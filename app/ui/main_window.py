"""Native macOS UI for the visual assistant."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.camera.capture import CameraCapture, CameraMetrics, CameraStatus, bgr_to_qimage
from app.camera.frame_buffer import CameraFrame
from app.config import AppConfig
from app.detection.detector import DetectionMetrics, DetectionResult
from app.detection.worker import DetectionThread
from app.perception.engine import AssistantState, SceneState
from app.ocr.engine import OcrRequest, format_ocr_results
from app.ocr.worker import OcrThread
from app.scene.conversation import QuestionRoute, QuestionRouter
from app.scene.scene_state import SceneContextManager
from app.speech.qt_tts import SpeechController
from app.ui.helpers import short_status_detail
from app.ui.overlay import draw_detections
from app.voice.worker import VoiceThread
from app.vision.object_focus import crop_detection, select_focus_target
from app.vision.vision_model import VisionAnalysisRequest
from app.vision.worker import VlmThread

logger = logging.getLogger(__name__)


class StatusIndicator(QLabel):
    """Small colored status pill used in the status bar."""

    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label = label
        self.set_active(False)

    def set_active(self, active: bool, detail: str = "", *, tooltip: str = "") -> None:
        color = "#34c759" if active else "#8e8e93"
        suffix = f" {detail}" if detail else ""
        self.setText(f'<span style="color:{color};">●</span> {self._label}{suffix}')
        self.setTextFormat(Qt.TextFormat.RichText)
        tip = tooltip or (f"{self._label}: {detail.lstrip('()')}" if detail else self._label)
        self.setToolTip(tip)


class MainWindow(QMainWindow):
    """Primary application window with live camera preview."""

    def __init__(
        self,
        config: AppConfig,
        camera: CameraCapture,
        detection_thread: DetectionThread,
        vlm_thread: VlmThread,
        ocr_thread: Optional[OcrThread] = None,
        speech_controller: Optional[SpeechController] = None,
        voice_thread: Optional[VoiceThread] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._camera = camera
        self._detection_thread = detection_thread
        self._vlm_thread = vlm_thread
        self._ocr_thread = ocr_thread
        self._speech = speech_controller
        self._voice_thread = voice_thread
        self._scene_context = SceneContextManager(max_turns=config.conversation_max_turns)
        self._question_router = QuestionRouter(stale_frame_gap=config.context_stale_frames)
        self._assistant_state = "INITIALIZING"
        self._latest_frame: Optional[CameraFrame] = None
        self._latest_result: Optional[DetectionResult] = None
        self._latest_detection_metrics: Optional[DetectionMetrics] = None
        self._latest_camera_metrics: Optional[CameraMetrics] = None
        self._latest_scene_state: Optional[SceneState] = None
        self._last_vlm_latency_ms: Optional[float] = None
        self._analyzing = False
        self._vlm_available = False
        self._ocr_available = False
        self._ocr_running = False
        self._tts_available = False
        self._voice_available = False
        self._voice_listening = False
        self._voice_transcribing = False
        self._pending_read_response = False
        self._pending_question: Optional[str] = None
        self._last_analysis_image: Optional[np.ndarray] = None

        self.setWindowTitle("What Am I Looking At?")
        self.setMinimumSize(960, 720)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("👁 What Am I Looking At?")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)

        self._preview = QLabel("Starting camera…")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(420)
        self._preview.setStyleSheet(
            "background-color: #1c1c1e; color: #aeaeb2; border-radius: 12px; font-size: 16px;"
        )
        self._preview.setScaledContents(True)
        layout.addWidget(self._preview, stretch=1)

        self._response = QLabel(
            "The assistant will describe the scene automatically when it stabilizes."
        )
        self._response.setWordWrap(True)
        self._response.setMinimumHeight(72)
        self._response.setStyleSheet(
            "background-color: #2c2c2e; color: #f2f2f7; padding: 14px; border-radius: 10px;"
        )
        layout.addWidget(self._response)

        self._description = QLabel("Loading YOLO26…")
        self._description.setWordWrap(True)
        self._description.setStyleSheet("color: #aeaeb2; font-size: 13px;")
        layout.addWidget(self._description)

        focus_row = QHBoxLayout()
        focus_label = QLabel("Focus:")
        focus_label.setStyleSheet("color: #aeaeb2; font-size: 13px;")
        focus_row.addWidget(focus_label)
        self._focus_combo = QComboBox()
        self._focus_combo.addItem("Full scene", None)
        self._focus_combo.setEnabled(False)
        self._focus_combo.currentIndexChanged.connect(self._on_focus_changed)
        focus_row.addWidget(self._focus_combo, stretch=1)
        layout.addLayout(focus_row)

        action_row = QHBoxLayout()
        self._question_input = QLineEdit()
        self._question_input.setPlaceholderText("Ask about what you're looking at…")
        self._question_input.setEnabled(False)
        self._question_input.returnPressed.connect(self._on_ask_clicked)
        self._question_input.setStyleSheet(
            "padding: 10px 14px; font-size: 14px; border-radius: 8px;"
        )
        action_row.addWidget(self._question_input, stretch=1)

        self._mic_button = QPushButton("Mic")
        self._mic_button.setEnabled(False)
        self._mic_button.setToolTip("Push-to-talk: click to record, click Stop when done (Esc to cancel)")
        self._mic_button.clicked.connect(self._on_mic_clicked)
        self._mic_button.setStyleSheet(
            "padding: 10px 18px; font-size: 14px; font-weight: 600;"
        )
        action_row.addWidget(self._mic_button)

        self._ask_button = QPushButton("Ask")
        self._ask_button.setEnabled(False)
        self._ask_button.setToolTip("Send a typed question about the scene")
        self._ask_button.clicked.connect(self._on_ask_clicked)
        self._ask_button.setStyleSheet(
            "padding: 10px 18px; font-size: 14px; font-weight: 600;"
        )
        action_row.addWidget(self._ask_button)

        self._analyze_button = QPushButton("Analyze")
        self._analyze_button.setEnabled(False)
        self._analyze_button.setToolTip("Re-run scene analysis now (bypasses cooldown)")
        self._analyze_button.clicked.connect(self._on_analyze_clicked)
        self._analyze_button.setStyleSheet(
            "padding: 10px 18px; font-size: 14px; font-weight: 600;"
        )
        action_row.addWidget(self._analyze_button)

        self._speak_button = QPushButton("Speak")
        self._speak_button.setEnabled(False)
        self._speak_button.setToolTip("Read the current response aloud")
        self._speak_button.clicked.connect(self._on_speak_clicked)
        self._speak_button.setStyleSheet(
            "padding: 10px 18px; font-size: 14px; font-weight: 600;"
        )
        action_row.addWidget(self._speak_button)

        self._clear_button = QPushButton("Clear")
        self._clear_button.setToolTip("Clear the response and scene conversation memory")
        self._clear_button.clicked.connect(self._on_clear_clicked)
        self._clear_button.setStyleSheet(
            "padding: 10px 18px; font-size: 14px; font-weight: 600;"
        )
        action_row.addWidget(self._clear_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_widget = QWidget()
        status_widget.setLayout(status_layout)

        self._camera_indicator = StatusIndicator("Camera")
        self._yolo_indicator = StatusIndicator("YOLO26")
        self._vlm_indicator = StatusIndicator("VLM")
        self._ocr_indicator = StatusIndicator("OCR")
        self._tts_indicator = StatusIndicator("TTS")
        self._voice_indicator = StatusIndicator("Voice")
        self._fps_label = QLabel("FPS —")
        self._fps_label.setStyleSheet("color: #d1d1d6;")

        status_layout.addWidget(self._camera_indicator)
        status_layout.addSpacing(12)
        status_layout.addWidget(self._yolo_indicator)
        status_layout.addSpacing(12)
        status_layout.addWidget(self._vlm_indicator)
        status_layout.addSpacing(12)
        status_layout.addWidget(self._ocr_indicator)
        status_layout.addSpacing(12)
        status_layout.addWidget(self._tts_indicator)
        status_layout.addSpacing(12)
        status_layout.addWidget(self._voice_indicator)
        status_layout.addStretch()
        status_layout.addWidget(self._fps_label)

        status_bar.addPermanentWidget(status_widget, 1)
        self._state_label = QLabel("State: INITIALIZING")
        status_bar.addWidget(self._state_label)

        self._yolo_indicator.set_active(False, "(loading)")
        self._vlm_indicator.set_active(False, "(checking)")
        if self._config.ocr_enabled and self._ocr_thread is not None:
            self._ocr_indicator.set_active(False, "(checking)")
        else:
            self._ocr_indicator.set_active(False, "(off)")
        if self._config.tts_enabled and self._speech is not None:
            self._refresh_tts_indicator()
        else:
            self._tts_indicator.set_active(False, "(off)")
        if self._config.voice_enabled and self._voice_thread is not None:
            self._voice_indicator.set_active(False, "(checking)")
        else:
            self._voice_indicator.set_active(False, "(off)")

        self._camera.frame_ready.connect(self._on_frame_ready)
        self._camera.status_changed.connect(self._on_status_changed)
        self._camera.metrics_updated.connect(self._on_camera_metrics)
        self._camera.error_occurred.connect(self._on_camera_error)

        self._detection_thread.model_loaded.connect(self._on_yolo_loaded)
        self._detection_thread.model_failed.connect(self._on_yolo_failed)
        self._detection_thread.result_ready.connect(self._on_detection_result)
        self._detection_thread.metrics_updated.connect(self._on_detection_metrics)

        self._vlm_thread.availability_checked.connect(self._on_vlm_availability)
        self._vlm_thread.analysis_started.connect(self._on_vlm_started)
        self._vlm_thread.analysis_completed.connect(self._on_vlm_completed)
        self._vlm_thread.analysis_rejected.connect(self._on_vlm_rejected)

        if self._ocr_thread is not None:
            self._ocr_thread.availability_checked.connect(self._on_ocr_availability)
            self._ocr_thread.recognition_started.connect(self._on_ocr_started)
            self._ocr_thread.recognition_completed.connect(self._on_ocr_completed)
            self._ocr_thread.recognition_rejected.connect(self._on_ocr_rejected)

        if self._voice_thread is not None:
            self._voice_thread.availability_checked.connect(self._on_voice_availability)
            self._voice_thread.listening_started.connect(self._on_voice_listening_started)
            self._voice_thread.listening_stopped.connect(self._on_voice_listening_stopped)
            self._voice_thread.transcription_completed.connect(self._on_voice_transcription)
            self._voice_thread.transcription_failed.connect(self._on_voice_failed)

        escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        escape_shortcut.activated.connect(self._on_escape_pressed)

    def _is_busy(self) -> bool:
        """True when user-facing work should block new questions."""
        return (
            self._analyzing
            or self._voice_listening
            or self._voice_transcribing
            or (self._ocr_running and self._pending_read_response)
        )

    def _sync_interaction_state(self) -> None:
        busy = self._is_busy()
        can_ask = self._vlm_available and not busy
        self._ask_button.setEnabled(can_ask)
        self._question_input.setEnabled(can_ask)
        self._analyze_button.setEnabled(self._vlm_available and not self._is_busy())
        self._mic_button.setEnabled(
            self._voice_available and (self._voice_listening or not busy)
        )
        self._speak_button.setEnabled(self._tts_available and not busy)
        self._clear_button.setEnabled(not busy)

    @Slot()
    def _on_escape_pressed(self) -> None:
        if self._voice_listening and self._voice_thread is not None:
            self._voice_thread.stop_listening()

    @Slot()
    def _on_clear_clicked(self) -> None:
        if self._is_busy():
            return
        self._scene_context.reset_scene()
        self._pending_question = None
        self._pending_read_response = False
        self._response.setText("Scene memory cleared. Point the camera and wait for READY, or click Analyze.")
        self._question_input.clear()
        if self._speech is not None:
            self._speech.stop()

    @Slot()
    def _on_mic_clicked(self) -> None:
        if self._voice_thread is None or not self._voice_available:
            return
        if self._voice_listening:
            self._voice_thread.stop_listening()
        else:
            self._voice_thread.start_listening()

    @Slot(bool, str)
    def _on_voice_availability(self, available: bool, message: str) -> None:
        self._voice_available = available
        if available:
            self._voice_indicator.set_active(True, tooltip=message)
        else:
            detail = "(off)" if not self._config.voice_enabled else short_status_detail(message)
            self._voice_indicator.set_active(False, detail, tooltip=message)
            if self._config.voice_enabled:
                logger.warning("Voice input unavailable: %s", message)
        self._sync_interaction_state()

    @Slot()
    def _on_voice_listening_started(self) -> None:
        self._voice_listening = True
        self._mic_button.setText("Stop")
        self._mic_button.setStyleSheet(
            "padding: 10px 18px; font-size: 14px; font-weight: 600; background-color: #ff453a; color: white;"
        )
        self._response.setText("Listening… click Stop when you're done (Esc to cancel).")
        self._sync_interaction_state()

    @Slot()
    def _on_voice_listening_stopped(self) -> None:
        self._voice_listening = False
        self._voice_transcribing = True
        self._mic_button.setText("Mic")
        self._mic_button.setStyleSheet(
            "padding: 10px 18px; font-size: 14px; font-weight: 600;"
        )
        self._response.setText("Transcribing… (first run may download the speech model)")
        self._sync_interaction_state()

    @Slot(object)
    def _on_voice_transcription(self, result) -> None:
        self._voice_transcribing = False
        self._question_input.setText(result.text)
        self._response.setText(f'Heard: "{result.text}"')
        logger.info("Voice transcript accepted in %.0f ms", result.latency_ms)
        self._handle_user_question(result.text)
        self._question_input.clear()
        self._sync_interaction_state()

    @Slot(str)
    def _on_voice_failed(self, message: str) -> None:
        self._voice_transcribing = False
        if message:
            logger.info("Voice input failed: %s", message)
        self._response.setText(message or "Voice input failed.")
        self._sync_interaction_state()

    @Slot()
    def _on_analyze_clicked(self) -> None:
        request = self._build_analysis_request()
        if request is None:
            self._response.setText("No frame available to analyze yet. Wait for the camera and YOLO26.")
            return
        request = self._apply_object_focus(request)
        self._pending_question = None
        self._submit_vlm_request(request)

    @Slot()
    def _on_ask_clicked(self) -> None:
        question = self._question_input.text().strip()
        if not question:
            return
        self._handle_user_question(question)
        self._question_input.clear()

    def _handle_user_question(self, question: str) -> None:
        if not self._vlm_available:
            self._response.setText("VLM is not available. Configure Ollama to ask questions.")
            return

        self._scene_context.add_user_turn(question)
        assistant_state = (
            self._latest_scene_state.assistant_state
            if self._latest_scene_state is not None
            else AssistantState.WATCHING
        )
        latest_frame_id = self._latest_frame.frame_id if self._latest_frame is not None else -1
        scene = self._scene_context.scene
        plan = self._question_router.route(
            question,
            has_description=bool(scene.description),
            detections=list(scene.objects or (self._latest_result.detections if self._latest_result else [])),
            analyzed_frame_id=scene.frame_id,
            latest_frame_id=latest_frame_id,
            has_cached_frame=scene.analyzed_frame_bgr is not None,
            assistant_state=assistant_state,
            last_description=scene.description,
            ocr_results=list(scene.ocr_results),
            ocr_enabled=self._config.ocr_enabled and self._ocr_available,
        )

        if plan.route == QuestionRoute.NO_SCENE:
            self._response.setText(plan.reason or "No scene context yet.")
            return

        if plan.route == QuestionRoute.OCR_READ:
            self._pending_read_response = True
            if not self._request_ocr(respond_in_ui=True):
                self._pending_read_response = False
                self._response.setText("No frame available for OCR yet.")
            return

        if plan.route == QuestionRoute.CONTEXT_ONLY and plan.answer:
            self._scene_context.add_assistant_turn(plan.answer)
            self._response.setText(plan.answer)
            logger.info("Answered from scene context: %s", plan.reason)
            self._maybe_speak(plan.answer, force=plan.speak_aloud)
            return

        use_cached = plan.route == QuestionRoute.VLM_CACHED_FRAME
        request = self._build_analysis_request(
            user_question=question,
            prefer_cached_frame=use_cached,
            manual=True,
        )
        if request is None:
            self._response.setText("No frame available to analyze yet. Wait for the camera and YOLO26.")
            return
        request = self._apply_object_focus(request, question)
        self._pending_question = question
        self._submit_vlm_request(request)

    def _refresh_tts_indicator(self) -> None:
        if self._speech is None:
            self._tts_indicator.set_active(False, "(off)")
            return
        self._tts_available = self._speech.available
        if self._tts_available:
            self._tts_indicator.set_active(True, tooltip=self._speech.message if self._speech else "")
        else:
            detail = "(off)" if not self._config.tts_enabled else "(unavailable)"
            message = self._speech.message if self._speech else ""
            self._tts_indicator.set_active(False, detail, tooltip=message or detail)
            if self._config.tts_enabled:
                logger.warning("TTS unavailable: %s", message)
        self._sync_interaction_state()

    def _maybe_speak(self, text: str, *, force: bool = False) -> None:
        if self._speech is None or not self._tts_available:
            return
        if not force and not self._config.tts_auto_speak:
            return
        self._speech.speak(text)

    @Slot()
    def _on_speak_clicked(self) -> None:
        text = self._response.text().strip()
        if not text:
            text = self._scene_context.scene.description
        if text:
            self._maybe_speak(text, force=True)

    def _submit_vlm_request(self, request: VisionAnalysisRequest) -> None:
        image = request.image_bgr
        self._last_analysis_image = image.copy() if image is not None else None
        self._vlm_thread.request_analysis(request)

    def _apply_object_focus(
        self,
        request: VisionAnalysisRequest,
        user_question: Optional[str] = None,
    ) -> VisionAnalysisRequest:
        if not self._config.object_crop_enabled:
            return request

        image = request.image_bgr
        height, width = image.shape[:2]
        focus_track_id = self._focus_combo.currentData()
        selection = select_focus_target(
            request.detections,
            question=user_question,
            focus_track_id=focus_track_id,
            frame_width=width,
            frame_height=height,
            allow_auto=focus_track_id is None,
        )
        if selection is None:
            return request

        crop = crop_detection(
            image,
            selection.detection,
            padding_ratio=self._config.object_crop_padding,
        )
        logger.info("Object focus: %s", selection.reason)
        self._scene_context.scene.focused_object = selection.detection
        return VisionAnalysisRequest(
            image_bgr=crop,
            frame_id=request.frame_id,
            detections=request.detections,
            user_question=request.user_question,
            ocr_text=request.ocr_text,
            previous_context=request.previous_context,
            manual=request.manual,
            focused_detection=selection.detection,
            is_object_crop=True,
        )

    def _build_analysis_request(
        self,
        *,
        user_question: Optional[str] = None,
        prefer_cached_frame: bool = False,
        manual: bool = True,
    ) -> Optional[VisionAnalysisRequest]:
        scene = self._scene_context.scene
        if prefer_cached_frame and scene.analyzed_frame_bgr is not None:
            detections = list(scene.objects)
            return VisionAnalysisRequest(
                image_bgr=scene.analyzed_frame_bgr.copy(),
                frame_id=scene.frame_id,
                detections=detections,
                user_question=user_question,
                ocr_text=list(scene.ocr_text) or None,
                previous_context=self._scene_context.previous_context(),
                manual=manual,
            )

        selected = self._detection_thread.last_selected_frame
        if selected is not None:
            candidate = selected.candidate
            return VisionAnalysisRequest(
                image_bgr=candidate.camera_frame.data.copy(),
                frame_id=candidate.camera_frame.frame_id,
                detections=list(candidate.result.detections),
                user_question=user_question,
                ocr_text=list(scene.ocr_text) or None,
                previous_context=self._scene_context.previous_context(),
                manual=manual,
            )

        if self._latest_frame is None:
            return None

        detections = self._latest_result.detections if self._latest_result else []
        return VisionAnalysisRequest(
            image_bgr=self._latest_frame.data.copy(),
            frame_id=self._latest_frame.frame_id,
            detections=list(detections),
            user_question=user_question,
            ocr_text=list(scene.ocr_text) or None,
            previous_context=self._scene_context.previous_context(),
            manual=manual,
        )

    def _ocr_source_image(self) -> Optional[tuple[np.ndarray, int]]:
        scene = self._scene_context.scene
        if scene.analyzed_frame_bgr is not None:
            image = scene.analyzed_frame_bgr.copy()
            frame_id = scene.frame_id if scene.frame_id >= 0 else (self._latest_frame.frame_id if self._latest_frame else -1)
        elif self._latest_frame is not None:
            image = self._latest_frame.data.copy()
            frame_id = self._latest_frame.frame_id
        else:
            return None

        detections = self._latest_result.detections if self._latest_result else []
        focus_track_id = self._focus_combo.currentData()
        selection = select_focus_target(
            list(detections),
            focus_track_id=focus_track_id,
            frame_width=image.shape[1],
            frame_height=image.shape[0],
            allow_auto=focus_track_id is not None,
        )
        if selection is not None:
            image = crop_detection(
                image,
                selection.detection,
                padding_ratio=self._config.object_crop_padding,
            )
        return image, frame_id

    def _request_ocr(self, *, respond_in_ui: bool = False) -> bool:
        if not self._config.ocr_enabled or self._ocr_thread is None or not self._ocr_available:
            return False
        source = self._ocr_source_image()
        if source is None:
            return False
        image, frame_id = source
        self._ocr_thread.request_recognition(
            OcrRequest(image_bgr=image, frame_id=frame_id, respond_in_ui=respond_in_ui)
        )
        return True

    def _maybe_auto_ocr(self, scene_state: SceneState) -> None:
        if not self._config.ocr_enabled or not self._config.ocr_auto_on_ready:
            return
        if not self._ocr_available or self._ocr_running:
            return
        if scene_state.selected_frame is None:
            return
        self._request_ocr(respond_in_ui=False)

    def _maybe_auto_analyze(self, scene_state: SceneState) -> None:
        if not self._config.vlm_auto_analyze:
            return
        if not self._vlm_available or self._analyzing:
            return
        if scene_state.selected_frame is None:
            return

        selected = scene_state.selected_frame.candidate
        request = VisionAnalysisRequest(
            image_bgr=selected.camera_frame.data.copy(),
            frame_id=selected.camera_frame.frame_id,
            detections=list(selected.result.detections),
            ocr_text=None,
            previous_context=self._scene_context.previous_context(),
            manual=False,
        )
        self._submit_vlm_request(request)

    @Slot(object)
    def _on_frame_ready(self, frame: CameraFrame) -> None:
        self._latest_frame = frame
        self._render_frame(frame.data)

    @Slot(object, object, object)
    def _on_detection_result(
        self, frame: CameraFrame, result: DetectionResult, scene_state: SceneState
    ) -> None:
        self._latest_frame = frame
        self._latest_result = result
        self._latest_scene_state = scene_state
        if scene_state.change.rising_edge:
            self._scene_context.reset_scene()
        self._vlm_thread.update_latest_frame_id(frame.frame_id)
        self._render_frame(frame.data)
        self._update_detection_summary(result)
        self._refresh_focus_selector(result)
        if not self._analyzing:
            self._state_label.setText(f"State: {scene_state.assistant_state.name}")
        self._maybe_auto_analyze(scene_state)
        self._maybe_auto_ocr(scene_state)

    @Slot(object)
    def _on_detection_metrics(self, metrics: DetectionMetrics) -> None:
        self._latest_detection_metrics = metrics
        self._refresh_fps_label()

    def _ocr_status_detail(self, available: bool, message: str) -> str:
        if not self._config.ocr_enabled:
            return "(off)"
        if available:
            return ""
        lowered = message.lower()
        if "not installed" in lowered:
            return "(not installed)"
        if "disabled" in lowered:
            return "(off)"
        return f"({message})" if message else "(unavailable)"

    @Slot(bool, str)
    def _on_ocr_availability(self, available: bool, message: str) -> None:
        self._ocr_available = available
        if available:
            self._ocr_indicator.set_active(True, tooltip=message)
        else:
            self._ocr_indicator.set_active(
                False,
                self._ocr_status_detail(available, message),
                tooltip=message,
            )
            if self._config.ocr_enabled:
                logger.warning("OCR unavailable: %s", message)

    @Slot(int)
    def _on_ocr_started(self, frame_id: int) -> None:
        self._ocr_running = True
        self._sync_interaction_state()
        if self._pending_read_response:
            self._response.setText(f"Reading text from frame #{frame_id}…")

    @Slot(object)
    def _on_ocr_completed(self, analysis) -> None:
        self._ocr_running = False
        self._sync_interaction_state()
        if not analysis.ok:
            if self._pending_read_response:
                self._response.setText(f"OCR error: {analysis.error}")
                self._pending_read_response = False
            return

        self._scene_context.apply_ocr(analysis)
        if self._pending_read_response:
            answer = format_ocr_results(analysis.results)
            self._scene_context.add_assistant_turn(answer)
            self._response.setText(answer)
            self._pending_read_response = False
            self._maybe_speak(answer)
        elif analysis.text_lines:
            logger.info("OCR captured %d lines for frame #%d", len(analysis.text_lines), analysis.frame_id)

    @Slot(str)
    def _on_ocr_rejected(self, message: str) -> None:
        if message:
            logger.info("OCR request rejected: %s", message)
        if self._pending_read_response:
            self._response.setText(message or "OCR is busy.")
            self._pending_read_response = False
            self._sync_interaction_state()

    @Slot(str)
    def _on_yolo_loaded(self, device: str) -> None:
        self._yolo_indicator.set_active(True, f"({device})")
        self._description.setText(
            "YOLO26 + tracking is running. The scene will be described automatically when stable."
        )
        logger.info("YOLO26 ready on %s", device)

    @Slot(str)
    def _on_yolo_failed(self, message: str) -> None:
        self._yolo_indicator.set_active(False, "(error)")
        self._description.setText(
            f"YOLO26 unavailable: {message}\nCamera preview will continue without detection."
        )
        logger.error("YOLO26 failed to load: %s", message)

    @Slot(bool, str)
    def _on_vlm_availability(self, available: bool, message: str) -> None:
        self._vlm_available = available
        if available:
            self._vlm_indicator.set_active(True, tooltip=message)
            if self._config.vlm_auto_analyze:
                self._description.setText(f"{message}. Auto-analyze is on when the scene stabilizes.")
            else:
                self._description.setText(f"{message}. Click Analyze when ready.")
        else:
            self._vlm_indicator.set_active(
                False,
                short_status_detail(message) or "(unavailable)",
                tooltip=message,
            )
            self._response.setText(f"VLM unavailable\n\n{message}")
            self._description.setText("Camera and YOLO26 still work without the VLM.")
        self._sync_interaction_state()

    @Slot(int, bool)
    def _on_vlm_started(self, frame_id: int, manual: bool) -> None:
        self._analyzing = True
        self._sync_interaction_state()
        self._state_label.setText("State: ANALYZING")
        if self._pending_question:
            prefix = "Thinking"
            self._response.setText(f'{prefix} about: "{self._pending_question}"…')
        else:
            prefix = "Analyzing" if manual else "Auto-analyzing"
            self._response.setText(f"{prefix} frame #{frame_id}…")

    @Slot(object)
    def _on_vlm_completed(self, result) -> None:
        self._analyzing = False
        self._pending_question = None
        self._last_vlm_latency_ms = result.latency_ms
        self._refresh_fps_label()
        self._sync_interaction_state()

        if not result.ok:
            self._response.setText(f"VLM error: {result.error}")
            self._state_label.setText("State: ERROR")
            return

        detections = self._latest_result.detections if self._latest_result else []
        self._scene_context.apply_analysis(
            result,
            detections,
            image_bgr=self._last_analysis_image,
        )
        self._response.setText(result.description)
        self._maybe_speak(result.description)
        state_name = (
            self._latest_scene_state.assistant_state.name
            if self._latest_scene_state
            else "READY"
        )
        self._state_label.setText(f"State: {state_name}")

    @Slot(str)
    def _on_vlm_rejected(self, message: str) -> None:
        if message:
            logger.info("VLM request rejected: %s", message)
        if self._pending_question:
            self._response.setText(message)

    def _render_frame(self, frame_bgr: np.ndarray) -> None:
        display = frame_bgr
        if self._latest_result and self._latest_result.detections:
            focus_track_id = self._focus_combo.currentData()
            display = draw_detections(
                frame_bgr,
                self._latest_result.detections,
                focus_track_id=focus_track_id,
            )
        image = bgr_to_qimage(display)
        self._preview.setPixmap(QPixmap.fromImage(image))

    def _refresh_focus_selector(self, result: DetectionResult) -> None:
        current = self._focus_combo.currentData()
        self._focus_combo.blockSignals(True)
        self._focus_combo.clear()
        self._focus_combo.addItem("Full scene", None)
        for detection in sorted(result.detections, key=lambda det: det.confidence, reverse=True):
            if detection.track_id is not None:
                label = f"#{detection.track_id} {detection.class_name}"
                self._focus_combo.addItem(label, detection.track_id)
        self._focus_combo.setEnabled(bool(result.detections))
        if current is not None:
            index = self._focus_combo.findData(current)
            if index >= 0:
                self._focus_combo.setCurrentIndex(index)
        self._focus_combo.blockSignals(False)

    @Slot(int)
    def _on_focus_changed(self, _index: int) -> None:
        if self._latest_frame is not None:
            self._render_frame(self._latest_frame.data)

    def _update_detection_summary(self, result: DetectionResult) -> None:
        if not result.detections:
            self._description.setText("No objects detected. Analyze still works on the full scene.")
            return

        lines = [
            self._format_detection_line(det)
            for det in sorted(result.detections, key=lambda d: d.confidence, reverse=True)[:5]
        ]
        extra = len(result.detections) - len(lines)
        summary = "Detected: " + ", ".join(lines)
        if extra > 0:
            summary += f" (+{extra} more)"

        if self._latest_scene_state is not None:
            scene = self._latest_scene_state
            if scene.assistant_state == AssistantState.WAITING_FOR_STABILITY:
                summary += (
                    f" | Stabilizing {scene.stability.stable_frames}/"
                    f"{scene.stability.required_frames}"
                )
            elif scene.selected_frame is not None:
                selected = scene.selected_frame.candidate
                summary += (
                    f" | Best frame #{selected.camera_frame.frame_id} "
                    f"(score {selected.score.total:.2f})"
                )

        focus_track_id = self._focus_combo.currentData()
        if focus_track_id is not None:
            summary += f" | Focus #{focus_track_id}"

        self._description.setText(summary)

    def _format_detection_line(self, detection) -> str:
        if detection.track_id is not None:
            return f"#{detection.track_id} {detection.class_name} — {detection.confidence:.0%}"
        return f"{detection.class_name} — {detection.confidence:.0%}"

    @Slot(object)
    def _on_status_changed(self, status: CameraStatus) -> None:
        active = status in {CameraStatus.READY, CameraStatus.RUNNING}
        detail = ""
        if status == CameraStatus.PERMISSION_DENIED:
            detail = "(denied)"
        elif status == CameraStatus.ERROR:
            detail = "(error)"
        self._camera_indicator.set_active(active, detail)

        if self._latest_scene_state is None and not self._analyzing:
            state_map = {
                CameraStatus.INITIALIZING: "INITIALIZING",
                CameraStatus.READY: "CAMERA_READY",
                CameraStatus.RUNNING: "WATCHING",
                CameraStatus.STOPPED: "STOPPED",
                CameraStatus.ERROR: "ERROR",
                CameraStatus.PERMISSION_DENIED: "ERROR",
            }
            self._state_label.setText(f"State: {state_map.get(status, 'ERROR')}")

        if status == CameraStatus.PERMISSION_DENIED:
            self._preview.setText(
                "Camera permission denied.\n\n"
                "Enable camera access for this app in\n"
                "System Settings → Privacy & Security → Camera."
            )
            self._preview.setPixmap(QPixmap())
        elif status == CameraStatus.ERROR:
            self._preview.setText("Camera error. Attempting recovery…")
            self._preview.setPixmap(QPixmap())

    @Slot(object)
    def _on_camera_metrics(self, metrics: CameraMetrics) -> None:
        self._latest_camera_metrics = metrics
        self._refresh_fps_label()

    def _refresh_fps_label(self) -> None:
        camera_fps = self._latest_camera_metrics.capture_fps if self._latest_camera_metrics else 0.0
        width = self._latest_camera_metrics.width if self._latest_camera_metrics else 0
        height = self._latest_camera_metrics.height if self._latest_camera_metrics else 0
        dropped = self._latest_camera_metrics.dropped_frames if self._latest_camera_metrics else 0

        parts = [
            f"Cam {camera_fps:4.1f} FPS",
            f"{width}×{height}",
            f"dropped {dropped}",
        ]
        if self._latest_detection_metrics:
            parts.append(f"YOLO {self._latest_detection_metrics.fps:4.1f} FPS")
            parts.append(f"{self._latest_detection_metrics.inference_ms:4.0f} ms")
        if self._last_vlm_latency_ms is not None:
            parts.append(f"VLM {self._last_vlm_latency_ms:4.0f} ms")
        if self._scene_context.scene.ocr_text:
            parts.append(f"OCR {len(self._scene_context.scene.ocr_text)} lines")
        self._fps_label.setText("  |  ".join(parts))

    @Slot(str)
    def _on_camera_error(self, message: str) -> None:
        logger.error("Camera UI error: %s", message)
        self._description.setText(f"Camera error: {message}")

    def closeEvent(self, event) -> None:  # noqa: N802
        logger.info("Closing application")
        if self._speech is not None:
            self._speech.stop()
        if self._voice_thread is not None:
            self._voice_thread.stop()
        if self._ocr_thread is not None:
            self._ocr_thread.stop()
        self._vlm_thread.stop()
        self._detection_thread.stop()
        self._camera.stop()
        super().closeEvent(event)
