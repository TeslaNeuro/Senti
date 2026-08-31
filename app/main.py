"""Application bootstrap and permission handling."""

from __future__ import annotations

from app.qt_bootstrap import configure_qt_environment

configure_qt_environment()

import logging
import sys

from PySide6.QtCore import QCoreApplication, QEventLoop, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from app.camera.capture import CameraCapture
from app.config import AppConfig, setup_logging
from app.warnings_filter import configure_runtime_warnings
from app.detection.worker import DetectionThread
from app.ocr.worker import OcrThread
from app.speech.qt_tts import SpeechController
from app.ui.main_window import MainWindow
from app.vision.worker import VlmThread
from app.voice.worker import VoiceThread

logger = logging.getLogger(__name__)


def _configure_macos_app() -> None:
    QCoreApplication.setOrganizationName("Senti")
    QCoreApplication.setApplicationName("What Am I Looking At?")
    QCoreApplication.setApplicationVersion("0.1.0")


def _ensure_qt_plugin_paths() -> None:
    """Register PySide6 plugin directory with Qt (backup to env vars)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("PySide6")
    if spec is None or not spec.origin:
        return
    plugins = Path(spec.origin).resolve().parent / "Qt" / "plugins"
    if plugins.is_dir():
        QCoreApplication.addLibraryPath(str(plugins))


def _request_camera_permission(app: QApplication) -> bool:
    """Request macOS camera permission via Qt 6 permission API."""
    try:
        from PySide6.QtCore import QCameraPermission
    except ImportError:
        logger.warning("QCameraPermission unavailable; proceeding without explicit check.")
        return True

    permission = QCameraPermission()
    status = app.checkPermission(permission)
    if status == Qt.PermissionStatus.Granted:
        return True
    if status == Qt.PermissionStatus.Denied:
        return False

    loop = QEventLoop()
    granted = False

    def on_result(result) -> None:
        nonlocal granted
        granted = result.status() == Qt.PermissionStatus.Granted
        loop.quit()

    app.requestPermission(permission, on_result)
    loop.exec()
    return granted or app.checkPermission(permission) == Qt.PermissionStatus.Granted


def _request_microphone_permission(app: QApplication) -> bool:
    """Request macOS microphone permission via Qt 6 permission API."""
    try:
        from PySide6.QtCore import QMicrophonePermission
    except ImportError:
        logger.warning("QMicrophonePermission unavailable; proceeding without explicit check.")
        return True

    permission = QMicrophonePermission()
    status = app.checkPermission(permission)
    if status == Qt.PermissionStatus.Granted:
        return True
    if status == Qt.PermissionStatus.Denied:
        return False

    loop = QEventLoop()
    granted = False

    def on_result(result) -> None:
        nonlocal granted
        granted = result.status() == Qt.PermissionStatus.Granted
        loop.quit()

    app.requestPermission(permission, on_result)
    loop.exec()
    return granted or app.checkPermission(permission) == Qt.PermissionStatus.Granted


def main() -> int:
    _configure_macos_app()
    config = AppConfig.load()
    config.validate()
    configure_runtime_warnings(debug=config.debug_mode)
    setup_logging(config)

    _ensure_qt_plugin_paths()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if not _request_camera_permission(app):
        QMessageBox.critical(
            None,
            "Camera Permission Required",
            "Camera access is required for this visual assistant.\n\n"
            "Enable it in System Settings → Privacy & Security → Camera, "
            "then relaunch the app.",
        )
        return 1

    if config.voice_enabled and not _request_microphone_permission(app):
        QMessageBox.warning(
            None,
            "Microphone Permission Required",
            "Voice input is enabled but microphone access was denied.\n\n"
            "Enable it in System Settings → Privacy & Security → Microphone, "
            "then relaunch the app.",
        )

    camera = CameraCapture(config)
    detection_thread = DetectionThread(config, camera)
    vlm_thread = VlmThread(config)
    ocr_thread = OcrThread(config) if config.ocr_enabled else None
    speech_controller = SpeechController(config) if config.tts_enabled else None
    voice_thread = VoiceThread(config) if config.voice_enabled else None
    window = MainWindow(
        config,
        camera,
        detection_thread,
        vlm_thread,
        ocr_thread,
        speech_controller,
        voice_thread,
    )
    window.show()

    camera.start()
    detection_thread.start()
    vlm_thread.start()
    if ocr_thread is not None:
        ocr_thread.start()
    if voice_thread is not None:
        voice_thread.start()

    exit_code = app.exec()
    if voice_thread is not None:
        voice_thread.stop()
    if ocr_thread is not None:
        ocr_thread.stop()
    vlm_thread.stop()
    detection_thread.stop()
    camera.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
