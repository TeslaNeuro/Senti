# Copyright (c) 2026 Arshia Keshvari
# SPDX-License-Identifier: MIT
#
# This file is part of Senti.
# Licensed under the MIT License. See the LICENSE file for details.

"""Ollama-backed local vision model."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

import cv2
import numpy as np

from app.config import AppConfig
from app.vision.ollama_models import (
    OllamaModelInfo,
    model_matches_request,
    parse_ollama_models,
    resolve_ollama_model,
)
from app.vision.vision_model import (
    VLM_SYSTEM_PROMPT,
    VisionAnalysisRequest,
    VisionAnalysisResult,
    VisionModel,
    build_vlm_user_prompt,
)

logger = logging.getLogger(__name__)


def encode_image_jpeg_base64(image_bgr: np.ndarray, max_width: int = 1280) -> str:
    """Encode a BGR frame as a base64 JPEG, resizing if needed."""
    image = image_bgr
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(
            image,
            (max_width, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to encode image for VLM.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class OllamaVisionModel(VisionModel):
    """Local VLM via Ollama's chat API."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._base_url = config.vlm_base_url.rstrip("/")
        self._requested_model = config.vlm_model or "gemma4"
        self._model = self._requested_model

    @property
    def model_name(self) -> str:
        return self._model

    def _fetch_show_capabilities(self, model_name: str) -> tuple[str, ...]:
        """Fetch authoritative capabilities from Ollama's show API."""
        payload = json.dumps({"name": model_name}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/api/show",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        return tuple(data.get("capabilities") or ())

    def _fetch_models(self) -> list[OllamaModelInfo]:
        request = urllib.request.Request(
            f"{self._base_url}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = parse_ollama_models(payload)

        # Ollama's /api/tags can omit vision for some models (e.g. gemma4:latest).
        enriched: list[OllamaModelInfo] = []
        for model in models:
            if (
                model_matches_request(self._requested_model, model.name)
                and "vision" not in model.capabilities
            ):
                try:
                    capabilities = self._fetch_show_capabilities(model.name)
                    enriched.append(OllamaModelInfo(model.name, capabilities))
                    continue
                except Exception:
                    logger.debug("Failed to enrich capabilities for %s", model.name, exc_info=True)
            enriched.append(model)
        return enriched

    def _resolve_model(self) -> tuple[bool, str]:
        try:
            models = self._fetch_models()
            resolved, error = resolve_ollama_model(self._requested_model, models)
            if error:
                return False, error
            assert resolved is not None
            self._model = resolved
            if resolved != self._requested_model:
                return True, f"Ollama ready ({resolved})"
            return True, f"Ollama ready ({self._model})"
        except urllib.error.URLError as exc:
            return False, f"Ollama not reachable at {self._base_url}: {exc.reason}"
        except Exception as exc:
            return False, str(exc)

    def check_availability(self) -> tuple[bool, str]:
        return self._resolve_model()

    async def analyze(self, request: VisionAnalysisRequest) -> VisionAnalysisResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.analyze_sync, request)

    def analyze_sync(self, request: VisionAnalysisRequest) -> VisionAnalysisResult:
        started = time.perf_counter()
        try:
            available, message = self._resolve_model()
            if not available:
                raise RuntimeError(message)

            image_b64 = encode_image_jpeg_base64(request.image_bgr)
            user_prompt = build_vlm_user_prompt(request)
            body = {
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": VLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_prompt,
                        "images": [image_b64],
                    },
                ],
            }
            payload = json.dumps(body).encode("utf-8")
            http_request = urllib.request.Request(
                f"{self._base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(http_request, timeout=120) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                try:
                    error_message = json.loads(detail).get("error", detail)
                except json.JSONDecodeError:
                    error_message = detail or exc.reason
                raise RuntimeError(f"Ollama error ({exc.code}): {error_message}") from exc
            message = data.get("message", {})
            content = (message.get("content") or "").strip()
            if not content:
                raise RuntimeError("VLM returned an empty response.")
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.info("VLM request completed: %.1fs", latency_ms / 1000.0)
            return VisionAnalysisResult(
                description=content,
                latency_ms=latency_ms,
                model=self._model,
                frame_id=request.frame_id,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("VLM request failed")
            return VisionAnalysisResult(
                description="",
                latency_ms=latency_ms,
                model=self._model,
                frame_id=request.frame_id,
                error=str(exc),
            )


def create_vision_model(config: AppConfig) -> VisionModel:
    """Factory for the configured local VLM runtime."""
    runtime = config.vlm_runtime.strip().lower()
    if runtime == "ollama":
        return OllamaVisionModel(config)
    raise ValueError(f"Unsupported VLM_RUNTIME: {config.vlm_runtime}")
