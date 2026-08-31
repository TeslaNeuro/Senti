"""Helpers for resolving Ollama model names and capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class OllamaModelInfo:
    name: str
    capabilities: tuple[str, ...]


def parse_ollama_models(payload: dict) -> list[OllamaModelInfo]:
    models: list[OllamaModelInfo] = []
    for item in payload.get("models", []):
        name = item.get("name", "").strip()
        if not name:
            continue
        caps = tuple(item.get("capabilities") or ())
        models.append(OllamaModelInfo(name=name, capabilities=caps))
    return models


def _base_name(model_name: str) -> str:
    return model_name.split(":", 1)[0]


def _matches_requested(requested: str, candidate: str) -> bool:
    if candidate == requested:
        return True
    if ":" not in requested and candidate.startswith(f"{requested}:"):
        return True
    return False


def model_matches_request(requested: str, candidate: str) -> bool:
    """Return True when an installed model satisfies a configured model name."""
    return _matches_requested(requested.strip(), candidate.strip())


def resolve_ollama_model(
    requested: str,
    models: Iterable[OllamaModelInfo],
    *,
    require_vision: bool = True,
) -> tuple[str | None, str | None]:
    """Resolve a configured model name to an installed Ollama model."""
    requested = requested.strip()
    if not requested:
        return None, "VLM_MODEL must be set."

    model_list = list(models)
    if not model_list:
        return None, "No models are installed in Ollama."

    exact = [model for model in model_list if model.name == requested]
    if exact:
        chosen = exact[0]
        if require_vision and "vision" not in chosen.capabilities:
            return None, (
                f"Model '{chosen.name}' is installed but does not support images. "
                "Install a vision-capable model such as gemma4:latest, llava, or qwen3.5:9b."
            )
        return chosen.name, None

    candidates = [model for model in model_list if _matches_requested(requested, model.name)]
    if not candidates:
        installed = ", ".join(model.name for model in model_list[:5])
        suffix = "..." if len(model_list) > 5 else ""
        return None, (
            f"Model '{requested}' not found in Ollama. Installed: {installed}{suffix}"
        )

    vision_candidates = [model for model in candidates if "vision" in model.capabilities]
    if require_vision:
        if vision_candidates:
            return vision_candidates[0].name, None
        names = ", ".join(model.name for model in candidates)
        return None, (
            f"Found {names}, but none support images. "
            f"Set VLM_MODEL to a vision model or run: ollama pull {requested}"
        )

    return candidates[0].name, None
