"""Tests for Ollama model resolution."""

from app.vision.ollama_models import OllamaModelInfo, resolve_ollama_model


def _models() -> list[OllamaModelInfo]:
    return [
        OllamaModelInfo("gemma4:12b-mlx", ("completion", "tools", "thinking")),
        OllamaModelInfo("qwen3.5:9b", ("vision", "completion", "tools", "thinking")),
        OllamaModelInfo("llava:latest", ("vision", "completion")),
    ]


def test_resolve_exact_vision_model() -> None:
    resolved, error = resolve_ollama_model("qwen3.5:9b", _models())
    assert error is None
    assert resolved == "qwen3.5:9b"


def test_resolve_family_prefers_vision_variant() -> None:
    resolved, error = resolve_ollama_model("llava", _models())
    assert error is None
    assert resolved == "llava:latest"


def test_resolve_gemma4_without_vision_fails() -> None:
    models = [
        OllamaModelInfo("gemma4:12b-mlx", ("completion", "tools", "thinking")),
        OllamaModelInfo("qwen3.5:9b", ("vision", "completion", "tools", "thinking")),
        OllamaModelInfo("llava:latest", ("vision", "completion")),
    ]
    resolved, error = resolve_ollama_model("gemma4", models)
    assert resolved is None
    assert error is not None
    assert "does not support images" in error or "none support images" in error


def test_resolve_gemma4_when_show_api_lists_vision() -> None:
    models = [
        OllamaModelInfo("gemma4:latest", ("completion", "tools", "thinking", "vision")),
        OllamaModelInfo("qwen3.5:9b", ("vision", "completion", "tools", "thinking")),
    ]
    resolved, error = resolve_ollama_model("gemma4", models)
    assert error is None
    assert resolved == "gemma4:latest"


def test_resolve_missing_model_reports_installed_names() -> None:
    resolved, error = resolve_ollama_model("moondream", _models())
    assert resolved is None
    assert error is not None
    assert "not found" in error
