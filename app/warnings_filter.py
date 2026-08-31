"""Runtime warning filters for cleaner local development logs."""

from __future__ import annotations

import warnings


def configure_runtime_warnings(*, debug: bool = False) -> None:
    """Hide known third-party warnings unless debug mode is on."""
    if debug:
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.quantize_per_tensor.*",
        category=UserWarning,
    )
