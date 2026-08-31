"""Small UI helper utilities."""

from __future__ import annotations


def short_status_detail(message: str, *, max_len: int = 22) -> str:
    """Compact a status message for the status bar."""
    compact = " ".join(message.split())
    if not compact:
        return ""
    if len(compact) <= max_len:
        return f"({compact})"
    return f"({compact[: max_len - 1]}…)"
