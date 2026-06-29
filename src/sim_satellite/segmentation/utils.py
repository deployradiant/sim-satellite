"""Small shared helpers for segmentation workflows."""

from __future__ import annotations

from collections.abc import Callable

ProgressReporter = Callable[[str], None]


def _should_report(current: int, total: int) -> bool:
    if total <= 0:
        return False
    if current == 1 or current == total or total <= 10:
        return True
    step = max(1, total // 10)
    return current % step == 0


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)
