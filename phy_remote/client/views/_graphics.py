"""Shared fastplotlib graphic helpers."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def safe_delete(subplot, graphic) -> None:
    """Delete a fastplotlib graphic from a subplot.

    Hides the graphic first (instant visual effect), then calls delete_graphic
    to free GPU memory.  If delete_graphic raises the graphic is already hidden
    so it won't accumulate visually.
    """
    try:
        graphic.visible = False
    except Exception:
        pass
    try:
        subplot.delete_graphic(graphic)
    except Exception as exc:
        logger.debug("delete_graphic failed after hide: %s", exc)
