"""
Axis-zoom event filter for fastplotlib Qt canvases.

Installed at QApplication level so it sees wheel events regardless of
which child widget inside the canvas actually receives them.

  Cmd  + scroll  →  zoom Y axis only
  Shift + scroll →  zoom X axis only
  plain scroll   →  left to fastplotlib's built-in pan-zoom

On macOS, Qt maps Command (⌘) → ControlModifier.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QApplication


class AxisZoomFilter(QObject):
    _STEP = 1.12  # zoom factor per scroll notch

    def __init__(self, fig, subplots: list, canvas, parent=None):
        super().__init__(parent)
        self._fig      = fig
        self._subplots = subplots
        self._canvas   = canvas   # top-level QWidget for this figure

    def eventFilter(self, obj, event) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False

        # Only act when the event target is inside our canvas hierarchy
        w = obj
        while w is not None:
            if w is self._canvas:
                break
            w = w.parent()
        else:
            return False

        mods  = event.modifiers()
        delta = event.angleDelta().y()
        if delta == 0:
            return False

        cmd   = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        if not cmd and not shift:
            return False  # plain scroll — let fastplotlib handle it

        factor = self._STEP ** (delta / 120)

        for sp in self._subplots:
            try:
                state = sp.camera.get_state()
                new   = dict(state)
                if cmd:
                    new['height'] = abs(state['height']) / factor
                else:
                    new['width']  = abs(state['width'])  / factor
                new['zoom'] = 1.0
                sp.camera.set_state(new)
            except Exception:
                pass
        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass
        return True  # consume — prevent fastplotlib from also zooming


def install_zoom_filter(fig, subplots: list, parent) -> AxisZoomFilter:
    """Create and register the filter at app level. Call once per view."""
    canvas = fig.canvas
    filt   = AxisZoomFilter(fig, subplots, canvas, parent)
    QApplication.instance().installEventFilter(filt)
    return filt


class SyncedZoomFilter(QObject):
    """Zoom filter that synchronises ALL subplots together on every scroll event.

    Plain scroll   → zoom all subplots (width + height) by the same factor.
    Cmd  + scroll  → zoom Y only across all subplots.
    Shift + scroll → zoom X only across all subplots.

    All scroll events are consumed so fastplotlib never gets a chance to zoom
    just the hovered subplot independently.
    """

    _STEP = 1.12   # zoom factor per scroll notch

    def __init__(self, fig, subplots: list, canvas, parent=None):
        super().__init__(parent)
        self._fig      = fig
        self._subplots = subplots
        self._canvas   = canvas

    def eventFilter(self, obj, event) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False

        # Only act when the event is inside our canvas hierarchy
        w = obj
        while w is not None:
            if w is self._canvas:
                break
            w = w.parent()
        else:
            return False

        delta = event.angleDelta().y()
        if delta == 0:
            return False

        mods   = event.modifiers()
        cmd    = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift  = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        factor = self._STEP ** (delta / 120)

        for sp in self._subplots:
            try:
                state = sp.camera.get_state()
                new   = dict(state)
                if cmd:
                    new["height"] = abs(state["height"]) / factor
                elif shift:
                    new["width"]  = abs(state["width"])  / factor
                else:
                    new["width"]  = abs(state["width"])  / factor
                    new["height"] = abs(state["height"]) / factor
                new["zoom"] = 1.0
                sp.camera.set_state(new)
            except Exception:
                pass

        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass

        return True   # consume — prevent fpl from zooming only the hovered subplot


def install_synced_zoom_filter(fig, subplots: list, parent) -> SyncedZoomFilter:
    """Like install_zoom_filter but uses SyncedZoomFilter (all subplots move together)."""
    canvas = fig.canvas
    filt   = SyncedZoomFilter(fig, subplots, canvas, parent)
    QApplication.instance().installEventFilter(filt)
    return filt
