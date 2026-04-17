"""ISI histogram widget — inter-spike interval distribution (fastplotlib).

X-axis: log10(ISI / ms).  Tick labels are added manually at key ms positions.
A yellow line marks the 2 ms refractory boundary (ISI violations).

Layout mirrors the original phy ISIView: each selected cluster gets its own
subplot row, stacked vertically.  A fixed 2-row figure is created once at
startup; the second row is cleared when only one cluster is selected.

Ctrl+scroll  — widen / narrow the x-axis window (re-bins the data)
Alt+scroll   — increase / decrease number of bins (finer / coarser)
"""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QEvent, QObject, QTimer
from PyQt6.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget

from phy_remote.client.views._colors import cluster_color
from phy_remote.client.views._graphics import safe_delete

logger = logging.getLogger(__name__)

# Defaults matching phy's ISIView: 0.5 ms – 500 ms, log-spaced, 60 bins
_DEFAULT_X_MIN_MS = 0.5
_DEFAULT_X_MAX_MS = 500.0
_DEFAULT_N_BINS   = 60

_REFRACTORY_MS = 2.0
_TICK_MS       = [1, 2, 5, 10, 50, 100, 500]
_MAX_CLUSTERS  = 2


class ISIWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._fig       = None
        self._subplots: list = []

        # Per-row state (indexed 0..1)
        self._hist_graphics: list       = [None, None]
        self._viol_lines:    list       = [None, None]
        self._tick_texts:    list[list] = [[], []]

        # Mutable parameters — changed by wheel events
        self._x_min_ms = _DEFAULT_X_MIN_MS
        self._x_max_ms = _DEFAULT_X_MAX_MS
        self._n_bins   = _DEFAULT_N_BINS

        # Cache last spike data for re-render on param change
        self._last_spike_data: dict[int, np.ndarray] = {}

        try:
            import fastplotlib as fpl
            self._fig = fpl.Figure(shape=(_MAX_CLUSTERS, 1), canvas="qt")
            for r in range(_MAX_CLUSTERS):
                sp = self._fig[r, 0]
                try:
                    sp.camera = "2d"
                    sp.axes.visible = True
                    sp.title.visible = False
                except Exception:
                    pass
                self._subplots.append(sp)
            self._fig.show()
            canvas = self._fig.canvas
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumSize(160, 120)
            layout.addWidget(canvas)
            self._wheel_filter = _HistWheelFilter(self, canvas, self)
            QApplication.instance().installEventFilter(self._wheel_filter)
        except Exception as exc:
            logger.warning("ISIWidget: fastplotlib init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_spike_data(self, spike_data: dict[int, np.ndarray]) -> None:
        """spike_data: {cluster_id -> (n_spikes, 2) array [time, amplitude]}"""
        if self._fig is None:
            return
        self._last_spike_data = spike_data
        self._rerender()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _rerender(self) -> None:
        if self._fig is None:
            return
        try:
            self._render(self._last_spike_data)
        except Exception as exc:
            logger.exception("ISIWidget render failed: %s", exc)

    def _render(self, spike_data: dict[int, np.ndarray]) -> None:
        items = list(spike_data.items())[:_MAX_CLUSTERS]
        n = len(items)

        x_min = max(self._x_min_ms, 1e-3)
        x_max = max(self._x_max_ms, x_min * 2.0)
        bins     = np.logspace(np.log10(x_min), np.log10(x_max), self._n_bins + 1)
        log_bins = np.log10(bins).astype(np.float32)

        for row in range(_MAX_CLUSTERS):
            sp = self._subplots[row]

            if row >= n:
                # Clear this row — cluster no longer selected
                if self._hist_graphics[row] is not None:
                    safe_delete(sp, self._hist_graphics[row])
                    self._hist_graphics[row] = None
                if self._viol_lines[row] is not None:
                    safe_delete(sp, self._viol_lines[row])
                    self._viol_lines[row] = None
                for t in self._tick_texts[row]:
                    try:
                        sp.delete_graphic(t)
                    except Exception:
                        pass
                self._tick_texts[row].clear()
                continue

            cid, data = items[row]
            times   = np.sort(data[:, 0])
            isis_ms = np.diff(times) * 1000.0

            if len(isis_ms) == 0:
                continue

            counts, _ = np.histogram(isis_ms, bins=bins)
            max_count = int(counts.max())

            n_viol = int(np.sum(isis_ms < _REFRACTORY_MS))
            pct    = 100.0 * n_viol / max(1, len(isis_ms))
            logger.debug("Cluster %d ISI violations: %.1f%% (%d / %d)",
                         cid, pct, n_viol, len(isis_ms))

            xy   = self._step_hist(counts, log_bins)
            rgba = np.array(cluster_color(row, alpha=0.7), dtype=np.float32)

            if self._hist_graphics[row] is not None:
                old = self._hist_graphics[row]
                if old.data.value.shape[0] == len(xy):
                    xyz = np.zeros((len(xy), 3), dtype=np.float32)
                    xyz[:, :2] = xy
                    old.data[:] = xyz
                else:
                    safe_delete(sp, old)
                    self._hist_graphics[row] = sp.add_line(xy, colors=rgba, thickness=1.5)
            else:
                self._hist_graphics[row] = sp.add_line(xy, colors=rgba, thickness=1.5)

            # ------------------------------------------------------------------
            # 2 ms refractory violation line
            # ------------------------------------------------------------------
            y_top  = float(max_count) * 1.05 if max_count > 0 else 1.0
            x_viol = float(np.log10(_REFRACTORY_MS))
            viol_xy = np.array([[x_viol, 0.0], [x_viol, y_top]], dtype=np.float32)

            if self._viol_lines[row] is not None:
                old = self._viol_lines[row]
                try:
                    if old.data.value.shape[0] == 2:
                        xyz = np.zeros((2, 3), dtype=np.float32)
                        xyz[:, :2] = viol_xy
                        old.data[:] = xyz
                    else:
                        sp.delete_graphic(old)
                        self._viol_lines[row] = sp.add_line(
                            viol_xy, colors=(1.0, 0.9, 0.0, 1.0), thickness=2.5
                        )
                except Exception:
                    self._viol_lines[row] = None
            if self._viol_lines[row] is None:
                try:
                    self._viol_lines[row] = sp.add_line(
                        viol_xy, colors=(1.0, 0.9, 0.0, 1.0), thickness=2.5
                    )
                except Exception as exc:
                    logger.debug("Could not draw violation line: %s", exc)

            # ------------------------------------------------------------------
            # X-axis tick labels at key ms values
            # ------------------------------------------------------------------
            self._update_tick_labels(row, sp, y_top)

        QTimer.singleShot(50, self._auto_scale)

    def _update_tick_labels(self, row: int, sp, y_top: float) -> None:
        """Draw text labels at major ms positions on the x-axis."""
        for t in self._tick_texts[row]:
            try:
                sp.delete_graphic(t)
            except Exception:
                pass
        self._tick_texts[row].clear()

        x_min = max(self._x_min_ms, 1e-3)
        x_max = max(self._x_max_ms, x_min * 2.0)
        y_label = -y_top * 0.08

        for ms in _TICK_MS:
            if ms < x_min * 0.5 or ms > x_max * 2.0:
                continue
            lx = float(np.log10(ms))
            try:
                txt = sp.add_text(f"{ms}ms", position=(lx, y_label, 0),
                                  font_size=10, face_color="white")
                self._tick_texts[row].append(txt)
            except Exception:
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _step_hist(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
        x_stair = np.repeat(edges, 2)[1:-1]
        y_stair = np.repeat(counts, 2).astype(np.float32)
        x_full  = np.concatenate([[edges[0]], x_stair, [edges[-1]]]).astype(np.float32)
        y_full  = np.concatenate([[0.0], y_stair, [0.0]]).astype(np.float32)
        return np.column_stack([x_full, y_full])

    def _auto_scale(self) -> None:
        if self._fig is None:
            return
        n = max(1, len(self._last_spike_data))
        for row in range(min(n, _MAX_CLUSTERS)):
            try:
                sp = self._subplots[row]
                sp.auto_scale()
                state = sp.camera.get_state()
                pos = state['position'].copy()
                pos[2] = state['depth'] / 2
                sp.camera.set_state({
                    'position':        pos,
                    'fov':             0.0,
                    'width':           abs(state['width'])  * 1.15,
                    'height':          abs(state['height']) * 1.25,
                    'depth':           state['depth'],
                    'zoom':            1.0,
                    'maintain_aspect': False,
                })
            except Exception as exc:
                logger.debug("ISIWidget auto_scale row %d: %s", row, exc)
        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Wheel filter: Ctrl+scroll → widen/narrow window; Alt+scroll → more/fewer bins
# ---------------------------------------------------------------------------

class _HistWheelFilter(QObject):
    _STEP = 1.2

    def __init__(self, view, canvas, parent=None):
        super().__init__(parent)
        self._view   = view
        self._canvas = canvas

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.Wheel:
            return False
        w = obj
        while w is not None:
            if w is self._canvas:
                break
            w = w.parent()
        else:
            return False

        from PyQt6.QtCore import Qt
        mods  = event.modifiers()
        delta = event.angleDelta().y()
        if delta == 0:
            return False

        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        alt  = bool(mods & Qt.KeyboardModifier.AltModifier)

        if not ctrl and not alt:
            return False

        factor = self._STEP ** (delta / 120)
        v = self._view

        if ctrl:
            v._x_max_ms = max(v._x_min_ms * 2, v._x_max_ms / factor)
            v._rerender()
        elif alt:
            v._n_bins = max(10, int(v._n_bins * factor))
            v._rerender()

        return True
