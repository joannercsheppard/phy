"""Correlogram view — auto/cross-correlograms (fastplotlib).

A fixed 2×2 figure is created once at startup.  When 1 cluster is selected
only [0,0] is used (auto-correlogram).  When 2 clusters are selected all
four cells are used.  No figure recreation avoids a rendercanvas crash where
the render timer fires on a just-deleted QRenderWidget.

Ctrl+scroll  — widen / narrow the lag window (re-bins the data)
Alt+scroll   — increase / decrease number of bins (finer / coarser)
"""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget

from phy_remote.client.views._colors import cluster_color
from phy_remote.client.views._graphics import safe_delete

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_MS = 50.0    # ± window in ms
_DEFAULT_N_BINS    = 100     # number of bins across the full window
_MAX_T             = 2_500   # max spikes per cluster for lag computation
_MAX_CLUSTERS      = 2       # fixed grid size


class CorrelogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._fig = None
        self._line_grid: dict[tuple[int, int], object] = {}

        # Mutable window / resolution — changed by scroll events
        self._window_ms = _DEFAULT_WINDOW_MS
        self._n_bins    = _DEFAULT_N_BINS

        # Cache the last spike-time dict so we can re-render on param change
        self._last_spike_times: dict[int, np.ndarray] = {}

        # Create a fixed 2×2 figure once — never recreated
        try:
            import fastplotlib as fpl
            self._fig = fpl.Figure(shape=(_MAX_CLUSTERS, _MAX_CLUSTERS), canvas="qt")
            for r in range(_MAX_CLUSTERS):
                for c in range(_MAX_CLUSTERS):
                    sp = self._fig[r, c]
                    try:
                        sp.camera = "2d"
                        sp.axes.visible = False
                        sp.title.visible = False
                    except Exception:
                        pass
            self._fig.show()
            canvas = self._fig.canvas
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumSize(160, 120)
            self._layout.addWidget(canvas)
            self._wheel_filter = _HistWheelFilter(self, canvas, self)
            QApplication.instance().installEventFilter(self._wheel_filter)
        except Exception as exc:
            logger.warning("CorrelogramWidget: fastplotlib init failed: %s", exc)

    def set_spike_times(self, spike_times: dict[int, np.ndarray]) -> None:
        if not spike_times:
            return
        self._last_spike_times = spike_times
        self._rerender()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _rerender(self) -> None:
        try:
            self._render(self._last_spike_times)
        except Exception as exc:
            logger.exception("CorrelogramWidget render failed: %s", exc)

    def _render(self, spike_times: dict[int, np.ndarray]) -> None:
        if self._fig is None:
            return
        items = list(spike_times.items())[:_MAX_CLUSTERS]
        n = len(items)

        bins = np.linspace(-self._window_ms, self._window_ms, self._n_bins + 1)
        window_s = self._window_ms / 1000.0

        active_keys: set[tuple[int, int]] = set()

        for i in range(n):
            cid_i, t_i = items[i]
            t_i = self._subsample(t_i, cid_i)
            for j in range(n):
                cid_j, t_j = items[j]
                t_j = self._subsample(t_j, cid_j)

                lags = self._compute_lags(t_i, t_j, window_s, symmetric=(i == j))
                counts, _ = np.histogram(lags, bins=bins)
                xy = self._step_hist(counts, bins.astype(np.float32))

                if i == j:
                    color = cluster_color(i, alpha=0.85)
                else:
                    r0, g0, b0, _ = cluster_color(i, alpha=1.0)
                    r1, g1, b1, _ = cluster_color(j, alpha=1.0)
                    color = ((r0 + r1) / 2, (g0 + g1) / 2, (b0 + b1) / 2, 0.65)

                sp = self._fig[i, j]
                key = (i, j)
                active_keys.add(key)

                if key in self._line_grid:
                    old = self._line_grid[key]
                    if old.data.value.shape[0] == len(xy):
                        xyz = np.zeros((len(xy), 3), dtype=np.float32)
                        xyz[:, :2] = xy
                        old.data[:] = xyz
                    else:
                        safe_delete(sp, old)
                        if len(xy) > 1:
                            self._line_grid[key] = sp.add_line(xy, colors=color, thickness=1.5)
                        else:
                            self._line_grid.pop(key, None)
                elif len(xy) > 1:
                    self._line_grid[key] = sp.add_line(xy, colors=color, thickness=1.5)

        # Clear graphics from cells no longer in use (e.g. 2→1 cluster)
        for key in list(self._line_grid.keys()):
            if key not in active_keys:
                i, j = key
                safe_delete(self._fig[i, j], self._line_grid.pop(key))

        QTimer.singleShot(50, self._auto_scale)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _subsample(times: np.ndarray, seed: int) -> np.ndarray:
        t = np.asarray(times, dtype=np.float64)
        if len(t) > _MAX_T:
            t = np.sort(np.random.default_rng(seed).choice(t, _MAX_T, replace=False))
        return t

    @staticmethod
    def _compute_lags(
        t0: np.ndarray, t1: np.ndarray, window_s: float, symmetric: bool
    ) -> np.ndarray:
        t0 = np.sort(t0)
        t1 = np.sort(t1)
        out = []
        for t in t0:
            lo = np.searchsorted(t1, t - window_s)
            hi = np.searchsorted(t1, t + window_s)
            if hi > lo:
                d = t1[lo:hi] - t
                if symmetric:
                    d = d[d != 0.0]
                out.append(d)
        if not out:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(out) * 1000.0

    @staticmethod
    def _step_hist(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
        x_stair = np.repeat(edges, 2)[1:-1]
        y_stair = np.repeat(counts, 2).astype(np.float32)
        x_full = np.concatenate([[edges[0]], x_stair, [edges[-1]]]).astype(np.float32)
        y_full = np.concatenate([[0.0], y_stair, [0.0]]).astype(np.float32)
        return np.column_stack([x_full, y_full])

    def _auto_scale(self) -> None:
        if self._fig is None:
            return
        try:
            n = max(1, len(self._last_spike_times))
            for r in range(min(n, _MAX_CLUSTERS)):
                for c in range(min(n, _MAX_CLUSTERS)):
                    sp = self._fig[r, c]
                    sp.auto_scale()
                    state = sp.camera.get_state()
                    pos = state['position'].copy()
                    pos[2] = state['depth'] / 2
                    sp.camera.set_state({
                        'position':        pos,
                        'fov':             0.0,
                        'width':           abs(state['width'])  * 1.1,
                        'height':          abs(state['height']) * 1.1,
                        'depth':           state['depth'],
                        'zoom':            1.0,
                        'maintain_aspect': False,
                    })
            self._fig.canvas.request_draw()
        except Exception as exc:
            logger.debug("CorrelogramWidget auto_scale: %s", exc)


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
            v._window_ms = max(1.0, v._window_ms / factor)
            v._rerender()
        elif alt:
            v._n_bins = max(10, int(v._n_bins * factor))
            v._rerender()

        return True
