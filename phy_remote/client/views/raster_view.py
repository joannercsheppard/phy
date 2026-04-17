"""Raster view — spike-time raster per cluster (fastplotlib)."""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from phy_remote.client.views._zoom import install_zoom_filter
from phy_remote.client.views._colors import cluster_color
from phy_remote.client.views._graphics import safe_delete

logger = logging.getLogger(__name__)
_MAX_POINTS = 5_000


class RasterWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._fig = None
        self._subplot = None
        self._scatter_graphics: dict[int, object] = {}
        self._cid_to_slot: dict[int, int] = {}

        try:
            import fastplotlib as fpl
            self._fig = fpl.Figure(canvas="qt")
            self._subplot = self._fig[0, 0]
            try:
                self._subplot.camera = "2d"
                self._subplot.axes.visible = False
                self._subplot.title.visible = False
            except Exception:
                pass
            self._fig.show()
            canvas = self._fig.canvas
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumSize(160, 120)
            layout.addWidget(canvas)
            self._zoom_filter = install_zoom_filter(self._fig, [self._subplot], self)
        except Exception as exc:
            logger.warning("RasterWidget: fastplotlib init failed: %s", exc)

    def set_spike_times(self, spike_times: dict[int, np.ndarray]) -> None:
        if self._fig is None or not spike_times:
            return
        try:
            self._render(spike_times)
        except Exception as exc:
            logger.exception("RasterWidget render failed: %s", exc)

    def _render(self, spike_times: dict[int, np.ndarray]) -> None:
        sp = self._subplot

        for cid in list(self._scatter_graphics.keys()):
            if cid not in spike_times:
                safe_delete(sp, self._scatter_graphics.pop(cid))
                self._cid_to_slot.pop(cid, None)

        for idx, (cid, times) in enumerate(spike_times.items()):
            t = np.asarray(times, dtype=np.float32)
            n = len(t)
            if n > _MAX_POINTS:
                rng = np.random.default_rng(seed=cid)
                sel = rng.choice(n, _MAX_POINTS, replace=False)
                t = np.sort(t[sel])
            y = np.full(len(t), float(idx + 1), dtype=np.float32)
            xy = np.column_stack([t, y]).astype(np.float32)
            rgba = np.array(cluster_color(idx, alpha=0.8), dtype=np.float32)
            colors_arr = np.broadcast_to(rgba, (len(xy), 4)).copy()

            slot_changed = self._cid_to_slot.get(cid) != idx
            if cid in self._scatter_graphics and not slot_changed:
                sc = self._scatter_graphics[cid]
                if sc.data.value.shape[0] == len(xy):
                    xyz = np.zeros((len(xy), 3), dtype=np.float32)
                    xyz[:, :2] = xy
                    sc.data[:] = xyz
                else:
                    safe_delete(sp, sc)
                    self._scatter_graphics[cid] = sp.add_scatter(xy, sizes=2, colors=colors_arr)
            else:
                if cid in self._scatter_graphics:
                    safe_delete(sp, self._scatter_graphics.pop(cid))
                self._scatter_graphics[cid] = sp.add_scatter(xy, sizes=2, colors=colors_arr)
            self._cid_to_slot[cid] = idx

        QTimer.singleShot(50, self._auto_scale)

    def _auto_scale(self) -> None:
        if self._fig is None:
            return
        try:
            sp = self._subplot
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
            logger.debug("RasterWidget auto_scale: %s", exc)
