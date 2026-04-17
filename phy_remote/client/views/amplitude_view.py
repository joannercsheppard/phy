"""Amplitude-over-time widget — spike amplitude scatter per cluster (fastplotlib)."""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from phy_remote.client.views._zoom import install_zoom_filter
from phy_remote.client.views._colors import cluster_color
from phy_remote.client.views._graphics import safe_delete

logger = logging.getLogger(__name__)


class AmplitudeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel("Amplitude")
        hdr.setFixedHeight(20)
        hdr.setStyleSheet(
            "background:#252525;color:#aaa;font-size:11px;"
            "padding-left:6px;border-bottom:1px solid #333;"
        )
        layout.addWidget(hdr)

        self._fig     = None
        self._subplot = None
        self._scatter_graphics: dict[int, object] = {}
        self._cid_to_slot: dict[int, int] = {}

        # Add canvas directly to layout — always visible so wgpu can
        # create its Metal surface before any data arrives.
        try:
            import fastplotlib as fpl
            self._fig     = fpl.Figure(canvas="qt")
            self._subplot = self._fig[0, 0]
            try:
                self._subplot.camera = "2d"
            except Exception:
                pass
            try:
                self._subplot.axes.visible = False
                self._subplot.title.visible = False
            except Exception:
                pass
            self._fig.show()   # wires up wgpu render loop
            canvas = self._fig.canvas
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumSize(160, 120)
            layout.addWidget(canvas)  # reparents canvas into our widget
            self._zoom_filter = install_zoom_filter(self._fig, [self._subplot], self)
        except Exception as exc:
            logger.warning("AmplitudeWidget: fastplotlib init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_spike_data(self, spike_data: dict[int, np.ndarray]) -> None:
        """spike_data: {cluster_id -> (n_spikes, 2) array [time, amplitude]}"""
        if self._fig is None or not spike_data:
            return
        try:
            self._render(spike_data)
        except Exception as exc:
            logger.exception("AmplitudeWidget render failed: %s", exc)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, spike_data: dict[int, np.ndarray]) -> None:
        sp = self._subplot

        for cid in list(self._scatter_graphics.keys()):
            if cid not in spike_data:
                safe_delete(sp, self._scatter_graphics.pop(cid))
                self._cid_to_slot.pop(cid, None)

        for idx, (cid, data) in enumerate(spike_data.items()):
            times = data[:, 0].astype(np.float32)
            amps  = data[:, 1].astype(np.float32)
            xy = np.column_stack([times, amps]).astype(np.float32)
            rgba = np.array(cluster_color(idx, alpha=0.6), dtype=np.float32)
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
                    self._scatter_graphics[cid] = sp.add_scatter(xy, sizes=3, colors=colors_arr)
            else:
                if cid in self._scatter_graphics:
                    safe_delete(sp, self._scatter_graphics.pop(cid))
                self._scatter_graphics[cid] = sp.add_scatter(xy, sizes=3, colors=colors_arr)
            self._cid_to_slot[cid] = idx

        QTimer.singleShot(50, lambda: self._auto_scale(sp))

    def _auto_scale(self, sp) -> None:
        try:
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
            logger.debug("AmplitudeWidget auto_scale: %s", exc)
