"""Raster view — spike-time raster for selected clusters."""
from __future__ import annotations

import io
import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

_COLORS = [
    (0.92, 0.15, 0.15),
    (0.15, 0.47, 0.90),
    (0.10, 0.72, 0.42),
    (0.92, 0.58, 0.08),
]


class RasterWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel("Select a cluster")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setStyleSheet("background:#1e1e1e; color:#aaaaaa; font-size:12px;")
        layout.addWidget(self._label)

    def set_spike_times(self, spike_times: dict[int, np.ndarray]) -> None:
        if not spike_times:
            self._label.setText("No spikes")
            self._label.setPixmap(QPixmap())
            return
        try:
            pm = self._render(spike_times)
            self._label.setPixmap(pm.scaled(
                self._label.width(), self._label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        except Exception as exc:
            logger.exception("Raster render failed")
            self._label.setText(f"Raster error:\n{exc}")

    def _render(self, spike_times: dict[int, np.ndarray]) -> QPixmap:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = Figure(figsize=(3.6, 2.2), facecolor="#1e1e1e")
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.18)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("#2a2a2a")
        ax.set_title("Raster", color="#ccc", fontsize=9, pad=4)
        ax.set_xlabel("Time (s)", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#555")

        for i, (cid, times) in enumerate(spike_times.items()):
            t = np.asarray(times, dtype=np.float64)
            if len(t) > 4000:
                rng = np.random.default_rng(seed=cid)
                sel = rng.choice(len(t), 4000, replace=False)
                t = np.sort(t[sel])
            y = np.full(len(t), i + 1, dtype=np.float32)
            c = _COLORS[i % len(_COLORS)]
            ax.scatter(t, y, s=2, color=c, alpha=0.8, linewidths=0, rasterized=True)

        ax.set_yticks(np.arange(1, len(spike_times) + 1))
        ax.set_yticklabels([str(cid) for cid in spike_times.keys()], color="#aaa", fontsize=7)

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        import matplotlib.pyplot as plt
        plt.close(fig)
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.read()))
