"""ISI histogram widget — inter-spike interval distribution per cluster."""
from __future__ import annotations
import io, logging
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy

logger = logging.getLogger(__name__)

_BG = "#1e1e1e"; _AXES = "#2a2a2a"
CLUSTER_COLORS = [
    (0.122,0.467,0.706),(1.000,0.498,0.055),(0.173,0.627,0.173),
    (0.839,0.153,0.157),(0.580,0.404,0.741),(0.549,0.337,0.294),
    (0.890,0.467,0.761),(0.498,0.498,0.498),(0.737,0.741,0.133),(0.090,0.745,0.812),
]

class ISIWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel("Select a cluster")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setStyleSheet("background:#1e1e1e; color:#aaaaaa; font-size:12px;")
        layout.addWidget(self._label)

    def set_spike_data(self, spike_data: dict[int, np.ndarray]) -> None:
        """spike_data: {cluster_id -> (n_spikes, 2) array [time, amplitude]}"""
        if not spike_data:
            self._label.setText("No data"); return
        try:
            pixmap = self._render(spike_data)
            self._label.setPixmap(pixmap.scaled(
                self._label.width(), self._label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        except Exception as exc:
            logger.exception("ISI render failed")
            self._label.setText(f"ISI error:\n{exc}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        pm = self._label.pixmap()
        if pm and not pm.isNull():
            self._label.setPixmap(pm.scaled(
                self._label.width(), self._label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def _render(self, spike_data: dict[int, np.ndarray]) -> QPixmap:
        import matplotlib; matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = Figure(figsize=(4, 2.5), facecolor=_BG)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.18)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(_AXES)
        for spine in ax.spines.values(): spine.set_color("#555")
        ax.tick_params(colors="#aaa", labelsize=8)
        ax.set_xlabel("ISI (ms)", color="#aaa", fontsize=9)
        ax.set_ylabel("Count", color="#aaa", fontsize=9)
        ax.set_title("ISI histogram", color="#ccc", fontsize=9, pad=4)

        bins = np.logspace(np.log10(0.5), np.log10(500), 60)
        for idx, (cid, data) in enumerate(spike_data.items()):
            times = np.sort(data[:, 0])
            isis = np.diff(times) * 1000  # ms
            if len(isis) == 0: continue
            r, g, b = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
            ax.hist(isis, bins=bins, color=(r,g,b,0.7), label=f"#{cid}", histtype="stepfilled")

        ax.axvline(1.5, color="#ff4444", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 500)
        if len(spike_data) > 1:
            ax.legend(fontsize=7, framealpha=0.3, labelcolor="#ccc",
                      facecolor="#333", edgecolor="#555")

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        import matplotlib.pyplot as plt; plt.close(fig)
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.read()))
