"""Feature cloud view — dense point cloud style features."""
from __future__ import annotations

import io
import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

_BG = "#1e1e1e"
_AXES = "#2a2a2a"


class FeatureCloudWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel("Select a cluster")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setStyleSheet("background:#1e1e1e; color:#aaaaaa; font-size:12px;")
        layout.addWidget(self._label)

    def set_feature_data(self, feature_data: dict[int, np.ndarray]) -> None:
        if not feature_data:
            self._label.setText("No features")
            self._label.setPixmap(QPixmap())
            return
        try:
            pm = self._render(feature_data)
            self._label.setPixmap(pm.scaled(
                self._label.width(), self._label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        except Exception as exc:
            logger.exception("Feature cloud render failed")
            self._label.setText(f"Feature cloud error:\n{exc}")

    def _render(self, feature_data: dict[int, np.ndarray]) -> QPixmap:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        all_points = []
        for arr in feature_data.values():
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim == 1:
                a = a[:, np.newaxis]
            if a.ndim > 2:
                a = a.reshape(a.shape[0], -1)
            if a.shape[1] < 2:
                a = np.column_stack([a[:, 0], np.zeros(a.shape[0], dtype=np.float32)])
            else:
                a = a[:, :2]
            all_points.append(a)
        pts = np.vstack(all_points) if all_points else np.empty((0, 2), dtype=np.float32)
        if len(pts) > 50_000:
            rng = np.random.default_rng(seed=1234)
            sel = rng.choice(len(pts), 50_000, replace=False)
            pts = pts[sel]

        fig = Figure(figsize=(4.0, 3.2), facecolor=_BG)
        fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.13)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(_AXES)
        ax.set_title("Feature cloud", color="#ccc", fontsize=9, pad=4)
        ax.tick_params(colors="#aaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#555")
        if len(pts):
            ax.hexbin(pts[:, 0], pts[:, 1], gridsize=70, bins="log", cmap="magma", mincnt=1)

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        import matplotlib.pyplot as plt
        plt.close(fig)
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.read()))
