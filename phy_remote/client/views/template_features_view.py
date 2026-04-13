"""Template features view — 2D projection scatter for selected clusters."""
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
_COLORS = [
    (0.92, 0.15, 0.15),
    (0.15, 0.47, 0.90),
    (0.10, 0.72, 0.42),
    (0.92, 0.58, 0.08),
]


class TemplateFeaturesWidget(QWidget):
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
        if len(feature_data) != 2:
            self._label.setText("Template features needs exactly 2 selected clusters")
            self._label.setPixmap(QPixmap())
            return
        try:
            pm = self._render(feature_data)
            self._label.setPixmap(pm.scaled(
                self._label.width(), self._label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        except Exception as exc:
            logger.exception("Template features render failed")
            self._label.setText(f"Template features error:\n{exc}")

    def _flatten(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            return arr[:, np.newaxis]
        if arr.ndim > 2:
            return arr.reshape(arr.shape[0], -1)
        return arr

    def _render(self, feature_data: dict[int, np.ndarray]) -> QPixmap:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = Figure(figsize=(3.8, 3.2), facecolor=_BG)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.14)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(_AXES)
        ax.set_xlabel("Feature 1", color="#aaa", fontsize=9)
        ax.set_ylabel("Feature 2", color="#aaa", fontsize=9)
        ax.set_title("Template features", color="#ccc", fontsize=9, pad=4)
        ax.tick_params(colors="#aaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#555")

        items = list(feature_data.items())
        cid0, arr0 = items[0]
        cid1, arr1 = items[1]
        v0 = self._flatten(arr0)
        v1 = self._flatten(arr1)

        # Phy template-feature style approximation: project each spike vector onto
        # the two selected cluster reference vectors.
        r0 = v0.mean(axis=0)
        r1 = v1.mean(axis=0)
        r0 /= max(1e-12, float(np.linalg.norm(r0)))
        r1 /= max(1e-12, float(np.linalg.norm(r1)))

        for i, (cid, vecs) in enumerate([(cid0, v0), (cid1, v1)]):
            x = vecs @ r0
            y = vecs @ r1
            n = len(vecs)
            if n > 8000:
                rng = np.random.default_rng(seed=cid)
                sel = rng.choice(n, 8000, replace=False)
                x, y = x[sel], y[sel]
            c = _COLORS[i % len(_COLORS)]
            ax.scatter(x, y, s=1, color=c, alpha=0.35, linewidths=0, label=f"#{cid}", rasterized=True)
        if len(feature_data) > 1:
            ax.legend(fontsize=7, framealpha=0.3, labelcolor="#ccc", facecolor="#333", edgecolor="#555")

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        import matplotlib.pyplot as plt
        plt.close(fig)
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.read()))
