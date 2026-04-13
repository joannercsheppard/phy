"""Feature view — Phy-style matrix of feature projections."""
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


class FeatureViewWidget(QWidget):
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
            logger.exception("Feature view render failed")
            self._label.setText(f"Feature view error:\n{exc}")

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            a = a[:, np.newaxis]
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
        if a.shape[1] < 2:
            a = np.column_stack([a[:, 0], np.zeros(a.shape[0], dtype=np.float32)])
        return a

    @staticmethod
    def _to_phy_like_dims(arr: np.ndarray) -> np.ndarray:
        """
        Convert model.get_features output to 4 dims similar to Phy defaults:
        [ch0-PC1, ch1-PC1, ch0-PC2, ch1-PC2] when available.
        """
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 3:
            n_spikes, n_ch, n_pc = a.shape
            c0 = 0
            c1 = min(1, n_ch - 1)
            p0 = 0
            p1 = min(1, n_pc - 1)
            cols = [
                a[:, c0, p0],
                a[:, c1, p0],
                a[:, c0, p1],
                a[:, c1, p1],
            ]
            return np.column_stack(cols).astype(np.float32)
        if a.ndim == 2:
            if a.shape[1] < 4:
                pad = np.zeros((a.shape[0], 4 - a.shape[1]), dtype=np.float32)
                a = np.concatenate([a, pad], axis=1)
            return a[:, :4].astype(np.float32)
        if a.ndim == 1:
            z = np.zeros((len(a), 4), dtype=np.float32)
            z[:, 0] = a
            return z
        return np.zeros((0, 4), dtype=np.float32)

    def _render(self, feature_data: dict[int, np.ndarray]) -> QPixmap:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        projected = [(cid, self._to_phy_like_dims(arr)) for cid, arr in feature_data.items()]
        dims = [0, 1, 2, 3]

        fig = Figure(figsize=(4.8, 4.6), facecolor=_BG)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.06, wspace=0.08, hspace=0.08)
        for r, dy in enumerate(dims):
            for c, dx in enumerate(dims):
                ax = fig.add_subplot(4, 4, r * 4 + c + 1)
                ax.set_facecolor(_AXES)
                for spine in ax.spines.values():
                    spine.set_color("#3f3f3f")
                ax.tick_params(colors="#777", labelsize=6, length=1)
                if r < 3:
                    ax.set_xticklabels([])
                if c > 0:
                    ax.set_yticklabels([])
                if r == 3:
                    ax.set_xlabel(f"PC{dx+1}", color="#888", fontsize=6, labelpad=1)
                if c == 0:
                    ax.set_ylabel(f"PC{dy+1}", color="#888", fontsize=6, labelpad=1)

                for i, (cid, arr) in enumerate(projected):
                    x = arr[:, dx]
                    y = arr[:, dy]
                    n = len(x)
                    if n > 2500:
                        rng = np.random.default_rng(seed=cid * 31 + r * 7 + c)
                        sel = rng.choice(n, 2500, replace=False)
                        x, y = x[sel], y[sel]
                    col = _COLORS[i % len(_COLORS)]
                    ax.scatter(x, y, s=1, color=col, alpha=0.30, linewidths=0, rasterized=True)

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        import matplotlib.pyplot as plt
        plt.close(fig)
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.read()))


# Backward-compatible alias for existing imports.
FeatureCloudWidget = FeatureViewWidget
