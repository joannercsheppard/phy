"""Correlogram view — auto/cross-correlograms for selected clusters."""
from __future__ import annotations

import io
import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class CorrelogramWidget(QWidget):
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
            logger.exception("Correlogram render failed")
            self._label.setText(f"Correlogram error:\n{exc}")

    def _render(self, spike_times: dict[int, np.ndarray]) -> QPixmap:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        items = list(spike_times.items())[:4]
        n = len(items)
        fig = Figure(figsize=(3.2, 3.0), facecolor="#1e1e1e")
        fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.08, wspace=0.15, hspace=0.15)
        bins = np.linspace(-50.0, 50.0, 101)
        window_s = 0.05

        def _lags_ms(t0: np.ndarray, t1: np.ndarray, symmetric: bool) -> np.ndarray:
            # Pairwise lag collection within +-window using searchsorted windows.
            out = []
            t1 = np.asarray(t1, dtype=np.float64)
            for t in np.asarray(t0, dtype=np.float64):
                lo = np.searchsorted(t1, t - window_s, side="left")
                hi = np.searchsorted(t1, t + window_s, side="right")
                if hi > lo:
                    d = t1[lo:hi] - t
                    if symmetric:
                        d = d[d != 0.0]
                    out.append(d)
            if not out:
                return np.empty(0, dtype=np.float64)
            return np.concatenate(out) * 1000.0

        for i in range(n):
            cid_i, t_i = items[i]
            t_i = np.asarray(t_i, dtype=np.float64)
            if len(t_i) > 2500:
                t_i = np.sort(np.random.default_rng(cid_i).choice(t_i, 2500, replace=False))
            for j in range(n):
                cid_j, t_j = items[j]
                t_j = np.asarray(t_j, dtype=np.float64)
                if len(t_j) > 2500:
                    t_j = np.sort(np.random.default_rng(cid_j).choice(t_j, 2500, replace=False))
                ax = fig.add_subplot(n, n, i * n + j + 1)
                ax.set_facecolor("#2a2a2a")
                for spine in ax.spines.values():
                    spine.set_color("#444")
                ax.tick_params(colors="#888", labelsize=6, length=2)
                lags = _lags_ms(t_i, t_j, symmetric=(i == j))
                color = (0.92, 0.15, 0.15, 0.85) if i == j else (0.15, 0.47, 0.90, 0.75)
                if len(lags):
                    ax.hist(lags, bins=bins, color=color)
                ax.axvline(0.0, color="#dddddd", lw=0.6, alpha=0.7)
                if j == 0:
                    ax.set_ylabel(str(cid_i), color="#aaa", fontsize=6)
                else:
                    ax.set_yticklabels([])
                if i == n - 1:
                    ax.set_xlabel(str(cid_j), color="#aaa", fontsize=6)
                else:
                    ax.set_xticklabels([])

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        import matplotlib.pyplot as plt
        plt.close(fig)
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.read()))
