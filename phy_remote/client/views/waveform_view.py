"""
WaveformWidget — renders spike waveforms via matplotlib Agg → QPixmap.

Uses the non-interactive Agg backend (renders to bytes, no Qt canvas
embedding required) so it works regardless of which matplotlib backend
the environment has configured.  The resulting image is displayed in a
QLabel that stretches to fill the central widget area.

One subplot per channel, vertical column, highest channel at top.
Individual spike traces (semi-transparent) + per-cluster mean (solid).
"""

from __future__ import annotations

import io
import logging

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy

logger = logging.getLogger(__name__)

# tab10 palette (RGB 0-1)
CLUSTER_COLORS = [
    (0.122, 0.467, 0.706),
    (1.000, 0.498, 0.055),
    (0.173, 0.627, 0.173),
    (0.839, 0.153, 0.157),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.498, 0.498, 0.498),
    (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.812),
]

_BG   = "#1e1e1e"
_AXES = "#2a2a2a"


def _rgba(idx: int, alpha: float = 1.0) -> tuple:
    r, g, b = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
    return (r, g, b, alpha)


class WaveformWidget(QWidget):
    """
    Embeddable widget showing multi-cluster waveforms rendered via
    matplotlib Agg backend displayed as a QPixmap.

    Parameters
    ----------
    max_spikes   : cap on individual spike traces per cluster/channel
    max_channels : cap on channel subplots shown
    """

    channel_clicked = pyqtSignal(int)   # future use

    def __init__(
        self,
        max_spikes: int = 50,
        max_channels: int = 16,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.max_spikes   = max_spikes
        self.max_channels = max_channels

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel("Select a cluster to display waveforms")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._label.setStyleSheet(
            "background: #1e1e1e; color: #aaaaaa; font-size: 14px;"
        )
        layout.addWidget(self._label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_waveforms(
        self,
        data_per_cluster: dict[int, np.ndarray],
    ) -> None:
        """
        Render waveforms.

        data_per_cluster values may be:
          (n_samples, n_channels)             — template / mean only
          (n_spikes, n_samples, n_channels)   — individual spikes + mean
        """
        if not data_per_cluster:
            self._label.setText("No waveforms to display")
            return

        # Normalise to (n_spikes, n_samples, n_channels)
        normalised: dict[int, np.ndarray] = {}
        for cid, arr in data_per_cluster.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                normalised[cid] = arr[np.newaxis]
            elif arr.ndim == 3:
                normalised[cid] = arr
            else:
                logger.error("Cluster %d: unexpected shape %s", cid, arr.shape)

        if not normalised:
            return

        shapes = {a.shape[1:] for a in normalised.values()}
        if len(shapes) > 1:
            logger.error("Mismatched waveform shapes: %s", shapes)
            return
        _n_samples, n_channels = shapes.pop()
        n_channels = min(n_channels, self.max_channels)

        try:
            pixmap = self._render(normalised, n_channels)
            self._label.setPixmap(
                pixmap.scaled(
                    self._label.width(),
                    self._label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        except Exception as exc:
            logger.exception("Waveform render failed")
            self._label.setText(f"Render error:\n{exc}")

    def clear(self) -> None:
        self._label.setText("Select a cluster to display waveforms")

    def resizeEvent(self, event) -> None:
        """Re-scale the existing pixmap when the widget is resized."""
        super().resizeEvent(event)
        pm = self._label.pixmap()
        if pm and not pm.isNull():
            self._label.setPixmap(
                pm.scaled(
                    self._label.width(),
                    self._label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render(
        self,
        normalised: dict[int, np.ndarray],
        n_channels: int,
    ) -> QPixmap:
        """Draw via matplotlib Agg → PNG bytes → QPixmap."""
        import matplotlib
        matplotlib.use("Agg")                   # non-interactive, always works
        from matplotlib.figure import Figure as MplFigure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        height_per_ch = 0.7
        fig_h = max(4.0, n_channels * height_per_ch)
        fig = MplFigure(figsize=(6, fig_h), facecolor=_BG)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.02,
                            hspace=0.05)

        axes = [fig.add_subplot(n_channels, 1, ch + 1)
                for ch in range(n_channels)]

        for ax in axes:
            ax.set_facecolor(_AXES)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        for clu_idx, (clu_id, waveforms) in enumerate(normalised.items()):
            n_spikes = waveforms.shape[0]
            sc = _rgba(clu_idx, alpha=0.18)
            mc = _rgba(clu_idx, alpha=1.00)

            indices = np.arange(n_spikes)
            if n_spikes > self.max_spikes:
                rng = np.random.default_rng(seed=clu_id)
                indices = rng.choice(indices, size=self.max_spikes, replace=False)
                indices.sort()

            for ch in range(n_channels):
                ax = axes[ch]
                if n_spikes > 1:
                    for i in indices:
                        ax.plot(waveforms[i, :, ch], color=sc, linewidth=0.4,
                                rasterized=True)
                mean_wf = waveforms[:, :, ch].mean(axis=0)
                ax.plot(mean_wf, color=mc, linewidth=1.5)

        # channel labels
        for ch, ax in enumerate(axes):
            ax.set_ylabel(f"{ch}", fontsize=6, color="#666666",
                          rotation=0, labelpad=14, va="center")

        # render to PNG bytes
        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        import matplotlib.pyplot as plt
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.read()

        qimage = QImage.fromData(png_bytes)
        return QPixmap.fromImage(qimage)
