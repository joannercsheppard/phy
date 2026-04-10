"""
WaveformWidget — PyQt6 widget that renders spike waveforms with fastplotlib.

Layout
------
One subplot per channel, arranged in a single column mirroring probe depth
(top = highest channel index, bottom = lowest, matching phy's probe view).

Each subplot shows:
  - Individual spike traces (semi-transparent, up to max_spikes per cluster)
  - Per-cluster mean in a solid contrasting colour

When templates only are available (before real waveforms arrive) the mean is
drawn as a single solid line — the individual spikes appear once the real
waveform fetch completes.

Cluster colours follow the same 10-colour cycle used throughout phy-remote
(see CLUSTER_COLORS below).

Dependencies
------------
    pip install fastplotlib[notebook] wgpu PyQt6
The wgpu Metal backend is selected automatically on macOS.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

# 10-colour qualitative palette (matches matplotlib tab10, normalised 0-1)
CLUSTER_COLORS: list[tuple[float, float, float, float]] = [
    (0.122, 0.467, 0.706, 1.0),
    (1.000, 0.498, 0.055, 1.0),
    (0.173, 0.627, 0.173, 1.0),
    (0.839, 0.153, 0.157, 1.0),
    (0.580, 0.404, 0.741, 1.0),
    (0.549, 0.337, 0.294, 1.0),
    (0.890, 0.467, 0.761, 1.0),
    (0.498, 0.498, 0.498, 1.0),
    (0.737, 0.741, 0.133, 1.0),
    (0.090, 0.745, 0.812, 1.0),
]


def _cluster_color(
    index: int, alpha: float = 0.25
) -> tuple[float, float, float, float]:
    r, g, b, _ = CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
    return (r, g, b, alpha)


def _cluster_mean_color(index: int) -> tuple[float, float, float, float]:
    return CLUSTER_COLORS[index % len(CLUSTER_COLORS)]


# ---------------------------------------------------------------------------
# WaveformWidget
# ---------------------------------------------------------------------------

class WaveformWidget(QWidget):
    """
    Embeddable Qt widget that shows multi-cluster, multi-channel waveforms.

    Signals
    -------
    channel_clicked(int)
        Emitted when the user clicks a channel subplot (channel index).

    Parameters
    ----------
    max_spikes : int
        Maximum number of individual spike traces drawn per cluster per channel.
    max_channels : int
        Maximum number of channel subplots to render.
    parent : QWidget, optional
    """

    channel_clicked = pyqtSignal(int)

    def __init__(
        self,
        max_spikes: int = 50,
        max_channels: int = 16,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.max_spikes = max_spikes
        self.max_channels = max_channels

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        self._fpl_widget: QWidget | None = None
        self._fig = None          # fastplotlib Figure
        self._n_channels: int = 0

        self._placeholder = QLabel("Select a cluster to display waveforms")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._layout.addWidget(self._placeholder)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_waveforms(
        self,
        data_per_cluster: dict[int, np.ndarray],
    ) -> None:
        """
        Render waveforms for one or more clusters.

        Parameters
        ----------
        data_per_cluster : dict[cluster_id -> ndarray]
            Each array has shape either:
            - ``(n_samples, n_channels)``             — template / mean only
            - ``(n_spikes, n_samples, n_channels)``   — individual spikes + mean

        Calling this method a second time for the same channel layout (e.g.
        upgrading from template to real waveforms) clears and redraws in place
        without rebuilding the figure, so the transition is smooth.
        """
        if not data_per_cluster:
            self._show_placeholder("No waveforms to display")
            return

        try:
            import fastplotlib as fpl
        except ImportError:
            self._show_placeholder(
                "fastplotlib not installed.\n"
                "Run: pip install fastplotlib wgpu"
            )
            return

        # Normalise everything to (n_spikes, n_samples, n_channels)
        # A template (n_samples, n_channels) becomes a 1-spike array.
        normalised: dict[int, np.ndarray] = {}
        for cid, arr in data_per_cluster.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                normalised[cid] = arr[np.newaxis]    # (1, n_samples, n_channels)
            elif arr.ndim == 3:
                normalised[cid] = arr
            else:
                logger.error(
                    "Cluster %d: unexpected waveform shape %s — skipping", cid, arr.shape
                )

        if not normalised:
            return

        shapes = {a.shape[1:] for a in normalised.values()}
        if len(shapes) > 1:
            logger.error("Mismatched waveform shapes across clusters: %s", shapes)
            return
        _n_samples, n_channels = shapes.pop()
        n_channels = min(n_channels, self.max_channels)

        self._rebuild_figure(fpl, n_channels)

        for clu_idx, (clu_id, waveforms) in enumerate(normalised.items()):
            self._plot_waveforms(clu_idx, clu_id, waveforms, n_channels)

        if self._fig is not None:
            self._fig.canvas.set_logical_size(
                self._fpl_widget.width(), self._fpl_widget.height()
            )

    def clear(self) -> None:
        """Remove all rendered content and show the placeholder."""
        self._show_placeholder("Select a cluster to display waveforms")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _show_placeholder(self, text: str) -> None:
        self._placeholder.setText(text)
        if self._fpl_widget is not None:
            self._fpl_widget.hide()
        self._placeholder.show()

    def _rebuild_figure(self, fpl, n_channels: int) -> None:
        """Create (or re-use) the fastplotlib Figure with n_channels subplots."""
        if self._fig is not None and self._n_channels == n_channels:
            # Re-use existing figure: just clear each subplot
            for row in range(n_channels):
                self._fig[row, 0].clear()
            return

        # Tear down old figure/widget
        if self._fpl_widget is not None:
            self._layout.removeWidget(self._fpl_widget)
            self._fpl_widget.deleteLater()
            self._fpl_widget = None
            self._fig = None

        self._placeholder.hide()

        # Build new figure: n_channels rows × 1 column
        self._fig = fpl.Figure(
            shape=(n_channels, 1),
            canvas="qt",
        )
        self._n_channels = n_channels

        self._fpl_widget = self._fig.canvas
        self._layout.addWidget(self._fpl_widget)
        self._fpl_widget.show()

        # Connect click events
        for row in range(n_channels):
            self._fig[row, 0].canvas.add_event_handler(
                lambda event, ch=row: self._on_channel_click(event, ch),
                "click",
            )

    def _plot_waveforms(
        self,
        clu_idx: int,
        clu_id: int,
        waveforms: np.ndarray,   # (n_spikes, n_samples, n_channels)
        n_channels: int,
    ) -> None:
        """
        Draw individual spike traces (semi-transparent) + mean (solid).

        When n_spikes == 1 (template) only the mean line is drawn.
        """
        n_spikes = waveforms.shape[0]
        spike_color = _cluster_color(clu_idx, alpha=0.20)
        mean_color  = _cluster_mean_color(clu_idx)

        # Subsample if too many spikes
        indices = np.arange(n_spikes)
        if n_spikes > self.max_spikes:
            rng = np.random.default_rng(seed=clu_id)
            indices = rng.choice(indices, size=self.max_spikes, replace=False)
            indices.sort()

        for ch in range(n_channels):
            subplot = self._fig[ch, 0]

            # Individual traces (skip for templates, i.e. n_spikes == 1)
            if n_spikes > 1:
                for i in indices:
                    subplot.add_line(
                        waveforms[i, :, ch],
                        colors=spike_color,
                        thickness=1.0,
                    )

            # Mean / template line
            mean_wf = waveforms[:, :, ch].mean(axis=0)
            subplot.add_line(
                mean_wf,
                colors=mean_color,
                thickness=2.5,
            )

        logger.debug(
            "Plotted %d waveform(s) for cluster %d (%d channels)",
            n_spikes, clu_id, n_channels,
        )

    def _on_channel_click(self, event, channel_idx: int) -> None:
        self.channel_clicked.emit(channel_idx)
