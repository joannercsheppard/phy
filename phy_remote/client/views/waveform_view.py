"""
WaveformWidget — fastplotlib waveform view with spatial probe layout.

Each selected cluster is rendered with its mean waveform ± std band at every
template channel, positioned on screen according to the real probe geometry
(depth on Y, horizontal offset on X).  Up to `max_spikes` raw spike snippets
are drawn behind the mean as semi-transparent lines and can be shown/hidden
with the "Show raw" toggle button.

Cluster roles
-------------
  index 0  (primary / selected)  →  red
  index 1+ (comparison)          →  blue / teal / amber / …

Camera
------
Uses fastplotlib's 2D pan-zoom camera.  `maintain_aspect = False` so the
X (time) and Y (depth) axes can be stretched independently:
  • Scroll           — zoom in/out centred on cursor
  • Ctrl + drag      — pan (default wgpu binding)
  • Right-click drag — zoom (default wgpu binding)

Fallback
--------
If fastplotlib / wgpu is not installed the widget falls back to a static
matplotlib Agg → QPixmap render (same as before).
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

_SELECTED_COLOR = (0.92, 0.15, 0.15, 1.0)   # vivid red   — primary selection
_COMPARE_COLORS = [
    (0.15, 0.47, 0.90, 1.0),   # blue
    (0.10, 0.72, 0.42, 1.0),   # teal-green
    (0.92, 0.58, 0.08, 1.0),   # amber
    (0.65, 0.15, 0.90, 1.0),   # purple
]
_STD_ALPHA = 0.22   # std band lines
_RAW_ALPHA = 0.11   # raw spike traces

_MPL_TAB10 = [         # matplotlib tab10 palette for fallback
    (0.122, 0.467, 0.706), (1.000, 0.498, 0.055), (0.173, 0.627, 0.173),
    (0.839, 0.153, 0.157), (0.580, 0.404, 0.741),
]


def _a(color: tuple, alpha: float) -> tuple:
    """Return *color* with alpha replaced."""
    return (color[0], color[1], color[2], alpha)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class WaveformWidget(QWidget):
    """
    Embeddable waveform widget.

    Parameters
    ----------
    max_spikes   : cap on raw spike traces drawn per cluster / channel
    max_channels : cap on channel subplots rendered
    """

    channel_clicked = pyqtSignal(int)

    def __init__(
        self,
        max_spikes: int = 50,
        max_channels: int = 16,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.max_spikes   = max_spikes
        self.max_channels = max_channels

        # Set externally once on startup via set_channel_positions()
        self._channel_positions: np.ndarray | None = None
        self._show_raw = False

        # fastplotlib handles
        self._fig          = None
        self._subplot      = None
        self._canvas_widget: QWidget | None = None
        self._raw_graphics: list = []   # line graphics that obey the raw toggle

        # ---- layout ----
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # thin toolbar
        bar = QWidget()
        bar.setFixedHeight(26)
        bar.setStyleSheet("background:#252525;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(6, 2, 6, 2)
        self._raw_btn = QPushButton("Show raw spikes")
        self._raw_btn.setCheckable(True)
        self._raw_btn.setFixedHeight(20)
        self._raw_btn.setStyleSheet(
            "QPushButton{font-size:11px;padding:0 6px;"
            "background:#333;color:#bbb;border:1px solid #555;border-radius:2px;}"
            "QPushButton:checked{background:#555;color:#fff;}"
        )
        self._raw_btn.toggled.connect(self._toggle_raw)
        bl.addWidget(self._raw_btn)
        bl.addStretch()
        outer.addWidget(bar)

        # canvas / placeholder area
        self._canvas_area = QWidget()
        self._canvas_area.setStyleSheet("background:#1e1e1e;")
        self._canvas_layout = QVBoxLayout(self._canvas_area)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QLabel("Select a cluster to display waveforms")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._placeholder.setStyleSheet("color:#666;font-size:14px;")
        self._canvas_layout.addWidget(self._placeholder)
        outer.addWidget(self._canvas_area)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_channel_positions(self, positions: np.ndarray) -> None:
        """Pass the (n_channels, 2) probe channel-positions array [x, y] µm."""
        self._channel_positions = np.asarray(positions, dtype=np.float32)
        logger.info("Channel positions loaded: %d channels", len(positions))

    def set_waveforms(
        self,
        data_per_cluster: "dict[int, tuple[np.ndarray, list[int]] | np.ndarray]",
    ) -> None:
        """
        Render waveforms for one or more clusters.

        Parameters
        ----------
        data_per_cluster
            Mapping of ``cluster_id → value`` where *value* is either:

            * ``(array, channel_ids)`` — preferred; enables spatial layout
            * bare ``np.ndarray``      — backwards-compatible; linear layout

            Array shapes accepted:

            * ``(n_samples, n_channels)``           — template (mean only)
            * ``(n_spikes, n_samples, n_channels)``  — individual spikes + mean
        """
        if not data_per_cluster:
            self._show_placeholder("No waveforms to display")
            return

        # Normalise to {cid: (arr3d, ch_ids)} with arr3d always 3D
        normalised: dict[int, tuple[np.ndarray, list[int]]] = {}
        for cid, val in data_per_cluster.items():
            if isinstance(val, tuple):
                arr, ch_ids = val[0], list(val[1])
            else:
                arr = val
                ch_ids = list(range(val.shape[-1] if val.ndim >= 2 else 1))
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]           # (1, n_samples, n_ch)
            if arr.ndim != 3:
                logger.error("Cluster %d: unexpected shape %s — skipped", cid, arr.shape)
                continue
            n_ch = min(arr.shape[2], self.max_channels)
            normalised[cid] = (arr[:, :, :n_ch], ch_ids[:n_ch])

        if not normalised:
            return

        # ---- try fastplotlib ----
        try:
            import fastplotlib as fpl
            self._render_fpl(fpl, normalised)
            return
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("fastplotlib render failed (%s) — falling back to matplotlib", exc)

        # ---- matplotlib fallback ----
        self._render_mpl(normalised)

    def clear(self) -> None:
        self._show_placeholder("Select a cluster to display waveforms")

    # ------------------------------------------------------------------
    # fastplotlib rendering
    # ------------------------------------------------------------------

    def _render_fpl(self, fpl, normalised: dict) -> None:
        """Build (or rebuild) the fastplotlib figure and draw all clusters."""
        primary_cid = next(iter(normalised))
        _, primary_ch_ids = normalised[primary_cid]

        self._ensure_fpl_figure(fpl)

        # Clear previous content
        self._subplot.clear()
        self._raw_graphics = []

        # Compute spatial layout from the primary cluster's channels
        positions, x_scale, amp_scale = self._probe_layout(primary_ch_ids, normalised)

        for clu_idx, (cid, (arr3, ch_ids)) in enumerate(normalised.items()):
            n_ch = arr3.shape[2]
            base = _SELECTED_COLOR if clu_idx == 0 else _COMPARE_COLORS[
                (clu_idx - 1) % len(_COMPARE_COLORS)
            ]
            mean_wf = arr3.mean(axis=0)     # (n_samples, n_ch)
            std_wf  = arr3.std(axis=0)

            # Spike index subset for raw traces
            n_spikes = arr3.shape[0]
            indices = np.arange(n_spikes)
            if n_spikes > self.max_spikes:
                rng = np.random.default_rng(seed=cid)
                indices = rng.choice(indices, self.max_spikes, replace=False)
                indices.sort()

            for ch in range(n_ch):
                pos = positions[ch] if ch < len(positions) else np.array([0.0, ch * 40.0])

                # Raw spikes (behind everything)
                if n_spikes > 1:
                    raw_color = _a(base, _RAW_ALPHA)
                    for i in indices:
                        g = self._subplot.add_line(
                            self._xy(arr3[i, :, ch], pos, x_scale, amp_scale),
                            colors=raw_color, thickness=0.8,
                        )
                        g.visible = self._show_raw
                        self._raw_graphics.append(g)

                # ±std band
                std_c = _a(base, _STD_ALPHA)
                for sign in (+1.0, -1.0):
                    self._subplot.add_line(
                        self._xy(mean_wf[:, ch] + sign * std_wf[:, ch],
                                 pos, x_scale, amp_scale),
                        colors=std_c, thickness=1.0,
                    )

                # Mean — on top, fully opaque, thicker
                self._subplot.add_line(
                    self._xy(mean_wf[:, ch], pos, x_scale, amp_scale),
                    colors=base, thickness=2.5,
                )

        self._placeholder.hide()
        self._canvas_widget.show()

    def _ensure_fpl_figure(self, fpl) -> None:
        """Create the fastplotlib Figure once; subsequent calls are no-ops."""
        if self._fig is not None:
            return

        self._fig = fpl.Figure(canvas="qt")
        self._subplot = self._fig[0, 0]

        # 2D camera — independent axis scaling (maintain_aspect=False)
        try:
            self._subplot.camera = "2d"
            self._subplot.camera.maintain_aspect = False
        except Exception as exc:
            logger.debug("fpl camera setup: %s", exc)

        self._canvas_widget = self._fig.canvas
        self._canvas_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._canvas_layout.addWidget(self._canvas_widget)
        logger.info("fastplotlib figure created")

    def _probe_layout(
        self,
        ch_ids: list[int],
        normalised: dict,
    ) -> tuple[np.ndarray, float, float]:
        """
        Return ``(positions, x_scale, amp_scale)`` for the given channel list.

        *positions* : (n_ch, 2) float32, channel centres in probe µm
        *x_scale*   : µm each trace spans horizontally (~channel pitch)
        *amp_scale* : sample-value → µm, scaled so peak-to-peak ≈ 45% of pitch
        """
        n_ch = len(ch_ids)

        if (
            self._channel_positions is not None
            and n_ch > 0
            and max(ch_ids) < len(self._channel_positions)
        ):
            positions = self._channel_positions[np.array(ch_ids)].copy()
        else:
            # Linear fallback: evenly spaced, top-to-bottom
            positions = np.zeros((n_ch, 2), dtype=np.float32)
            positions[:, 1] = np.linspace(n_ch * 40.0, 0.0, n_ch, dtype=np.float32)

        # Estimate minimum vertical pitch
        if n_ch > 1:
            y = np.sort(positions[:, 1])
            diffs = np.diff(y)
            diffs = diffs[diffs > 0.5]
            y_pitch = float(np.min(diffs)) if len(diffs) else 40.0
        else:
            y_pitch = 40.0

        x_scale = y_pitch * 0.85

        # Amplitude scale: largest mean peak-to-peak → 45% of pitch
        ptp_max = max(
            (float(np.ptp(arr3.mean(axis=0)[:, ch]))
             for arr3, _ in normalised.values()
             for ch in range(arr3.shape[2])),
            default=1.0,
        )
        amp_scale = (y_pitch * 0.45 / ptp_max) if ptp_max > 1e-9 else 1.0

        return positions, x_scale, amp_scale

    @staticmethod
    def _xy(
        wf: np.ndarray,       # (n_samples,)
        pos: np.ndarray,      # [x_um, y_um]
        x_scale: float,
        amp_scale: float,
    ) -> np.ndarray:
        """Build a (n_samples, 2) float32 xy array for one channel trace."""
        n = len(wf)
        x = pos[0] + np.linspace(-x_scale / 2.0, x_scale / 2.0, n, dtype=np.float32)
        y = pos[1] + (wf * amp_scale).astype(np.float32)
        return np.column_stack([x, y])

    def _toggle_raw(self, checked: bool) -> None:
        self._show_raw = checked
        for g in self._raw_graphics:
            try:
                g.visible = checked
            except Exception:
                pass

    # ------------------------------------------------------------------
    # matplotlib fallback
    # ------------------------------------------------------------------

    def _render_mpl(self, normalised: dict) -> None:
        """Render a static matplotlib Agg image into the placeholder QLabel."""
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure as MplFig

        # All clusters should share the same n_samples and n_ch
        shapes = {arr3.shape[1:] for arr3, _ in normalised.values()}
        if len(shapes) > 1:
            logger.error("Mismatched waveform shapes: %s", shapes)
            return
        _n_samples, n_ch = shapes.pop()

        _BG   = "#1e1e1e"
        _AXES = "#2a2a2a"

        fig = MplFig(
            figsize=(6, max(4.0, n_ch * 0.7)),
            facecolor=_BG,
        )
        fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.02, hspace=0.05)
        axes = [fig.add_subplot(n_ch, 1, ch + 1) for ch in range(n_ch)]
        for ax in axes:
            ax.set_facecolor(_AXES)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        for clu_idx, (cid, (arr3, _)) in enumerate(normalised.items()):
            c = _MPL_TAB10[clu_idx % len(_MPL_TAB10)]
            mean_wf = arr3.mean(axis=0)
            std_wf  = arr3.std(axis=0)

            indices = np.arange(arr3.shape[0])
            if arr3.shape[0] > self.max_spikes:
                rng = np.random.default_rng(seed=cid)
                indices = rng.choice(indices, self.max_spikes, replace=False)
                indices.sort()

            for ch in range(n_ch):
                ax = axes[ch]
                xs = np.arange(arr3.shape[1])

                if arr3.shape[0] > 1 and self._show_raw:
                    for i in indices:
                        ax.plot(xs, arr3[i, :, ch], color=(*c, 0.10), lw=0.4,
                                rasterized=True)

                ax.fill_between(
                    xs,
                    mean_wf[:, ch] - std_wf[:, ch],
                    mean_wf[:, ch] + std_wf[:, ch],
                    color=(*c, 0.20),
                )
                ax.plot(xs, mean_wf[:, ch], color=c, lw=1.5)

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        import matplotlib.pyplot as plt
        plt.close(fig)
        buf.seek(0)
        pm = QPixmap.fromImage(QImage.fromData(buf.read()))

        self._placeholder.setPixmap(
            pm.scaled(
                self._placeholder.width(),
                self._placeholder.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._placeholder.setStyleSheet("background:#1e1e1e;")
        self._placeholder.show()
        if self._canvas_widget is not None:
            self._canvas_widget.hide()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _show_placeholder(self, text: str) -> None:
        self._placeholder.setPixmap(QPixmap())
        self._placeholder.setText(text)
        self._placeholder.setStyleSheet("color:#666;font-size:14px;")
        if self._canvas_widget is not None:
            self._canvas_widget.hide()
        self._placeholder.show()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Rescale matplotlib pixmap when widget resizes
        if self._canvas_widget is None or not self._canvas_widget.isVisible():
            pm = self._placeholder.pixmap()
            if pm and not pm.isNull():
                self._placeholder.setPixmap(
                    pm.scaled(
                        self._placeholder.width(),
                        self._placeholder.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
