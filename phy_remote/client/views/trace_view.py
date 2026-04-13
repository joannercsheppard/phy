"""
TraceView — fastplotlib raw voltage trace display.

All channel traces in the depth window are rendered as a **single
NaN-separated Line object** (one GPU draw call).  Spike tick marks for
each visible cluster are each a single NaN-separated Line (one draw call
per cluster, regardless of how many channels or spikes).

Layout (world-space coordinates)
---------------------------------
  X axis : time in seconds
  Y axis : probe depth in µm  (shallow at top → large Y value; deep at
            bottom → small Y value, matching phy's probe view convention)

Each channel trace is centred on its probe Y position; amplitude is
scaled so the median peak-to-peak spans ≈ 60 % of the inter-channel
pitch.

Interactions (fastplotlib / wgpu)
----------------------------------
  Pan             : left-drag (wgpu default 2-D camera)
  Time zoom       : scroll wheel  (wgpu default)
  Amplitude zoom  : Ctrl + scroll — changes amp_scale and redraws in place
  Channel scroll  : Shift + scroll — moves depth window, triggers re-fetch
  Go to spike     : click near any tick mark → centers view on that spike

Fetching strategy
-----------------
Traces are fetched in a daemon thread (new PhyTransport per request, so
the main ZMQ socket is never shared).  A 200 ms debounce prevents
hammering the server while the user is scrolling.  A 50 ms polling timer
on the main thread applies results without blocking the event loop.

Fallback
--------
If fastplotlib / wgpu is not importable, the dock shows a clear error
message.  There is no matplotlib fallback because the trace view requires
a live interactive canvas.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRACE_COLOR = (0.60, 0.60, 0.60, 0.85)   # grey, slightly transparent

# Tick colours: index 0 = primary (selected) cluster, rest = comparison
_TICK_COLORS = [
    (0.92, 0.15, 0.15, 1.0),   # red
    (0.15, 0.47, 0.90, 1.0),   # blue
    (0.10, 0.72, 0.42, 1.0),   # teal
    (0.92, 0.58, 0.08, 1.0),   # amber
    (0.65, 0.15, 0.90, 1.0),   # purple
]

_DEFAULT_T_WINDOW = 0.5     # seconds shown initially
_DEFAULT_N_CH     = 32      # channels in depth window
_MAX_DISPLAY_PTS  = 3_000   # max samples per channel sent from server


# ---------------------------------------------------------------------------
# Line-data builders (module level — used outside the class too)
# ---------------------------------------------------------------------------

def build_nan_lines(traces_um: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
    """
    Pack N channel traces into a single NaN-separated array for one GPU draw.

    Parameters
    ----------
    traces_um : (n_ch, n_s) float32  — Y positions in µm (amplitude already
                                        added to the channel depth offset)
    t_arr     : (n_s,) float32       — time in seconds

    Returns
    -------
    (n_ch * (n_s + 1), 2) float32
        Each channel block is ``n_s`` (t, y) rows followed by one (NaN, NaN)
        separator so the GPU renderer lifts the pen between channels.
    """
    n_ch, n_s = traces_um.shape
    # Shape: (n_ch, n_s+1, 2) — last sample slot in each row stays NaN
    out = np.full((n_ch, n_s + 1, 2), np.nan, dtype=np.float32)
    out[:, :n_s, 0] = t_arr[np.newaxis, :]   # broadcast time across channels
    out[:, :n_s, 1] = traces_um
    return out.reshape(-1, 2)


def build_tick_data(
    spike_times: np.ndarray,
    ch_y: float,
    half_height: float,
) -> np.ndarray:
    """
    Build vertical tick marks for one cluster on one channel.

    Pattern per spike: [t, ch_y − h], [t, ch_y + h], [NaN, NaN]

    Returns (3 * n, 2) float32.
    """
    n = len(spike_times)
    if n == 0:
        return np.empty((0, 2), dtype=np.float32)
    out = np.full((n * 3, 2), np.nan, dtype=np.float32)
    t = spike_times.astype(np.float32)
    out[0::3, 0] = t;  out[0::3, 1] = ch_y - half_height
    out[1::3, 0] = t;  out[1::3, 1] = ch_y + half_height
    # [2::3] stays NaN — pen-lift between ticks
    return out


# ---------------------------------------------------------------------------
# TraceWidget
# ---------------------------------------------------------------------------

class TraceWidget(QWidget):
    """
    Interactive raw-trace viewer.

    Parameters
    ----------
    host, port : ZMQ server coordinates — TraceWidget creates its own
                 transport per request, never sharing the main socket.
    n_visible_ch : number of channels shown simultaneously
    t_window     : initial visible time span (seconds)
    """

    spike_clicked = pyqtSignal(float)   # world-space time when user clicks a tick

    def __init__(
        self,
        host: str,
        port: int,
        n_visible_ch: int = _DEFAULT_N_CH,
        t_window: float = _DEFAULT_T_WINDOW,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._port = port
        self._n_visible   = n_visible_ch
        self._t_window    = t_window
        self._t_center    = 0.0
        self._amp_scale   = 1.0     # µm per µV (adjusted after first fetch)
        self._filtered    = False
        self._ch_start_idx = 0      # index into _ch_sorted

        # Probe geometry
        self._ch_positions: np.ndarray | None = None  # (n_ch_total, 2)
        self._ch_sorted: list[int] = []               # channel indices by depth ↓

        # Buffer: last fetched trace block
        self._buf_traces: np.ndarray | None = None    # (n_ch_vis, n_samples)
        self._buf_t_start = 0.0
        self._buf_t_end   = 0.0
        self._buf_ch_ids: list[int] = []
        self._buf_sr      = 30_000.0

        # Spike overlay data: {cid: (spike_times_1d, home_ch_idx, rgba)}
        self._cluster_data: dict[int, tuple] = {}

        # fastplotlib handles
        self._fig            = None
        self._subplot        = None
        self._canvas_widget: QWidget | None = None
        self._trace_line: Any = None            # single NaN-sep Line for all channels
        self._tick_lines: dict[int, Any] = {}   # one NaN-sep Line per cluster
        self._fpl_ready      = False

        # Async fetch machinery
        self._fetch_cancelled  = False
        self._pending: dict | None = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._do_fetch)

        # Poll for results from fetch thread
        self._poll = QTimer(self)
        self._poll.setInterval(50)
        self._poll.timeout.connect(self._apply_pending)
        self._poll.start()

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_channel_positions(self, positions: np.ndarray) -> None:
        """Supply probe channel positions (n_ch, 2) [x, y] µm."""
        self._ch_positions = np.asarray(positions, dtype=np.float32)
        # Sort channels deepest-first so index 0 → shallowest at top of plot
        order = np.argsort(self._ch_positions[:, 1])[::-1]
        self._ch_sorted = order.tolist()
        logger.info("TraceWidget: %d channels", len(positions))

    def set_cluster_data(
        self,
        cluster_data: "dict[int, tuple[np.ndarray, int, tuple]]",
    ) -> None:
        """
        Set spike overlay data.

        Parameters
        ----------
        cluster_data : {cluster_id: (spike_times, home_channel_idx, rgba)}
            spike_times     : 1-D float64 array (seconds, all spikes)
            home_channel_idx: integer index into the full channel array
            rgba            : 4-tuple colour
        """
        self._cluster_data = dict(cluster_data)
        if self._fpl_ready and self._buf_traces is not None:
            self._rebuild_ticks()

    def go_to_time(self, t: float) -> None:
        """Center the view on *t* seconds and trigger a data fetch."""
        self._t_center = float(t)
        self._schedule_fetch()

    def go_to_first_spike(self) -> None:
        """Jump to the first spike of the primary (index-0) cluster."""
        for _cid, (times, _ch, _col) in self._cluster_data.items():
            if len(times):
                self.go_to_time(float(times.min()))
            return

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- toolbar ----
        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet("background:#252525;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(6, 2, 6, 2)
        bl.setSpacing(8)

        def _btn(label: str) -> QPushButton:
            b = QPushButton(label)
            b.setCheckable(False)
            b.setFixedHeight(20)
            b.setStyleSheet(
                "QPushButton{font-size:11px;padding:0 6px;background:#333;"
                "color:#bbb;border:1px solid #555;border-radius:2px;}"
                "QPushButton:hover{background:#444;}"
                "QPushButton:checked{background:#555;color:#fff;}"
            )
            return b

        self._filter_btn = _btn("HP filter")
        self._filter_btn.setCheckable(True)
        self._filter_btn.toggled.connect(self._on_filter_toggled)

        self._goto_btn = _btn("⟩| first spike")
        self._goto_btn.clicked.connect(self.go_to_first_spike)

        self._status = QLabel("Waiting for data…")
        self._status.setStyleSheet("color:#666;font-size:11px;")

        bl.addWidget(self._filter_btn)
        bl.addWidget(self._goto_btn)
        bl.addStretch()
        bl.addWidget(self._status)
        outer.addWidget(bar)

        # ---- canvas area ----
        self._canvas_area = QWidget()
        self._canvas_area.setStyleSheet("background:#1a1a1a;")
        self._canvas_layout = QVBoxLayout(self._canvas_area)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QLabel(
            "Select a cluster — trace view will load automatically"
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._placeholder.setStyleSheet("color:#555;font-size:13px;")
        self._canvas_layout.addWidget(self._placeholder)
        outer.addWidget(self._canvas_area)

    # ------------------------------------------------------------------
    # Fetch pipeline
    # ------------------------------------------------------------------

    def _schedule_fetch(self) -> None:
        self._fetch_cancelled = True
        self._debounce.start()

    def _do_fetch(self) -> None:
        if self._ch_positions is None:
            return
        ch_ids = self._visible_ch_ids()
        half   = self._t_window / 2.0
        t0     = max(0.0, self._t_center - half)
        t1     = self._t_center + half

        self._fetch_cancelled = False
        self._status.setText("Loading traces…")

        def _worker() -> None:
            from phy_remote.client.transport import PhyTransport, TransportError
            try:
                tr = PhyTransport(host=self._host, port=self._port)
                try:
                    traces, hdr = tr.get_traces(
                        t0, t1, ch_ids,
                        filtered=self._filtered,
                        max_samples=_MAX_DISPLAY_PTS,
                    )
                finally:
                    tr.close()
            except Exception as exc:
                if not self._fetch_cancelled:
                    self._pending = {"error": str(exc)}
                return
            if not self._fetch_cancelled:
                self._pending = {
                    "traces":   traces,
                    "t_start":  hdr.get("t_start", t0),
                    "t_end":    hdr.get("t_end",   t1),
                    "sr":       float(hdr.get("sample_rate", 30_000)),
                    "ch_ids":   hdr.get("channel_ids", ch_ids),
                }

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_pending(self) -> None:
        if self._pending is None:
            return
        data = self._pending
        self._pending = None

        if "error" in data:
            self._status.setText(f"Error: {data['error']}")
            return

        self._buf_traces  = data["traces"].astype(np.float32)
        self._buf_t_start = float(data["t_start"])
        self._buf_t_end   = float(data["t_end"])
        self._buf_sr      = float(data["sr"])
        self._buf_ch_ids  = list(data["ch_ids"])

        self._auto_scale_amplitude()
        self._render()

    def _auto_scale_amplitude(self) -> None:
        """Set amp_scale so median channel ptp ≈ 60 % of inter-channel pitch."""
        if self._buf_traces is None or len(self._buf_ch_ids) < 2:
            return
        ptp = float(np.median(np.ptp(self._buf_traces, axis=1)))
        if ptp < 1e-9:
            return
        ys = self._ch_positions[self._buf_ch_ids, 1]
        diffs = np.abs(np.diff(np.sort(ys)))
        diffs = diffs[diffs > 0.1]
        pitch = float(np.min(diffs)) if len(diffs) else 40.0
        self._amp_scale = (pitch * 0.60) / ptp

    def _visible_ch_ids(self) -> list[int]:
        if not self._ch_sorted:
            return list(range(min(self._n_visible, 96)))
        start = max(0, self._ch_start_idx)
        end   = min(start + self._n_visible, len(self._ch_sorted))
        return [self._ch_sorted[i] for i in range(start, end)]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if self._buf_traces is None:
            return
        try:
            import fastplotlib as fpl
        except ImportError:
            self._placeholder.setText(
                "TraceView requires fastplotlib.\n"
                "pip install fastplotlib wgpu"
            )
            return
        try:
            self._ensure_figure(fpl)
            self._update_traces()
            self._rebuild_ticks()
            self._placeholder.hide()
            self._canvas_widget.show()
        except Exception as exc:
            logger.exception("TraceWidget render error")
            self._placeholder.setText(f"Render error:\n{exc}")

    def _ensure_figure(self, fpl) -> None:
        if self._fpl_ready:
            return

        self._fig     = fpl.Figure(canvas="qt")
        self._subplot = self._fig[0, 0]

        # Independent-axis 2-D pan/zoom camera
        try:
            self._subplot.camera = "2d"
            self._subplot.camera.maintain_aspect = False
        except Exception as exc:
            logger.debug("fpl camera: %s", exc)

        # Event handlers
        try:
            self._fig.canvas.add_event_handler(self._on_wheel,  "wheel")
            self._fig.canvas.add_event_handler(self._on_click,  "click")
            self._fig.canvas.add_event_handler(self._on_key,    "key_down")
        except Exception as exc:
            logger.debug("fpl events: %s", exc)

        self._canvas_widget = self._fig.canvas
        self._canvas_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._canvas_layout.addWidget(self._canvas_widget)
        self._fpl_ready = True

    def _update_traces(self) -> None:
        """Rebuild the single NaN-separated trace Line from buffered data."""
        traces  = self._buf_traces
        t_start = self._buf_t_start
        t_end   = self._buf_t_end
        n_ch, n_s = traces.shape

        t_arr = np.linspace(t_start, t_end, n_s, dtype=np.float32)

        # Channel Y offsets in probe µm
        if self._ch_positions is not None and self._buf_ch_ids:
            ys = self._ch_positions[np.array(self._buf_ch_ids), 1].astype(np.float32)
        else:
            ys = np.arange(n_ch, dtype=np.float32) * 40.0

        # traces_um: each channel signal offset to its probe depth
        traces_um = ys[:, np.newaxis] + traces * self._amp_scale

        line_data = build_nan_lines(traces_um, t_arr)

        if self._trace_line is None:
            self._trace_line = self._subplot.add_line(
                line_data, colors=_TRACE_COLOR, thickness=1.2,
            )
            # Fit camera on first draw
            try:
                self._subplot.auto_scale()
            except Exception:
                pass
        else:
            try:
                self._trace_line.data[:] = line_data
            except (ValueError, TypeError):
                # Shape changed — remove and recreate
                try:
                    self._subplot.remove(self._trace_line)
                except Exception:
                    pass
                self._trace_line = self._subplot.add_line(
                    line_data, colors=_TRACE_COLOR, thickness=1.2,
                )

        self._status.setText(
            f"{t_start:.3f} – {t_end:.3f} s  ·  "
            f"{n_ch} ch  ·  {'HP' if self._filtered else 'raw'}"
        )

    def _rebuild_ticks(self) -> None:
        """Rebuild per-cluster spike tick Lines for the current time window."""
        if not self._fpl_ready or self._buf_traces is None:
            return

        t0 = self._buf_t_start
        t1 = self._buf_t_end

        # Tick height = 35 % of inter-channel pitch
        if self._ch_positions is not None and len(self._buf_ch_ids) > 1:
            ys = self._ch_positions[np.array(self._buf_ch_ids), 1]
            diffs = np.abs(np.diff(np.sort(ys)))
            diffs = diffs[diffs > 0.1]
            pitch = float(np.min(diffs)) if len(diffs) else 40.0
        else:
            pitch = 40.0
        tick_h = pitch * 0.35

        # Remove lines for clusters that are no longer selected
        stale = set(self._tick_lines) - set(self._cluster_data)
        for cid in stale:
            try:
                self._subplot.remove(self._tick_lines.pop(cid))
            except Exception:
                self._tick_lines.pop(cid, None)

        for clu_idx, (cid, (times, home_ch, color)) in enumerate(
            self._cluster_data.items()
        ):
            # Filter spikes to visible window
            mask = (times >= t0) & (times <= t1)
            vis_times = times[mask].astype(np.float32)

            # Home channel not in the visible depth window → hide ticks
            if home_ch not in self._buf_ch_ids:
                if cid in self._tick_lines:
                    try:
                        self._tick_lines[cid].visible = False
                    except Exception:
                        pass
                continue

            ch_y = float(self._ch_positions[home_ch, 1]) \
                if self._ch_positions is not None else float(clu_idx * 40)

            tick_data = build_tick_data(vis_times, ch_y, tick_h)

            if len(tick_data) == 0:
                if cid in self._tick_lines:
                    try:
                        self._tick_lines[cid].visible = False
                    except Exception:
                        pass
                continue

            if cid in self._tick_lines:
                line = self._tick_lines[cid]
                try:
                    line.data[:] = tick_data
                except (ValueError, TypeError):
                    try:
                        self._subplot.remove(line)
                    except Exception:
                        pass
                    line = self._subplot.add_line(
                        tick_data, colors=color, thickness=1.8,
                    )
                    self._tick_lines[cid] = line
                try:
                    line.colors = color
                    line.visible = True
                except Exception:
                    pass
            else:
                line = self._subplot.add_line(
                    tick_data, colors=color, thickness=1.8,
                )
                # Attach metadata for click hit-testing
                line._spike_times  = times    # type: ignore[attr-defined]
                line._cluster_id   = cid      # type: ignore[attr-defined]
                self._tick_lines[cid] = line

    # ------------------------------------------------------------------
    # Event handlers (fastplotlib / wgpu)
    # ------------------------------------------------------------------

    def _on_wheel(self, event) -> None:
        dy = float(getattr(event, "dy", 0) or 0)
        if dy == 0:
            return

        mods = getattr(event, "modifiers", None) or ()
        ctrl  = any("ctrl"  in str(m).lower() for m in mods)
        shift = any("shift" in str(m).lower() for m in mods)

        if ctrl:
            # Amplitude zoom: rescale in-place, no network fetch
            factor = 1.15 if dy < 0 else 1.0 / 1.15
            self._amp_scale *= factor
            if self._buf_traces is not None:
                try:
                    self._update_traces()
                    self._rebuild_ticks()
                except Exception as exc:
                    logger.debug("amp zoom redraw: %s", exc)
            try:
                event.handled = True
            except Exception:
                pass

        elif shift:
            # Depth window scroll: move channel window, re-fetch
            step = max(1, self._n_visible // 4)
            self._ch_start_idx = max(
                0,
                min(
                    len(self._ch_sorted) - self._n_visible,
                    self._ch_start_idx + (step if dy < 0 else -step),
                ),
            )
            self._schedule_fetch()
            try:
                event.handled = True
            except Exception:
                pass
        # Plain scroll: let fastplotlib pan the time axis natively.
        # After panning, check if we've drifted outside the buffer.
        # (Handled asynchronously below via _check_buffer_coverage.)

    def _on_key(self, event) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key in ("arrowup", "up", "k"):
            step = max(1, self._n_visible // 4)
            self._ch_start_idx = max(0, self._ch_start_idx - step)
            self._schedule_fetch()
        elif key in ("arrowdown", "down", "j"):
            step = max(1, self._n_visible // 4)
            self._ch_start_idx = min(
                len(self._ch_sorted) - self._n_visible,
                self._ch_start_idx + step,
            )
            self._schedule_fetch()

    def _on_click(self, event) -> None:
        """Click on or near a tick mark → center view on that spike."""
        # Convert screen click to world-space time coordinate
        t_click = self._screen_to_time(event)
        if t_click is None:
            return

        tol = self._t_window * 0.03   # 3 % of visible window

        best_t:    float | None = None
        best_dist: float        = float("inf")

        for _cid, line in self._tick_lines.items():
            times = getattr(line, "_spike_times", None)
            if times is None:
                continue
            try:
                if not line.visible:
                    continue
            except Exception:
                pass
            vis = times[(times >= self._buf_t_start) & (times <= self._buf_t_end)]
            if len(vis) == 0:
                continue
            d = np.abs(vis - t_click)
            idx = int(np.argmin(d))
            if float(d[idx]) < best_dist:
                best_dist = float(d[idx])
                best_t = float(vis[idx])

        if best_t is not None and best_dist < tol:
            self._t_center = best_t
            self._schedule_fetch()
            self.spike_clicked.emit(best_t)

    def _screen_to_time(self, event) -> "float | None":
        """
        Convert a click event to a time value (seconds) in world space.

        Tries the fastplotlib/pygfx world-position APIs in order; falls back
        to a linear interpolation from screen x fraction.
        """
        # Method 1: pygfx pick_info world position
        try:
            wp = event.pick_info.get("world_object")
            if wp is not None:
                return float(wp.world.x)
        except Exception:
            pass

        # Method 2: event.world_position
        try:
            return float(event.world_position[0])
        except Exception:
            pass

        # Method 3: linear interpolation from screen x fraction
        try:
            sx = float(event.x)
            sw = float(self._canvas_widget.width())
            if sw > 0:
                frac = sx / sw
                return self._buf_t_start + frac * (self._buf_t_end - self._buf_t_start)
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Toolbar slots
    # ------------------------------------------------------------------

    def _on_filter_toggled(self, checked: bool) -> None:
        self._filtered = checked
        self._schedule_fetch()

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:
        """Trigger an initial fetch when the dock is first shown."""
        super().showEvent(event)
        if self._ch_positions is not None and self._buf_traces is None:
            self._schedule_fetch()
