"""TraceView — raw voltage trace display using fastplotlib.

Matches phy2 TraceView behaviour:
- Channels offset vertically by probe position * y_scale
- add_line_collection for all traces (one GPU call)
- Spike overlay = vertical colored tick marks on best channel (NOT waveform snippets)
- Raw data only reloaded on time navigation; cluster change only updates tick colors
- HP filter: 3rd-order Butterworth 300 Hz zero-phase (if hp_filtered=False)
- Navigation: Alt+←/→ (scroll 50%), F/B (full window), Ctrl+wheel (time zoom),
  Alt+wheel (amplitude), right-click drag (pan time)
"""
from __future__ import annotations

import colorsys
import logging
import threading
from typing import Any

import numpy as np
from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QPushButton,
    QRubberBand, QSizePolicy, QVBoxLayout, QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_S       = 1.0       # default display window in seconds
_SHIFT_FRAC     = 0.5       # Alt+arrow scrolls this fraction of window
_TRACE_COLOR    = (0.45, 0.45, 0.45, 0.90)
_MAX_SAMPLES    = 3_000     # server-side downsampling cap

# Cluster colors index 0=primary, 1=comparison, …
_CLUSTER_COLORS = [
    (0.92, 0.15, 0.15, 1.0),
    (0.15, 0.47, 0.90, 1.0),
    (0.15, 0.72, 0.35, 1.0),
    (0.92, 0.58, 0.08, 1.0),
    (0.65, 0.15, 0.90, 1.0),
]


class TraceWidget(QWidget):
    spike_clicked = pyqtSignal(float)

    def __init__(self, host: str, port: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._host     = host
        self._port     = port
        self._t_start  = 0.0
        self._window_s = _WINDOW_S
        self._y_scale  = 1.0       # amplitude multiplier
        self._filtered = False

        # Probe geometry
        self._ch_positions: np.ndarray | None = None   # (n_ch, 2) µm
        self._ch_order:     list[int] = []             # channel indices deepest→shallowest
        self._ch_y:         np.ndarray | None = None   # (n_ch,) y offset per channel id

        # Trace buffer — only reloaded on time navigation
        self._buf_traces:  np.ndarray | None = None    # (n_ch, n_s) CMR+scaled
        self._buf_t_arr:   np.ndarray | None = None    # (n_s,) seconds
        self._buf_ch_ids:  list[int] = []
        self._buf_sr       = 30_000.0

        # Spike overlay for SELECTED clusters: {cid: (spike_times_s, home_ch, rgba)}
        self._cluster_data: dict[int, tuple] = {}
        # Best channel for ALL clusters — set once at startup
        self._all_best_channels: dict[int, int] = {}
        # Spikes fetched in the current window: (n, 2) float64 [time_s, cluster_id]
        self._window_spikes: np.ndarray | None = None

        # Right-click pan state
        self._pan_last_x: float | None = None

        # fastplotlib handles
        self._fig:            Any = None
        self._subplot:        Any = None
        self._canvas_widget:  QWidget | None = None
        self._line_collection: Any = None   # all channel traces
        self._tick_lines:      dict[int, Any] = {}   # per-cluster tick marks
        self._fpl_ready        = False

        # Async fetch
        self._fetch_seq  = 0
        self._pending: dict | None = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._do_fetch)

        self._poll = QTimer(self)
        self._poll.setInterval(40)
        self._poll.timeout.connect(self._apply_pending)
        self._poll.start()

        self._build_ui()

        try:
            import fastplotlib as fpl
            self._init_fpl(fpl)
        except Exception as exc:
            logger.warning("TraceWidget: fastplotlib init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_channel_positions(self, positions: np.ndarray) -> None:
        self._ch_positions = np.asarray(positions, dtype=np.float32)
        # Sort shallowest→deepest so first in list = top of display
        order = np.argsort(self._ch_positions[:, 1])[::-1]
        self._ch_order = order.tolist()
        n = len(positions)
        # y offset = probe y position (µm) — scaled later by y_scale
        self._ch_y = self._ch_positions[:, 1].copy()   # indexed by channel id
        logger.info("TraceWidget: %d channels", n)
        self._schedule_fetch()

    def set_cluster_data(
        self,
        cluster_data: "dict[int, tuple[np.ndarray, int, tuple]]",
    ) -> None:
        """Update spike overlay WITHOUT reloading raw data."""
        self._cluster_data = dict(cluster_data)
        if self._fpl_ready and self._buf_traces is not None:
            self._rebuild_ticks()
            try:
                self._fig.canvas.request_draw()
            except Exception:
                pass

    def set_all_best_channels(self, best: "dict[int, int]") -> None:
        """Set best-channel mapping for all clusters (for full spike overlay)."""
        self._all_best_channels = dict(best)
        if self._fpl_ready and self._buf_traces is not None:
            self._rebuild_ticks()
            try:
                self._fig.canvas.request_draw()
            except Exception:
                pass

    def go_to_time(self, t: float) -> None:
        self._t_start = max(0.0, t - self._window_s / 2)
        self._schedule_fetch()

    def go_to_first_spike(self) -> None:
        for _cid, (times, _ch, _col) in self._cluster_data.items():
            if len(times):
                self.go_to_time(float(times[0]))
                return

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet("background:#252525;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(6, 2, 6, 2)
        bl.setSpacing(6)

        self._filter_btn = self._make_btn("HP filter", checkable=True)
        self._filter_btn.toggled.connect(self._on_filter_toggled)
        self._spike_btn  = self._make_btn("⟩| spike")
        self._spike_btn.clicked.connect(self.go_to_first_spike)

        self._status = QLabel("Waiting for data…")
        self._status.setStyleSheet("color:#666;font-size:11px;")

        bl.addWidget(self._filter_btn)
        bl.addWidget(self._spike_btn)
        bl.addStretch()
        bl.addWidget(self._status)
        outer.addWidget(bar)

        self._canvas_area = QWidget()
        self._canvas_area.setStyleSheet("background:#111;")
        self._canvas_layout = QVBoxLayout(self._canvas_area)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QLabel("Select a cluster — traces will load automatically")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._placeholder.setStyleSheet("color:#555;font-size:13px;")
        self._canvas_layout.addWidget(self._placeholder)
        outer.addWidget(self._canvas_area)

    @staticmethod
    def _make_btn(label: str, checkable: bool = False) -> QPushButton:
        b = QPushButton(label)
        b.setCheckable(checkable)
        b.setFixedHeight(20)
        b.setStyleSheet(
            "QPushButton{font-size:11px;padding:0 6px;background:#333;"
            "color:#bbb;border:1px solid #555;border-radius:2px;}"
            "QPushButton:checked{background:#555;color:#fff;}"
            "QPushButton:hover{background:#444;}"
        )
        return b

    # ------------------------------------------------------------------
    # fastplotlib setup
    # ------------------------------------------------------------------

    def _init_fpl(self, fpl) -> None:
        self._fig     = fpl.Figure(canvas="qt")
        self._subplot = self._fig[0, 0]
        try:
            self._subplot.camera = "2d"
            self._subplot.camera.maintain_aspect = False
        except Exception:
            pass
        try:
            self._subplot.axes.visible = False
            self._subplot.title.visible = False
        except Exception:
            pass
        self._fig.show()
        self._canvas_widget = self._fig.canvas
        self._canvas_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._canvas_widget.setMinimumHeight(60)
        self._canvas_layout.addWidget(self._canvas_widget)
        self._canvas_widget.hide()

        # Event filters
        self._event_filter = _TraceEventFilter(self, self._canvas_widget, self)
        QApplication.instance().installEventFilter(self._event_filter)
        self._box_zoom_filter = _BoxZoomFilter(self, self._canvas_widget, self)
        QApplication.instance().installEventFilter(self._box_zoom_filter)

        # Right-click drag via fpl canvas events
        try:
            self._fig.canvas.add_event_handler(self._on_pointer_down,  "pointer_down")
            self._fig.canvas.add_event_handler(self._on_pointer_move,  "pointer_move")
            self._fig.canvas.add_event_handler(self._on_pointer_up,    "pointer_up")
        except Exception as exc:
            logger.debug("TraceWidget: canvas events not available: %s", exc)

        self._fpl_ready = True

    # ------------------------------------------------------------------
    # Fetch pipeline  (only called on time navigation)
    # ------------------------------------------------------------------

    def _schedule_fetch(self) -> None:
        self._fetch_seq += 1
        self._debounce.start()

    def _do_fetch(self) -> None:
        if self._ch_positions is None:
            return
        t0  = self._t_start
        t1  = t0 + self._window_s
        seq = self._fetch_seq

        def _worker() -> None:
            from phy_remote.client.transport import PhyTransport
            try:
                tr = PhyTransport(host=self._host, port=self._port)
                try:
                    traces, hdr = tr.get_traces(
                        t0, t1,
                        channel_ids=None,
                        filtered=self._filtered,
                        max_samples=_MAX_SAMPLES,
                    )
                    window_spikes = tr.get_spikes_in_window(t0, t1)
                finally:
                    tr.close()
            except Exception as exc:
                if seq == self._fetch_seq:
                    self._pending = {"error": str(exc)}
                return
            if seq == self._fetch_seq:
                self._pending = {
                    "traces":        traces,
                    "window_spikes": window_spikes,
                    "t_start": float(hdr.get("t_start", t0)),
                    "t_end":   float(hdr.get("t_end",   t1)),
                    "sr":      float(hdr.get("sample_rate", 30_000)),
                    "ch_ids":  hdr.get("channel_ids", list(range(traces.shape[0]))),
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

        traces        = np.array(data["traces"], dtype=np.float32)   # (n_ch, n_s) writable
        self._window_spikes = data.get("window_spikes")             # (n, 2) [time, cid]
        t_start = data["t_start"]
        t_end   = data["t_end"]
        sr      = data["sr"]
        ch_ids  = list(data["ch_ids"])

        self._buf_sr     = sr
        self._buf_ch_ids = ch_ids

        n_ch, n_s = traces.shape
        self._buf_t_arr = np.linspace(t_start, t_end, n_s, dtype=np.float32)

        # CMR: subtract median across channels at each sample
        traces -= np.median(traces, axis=0)

        # Auto-scale: normalise so the 1%-99% range spans one channel spacing
        q01 = float(np.quantile(traces, 0.01))
        q99 = float(np.quantile(traces, 0.99))
        span = max(abs(q01), abs(q99), 1e-9)
        # Compute channel pitch for scaling reference
        if self._ch_y is not None and len(ch_ids) > 1:
            ys = self._ch_y[np.array(ch_ids)]
            diffs = np.abs(np.diff(np.sort(ys)))
            diffs = diffs[diffs > 0.5]
            pitch = float(np.min(diffs)) if len(diffs) else 40.0
        else:
            pitch = 40.0
        self._buf_traces = traces * (pitch * 0.45 / span) * self._y_scale

        self._render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if not self._fpl_ready or self._buf_traces is None:
            return
        try:
            self._update_line_collection()
            self._rebuild_ticks()
            self._placeholder.hide()
            self._canvas_widget.show()
            n_ch = len(self._buf_ch_ids)
            t0   = float(self._buf_t_arr[0])
            t1   = float(self._buf_t_arr[-1])
            self._status.setText(
                f"{t0:.3f}–{t1:.3f} s  ·  {n_ch} ch  ·  "
                f"{'HP' if self._filtered else 'raw'}"
            )
            pre = getattr(getattr(self, "_box_zoom_filter", None), "_pre_zoom_state", None)
            if pre:
                # Re-apply y-zoom now and again after fpl's own render pass
                self._box_zoom_filter._apply_y_zoom(pre["y_mid"], pre["y_height"])
                QTimer.singleShot(80, lambda: self._box_zoom_filter._apply_y_zoom(
                    pre["y_mid"], pre["y_height"]
                ))
            else:
                QTimer.singleShot(60, self._fit_camera)
        except Exception as exc:
            logger.exception("TraceWidget render error: %s", exc)

    def _channel_offsets(self, ch_ids: list[int]) -> np.ndarray:
        """Y offset for each channel id in µm probe space."""
        if self._ch_y is not None:
            return np.array(
                [self._ch_y[c] if c < len(self._ch_y) else float(i)
                 for i, c in enumerate(ch_ids)],
                dtype=np.float32,
            )
        return np.arange(len(ch_ids), dtype=np.float32) * 40.0

    def _update_line_collection(self) -> None:
        """Rebuild the trace line collection — one (n_s, 2) array per channel."""
        traces  = self._buf_traces    # (n_ch, n_s)
        t_arr   = self._buf_t_arr     # (n_s,)
        ch_ids  = self._buf_ch_ids
        offsets = self._channel_offsets(ch_ids)   # (n_ch,)
        n_ch, n_s = traces.shape

        lines = []
        for i in range(n_ch):
            xy = np.empty((n_s, 2), dtype=np.float32)
            xy[:, 0] = t_arr
            xy[:, 1] = offsets[i] + traces[i]
            lines.append(xy)

        sp = self._subplot

        # Try add_line_collection; fall back to single NaN-separated line
        if self._line_collection is not None:
            try:
                sp.delete_graphic(self._line_collection)
            except Exception:
                pass
            self._line_collection = None

        try:
            self._line_collection = sp.add_line_collection(
                data=lines,
                colors=_TRACE_COLOR,
                thickness=0.5,
            )
        except AttributeError:
            # Fallback: NaN-separated single line
            n_pts = n_ch * (n_s + 1)
            out = np.full((n_pts, 2), np.nan, dtype=np.float32)
            for i in range(n_ch):
                s = i * (n_s + 1)
                out[s:s + n_s, 0] = t_arr
                out[s:s + n_s, 1] = offsets[i] + traces[i]
            self._line_collection = sp.add_line(out, colors=_TRACE_COLOR, thickness=1.0)

    def _rebuild_ticks(self) -> None:
        """Overlay coloured waveform snippets from the live trace at each spike time.

        For every spike in the current window we extract the actual raw-trace
        segment (±_SPIKE_HALF_MS milliseconds) on each relevant channel and
        draw it in the cluster colour on top of the grey trace.  The result
        looks identical to phy2's trace view: the waveform shape itself is
        coloured, not a separate tick glyph.
        """
        if not self._fpl_ready or self._buf_t_arr is None or self._buf_traces is None:
            return
        if self._window_spikes is None and not self._cluster_data:
            return

        sp      = self._subplot
        traces  = self._buf_traces    # (n_ch, n_s)
        t_arr   = self._buf_t_arr     # (n_s,)
        ch_ids  = self._buf_ch_ids
        offsets = self._channel_offsets(ch_ids)   # (n_ch,) µm
        t0, t1  = float(t_arr[0]), float(t_arr[-1])

        # Half-width of highlighted snippet in samples (~1 ms each side)
        # Compute half-window in display-buffer indices so the highlight always
        # covers ±1 ms of real time regardless of how heavily the data is
        # downsampled.  _buf_t_arr is the actual time axis of the displayed
        # traces, so its spacing already accounts for min-max decimation.
        n_display = len(t_arr)
        display_duration = float(t_arr[-1]) - float(t_arr[0])
        dt_display = display_duration / max(n_display - 1, 1)   # seconds per display sample
        half_win = max(1, int(0.001 / dt_display))              # indices for 1 ms

        # Build per-cluster spike times
        spikes_by_cid: dict[int, np.ndarray] = {}
        if self._window_spikes is not None and len(self._window_spikes):
            ws = self._window_spikes
            for cid in np.unique(ws[:, 1]).astype(int):
                spikes_by_cid[cid] = ws[ws[:, 1] == cid, 0]
        else:
            for cid, (times, _ch, _col) in self._cluster_data.items():
                mask = (times >= t0) & (times <= t1)
                spikes_by_cid[cid] = times[mask]

        selected = set(self._cluster_data.keys())
        sel_list = list(self._cluster_data.keys())

        def _cluster_color(cid: int, selected_idx: int | None) -> tuple:
            if selected_idx is not None:
                return _CLUSTER_COLORS[selected_idx % len(_CLUSTER_COLORS)]
            rng = np.random.default_rng(seed=abs(cid) % (2**31))
            h = rng.uniform(0, 1)
            r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.75)
            return (r, g, b, 0.7)

        # Compute which (cid, ch) keys should exist
        active_keys: set[tuple[int, int]] = set()
        for cid, spike_times in spikes_by_cid.items():
            if len(spike_times) == 0:
                continue
            for ch in self._channels_for_cid(cid):
                if ch in ch_ids:
                    active_keys.add((cid, ch))

        # Remove stale graphics
        for key in list(self._tick_lines.keys()):
            if key not in active_keys:
                try:
                    sp.delete_graphic(self._tick_lines.pop(key))
                except Exception:
                    self._tick_lines.pop(key, None)

        for cid, spike_times in spikes_by_cid.items():
            if len(spike_times) == 0:
                continue

            sel_idx = sel_list.index(cid) if cid in selected else None
            rgba    = _cluster_color(cid, sel_idx)
            thick   = 2.5 if cid in selected else 1.5

            for ch in self._channels_for_cid(cid):
                if ch not in ch_ids:
                    continue
                ch_idx = ch_ids.index(ch)
                y_off  = float(offsets[ch_idx])

                # Build one snippet per spike from the actual trace data
                snippets = []
                for t_spike in spike_times:
                    idx = int(np.searchsorted(t_arr, t_spike))
                    i0  = max(0, idx - half_win)
                    i1  = min(len(t_arr), idx + half_win + 1)
                    if i1 - i0 < 2:
                        continue
                    xy = np.empty((i1 - i0, 2), dtype=np.float32)
                    xy[:, 0] = t_arr[i0:i1]
                    xy[:, 1] = y_off + traces[ch_idx, i0:i1]
                    snippets.append(xy)

                if not snippets:
                    continue

                key = (cid, ch)
                # Replace existing graphic
                if key in self._tick_lines:
                    try:
                        sp.delete_graphic(self._tick_lines.pop(key))
                    except Exception:
                        self._tick_lines.pop(key, None)

                try:
                    self._tick_lines[key] = sp.add_line_collection(
                        data=snippets, colors=rgba, thickness=thick,
                    )
                except Exception as exc:
                    logger.warning("TraceWidget: spike highlight failed: %s", exc)

    def _channels_for_cid(self, cid: int) -> list[int]:
        """Return the channels to draw ticks on for a cluster."""
        if cid in self._cluster_data:
            chs = self._cluster_data[cid][1]   # list of channels (selected cluster)
            return chs if isinstance(chs, list) else [chs]
        # Unselected cluster: use pre-loaded top-N channels
        best = self._all_best_channels.get(cid)
        if best is None:
            return []
        return best if isinstance(best, list) else [best]

    def _fit_camera(self) -> None:
        """Fit camera to current traces — called after each data reload."""
        if self._fig is None or self._buf_t_arr is None:
            return
        try:
            sp = self._subplot
            offsets = self._channel_offsets(self._buf_ch_ids)
            t0  = float(self._buf_t_arr[0])
            t1  = float(self._buf_t_arr[-1])
            y0  = float(offsets.min())
            y1  = float(offsets.max())
            mid_x = (t0 + t1) / 2
            mid_y = (y0 + y1) / 2
            w = (t1 - t0) * 1.04
            h = (y1 - y0 + 40) * 1.08   # 40 µm padding for top/bottom channel

            state = sp.camera.get_state()
            sp.camera.set_state({
                'position':        np.array([mid_x, mid_y, state['position'][2]]),
                'fov':             0.0,
                'width':           w,
                'height':          h,
                'depth':           state['depth'],
                'zoom':            1.0,
                'maintain_aspect': False,
            })
            self._fig.canvas.request_draw()
        except Exception as exc:
            logger.debug("TraceWidget fit_camera: %s", exc)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _scroll(self, frac: float) -> None:
        """Shift window by frac * window_s."""
        self._t_start = max(0.0, self._t_start + frac * self._window_s)
        self._schedule_fetch()

    def _zoom_time(self, factor: float) -> None:
        """Zoom time axis: factor > 1 widens, < 1 narrows."""
        mid = self._t_start + self._window_s / 2
        self._window_s = max(0.01, self._window_s * factor)
        self._t_start  = max(0.0, mid - self._window_s / 2)
        self._schedule_fetch()

    def _zoom_amp(self, factor: float) -> None:
        """Scale amplitude without re-fetching."""
        self._y_scale = float(np.clip(self._y_scale * factor, 0.01, 200.0))
        if self._buf_traces is not None:
            # Re-apply scale: undo old, apply new
            self._buf_traces *= factor
            if self._fpl_ready:
                try:
                    self._update_line_collection()
                    self._fig.canvas.request_draw()
                except Exception as exc:
                    logger.debug("amp zoom redraw: %s", exc)

    def _reset_zoom(self) -> None:
        """Double-click: restore pre-box-zoom state, or fit camera to current data."""
        pre = getattr(self._box_zoom_filter, "_pre_zoom_state", None)
        if pre is not None:
            self._t_start  = pre["t_start"]
            self._window_s = pre["window_s"]
            self._box_zoom_filter._pre_zoom_state = None
            self._schedule_fetch()
        else:
            self._fit_camera()

    def _on_filter_toggled(self, checked: bool) -> None:
        self._filtered = checked
        self._schedule_fetch()

    # ------------------------------------------------------------------
    # Right-click pan (fpl pointer events)
    # ------------------------------------------------------------------

    def _on_pointer_down(self, event) -> None:
        if getattr(event, "button", None) == 2:   # right button
            self._pan_last_x = getattr(event, "x", None)

    def _on_pointer_move(self, event) -> None:
        if self._pan_last_x is None:
            return
        if getattr(event, "button", None) != 2:
            self._pan_last_x = None
            return
        x = getattr(event, "x", None)
        if x is None:
            return
        dx_px = x - self._pan_last_x
        self._pan_last_x = x
        # Convert pixel delta to time delta
        try:
            w_px = self._canvas_widget.width()
            if w_px > 0:
                dt = -(dx_px / w_px) * self._window_s
                self._t_start = max(0.0, self._t_start + dt)
                self._schedule_fetch()
        except Exception:
            pass

    def _on_pointer_up(self, event) -> None:
        self._pan_last_x = None

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._ch_positions is not None and self._buf_traces is None:
            self._schedule_fetch()


# ---------------------------------------------------------------------------
# Qt event filter: wheel + keyboard navigation
# ---------------------------------------------------------------------------

class _TraceEventFilter(QObject):
    _WHEEL_STEP = 1.20

    def __init__(self, widget: TraceWidget, canvas, parent=None):
        super().__init__(parent)
        self._w      = widget
        self._canvas = canvas

    def _over_canvas(self, obj) -> bool:
        w = obj
        while w is not None:
            if w is self._canvas:
                return True
            w = w.parent()
        return False

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            if not self._over_canvas(obj):
                return False
            mods  = event.modifiers()
            delta = event.angleDelta().y()
            if delta == 0:
                return False

            ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
            meta = bool(mods & Qt.KeyboardModifier.MetaModifier)
            alt  = bool(mods & Qt.KeyboardModifier.AltModifier)
            factor = self._WHEEL_STEP ** (delta / 120)

            if ctrl or meta:
                # Ctrl/Cmd+wheel → zoom time axis
                self._w._zoom_time(1.0 / factor)
            elif alt:
                # Alt+wheel → amplitude zoom
                self._w._zoom_amp(factor)
            else:
                # Plain scroll → pan time (scroll right = forward)
                self._w._scroll(0.1 * (-1 if delta > 0 else 1))
            return True   # always consume so fpl camera doesn't interfere

        if event.type() == QEvent.Type.KeyPress:
            if not self._over_canvas(obj):
                return False
            mods = event.modifiers()
            key  = event.key()
            alt  = bool(mods & Qt.KeyboardModifier.AltModifier)

            if key == Qt.Key.Key_Left and alt:
                self._w._scroll(-_SHIFT_FRAC)
                return True
            if key == Qt.Key.Key_Right and alt:
                self._w._scroll(+_SHIFT_FRAC)
                return True
            if key == Qt.Key.Key_F:
                self._w._scroll(+1.0)
                return True
            if key == Qt.Key.Key_B:
                self._w._scroll(-1.0)
                return True

        return False


# ---------------------------------------------------------------------------
# Box-zoom event filter: left-drag draws rubber band, release zooms,
# double-click restores previous view.
# ---------------------------------------------------------------------------

class _BoxZoomFilter(QObject):
    _MIN_DRAG_PX = 8   # ignore tiny accidental drags

    def __init__(self, widget: "TraceWidget", canvas: QWidget, parent=None):
        super().__init__(parent)
        self._w      = widget
        self._canvas = canvas
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, canvas)
        self._origin: QPoint | None = None
        self._pre_zoom_state: dict | None = None

    def _over_canvas(self, obj) -> bool:
        w = obj
        while w is not None:
            if w is self._canvas:
                return True
            w = w.parent()
        return False

    def eventFilter(self, obj, event) -> bool:
        if not self._over_canvas(obj):
            return False

        t = event.type()

        if t == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self._origin = event.pos()
                self._rubber.setGeometry(QRect(self._origin, self._origin))
                self._rubber.show()
                return True

        elif t == QEvent.Type.MouseMove:
            if self._origin is not None:
                self._rubber.setGeometry(
                    QRect(self._origin, event.pos()).normalized()
                )
                return True

        elif t == QEvent.Type.MouseButtonRelease:
            if self._origin is not None and event.button() == Qt.MouseButton.LeftButton:
                self._rubber.hide()
                rect = QRect(self._origin, event.pos()).normalized()
                self._origin = None
                if rect.width() > self._MIN_DRAG_PX and rect.height() > self._MIN_DRAG_PX:
                    self._apply_zoom(rect)
                return True

        elif t == QEvent.Type.MouseButtonDblClick:
            if event.button() == Qt.MouseButton.LeftButton:
                self._origin = None
                self._rubber.hide()
                self._w._reset_zoom()
                return True

        return False

    def _pixel_to_data(self, px: int, py: int) -> tuple[float, float]:
        """Convert canvas pixel (px, py) → data (time_s, y_µm)."""
        try:
            state = self._w._subplot.camera.get_state()
            cx = float(state["position"][0])
            cy = float(state["position"][1])
            dw = float(state["width"])
            dh = float(state["height"])
            W  = self._canvas.width()
            H  = self._canvas.height()
            if W <= 0 or H <= 0:
                return 0.0, 0.0
            t = cx + (px / W - 0.5) * dw
            y = cy + (0.5 - py / H) * dh   # screen y is flipped vs data y
            return t, y
        except Exception:
            return 0.0, 0.0

    def _apply_zoom(self, rect: QRect) -> None:
        t0, y1 = self._pixel_to_data(rect.left(),  rect.top())
        t1, y0 = self._pixel_to_data(rect.right(), rect.bottom())

        t0 = max(0.0, t0)
        t1 = max(t0 + 0.001, t1)

        # Save state so double-click can restore
        self._pre_zoom_state = {
            "t_start":  self._w._t_start,
            "window_s": self._w._window_s,
            "y_mid":    (y0 + y1) / 2.0,
            "y_height": abs(y1 - y0) * 1.05,
        }

        # Update time window → triggers fetch + full re-render
        self._w._t_start  = t0
        self._w._window_s = t1 - t0
        self._w._schedule_fetch()

        # Immediately narrow camera y to the selected band
        self._apply_y_zoom(self._pre_zoom_state["y_mid"],
                           self._pre_zoom_state["y_height"])

    def _apply_y_zoom(self, mid_y: float, h: float) -> None:
        try:
            sp    = self._w._subplot
            state = sp.camera.get_state()
            # X range from the actual fetched data (updated after each fetch)
            t_arr = self._w._buf_t_arr
            if t_arr is not None and len(t_arr) > 1:
                t0, t1 = float(t_arr[0]), float(t_arr[-1])
                mid_x  = (t0 + t1) / 2.0
                w      = (t1 - t0) * 1.04
            else:
                mid_x = state["position"][0]
                w     = state["width"]
            sp.camera.set_state({
                "position":        np.array([mid_x, mid_y, state["position"][2]]),
                "fov":             0.0,
                "width":           w,
                "height":          h,
                "depth":           state["depth"],
                "zoom":            1.0,
                "maintain_aspect": False,
            })
            self._w._fig.canvas.request_draw()
        except Exception as exc:
            logger.debug("BoxZoom: camera set failed: %s", exc)
