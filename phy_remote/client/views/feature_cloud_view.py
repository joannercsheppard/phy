"""FeatureView — 4×4 PC feature scatter grid, matching phy2 conventions.

Grid layout (matches phy's default _get_default_grid):

    time,0A | 1A,0A | 0B,0A | 1B,0A
    0A,1A   | time,1A | 0B,1A | 1B,1A
    0A,0B   | 1A,0B | time,0B | 1B,0B
    0A,1B   | 1A,1B | 0B,1B | time,1B

Each dim spec is "<ch_rel><PC_letter>" or "time".
Labels shown are "<channel_id><PC_letter>", e.g. "127A", "128B".
Channels are ordered best-first (index 0 = highest-amplitude channel).

Lasso selection: click the Lasso button, then drag a polygon on any subplot.
Selected spikes are emitted via the `spikes_selected` signal.
"""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygon
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from phy_remote.client.views._zoom import install_synced_zoom_filter
from phy_remote.client.views._colors import cluster_color
from phy_remote.client.views._graphics import safe_delete

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid definition — matches phy2's _get_default_grid()
# Each cell: (dim_x, dim_y).  "time" = spike time axis.
# "<n><L>" = relative channel index n, PC component letter L (A=0, B=1, …).
# ---------------------------------------------------------------------------
_GRID: list[list[tuple[str, str]]] = [
    [("time", "0A"), ("1A", "0A"), ("0B", "0A"), ("1B", "0A")],
    [("0A",  "1A"), ("time", "1A"), ("0B", "1A"), ("1B", "1A")],
    [("0A",  "0B"), ("1A",  "0B"), ("time", "0B"), ("1B", "0B")],
    [("0A",  "1B"), ("1A",  "1B"), ("0B",  "1B"), ("time", "1B")],
]
_N_ROWS = _N_COLS = 4
_PC = "ABCDEFGHIJ"        # PC component → index: A=0, B=1, …
_MAX_SPIKES = 2_500       # max displayed points per cluster per subplot

_LASSO_COLOR = np.array([1.0, 0.85, 0.0, 1.0], dtype=np.float32)   # bright yellow


def _dim_color(rgba: np.ndarray) -> np.ndarray:
    """Return a faded version of *rgba* for non-selected points during a lasso."""
    out = rgba.copy()
    out[3] = 0.10   # very transparent
    return out


def _local_highlight_mask(
    scatter_orig: "np.ndarray | None",
    lasso_orig: np.ndarray,
    n: int,
) -> np.ndarray:
    """Boolean mask of length *n* — True where scatter point is in *lasso_orig*.

    *scatter_orig* is the original spike index for each scatter point (or None
    meaning positions 0..n-1 were used with no subsampling).
    *lasso_orig* is the set of original spike indices selected by the lasso.
    """
    if scatter_orig is None:
        positions = np.arange(n)
    else:
        positions = scatter_orig
    lasso_set = set(lasso_orig.tolist())
    return np.array([p in lasso_set for p in positions], dtype=bool)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon test.

    Parameters
    ----------
    points  : (N, 2) float64
    polygon : (M, 2) float64, vertices in order (open or closed)

    Returns bool mask of length N.
    """
    inside = np.zeros(len(points), dtype=bool)
    px, py = points[:, 0], points[:, 1]
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        cross = (y1 > py) != (y2 > py)
        denom = y2 - y1 if y2 != y1 else 1e-10
        t = (x2 - x1) * (py - y1) / denom + x1
        inside ^= cross & (px < t)
        x1, y1 = x2, y2
    return inside


# ---------------------------------------------------------------------------
# Lasso overlay
# ---------------------------------------------------------------------------

class _LassoOverlay(QWidget):
    """Transparent QWidget layered on top of the fpl canvas.

    Invisible and pass-through when lasso mode is off.
    When active: captures mouse events, draws the polygon, emits
    ``spikes_selected`` with ``{cid: np.ndarray_of_local_scatter_indices}``.
    """

    spikes_selected = pyqtSignal(dict)

    def __init__(self, canvas: QWidget, fig, feature_widget: "FeatureViewWidget"):
        super().__init__(canvas)
        self._canvas = canvas
        self._fig    = fig
        self._fw     = feature_widget

        self._poly: list[QPoint] = []
        self._dragging = False
        self._subplot_rc: tuple[int, int] | None = None

        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Start transparent so canvas events pass through normally
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setGeometry(canvas.rect())
        self.show()
        self.raise_()

        # Track canvas resize so we stay aligned
        canvas.installEventFilter(self)

    # ---- activation -------------------------------------------------------

    def set_active(self, active: bool) -> None:
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not active)
        self.setGeometry(self._canvas.rect())
        self.raise_()
        if not active:
            self._poly.clear()
            self._dragging = False
            self._subplot_rc = None
            self.update()

    # ---- event filter (resize tracking) -----------------------------------

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self._canvas and event.type() == QEvent.Type.Resize:
            self.setGeometry(self._canvas.rect())
        return False

    # ---- mouse ------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            px, py = int(event.position().x()), int(event.position().y())
            self._dragging = True
            self._poly = [QPoint(px, py)]
            self._subplot_rc = self._fw._subplot_at_pixel(px, py)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._poly.append(event.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            if len(self._poly) >= 3 and self._subplot_rc is not None:
                self._emit_selection()
            self._poly.clear()
            self._subplot_rc = None
            self.update()

    # ---- painting ---------------------------------------------------------

    def paintEvent(self, event) -> None:
        if not self._poly:
            return
        p = QPainter(self)
        p.setPen(QPen(QColor(255, 200, 0, 210), 1.5, Qt.PenStyle.DashLine))
        p.setBrush(QColor(255, 200, 0, 22))
        p.drawPolygon(QPolygon(self._poly))

    # ---- lasso logic ------------------------------------------------------

    def _emit_selection(self) -> None:
        r, c = self._subplot_rc
        # Convert polygon vertices from pixel → data coords in this subplot
        poly_data = np.array(
            [self._fw._pixel_to_data(pt.x(), pt.y(), r, c) for pt in self._poly],
            dtype=np.float64,
        )
        scatter_data = self._fw.get_scatter_data_for(r, c)
        # local_sel: {cid: local scatter indices inside polygon}
        local_sel: dict[int, np.ndarray] = {}
        for cid, pts in scatter_data.items():
            if len(pts) == 0:
                continue
            mask = _points_in_polygon(pts.astype(np.float64), poly_data)
            if mask.any():
                local_sel[cid] = np.where(mask)[0]

        if local_sel:
            total = sum(len(v) for v in local_sel.values())
            self._fw._status.setText(
                f"Lasso: {total} spike(s) across {len(local_sel)} cluster(s)"
            )
            # Highlight selected spikes in yellow across all subplots
            self._fw.apply_lasso_highlight(r, c, local_sel)
            self.spikes_selected.emit(local_sel)
        else:
            self._fw._status.setText("Lasso: no spikes inside selection")
            self._fw.clear_lasso_highlight()


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class FeatureViewWidget(QWidget):
    """4×4 grid of PC feature scatter plots, matching phy2's FeatureView layout."""

    spikes_selected = pyqtSignal(dict)   # {cid: np.ndarray of local scatter indices}

    def __init__(self, parent=None):
        super().__init__(parent)

        self._fig: object | None = None
        self._canvas: QWidget | None = None
        self._lasso_overlay: _LassoOverlay | None = None

        # {(r, c, cid): scatter graphic}
        self._scatters: dict[tuple[int, int, int], object] = {}
        # {(r, c, cid): np.ndarray of original spike indices shown in that scatter}
        # None means all spikes were shown (no subsampling)
        self._scatter_sel: dict[tuple[int, int, int], "np.ndarray | None"] = {}
        # {cid: set of original spike indices currently highlighted by lasso}
        self._lasso_sel: dict[int, "np.ndarray"] = {}
        # {cid: {times, features (n,ch,pc), ch_ids, display_idx}}
        self._cdata: dict[int, dict] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- toolbar ---
        bar = QWidget()
        bar.setFixedHeight(26)
        bar.setStyleSheet("background:#252525;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(4, 2, 4, 2)
        bl.setSpacing(4)

        self._lasso_btn = QPushButton("⬡ Lasso")
        self._lasso_btn.setCheckable(True)
        self._lasso_btn.setFixedHeight(20)
        self._lasso_btn.setStyleSheet(
            "color:#ccc;background:#3a3a3a;border:1px solid #555;"
            "border-radius:3px;font-size:11px;padding:0 6px;"
        )
        self._lasso_btn.toggled.connect(self._on_lasso_toggled)
        bl.addWidget(self._lasso_btn)
        bl.addStretch()
        self._status = QLabel("")
        self._status.setStyleSheet("color:#888;font-size:10px;")
        bl.addWidget(self._status)
        layout.addWidget(bar)

        # --- fastplotlib canvas ---
        try:
            import fastplotlib as fpl
            self._init_fpl(fpl, layout)
        except Exception as exc:
            logger.warning("FeatureViewWidget: fastplotlib init failed: %s", exc)
            ph = QLabel("fastplotlib unavailable")
            ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(ph)

    # ------------------------------------------------------------------
    # fpl init
    # ------------------------------------------------------------------

    def _init_fpl(self, fpl, layout: QVBoxLayout) -> None:
        self._fig = fpl.Figure(shape=(_N_ROWS, _N_COLS), canvas="qt")
        for r in range(_N_ROWS):
            for c in range(_N_COLS):
                sp = self._fig[r, c]
                try:
                    sp.camera = "2d"
                except Exception:
                    pass
                try:
                    sp.axes.visible = False
                except Exception:
                    pass
                try:
                    sp.title.visible = False
                except Exception:
                    pass

        self._fig.show()
        self._canvas = self._fig.canvas
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._canvas.setMinimumSize(200, 200)
        layout.addWidget(self._canvas)

        all_sps = [self._fig[r, c] for r in range(_N_ROWS) for c in range(_N_COLS)]
        install_synced_zoom_filter(self._fig, all_sps, self)

        self._lasso_overlay = _LassoOverlay(self._canvas, self._fig, self)
        self._lasso_overlay.spikes_selected.connect(self.spikes_selected)

    # ------------------------------------------------------------------
    # Public API  (compatible with existing main_window calls)
    # ------------------------------------------------------------------

    def set_feature_data(
        self,
        feature_data: "dict[int, np.ndarray]",
        spike_times: "dict[int, np.ndarray] | None" = None,
        channel_ids: "dict[int, list[int]] | None" = None,
    ) -> None:
        """
        Parameters
        ----------
        feature_data : {cid: array}
            Array shape (n_spikes, n_channels, n_pcs) or (n_spikes, n_features).
        spike_times  : {cid: 1-D float array in seconds}, used for "time" axis.
        channel_ids  : {cid: list[int]} physical channel IDs ordered best-first.
            If omitted, labels fall back to relative indices ("0A", "1A", …).
        """
        if self._fig is None:
            return
        if not feature_data:
            self._clear_all()
            return

        times   = spike_times  or {}
        ch_map  = channel_ids  or {}

        new_cdata: dict[int, dict] = {}
        for idx, cid in enumerate(feature_data):
            arr = np.asarray(feature_data[cid], dtype=np.float32)
            # Normalise to 3-D (n_spikes, n_ch, n_pc)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            ch_ids = ch_map.get(cid) or list(range(arr.shape[1]))
            new_cdata[cid] = {
                "features": arr,
                "times":    times.get(cid, np.empty(0, np.float32)),
                "ch_ids":   ch_ids,
                "idx":      idx,
            }

        self._cdata = new_cdata

        # Log what we received so mismatched shapes are easy to spot
        for cid, cdat in new_cdata.items():
            f = cdat["features"]
            logger.info(
                "FeatureView cid=%d  features shape=%s  ch_ids=%s  spike_times_len=%d",
                cid, f.shape, cdat["ch_ids"][:4], len(cdat["times"]),
            )

        try:
            self._render()
        except Exception as exc:
            logger.exception("FeatureViewWidget render: %s", exc)

        # Update status to show what is actually being displayed
        if new_cdata:
            sample = next(iter(new_cdata.values()))
            f = sample["features"]
            n_ch  = f.shape[1] if f.ndim == 3 else "?"
            n_pc  = f.shape[2] if f.ndim == 3 else "?"
            n_spk = f.shape[0]
            ch_labels = [
                f"{sample['ch_ids'][i]}{_PC[j]}"
                for i in range(min(2, int(n_ch) if isinstance(n_ch, int) else 0))
                for j in range(min(2, int(n_pc) if isinstance(n_pc, int) else 0))
            ]
            self._status.setText(
                f"{n_spk} spikes · {n_ch} ch × {n_pc} PC · "
                + (", ".join(ch_labels) + "…" if ch_labels else "")
            )

    # ------------------------------------------------------------------
    # Coordinate helpers (used by _LassoOverlay)
    # ------------------------------------------------------------------

    def _subplot_at_pixel(self, px: int, py: int) -> "tuple[int, int] | None":
        """Return (row, col) of the subplot that contains canvas pixel (px, py)."""
        if self._canvas is None:
            return None
        w, h = self._canvas.width(), self._canvas.height()
        if w <= 0 or h <= 0:
            return None
        c = min(_N_COLS - 1, int(px / w * _N_COLS))
        r = min(_N_ROWS - 1, int(py / h * _N_ROWS))
        return r, c

    def _pixel_to_data(self, px: int, py: int, r: int, c: int) -> "tuple[float, float]":
        """Convert canvas pixel to data coordinates in subplot (r, c)."""
        sp    = self._fig[r, c]
        state = sp.camera.get_state()
        cx    = float(state["position"][0])
        cy    = float(state["position"][1])
        dw    = float(state["width"])
        dh    = float(state["height"])
        cw    = self._canvas.width()
        ch    = self._canvas.height()
        sp_w  = cw / _N_COLS
        sp_h  = ch / _N_ROWS
        lx    = px - c * sp_w
        ly    = py - r * sp_h
        x = cx + (lx / sp_w - 0.5) * dw
        y = cy + (0.5 - ly / sp_h) * dh
        return x, y

    def get_scatter_data_for(self, r: int, c: int) -> "dict[int, np.ndarray]":
        """Return {cid: (N, 2) float32} of current scatter points in subplot (r, c)."""
        out: dict[int, np.ndarray] = {}
        for (kr, kc, kcid), sc in self._scatters.items():
            if kr != r or kc != c:
                continue
            try:
                val = sc.data.value  # (N, 3) xyz
                out[kcid] = val[:, :2].copy()
            except Exception:
                pass
        return out

    def apply_lasso_highlight(
        self,
        r: int,
        c: int,
        local_sel: "dict[int, np.ndarray]",
    ) -> None:
        """Highlight lasso-selected spikes in yellow across all 16 subplots.

        Parameters
        ----------
        r, c       : subplot where the lasso was drawn (used to resolve local→original)
        local_sel  : {cid: local_scatter_indices} as returned by _LassoOverlay
        """
        if self._fig is None:
            return

        # Convert local scatter indices in the lasso subplot → original spike indices
        self._lasso_sel = {}
        for cid, local_idx in local_sel.items():
            scatter_orig = self._scatter_sel.get((r, c, cid))
            if scatter_orig is None:
                orig = local_idx.astype(np.intp)
            else:
                orig = scatter_orig[local_idx]
            self._lasso_sel[cid] = orig

        # Recolour every scatter across all subplots
        for (kr, kc, kcid), sc in self._scatters.items():
            cdat = self._cdata.get(kcid)
            if cdat is None:
                continue
            base_rgba = np.array(cluster_color(cdat["idx"], alpha=0.45), dtype=np.float32)
            try:
                n = sc.data.value.shape[0]
            except Exception:
                continue

            orig_sel = self._lasso_sel.get(kcid)
            if orig_sel is not None and len(orig_sel):
                scatter_orig = self._scatter_sel.get((kr, kc, kcid))
                local_mask   = _local_highlight_mask(scatter_orig, orig_sel, n)
                colors       = np.broadcast_to(base_rgba, (n, 4)).copy()
                colors[~local_mask] = _dim_color(base_rgba)
                colors[local_mask]  = _LASSO_COLOR
            else:
                colors = np.broadcast_to(base_rgba, (n, 4)).copy()

            try:
                sc.colors[:] = colors
            except Exception as exc:
                logger.debug("apply_lasso_highlight colors update: %s", exc)

        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass

    def clear_lasso_highlight(self) -> None:
        """Reset all scatter colours to their original cluster colour."""
        self._lasso_sel.clear()
        if self._fig is None:
            return
        for (kr, kc, kcid), sc in self._scatters.items():
            cdat = self._cdata.get(kcid)
            if cdat is None:
                continue
            base_rgba = np.array(cluster_color(cdat["idx"], alpha=0.45), dtype=np.float32)
            try:
                n = sc.data.value.shape[0]
                sc.colors[:] = np.broadcast_to(base_rgba, (n, 4)).copy()
            except Exception:
                pass
        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _clear_all(self) -> None:
        if self._fig is None:
            return
        for key in list(self._scatters):
            r, c, _ = key
            safe_delete(self._fig[r, c], self._scatters.pop(key))
        self._scatter_sel.clear()
        self._lasso_sel.clear()
        self._cdata.clear()

    def _get_dim(self, dim: str, cdat: dict) -> "np.ndarray | None":
        """Extract 1-D float32 array for a dimension spec ("time", "0A", "1B"…)."""
        if dim == "time":
            t = cdat["times"]
            return t.astype(np.float32) if len(t) else None

        if not dim[:-1].isdecimal():
            return None
        ch_rel   = int(dim[:-1])
        pc_letter = dim[-1]
        if pc_letter not in _PC:
            return None
        pc_idx = _PC.index(pc_letter)

        feat = cdat["features"]   # (n_spikes, n_ch, n_pc)
        if feat.ndim != 3:
            return None
        n_ch = feat.shape[1]
        n_pc = feat.shape[2]
        if ch_rel >= n_ch or pc_idx >= n_pc:
            return None
        return feat[:, ch_rel, pc_idx]

    def _label(self, dim: str, ch_ids: "list[int]") -> str:
        """Build a human-readable axis label, e.g. "127A" from "0A"."""
        if dim == "time":
            return "time"
        if not dim[:-1].isdecimal():
            return dim
        ch_rel    = int(dim[:-1])
        pc_letter = dim[-1]
        if ch_rel < len(ch_ids):
            return f"{ch_ids[ch_rel]}{pc_letter}"
        return dim

    def _render(self) -> None:
        if self._fig is None or not self._cdata:
            return

        active_cids = set(self._cdata.keys())

        for r in range(_N_ROWS):
            for c in range(_N_COLS):
                dim_x, dim_y = _GRID[r][c]
                sp = self._fig[r, c]

                # Remove scatters for clusters no longer selected
                for key in list(self._scatters):
                    kr, kc, kcid = key
                    if kr == r and kc == c and kcid not in active_cids:
                        safe_delete(sp, self._scatters.pop(key))

                for cid, cdat in self._cdata.items():
                    x = self._get_dim(dim_x, cdat)
                    y = self._get_dim(dim_y, cdat)

                    key = (r, c, cid)
                    if x is None or y is None:
                        if key in self._scatters:
                            safe_delete(sp, self._scatters.pop(key))
                        continue

                    n = min(len(x), len(y))
                    x, y = x[:n], y[:n]

                    if n > _MAX_SPIKES:
                        rng = np.random.default_rng(seed=cid * 31 + r * 7 + c)
                        sel = rng.choice(n, _MAX_SPIKES, replace=False)
                        sel.sort()
                        x, y = x[sel], y[sel]
                        self._scatter_sel[key] = sel
                    else:
                        self._scatter_sel[key] = None   # all points, no subsampling

                    xy         = np.column_stack([x, y]).astype(np.float32)
                    base_rgba  = np.array(
                        cluster_color(cdat["idx"], alpha=0.45), dtype=np.float32
                    )
                    colors_arr = np.broadcast_to(base_rgba, (len(xy), 4)).copy()

                    # Apply any existing lasso highlight
                    orig_sel = self._lasso_sel.get(cid)
                    if orig_sel is not None and len(orig_sel):
                        scatter_orig = self._scatter_sel[key]
                        local_mask = _local_highlight_mask(scatter_orig, orig_sel, len(xy))
                        colors_arr[~local_mask] = _dim_color(base_rgba)
                        colors_arr[local_mask]  = _LASSO_COLOR

                    if key in self._scatters:
                        sc = self._scatters[key]
                        try:
                            if sc.data.value.shape[0] == len(xy):
                                xyz = np.zeros((len(xy), 3), dtype=np.float32)
                                xyz[:, :2] = xy
                                sc.data[:] = xyz
                                sc.colors[:] = colors_arr
                                continue   # updated in place — skip add_scatter
                            else:
                                safe_delete(sp, self._scatters.pop(key))
                        except Exception:
                            safe_delete(sp, self._scatters.pop(key, None))

                    try:
                        self._scatters[key] = sp.add_scatter(
                            xy, sizes=2, colors=colors_arr
                        )
                    except Exception as exc:
                        logger.warning("FeatureViewWidget add_scatter: %s", exc)

        QTimer.singleShot(50, self._auto_scale)

    def _auto_scale(self) -> None:
        """Fit each subplot's camera to the bounding box of its scatter data."""
        if self._fig is None:
            return
        try:
            for r in range(_N_ROWS):
                for c in range(_N_COLS):
                    sp = self._fig[r, c]
                    # Collect all points in this subplot
                    pts_list = []
                    for (kr, kc, _), sc in self._scatters.items():
                        if kr != r or kc != c:
                            continue
                        try:
                            val = sc.data.value  # (N, 3)
                            pts_list.append(val[:, :2])
                        except Exception:
                            pass
                    if not pts_list:
                        continue
                    pts = np.concatenate(pts_list, axis=0)
                    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
                    y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
                    # Add 10 % padding; guard against zero-range axes
                    dx = max(x1 - x0, 1e-6) * 1.10
                    dy = max(y1 - y0, 1e-6) * 1.10
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    state = sp.camera.get_state()
                    pos = state["position"].copy()
                    pos[0], pos[1] = cx, cy
                    pos[2] = state["depth"] / 2
                    sp.camera.set_state({
                        "position":        pos,
                        "fov":             0.0,
                        "width":           dx,
                        "height":          dy,
                        "depth":           state["depth"],
                        "zoom":            1.0,
                        "maintain_aspect": False,
                    })
            self._fig.canvas.request_draw()
        except Exception as exc:
            logger.debug("FeatureViewWidget auto_scale: %s", exc)

    # ------------------------------------------------------------------
    # Lasso toggle
    # ------------------------------------------------------------------

    def _on_lasso_toggled(self, checked: bool) -> None:
        if self._lasso_overlay is not None:
            self._lasso_overlay.set_active(checked)
        self._lasso_btn.setText("⬡ Lasso (ON)" if checked else "⬡ Lasso")
        if not checked:
            self._status.setText("")
            self.clear_lasso_highlight()


# Backwards-compat alias used in main_window
FeatureCloudWidget = FeatureViewWidget
