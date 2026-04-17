"""Amplitude views — spike amplitude scatter plots using fastplotlib.

AmplitudeTimeWidget
    Scatter of spike amplitude vs. recording time for selected clusters.
    Decimates to at most _MAX_PTS points per cluster for GPU performance.

TemplateAmplitudeWidget
    Scatter of mean template amplitude vs. cluster rank for ALL clusters.
    Selected clusters are highlighted; all others are shown as faded grey.
    Refreshes whenever cluster selection or cluster list changes.
"""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from phy_remote.client.views._zoom import install_zoom_filter
from phy_remote.client.views._colors import cluster_color
from phy_remote.client.views._graphics import safe_delete

logger = logging.getLogger(__name__)

_MAX_PTS   = 100_000   # max scatter points per cluster (AmplitudeTimeWidget)
_PT_SIZE   = 2
_ALPHA_SEL = 0.70      # selected cluster point alpha

# TemplateAmplitudeWidget — label colours matching phy2 conventions
_LABEL_RGBA: dict[str, np.ndarray] = {
    "good":     np.array([0.22, 0.75, 0.38, 0.80], dtype=np.float32),   # green
    "mua":      np.array([0.90, 0.55, 0.12, 0.80], dtype=np.float32),   # orange
    "noise":    np.array([0.82, 0.22, 0.22, 0.80], dtype=np.float32),   # red
    "unsorted": np.array([0.55, 0.55, 0.55, 0.55], dtype=np.float32),   # grey
}
_BASE_PT_SIZE = 4    # dots for all clusters
_SEL_PT_SIZE  = 8    # highlighted dot for selected clusters


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _decimate(arr: np.ndarray, max_pts: int, seed: int = 0) -> np.ndarray:
    """Random-sample *arr* to at most *max_pts* rows. Deterministic."""
    if len(arr) <= max_pts:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), max_pts, replace=False)
    idx.sort()
    return arr[idx]


def _fit_camera(sp, pts: np.ndarray, fig) -> None:
    """Set *sp*'s camera to fit *pts* with 5 % padding."""
    if len(pts) == 0:
        return
    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
    dx = max(x1 - x0, 1e-6) * 1.05
    dy = max(y1 - y0, 1e-6) * 1.10
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    try:
        state = sp.camera.get_state()
        pos = state["position"].copy()
        pos[0], pos[1] = cx, cy
        pos[2] = state["depth"] / 2
        sp.camera.set_state({
            "position": pos, "fov": 0.0,
            "width": dx, "height": dy,
            "depth": state["depth"], "zoom": 1.0,
            "maintain_aspect": False,
        })
        fig.canvas.request_draw()
    except Exception as exc:
        logger.debug("_fit_camera: %s", exc)


def _make_fpl_subplot(layout: QVBoxLayout, parent: QWidget):
    """Create a single-subplot fastplotlib figure, return (fig, subplot, canvas)."""
    try:
        import fastplotlib as fpl
        fig = fpl.Figure(canvas="qt")
        sp  = fig[0, 0]
        try:
            sp.camera = "2d"
        except Exception:
            pass
        try:
            sp.axes.visible = False
            sp.title.visible = False
        except Exception:
            pass
        fig.show()
        canvas = fig.canvas
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas.setMinimumSize(160, 80)
        layout.addWidget(canvas)
        return fig, sp, canvas
    except Exception as exc:
        logger.warning("fastplotlib init failed: %s", exc)
        ph = QLabel("fastplotlib unavailable")
        ph.setAlignment(
            __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignCenter
        )
        layout.addWidget(ph)
        return None, None, None


# ---------------------------------------------------------------------------
# AmplitudeTimeWidget
# ---------------------------------------------------------------------------

class AmplitudeTimeWidget(QWidget):
    """Spike amplitude vs. time scatter for selected clusters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel("Amplitude vs time")
        hdr.setFixedHeight(18)
        hdr.setStyleSheet(
            "background:#252525;color:#aaa;font-size:10px;"
            "padding-left:6px;border-bottom:1px solid #333;"
        )
        layout.addWidget(hdr)

        self._fig:    object | None = None
        self._sp:     object | None = None
        self._scatters: dict[int, object] = {}
        self._all_pts:  np.ndarray | None = None   # stacked for camera fit

        self._fig, self._sp, _ = _make_fpl_subplot(layout, self)
        if self._fig is not None:
            install_zoom_filter(self._fig, [self._sp], self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_spike_data(self, spike_data: "dict[int, np.ndarray]") -> None:
        """spike_data: {cluster_id -> (n_spikes, 2) array [time_s, amplitude]}"""
        if self._fig is None:
            return
        if not spike_data:
            self._clear()
            return
        try:
            self._render(spike_data)
        except Exception as exc:
            logger.exception("AmplitudeTimeWidget render: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _clear(self) -> None:
        for sc in list(self._scatters.values()):
            safe_delete(self._sp, sc)
        self._scatters.clear()
        self._all_pts = None

    def _render(self, spike_data: "dict[int, np.ndarray]") -> None:
        sp = self._sp
        active = set(spike_data.keys())

        # Remove stale
        for cid in list(self._scatters):
            if cid not in active:
                safe_delete(sp, self._scatters.pop(cid))

        all_chunks = []
        for slot, (cid, data) in enumerate(spike_data.items()):
            raw = data[:, :2].astype(np.float32)          # (n, 2) [time, amp]
            pts = _decimate(raw, _MAX_PTS, seed=cid)
            all_chunks.append(pts)

            rgba = np.array(cluster_color(slot, alpha=_ALPHA_SEL), dtype=np.float32)
            colors = np.broadcast_to(rgba, (len(pts), 4)).copy()

            if cid in self._scatters:
                sc = self._scatters[cid]
                try:
                    if sc.data.value.shape[0] == len(pts):
                        xyz = np.zeros((len(pts), 3), dtype=np.float32)
                        xyz[:, :2] = pts
                        sc.data[:] = xyz
                        sc.colors[:] = colors
                        continue
                    else:
                        safe_delete(sp, self._scatters.pop(cid))
                except Exception:
                    safe_delete(sp, self._scatters.pop(cid, None))

            try:
                self._scatters[cid] = sp.add_scatter(pts, sizes=_PT_SIZE, colors=colors)
            except Exception as exc:
                logger.warning("AmplitudeTimeWidget add_scatter: %s", exc)

        if all_chunks:
            self._all_pts = np.concatenate(all_chunks, axis=0)
            QTimer.singleShot(50, self._fit)

    def _fit(self) -> None:
        if self._all_pts is not None and self._fig is not None:
            _fit_camera(self._sp, self._all_pts, self._fig)


# ---------------------------------------------------------------------------
# TemplateAmplitudeWidget
# ---------------------------------------------------------------------------

class TemplateAmplitudeWidget(QWidget):
    """All-cluster scatter: amplitude (y) vs depth (x), coloured by label.

    Mirrors phy2's ClusterScatterView:
    - Every cluster is a dot coloured by its label (good/mua/noise/unsorted).
    - Selected clusters are rendered on top in brighter cluster colours.
    - Falls back to cluster index on x when depth is unavailable.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel("Amplitude vs depth")
        hdr.setFixedHeight(18)
        hdr.setStyleSheet(
            "background:#252525;color:#aaa;font-size:10px;"
            "padding-left:6px;border-bottom:1px solid #333;"
        )
        layout.addWidget(hdr)

        self._fig:      object | None = None
        self._sp:       object | None = None
        self._sc_all:   object | None = None   # all-cluster scatter (label colours)
        self._sc_sel:   object | None = None   # selected-cluster overlay
        self._clusters: list[dict]    = []
        self._selected: list[int]     = []

        self._fig, self._sp, _ = _make_fpl_subplot(layout, self)
        if self._fig is not None:
            install_zoom_filter(self._fig, [self._sp], self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_cluster_info(self, clusters: "list[dict]") -> None:
        """Receive full cluster list (called at startup and after merge/undo)."""
        self._clusters = clusters
        self._render()

    def set_selected(self, cluster_ids: "list[int]") -> None:
        """Highlight these cluster IDs."""
        self._selected = list(cluster_ids)
        self._render()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if self._fig is None or not self._clusters:
            return
        sp = self._sp

        has_depth = any("depth" in c for c in self._clusters)

        # Build per-cluster (x, y) and colour from label
        pts_list:   list[list[float]] = []
        color_list: list[np.ndarray]  = []
        for i, c in enumerate(self._clusters):
            x = float(c["depth"]) if has_depth else float(i)
            y = float(c.get("amplitude", 0.0))
            pts_list.append([x, y])
            lbl = c.get("label", "unsorted")
            color_list.append(_LABEL_RGBA.get(lbl, _LABEL_RGBA["unsorted"]))

        all_pts    = np.array(pts_list,  dtype=np.float32)
        all_colors = np.array(color_list, dtype=np.float32)

        if self._sc_all is not None:
            try:
                safe_delete(sp, self._sc_all)
            except Exception:
                pass
        try:
            self._sc_all = sp.add_scatter(all_pts, sizes=_BASE_PT_SIZE, colors=all_colors)
        except Exception as exc:
            logger.warning("TemplateAmplitudeWidget add_scatter all: %s", exc)

        # Selected overlay — brighter cluster colours on top
        if self._sc_sel is not None:
            try:
                safe_delete(sp, self._sc_sel)
            except Exception:
                pass
            self._sc_sel = None

        if self._selected:
            id_to_cluster = {int(c["id"]): c for c in self._clusters}
            sel_pts, sel_cols = [], []
            for slot, cid in enumerate(self._selected):
                c = id_to_cluster.get(int(cid))
                if c is None:
                    continue
                x = float(c["depth"]) if has_depth else 0.0
                y = float(c.get("amplitude", 0.0))
                sel_pts.append([x, y])
                sel_cols.append(list(cluster_color(slot, alpha=0.95)))

            if sel_pts:
                try:
                    self._sc_sel = sp.add_scatter(
                        np.array(sel_pts, dtype=np.float32),
                        sizes=_SEL_PT_SIZE,
                        colors=np.array(sel_cols, dtype=np.float32),
                    )
                except Exception as exc:
                    logger.warning("TemplateAmplitudeWidget add_scatter sel: %s", exc)

        QTimer.singleShot(50, self._fit)

    def _fit(self) -> None:
        if self._fig is None or not self._clusters:
            return
        has_depth = any("depth" in c for c in self._clusters)
        xs = [float(c["depth"]) if has_depth else float(i)
              for i, c in enumerate(self._clusters)]
        ys = [float(c.get("amplitude", 0.0)) for c in self._clusters]
        pts = np.array([[min(xs), min(ys)], [max(xs), max(ys)]], dtype=np.float32)
        _fit_camera(self._sp, pts, self._fig)


# Backwards-compat alias
AmplitudeWidget = AmplitudeTimeWidget
