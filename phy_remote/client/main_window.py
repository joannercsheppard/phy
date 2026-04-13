"""
MainWindow — PyQt6 main window for phy-remote.

Layout
------
+--------------------------------------------------+
|  Menu bar                                        |
+--------------------+-----------------------------+
|  Cluster table     |  Waveform view (central)    |
|  (left dock)       |                             |
|  id | label | fr   |                             |
|  id | label | fr   |                             |
+--------------------+-----------------------------+
|  Status bar   [label buttons: g  m  n  u]        |
+--------------------------------------------------+

Keyboard shortcuts (when cluster table has focus or anywhere):
  g  →  label selected clusters "good"
  m  →  label selected clusters "mua"
  n  →  label selected clusters "noise"
  u  →  label selected clusters "unsorted"
  j  →  move selection down one cluster
  k  →  move selection up one cluster
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Sequence

import numpy as np

from PyQt6.QtCore import (
    Qt, QObject,
    QAbstractTableModel, QModelIndex, QItemSelectionModel,
    pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import QColor, QKeySequence, QShortcut, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QTableView, QHeaderView, QStatusBar, QLabel, QPushButton,
    QAbstractItemView, QSizePolicy,
)

from phy_remote.client.transport import PhyTransport, TransportError
from phy_remote.client.views.waveform_view import WaveformWidget
from phy_remote.client.views.isi_view import ISIWidget
from phy_remote.client.views.amplitude_view import AmplitudeWidget
from phy_remote.client.views.trace_view import TraceWidget
from phy_remote.client.views.similarity_view import SimilarityWidget
from phy_remote.client.views.template_features_view import TemplateFeaturesWidget
from phy_remote.client.views.feature_cloud_view import FeatureViewWidget
from phy_remote.client.views.probe_view import ProbeWidget
from phy_remote.client.views.raster_view import RasterWidget
from phy_remote.client.views.correlogram_view import CorrelogramWidget
from phy_remote.client.views.console_view import ConsoleWidget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label colours  (match phy conventions)
# ---------------------------------------------------------------------------

LABEL_COLORS: dict[str, QColor] = {
    "good":      QColor(100, 200, 100),   # green
    "mua":       QColor(220, 160,  60),   # orange
    "noise":     QColor(160,  60,  60),   # red
    "unsorted":  QColor(180, 180, 180),   # grey
}
TEXT_COLOR = QColor(230, 230, 230)


# ---------------------------------------------------------------------------
# Cluster table model
# ---------------------------------------------------------------------------

_COLUMNS = ["ID", "Label", "Spikes", "Ampl (µV)", "FR (Hz)"]
_COL_ID, _COL_LABEL, _COL_SPIKES, _COL_AMPL, _COL_FR = range(5)


class ClusterTableModel(QAbstractTableModel):
    """Qt model backed by a list of cluster-info dicts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []

    def load(self, clusters: list[dict]) -> None:
        self.beginResetModel()
        self._rows = clusters
        self.endResetModel()

    def update_label(self, cluster_id: int, label: str) -> None:
        for i, row in enumerate(self._rows):
            if row["id"] == cluster_id:
                row["label"] = label
                idx = self.index(i, _COL_LABEL)
                self.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DisplayRole,
                                                  Qt.ItemDataRole.BackgroundRole])
                return

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(_COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return _COLUMNS[section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == _COL_ID:     return str(row["id"])
            if col == _COL_LABEL:  return row["label"]
            if col == _COL_SPIKES: return str(row["n_spikes"])
            if col == _COL_AMPL:   return f'{row["amplitude"]:.1f}'
            if col == _COL_FR:     return f'{row["fr"]:.2f}'

        if role == Qt.ItemDataRole.BackgroundRole and col == _COL_LABEL:
            return LABEL_COLORS.get(row["label"], LABEL_COLORS["unsorted"])

        if role == Qt.ItemDataRole.ForegroundRole and col == _COL_LABEL:
            return TEXT_COLOR

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col == _COL_LABEL:
                return Qt.AlignmentFlag.AlignCenter
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        if role == Qt.ItemDataRole.UserRole:
            return row  # entire dict for sorting

        return None

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        self.layoutAboutToBeChanged.emit()
        reverse = (order == Qt.SortOrder.DescendingOrder)
        key = {
            _COL_ID:     lambda r: r["id"],
            _COL_LABEL:  lambda r: r["label"],
            _COL_SPIKES: lambda r: r["n_spikes"],
            _COL_AMPL:   lambda r: r["amplitude"],
            _COL_FR:     lambda r: r["fr"],
        }[column]
        self._rows.sort(key=key, reverse=reverse)
        self.layoutChanged.emit()

    def cluster_id_at_row(self, row: int) -> int:
        return self._rows[row]["id"]


# ---------------------------------------------------------------------------
# Cluster table dock
# ---------------------------------------------------------------------------

class _ClusterTableDock(QDockWidget):
    clusters_selected = pyqtSignal(list)   # list[int]

    def __init__(self, parent=None):
        super().__init__("Clusters", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self._model = ClusterTableModel()
        self._view = QTableView()
        self._view.setModel(self._model)
        self._view.setSortingEnabled(True)
        self._view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._view.horizontalHeader().setSectionResizeMode(
            _COL_LABEL, QHeaderView.ResizeMode.Stretch
        )
        self._view.horizontalHeader().setSectionResizeMode(
            _COL_ID, QHeaderView.ResizeMode.ResizeToContents
        )
        self._view.setAlternatingRowColors(True)
        self._view.verticalHeader().setVisible(False)
        self._view.setShowGrid(False)
        mono = QFont("Menlo", 11)
        self._view.setFont(mono)

        self._view.selectionModel().selectionChanged.connect(self._on_selection_changed)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self.setWidget(container)

    def populate(self, clusters: list[dict]) -> None:
        self._model.load(clusters)
        self._view.sortByColumn(_COL_ID, Qt.SortOrder.AscendingOrder)

    def update_label(self, cluster_id: int, label: str) -> None:
        self._model.update_label(cluster_id, label)

    def selected_cluster_ids(self) -> list[int]:
        rows = {idx.row() for idx in self._view.selectionModel().selectedRows()}
        return [self._model.cluster_id_at_row(r) for r in sorted(rows)]

    def adjacent_cluster_ids(self, cluster_ids: list[int], n: int = 3) -> list[int]:
        """Return up to n cluster ids before and n after the current selection."""
        rows = sorted({idx.row() for idx in self._view.selectionModel().selectedRows()})
        if not rows:
            return []
        total = self._model.rowCount()
        result = []
        for delta in range(1, n + 1):
            if rows[0] - delta >= 0:
                result.append(self._model.cluster_id_at_row(rows[0] - delta))
            if rows[-1] + delta < total:
                result.append(self._model.cluster_id_at_row(rows[-1] + delta))
        return result

    def move_selection(self, delta: int) -> None:
        """Move selection by delta rows (negative = up, positive = down)."""
        current = self._view.selectionModel().currentIndex()
        if not current.isValid():
            new_row = 0 if delta > 0 else self._model.rowCount() - 1
        else:
            new_row = max(0, min(self._model.rowCount() - 1, current.row() + delta))
        new_index = self._model.index(new_row, 0)
        self._view.selectionModel().setCurrentIndex(
            new_index,
            QItemSelectionModel.SelectionFlag.ClearAndSelect |
            QItemSelectionModel.SelectionFlag.Rows,
        )
        self._view.scrollTo(new_index)

    def _on_selection_changed(self, *_) -> None:
        ids = self.selected_cluster_ids()
        if ids:
            self.clusters_selected.emit(ids)


# ---------------------------------------------------------------------------
# LRU cache
# ---------------------------------------------------------------------------

class _LRUCache:
    """Simple LRU cache keyed by cluster id."""

    def __init__(self, maxsize: int = 20):
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key):
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key) -> bool:
        return key in self._data


# ---------------------------------------------------------------------------
# Async fetch workers
# ---------------------------------------------------------------------------

class _FetchWorker(QObject):
    """
    Fetches waveforms in two stages using a plain Python thread.
    PyQt6 queues cross-thread signal emissions to the main event loop
    automatically, so no QThread / moveToThread needed.
    """
    template_ready   = pyqtSignal(dict)   # {cid: (n_samples, n_channels)}
    features_ready   = pyqtSignal(dict)   # {cid: features_array}
    template_features_ready = pyqtSignal(dict)  # {cid: template_features_array}
    spike_data_ready = pyqtSignal(dict)   # {cid: (n_spikes, 2) [time, amp]}
    finished         = pyqtSignal(dict)   # {cid: (n_spikes, n_samples, n_channels)}
    error            = pyqtSignal(str)

    def __init__(
        self,
        transport: PhyTransport,
        cluster_ids: list[int],
        n_spikes: int,
        skip_templates: bool = False,
    ):
        super().__init__()
        self.transport = transport
        self.cluster_ids = cluster_ids
        self.n_spikes = n_spikes
        self.skip_templates = skip_templates
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):
        if not self.skip_templates:
            templates = {}
            for cid in self.cluster_ids:
                if self._cancelled:
                    return
                try:
                    hdr, tmpl = self.transport.get_templates(cid)
                    ch_ids = hdr.get("channel_ids", list(range(tmpl.shape[-1])))
                    templates[cid] = (tmpl, ch_ids)
                except Exception as exc:
                    self.error.emit(f"Template fetch failed for cluster {cid}: {exc}")
                    return
            if not self._cancelled:
                self.template_ready.emit(templates)

        # Stage 1.5: features
        features = {}
        for cid in self.cluster_ids:
            if self._cancelled:
                return
            try:
                _, f = self.transport.get_features(cid)
                features[cid] = f
            except Exception as exc:
                logger.debug("Feature fetch skipped for cluster %d: %s", cid, exc)
        if not self._cancelled and features:
            self.features_ready.emit(features)

        # Stage 1.6: template features (if available)
        template_features = {}
        for cid in self.cluster_ids:
            if self._cancelled:
                return
            try:
                _, tf = self.transport.get_template_features(cid)
                template_features[cid] = tf
            except Exception as exc:
                logger.debug("Template features unavailable for cluster %d: %s", cid, exc)
        if not self._cancelled and template_features:
            self.template_features_ready.emit(template_features)

        # Stage 2: spike times + amplitudes (fast — all in RAM)
        spike_data = {}
        for cid in self.cluster_ids:
            if self._cancelled:
                return
            try:
                spike_data[cid] = self.transport.get_spike_data(cid)
            except Exception as exc:
                self.error.emit(f"Spike data fetch failed for cluster {cid}: {exc}")
                return
        if not self._cancelled:
            self.spike_data_ready.emit(spike_data)

        # Stage 3: real waveforms (slow — disk read)
        waveforms = {}
        for cid in self.cluster_ids:
            if self._cancelled:
                return
            try:
                hdr, w = self.transport.get_waveforms(cid, self.n_spikes)
                ch_ids = hdr.get("channel_ids", list(range(w.shape[-1])))
                waveforms[cid] = (w, ch_ids)
            except Exception as exc:
                self.error.emit(f"Waveform fetch failed for cluster {cid}: {exc}")
                return
        if not self._cancelled:
            self.finished.emit(waveforms)


class _PrefetchWorker(QObject):
    """
    Background worker that silently pre-warms the template cache for
    clusters adjacent to the current selection.  Creates its own
    transport so it never shares a ZMQ socket with _FetchWorker.
    """
    template_ready = pyqtSignal(int, object)   # (cluster_id, ndarray)
    finished       = pyqtSignal()

    def __init__(self, host: str, port: int, cluster_ids: list[int]):
        super().__init__()
        self._host = host
        self._port = port
        self.cluster_ids = list(cluster_ids)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):
        try:
            transport = PhyTransport(host=self._host, port=self._port)
        except Exception as exc:
            logger.debug("Prefetch transport failed: %s", exc)
            self.finished.emit()
            return
        try:
            for cid in self.cluster_ids:
                if self._cancelled:
                    break
                try:
                    _, tmpl = transport.get_templates(cid)
                    if not self._cancelled:
                        self.template_ready.emit(cid, tmpl)
                except Exception as exc:
                    logger.debug("Prefetch skipped cluster %d: %s", cid, exc)
        finally:
            transport.close()
            self.finished.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Parameters
    ----------
    transport : PhyTransport
    n_spikes  : int   max spikes to fetch per cluster for display
    """

    def __init__(self, transport: PhyTransport, n_spikes: int = 50):
        super().__init__()
        self.transport = transport
        self.n_spikes = n_spikes
        self._fetch_worker:    _FetchWorker    | None = None
        self._prefetch_worker: _PrefetchWorker | None = None

        # Caches: templates are cheap (fast to fetch, ~KB each),
        # waveforms are expensive (slow to fetch, ~MB each).
        self._template_cache = _LRUCache(maxsize=30)
        self._waveform_cache = _LRUCache(maxsize=10)
        self._current_cluster_ids: list[int] = []
        # home channel for each cluster (populated from template headers)
        self._home_channels: dict[int, int] = {}
        self._template_channels: dict[int, list[int]] = {}
        # spike times for each cluster (populated from spike_data)
        self._spike_times: dict[int, np.ndarray] = {}
        self._features: dict[int, np.ndarray] = {}
        self._template_features: dict[int, np.ndarray] = {}

        self.setWindowTitle("phy-remote")
        self.resize(1700, 950)
        self.setDockNestingEnabled(True)

        # Central widget — waveform view
        self._waveform_view = WaveformWidget(max_spikes=n_spikes, parent=self)
        self.setCentralWidget(self._waveform_view)

        # Left dock — cluster table
        self._cluster_dock = _ClusterTableDock(self)
        self._cluster_dock.clusters_selected.connect(self._on_clusters_selected)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._cluster_dock)
        self._cluster_dock.setMinimumWidth(260)

        # Bottom docks — ISI and Amplitude
        self._isi_dock = QDockWidget("ISI", self)
        self._isi_view = ISIWidget(self)
        self._isi_dock.setWidget(self._isi_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._isi_dock)

        self._amp_dock = QDockWidget("Amplitude", self)
        self._amp_view = AmplitudeWidget(self)
        self._amp_dock.setWidget(self._amp_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._amp_dock)

        # Right dock — trace view
        self._trace_dock = QDockWidget("Traces", self)
        self._trace_view = TraceWidget(
            host=transport.host,
            port=transport.port,
            parent=self,
        )
        self._trace_dock.setWidget(self._trace_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._trace_dock)

        self._similarity_dock = QDockWidget("Similarity", self)
        self._similarity_view = SimilarityWidget(self)
        self._similarity_dock.setWidget(self._similarity_view)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._similarity_dock)

        # Additional phy-style panes (placeholder widgets until implemented)
        self._template_feat_dock = QDockWidget("Template features", self)
        self._template_feat_view = TemplateFeaturesWidget(self)
        self._template_feat_dock.setWidget(self._template_feat_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._template_feat_dock)

        self._feature_cloud_dock = QDockWidget("Feature view", self)
        self._feature_cloud_view = FeatureViewWidget(self)
        self._feature_cloud_dock.setWidget(self._feature_cloud_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._feature_cloud_dock)

        self._probe_dock = QDockWidget("Probe", self)
        self._probe_view = ProbeWidget(self)
        self._probe_dock.setWidget(self._probe_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._probe_dock)

        self._raster_dock = QDockWidget("Raster", self)
        self._raster_view = RasterWidget(self)
        self._raster_dock.setWidget(self._raster_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._raster_dock)

        self._correlogram_dock = QDockWidget("Correlogram", self)
        self._correlogram_view = CorrelogramWidget(self)
        self._correlogram_dock.setWidget(self._correlogram_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._correlogram_dock)

        self._console_dock = QDockWidget("Console", self)
        self._console_view = ConsoleWidget(self)
        self._console_dock.setWidget(self._console_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._console_dock)

        # Build a phy-like layout: left cluster table, middle waveform + top feature,
        # large right traces, and several small bottom analysis panes.
        self.splitDockWidget(self._cluster_dock, self._template_feat_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(self._template_feat_dock, self._trace_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(self._cluster_dock, self._similarity_dock, Qt.Orientation.Vertical)
        self.splitDockWidget(self._template_feat_dock, self._feature_cloud_dock, Qt.Orientation.Vertical)

        self.splitDockWidget(self._cluster_dock, self._isi_dock, Qt.Orientation.Vertical)
        self.splitDockWidget(self._isi_dock, self._amp_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(self._amp_dock, self._probe_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(self._probe_dock, self._raster_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(self._raster_dock, self._correlogram_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(self._correlogram_dock, self._console_dock, Qt.Orientation.Horizontal)

        self.resizeDocks(
            [self._cluster_dock, self._template_feat_dock, self._trace_dock],
            [260, 680, 720],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self._cluster_dock, self._similarity_dock],
            [620, 280],
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self._template_feat_dock, self._feature_cloud_dock],
            [250, 700],
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [
                self._isi_dock, self._amp_dock, self._probe_dock,
                self._raster_dock, self._correlogram_dock, self._console_dock,
            ],
            [220, 220, 220, 220, 220, 220],
            Qt.Orientation.Horizontal,
        )

        # Status bar + label buttons
        self._status_label = QLabel("Connecting…")
        status_bar = QStatusBar()
        status_bar.addWidget(self._status_label, stretch=1)
        status_bar.addPermanentWidget(self._make_label_buttons())
        self.setStatusBar(status_bar)

        # Menu
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self._cluster_dock.toggleViewAction())
        view_menu.addAction(self._isi_dock.toggleViewAction())
        view_menu.addAction(self._amp_dock.toggleViewAction())
        view_menu.addAction(self._trace_dock.toggleViewAction())
        view_menu.addAction(self._similarity_dock.toggleViewAction())
        view_menu.addAction(self._template_feat_dock.toggleViewAction())
        view_menu.addAction(self._feature_cloud_dock.toggleViewAction())
        view_menu.addAction(self._probe_dock.toggleViewAction())
        view_menu.addAction(self._raster_dock.toggleViewAction())
        view_menu.addAction(self._correlogram_dock.toggleViewAction())
        view_menu.addAction(self._console_dock.toggleViewAction())

        # Keyboard shortcuts — labelling
        self._add_label_shortcut("g", "good")
        self._add_label_shortcut("m", "mua")
        self._add_label_shortcut("n", "noise")
        self._add_label_shortcut("u", "unsorted")

        # Keyboard shortcuts — navigation (j = down, k = up, vim-style)
        self._add_nav_shortcut("j", +1)
        self._add_nav_shortcut("k", -1)

        # Load channel positions then cluster table
        self._load_channel_positions()
        self._load_cluster_info()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _make_label_buttons(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)
        for label, shortcut in [("Good [g]", "good"), ("MUA [m]", "mua"),
                                  ("Noise [n]", "noise"), ("Unsorted [u]", "unsorted")]:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            color = LABEL_COLORS[shortcut].name()
            btn.setStyleSheet(
                f"QPushButton {{ background: {color}; color: white; "
                f"border: none; border-radius: 3px; padding: 0 6px; }}"
                f"QPushButton:hover {{ opacity: 0.8; }}"
            )
            btn.clicked.connect(lambda checked, lbl=shortcut: self._apply_label(lbl))
            layout.addWidget(btn)
        return w

    def _add_label_shortcut(self, key: str, label: str) -> None:
        sc = QShortcut(QKeySequence(key), self)
        sc.activated.connect(lambda lbl=label: self._apply_label(lbl))

    def _add_nav_shortcut(self, key: str, delta: int) -> None:
        sc = QShortcut(QKeySequence(key), self)
        sc.activated.connect(lambda d=delta: self._navigate_cluster(d))

    def _navigate_cluster(self, delta: int) -> None:
        """Move the cluster table selection by delta rows."""
        self._cluster_dock.move_selection(delta)

    @staticmethod
    def _fmt_ids(ids: list[int]) -> str:
        s = ", ".join(str(c) for c in ids[:4])
        if len(ids) > 4:
            s += f" (+{len(ids) - 4})"
        return s

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _load_channel_positions(self) -> None:
        try:
            positions = self.transport.get_channel_positions()
            self._waveform_view.set_channel_positions(positions)
            self._trace_view.set_channel_positions(positions)
            self._probe_view.set_channel_positions(positions)
            logger.info("Loaded %d channel positions", len(positions))
        except Exception as exc:
            logger.warning("Could not load channel positions: %s", exc)

    def _load_cluster_info(self) -> None:
        self._set_status("Loading clusters…")
        try:
            clusters = self.transport.get_cluster_info()
            self._cluster_dock.populate(clusters)
            self._similarity_view.set_cluster_info(clusters)
            n_good = sum(1 for c in clusters if c["label"] == "good")
            self._set_status(
                f"{len(clusters)} clusters  —  {n_good} good"
            )
            self._console_view.log(f"Loaded {len(clusters)} clusters")
        except Exception as exc:
            logger.error("Could not load cluster info: %s", exc)
            self._set_status(f"Error: {exc}")

    # ------------------------------------------------------------------
    # Cluster selection → async waveform fetch with cache + prefetch
    # ------------------------------------------------------------------

    def _on_clusters_selected(self, cluster_ids: list[int]) -> None:
        self._current_cluster_ids = cluster_ids
        primary_cluster_id = cluster_ids[-1] if cluster_ids else None
        # Reset feature-based panes; they will repopulate from model-backed RPCs.
        self._feature_cloud_view.set_feature_data({})
        self._template_feat_view.set_feature_data({})

        # Update similarity table from the last selected cluster (Phy behavior).
        if primary_cluster_id is not None:
            try:
                rows = self.transport.get_similar_clusters(primary_cluster_id, limit=100)
                self._similarity_view.set_similarity_rows(primary_cluster_id, rows)
            except Exception as exc:
                logger.warning("Similarity fetch failed: %s", exc)
                self._similarity_view.set_similarity_rows(primary_cluster_id, [])
            self._console_view.log(f"Selected clusters: {self._fmt_ids(cluster_ids)}")

        # Cancel any in-flight workers (they check _cancelled before emitting)
        if self._prefetch_worker is not None:
            self._prefetch_worker.cancel()
        if self._fetch_worker is not None:
            self._fetch_worker.cancel()

        # --- Check template cache ---
        cached_templates = {
            cid: self._template_cache.get(cid)
            for cid in cluster_ids
            if cid in self._template_cache
        }

        if len(cached_templates) == len(cluster_ids):
            # All templates in cache → show immediately
            self._waveform_view.set_waveforms(cached_templates)

            # Check if real waveforms are also cached
            cached_waveforms = {
                cid: self._waveform_cache.get(cid)
                for cid in cluster_ids
                if cid in self._waveform_cache
            }
            if len(cached_waveforms) == len(cluster_ids):
                self._set_status(f"Cluster(s) {self._fmt_ids(cluster_ids)}  (cached)")
                self._start_prefetch(cluster_ids)
                return

            # Show templates now; fetch only real waveforms in background
            self._set_status(
                f"Cluster(s) {self._fmt_ids(cluster_ids)}  — loading waveforms…"
            )
            self._start_fetch(cluster_ids, skip_templates=True)
            return

        # Not cached → full two-stage fetch
        self._set_status(f"Fetching: {self._fmt_ids(cluster_ids)}…")
        self._start_fetch(cluster_ids, skip_templates=False)

    def _start_fetch(self, cluster_ids: list[int], skip_templates: bool) -> None:
        worker = _FetchWorker(
            self.transport, cluster_ids, self.n_spikes,
            skip_templates=skip_templates,
        )
        if not skip_templates:
            worker.template_ready.connect(self._on_templates_ready)
        worker.features_ready.connect(self._on_features_ready)
        worker.template_features_ready.connect(self._on_template_features_ready)
        worker.spike_data_ready.connect(self._on_spike_data_ready)
        worker.finished.connect(self._on_waveforms_ready)
        worker.error.connect(self._on_fetch_error)
        self._fetch_worker = worker
        threading.Thread(target=worker.run, daemon=True).start()

    def _start_prefetch(self, current_ids: list[int]) -> None:
        """Silently warm the template cache for adjacent clusters."""
        adjacent = self._cluster_dock.adjacent_cluster_ids(current_ids, n=3)
        to_fetch = [cid for cid in adjacent if cid not in self._template_cache]
        if not to_fetch:
            return

        worker = _PrefetchWorker(self.transport.host, self.transport.port, to_fetch)
        worker.template_ready.connect(self._on_prefetch_template)
        self._prefetch_worker = worker
        threading.Thread(target=worker.run, daemon=True).start()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    @pyqtSlot(dict)
    def _on_templates_ready(self, templates: dict) -> None:
        # templates: {cid: (array, ch_ids)}
        logger.info("Templates ready for clusters %s", list(templates.keys()))
        for cid, (arr, ch_ids) in templates.items():
            self._template_cache.put(cid, (arr, ch_ids))
            # ch_ids[0] is the best (home) channel
            if ch_ids:
                self._home_channels[cid] = int(ch_ids[0])
                self._template_channels[cid] = [int(c) for c in ch_ids]
        if set(templates.keys()) == set(self._current_cluster_ids):
            self._set_status(
                f"Cluster(s) {self._fmt_ids(list(templates.keys()))}  — loading waveforms…"
            )
            self._waveform_view.set_waveforms(templates)
        else:
            logger.info("Templates discarded (selection changed to %s)", self._current_cluster_ids)

    @pyqtSlot(dict)
    def _on_features_ready(self, features: dict) -> None:
        for cid, arr in features.items():
            self._features[cid] = arr
        if set(features.keys()) == set(self._current_cluster_ids):
            self._feature_cloud_view.set_feature_data(features)

    @pyqtSlot(dict)
    def _on_template_features_ready(self, template_features: dict) -> None:
        for cid, arr in template_features.items():
            self._template_features[cid] = arr
        if set(template_features.keys()) == set(self._current_cluster_ids):
            self._template_feat_view.set_feature_data(template_features)

    # Tick colours (index 0 = primary/selected cluster)
    _TICK_COLORS = [
        (0.92, 0.15, 0.15, 1.0),
        (0.15, 0.47, 0.90, 1.0),
        (0.10, 0.72, 0.42, 1.0),
        (0.92, 0.58, 0.08, 1.0),
        (0.65, 0.15, 0.90, 1.0),
    ]

    @pyqtSlot(dict)
    def _on_spike_data_ready(self, spike_data: dict) -> None:
        logger.info("Spike data ready for clusters %s", list(spike_data.keys()))
        # Cache spike times
        for cid, data in spike_data.items():
            self._spike_times[cid] = data[:, 0].astype(np.float64)

        if set(spike_data.keys()) == set(self._current_cluster_ids):
            self._isi_view.set_spike_data(spike_data)
            self._amp_view.set_spike_data(spike_data)
            self._raster_view.set_spike_times(self._spike_times_for_current())
            self._correlogram_view.set_spike_times(self._spike_times_for_current())
            self._update_trace_clusters(list(spike_data.keys()))

    @pyqtSlot(dict)
    def _on_waveforms_ready(self, waveforms: dict) -> None:
        # waveforms: {cid: (array, ch_ids)}
        logger.info("Waveforms ready for clusters %s", list(waveforms.keys()))
        for cid, val in waveforms.items():
            self._waveform_cache.put(cid, val)
        if set(waveforms.keys()) == set(self._current_cluster_ids):
            self._set_status(f"Cluster(s) {self._fmt_ids(list(waveforms.keys()))}")
            self._waveform_view.set_waveforms(waveforms)
            # Update trace overlays now that we have home channels confirmed
            self._update_trace_clusters(list(waveforms.keys()))
            self._start_prefetch(self._current_cluster_ids)
        else:
            logger.info("Waveforms discarded (selection changed to %s)", self._current_cluster_ids)

    def _update_trace_clusters(self, cluster_ids: list[int]) -> None:
        """Build cluster_data dict and push it to the trace view."""
        cluster_data = {}
        for idx, cid in enumerate(cluster_ids):
            times   = self._spike_times.get(cid)
            home_ch = self._home_channels.get(cid)
            if times is None or home_ch is None:
                continue
            color = self._TICK_COLORS[idx % len(self._TICK_COLORS)]
            cluster_data[cid] = (times, home_ch, color)

        if not cluster_data:
            return

        self._trace_view.set_cluster_data(cluster_data)
        selected_channels = []
        for cid, (_times, home_ch, color) in cluster_data.items():
            channels = self._template_channels.get(cid, [home_ch])[:8]
            selected_channels.append((channels, color))
        self._probe_view.set_selected_channels(selected_channels)

        # Center traces on first selected spike when available.
        first_times = cluster_data[cluster_ids[0]][0]
        if len(first_times):
            self._trace_view.go_to_time(float(first_times[0]))
        else:
            self._trace_view.go_to_time(0.0)

    @pyqtSlot(str)
    def _on_fetch_error(self, msg: str) -> None:
        logger.error("Fetch error: %s", msg)
        self._set_status(f"Error: {msg}")

    @pyqtSlot(int, object)
    def _on_prefetch_template(self, cluster_id: int, template) -> None:
        self._template_cache.put(cluster_id, template)
        logger.debug("Prefetched template for cluster %d", cluster_id)

    # ------------------------------------------------------------------
    # Labelling
    # ------------------------------------------------------------------

    def _apply_label(self, label: str) -> None:
        cluster_ids = self._cluster_dock.selected_cluster_ids()
        if not cluster_ids:
            self._set_status("No clusters selected")
            return
        try:
            self.transport.label_cluster(cluster_ids, label)
        except TransportError as exc:
            self._set_status(f"Label error: {exc}")
            return
        for cid in cluster_ids:
            self._cluster_dock.update_label(cid, label)
        clu_str = ", ".join(str(c) for c in cluster_ids)
        self._set_status(f"Labelled {clu_str} → {label}")
        logger.info("Labelled cluster(s) %s as %r", cluster_ids, label)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _spike_times_for_current(self) -> dict[int, np.ndarray]:
        out = {}
        for cid in self._current_cluster_ids:
            if cid in self._spike_times:
                out[cid] = self._spike_times[cid]
        return out

    def _features_for_current(self) -> dict[int, np.ndarray]:
        out = {}
        for cid in self._current_cluster_ids:
            if cid in self._features:
                out[cid] = self._features[cid]
        return out

    def closeEvent(self, event) -> None:
        if self._fetch_worker is not None:
            self._fetch_worker.cancel()
        if self._prefetch_worker is not None:
            self._prefetch_worker.cancel()
        super().closeEvent(event)
