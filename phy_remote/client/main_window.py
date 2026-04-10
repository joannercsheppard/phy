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
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from PyQt6.QtCore import (
    Qt, QThread, QObject, QSortFilterProxyModel,
    QAbstractTableModel, QModelIndex, pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import QColor, QKeySequence, QShortcut, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QTableView, QHeaderView, QStatusBar, QLabel, QPushButton,
    QAbstractItemView, QSizePolicy,
)

from phy_remote.client.transport import PhyTransport, TransportError
from phy_remote.client.views.waveform_view import WaveformWidget

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

    def _on_selection_changed(self, *_) -> None:
        ids = self.selected_cluster_ids()
        if ids:
            self.clusters_selected.emit(ids)


# ---------------------------------------------------------------------------
# Async fetch worker
# ---------------------------------------------------------------------------

class _FetchWorker(QObject):
    """
    Fetches waveforms for selected clusters in two stages:
      1. Templates (instant) — emitted via template_ready so the view
         can show something immediately.
      2. Real spike waveforms on template channels — emitted via finished
         once extraction completes.
    """
    template_ready = pyqtSignal(dict)   # {cluster_id: ndarray (n_samples, n_channels)}
    finished       = pyqtSignal(dict)   # {cluster_id: ndarray (n_spikes, n_samples, n_channels)}
    error          = pyqtSignal(str)

    def __init__(self, transport, cluster_ids, n_spikes):
        super().__init__()
        self.transport = transport
        self.cluster_ids = cluster_ids
        self.n_spikes = n_spikes

    @pyqtSlot()
    def run(self):
        # Stage 1: templates (fast)
        templates = {}
        for cid in self.cluster_ids:
            try:
                _, tmpl = self.transport.get_templates(cid)
                templates[cid] = tmpl
            except Exception as exc:
                self.error.emit(f"Template fetch failed for cluster {cid}: {exc}")
                return
        self.template_ready.emit(templates)

        # Stage 2: real waveforms on template channels (slower)
        waveforms = {}
        for cid in self.cluster_ids:
            try:
                _, w = self.transport.get_waveforms(cid, self.n_spikes)
                waveforms[cid] = w
            except Exception as exc:
                self.error.emit(f"Waveform fetch failed for cluster {cid}: {exc}")
                return
        self.finished.emit(waveforms)


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
        self._fetch_thread: QThread | None = None

        self.setWindowTitle("phy-remote")
        self.resize(1400, 900)

        # Central widget — waveform view
        self._waveform_view = WaveformWidget(max_spikes=n_spikes, parent=self)
        self.setCentralWidget(self._waveform_view)

        # Left dock — cluster table
        self._cluster_dock = _ClusterTableDock(self)
        self._cluster_dock.clusters_selected.connect(self._on_clusters_selected)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._cluster_dock)

        # Status bar + label buttons
        self._status_label = QLabel("Connecting…")
        status_bar = QStatusBar()
        status_bar.addWidget(self._status_label, stretch=1)
        status_bar.addPermanentWidget(self._make_label_buttons())
        self.setStatusBar(status_bar)

        # Menu
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self._cluster_dock.toggleViewAction())

        # Keyboard shortcuts for labelling
        self._add_label_shortcut("g", "good")
        self._add_label_shortcut("m", "mua")
        self._add_label_shortcut("n", "noise")
        self._add_label_shortcut("u", "unsorted")

        # Load cluster table
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

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _load_cluster_info(self) -> None:
        self._set_status("Loading clusters…")
        try:
            clusters = self.transport.get_cluster_info()
            self._cluster_dock.populate(clusters)
            n_good = sum(1 for c in clusters if c["label"] == "good")
            self._set_status(
                f"{len(clusters)} clusters  —  {n_good} good"
            )
        except Exception as exc:
            logger.error("Could not load cluster info: %s", exc)
            self._set_status(f"Error: {exc}")

    # ------------------------------------------------------------------
    # Cluster selection → async waveform fetch
    # ------------------------------------------------------------------

    def _on_clusters_selected(self, cluster_ids: list[int]) -> None:
        if self._fetch_thread and self._fetch_thread.isRunning():
            self._fetch_thread.quit()
            self._fetch_thread.wait(300)

        clu_str = ", ".join(str(c) for c in cluster_ids[:4])
        if len(cluster_ids) > 4:
            clu_str += f" (+{len(cluster_ids) - 4})"
        self._set_status(f"Fetching waveforms: {clu_str}…")

        worker = _FetchWorker(self.transport, cluster_ids, self.n_spikes)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.template_ready.connect(self._on_templates_ready)
        worker.finished.connect(self._on_waveforms_ready)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_fetch_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        self._fetch_thread = thread
        thread.start()

    @pyqtSlot(dict)
    def _on_waveforms_ready(self, templates_per_cluster: dict) -> None:
        ids = ", ".join(str(k) for k in templates_per_cluster)
        self._set_status(f"Templates: {ids}")
        self._waveform_view.set_waveforms(templates_per_cluster)

    @pyqtSlot(str)
    def _on_fetch_error(self, msg: str) -> None:
        logger.error("Fetch error: %s", msg)
        self._set_status(f"Error: {msg}")

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

    def closeEvent(self, event) -> None:
        if self._fetch_thread and self._fetch_thread.isRunning():
            self._fetch_thread.quit()
            self._fetch_thread.wait(1000)
        super().closeEvent(event)
