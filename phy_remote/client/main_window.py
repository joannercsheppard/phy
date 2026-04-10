"""
MainWindow — PyQt6 main window for phy-remote.

Layout
------
+-----------------------------------------------+
|  Menu bar                                     |
+-------------------+---------------------------+
|  Cluster list     |  Waveform view (central)  |
|  (left dock)      |                           |
|                   |                           |
+-------------------+---------------------------+
|  Status bar                                   |
+-----------------------------------------------+

Cluster selection triggers an async ZMQ fetch so the UI never blocks.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout,
    QListWidget, QListWidgetItem, QStatusBar, QLabel,
    QApplication,
)

from phy_remote.client.transport import PhyTransport, TransportError
from phy_remote.client.views.waveform_view import WaveformWidget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async fetch worker
# ---------------------------------------------------------------------------

class _FetchWorker(QObject):
    """
    Runs in a QThread.  Fetches waveforms for a set of cluster ids via ZMQ
    and emits the result back to the main thread.
    """

    # Emitted on success: {cluster_id: waveform_array, ...}
    finished = pyqtSignal(dict)
    # Emitted on any error
    error = pyqtSignal(str)

    def __init__(
        self,
        transport: PhyTransport,
        cluster_ids: Sequence[int],
        n_spikes: int = 50,
    ) -> None:
        super().__init__()
        self.transport = transport
        self.cluster_ids = list(cluster_ids)
        self.n_spikes = n_spikes

    @pyqtSlot()
    def run(self) -> None:
        result: dict[int, np.ndarray] = {}
        for clu_id in self.cluster_ids:
            try:
                _, waveforms = self.transport.get_waveforms(clu_id, self.n_spikes)
                result[clu_id] = waveforms
            except TransportError as exc:
                self.error.emit(f"Cluster {clu_id}: {exc}")
                return
            except Exception as exc:
                self.error.emit(f"Unexpected error fetching cluster {clu_id}: {exc}")
                return
        self.finished.emit(result)


# ---------------------------------------------------------------------------
# Cluster list dock widget
# ---------------------------------------------------------------------------

class _ClusterListDock(QDockWidget):
    """Dock widget containing a list of cluster ids."""

    clusters_selected = pyqtSignal(list)  # list[int]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Clusters", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._list)
        self.setWidget(container)

    def populate(self, cluster_ids: np.ndarray) -> None:
        self._list.clear()
        for clu_id in cluster_ids:
            item = QListWidgetItem(str(int(clu_id)))
            item.setData(Qt.ItemDataRole.UserRole, int(clu_id))
            self._list.addItem(item)

    def _on_selection_changed(self) -> None:
        selected = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._list.selectedItems()
        ]
        if selected:
            self.clusters_selected.emit(selected)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Top-level window for phy-remote.

    Parameters
    ----------
    transport : PhyTransport
        Open ZMQ transport to the server.
    n_spikes : int
        Number of spikes to fetch per cluster for display.
    """

    def __init__(
        self,
        transport: PhyTransport,
        n_spikes: int = 50,
    ) -> None:
        super().__init__()
        self.transport = transport
        self.n_spikes = n_spikes

        self._fetch_thread: QThread | None = None
        self._fetch_worker: _FetchWorker | None = None

        self.setWindowTitle("phy-remote")
        self.resize(1200, 800)

        # Central widget — waveform view
        self._waveform_view = WaveformWidget(max_spikes=n_spikes, parent=self)
        self._waveform_view.channel_clicked.connect(self._on_channel_clicked)
        self.setCentralWidget(self._waveform_view)

        # Left dock — cluster list
        self._cluster_dock = _ClusterListDock(self)
        self._cluster_dock.clusters_selected.connect(self._on_clusters_selected)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._cluster_dock)

        # Status bar
        self._status_label = QLabel("Connecting…")
        status_bar = QStatusBar()
        status_bar.addWidget(self._status_label)
        self.setStatusBar(status_bar)

        # Menu bar
        self._build_menus()

        # Fetch cluster list
        self._load_cluster_list()

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _build_menus(self) -> None:
        menu = self.menuBar()

        view_menu = menu.addMenu("View")
        view_menu.addAction(
            self._cluster_dock.toggleViewAction()
        )

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _load_cluster_list(self) -> None:
        self._set_status("Fetching cluster list…")
        try:
            cluster_ids = self.transport.get_cluster_ids()
            self._cluster_dock.populate(cluster_ids)
            self._set_status(f"{len(cluster_ids)} clusters loaded")
        except Exception as exc:
            logger.error("Could not load cluster ids: %s", exc)
            self._set_status(f"Error: {exc}")

    # ------------------------------------------------------------------
    # Cluster selection → async fetch
    # ------------------------------------------------------------------

    def _on_clusters_selected(self, cluster_ids: list[int]) -> None:
        # Cancel any in-flight fetch
        if self._fetch_thread is not None and self._fetch_thread.isRunning():
            self._fetch_thread.quit()
            self._fetch_thread.wait(500)

        clu_str = ", ".join(str(c) for c in cluster_ids[:5])
        if len(cluster_ids) > 5:
            clu_str += f" … (+{len(cluster_ids) - 5} more)"
        self._set_status(f"Fetching waveforms for cluster(s) {clu_str}…")

        worker = _FetchWorker(self.transport, cluster_ids, self.n_spikes)
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_waveforms_ready)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_fetch_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)

        self._fetch_worker = worker
        self._fetch_thread = thread
        thread.start()

    @pyqtSlot(dict)
    def _on_waveforms_ready(self, waveforms_per_cluster: dict) -> None:
        n_total = sum(v.shape[0] for v in waveforms_per_cluster.values())
        n_clu = len(waveforms_per_cluster)
        self._set_status(
            f"Displaying {n_total} spikes across {n_clu} cluster(s)"
        )
        self._waveform_view.set_waveforms(waveforms_per_cluster)

    @pyqtSlot(str)
    def _on_fetch_error(self, msg: str) -> None:
        logger.error("Fetch error: %s", msg)
        self._set_status(f"Error: {msg}")

    # ------------------------------------------------------------------
    # Channel click
    # ------------------------------------------------------------------

    def _on_channel_clicked(self, channel_idx: int) -> None:
        logger.debug("Channel %d clicked", channel_idx)
        self._set_status(f"Channel {channel_idx} selected")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)
        logger.info(text)

    def closeEvent(self, event) -> None:
        # Clean up the fetch thread if still running
        if self._fetch_thread is not None and self._fetch_thread.isRunning():
            self._fetch_thread.quit()
            self._fetch_thread.wait(1000)
        super().closeEvent(event)
