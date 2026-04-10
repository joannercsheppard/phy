"""
ZMQ REQ transport — runs on the MacBook client.

Typical setup
-------------
    # On cluster:  already running PhyServer on port 5555 (loopback)
    # On MacBook:  ssh -L 5555:localhost:5555 user@cluster

    from phy_remote.client.transport import PhyTransport
    t = PhyTransport(port=5555)
    header, waveforms = t.get_waveforms(cluster_id=42)
    t.close()

    # Or use as a context manager:
    with PhyTransport() as t:
        header, waveforms = t.get_waveforms(cluster_id=42)
"""

import logging

import numpy as np
import zmq

from phy_remote.shared.protocol import (
    CMD_PING,
    CMD_GET_WAVEFORMS,
    CMD_GET_SPIKE_TIMES,
    CMD_GET_FEATURES,
    CMD_GET_TEMPLATES,
    CMD_GET_CLUSTER_IDS,
    CMD_GET_CLUSTER_INFO,
    CMD_LABEL_CLUSTER,
    CMD_GET_SPIKE_DATA,
    encode_request,
    decode_response,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS = 60_000  # 60 s — on-the-fly waveform extraction can be slow


class TransportError(RuntimeError):
    """Raised when the server returns an error response."""


class PhyTransport:
    """
    Thin ZMQ REQ wrapper around the phy-remote protocol.

    Parameters
    ----------
    host : str
        Hostname / IP of the server.  When using an SSH tunnel, keep this
        as "127.0.0.1" (the local tunnel endpoint).
    port : int
        Port number (must match ``PhyServer`` / SSH tunnel).
    timeout_ms : int
        Socket receive timeout in milliseconds.  Raises ``zmq.Again`` on
        expiry so the UI can display a meaningful error instead of hanging.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ):
        self.host = host
        self.port = port
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        addr = f"tcp://{host}:{port}"
        self._sock.connect(addr)
        logger.info("PhyTransport connected to %s", addr)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self._sock.close(linger=0)
        self._ctx.term()

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the server responds to a ping."""
        header, _ = self._call(CMD_PING)
        return header.get("pong", False)

    def get_waveforms(
        self, cluster_id: int, n_spikes: int = 100
    ) -> tuple[dict, np.ndarray]:
        """
        Fetch waveforms for *cluster_id*.

        Returns
        -------
        header : dict
            Response metadata (cluster_id, n_spikes, dtype, shape).
        waveforms : np.ndarray, shape (n_spikes, n_samples, n_channels), float32
        """
        header, array = self._call(
            CMD_GET_WAVEFORMS, cluster_id=cluster_id, n_spikes=n_spikes
        )
        if array is None:
            raise TransportError("server sent no array for get_waveforms")
        return header, array

    def get_spike_times(self, cluster_id: int) -> tuple[dict, np.ndarray]:
        """
        Fetch spike times (seconds) for *cluster_id*.

        Returns
        -------
        header : dict
        times : np.ndarray, shape (n_spikes,), float64
        """
        header, array = self._call(CMD_GET_SPIKE_TIMES, cluster_id=cluster_id)
        if array is None:
            raise TransportError("server sent no array for get_spike_times")
        return header, array

    def get_features(self, cluster_id: int) -> tuple[dict, np.ndarray]:
        """
        Fetch PC features for *cluster_id*.

        Returns
        -------
        header : dict
        features : np.ndarray, float32
        """
        header, array = self._call(CMD_GET_FEATURES, cluster_id=cluster_id)
        if array is None:
            raise TransportError("server sent no array for get_features")
        return header, array

    def get_templates(self, cluster_id: int) -> tuple[dict, np.ndarray]:
        """
        Fetch the mean template waveform for *cluster_id*.

        Returns
        -------
        header : dict
            Includes ``channel_ids`` list.
        template : np.ndarray, shape (n_samples, n_channels), float32
        """
        header, array = self._call(CMD_GET_TEMPLATES, cluster_id=cluster_id)
        if array is None:
            raise TransportError("server sent no array for get_templates")
        return header, array

    def get_cluster_ids(self) -> np.ndarray:
        """Return all cluster ids as a 1-D int32 array."""
        _, array = self._call(CMD_GET_CLUSTER_IDS)
        if array is None:
            raise TransportError("server sent no array for get_cluster_ids")
        return array

    def get_cluster_info(self) -> list[dict]:
        """
        Return summary info for all clusters.

        Returns
        -------
        list of dicts, each with keys:
            id, label, n_spikes, amplitude, fr
        """
        header, _ = self._call(CMD_GET_CLUSTER_INFO)
        return header["clusters"]

    def get_spike_data(self, cluster_id: int) -> np.ndarray:
        """
        Fetch per-spike times and amplitudes for *cluster_id*.

        Returns
        -------
        data : np.ndarray, shape (n_spikes, 2), float64
            column 0 = spike time (seconds)
            column 1 = spike amplitude
        """
        _, array = self._call(CMD_GET_SPIKE_DATA, cluster_id=cluster_id)
        if array is None:
            raise TransportError("server sent no array for get_spike_data")
        return array

    def label_cluster(
        self, cluster_ids: "int | list[int]", label: str
    ) -> None:
        """
        Set the label for one or more clusters and save to disk on the server.

        Parameters
        ----------
        cluster_ids : int or list[int]
        label : "good" | "mua" | "noise" | "unsorted"
        """
        if isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]
        self._call(CMD_LABEL_CLUSTER, cluster_ids=cluster_ids, label=label)

    # ------------------------------------------------------------------
    # Low-level send/recv
    # ------------------------------------------------------------------

    def _call(self, cmd: str, **kwargs) -> tuple[dict, "np.ndarray | None"]:
        frames = encode_request(cmd, **kwargs)
        self._sock.send_multipart(frames)
        reply = self._sock.recv_multipart()
        header, array = decode_response(reply)
        if header.get("status") == "error":
            raise TransportError(header.get("error", "unknown server error"))
        return header, array
