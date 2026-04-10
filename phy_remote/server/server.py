"""
ZMQ REP server — runs headless on the HPC cluster.

Usage
-----
    from phy_remote.server.server import PhyServer
    server = PhyServer(model=my_template_model, port=5555)
    server.serve_forever()

The server is intentionally synchronous (one request at a time) to keep the
code simple and because phy's model layer is not thread-safe.  If latency
becomes an issue, move to a ROUTER/DEALER pattern later.
"""

import json
import logging
import signal
import socket
import threading

import numpy as np
import zmq

from phy_remote.shared.protocol import (
    CMD_PING,
    CMD_GET_WAVEFORMS,
    CMD_GET_SPIKE_TIMES,
    CMD_GET_FEATURES,
    CMD_GET_TEMPLATES,
    CMD_GET_CLUSTER_IDS,
    decode_request,
    encode_response,
)

logger = logging.getLogger(__name__)


class PhyServer:
    """
    Minimal ZMQ REP server wrapping a phy TemplateModel.

    Parameters
    ----------
    model : TemplateModel or any object with the expected attributes
        The data source.  Pass *None* to start without a model (useful for
        testing the transport layer only — only CMD_PING will work).
    port : int
        TCP port to bind.  SSH-tunnel this to the client.
    host : str
        Interface to bind.  Defaults to loopback; change to "0.0.0.0" only
        on a trusted network.
    """

    def __init__(self, model=None, port: int = 5555, host: str = "127.0.0.1"):
        self.model = model
        self.port = port
        self.host = host
        self._running = False

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._sock.bind(addr)
        logger.info("PhyServer bound to %s", addr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        """
        Block and serve requests until stop() is called or a signal arrives.

        The socket is created, polled, and destroyed entirely within this
        method so it is always owned by a single thread (ZMQ requirement).
        """
        self._running = True
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

        import os
        hostname = socket.getfqdn()
        logger.info("PhyServer ready (pid %d, host %s)", os.getpid(), hostname)
        # Machine-parseable ready token (also emitted by __main__.py before
        # calling serve_forever, but repeated here for library use)
        print(f"PHY_REMOTE_READY host={hostname} port={self.port}", flush=True)

        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        try:
            while self._running:
                events = dict(poller.poll(timeout=100))  # 100 ms poll interval
                if self._sock not in events:
                    continue
                frames = self._sock.recv_multipart()
                self._sock.send_multipart(self._dispatch(frames))
        finally:
            self._sock.close(linger=0)
            logger.info("PhyServer loop exited")

    def stop(self) -> None:
        """Thread-safe: signal the serve_forever loop to exit."""
        self._running = False

    def close(self) -> None:
        """Stop the server and release the ZMQ context.

        Call *after* the thread running serve_forever has been joined so the
        socket is guaranteed closed before ctx.term().
        """
        self._running = False
        self._ctx.term()
        logger.info("PhyServer context terminated")

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, frames: list[bytes]) -> list[bytes]:
        try:
            req = decode_request(frames)
        except Exception as exc:
            return encode_response(status="error", error=f"bad request: {exc}")

        cmd = req.get("cmd", "")
        logger.debug("cmd=%s args=%s", cmd, {k: v for k, v in req.items() if k != "cmd"})

        handler = {
            CMD_PING: self._handle_ping,
            CMD_GET_WAVEFORMS: self._handle_get_waveforms,
            CMD_GET_SPIKE_TIMES: self._handle_get_spike_times,
            CMD_GET_FEATURES: self._handle_get_features,
            CMD_GET_TEMPLATES: self._handle_get_templates,
            CMD_GET_CLUSTER_IDS: self._handle_get_cluster_ids,
        }.get(cmd)

        if handler is None:
            return encode_response(status="error", error=f"unknown command: {cmd!r}")

        try:
            return handler(req)
        except Exception as exc:
            logger.exception("error handling %s", cmd)
            return encode_response(status="error", error=str(exc))

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _handle_ping(self, req: dict) -> list[bytes]:
        return encode_response(pong=True)

    def _handle_get_waveforms(self, req: dict) -> list[bytes]:
        """
        Return waveforms for a cluster.

        Request fields
        --------------
        cluster_id : int
        n_spikes   : int, optional  max spikes to return (default 100)
        """
        self._require_model()
        cluster_id = int(req["cluster_id"])
        n_spikes = int(req.get("n_spikes", 100))

        spike_ids = self.model.get_cluster_spikes(cluster_id)
        if len(spike_ids) > n_spikes:
            rng = np.random.default_rng(seed=cluster_id)
            spike_ids = rng.choice(spike_ids, size=n_spikes, replace=False)
            spike_ids.sort()

        # shape: (n_spikes, n_samples, n_channels)
        waveforms = self.model.get_waveforms(spike_ids)
        if waveforms is None:
            return encode_response(status="error", error="no waveforms available")

        arr = np.asarray(waveforms, dtype=np.float32)
        return encode_response(array=arr, cluster_id=cluster_id, n_spikes=len(spike_ids))

    def _handle_get_spike_times(self, req: dict) -> list[bytes]:
        """Return spike times (seconds) for a cluster."""
        self._require_model()
        cluster_id = int(req["cluster_id"])
        spike_ids = self.model.get_cluster_spikes(cluster_id)
        times = np.asarray(self.model.spike_times[spike_ids], dtype=np.float64)
        return encode_response(array=times, cluster_id=cluster_id)

    def _handle_get_features(self, req: dict) -> list[bytes]:
        """Return PC features for a cluster."""
        self._require_model()
        cluster_id = int(req["cluster_id"])
        spike_ids = self.model.get_cluster_spikes(cluster_id)
        features = self.model.get_features(spike_ids, None)
        if features is None:
            return encode_response(status="error", error="no features available")
        arr = np.asarray(features.data, dtype=np.float32)
        return encode_response(array=arr, cluster_id=cluster_id)

    def _handle_get_templates(self, req: dict) -> list[bytes]:
        """
        Return the mean template waveform for a cluster.

        Request fields
        --------------
        cluster_id : int

        Response array shape: (n_samples, n_channels), float32
        """
        self._require_model()
        cluster_id = int(req["cluster_id"])

        # TemplateModel exposes get_template() returning a Bunch with
        # .template (n_samples, n_channels) and .channel_ids
        template = self.model.get_template(cluster_id)
        if template is None:
            return encode_response(status="error", error="no template available")

        arr = np.asarray(template.template, dtype=np.float32)
        channel_ids = template.channel_ids.tolist()
        return encode_response(array=arr, cluster_id=cluster_id, channel_ids=channel_ids)

    def _handle_get_cluster_ids(self, req: dict) -> list[bytes]:
        """Return the list of all cluster ids as a 1-D int32 array."""
        self._require_model()
        cluster_ids = np.asarray(self.model.cluster_ids, dtype=np.int32)
        return encode_response(array=cluster_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("server started without a model")

    def _handle_signal(self, signum, frame) -> None:
        logger.info("received signal %d, stopping", signum)
        self._running = False  # poller loop checks this flag and exits cleanly
