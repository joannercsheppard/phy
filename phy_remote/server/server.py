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
    CMD_GET_CLUSTER_INFO,
    CMD_LABEL_CLUSTER,
    CMD_GET_SPIKE_DATA,
    CMD_GET_CHANNEL_POSITIONS,
    CMD_GET_TRACES,
    CMD_GET_SPIKES_IN_WINDOW,
    CMD_GET_SIMILAR_CLUSTERS,
    CMD_GET_TEMPLATE_FEATURES,
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
        self.host = host
        self._running = False

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)

        # Try the requested port; if it's taken walk up until one is free.
        for candidate in range(port, port + 20):
            try:
                self._sock.bind(f"tcp://{host}:{candidate}")
                self.port = candidate
                break
            except zmq.ZMQError:
                continue
        else:
            raise OSError(f"No free port found in range {port}–{port + 19}")

        logger.info("PhyServer bound to tcp://%s:%d", host, self.port)

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
            CMD_GET_CLUSTER_INFO: self._handle_get_cluster_info,
            CMD_LABEL_CLUSTER: self._handle_label_cluster,
            CMD_GET_SPIKE_DATA: self._handle_get_spike_data,
            CMD_GET_CHANNEL_POSITIONS: self._handle_get_channel_positions,
            CMD_GET_TRACES: self._handle_get_traces,
            CMD_GET_SPIKES_IN_WINDOW: self._handle_get_spikes_in_window,
            CMD_GET_SIMILAR_CLUSTERS: self._handle_get_similar_clusters,
            CMD_GET_TEMPLATE_FEATURES: self._handle_get_template_features,
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
        Return waveforms for a cluster, restricted to template channels.

        Reads only the ~10-20 channels near the probe site rather than all
        channels — same as phy does locally, and essential for long recordings
        where seeking across all channels is prohibitively slow.

        Request fields
        --------------
        cluster_id : int
        n_spikes   : int, optional  (default 50)
        """
        self._require_model()
        cluster_id = int(req["cluster_id"])
        n_spikes = int(req.get("n_spikes", 50))

        # Get the template's local channel_ids — typically 10-20 channels
        template = self.model.get_template(cluster_id)
        channel_ids = template.channel_ids if template is not None else None

        spike_ids = self.model.get_cluster_spikes(cluster_id)
        if len(spike_ids) > n_spikes:
            rng = np.random.default_rng(seed=cluster_id)
            spike_ids = rng.choice(spike_ids, size=n_spikes, replace=False)
            spike_ids.sort()

        # shape: (n_spikes, n_samples, n_template_channels)
        waveforms = self.model.get_waveforms(spike_ids, channel_ids)
        if waveforms is None:
            return encode_response(status="error", error="no waveforms available")

        arr = np.asarray(waveforms, dtype=np.float32)
        ch_list = channel_ids.tolist() if channel_ids is not None else []
        return encode_response(
            array=arr,
            cluster_id=cluster_id,
            n_spikes=len(spike_ids),
            channel_ids=ch_list,
        )

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
        # Match Phy's behavior: use best channels for this cluster.
        template = self.model.get_template(cluster_id)
        channel_ids = template.channel_ids if template is not None else None
        features = self.model.get_features(spike_ids, channel_ids)
        if features is None:
            return encode_response(status="error", error="no features available")
        arr = np.asarray(getattr(features, "data", features), dtype=np.float32)
        ch_list = channel_ids.tolist() if channel_ids is not None else []
        return encode_response(array=arr, cluster_id=cluster_id, channel_ids=ch_list)

    def _handle_get_template_features(self, req: dict) -> list[bytes]:
        """Return template_features for a cluster (KiloSort template_features.npy path)."""
        self._require_model()
        cluster_id = int(req["cluster_id"])
        if not hasattr(self.model, "get_template_features"):
            return encode_response(status="error", error="template features unavailable")
        spike_ids = self.model.get_cluster_spikes(cluster_id)
        tf = self.model.get_template_features(spike_ids)
        if tf is None:
            return encode_response(status="error", error="no template features available")
        arr = np.asarray(tf, dtype=np.float32)
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

    def _handle_get_cluster_info(self, req: dict) -> list[bytes]:
        """
        Return a summary table for all clusters.

        Response header contains a 'clusters' list, each entry:
          { id, label, n_spikes, amplitude, fr }

        No array frame — all data fits in the JSON header.
        """
        self._require_model()
        m = self.model

        # spike counts per cluster
        spike_clusters = m.spike_clusters
        cluster_ids = m.cluster_ids.tolist()
        counts = {int(cid): int(np.sum(spike_clusters == cid)) for cid in cluster_ids}

        # labels / groups (good / mua / noise / unsorted)
        groups = {}
        if hasattr(m, 'cluster_groups') and m.cluster_groups:
            groups = {int(k): str(v) for k, v in m.cluster_groups.items()}

        # mean amplitude per cluster (from spike amplitudes if available)
        amplitudes = {}
        if hasattr(m, 'amplitudes') and m.amplitudes is not None:
            for cid in cluster_ids:
                mask = spike_clusters == cid
                if mask.any():
                    amplitudes[int(cid)] = float(np.mean(m.amplitudes[mask]))

        # firing rate: n_spikes / recording duration
        duration = float(m.spike_times[-1]) if len(m.spike_times) else 1.0

        clusters = []
        for cid in cluster_ids:
            cid = int(cid)
            n = counts.get(cid, 0)
            clusters.append({
                "id": cid,
                "label": groups.get(cid, "unsorted"),
                "n_spikes": n,
                "amplitude": round(amplitudes.get(cid, 0.0), 1),
                "fr": round(n / duration, 2),
            })

        return encode_response(clusters=clusters)

    def _handle_get_spike_data(self, req: dict) -> list[bytes]:
        """
        Return per-spike times (s) and amplitudes for a cluster.

        Response array shape: (n_spikes, 2), float64
          column 0 = spike time in seconds
          column 1 = spike amplitude (arbitrary units from model.amplitudes,
                     or 0 if amplitudes are unavailable)
        """
        self._require_model()
        cluster_id = int(req["cluster_id"])
        spike_ids = self.model.get_cluster_spikes(cluster_id)
        times = np.asarray(self.model.spike_times[spike_ids], dtype=np.float64)

        if hasattr(self.model, 'amplitudes') and self.model.amplitudes is not None:
            amps = np.asarray(self.model.amplitudes[spike_ids], dtype=np.float64)
        else:
            amps = np.zeros(len(spike_ids), dtype=np.float64)

        data = np.column_stack([times, amps])   # (n_spikes, 2)
        return encode_response(array=data, cluster_id=cluster_id)

    def _handle_get_traces(self, req: dict) -> list[bytes]:
        """
        Return raw (or high-pass filtered) voltage traces for a time window.

        Request fields
        --------------
        t_start     : float  seconds
        t_end       : float  seconds
        channel_ids : list[int]  optional — all channels if omitted
        filter      : bool  optional  high-pass filter at 300 Hz (default False)
        max_samples : int   optional  downsample to at most this many columns
                            (default 3000 — enough for a ~1000 px wide display)

        Response array shape: (n_channels, n_samples) float32
        Response header extras: sample_rate, t_start, t_end, channel_ids
        """
        self._require_model()
        m = self.model
        sr = float(m.sample_rate)

        t_start = float(req["t_start"])
        t_end   = float(req["t_end"])
        channel_ids = req.get("channel_ids", None)
        do_filter   = bool(req.get("filter", False))
        max_samples = int(req.get("max_samples", 3000))

        n_total = m.traces.shape[0]
        s_start = max(0, int(t_start * sr))
        s_end   = min(n_total, int(t_end * sr))
        if s_start >= s_end:
            return encode_response(status="error", error="empty time range")

        # Read the trace chunk — force into a plain numpy array
        chunk = np.array(m.traces[s_start:s_end], dtype=np.float32)  # (n_s, n_ch)
        if channel_ids is not None:
            chunk = chunk[:, list(channel_ids)]
        else:
            channel_ids = list(range(chunk.shape[1]))

        chunk = chunk.T  # (n_ch, n_s) — channel-first for efficient wire transfer

        # Optional high-pass filter (300 Hz, zero-phase)
        if do_filter:
            try:
                from scipy.signal import butter, sosfiltfilt
                sos = butter(3, 300.0 / (sr / 2.0), btype="high", output="sos")
                chunk = sosfiltfilt(sos, chunk, axis=1).astype(np.float32)
            except Exception as exc:
                logger.warning("HP filter failed: %s", exc)

        # Downsample for display if needed
        n_s = chunk.shape[1]
        if n_s > max_samples:
            step = n_s // max_samples
            chunk = chunk[:, ::step]
            actual_t_end = t_start + (chunk.shape[1] * step) / sr
        else:
            actual_t_end = s_end / sr

        actual_t_start = s_start / sr
        return encode_response(
            array=chunk,
            sample_rate=sr,
            t_start=actual_t_start,
            t_end=actual_t_end,
            channel_ids=list(channel_ids),
        )

    def _handle_get_spikes_in_window(self, req: dict) -> list[bytes]:
        """
        Return spike times and cluster ids within a time window.

        Request fields
        --------------
        t_start     : float  seconds
        t_end       : float  seconds
        cluster_ids : list[int]  optional — if given, filter to these clusters

        Response array shape: (n_spikes, 2) float64  [time_s, cluster_id]
        """
        self._require_model()
        m = self.model
        times    = m.spike_times
        clusters = m.spike_clusters

        t_start = float(req["t_start"])
        t_end   = float(req["t_end"])

        mask = (times >= t_start) & (times <= t_end)
        if "cluster_ids" in req:
            mask &= np.isin(clusters, req["cluster_ids"])

        result = np.column_stack([
            times[mask].astype(np.float64),
            clusters[mask].astype(np.float64),
        ]) if mask.any() else np.empty((0, 2), dtype=np.float64)

        return encode_response(
            array=result,
            t_start=t_start,
            t_end=t_end,
        )

    def _handle_get_channel_positions(self, req: dict) -> list[bytes]:
        """Return probe channel positions as (n_channels, 2) float32 array [x, y] in µm."""
        self._require_model()
        positions = np.asarray(self.model.channel_positions, dtype=np.float32)
        return encode_response(array=positions)

    def _handle_get_similar_clusters(self, req: dict) -> list[bytes]:
        """
        Return a ranked similarity list for one selected cluster (Phy-style).

        Request fields
        --------------
        cluster_id : int
        limit      : int, optional (default 100)
        """
        self._require_model()
        m = self.model
        cluster_id = int(req["cluster_id"])
        limit = int(req.get("limit", 100))

        if not hasattr(m, "similar_templates") or m.similar_templates is None:
            return encode_response(similar_clusters=[])
        if not hasattr(m, "spike_templates") or m.spike_templates is None:
            return encode_response(similar_clusters=[])

        cluster_ids = [int(c) for c in np.asarray(m.cluster_ids, dtype=np.int64)]
        if cluster_id not in cluster_ids:
            return encode_response(similar_clusters=[])

        # Build cluster -> template-counts mapping, mirroring Phy's TemplateController logic.
        spike_clusters = np.asarray(m.spike_clusters)
        spike_templates = np.asarray(m.spike_templates)
        n_templates = int(getattr(m, "n_templates", int(np.max(spike_templates)) + 1))

        counts_by_cluster: dict[int, np.ndarray] = {}
        for cid in cluster_ids:
            spike_ids = np.nonzero(spike_clusters == cid)[0]
            st = spike_templates[spike_ids]
            counts_by_cluster[cid] = np.bincount(st, minlength=n_templates)

        temp_i = np.nonzero(counts_by_cluster[cluster_id])[0]
        if len(temp_i) == 0:
            return encode_response(similar_clusters=[])

        similar_templates = np.asarray(m.similar_templates)
        sims = np.max(similar_templates[temp_i, :], axis=0)

        def _sim_for_cluster(cj: int) -> float:
            temp_j = np.nonzero(counts_by_cluster[cj])[0]
            if len(temp_j) == 0:
                return 0.0
            return float(np.max(sims[temp_j]))

        # Precompute info table fields.
        counts = {int(cid): int(np.sum(spike_clusters == cid)) for cid in cluster_ids}
        groups = {}
        if hasattr(m, "cluster_groups") and m.cluster_groups:
            groups = {int(k): str(v) for k, v in m.cluster_groups.items()}
        amplitudes = {}
        if hasattr(m, "amplitudes") and m.amplitudes is not None:
            for cid in cluster_ids:
                mask = spike_clusters == cid
                if mask.any():
                    amplitudes[cid] = float(np.mean(m.amplitudes[mask]))
        duration = float(m.spike_times[-1]) if len(m.spike_times) else 1.0

        rows = []
        for cid in cluster_ids:
            if cid == cluster_id:
                continue
            s = _sim_for_cluster(cid)
            n = counts.get(cid, 0)
            rows.append({
                "id": cid,
                "similarity": round(s, 3),
                "n_spikes": n,
                "label": groups.get(cid, "unsorted"),
                "amplitude": round(amplitudes.get(cid, 0.0), 1),
                "fr": round(n / duration, 2),
            })

        rows.sort(key=lambda r: r["similarity"], reverse=True)
        if limit > 0:
            rows = rows[:limit]
        return encode_response(primary_cluster_id=cluster_id, similar_clusters=rows)

    def _handle_label_cluster(self, req: dict) -> list[bytes]:
        """
        Set the label for one or more clusters and save to disk.

        Request fields
        --------------
        cluster_ids : list[int]  (or a single int as 'cluster_id')
        label       : str  one of good / mua / noise / unsorted
        """
        self._require_model()
        VALID = {"good", "mua", "noise", "unsorted"}
        label = str(req.get("label", ""))
        if label not in VALID:
            return encode_response(
                status="error",
                error=f"invalid label {label!r}, must be one of {sorted(VALID)}",
            )

        # Accept either cluster_ids (list) or cluster_id (scalar)
        if "cluster_ids" in req:
            cluster_ids = [int(c) for c in req["cluster_ids"]]
        else:
            cluster_ids = [int(req["cluster_id"])]

        m = self.model
        if not hasattr(m, 'cluster_groups') or m.cluster_groups is None:
            m.cluster_groups = {}
        for cid in cluster_ids:
            m.cluster_groups[cid] = label

        # Persist: phy uses save_metadata to write cluster_group.tsv
        if hasattr(m, 'save_metadata'):
            m.save_metadata("group", m.cluster_groups)
        elif hasattr(m, 'save'):
            m.save()

        logger.info("Labelled cluster(s) %s as %r", cluster_ids, label)
        return encode_response(cluster_ids=cluster_ids, label=label)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("server started without a model")

    def _handle_signal(self, signum, frame) -> None:
        logger.info("received signal %d, stopping", signum)
        self._running = False  # poller loop checks this flag and exits cleanly
