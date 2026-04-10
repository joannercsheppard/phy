"""
ZMQ message protocol for phy-remote.

Wire format (multipart message):
  Frame 0 — JSON-encoded header  (always present)
  Frame 1 — raw numpy bytes      (only when header["has_array"] is True)

Request header fields:
  cmd      str    command name (see CMD_* constants)
  **kwargs        command-specific keyword arguments

Response header fields:
  status   str    "ok" | "error"
  error    str    error message (only when status == "error")
  has_array bool  True when frame 1 is present
  dtype    str    numpy dtype string  (only when has_array)
  shape    list   array shape as list of ints (only when has_array)
"""

import json
import numpy as np

# ---------------------------------------------------------------------------
# Command constants
# ---------------------------------------------------------------------------

CMD_PING = "ping"
CMD_GET_WAVEFORMS = "get_waveforms"
CMD_GET_SPIKE_TIMES = "get_spike_times"
CMD_GET_FEATURES = "get_features"
CMD_GET_TEMPLATES = "get_templates"
CMD_GET_CLUSTER_IDS = "get_cluster_ids"

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_request(cmd: str, **kwargs) -> list[bytes]:
    """Return a single-frame multipart message for a command request."""
    header = {"cmd": cmd, **kwargs}
    return [json.dumps(header).encode()]


def decode_request(frames: list[bytes]) -> dict:
    """Parse an incoming request from the server side."""
    return json.loads(frames[0])


def encode_response(
    *,
    status: str = "ok",
    error: str = "",
    array: "np.ndarray | None" = None,
    **extra,
) -> list[bytes]:
    """
    Build a response multipart message.

    Parameters
    ----------
    status : "ok" | "error"
    error  : error string (when status == "error")
    array  : optional numpy array to attach as frame 1
    **extra: additional metadata fields written into the header
    """
    header: dict = {"status": status, "has_array": array is not None, **extra}
    if status == "error":
        header["error"] = error
    if array is not None:
        header["dtype"] = array.dtype.str   # e.g. "<f4"
        header["shape"] = list(array.shape)
        return [json.dumps(header).encode(), array.tobytes()]
    return [json.dumps(header).encode()]


def decode_response(frames: list[bytes]) -> tuple[dict, "np.ndarray | None"]:
    """
    Parse a server response.

    Returns
    -------
    header : dict
    array  : np.ndarray or None
    """
    header = json.loads(frames[0])
    array = None
    if header.get("has_array"):
        dtype = np.dtype(header["dtype"])
        shape = tuple(header["shape"])
        array = np.frombuffer(frames[1], dtype=dtype).reshape(shape)
    return header, array
