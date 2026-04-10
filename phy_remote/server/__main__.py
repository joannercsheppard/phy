"""
Server entry point — run as:

    python -m phy_remote.server /path/to/params.py

or inside an LSF job (see bsub_template.sh).

On startup the server prints a ready line:
    PHY_REMOTE_READY host=<hostname> port=<port>
which the bsub template parses to emit the correct ssh tunnel command.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m phy_remote.server",
        description="phy-remote ZMQ server — runs headless on the HPC cluster",
    )
    p.add_argument(
        "params_path",
        help="Path to params.py (Kilosort / phy-format dataset)",
    )
    p.add_argument("--port", type=int, default=5555, help="ZMQ port (default: 5555)")
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Interface to bind (default: 127.0.0.1).  "
            "Use 127.0.0.1 when the client connects via an SSH tunnel "
            "(recommended).  Use 0.0.0.0 only on a trusted internal network."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # ------------------------------------------------------------------ #
    # Load the TemplateModel
    # ------------------------------------------------------------------ #
    from pathlib import Path
    params_path = Path(args.params_path).resolve()
    if not params_path.exists():
        logger.error("params.py not found: %s", params_path)
        sys.exit(1)

    try:
        from phylib.io.model import load_model
    except ImportError:
        logger.error(
            "phylib is not installed in this environment.  "
            "Activate the phy conda environment first."
        )
        sys.exit(1)

    logger.info("Loading model from %s …", params_path)
    try:
        model = load_model(params_path)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        sys.exit(1)

    logger.info(
        "Model loaded: %d clusters, %d spikes, %d channels",
        len(model.cluster_ids),
        len(model.spike_times),
        model.n_channels,
    )

    # ------------------------------------------------------------------ #
    # Start server
    # ------------------------------------------------------------------ #
    from phy_remote.server.server import PhyServer

    server = PhyServer(model=model, port=args.port, host=args.host)

    hostname = socket.gethostname()

    # Machine-parseable ready line — the bsub template greps for this
    # and formats the tunnel command for the user.
    print(
        f"PHY_REMOTE_READY host={hostname} port={args.port}",
        flush=True,
    )
    # Human-readable tunnel hint on stderr
    print(
        f"\n  To connect from your MacBook, run:\n\n"
        f"    ssh -N -J <login_node> -L {args.port}:localhost:{args.port} "
        f"<user>@{hostname}\n\n"
        f"  Then launch the client:\n\n"
        f"    python -m phy_remote.client.app --port {args.port}\n",
        file=sys.stderr,
        flush=True,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
