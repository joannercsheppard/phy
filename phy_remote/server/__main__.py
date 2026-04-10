"""
Server entry point — run as:

    python -m phy_remote.server /path/to/params.py

Tunnel strategy (same pattern as Janelia Jupyter jobs)
-------------------------------------------------------
Bind to 0.0.0.0 on the compute node, then forward from the login node to
the compute node over plain TCP.  No SSH keys between nodes required.

  Compute node:   python -m phy_remote.server params.py
                  (binds to 0.0.0.0:5555 by default)

  MacBook:        ssh -N -L 5555:<compute_node>:5555 sheppardj@login1.int.janelia.org
                  python -m phy_remote.client.app --port 5555
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
        default="0.0.0.0",
        help="Interface to bind (default: 0.0.0.0)",
    )
    p.add_argument(
        "--login-node",
        default="login1.int.janelia.org",
        help="Login node hostname, used to print the tunnel command",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _check_not_login_node() -> None:
    """Refuse to run if the hostname looks like a Janelia login node."""
    hostname = socket.gethostname()
    is_login = hostname.startswith("login") or "login" in hostname.split(".")[0]
    if is_login:
        print(
            f"\nERROR: Refusing to run on login node ({hostname}).\n\n"
            f"  Start an interactive job first:\n\n"
            f"    bsub -Is -q interactive -n 1 -R \"rusage[mem=8000]\" bash\n\n"
            f"  Then re-run this command.\n",
            file=sys.stderr,
        )
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    _check_not_login_node()

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
    user = os.environ.get("USER", "<user>")
    login_node = args.login_node

    print(f"PHY_REMOTE_READY host={hostname} port={args.port}", flush=True)
    print(
        f"\n  On your MacBook:\n\n"
        f"    ssh -N -L {args.port}:{hostname}:{args.port} "
        f"{user}@{login_node}\n\n"
        f"    python -m phy_remote.client.app --port {args.port}\n",
        file=sys.stderr,
        flush=True,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
