"""
Server entry point — run as:

    python -m phy_remote.server /path/to/params.py

or via the bsub template (see bsub_template.sh).

Tunnel strategy
---------------
Janelia compute nodes do not accept inbound SSH from the login node, so
ProxyJump (-J) fails.  Instead we use a *reverse tunnel*: the compute node
opens an outbound SSH connection to the login node and pushes the ZMQ port
there, then the MacBook connects to the login node.

    compute node  ──reverse tunnel──►  login1:5555
    MacBook       ──────────────────►  login1:5555  (normal SSH -L)

Pass --reverse-tunnel <login_node> to have the server set this up
automatically (requires passwordless SSH from compute → login node, i.e.
your cluster SSH key must be in ~/.ssh/authorized_keys on the login node).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time

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
        help="Interface to bind (default: 127.0.0.1)",
    )
    p.add_argument(
        "--reverse-tunnel",
        metavar="LOGIN_NODE",
        default=None,
        help=(
            "Automatically open a reverse SSH tunnel to LOGIN_NODE so the "
            "MacBook can connect via the login node.  Requires passwordless "
            "SSH from this compute node to LOGIN_NODE.  "
            "Example: --reverse-tunnel login1.int.janelia.org"
        ),
    )
    p.add_argument(
        "--user",
        default=None,
        help="Username on LOGIN_NODE (defaults to $USER)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _check_not_login_node() -> None:
    in_lsf_job = bool(os.environ.get("LSB_JOBID"))
    in_slurm_job = bool(os.environ.get("SLURM_JOB_ID"))
    if not in_lsf_job and not in_slurm_job:
        hostname = socket.getfqdn()
        print(
            f"\nERROR: Not running inside an LSF/SLURM job (LSB_JOBID and "
            f"SLURM_JOB_ID are both unset).\n"
            f"  Current host : {hostname}\n\n"
            f"  Running the server on a login node will consume shared resources.\n\n"
            f"  Start an interactive job first:\n\n"
            f"    bsub -Is -q interactive -n 1 -R \"rusage[mem=8000]\" bash\n\n"
            f"  Then re-run this command.\n\n"
            f"  (To skip this check for local testing: export LSB_JOBID=local)\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _open_reverse_tunnel(login_node: str, port: int, user: str | None) -> subprocess.Popen:
    """
    Open a reverse SSH tunnel from this compute node to login_node.

    Creates:  login_node:port  →  localhost:port  (on this compute node)

    Requires passwordless SSH (key-based auth) from compute → login node.
    """
    user_host = f"{user}@{login_node}" if user else login_node
    cmd = [
        "ssh",
        "-N",                        # no remote command
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=no",
        "-R", f"{port}:localhost:{port}",
        user_host,
    ]
    logger.info("Opening reverse tunnel: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    # Give SSH a moment to establish the tunnel before accepting connections
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError(
            f"Reverse tunnel to {user_host} failed immediately (exit {proc.returncode}).\n"
            f"  Check that key-based SSH from this node to {login_node} works:\n"
            f"    ssh {user_host} echo ok"
        )
    return proc


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
    # Reverse tunnel (optional)
    # ------------------------------------------------------------------ #
    tunnel_proc: subprocess.Popen | None = None
    login_node = args.reverse_tunnel

    if login_node:
        try:
            tunnel_proc = _open_reverse_tunnel(login_node, args.port, args.user)
            logger.info("Reverse tunnel established to %s:%d", login_node, args.port)
        except Exception as exc:
            logger.error("Could not open reverse tunnel: %s", exc)
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Start server
    # ------------------------------------------------------------------ #
    from phy_remote.server.server import PhyServer

    server = PhyServer(model=model, port=args.port, host=args.host)

    hostname = socket.getfqdn()
    user = args.user or os.environ.get("USER", "<user>")

    print(f"PHY_REMOTE_READY host={hostname} port={args.port}", flush=True)

    if login_node:
        print(
            f"\n  Reverse tunnel active → {login_node}:{args.port}\n\n"
            f"  On your MacBook:\n\n"
            f"    ssh -N -L {args.port}:localhost:{args.port} "
            f"{user}@{login_node}\n\n"
            f"    python -m phy_remote.client.app --port {args.port}\n",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"\n  On your MacBook:\n\n"
            f"    ssh -N -L {args.port}:localhost:{args.port} "
            f"{user}@{login_node or '<login_node>'}\n\n"
            f"    python -m phy_remote.client.app --port {args.port}\n",
            file=sys.stderr,
            flush=True,
        )

    try:
        server.serve_forever()
    finally:
        if tunnel_proc is not None:
            tunnel_proc.terminate()
            tunnel_proc.wait()
            logger.info("Reverse tunnel closed")


if __name__ == "__main__":
    main()
