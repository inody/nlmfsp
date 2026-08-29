#!/usr/bin/env python3
"""Reproduce one figure or table from the paper.

Each config under ``configs/paper/`` corresponds to a specific paper artifact and
carries every setting that artifact needs, so nothing has to be supplied on the
command line. This wrapper deliberately exposes only the handful of options that
cannot live in a config -- which machine to run on, which seed, where to write --
because the published settings were previously scattered across a problem JSON,
a shell script and environment variables, and that is what made them hard to
recover.

Examples::

    # List what can be reproduced
    uv run python scripts/reproduce.py --list

    # Run one artifact
    uv run python scripts/reproduce.py --config configs/paper/vneck/main_nn.json

    # A backend table needs several seeds; see MANIFEST.md for the exact set
    for SEED in 310 311 312 313 314 315 316 317; do
        uv run python scripts/reproduce.py \\
            --config configs/paper/vneck/backend_table_nn.json --seed "$SEED"
    done

The published numbers were measured on GPUs that are no longer available
(see the paper-vs-code audit), so timings will not match exactly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIGS = REPO_ROOT / "configs" / "paper"
ENTRY_POINT = REPO_ROOT / "examples" / "validate_generalized_schrodinger_bridge.py"


def available_configs() -> list[Path]:
    return sorted(PAPER_CONFIGS.glob("*/*.json"))


def print_available() -> None:
    configs = available_configs()
    if not configs:
        print("No paper configs found. Run scripts/build_paper_configs.py first.")
        return
    print("Reproducible paper artifacts:\n")
    for path in configs:
        description = json.loads(path.read_text()).get("description", "")
        # Provenance is appended in brackets; it belongs in MANIFEST.md, not here.
        summary = description.split(" [derived from run")[0]
        print(f"  {path.relative_to(REPO_ROOT)}")
        print(f"      {summary}\n")
    print("See configs/paper/manifest/MANIFEST.md for the runs these came from.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="Paper config under configs/paper/.")
    parser.add_argument("--list", action="store_true", help="List reproducible artifacts and exit.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the seed. Tables average several seeds; see MANIFEST.md.",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "mps", "auto"), default=None,
        help="Compute device. Resolved automatically unless given.",
    )
    parser.add_argument("--name", type=str, default=None, help="Scenario name for the output directory.")
    parser.add_argument("--resume", type=Path, default=None, help="Experiment directory to resume from.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list or args.config is None:
        print_available()
        return 0 if args.list else 2

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if not config_path.exists():
        print(f"error: no such config: {args.config}", file=sys.stderr)
        return 2

    command = [sys.executable, str(ENTRY_POINT), "--problem-config", str(config_path)]
    for flag, value in (("--seed", args.seed), ("--device", args.device),
                        ("--name", args.name), ("--resume", args.resume)):
        if value is not None:
            command += [flag, str(value)]

    print(f"Reproducing {config_path.relative_to(REPO_ROOT)}")
    print(f"  $ {' '.join(command)}\n")
    return subprocess.run(command, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
