#!/usr/bin/env python3
"""Run the N=8 resonance and finite-deformation variational scans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scta_numerics.variational import (
    run_variational_scans,
    smoke_configuration,
)


def load_config(path: Path) -> dict[str, Any]:
    """Load a variational JSON configuration."""

    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a JSON object.")
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments without starting an optimization."""

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=here / "variational_config.json",
        help="JSON configuration (default: variational_config.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "results" / "variational",
        help="Output root; checkpoints are written below OUTPUT/raw.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a compatible interrupted run from its checkpoint.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use N=4 and three physical points while retaining ten starts.",
    )
    return parser.parse_args()


def main() -> None:
    """Load, optionally reduce, and execute the configured scans."""

    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.smoke:
        config = smoke_configuration(config)
    numerics_root = Path(__file__).resolve().parent
    run_variational_scans(
        config,
        config_path,
        args.output.resolve(),
        numerics_root,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
