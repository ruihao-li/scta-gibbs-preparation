#!/usr/bin/env python3
"""Run the residual-scaling and endpoint Gibbs-state benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from scta_numerics.benchmarks import (
    load_config,
    run_benchmarks,
    smoke_configuration,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments without performing numerical work."""

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=here / "benchmark_config.json",
        help="JSON configuration (default: benchmark_config.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "results" / "benchmarks",
        help="Output root; raw files are written below OUTPUT/raw.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a small configuration that exercises every code path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the known benchmark files under OUTPUT/raw.",
    )
    return parser.parse_args()


def main() -> None:
    """Load, optionally reduce, and execute the configured benchmark."""

    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.smoke:
        config = smoke_configuration(config)
    numerics_root = Path(__file__).resolve().parent
    run_benchmarks(
        config,
        config_path,
        args.output.resolve(),
        numerics_root,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
