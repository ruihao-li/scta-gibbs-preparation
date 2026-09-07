"""Focused tests for analysis fitting, validation, and path handling."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


NUMERICS = Path(__file__).resolve().parents[1]
if str(NUMERICS) not in sys.path:
    sys.path.insert(0, str(NUMERICS))

import analyze_results as analysis  # noqa: E402


class TestAnalysisFits(unittest.TestCase):
    def test_power_law_fit_recovers_exact_exponent(self) -> None:
        x = np.array([1.0, 2.0, 4.0, 8.0])
        frame = pd.DataFrame({"x": x, "y": 3.0 * x**2})
        result = analysis.power_law_fit(
            frame,
            "x",
            "y",
            label="exact",
            aspect="test",
        )
        self.assertAlmostEqual(result["exponent"], 2.0, places=13)
        self.assertAlmostEqual(result["prefactor"], 3.0, places=13)

    def test_grouped_fit_uses_one_slope_and_distinct_prefactors(self) -> None:
        rows = []
        for group, prefactor in ((0.5, 2.0), (1.0, 7.0)):
            for x in (1.0, 2.0, 4.0, 8.0):
                rows.append(
                    {"x": x, "y": prefactor * x**1.5, "group": group}
                )
        fitted = analysis.grouped_power_law_fit(
            pd.DataFrame(rows), "x", "y", "group"
        )
        self.assertEqual(len(fitted), 2)
        for row in fitted:
            self.assertAlmostEqual(row["exponent"], 1.5, places=13)

    def test_crossover_uses_log_coordinate_interpolation(self) -> None:
        frame = pd.DataFrame(
            [
                {"scan": "coupling", "lambda": 0.5, "method": "bare", "error": 1.0},
                {
                    "scan": "coupling",
                    "lambda": 0.5,
                    "method": "analytic_first_order",
                    "error": 0.8,
                },
                {"scan": "coupling", "lambda": 1.0, "method": "bare", "error": 1.0},
                {
                    "scan": "coupling",
                    "lambda": 1.0,
                    "method": "analytic_first_order",
                    "error": 1.2,
                },
            ]
        )
        summary = analysis.crossover_summary(
            frame,
            scan="coupling",
            x="lambda",
            metric="error",
            descending=False,
        )
        self.assertTrue(summary["found"])
        self.assertAlmostEqual(summary["estimate"], np.sqrt(0.5), places=13)


class TestAnalysisInputs(unittest.TestCase):
    def setUp(self) -> None:
        model = {
            "boundary": "open",
            "h_bar": 1.25,
            "J": 1.0,
            "field_convention": "one-based alternating field",
        }
        self.benchmark_config = {
            "model": {**model, "delta": 0.25},
            "residual_scaling": {
                "N_values": [2],
                "primary_N": 2,
                "lambda_values": [0.1],
                "size_lambda": 0.05,
                "dense_check_lambdas": [0.1],
                "fit_lambda_min": 0.1,
                "fit_lambda_max": 0.1,
            },
            "thermal_accuracy": {
                "N": 2,
                "beta_values": [1.0],
                "lambda_values": [0.1],
                "fit_lambda_min": 0.1,
                "fit_lambda_max": 0.1,
            },
            "numerics": {
                "tolerance_sweep_lambdas": [0.05],
                "tolerance_sweep_values": [1.0e-13],
            },
        }
        self.variational_config = {
            "model": model,
            "variational_scans": {
                "resonance_lambda": 0.02,
                "resonance_delta_values": [0.25, 0.0],
                "coupling_delta": 0.25,
                "coupling_lambda_values": [0.02, 0.1],
                "random_starts": 1,
            },
        }

    def raw_frames(self) -> dict[str, pd.DataFrame]:
        residual = pd.DataFrame(
            [
                {"N": 2, "lambda": 0.1, "method": "bare"},
                {"N": 2, "lambda": 0.1, "method": "first_order"},
                {"N": 2, "lambda": 0.05, "method": "bare_size_scan"},
                {"N": 2, "lambda": 0.05, "method": "first_order_size_scan"},
            ]
        )
        algebra = pd.DataFrame([{"lambda": 0.1}])
        tolerance = pd.DataFrame(
            [{"lambda": 0.05, "prune_tolerance": 1.0e-13}]
        )
        thermal = pd.DataFrame(
            [
                {"N": 2, "beta": 1.0, "lambda": 0.1, "method": "bare"},
                {"N": 2, "beta": 1.0, "lambda": 0.1, "method": "first_order"},
            ]
        )
        methods = ("bare", "analytic_first_order", "variational")
        memberships = (
            ("point_000", "resonance", 0.25, 0.02),
            ("point_001", "resonance", 0.0, 0.02),
            ("point_000", "coupling", 0.25, 0.02),
            ("point_002", "coupling", 0.25, 0.1),
        )
        combined = pd.DataFrame(
            [
                {
                    "point_id": point_id,
                    "scan": scan,
                    "delta": delta,
                    "lambda": lam,
                    "method": method,
                    "diagnostic": 1.0,
                }
                for point_id, scan, delta, lam in memberships
                for method in methods
            ]
        )
        physical_points = (
            ("point_000", 0.25, 0.02),
            ("point_001", 0.0, 0.02),
            ("point_002", 0.25, 0.1),
        )
        starts = ("analytic_seed", "bare_seed", "random_00")
        optimizer = pd.DataFrame(
            [
                {
                    "point_id": point_id,
                    "delta": delta,
                    "lambda": lam,
                    "start": start,
                }
                for point_id, delta, lam in physical_points
                for start in starts
            ]
        )
        return {
            "residual": residual,
            "algebra": algebra,
            "tolerance_sweep": tolerance,
            "thermal": thermal,
            "combined": combined,
            "optimizer_runs": optimizer,
        }

    def test_exact_configured_coordinate_checks_and_missing_row(self) -> None:
        frames = self.raw_frames()
        checks = analysis.configured_completeness_checks(
            **frames,
            benchmark_config=self.benchmark_config,
            variational_config=self.variational_config,
        )
        self.assertTrue(all(check["passed"] for check in checks))

        frames["thermal"] = frames["thermal"].iloc[:-1].copy()
        checks = analysis.configured_completeness_checks(
            **frames,
            benchmark_config=self.benchmark_config,
            variational_config=self.variational_config,
        )
        failed = {check["check"] for check in checks if not check["passed"]}
        self.assertEqual(failed, {"thermal configured-coordinate completeness"})

    def test_shared_payload_and_point_ids_are_checked(self) -> None:
        frames = self.raw_frames()
        selector = (
            (frames["combined"].point_id == "point_000")
            & (frames["combined"].scan == "coupling")
            & (frames["combined"].method == "bare")
        )
        frames["combined"].loc[selector, "diagnostic"] = 2.0
        frames["optimizer_runs"].loc[0, "point_id"] = "point_001"
        checks = analysis.configured_completeness_checks(
            **frames,
            benchmark_config=self.benchmark_config,
            variational_config=self.variational_config,
        )
        failed = {check["check"] for check in checks if not check["passed"]}
        self.assertEqual(
            failed,
            {
                "optimizer configured-coordinate completeness",
                "shared-scan payload consistency",
            },
        )

    def test_coordinate_keys_do_not_round_nearby_floats_together(self) -> None:
        value = 0.1
        neighbor = np.nextafter(value, np.inf)
        self.assertNotEqual(
            analysis._coordinate_key((value,)),
            analysis._coordinate_key((neighbor,)),
        )

    def test_custom_paths_and_output_schema_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "raw-results"
            benchmark_raw = results / "benchmarks" / "raw"
            variational_raw = results / "variational" / "raw"
            benchmark_raw.mkdir(parents=True)
            variational_raw.mkdir(parents=True)
            benchmark_config_path = root / "benchmark.json"
            variational_config_path = root / "variational.json"
            benchmark_config_path.write_text(json.dumps(self.benchmark_config))
            variational_config_path.write_text(json.dumps(self.variational_config))
            benchmark_names = (
                "residual_scaling.csv",
                "algebra_checks.csv",
                "pauli_tolerance_sweep.csv",
                "thermal_accuracy.csv",
            )
            variational_names = ("scan_results.csv", "optimizer_runs.csv")
            for name in benchmark_names:
                (benchmark_raw / name).write_text("column\n", encoding="utf-8")
            for name in variational_names:
                (variational_raw / name).write_text("column\n", encoding="utf-8")
            benchmark_metadata = {
                "output_schema_version": 1,
                "status": "complete",
                "effective_config": self.benchmark_config,
                "output_file_sha256": {
                    name: analysis.sha256(benchmark_raw / name)
                    for name in benchmark_names
                },
            }
            variational_metadata = {
                "output_schema_version": 1,
                "status": "complete",
                "effective_config": self.variational_config,
                "output_file_sha256": {
                    name: analysis.sha256(variational_raw / name)
                    for name in variational_names
                },
            }
            (benchmark_raw / "run_metadata.json").write_text(
                json.dumps(benchmark_metadata)
            )
            (variational_raw / "run_metadata.json").write_text(
                json.dumps(variational_metadata)
            )
            processed = root / "derived" / "tables"
            figures = root / "derived" / "figures"
            analysis.configure_paths(
                results,
                processed,
                figures,
                benchmark_config_path,
                variational_config_path,
            )
            self.assertEqual(analysis.RAW, benchmark_raw)
            self.assertEqual(analysis.COMBINED_RAW, variational_raw)
            self.assertEqual(analysis.PROCESSED, processed)
            self.assertEqual(analysis.FIGURES, figures)
            processed.mkdir(parents=True)
            figures.mkdir(parents=True)
            manifest = analysis.write_hashes()
            self.assertIn(
                "source/benchmark_config.json", set(manifest["path"])
            )
            self.assertFalse(any(Path(label).is_absolute() for label in manifest["path"]))

            with self.assertRaisesRegex(ValueError, "overlap raw input roots"):
                analysis.configure_paths(
                    results,
                    benchmark_raw / "derived",
                    figures,
                    benchmark_config_path,
                    variational_config_path,
                )

            corrupted = benchmark_raw / "residual_scaling.csv"
            corrupted.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "failed its hash check"):
                analysis.configure_paths(
                    results,
                    processed,
                    figures,
                    benchmark_config_path,
                    variational_config_path,
                )
            corrupted.write_text("column\n", encoding="utf-8")

            benchmark_metadata["output_schema_version"] = 0
            (benchmark_raw / "run_metadata.json").write_text(
                json.dumps(benchmark_metadata)
            )
            with self.assertRaisesRegex(RuntimeError, "output schema"):
                analysis.configure_paths(
                    results,
                    processed,
                    figures,
                    benchmark_config_path,
                    variational_config_path,
                )

    def test_cli_exposes_separate_processed_and_figure_outputs(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "analyze_results.py",
                "--processed-output",
                "/tmp/tables",
                "--figures-output",
                "/tmp/figures",
            ],
        ):
            args = analysis.parse_args()
        self.assertEqual(args.processed_output, Path("/tmp/tables"))
        self.assertEqual(args.figures_output, Path("/tmp/figures"))


if __name__ == "__main__":
    unittest.main()
