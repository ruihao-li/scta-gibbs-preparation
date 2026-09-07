"""Tests for the direct-angle variational scans."""

from __future__ import annotations

import csv
import sys
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

NUMERICS = Path(__file__).resolve().parents[1]
if str(NUMERICS) not in sys.path:
    sys.path.insert(0, str(NUMERICS))

from scta_numerics.variational import (  # noqa: E402
    _completed_points,
    _discard_incomplete_points,
    _validate_checkpoint_rows,
    parameter_starts,
    run_variational_scans,
    scan_points,
    smoke_configuration,
    validate_variational_config,
)
from scta_numerics.benchmarks import (  # noqa: E402
    environment_metadata,
    write_csv,
    write_json,
)
from scta_numerics.model import (  # noqa: E402
    BenchmarkParameters,
    analytic_channel_angles,
    bond_channel_directions,
    core_hamiltonian,
    correction_unitary,
    deformation_hamiltonian,
    direct_angle_correction_unitary,
    field_from_loader_angle,
    loader_angle_from_field,
    product_core_state,
    product_core_state_from_angles,
)
from scta_numerics.operators import I2, SM, SP, X, Z  # noqa: E402
from scta_numerics.pauli import (  # noqa: E402
    direct_angle_residual_terms,
    terms_to_dense,
)


class TestVariationalScans(unittest.TestCase):
    def test_channel_directions_commute_and_factorize_exactly(self) -> None:
        pair, exchange = bond_channel_directions()
        self.assertLess(np.linalg.norm(pair + pair.conj().T), 1.0e-14)
        self.assertLess(np.linalg.norm(exchange + exchange.conj().T), 1.0e-14)
        self.assertLess(np.linalg.norm(pair @ exchange - exchange @ pair), 1.0e-14)
        theta_pair, theta_exchange = 0.37, -0.82
        direct = expm(theta_pair * pair + theta_exchange * exchange)
        factorized = expm(theta_pair * pair) @ expm(theta_exchange * exchange)
        self.assertLess(np.linalg.norm(direct - factorized), 1.0e-13)

    def test_direct_angles_reproduce_nonresonant_analytic_circuit(self) -> None:
        params = BenchmarkParameters(4, h_bar=1.25, delta=0.25, coupling=0.7)
        lam = 0.13
        theta_pair, theta_exchange, omitted = analytic_channel_angles(params, lam)
        self.assertFalse(omitted)
        direct = direct_angle_correction_unitary(
            params, theta_pair, theta_exchange
        )
        original = correction_unitary(params, lam)
        self.assertLess(np.linalg.norm(direct - original, ord="fro"), 1.0e-12)

    def test_resonant_analytic_seed_omits_only_exchange_angle(self) -> None:
        params = BenchmarkParameters(4, h_bar=1.25, delta=0.0, coupling=1.0)
        theta_pair, theta_exchange, omitted = analytic_channel_angles(params, 0.02)
        self.assertTrue(omitted)
        self.assertAlmostEqual(theta_pair, 0.004)
        self.assertEqual(theta_exchange, 0.0)
        correction = direct_angle_correction_unitary(params, theta_pair, 0.41)
        identity = np.eye(2**params.n_sites)
        self.assertLess(
            np.linalg.norm(correction.conj().T @ correction - identity, ord="fro"),
            1.0e-12,
        )

    def test_resonant_pair_generator_leaves_exchange_channel(self) -> None:
        h_value, coupling = 1.25, 0.7
        pair_direction, _ = bond_channel_directions()
        pair_generator = coupling * pair_direction / (4.0 * h_value)
        h_bond = -h_value * np.kron(Z, I2) - h_value * np.kron(I2, Z)
        deformation = coupling * np.kron(X, X)
        retained_exchange = coupling * (
            np.kron(SP, SM) + np.kron(SM, SP)
        )
        first_order_remainder = (
            h_bond @ pair_generator - pair_generator @ h_bond + deformation
        )
        self.assertLess(
            np.linalg.norm(first_order_remainder - retained_exchange), 1.0e-13
        )

    def test_direct_angle_sparse_residual_matches_dense_identity(self) -> None:
        params = BenchmarkParameters(4, h_bar=1.25, delta=0.0, coupling=1.0)
        lam = 0.071
        theta_pair, theta_exchange = 0.026, -0.31
        correction = direct_angle_correction_unitary(
            params, theta_pair, theta_exchange
        )
        dense = (
            correction.conj().T
            @ (core_hamiltonian(params) + lam * deformation_hamiltonian(params))
            @ correction
            - core_hamiltonian(params)
        )
        sparse = terms_to_dense(
            direct_angle_residual_terms(
                params, lam, theta_pair, theta_exchange
            )
        )
        self.assertLess(np.linalg.norm(dense - sparse, ord="fro"), 1.0e-11)

    def test_ten_starts_are_reproducible_at_exact_resonance(self) -> None:
        settings = {
            "beta": 1.0,
            "angle_bounds": [-np.pi, np.pi],
            "core_angle_bounds": [1.0e-6, np.pi / 2.0 - 1.0e-6],
            "random_seed": 1701,
            "random_starts": 8,
        }
        params = BenchmarkParameters(4, h_bar=1.25, delta=0.0, coupling=1.0)
        first, status = parameter_starts(params, 0.02, settings, point_index=7)
        second, second_status = parameter_starts(
            params, 0.02, settings, point_index=7
        )
        self.assertEqual(status, "pair_only_at_resonance")
        self.assertEqual(status, second_status)
        self.assertEqual([name for name, _ in first[:2]], ["analytic_seed", "bare_seed"])
        self.assertEqual(len(first), 10)
        self.assertEqual(first[0][1][1], 0.0)
        for (name_a, values_a), (name_b, values_b) in zip(first, second):
            self.assertEqual(name_a, name_b)
            self.assertTrue(np.array_equal(values_a, values_b))

    def test_loader_angle_bounds_exclude_singular_endpoints(self) -> None:
        settings = {
            "beta": 1.0,
            "angle_bounds": [-np.pi, np.pi],
            "core_angle_bounds": [0.0, np.pi / 2.0],
            "random_seed": 1701,
            "random_starts": 8,
        }
        params = BenchmarkParameters(4, h_bar=1.25, delta=0.25, coupling=1.0)
        with self.assertRaisesRegex(ValueError, "0 < lower < upper < pi/2"):
            parameter_starts(params, 0.02, settings, point_index=0)

    def test_production_configuration_is_valid(self) -> None:
        config = json.loads((NUMERICS / "variational_config.json").read_text())
        validate_variational_config(config)

    def test_configuration_rejects_duplicate_scan_points(self) -> None:
        config = json.loads(
            (NUMERICS / "variational_config.json").read_text(encoding="utf-8")
        )
        config["variational_scans"]["coupling_lambda_values"].append(0.05)
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            validate_variational_config(config)

    def test_configuration_rejects_nonintegral_maxls(self) -> None:
        config = json.loads(
            (NUMERICS / "variational_config.json").read_text(encoding="utf-8")
        )
        config["variational_scans"]["maxls"] = 3.5
        with self.assertRaisesRegex(ValueError, "integer"):
            validate_variational_config(config)

    def test_partial_checkpoint_rows_are_discarded_together(self) -> None:
        results = [
            {"point_id": "point_000", "scan": "resonance", "method": "bare"},
            {"point_id": "point_001", "scan": "resonance", "method": "bare"},
        ]
        optimizer = [
            {"point_id": "point_000", "start": "bare_seed"},
            {"point_id": "point_001", "start": "bare_seed"},
        ]
        clean_results, clean_optimizer = _discard_incomplete_points(
            results, optimizer, {"point_000"}
        )
        self.assertEqual({row["point_id"] for row in clean_results}, {"point_000"})
        self.assertEqual({row["point_id"] for row in clean_optimizer}, {"point_000"})

    def test_shared_point_requires_both_scan_memberships(self) -> None:
        point = {
            "point_id": "point_000",
            "scans": ["resonance", "coupling"],
        }
        results = [
            {
                "point_id": "point_000",
                "scan": "resonance",
                "method": method,
            }
            for method in ("bare", "analytic_first_order", "variational")
        ]
        optimizer = [
            {"point_id": "point_000", "start": f"start_{index}"}
            for index in range(10)
        ]
        self.assertEqual(
            _completed_points([point], results, optimizer, starts=10), set()
        )

    def test_resume_rejects_changed_effective_configuration(self) -> None:
        source_config = json.loads(
            (NUMERICS / "variational_config.json").read_text(encoding="utf-8")
        )
        source_config = smoke_configuration(source_config)
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            config_path = temporary_path / "config.json"
            config_path.write_text(json.dumps(source_config), encoding="utf-8")
            raw_dir = temporary_path / "output" / "raw"
            raw_dir.mkdir(parents=True)
            metadata = environment_metadata(config_path, NUMERICS, source_config)
            metadata.update(
                {
                    "effective_config": source_config,
                    "status": "running",
                    "elapsed_seconds": 0.0,
                }
            )
            write_json(raw_dir / "run_metadata.json", metadata)
            changed = deepcopy(source_config)
            changed["variational_scans"]["beta"] = 1.1
            with self.assertRaisesRegex(ValueError, "different effective configuration"):
                run_variational_scans(
                    changed,
                    config_path,
                    temporary_path / "output",
                    NUMERICS,
                    resume=True,
                )

    def test_running_checkpoint_recovers_from_mismatched_payload_hashes(self) -> None:
        config = json.loads(
            (NUMERICS / "variational_config.json").read_text(encoding="utf-8")
        )
        config = smoke_configuration(config)
        points = scan_points(config["variational_scans"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "output"
            raw = output / "raw"
            partial = [{
                "point_id": points[0]["point_id"],
                "delta": points[0]["delta"],
                "lambda": points[0]["lambda"],
                "scan": points[0]["scans"][0],
                "method": "bare",
            }]
            write_csv(raw / "scan_results.csv", partial)
            metadata = environment_metadata(config_path, NUMERICS, config)
            metadata.update({
                "status": "running",
                "elapsed_seconds": 0.0,
                "output_file_sha256": {"scan_results.csv": "stale"},
            })
            write_json(raw / "run_metadata.json", metadata)

            def fake_point(point, *_args):
                results = [
                    {
                        "point_id": point["point_id"],
                        "delta": point["delta"],
                        "lambda": point["lambda"],
                        "scan": scan,
                        "method": method,
                    }
                    for scan in point["scans"]
                    for method in ("bare", "analytic_first_order", "variational")
                ]
                optimizer = [
                    {
                        "point_id": point["point_id"],
                        "delta": point["delta"],
                        "lambda": point["lambda"],
                        "start": start,
                    }
                    for start in (
                        "analytic_seed",
                        "bare_seed",
                        *(f"random_{index:02d}" for index in range(8)),
                    )
                ]
                return results, optimizer

            with patch(
                "scta_numerics.variational.run_variational_point",
                side_effect=fake_point,
            ):
                run_variational_scans(
                    config, config_path, output, NUMERICS, resume=True
                )

            with (raw / "scan_results.csv").open(newline="", encoding="utf-8") as handle:
                results = list(csv.DictReader(handle))
            with (raw / "optimizer_runs.csv").open(newline="", encoding="utf-8") as handle:
                optimizer = list(csv.DictReader(handle))
            self.assertEqual(
                len(results), sum(len(point["scans"]) for point in points) * 3
            )
            self.assertEqual(len(optimizer), len(points) * 10)
            self.assertEqual(
                json.loads((raw / "run_metadata.json").read_text())["status"],
                "complete",
            )

    def test_loader_angle_parameterization_matches_product_gibbs_state(self) -> None:
        beta = 1.3
        fields = np.array([0.4, 1.1, 0.4, 1.1])
        angles = loader_angle_from_field(fields, beta)
        reconstructed_fields = field_from_loader_angle(angles, beta)
        state_from_fields, entropy_from_fields = product_core_state(fields, beta)
        state_from_angles, entropy_from_angles = product_core_state_from_angles(
            angles
        )
        self.assertLess(np.linalg.norm(reconstructed_fields - fields), 1.0e-13)
        self.assertLess(np.linalg.norm(state_from_angles - state_from_fields), 1.0e-13)
        self.assertAlmostEqual(entropy_from_angles, entropy_from_fields, places=13)

    def test_duplicate_physical_point_is_optimized_once(self) -> None:
        settings = {
            "resonance_lambda": 0.02,
            "resonance_delta_values": [0.5, 0.25, 0.0],
            "coupling_delta": 0.25,
            "coupling_lambda_values": [0.02, 0.3],
        }
        points = scan_points(settings)
        self.assertEqual(len(points), 4)
        shared = [
            point
            for point in points
            if point["delta"] == 0.25 and point["lambda"] == 0.02
        ]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]["scans"], ["resonance", "coupling"])

    def test_checkpoint_rejects_mismatched_point_coordinates(self) -> None:
        settings = {
            "resonance_lambda": 0.02,
            "resonance_delta_values": [0.5, 0.25, 0.00390625],
            "coupling_delta": 0.25,
            "coupling_lambda_values": [0.02, 0.05, 0.8],
        }
        checkpoint = [{
            "point_id": "point_000",
            "delta": "0.25",
            "lambda": "0.02",
            "scan": "resonance",
            "method": "bare",
        }]
        with self.assertRaisesRegex(ValueError, "configuration declares"):
            _validate_checkpoint_rows(
                scan_points(settings), checkpoint, [], {"analytic_seed"}
            )

    def test_checkpoint_rejects_unknown_optimizer_start(self) -> None:
        settings = {
            "resonance_lambda": 0.02,
            "resonance_delta_values": [0.5],
            "coupling_delta": 0.25,
            "coupling_lambda_values": [0.05],
        }
        checkpoint = [{
            "point_id": "point_000",
            "delta": "0.5",
            "lambda": "0.02",
            "start": "random_99",
        }]
        with self.assertRaisesRegex(ValueError, "unknown start name"):
            _validate_checkpoint_rows(
                scan_points(settings), [], checkpoint, {"analytic_seed"}
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
