"""Algebraic tests for the SCTA numerical benchmark."""

from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

NUMERICS = Path(__file__).resolve().parents[1]
if str(NUMERICS) not in sys.path:
    sys.path.insert(0, str(NUMERICS))

from scta_numerics.model import (  # noqa: E402
    BenchmarkParameters,
    bond_generator,
    core_hamiltonian,
    correction_unitary,
    deformation_hamiltonian,
    gibbs_state,
    graph_tail,
)
from scta_numerics.operators import (  # noqa: E402
    I2,
    X,
    Z,
    onsite_operator,
    partial_trace,
    trace_norm_hermitian,
)
from scta_numerics.pauli import corrected_residual_terms, terms_to_dense  # noqa: E402
from scta_numerics.benchmarks import validate_benchmark_config  # noqa: E402


class TestAlgebra(unittest.TestCase):
    def test_production_configuration_is_valid(self) -> None:
        config = json.loads((NUMERICS / "benchmark_config.json").read_text())
        validate_benchmark_config(config)

    def test_configuration_rejects_nonintegral_size(self) -> None:
        config = json.loads((NUMERICS / "benchmark_config.json").read_text())
        config["thermal_accuracy"]["N"] = 8.5
        with self.assertRaisesRegex(ValueError, "integer"):
            validate_benchmark_config(config)

    def test_configuration_rejects_invalid_fit_window(self) -> None:
        config = json.loads((NUMERICS / "benchmark_config.json").read_text())
        changed = deepcopy(config)
        changed["residual_scaling"]["fit_lambda_min"] = 0.2
        changed["residual_scaling"]["fit_lambda_max"] = 0.1
        with self.assertRaisesRegex(ValueError, "fit window"):
            validate_benchmark_config(changed)

    def test_bond_generator_is_antihermitian_and_cancels(self) -> None:
        h_left, h_right, coupling = 1.0, 1.5, 0.7
        generator = bond_generator(h_left, h_right, coupling)
        h_bond = -h_left * np.kron(Z, I2) - h_right * np.kron(I2, Z)
        deformation = coupling * np.kron(X, X)
        self.assertLess(np.linalg.norm(generator + generator.conj().T), 1.0e-13)
        self.assertLess(
            np.linalg.norm(h_bond @ generator - generator @ h_bond + deformation),
            1.0e-12,
        )

    def test_graph_tail_pullback(self) -> None:
        n_sites = 4
        ug = graph_tail(n_sites)
        self.assertLess(
            np.linalg.norm(ug.conj().T @ ug - np.eye(2**n_sites)), 1.0e-12
        )
        for site in range(n_sites):
            physical_z = onsite_operator(Z, site, n_sites)
            core_x = onsite_operator(X, site, n_sites)
            self.assertLess(
                np.linalg.norm(ug.conj().T @ physical_z @ ug - core_x), 1.0e-12
            )

    def test_sparse_pauli_residual_matches_dense(self) -> None:
        params = BenchmarkParameters(4)
        lam = 0.071
        h0 = core_hamiltonian(params)
        vc = deformation_hamiltonian(params)
        w = correction_unitary(params, lam)
        dense = w.conj().T @ (h0 + lam * vc) @ w - h0
        sparse = terms_to_dense(corrected_residual_terms(params, lam))
        self.assertLess(np.linalg.norm(dense - sparse, ord="fro"), 1.0e-11)

    def test_exact_gibbs_covariance(self) -> None:
        params = BenchmarkParameters(4)
        beta = 1.3
        lam = 0.08
        target = core_hamiltonian(params) + lam * deformation_hamiltonian(params)
        w = correction_unitary(params, lam)
        rho_target, _, _ = gibbs_state(target, beta)
        rho_reduced, _, _ = gibbs_state(w.conj().T @ target @ w, beta)
        self.assertLess(
            trace_norm_hermitian(w @ rho_reduced @ w.conj().T - rho_target),
            1.0e-11,
        )

    def test_partial_trace_of_product_state(self) -> None:
        rho_a = np.diag([0.7, 0.3]).astype(complex)
        rho_b = np.diag([0.2, 0.8]).astype(complex)
        rho_c = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=complex)
        rho = np.kron(np.kron(rho_a, rho_b), rho_c)
        reduced = partial_trace(rho, [0, 2], 3)
        self.assertLess(np.linalg.norm(reduced - np.kron(rho_a, rho_c)), 1.0e-13)

    def test_second_order_product_ordering_coefficient(self) -> None:
        params = BenchmarkParameters(4)
        h0 = core_hamiltonian(params)
        vc = deformation_hamiltonian(params)
        fields = params.fields()
        s_odd = np.zeros_like(h0)
        s_even = np.zeros_like(h0)
        from scta_numerics.operators import adjacent_operator

        for left in range(params.n_sites - 1):
            local = bond_generator(fields[left], fields[left + 1], params.coupling)
            if left % 2 == 0:
                s_odd += adjacent_operator(local, left, params.n_sites)
            else:
                s_even += adjacent_operator(local, left, params.n_sites)
        s_total = s_odd + s_even
        comm = lambda a, b: a @ b - b @ a
        expected = 0.5 * comm(vc, s_total) + 0.5 * comm(
            h0, comm(s_odd, s_even)
        )
        lam = 1.0e-5
        w = correction_unitary(params, lam)
        observed = (w.conj().T @ (h0 + lam * vc) @ w - h0) / lam**2
        relative = np.linalg.norm(observed - expected, ord="fro") / np.linalg.norm(
            expected, ord="fro"
        )
        self.assertLess(relative, 2.0e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
