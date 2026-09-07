"""Exact sparse Pauli propagation for the declared local residual."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .model import (
    BenchmarkParameters,
    correction_gate_layers,
    direct_angle_gate_layers,
)
from .operators import PAULI, PAULI_LABELS, kron_all, pauli_matrix


PauliTerms = dict[str, complex]


def core_pauli_terms(
    params: BenchmarkParameters, fields: np.ndarray | None = None
) -> PauliTerms:
    terms: PauliTerms = {}
    h = params.fields() if fields is None else np.asarray(fields, dtype=float)
    if h.shape != (params.n_sites,):
        raise ValueError("The field array must have one entry per site.")
    for site, field in enumerate(h):
        label = ["I"] * params.n_sites
        label[site] = "Z"
        terms["".join(label)] = complex(-field)
    return terms


def deformation_pauli_terms(params: BenchmarkParameters, lam: float = 1.0) -> PauliTerms:
    terms: PauliTerms = {}
    for left in range(params.n_sites - 1):
        label = ["I"] * params.n_sites
        label[left] = "X"
        label[left + 1] = "X"
        terms["".join(label)] = complex(lam * params.coupling)
    return terms


def add_terms(*collections: PauliTerms, tolerance: float = 0.0) -> PauliTerms:
    result: defaultdict[str, complex] = defaultdict(complex)
    for terms in collections:
        for label, coefficient in terms.items():
            result[label] += coefficient
    return {
        label: coefficient
        for label, coefficient in result.items()
        if abs(coefficient) > tolerance
    }


def scale_terms(terms: PauliTerms, factor: complex) -> PauliTerms:
    return {label: factor * coefficient for label, coefficient in terms.items()}


def local_pauli_conjugation_map(
    gate: np.ndarray, tolerance: float = 1.0e-14
) -> dict[str, dict[str, complex]]:
    """Map each two-qubit Pauli under ``P -> gate^dagger P gate``."""

    basis = {
        first + second: np.kron(PAULI[first], PAULI[second])
        for first in PAULI_LABELS
        for second in PAULI_LABELS
    }
    mapping: dict[str, dict[str, complex]] = {}
    for source, source_matrix in basis.items():
        transformed = gate.conj().T @ source_matrix @ gate
        coefficients: dict[str, complex] = {}
        for target, target_matrix in basis.items():
            coefficient = np.trace(target_matrix @ transformed) / 4.0
            if abs(coefficient) > tolerance:
                coefficients[target] = complex(coefficient)
        mapping[source] = coefficients
    return mapping


def conjugate_terms_by_adjacent_gate(
    terms: PauliTerms,
    gate: np.ndarray,
    left: int,
    *,
    tolerance: float = 1.0e-13,
) -> PauliTerms:
    """Conjugate a sparse full-system Pauli sum by one adjacent gate."""

    mapping = local_pauli_conjugation_map(gate, tolerance=tolerance / 10.0)
    output: defaultdict[str, complex] = defaultdict(complex)
    for label, coefficient in terms.items():
        local_source = label[left : left + 2]
        if local_source == "II":
            output[label] += coefficient
            continue
        for local_target, local_coefficient in mapping[local_source].items():
            target = label[:left] + local_target + label[left + 2 :]
            output[target] += coefficient * local_coefficient
    return {
        label: coefficient
        for label, coefficient in output.items()
        if abs(coefficient) > tolerance
    }


def conjugate_terms_by_layers(
    terms: PauliTerms,
    odd_gates: dict[int, np.ndarray],
    even_gates: dict[int, np.ndarray],
    *,
    tolerance: float = 1.0e-13,
) -> PauliTerms:
    """Apply ``W^dagger A W`` for ``W=U_odd U_even``.

    The Hamiltonian is first conjugated by ``U_odd`` and then by ``U_even``.
    """

    result = dict(terms)
    for left in sorted(odd_gates):
        result = conjugate_terms_by_adjacent_gate(
            result, odd_gates[left], left, tolerance=tolerance
        )
    for left in sorted(even_gates):
        result = conjugate_terms_by_adjacent_gate(
            result, even_gates[left], left, tolerance=tolerance
        )
    return result


def corrected_residual_terms(
    params: BenchmarkParameters,
    lam: float,
    *,
    alpha_pair: float = 1.0,
    alpha_exchange: float = 1.0,
    resonant_policy: str = "raise",
    output_core_fields: np.ndarray | None = None,
    tolerance: float = 1.0e-13,
) -> PauliTerms:
    """Exact Pauli decomposition of ``W^dagger(H_C+lambda V_C)W-H_C``."""

    input_terms = add_terms(
        core_pauli_terms(params),
        deformation_pauli_terms(params, lam),
        tolerance=tolerance,
    )
    odd_gates, even_gates = correction_gate_layers(
        params,
        lam,
        alpha_pair=alpha_pair,
        alpha_exchange=alpha_exchange,
        resonant_policy=resonant_policy,
    )
    reduced = conjugate_terms_by_layers(
        input_terms, odd_gates, even_gates, tolerance=tolerance
    )
    return add_terms(
        reduced,
        scale_terms(core_pauli_terms(params, output_core_fields), -1.0),
        tolerance=tolerance,
    )


def direct_angle_residual_terms(
    params: BenchmarkParameters,
    lam: float,
    theta_pair: float,
    theta_exchange: float,
    *,
    output_core_fields: np.ndarray | None = None,
    tolerance: float = 1.0e-13,
) -> PauliTerms:
    """Exact residual for a correction specified by finite channel angles.

    This is the direct-angle analogue of :func:`corrected_residual_terms`.  It
    remains well defined at ``params.delta == 0`` because neither variational
    angle contains a perturbative energy denominator.
    """

    input_terms = add_terms(
        core_pauli_terms(params),
        deformation_pauli_terms(params, lam),
        tolerance=tolerance,
    )
    odd_gates, even_gates = direct_angle_gate_layers(
        params, theta_pair, theta_exchange
    )
    reduced = conjugate_terms_by_layers(
        input_terms, odd_gates, even_gates, tolerance=tolerance
    )
    return add_terms(
        reduced,
        scale_terms(core_pauli_terms(params, output_core_fields), -1.0),
        tolerance=tolerance,
    )


def bare_residual_terms(params: BenchmarkParameters, lam: float) -> PauliTerms:
    return deformation_pauli_terms(params, lam)


def pauli_incident_strength(terms: PauliTerms) -> float:
    """Return the Pauli incident strength of the displayed decomposition.

    For the equal-support grouping used in the manuscript, this quantity is a
    computable upper bound on the intrinsic centered incident strength; the two
    quantities need not be equal.
    """

    if not terms:
        return 0.0
    n_sites = len(next(iter(terms)))
    incident = np.zeros(n_sites, dtype=float)
    for label, coefficient in terms.items():
        if set(label) == {"I"}:
            continue
        for site, pauli in enumerate(label):
            if pauli != "I":
                incident[site] += abs(coefficient)
    return float(np.max(incident))


def normalized_hilbert_schmidt_density(terms: PauliTerms) -> float:
    """Return ``||A||_2/sqrt(N 2^N)`` using Pauli orthogonality."""

    if not terms:
        return 0.0
    n_sites = len(next(iter(terms)))
    return float(np.sqrt(sum(abs(c) ** 2 for c in terms.values()) / n_sites))


def support_statistics(terms: PauliTerms) -> dict[str, float]:
    max_size = 0
    max_diameter = 0
    for label, coefficient in terms.items():
        if abs(coefficient) == 0.0:
            continue
        support = [site for site, pauli in enumerate(label) if pauli != "I"]
        if not support:
            continue
        max_size = max(max_size, len(support))
        max_diameter = max(max_diameter, support[-1] - support[0])
    return {
        "term_count": float(len(terms)),
        "max_support_size": float(max_size),
        "max_support_diameter": float(max_diameter),
    }


def coefficient_diagnostics(terms: PauliTerms) -> dict[str, float]:
    return {
        "max_imaginary_coefficient": float(
            max((abs(c.imag) for c in terms.values()), default=0.0)
        ),
        "identity_coefficient": float(
            abs(terms.get("I" * len(next(iter(terms))), 0.0)) if terms else 0.0
        ),
    }


def terms_to_dense(terms: PauliTerms) -> np.ndarray:
    if not terms:
        raise ValueError("Cannot infer matrix dimension from an empty decomposition.")
    n_sites = len(next(iter(terms)))
    result = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    for label, coefficient in terms.items():
        result += coefficient * pauli_matrix(label)
    return result

