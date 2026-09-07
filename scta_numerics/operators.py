"""Small dense-operator utilities with an explicit qubit-order convention.

Qubit 0 is the leftmost (most significant) tensor factor.  A computational
basis index therefore encodes ``|q_0 q_1 ... q_{N-1}>`` in ordinary binary
order.  The convention is used consistently by the dense and Pauli-string
implementations.
"""

from __future__ import annotations

from functools import reduce
from typing import Iterable, Sequence

import numpy as np


I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
HAD = (X + Z) / np.sqrt(2.0)
SP = (X + 1.0j * Y) / 2.0
SM = (X - 1.0j * Y) / 2.0

PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}
PAULI_LABELS = ("I", "X", "Y", "Z")


def kron_all(factors: Iterable[np.ndarray]) -> np.ndarray:
    """Kronecker product of ``factors`` in their displayed order."""

    factors = list(factors)
    if not factors:
        return np.array([[1.0]], dtype=complex)
    return reduce(np.kron, factors)


def pauli_matrix(label: str) -> np.ndarray:
    """Return the dense matrix of a full Pauli label such as ``"IXZ"``."""

    return kron_all(PAULI[p] for p in label)


def onsite_operator(op: np.ndarray, site: int, n_sites: int) -> np.ndarray:
    """Embed a one-qubit operator at ``site``."""

    if not 0 <= site < n_sites:
        raise IndexError(site)
    factors = [I2] * n_sites
    factors[site] = op
    return kron_all(factors)


def adjacent_operator(op: np.ndarray, left_site: int, n_sites: int) -> np.ndarray:
    """Embed a two-qubit operator on ``(left_site, left_site + 1)``."""

    if op.shape != (4, 4):
        raise ValueError("The embedded adjacent operator must be 4 x 4.")
    if not 0 <= left_site < n_sites - 1:
        raise IndexError(left_site)
    factors: list[np.ndarray] = []
    site = 0
    while site < n_sites:
        if site == left_site:
            factors.append(op)
            site += 2
        else:
            factors.append(I2)
            site += 1
    return kron_all(factors)


def disjoint_adjacent_layer(
    gates: dict[int, np.ndarray], n_sites: int
) -> np.ndarray:
    """Build one layer of pairwise-disjoint adjacent two-qubit gates.

    ``gates[i]`` acts on sites ``(i, i+1)``.  Overlapping keys are rejected.
    """

    starts = set(gates)
    if any(i + 1 in starts for i in starts):
        raise ValueError("Gate supports in one layer must be disjoint.")
    factors: list[np.ndarray] = []
    site = 0
    while site < n_sites:
        if site in gates:
            gate = gates[site]
            if gate.shape != (4, 4):
                raise ValueError("Every layer gate must be 4 x 4.")
            factors.append(gate)
            site += 2
        else:
            factors.append(I2)
            site += 1
    return kron_all(factors)


def partial_trace(rho: np.ndarray, keep: Sequence[int], n_sites: int) -> np.ndarray:
    """Trace all qubits except ``keep`` from an ``N``-qubit density matrix."""

    keep = tuple(sorted(set(keep)))
    if any(site < 0 or site >= n_sites for site in keep):
        raise IndexError("A retained site lies outside the system.")
    traced = [site for site in range(n_sites) if site not in keep]
    tensor = rho.reshape([2] * (2 * n_sites))
    # Trace from the highest site downward so the remaining axis indices stay
    # valid after each contraction.
    current_n = n_sites
    for site in sorted(traced, reverse=True):
        tensor = np.trace(tensor, axis1=site, axis2=site + current_n)
        current_n -= 1
    dim = 2 ** len(keep)
    return tensor.reshape((dim, dim))


def trace_norm_hermitian(op: np.ndarray) -> float:
    """Trace norm of a Hermitian matrix, with numerical symmetrization."""

    hermitian = (op + op.conj().T) / 2.0
    return float(np.sum(np.abs(np.linalg.eigvalsh(hermitian))))


def density_diagnostics(rho: np.ndarray) -> dict[str, float]:
    """Return normalization, Hermiticity, and positivity diagnostics."""

    hermiticity = float(np.linalg.norm(rho - rho.conj().T, ord="fro"))
    trace_error = float(abs(np.trace(rho) - 1.0))
    min_eigenvalue = float(np.min(np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)))
    return {
        "trace_error": trace_error,
        "hermiticity_error": hermiticity,
        "min_eigenvalue": min_eigenvalue,
    }

