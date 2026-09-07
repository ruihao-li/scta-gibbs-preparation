"""Alternating-field graph-stabilizer benchmark in the syndrome core frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp

from .operators import (
    HAD,
    I2,
    SM,
    SP,
    X,
    Z,
    adjacent_operator,
    disjoint_adjacent_layer,
    kron_all,
    onsite_operator,
)


FIELD_CONVENTION = "h_i = h_bar + (-1)^i delta for one-based i"


@dataclass(frozen=True)
class BenchmarkParameters:
    """Local parameters for the open alternating-field chain."""

    n_sites: int
    h_bar: float = 1.25
    delta: float = 0.25
    coupling: float = 1.0

    def __post_init__(self) -> None:
        if self.n_sites < 2:
            raise ValueError("The open-chain benchmark requires at least two sites.")
        if not all(np.isfinite(value) for value in (self.h_bar, self.delta, self.coupling)):
            raise ValueError("All model parameters must be finite.")

    def fields(self) -> np.ndarray:
        """Return ``h_i=h_bar+(-1)^i delta`` for one-based site labels."""

        one_based = np.arange(1, self.n_sites + 1)
        return self.h_bar + ((-1.0) ** one_based) * self.delta

def core_hamiltonian(params: BenchmarkParameters, fields: np.ndarray | None = None) -> np.ndarray:
    """Construct ``H_C,0=-sum_i h_i Z_i``."""

    h = params.fields() if fields is None else np.asarray(fields, dtype=float)
    if h.shape != (params.n_sites,):
        raise ValueError("The field array must have one entry per site.")
    dim = 2 ** params.n_sites
    result = np.zeros((dim, dim), dtype=complex)
    for site, value in enumerate(h):
        result -= value * onsite_operator(Z, site, params.n_sites)
    return result


def deformation_hamiltonian(params: BenchmarkParameters) -> np.ndarray:
    """Construct ``V_C=J sum_i X_i X_{i+1}`` with open boundaries."""

    dim = 2 ** params.n_sites
    result = np.zeros((dim, dim), dtype=complex)
    xx = np.kron(X, X)
    for left in range(params.n_sites - 1):
        result += params.coupling * adjacent_operator(xx, left, params.n_sites)
    return result


def bond_generator_components(
    h_left: float,
    h_right: float,
    coupling: float,
    *,
    resonant_policy: Literal["raise", "omit"] = "raise",
    tolerance: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pair and exchange pieces of the anti-Hermitian bond generator.

    The sum obeys ``[H_C, S_i] = -J X_i X_{i+1}`` whenever both transition
    frequencies are nonzero.  With ``resonant_policy='omit'``, only a channel
    whose denominator vanishes is omitted.
    """

    pair_direction, exchange_direction = bond_channel_directions()

    def divided(direction: np.ndarray, denominator: float, name: str) -> np.ndarray:
        if abs(denominator) <= tolerance:
            if resonant_policy == "omit":
                return np.zeros((4, 4), dtype=complex)
            raise ValueError(f"The {name} channel is resonant.")
        return coupling * direction / denominator

    pair = divided(pair_direction, 2.0 * (h_left + h_right), "pair")
    exchange = divided(exchange_direction, 2.0 * (h_left - h_right), "exchange")
    return pair, exchange


def bond_channel_directions() -> tuple[np.ndarray, np.ndarray]:
    """Return the denominator-free pair and exchange channel directions.

    Both matrices are anti-Hermitian.  The pair direction acts only on the
    even-parity subspace spanned by ``|00>`` and ``|11>``, whereas the exchange
    direction acts only on the odd-parity subspace spanned by ``|01>`` and
    ``|10>``.  They therefore commute exactly.
    """

    pair_direction = np.kron(SP, SP) - np.kron(SM, SM)
    exchange_direction = np.kron(SP, SM) - np.kron(SM, SP)
    return pair_direction, exchange_direction


def analytic_channel_angles(
    params: BenchmarkParameters,
    lam: float,
    *,
    resonance_tolerance: float = 1.0e-14,
) -> tuple[float, float, bool]:
    """Return direct pair/exchange angles for the analytic first-order seed.

    The exchange angle is the reference-orientation amplitude; its sign on an
    individual bond is supplied by :func:`direct_angle_gate_layers`.  At exact
    resonance the exchange channel has no perturbative inverse and is omitted,
    while the finite pair angle is retained.  The Boolean output records this
    omission explicitly.
    """

    pair_denominator = 4.0 * params.h_bar
    if abs(pair_denominator) <= resonance_tolerance:
        raise ValueError("The pair channel is resonant.")
    theta_pair = lam * params.coupling / pair_denominator

    exchange_denominator = 4.0 * params.delta
    exchange_omitted = abs(exchange_denominator) <= resonance_tolerance
    theta_exchange = (
        0.0
        if exchange_omitted
        else lam * params.coupling / exchange_denominator
    )
    return float(theta_pair), float(theta_exchange), exchange_omitted


def direct_angle_gate_layers(
    params: BenchmarkParameters,
    theta_pair: float,
    theta_exchange: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return brick-wall gates parameterized by finite channel angles.

    ``theta_exchange`` is a staggered amplitude.  With the field convention
    used by :class:`BenchmarkParameters`, the analytic exchange coefficient is
    negative on a zero-based even-left bond and positive on an odd-left bond.
    This orientation is retained at exact resonance, where the variational
    angle remains finite even though the perturbative denominator vanishes.
    """

    pair_direction, exchange_direction = bond_channel_directions()
    odd_gates: dict[int, np.ndarray] = {}
    even_gates: dict[int, np.ndarray] = {}
    for left in range(params.n_sites - 1):
        exchange_orientation = -1.0 if left % 2 == 0 else 1.0
        exponent = (
            float(theta_pair) * pair_direction
            + exchange_orientation * float(theta_exchange) * exchange_direction
        )
        gate = expm(exponent)
        (odd_gates if left % 2 == 0 else even_gates)[left] = gate
    return odd_gates, even_gates


def direct_angle_correction_unitary(
    params: BenchmarkParameters,
    theta_pair: float,
    theta_exchange: float,
) -> np.ndarray:
    """Construct the exact depth-two correction from direct channel angles."""

    odd_gates, even_gates = direct_angle_gate_layers(
        params, theta_pair, theta_exchange
    )
    odd_layer = disjoint_adjacent_layer(odd_gates, params.n_sites)
    even_layer = disjoint_adjacent_layer(even_gates, params.n_sites)
    return odd_layer @ even_layer


def bond_generator(
    h_left: float,
    h_right: float,
    coupling: float,
    *,
    alpha_pair: float = 1.0,
    alpha_exchange: float = 1.0,
    resonant_policy: Literal["raise", "omit"] = "raise",
) -> np.ndarray:
    pair, exchange = bond_generator_components(
        h_left,
        h_right,
        coupling,
        resonant_policy=resonant_policy,
    )
    return alpha_pair * pair + alpha_exchange * exchange


def correction_unitary(
    params: BenchmarkParameters,
    lam: float,
    *,
    alpha_pair: float = 1.0,
    alpha_exchange: float = 1.0,
    resonant_policy: Literal["raise", "omit"] = "raise",
) -> np.ndarray:
    """Construct ``W=e^{lambda S_odd} e^{lambda S_even}`` exactly."""

    odd_gates, even_gates = correction_gate_layers(
        params,
        lam,
        alpha_pair=alpha_pair,
        alpha_exchange=alpha_exchange,
        resonant_policy=resonant_policy,
    )
    odd_layer = disjoint_adjacent_layer(odd_gates, params.n_sites)
    even_layer = disjoint_adjacent_layer(even_gates, params.n_sites)
    return odd_layer @ even_layer


def correction_gate_layers(
    params: BenchmarkParameters,
    lam: float,
    *,
    alpha_pair: float = 1.0,
    alpha_exchange: float = 1.0,
    resonant_policy: Literal["raise", "omit"] = "raise",
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return the odd- and even-bond gates before embedding."""

    h = params.fields()
    odd_gates: dict[int, np.ndarray] = {}
    even_gates: dict[int, np.ndarray] = {}
    for left in range(params.n_sites - 1):
        generator = bond_generator(
            h[left],
            h[left + 1],
            params.coupling,
            alpha_pair=alpha_pair,
            alpha_exchange=alpha_exchange,
            resonant_policy=resonant_policy,
        )
        gate = expm(lam * generator)
        # A one-based odd bond has an even zero-based left endpoint.
        (odd_gates if left % 2 == 0 else even_gates)[left] = gate
    return odd_gates, even_gates


def graph_tail(n_sites: int) -> np.ndarray:
    """Return ``U_G=(prod CZ)(prod Hadamard)`` for the open chain graph."""

    hadamards = kron_all([HAD] * n_sites)
    phases = np.ones(2**n_sites, dtype=complex)
    for basis in range(2**n_sites):
        bits = [(basis >> (n_sites - 1 - site)) & 1 for site in range(n_sites)]
        parity = sum(bits[site] * bits[site + 1] for site in range(n_sites - 1))
        phases[basis] = -1.0 if parity % 2 else 1.0
    return phases[:, None] * hadamards


def gibbs_state(hamiltonian: np.ndarray, beta: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Return the Gibbs state, log partition function, and energy levels."""

    if beta <= 0.0:
        raise ValueError("The inverse temperature beta must be positive.")
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("The Hamiltonian must be a square matrix.")

    energies, vectors = np.linalg.eigh((hamiltonian + hamiltonian.conj().T) / 2.0)
    log_z = float(logsumexp(-beta * energies))
    probabilities = np.exp(-beta * energies - log_z)
    rho = (vectors * probabilities[None, :]) @ vectors.conj().T
    return rho, log_z, energies


def product_core_state(fields: np.ndarray, beta: float) -> tuple[np.ndarray, float]:
    """Gibbs state and entropy of ``-sum_i fields[i] Z_i``."""

    factors = []
    entropy = 0.0
    for field in np.asarray(fields, dtype=float):
        p_zero = 1.0 / (1.0 + np.exp(-2.0 * beta * field))
        p_one = 1.0 - p_zero
        factors.append(np.diag([p_zero, p_one]).astype(complex))
        for probability in (p_zero, p_one):
            if probability > 0.0:
                entropy -= probability * np.log(probability)
    return kron_all(factors), float(entropy)


def loader_angle_from_field(field: float | np.ndarray, beta: float) -> np.ndarray:
    """Convert the field in ``-field * Z`` to its Gibbs-loader angle.

    The convention is ``R_y(2 alpha)|0> = cos(alpha)|0> + sin(alpha)|1>``.
    For the Gibbs state at inverse temperature ``beta``, this gives
    ``tan(alpha) = exp(-beta * field)``.
    """

    if beta <= 0.0:
        raise ValueError("The inverse temperature beta must be positive.")
    values = np.asarray(field, dtype=float)
    return np.arctan(np.exp(-beta * values))


def field_from_loader_angle(angle: float | np.ndarray, beta: float) -> np.ndarray:
    """Return the product-core field represented by a Gibbs-loader angle."""

    if beta <= 0.0:
        raise ValueError("The inverse temperature beta must be positive.")
    values = np.asarray(angle, dtype=float)
    if np.any(values <= 0.0) or np.any(values >= np.pi / 2.0):
        raise ValueError("Every loader angle must lie strictly between 0 and pi/2.")
    return np.log(1.0 / np.tan(values)) / beta


def product_core_state_from_angles(
    angles: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Product-core state and entropy prepared by direct loader angles."""

    values = np.asarray(angles, dtype=float)
    if np.any(values < 0.0) or np.any(values > np.pi / 2.0):
        raise ValueError("Every loader angle must lie between 0 and pi/2.")
    factors = []
    entropy = 0.0
    for angle in values:
        p_zero = float(np.cos(angle) ** 2)
        p_one = float(np.sin(angle) ** 2)
        factors.append(np.diag([p_zero, p_one]).astype(complex))
        for probability in (p_zero, p_one):
            if probability > 0.0:
                entropy -= probability * np.log(probability)
    return kron_all(factors), float(entropy)


def exact_core_frame_hamiltonian(params: BenchmarkParameters, lam: float) -> np.ndarray:
    """Return ``H_C,0 + lambda V_C`` for the benchmark chain."""

    return core_hamiltonian(params) + lam * deformation_hamiltonian(params)
