"""Deterministic residual-scaling and Gibbs-state benchmarks.

This module implements the two nonvariational studies reported in Sec. V.A of
the manuscript.  Sparse Pauli propagation is used for the system-size scan,
whereas dense exact diagonalization is restricted to the N=8 Gibbs-state
comparisons and small algebraic checks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .model import (
    BenchmarkParameters,
    FIELD_CONVENTION,
    core_hamiltonian,
    correction_unitary,
    deformation_hamiltonian,
    gibbs_state,
    graph_tail,
    product_core_state,
)
from .operators import Z, adjacent_operator, density_diagnostics, partial_trace
from .operators import trace_norm_hermitian
from .pauli import (
    bare_residual_terms,
    coefficient_diagnostics,
    corrected_residual_terms,
    normalized_hilbert_schmidt_density,
    pauli_incident_strength,
    support_statistics,
    terms_to_dense,
)


BENCHMARK_OUTPUTS = (
    "residual_scaling.csv",
    "algebra_checks.csv",
    "pauli_tolerance_sweep.csv",
    "thermal_accuracy.csv",
    "run_metadata.json",
)
OUTPUT_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_hashes(paths: list[Path]) -> dict[str, str]:
    """Return filename-keyed hashes for the existing output files in ``paths``."""

    return {path.name: sha256_file(path) for path in paths if path.is_file()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write dictionaries while retaining first-seen column order."""

    if not rows:
        raise ValueError(f"No rows supplied for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write a JSON object with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def max_contiguous_local_error(
    rho: np.ndarray,
    sigma: np.ndarray,
    n_sites: int,
    region_size: int,
) -> tuple[float, int]:
    """Return the largest unhalved trace distance over contiguous regions."""

    if not 1 <= region_size <= n_sites:
        raise ValueError("region_size must lie between one and n_sites.")
    best_value = -1.0
    best_left = -1
    for left in range(n_sites - region_size + 1):
        keep = range(left, left + region_size)
        reduced_difference = partial_trace(rho - sigma, keep, n_sites)
        value = trace_norm_hermitian(reduced_difference)
        if value > best_value:
            best_value = value
            best_left = left
    return best_value, best_left


def expectation(rho: np.ndarray, observable: np.ndarray) -> float:
    """Return the real expectation value of a Hermitian observable."""

    return float(np.real(np.trace(rho @ observable)))


def max_physical_bond_observable_error(
    rho: np.ndarray,
    sigma: np.ndarray,
    n_sites: int,
) -> tuple[float, int]:
    """Return the largest nearest-neighbor ZZ expectation-value error."""

    zz = np.kron(Z, Z)
    best_value = -1.0
    best_left = -1
    for left in range(n_sites - 1):
        observable = adjacent_operator(zz, left, n_sites)
        value = abs(expectation(rho - sigma, observable))
        if value > best_value:
            best_value = value
            best_left = left
    return best_value, best_left


def free_energy(
    rho: np.ndarray,
    entropy: float,
    hamiltonian: np.ndarray,
    beta: float,
) -> float:
    """Return ``Tr(rho H) - S(rho)/beta``."""

    if beta <= 0.0:
        raise ValueError("The inverse temperature beta must be positive.")
    return expectation(rho, hamiltonian) - entropy / beta


def _require_finite_number(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _require_integer(value: Any, name: str, minimum: int) -> int:
    """Return an integer-valued input no smaller than ``minimum``."""

    number = _require_finite_number(value, name)
    if not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(number)


def _require_positive_sequence(
    values: Any, name: str, *, unique: bool = False
) -> list[float]:
    result = [_require_finite_number(value, name) for value in values]
    if not result or any(value <= 0.0 for value in result):
        raise ValueError(f"{name} must be a nonempty sequence of positive values.")
    if unique and len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicate values.")
    return result


def _validate_fit_window(
    settings: dict[str, Any], values: list[float], label: str
) -> None:
    lower = _require_finite_number(settings["fit_lambda_min"], f"{label} fit minimum")
    upper = _require_finite_number(settings["fit_lambda_max"], f"{label} fit maximum")
    if not 0.0 < lower <= upper:
        raise ValueError(f"{label} fit window must satisfy 0 < minimum <= maximum.")
    if sum(lower <= value <= upper for value in values) < 2:
        raise ValueError(f"{label} fit window must contain at least two grid points.")


def validate_benchmark_config(config: dict[str, Any]) -> None:
    """Validate assumptions required by the nonresonant benchmark."""

    model = config["model"]
    if model.get("boundary") != "open":
        raise ValueError("The reported benchmark uses open boundaries.")
    if model.get("field_convention") != FIELD_CONVENTION:
        raise ValueError(f"field_convention must be {FIELD_CONVENTION!r}.")
    h_bar = _require_finite_number(model["h_bar"], "model.h_bar")
    delta = _require_finite_number(model["delta"], "model.delta")
    _require_finite_number(model["J"], "model.J")
    if abs(h_bar) <= 1.0e-14 or abs(delta) <= 1.0e-14:
        raise ValueError("The first-order scaling benchmark must be nonresonant.")

    residual = config["residual_scaling"]
    n_values = [
        _require_integer(value, "residual_scaling.N_values", 2)
        for value in residual["N_values"]
    ]
    if not n_values or len(n_values) != len(set(n_values)):
        raise ValueError("residual_scaling.N_values must be nonempty and unique.")
    _require_integer(residual["primary_N"], "residual_scaling.primary_N", 2)
    residual_lambdas = _require_positive_sequence(
        residual["lambda_values"], "residual lambda values", unique=True
    )
    _validate_fit_window(residual, residual_lambdas, "residual")
    _require_positive_sequence(
        residual["dense_check_lambdas"],
        "dense-check lambda values",
        unique=True,
    )
    if _require_finite_number(residual["size_lambda"], "size_lambda") <= 0.0:
        raise ValueError("residual_scaling.size_lambda must be positive.")

    thermal = config["thermal_accuracy"]
    _require_integer(thermal["N"], "thermal_accuracy.N", 2)
    _require_positive_sequence(
        thermal["beta_values"], "inverse temperatures", unique=True
    )
    thermal_lambdas = _require_positive_sequence(
        thermal["lambda_values"], "thermal lambda values", unique=True
    )
    _validate_fit_window(thermal, thermal_lambdas, "thermal")

    numerics = config["numerics"]
    if _require_finite_number(
        numerics["pauli_prune_tolerance"], "pauli_prune_tolerance"
    ) <= 0.0:
        raise ValueError("pauli_prune_tolerance must be positive.")
    _require_positive_sequence(
        numerics["tolerance_sweep_values"],
        "tolerance-sweep tolerances",
        unique=True,
    )
    _require_positive_sequence(
        numerics["tolerance_sweep_lambdas"],
        "tolerance-sweep lambda values",
        unique=True,
    )
    _require_integer(numerics["tolerance_sweep_N"], "tolerance_sweep_N", 2)
    for name in ("tolerance_relative_spread_max", "algebra_tolerance"):
        if _require_finite_number(numerics[name], name) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    for name in ("fit_linear_band", "fit_quadratic_band"):
        band = [_require_finite_number(value, name) for value in numerics[name]]
        if len(band) != 2 or band[0] > band[1]:
            raise ValueError(f"{name} must contain an increasing pair of values.")


def smoke_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """Return a cheap configuration that exercises every benchmark pathway."""

    smoke = json.loads(json.dumps(config))
    residual = smoke["residual_scaling"]
    residual["N_values"] = [4, 6]
    residual["primary_N"] = 4
    residual["lambda_values"] = residual["lambda_values"][:3]
    residual["dense_check_lambdas"] = residual["dense_check_lambdas"][:1]
    thermal = smoke["thermal_accuracy"]
    thermal["N"] = 4
    thermal["beta_values"] = [1.0]
    thermal["lambda_values"] = thermal["lambda_values"][:3]
    smoke["numerics"]["tolerance_sweep_N"] = 6
    smoke["numerics"]["tolerance_sweep_lambdas"] = [0.05]
    return smoke


def _residual_row(
    params: BenchmarkParameters,
    lam: float,
    method: str,
    tolerance: float,
) -> dict[str, Any]:
    terms = (
        bare_residual_terms(params, lam)
        if method.startswith("bare")
        else corrected_residual_terms(params, lam, tolerance=tolerance)
    )
    row: dict[str, Any] = {
        "N": params.n_sites,
        "h_bar": params.h_bar,
        "delta": params.delta,
        "J": params.coupling,
        "lambda": lam,
        "method": method,
        "incident_strength": pauli_incident_strength(terms),
        "normalized_hs_density": normalized_hilbert_schmidt_density(terms),
    }
    row.update(support_statistics(terms))
    row.update(coefficient_diagnostics(terms))
    return row


def run_residual_scaling(config: dict[str, Any], raw_dir: Path) -> None:
    """Generate the coupling, size, algebra, and pruning-tolerance scans."""

    model = config["model"]
    settings = config["residual_scaling"]
    tolerance = float(config["numerics"]["pauli_prune_tolerance"])
    primary_params = BenchmarkParameters(
        int(settings["primary_N"]),
        float(model["h_bar"]),
        float(model["delta"]),
        float(model["J"]),
    )
    rows: list[dict[str, Any]] = []
    for lam in settings["lambda_values"]:
        rows.append(_residual_row(primary_params, float(lam), "bare", tolerance))
        rows.append(
            _residual_row(primary_params, float(lam), "first_order", tolerance)
        )
    for n_sites in settings["N_values"]:
        params = BenchmarkParameters(
            int(n_sites),
            float(model["h_bar"]),
            float(model["delta"]),
            float(model["J"]),
        )
        lam = float(settings["size_lambda"])
        rows.append(_residual_row(params, lam, "bare_size_scan", tolerance))
        rows.append(_residual_row(params, lam, "first_order_size_scan", tolerance))
    write_csv(raw_dir / "residual_scaling.csv", rows)

    check_params = BenchmarkParameters(
        4,
        float(model["h_bar"]),
        float(model["delta"]),
        float(model["J"]),
    )
    h0 = core_hamiltonian(check_params)
    deformation = deformation_hamiltonian(check_params)
    checks: list[dict[str, Any]] = []
    for lam_value in settings["dense_check_lambdas"]:
        lam = float(lam_value)
        correction = correction_unitary(check_params, lam)
        dense = correction.conj().T @ (h0 + lam * deformation) @ correction - h0
        sparse = terms_to_dense(
            corrected_residual_terms(check_params, lam, tolerance=tolerance)
        )
        scale = float(np.linalg.norm(dense, ord="fro"))
        if scale <= np.finfo(float).tiny:
            raise FloatingPointError("The dense residual norm vanished unexpectedly.")
        checks.append(
            {
                "lambda": lam,
                "relative_pauli_reconstruction_error": float(
                    np.linalg.norm(dense - sparse, ord="fro") / scale
                ),
                "residual_hermiticity_error": float(
                    np.linalg.norm(dense - dense.conj().T, ord="fro") / scale
                ),
                "unitarity_error": float(
                    np.linalg.norm(
                        correction.conj().T @ correction
                        - np.eye(correction.shape[0]),
                        ord="fro",
                    )
                ),
            }
        )
    write_csv(raw_dir / "algebra_checks.csv", checks)

    tolerance_params = BenchmarkParameters(
        int(config["numerics"]["tolerance_sweep_N"]),
        float(model["h_bar"]),
        float(model["delta"]),
        float(model["J"]),
    )
    tolerance_rows: list[dict[str, Any]] = []
    for lam_value in config["numerics"]["tolerance_sweep_lambdas"]:
        for prune_value in config["numerics"]["tolerance_sweep_values"]:
            prune_tolerance = float(prune_value)
            row = _residual_row(
                tolerance_params,
                float(lam_value),
                "first_order_tolerance_sweep",
                prune_tolerance,
            )
            row["prune_tolerance"] = prune_tolerance
            tolerance_rows.append(row)
    write_csv(raw_dir / "pauli_tolerance_sweep.csv", tolerance_rows)


def run_thermal_accuracy(config: dict[str, Any], raw_dir: Path) -> None:
    """Generate exact endpoint Gibbs-state and local-observable comparisons."""

    model = config["model"]
    settings = config["thermal_accuracy"]
    params = BenchmarkParameters(
        int(settings["N"]),
        float(model["h_bar"]),
        float(model["delta"]),
        float(model["J"]),
    )
    h0 = core_hamiltonian(params)
    deformation = deformation_hamiltonian(params)
    graph_unitary = graph_tail(params.n_sites)
    rows: list[dict[str, Any]] = []

    for beta_value in settings["beta_values"]:
        beta = float(beta_value)
        rho0, _ = product_core_state(params.fields(), beta)
        physical_bare = graph_unitary @ rho0 @ graph_unitary.conj().T
        for lam_value in settings["lambda_values"]:
            lam = float(lam_value)
            h_target = h0 + lam * deformation
            rho_exact_core, _, _ = gibbs_state(h_target, beta)
            correction = correction_unitary(params, lam)
            rho_corrected_core = correction @ rho0 @ correction.conj().T
            physical_exact = graph_unitary @ rho_exact_core @ graph_unitary.conj().T
            physical_corrected = (
                graph_unitary @ rho_corrected_core @ graph_unitary.conj().T
            )

            h_reduced = correction.conj().T @ h_target @ correction
            rho_reduced, _, _ = gibbs_state(h_reduced, beta)
            covariance_reconstruction = correction @ rho_reduced @ correction.conj().T
            covariance_error = trace_norm_hermitian(
                covariance_reconstruction - rho_exact_core
            )

            for method, approximate in (
                ("bare", physical_bare),
                ("first_order", physical_corrected),
            ):
                one_site, one_left = max_contiguous_local_error(
                    physical_exact, approximate, params.n_sites, 1
                )
                two_site, two_left = max_contiguous_local_error(
                    physical_exact, approximate, params.n_sites, 2
                )
                bond_error, bond_left = max_physical_bond_observable_error(
                    physical_exact, approximate, params.n_sites
                )
                rows.append(
                    {
                        "N": params.n_sites,
                        "beta": beta,
                        "lambda": lam,
                        "method": method,
                        "physical_one_site_trace_norm_max": one_site,
                        "one_site_argmax_left": one_left,
                        "physical_two_site_trace_norm_max": two_site,
                        "two_site_argmax_left": two_left,
                        "physical_ZZ_error_max": bond_error,
                        "ZZ_argmax_left": bond_left,
                        "gibbs_covariance_trace_error": covariance_error,
                        **density_diagnostics(approximate),
                    }
                )
    write_csv(raw_dir / "thermal_accuracy.csv", rows)


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _numpy_build_configuration() -> str:
    """Return a stable record of NumPy's build configuration."""

    config = getattr(np.__config__, "CONFIG", None)
    if config is None:
        return "unavailable"
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def environment_metadata(
    config_path: Path,
    repository_root: Path,
    effective_config: dict[str, Any],
) -> dict[str, Any]:
    """Record software versions, Git state, configuration, and code hashes."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    try:
        repository_status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        repository_status = ["unavailable"]

    numerics_root = Path(__file__).resolve().parents[1]
    execution_files = [
        config_path,
        numerics_root / "requirements.txt",
        *sorted(numerics_root.glob("*.py")),
        *sorted((numerics_root / "scta_numerics").glob("*.py")),
        *sorted((numerics_root / "tests").glob("*.py")),
    ]
    execution_hashes: dict[str, str] = {}
    for path in execution_files:
        if not path.is_file():
            continue
        try:
            label = str(path.relative_to(numerics_root))
        except ValueError:
            label = f"external_config/{path.name}"
        execution_hashes[label] = sha256_file(path)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_build_configuration": _numpy_build_configuration(),
        **_package_versions(),
        "git_commit": commit,
        "repository_status_short": repository_status,
        "execution_file_sha256": execution_hashes,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "config_file": config_path.name,
        "effective_config": effective_config,
    }


def run_benchmarks(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    repository_root: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Run the two deterministic benchmark groups and write raw results."""

    validate_benchmark_config(config)
    raw_dir = output_dir / "raw"
    existing = [raw_dir / name for name in BENCHMARK_OUTPUTS if (raw_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Benchmark output already exists ({names}); pass --overwrite or "
            "choose another output directory."
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = raw_dir / "run_metadata.json"
    started = time.perf_counter()
    metadata = environment_metadata(config_path, repository_root, config)
    metadata.update({"experiment": "scaling_and_thermal", "status": "running"})
    write_json(metadata_path, metadata)
    run_residual_scaling(config, raw_dir)
    run_thermal_accuracy(config, raw_dir)
    generated_paths = [
        raw_dir / name for name in BENCHMARK_OUTPUTS if name != "run_metadata.json"
    ]
    metadata.update(
        {
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "output_file_sha256": output_hashes(generated_paths),
        }
    )
    write_json(metadata_path, metadata)


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a JSON object.")
    return config
