"""Variational continuation through resonance and finite deformation."""

from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .benchmarks import (
    OUTPUT_SCHEMA_VERSION,
    _require_integer,
    environment_metadata,
    free_energy,
    max_contiguous_local_error,
    output_hashes,
    write_csv,
    write_json,
)
from .model import (
    BenchmarkParameters,
    FIELD_CONVENTION,
    analytic_channel_angles,
    direct_angle_correction_unitary,
    exact_core_frame_hamiltonian,
    field_from_loader_angle,
    gibbs_state,
    graph_tail,
    loader_angle_from_field,
    product_core_state_from_angles,
)
from .operators import density_diagnostics
from .pauli import (
    coefficient_diagnostics,
    direct_angle_residual_terms,
    normalized_hilbert_schmidt_density,
    pauli_incident_strength,
    support_statistics,
)


METHODS = ("bare", "analytic_first_order", "variational")


def validate_variational_config(config: dict[str, Any]) -> None:
    """Validate the parameter domain and scan grids used in Sec. V.B."""

    model = config["model"]
    settings = config["variational_scans"]
    if model.get("boundary") != "open":
        raise ValueError("The reported variational benchmark uses open boundaries.")
    if model.get("field_convention") != FIELD_CONVENTION:
        raise ValueError(f"field_convention must be {FIELD_CONVENTION!r}.")
    _require_integer(settings["N"], "variational_scans.N", 2)
    if float(settings["beta"]) <= 0.0:
        raise ValueError("variational_scans.beta must be positive.")
    _require_integer(settings["random_starts"], "random_starts", 0)
    _require_integer(settings["random_seed"], "random_seed", 0)
    _require_integer(settings["maxiter"], "maxiter", 1)
    _require_integer(settings.get("maxls", 50), "maxls", 1)
    for name in (
        "ftol",
        "gtol",
        "objective_comparison_atol",
        "objective_comparison_rtol",
    ):
        if not np.isfinite(float(settings[name])) or float(settings[name]) <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    angle_bounds = tuple(map(float, settings["angle_bounds"]))
    if (
        len(angle_bounds) != 2
        or not all(np.isfinite(angle_bounds))
        or not angle_bounds[0] < angle_bounds[1]
    ):
        raise ValueError("angle_bounds must contain increasing lower and upper bounds.")
    _validated_core_angle_bounds(settings)

    resonance_deltas = np.asarray(settings["resonance_delta_values"], dtype=float)
    resonance_plot = np.asarray(
        settings["resonance_plot_delta_values"], dtype=float
    )
    coupling_lambdas = np.asarray(settings["coupling_lambda_values"], dtype=float)
    coupling_plot = np.asarray(
        settings["coupling_plot_lambda_values"], dtype=float
    )
    if resonance_deltas.size == 0 or np.any(~np.isfinite(resonance_deltas)):
        raise ValueError("resonance_delta_values must be a nonempty finite grid.")
    if len(resonance_deltas) != len(set(map(float, resonance_deltas))):
        raise ValueError("resonance_delta_values must not contain duplicates.")
    if np.any(resonance_deltas < 0.0):
        raise ValueError("This benchmark scans nonnegative field staggerings.")
    if (
        coupling_lambdas.size == 0
        or np.any(~np.isfinite(coupling_lambdas))
        or np.any(coupling_lambdas <= 0.0)
    ):
        raise ValueError("coupling_lambda_values must be a nonempty positive grid.")
    if len(coupling_lambdas) != len(set(map(float, coupling_lambdas))):
        raise ValueError("coupling_lambda_values must not contain duplicates.")
    for values, name in (
        (resonance_plot, "resonance_plot_delta_values"),
        (coupling_plot, "coupling_plot_lambda_values"),
    ):
        if values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must be a nonempty finite grid.")
        if len(values) != len(set(map(float, values))):
            raise ValueError(f"{name} must not contain duplicates.")
    if float(settings["resonance_lambda"]) <= 0.0:
        raise ValueError("resonance_lambda must be positive.")
    if float(settings["coupling_delta"]) <= 0.0:
        raise ValueError("coupling_delta must be positive.")
    if not all(
        np.isclose(value, resonance_deltas).any() for value in resonance_plot
    ):
        raise ValueError("Every plotted resonance point must belong to the raw grid.")
    if not all(np.isclose(value, coupling_lambdas).any() for value in coupling_plot):
        raise ValueError("Every plotted coupling point must belong to the raw grid.")
    tolerance = float(config["numerics"]["pauli_prune_tolerance"])
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("pauli_prune_tolerance must be finite and positive.")

    for point in scan_points(settings):
        params = BenchmarkParameters(
            int(settings["N"]),
            float(model["h_bar"]),
            float(point["delta"]),
            float(model["J"]),
        )
        parameter_starts(
            params,
            float(point["lambda"]),
            settings,
            int(point["point_index"]),
        )


def _validated_core_angle_bounds(settings: dict[str, Any]) -> tuple[float, float]:
    """Return loader-angle bounds strictly inside the physical open interval."""

    bounds = tuple(map(float, settings["core_angle_bounds"]))
    if len(bounds) != 2:
        raise ValueError("core_angle_bounds must contain exactly two values.")
    lower, upper = bounds
    if not 0.0 < lower < upper < np.pi / 2.0:
        raise ValueError(
            "core_angle_bounds must satisfy 0 < lower < upper < pi/2."
        )
    return lower, upper


def alternating_loader_angles(
    n_sites: int, alpha_odd: float, alpha_even: float
) -> np.ndarray:
    """Return odd/even loader angles in zero-based site order."""

    return np.array(
        [alpha_odd if site % 2 == 0 else alpha_even for site in range(n_sites)],
        dtype=float,
    )


def direct_angle_objective_factory(
    params: BenchmarkParameters,
    beta: float,
    h_target: np.ndarray,
):
    """Construct the exact free-energy-density objective for four parameters."""

    cache: dict[tuple[float, ...], float] = {}

    def objective(theta: np.ndarray) -> float:
        key = tuple(map(float, theta))
        if key in cache:
            return cache[key]
        theta_pair, theta_exchange, alpha_odd, alpha_even = key
        angles = alternating_loader_angles(
            params.n_sites, alpha_odd, alpha_even
        )
        rho_core, entropy = product_core_state_from_angles(angles)
        correction = direct_angle_correction_unitary(
            params, theta_pair, theta_exchange
        )
        rho = correction @ rho_core @ correction.conj().T
        value = free_energy(rho, entropy, h_target, beta) / params.n_sites
        cache[key] = float(value)
        return float(value)

    return objective


def analytic_seed(
    params: BenchmarkParameters, lam: float, beta: float
) -> tuple[np.ndarray, str]:
    """Return the analytic deterministic seed and its resonance status."""

    theta_pair, theta_exchange, exchange_omitted = analytic_channel_angles(
        params, lam
    )
    angles = loader_angle_from_field(params.fields(), beta)
    status = "pair_only_at_resonance" if exchange_omitted else "full_nonresonant"
    return (
        np.array(
            [theta_pair, theta_exchange, float(angles[0]), float(angles[1])],
            dtype=float,
        ),
        status,
    )


def parameter_starts(
    params: BenchmarkParameters,
    lam: float,
    settings: dict[str, Any],
    point_index: int,
) -> tuple[list[tuple[str, np.ndarray]], str]:
    """Return two deterministic and exactly the configured random starts."""

    angle_bounds = tuple(map(float, settings["angle_bounds"]))
    core_angle_bounds = _validated_core_angle_bounds(settings)
    bounds = [angle_bounds, angle_bounds, core_angle_bounds, core_angle_bounds]
    beta = float(settings["beta"])
    analytic, analytic_status = analytic_seed(params, lam, beta)
    bare_angles = loader_angle_from_field(params.fields(), beta)
    bare = np.array([0.0, 0.0, bare_angles[0], bare_angles[1]])

    for value, (low, high) in zip(analytic, bounds):
        if not low <= value <= high:
            raise ValueError(
                "The analytic seed lies outside the configured parameter bounds."
            )

    starts: list[tuple[str, np.ndarray]] = [
        ("analytic_seed", analytic),
        ("bare_seed", bare.astype(float)),
    ]
    seed = np.random.SeedSequence(
        [int(settings["random_seed"]), int(point_index)]
    )
    rng = np.random.default_rng(seed)
    for start_index in range(int(settings["random_starts"])):
        random_start = np.array(
            [rng.uniform(low, high) for low, high in bounds], dtype=float
        )
        starts.append((f"random_{start_index:02d}", random_start))
    return starts, analytic_status


def scan_points(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Return unique physical points and their plot-scan memberships."""

    declared: list[tuple[str, float, float]] = []
    resonance_lambda = float(settings["resonance_lambda"])
    for delta in settings["resonance_delta_values"]:
        declared.append(("resonance", float(delta), resonance_lambda))
    coupling_delta = float(settings["coupling_delta"])
    for lam in settings["coupling_lambda_values"]:
        declared.append(("coupling", coupling_delta, float(lam)))

    unique: list[dict[str, Any]] = []
    lookup: dict[tuple[float, float], dict[str, Any]] = {}
    for scan, delta, lam in declared:
        key = (delta, lam)
        if key not in lookup:
            point = {
                "point_id": f"point_{len(unique):03d}",
                "point_index": len(unique),
                "delta": delta,
                "lambda": lam,
                "scans": [],
            }
            lookup[key] = point
            unique.append(point)
        lookup[key]["scans"].append(scan)
    return unique


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _completed_points(
    points: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    optimizer_rows: list[dict[str, Any]],
    starts: int,
) -> set[str]:
    result_records: dict[str, set[tuple[str, str]]] = {}
    optimizer_counts: dict[str, int] = {}
    for row in result_rows:
        result_records.setdefault(str(row["point_id"]), set()).add(
            (str(row["scan"]), str(row["method"]))
        )
    for row in optimizer_rows:
        point_id = str(row["point_id"])
        optimizer_counts[point_id] = optimizer_counts.get(point_id, 0) + 1
    completed: set[str] = set()
    for point in points:
        point_id = str(point["point_id"])
        expected = {
            (str(scan), method) for scan in point["scans"] for method in METHODS
        }
        if (
            result_records.get(point_id, set()) == expected
            and optimizer_counts.get(point_id, 0) == starts
        ):
            completed.add(point_id)
    return completed


def _validate_checkpoint_rows(
    points: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    optimizer_rows: list[dict[str, Any]],
    expected_start_names: set[str],
) -> None:
    """Check checkpoint identities, coordinates, memberships, and uniqueness."""

    declared = {str(point["point_id"]): point for point in points}

    def validate_identity(row: dict[str, Any], table: str) -> dict[str, Any]:
        point_id = str(row["point_id"])
        if point_id not in declared:
            raise ValueError(f"The {table} checkpoint contains unknown {point_id}.")
        point = declared[point_id]
        coordinate = (float(row["delta"]), float(row["lambda"]))
        expected = (float(point["delta"]), float(point["lambda"]))
        if coordinate != expected:
            raise ValueError(
                f"The {table} checkpoint maps {point_id} to {coordinate}, "
                f"but the configuration declares {expected}."
            )
        return point

    for row in result_rows:
        point = validate_identity(row, "results")
        if str(row["scan"]) not in set(map(str, point["scans"])):
            raise ValueError("The results checkpoint contains an invalid scan membership.")
        if str(row["method"]) not in METHODS:
            raise ValueError("The results checkpoint contains an unknown method.")
    for row in optimizer_rows:
        validate_identity(row, "optimizer")
        if str(row["start"]) not in expected_start_names:
            raise ValueError("The optimizer checkpoint contains an unknown start name.")

    result_keys = [
        (str(row["point_id"]), str(row["scan"]), str(row["method"]))
        for row in result_rows
    ]
    optimizer_keys = [
        (str(row["point_id"]), str(row["start"])) for row in optimizer_rows
    ]
    if len(result_keys) != len(set(result_keys)):
        raise ValueError("The results checkpoint contains duplicate rows.")
    if len(optimizer_keys) != len(set(optimizer_keys)):
        raise ValueError("The optimizer checkpoint contains duplicate rows.")


def _discard_incomplete_points(
    result_rows: list[dict[str, Any]],
    optimizer_rows: list[dict[str, Any]],
    completed: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove partial point records so an interrupted point can be rerun safely."""

    seen = {
        str(row["point_id"]) for row in result_rows + optimizer_rows
    }
    incomplete = seen - completed
    if not incomplete:
        return result_rows, optimizer_rows
    return (
        [row for row in result_rows if str(row["point_id"]) not in incomplete],
        [row for row in optimizer_rows if str(row["point_id"]) not in incomplete],
    )


def _evaluate_method(
    *,
    point: dict[str, Any],
    params: BenchmarkParameters,
    beta: float,
    h_target: np.ndarray,
    physical_exact: np.ndarray,
    exact_free_energy_density: float,
    graph_unitary: np.ndarray,
    method: str,
    theta: np.ndarray,
    winning_start: str,
    analytic_status: str,
    tolerance: float,
) -> dict[str, Any]:
    theta_pair, theta_exchange, alpha_odd, alpha_even = map(float, theta)
    angles = alternating_loader_angles(
        params.n_sites, alpha_odd, alpha_even
    )
    fields = field_from_loader_angle(angles, beta)
    rho_core, entropy = product_core_state_from_angles(angles)
    correction = direct_angle_correction_unitary(
        params, theta_pair, theta_exchange
    )
    rho_approx_core = correction @ rho_core @ correction.conj().T
    physical_approx = graph_unitary @ rho_approx_core @ graph_unitary.conj().T
    free_energy_density = (
        free_energy(rho_approx_core, entropy, h_target, beta) / params.n_sites
    )
    local_error, argmax_left = max_contiguous_local_error(
        physical_exact, physical_approx, params.n_sites, 2
    )
    residual = direct_angle_residual_terms(
        params,
        float(point["lambda"]),
        theta_pair,
        theta_exchange,
        output_core_fields=fields,
        tolerance=tolerance,
    )
    residual_support = support_statistics(residual)
    residual_coefficients = coefficient_diagnostics(residual)
    return {
        "point_id": point["point_id"],
        "N": params.n_sites,
        "beta": beta,
        "h_bar": params.h_bar,
        "delta": params.delta,
        "J": params.coupling,
        "lambda": point["lambda"],
        "method": method,
        "analytic_status": analytic_status,
        "free_energy_density": free_energy_density,
        "exact_free_energy_density": exact_free_energy_density,
        "free_energy_gap_density": free_energy_density - exact_free_energy_density,
        "physical_two_site_trace_norm_max": local_error,
        "two_site_argmax_left": argmax_left,
        "residual_incident_strength": pauli_incident_strength(residual),
        "residual_normalized_hs_density": normalized_hilbert_schmidt_density(
            residual
        ),
        "residual_term_count": residual_support["term_count"],
        "residual_max_support_size": residual_support["max_support_size"],
        "residual_max_support_diameter": residual_support[
            "max_support_diameter"
        ],
        "residual_max_imaginary_coefficient": residual_coefficients[
            "max_imaginary_coefficient"
        ],
        "residual_identity_coefficient": residual_coefficients[
            "identity_coefficient"
        ],
        "theta_pair": theta_pair,
        "theta_exchange": theta_exchange,
        "core_alpha_odd": alpha_odd,
        "core_alpha_even": alpha_even,
        "core_alpha_odd_shift": alpha_odd
        - float(loader_angle_from_field(params.fields()[0], beta)),
        "core_alpha_even_shift": alpha_even
        - float(loader_angle_from_field(params.fields()[1], beta)),
        "winning_start": winning_start if method == "variational" else "",
        "eta_pair": abs(float(point["lambda"]) * params.coupling)
        / (4.0 * abs(params.h_bar)),
        "eta_exchange": (
            np.inf
            if params.delta == 0.0
            else abs(float(point["lambda"]) * params.coupling)
            / (4.0 * abs(params.delta))
        ),
        **density_diagnostics(physical_approx),
    }


def run_variational_point(
    point: dict[str, Any],
    model: dict[str, Any],
    settings: dict[str, Any],
    graph_unitary: np.ndarray,
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Optimize and evaluate one unique ``(delta, lambda)`` point."""

    params = BenchmarkParameters(
        int(settings["N"]),
        float(model["h_bar"]),
        float(point["delta"]),
        float(model["J"]),
    )
    beta = float(settings["beta"])
    lam = float(point["lambda"])
    h_target = exact_core_frame_hamiltonian(params, lam)
    rho_exact_core, log_z, _ = gibbs_state(h_target, beta)
    exact_free_energy_density = -log_z / (beta * params.n_sites)
    physical_exact = graph_unitary @ rho_exact_core @ graph_unitary.conj().T
    objective = direct_angle_objective_factory(params, beta, h_target)

    starts, analytic_status = parameter_starts(
        params, lam, settings, int(point["point_index"])
    )
    core_angle_bounds = _validated_core_angle_bounds(settings)
    bounds = [
        tuple(map(float, settings["angle_bounds"])),
        tuple(map(float, settings["angle_bounds"])),
        core_angle_bounds,
        core_angle_bounds,
    ]
    optimizer_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for start_name, start in starts:
        started = time.perf_counter()
        optimized = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": int(settings["maxiter"]),
                "ftol": float(settings["ftol"]),
                "gtol": float(settings["gtol"]),
                "maxls": int(settings.get("maxls", 50)),
            },
        )
        elapsed = time.perf_counter() - started
        jac_norm = (
            float(np.linalg.norm(optimized.jac))
            if optimized.jac is not None
            else np.nan
        )
        row = {
            "point_id": point["point_id"],
            "delta": point["delta"],
            "lambda": point["lambda"],
            "analytic_status": analytic_status,
            "start": start_name,
            "start_theta_pair": start[0],
            "start_theta_exchange": start[1],
            "start_core_alpha_odd": start[2],
            "start_core_alpha_even": start[3],
            "success": bool(optimized.success),
            "status": int(optimized.status),
            "iterations": int(optimized.nit),
            "function_evaluations": int(optimized.nfev),
            "elapsed_seconds": elapsed,
            "objective_density": float(optimized.fun),
            "gradient_norm": jac_norm,
            "theta_pair": float(optimized.x[0]),
            "theta_exchange": float(optimized.x[1]),
            "core_alpha_odd": float(optimized.x[2]),
            "core_alpha_even": float(optimized.x[3]),
            "message": str(optimized.message),
        }
        optimizer_rows.append(row)
        if np.isfinite(optimized.fun):
            candidates.append((float(optimized.fun), optimized.x.copy(), row))

    successful = [candidate for candidate in candidates if candidate[2]["success"]]
    eligible = successful if successful else candidates
    if not eligible:
        raise RuntimeError(f"All optimizer runs failed at {point['point_id']}.")
    eligible.sort(key=lambda item: item[0])
    _, best_theta, best_log = eligible[0]

    analytic_theta, _ = analytic_seed(params, lam, beta)
    bare_angles = loader_angle_from_field(params.fields(), beta)
    bare_theta = np.array(
        [0.0, 0.0, float(bare_angles[0]), float(bare_angles[1])]
    )
    method_parameters = {
        "bare": bare_theta,
        "analytic_first_order": analytic_theta,
        "variational": np.asarray(best_theta, dtype=float),
    }
    base_rows = [
        _evaluate_method(
            point=point,
            params=params,
            beta=beta,
            h_target=h_target,
            physical_exact=physical_exact,
            exact_free_energy_density=exact_free_energy_density,
            graph_unitary=graph_unitary,
            method=method,
            theta=theta,
            winning_start=str(best_log["start"]),
            analytic_status=analytic_status,
            tolerance=tolerance,
        )
        for method, theta in method_parameters.items()
    ]

    result_rows: list[dict[str, Any]] = []
    for scan in point["scans"]:
        for row in base_rows:
            result_rows.append({**row, "scan": scan})
    return result_rows, optimizer_rows


def smoke_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """Return a cheap configuration that still exercises resonance and ten starts."""

    smoke = deepcopy(config)
    settings = smoke["variational_scans"]
    settings["N"] = 4
    settings["resonance_delta_values"] = [0.25, 0.0]
    settings["resonance_plot_delta_values"] = [0.25]
    settings["coupling_lambda_values"] = [0.05]
    settings["coupling_plot_lambda_values"] = [0.05]
    settings["maxiter"] = min(int(settings["maxiter"]), 100)
    return smoke


def run_variational_scans(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    repository_root: Path,
    *,
    resume: bool = False,
) -> None:
    """Run the checkpointed resonance and finite-deformation scans."""

    validate_variational_config(config)
    settings = config["variational_scans"]
    model = config["model"]
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_path = raw_dir / "scan_results.csv"
    optimizer_path = raw_dir / "optimizer_runs.csv"
    metadata_path = raw_dir / "run_metadata.json"
    existing = [
        path
        for path in (results_path, optimizer_path, metadata_path)
        if path.exists()
    ]
    if not resume and existing:
        raise FileExistsError(
            f"Variational output already exists under {raw_dir}; use --resume or "
            "choose another output directory."
        )

    result_rows = _load_rows(results_path) if resume else []
    optimizer_rows = _load_rows(optimizer_path) if resume else []
    if resume and (result_rows or optimizer_rows) and not metadata_path.exists():
        raise ValueError("Cannot resume checkpoint rows without run_metadata.json.")
    previous_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if resume and metadata_path.exists()
        else {}
    )
    current_metadata = environment_metadata(config_path, repository_root, config)
    if previous_metadata:
        if previous_metadata.get("status") not in {"running", "complete"}:
            raise ValueError("Checkpoint metadata has an unsupported run status.")
        if previous_metadata.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
            raise ValueError("Refusing to resume a different output schema version.")
        if previous_metadata.get("effective_config") != config:
            raise ValueError(
                "Refusing to resume with a different effective configuration."
            )
        if previous_metadata.get("execution_file_sha256") != current_metadata.get(
            "execution_file_sha256"
        ):
            raise ValueError(
                "Refusing to resume after the recorded source files changed."
            )
        for package in (
            "python",
            "platform",
            "numpy",
            "numpy_build_configuration",
            "scipy",
        ):
            if previous_metadata.get(package) != current_metadata.get(package):
                raise ValueError(
                    f"Refusing to resume with a different {package} version."
                )
        checkpoint_paths = [results_path, optimizer_path]
        recorded_hashes = previous_metadata.get("output_file_sha256", {})
        actual_hashes = output_hashes(checkpoint_paths)
        if recorded_hashes != actual_hashes:
            if previous_metadata.get("status") == "complete":
                raise ValueError("Refusing to resume modified checkpoint CSV files.")
            for path in checkpoint_paths:
                path.unlink(missing_ok=True)
            result_rows = []
            optimizer_rows = []
            previous_metadata["output_file_sha256"] = {}
    previous_elapsed = float(previous_metadata.get("elapsed_seconds", 0.0))
    expected_start_names = {"analytic_seed", "bare_seed"} | {
        f"random_{index:02d}" for index in range(int(settings["random_starts"]))
    }
    expected_starts = len(expected_start_names)
    points = scan_points(settings)
    _validate_checkpoint_rows(
        points, result_rows, optimizer_rows, expected_start_names
    )
    completed = _completed_points(
        points, result_rows, optimizer_rows, expected_starts
    )
    result_rows, optimizer_rows = _discard_incomplete_points(
        result_rows, optimizer_rows, completed
    )
    graph_unitary = graph_tail(int(settings["N"]))
    tolerance = float(config["numerics"]["pauli_prune_tolerance"])
    run_started = time.perf_counter()

    if previous_metadata:
        metadata = dict(previous_metadata)
        resume_events = list(metadata.get("resume_events", []))
        resume_events.append(
            {
                "resumed_at_utc": current_metadata["generated_at_utc"],
                "git_commit": current_metadata["git_commit"],
                "repository_status_short": current_metadata[
                    "repository_status_short"
                ],
            }
        )
        metadata["resume_events"] = resume_events
    else:
        metadata = current_metadata
        metadata["resume_events"] = []
        metadata["output_file_sha256"] = {}
    metadata.update(
        {
            "experiment": "variational_resonance_and_deformation",
            "status": "running",
            "completed_points": len(completed),
            "total_unique_points": len(points),
            "elapsed_seconds": previous_elapsed,
            "incremental_elapsed_seconds": 0.0,
            "output_file_sha256": output_hashes(
                [results_path, optimizer_path]
            ),
        }
    )
    write_json(metadata_path, metadata)

    for ordinal, point in enumerate(points, start=1):
        if point["point_id"] in completed:
            print(
                f"[{ordinal}/{len(points)}] skipping completed {point['point_id']} "
                f"(delta={point['delta']}, lambda={point['lambda']})",
                flush=True,
            )
            continue
        point_started = time.perf_counter()
        print(
            f"[{ordinal}/{len(points)}] optimizing {point['point_id']} "
            f"(delta={point['delta']}, lambda={point['lambda']})",
            flush=True,
        )
        new_results, new_optimizer_rows = run_variational_point(
            point, model, settings, graph_unitary, tolerance
        )
        result_rows.extend(new_results)
        optimizer_rows.extend(new_optimizer_rows)
        write_csv(results_path, result_rows)
        write_csv(optimizer_path, optimizer_rows)
        completed.add(str(point["point_id"]))
        metadata["completed_points"] = len(completed)
        metadata["elapsed_seconds"] = (
            previous_elapsed + time.perf_counter() - run_started
        )
        metadata["incremental_elapsed_seconds"] = time.perf_counter() - run_started
        metadata["output_file_sha256"] = output_hashes(
            [results_path, optimizer_path]
        )
        write_json(metadata_path, metadata)
        print(
            f"    completed in {time.perf_counter() - point_started:.1f} s",
            flush=True,
        )

    metadata["status"] = "complete"
    metadata["completed_points"] = len(completed)
    metadata["elapsed_seconds"] = previous_elapsed + time.perf_counter() - run_started
    metadata["incremental_elapsed_seconds"] = time.perf_counter() - run_started
    metadata["output_file_sha256"] = output_hashes([results_path, optimizer_path])
    write_json(metadata_path, metadata)
