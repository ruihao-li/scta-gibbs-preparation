#!/usr/bin/env python3
"""Analyze saved SCTA results, generate manuscript figures, and verify claims.

This script deliberately contains no Hamiltonian simulation.  It reads the raw
CSV files produced by ``run_benchmarks.py`` and ``run_variational.py`` so that
execution and analysis are independently
reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scta_numerics.analysis_figures import generate_manuscript_figures


HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "benchmarks" / "raw"
COMBINED_RAW = HERE / "results" / "variational" / "raw"
PROCESSED = HERE / "results" / "processed"
FIGURES = HERE / "figures"
BENCHMARK_CONFIG_PATH = HERE / "benchmark_config.json"
VARIATIONAL_CONFIG_PATH = HERE / "variational_config.json"
LIVE_CONFIG: dict[str, Any] = {}
LIVE_COMBINED_CONFIG: dict[str, Any] = {}
RUN_METADATA: dict[str, Any] = {}
COMBINED_METADATA: dict[str, Any] = {}
CONFIG: dict[str, Any] = {}
COMBINED_CONFIG: dict[str, Any] = {}
OUTPUT_SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_hashes(
    metadata: dict[str, Any], raw_dir: Path, filenames: tuple[str, ...], label: str
) -> None:
    """Verify completed raw CSVs against their generation-time digests."""

    recorded = metadata.get("output_file_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(filenames):
        raise RuntimeError(f"The {label} output-hash record is incomplete.")
    for filename in filenames:
        path = raw_dir / filename
        if not path.is_file() or sha256(path) != recorded[filename]:
            raise RuntimeError(f"The {label} raw file {filename} failed its hash check.")


def data_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return settings for the deterministic data consumed by this analysis."""

    normalized = {
        key: json.loads(json.dumps(config.get(key, {})))
        for key in ("model", "residual_scaling", "thermal_accuracy", "numerics")
    }
    for settings in normalized.values():
        if isinstance(settings, dict):
            for key in tuple(settings):
                if "fit_lambda" in key:
                    settings.pop(key)
    return normalized


def configure_paths(
    results_root: Path,
    processed_output: Path,
    figures_output: Path,
    benchmark_config: Path,
    variational_config: Path,
) -> None:
    """Set paths and verify that raw data match the live configurations."""

    global RAW, COMBINED_RAW, PROCESSED, FIGURES
    global BENCHMARK_CONFIG_PATH, VARIATIONAL_CONFIG_PATH
    global LIVE_CONFIG, LIVE_COMBINED_CONFIG, RUN_METADATA, COMBINED_METADATA
    global CONFIG, COMBINED_CONFIG

    candidate_raw = results_root / "benchmarks" / "raw"
    candidate_variational_raw = results_root / "variational" / "raw"
    outputs = (processed_output, figures_output)
    raw_inputs = (candidate_raw, candidate_variational_raw)
    for output in outputs:
        for raw_input in raw_inputs:
            if (
                output == raw_input
                or output in raw_input.parents
                or raw_input in output.parents
            ):
                raise ValueError("Analysis outputs must not overlap raw input roots.")
    if (
        processed_output == figures_output
        or processed_output in figures_output.parents
        or figures_output in processed_output.parents
    ):
        raise ValueError("Processed and figure output roots must not overlap.")

    RAW = candidate_raw
    COMBINED_RAW = candidate_variational_raw
    PROCESSED = processed_output
    FIGURES = figures_output
    BENCHMARK_CONFIG_PATH = benchmark_config
    VARIATIONAL_CONFIG_PATH = variational_config
    LIVE_CONFIG = json.loads(benchmark_config.read_text(encoding="utf-8"))
    LIVE_COMBINED_CONFIG = json.loads(variational_config.read_text(encoding="utf-8"))
    for key in ("boundary", "h_bar", "J", "field_convention"):
        if LIVE_CONFIG["model"].get(key) != LIVE_COMBINED_CONFIG["model"].get(key):
            raise RuntimeError(
                f"The benchmark and variational configurations disagree on model.{key}."
            )

    benchmark_metadata_path = RAW / "run_metadata.json"
    if not benchmark_metadata_path.exists():
        raise FileNotFoundError(
            "Run run_benchmarks.py before analyzing the saved results."
        )
    RUN_METADATA = json.loads(benchmark_metadata_path.read_text(encoding="utf-8"))
    if RUN_METADATA.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported deterministic benchmark output schema.")
    if RUN_METADATA.get("status") != "complete":
        raise RuntimeError("The deterministic benchmark run is not complete.")
    verify_raw_hashes(
        RUN_METADATA,
        RAW,
        (
            "residual_scaling.csv",
            "algebra_checks.csv",
            "pauli_tolerance_sweep.csv",
            "thermal_accuracy.csv",
        ),
        "deterministic benchmark",
    )
    if data_generation_config(
        RUN_METADATA.get("effective_config", {})
    ) != data_generation_config(LIVE_CONFIG):
        raise RuntimeError(
            "The live benchmark configuration differs from the one embedded in "
            "the raw run. Rerun the benchmarks before analyzing these data."
        )
    CONFIG = LIVE_CONFIG

    variational_metadata_path = COMBINED_RAW / "run_metadata.json"
    if not variational_metadata_path.exists():
        raise FileNotFoundError(
            "Run run_variational.py before analyzing the saved results."
        )
    COMBINED_METADATA = json.loads(
        variational_metadata_path.read_text(encoding="utf-8")
    )
    if COMBINED_METADATA.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported variational output schema.")
    if COMBINED_METADATA.get("effective_config") != LIVE_COMBINED_CONFIG:
        raise RuntimeError(
            "The live variational configuration differs from the one embedded in "
            "the raw run. Rerun the scans before analyzing these data."
        )
    if COMBINED_METADATA.get("status") != "complete":
        raise RuntimeError("The variational production run is not complete.")
    verify_raw_hashes(
        COMBINED_METADATA,
        COMBINED_RAW,
        ("scan_results.csv", "optimizer_runs.csv"),
        "variational",
    )
    COMBINED_CONFIG = COMBINED_METADATA["effective_config"]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / name, float_precision="round_trip")


def load_combined(name: str) -> pd.DataFrame:
    return pd.read_csv(COMBINED_RAW / name, float_precision="round_trip")


def configured_scan_data(data: pd.DataFrame, scan: str) -> pd.DataFrame:
    """Return the explicitly configured plotting subset of one combined scan."""

    settings = COMBINED_CONFIG["variational_scans"]
    if scan == "resonance":
        parameter = "delta"
        values = np.asarray(settings["resonance_plot_delta_values"], dtype=float)
    elif scan == "coupling":
        parameter = "lambda"
        values = np.asarray(settings["coupling_plot_lambda_values"], dtype=float)
    else:
        raise ValueError(f"Unknown combined scan {scan!r}.")
    coordinates = data[parameter].to_numpy(dtype=float)
    selected = np.isin(coordinates, values)
    return data[(data.scan == scan).to_numpy() & selected].copy()


def combined_plot_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return both plotted scans while retaining nonplotted raw diagnostics."""

    return pd.concat(
        [configured_scan_data(data, "resonance"), configured_scan_data(data, "coupling")],
        ignore_index=True,
    )


def _coordinate_key(values: tuple[Any, ...]) -> tuple[Any, ...]:
    """Encode strings and exact binary64 values for CSV/config comparison."""

    normalized: list[Any] = []
    for value in values:
        if isinstance(value, str):
            normalized.append(("string", value))
        else:
            normalized.append(("float", float(value).hex()))
    return tuple(normalized)


def _configured_variational_points(
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct the runner's ordered physical points from the configuration."""

    declared = [
        ("resonance", float(delta), float(settings["resonance_lambda"]))
        for delta in settings["resonance_delta_values"]
    ]
    declared.extend(
        ("coupling", float(settings["coupling_delta"]), float(lam))
        for lam in settings["coupling_lambda_values"]
    )
    points: list[dict[str, Any]] = []
    lookup: dict[tuple[float, float], dict[str, Any]] = {}
    for scan, delta, lam in declared:
        coordinate = (delta, lam)
        if coordinate not in lookup:
            point = {
                "point_id": f"point_{len(points):03d}",
                "delta": delta,
                "lambda": lam,
                "scans": [],
            }
            lookup[coordinate] = point
            points.append(point)
        lookup[coordinate]["scans"].append(scan)
    return points


def _shared_scan_payload_check(combined: pd.DataFrame) -> dict[str, Any]:
    """Require shared physical points to agree outside their scan label."""

    required = {"point_id", "method", "scan"}
    if not required.issubset(combined.columns):
        consistent = False
        shared_groups = 0
    else:
        consistent = True
        shared_groups = 0
        payload_columns = [column for column in combined.columns if column != "scan"]
        for _, group in combined.groupby(["point_id", "method"], dropna=False):
            if len(group) <= 1:
                continue
            shared_groups += 1
            if len(group.loc[:, payload_columns].drop_duplicates()) != 1:
                consistent = False
                break
    return {
        "aspect": "input",
        "check": "shared-scan payload consistency",
        "value": float(shared_groups),
        "criterion": "all duplicated point/method rows agree outside scan",
        "passed": bool(consistent),
    }


def _table_completeness_check(
    *,
    aspect: str,
    name: str,
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    expected: set[tuple[Any, ...]],
) -> dict[str, Any]:
    """Check an exact configured coordinate set and reject duplicate rows."""

    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        actual_keys: list[tuple[Any, ...]] = []
        passed = False
    else:
        actual_keys = [
            _coordinate_key(tuple(row))
            for row in frame.loc[:, list(columns)].itertuples(index=False, name=None)
        ]
        passed = len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected
    criterion = (
        f"exactly {len(expected)} configured rows with unique "
        + ", ".join(columns)
    )
    if missing_columns:
        criterion += "; required columns: " + ", ".join(missing_columns)
    return {
        "aspect": aspect,
        "check": name,
        "value": float(len(actual_keys)),
        "criterion": criterion,
        "passed": bool(passed),
    }


def configured_completeness_checks(
    *,
    residual: pd.DataFrame,
    algebra: pd.DataFrame,
    tolerance_sweep: pd.DataFrame,
    thermal: pd.DataFrame,
    combined: pd.DataFrame,
    optimizer_runs: pd.DataFrame,
    benchmark_config: dict[str, Any],
    variational_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return exact grid/schema checks for every raw table consumed here."""

    residual_settings = benchmark_config["residual_scaling"]
    primary_n = int(residual_settings["primary_N"])
    residual_expected = {
        _coordinate_key((primary_n, lam, method))
        for lam in residual_settings["lambda_values"]
        for method in ("bare", "first_order")
    }
    residual_expected.update(
        {
            _coordinate_key((n_sites, residual_settings["size_lambda"], method))
            for n_sites in residual_settings["N_values"]
            for method in ("bare_size_scan", "first_order_size_scan")
        }
    )

    algebra_expected = {
        _coordinate_key((lam,)) for lam in residual_settings["dense_check_lambdas"]
    }
    numerics = benchmark_config["numerics"]
    tolerance_expected = {
        _coordinate_key((lam, tolerance))
        for lam in numerics["tolerance_sweep_lambdas"]
        for tolerance in numerics["tolerance_sweep_values"]
    }

    thermal_settings = benchmark_config["thermal_accuracy"]
    thermal_n = int(thermal_settings["N"])
    thermal_expected = {
        _coordinate_key((thermal_n, beta, lam, method))
        for beta in thermal_settings["beta_values"]
        for lam in thermal_settings["lambda_values"]
        for method in ("bare", "first_order")
    }

    scan_settings = variational_config["variational_scans"]
    methods = ("bare", "analytic_first_order", "variational")
    physical_points = _configured_variational_points(scan_settings)
    combined_expected = {
        _coordinate_key(
            (point["point_id"], scan, point["delta"], point["lambda"], method)
        )
        for point in physical_points
        for scan in point["scans"]
        for method in methods
    }
    starts = ("analytic_seed", "bare_seed") + tuple(
        f"random_{index:02d}" for index in range(int(scan_settings["random_starts"]))
    )
    optimizer_expected = {
        _coordinate_key((point["point_id"], point["delta"], point["lambda"], start))
        for point in physical_points
        for start in starts
    }

    return [
        _table_completeness_check(
            aspect="input",
            name="residual configured-coordinate completeness",
            frame=residual,
            columns=("N", "lambda", "method"),
            expected=residual_expected,
        ),
        _table_completeness_check(
            aspect="input",
            name="algebra-check configured-coordinate completeness",
            frame=algebra,
            columns=("lambda",),
            expected=algebra_expected,
        ),
        _table_completeness_check(
            aspect="input",
            name="Pauli-tolerance configured-coordinate completeness",
            frame=tolerance_sweep,
            columns=("lambda", "prune_tolerance"),
            expected=tolerance_expected,
        ),
        _table_completeness_check(
            aspect="input",
            name="thermal configured-coordinate completeness",
            frame=thermal,
            columns=("N", "beta", "lambda", "method"),
            expected=thermal_expected,
        ),
        _table_completeness_check(
            aspect="input",
            name="variational-scan configured-coordinate completeness",
            frame=combined,
            columns=("point_id", "scan", "delta", "lambda", "method"),
            expected=combined_expected,
        ),
        _table_completeness_check(
            aspect="input",
            name="optimizer configured-coordinate completeness",
            frame=optimizer_runs,
            columns=("point_id", "delta", "lambda", "start"),
            expected=optimizer_expected,
        ),
        _shared_scan_payload_check(combined),
    ]


def power_law_fit(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    label: str,
    aspect: str,
    trim: str = "none",
) -> dict[str, Any]:
    selected = data[np.isfinite(data[x]) & np.isfinite(data[y])]
    selected = selected[(selected[x] > 0.0) & (selected[y] > 0.0)].sort_values(x)
    if x_min is not None:
        selected = selected[selected[x] >= x_min * (1.0 - 1.0e-9)]
    if x_max is not None:
        selected = selected[selected[x] <= x_max * (1.0 + 1.0e-9)]
    if trim == "drop_smallest" and len(selected) > 3:
        selected = selected.iloc[1:]
    elif trim == "drop_largest" and len(selected) > 3:
        selected = selected.iloc[:-1]
    if len(selected) < 3:
        raise ValueError(f"At least three points are required for {label}.")
    log_x = np.log(selected[x].to_numpy(dtype=float))
    log_y = np.log(selected[y].to_numpy(dtype=float))
    slope, intercept = np.polyfit(log_x, log_y, 1)
    prediction = intercept + slope * log_x
    residual_sum = float(np.sum((log_y - prediction) ** 2))
    total_sum = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    centered_log_x = log_x - np.mean(log_x)
    slope_standard_error = float(
        np.sqrt(
            residual_sum
            / (len(log_x) - 2)
            / np.sum(centered_log_x**2)
        )
    )
    return {
        "aspect": aspect,
        "label": label,
        "x": x,
        "y": y,
        "trim": trim,
        "n_points": len(selected),
        "x_min": float(selected[x].min()),
        "x_max": float(selected[x].max()),
        "exponent": float(slope),
        "exponent_standard_error": slope_standard_error,
        "log_prefactor": float(intercept),
        "prefactor": float(np.exp(intercept)),
        "r_squared": r_squared,
    }


def grouped_power_law_fit(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
) -> list[dict[str, Any]]:
    """Fit one exponent with an independent prefactor for each group."""
    selected = data[
        np.isfinite(data[x])
        & np.isfinite(data[y])
        & np.isfinite(data[group])
    ]
    selected = selected[(selected[x] > 0.0) & (selected[y] > 0.0)].copy()
    if x_min is not None:
        selected = selected[selected[x] >= x_min * (1.0 - 1.0e-9)]
    if x_max is not None:
        selected = selected[selected[x] <= x_max * (1.0 + 1.0e-9)]
    group_values = sorted(selected[group].unique())
    if len(selected) <= len(group_values) + 1:
        raise ValueError("Insufficient data for a grouped power-law fit.")

    log_x = np.log(selected[x].to_numpy(dtype=float))
    log_y = np.log(selected[y].to_numpy(dtype=float))
    group_array = selected[group].to_numpy()
    design = np.column_stack(
        [log_x]
        + [(group_array == value).astype(float) for value in group_values]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, log_y, rcond=None)
    prediction = design @ coefficients
    residual_sum = float(np.sum((log_y - prediction) ** 2))
    degrees_of_freedom = len(log_y) - rank
    residual_variance = residual_sum / degrees_of_freedom
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    slope_standard_error = float(np.sqrt(max(covariance[0, 0], 0.0)))
    total_sum = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0

    rows: list[dict[str, Any]] = []
    for index, value in enumerate(group_values):
        log_prefactor = float(coefficients[index + 1])
        rows.append(
            {
                "group": value,
                "n_points": len(selected),
                "n_groups": len(group_values),
                "exponent": float(coefficients[0]),
                "exponent_standard_error": slope_standard_error,
                "log_prefactor": log_prefactor,
                "prefactor": float(np.exp(log_prefactor)),
                "r_squared": r_squared,
            }
        )
    return rows


def add_fit_with_sensitivity(
    rows: list[dict[str, Any]],
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    x_min: float | None,
    x_max: float | None,
    label: str,
    aspect: str,
) -> None:
    for trim in ("none", "drop_smallest", "drop_largest"):
        rows.append(
            power_law_fit(
                data,
                x,
                y,
                x_min=x_min,
                x_max=x_max,
                label=label,
                aspect=aspect,
                trim=trim,
            )
        )


def fit_table() -> pd.DataFrame:
    fit_rows: list[dict[str, Any]] = []

    residual = load("residual_scaling.csv")
    rset = CONFIG["residual_scaling"]
    for method in ("bare", "first_order"):
        data = residual[residual.method == method]
        add_fit_with_sensitivity(
            fit_rows,
            data,
            "lambda",
            "incident_strength",
            x_min=rset["fit_lambda_min"],
            x_max=rset["fit_lambda_max"],
            label=f"residual_{method}",
            aspect="residual",
        )

    thermal = load("thermal_accuracy.csv")
    tset = CONFIG["thermal_accuracy"]
    for beta in tset["beta_values"]:
        for method in ("bare", "first_order"):
            data = thermal[(thermal.beta == beta) & (thermal.method == method)]
            label = f"thermal_beta_{beta:g}_{method}"
            add_fit_with_sensitivity(
                fit_rows,
                data,
                "lambda",
                "physical_two_site_trace_norm_max",
                x_min=tset["fit_lambda_min"],
                x_max=tset["fit_lambda_max"],
                label=label,
                aspect="thermal",
            )
            zz_label = f"thermal_ZZ_beta_{beta:g}_{method}"
            add_fit_with_sensitivity(
                fit_rows,
                data,
                "lambda",
                "physical_ZZ_error_max",
                x_min=tset["fit_lambda_min"],
                x_max=tset["fit_lambda_max"],
                label=zz_label,
                aspect="thermal_ZZ",
            )
    return pd.DataFrame(fit_rows)


def thermal_grouped_fit_table() -> pd.DataFrame:
    data = load("thermal_accuracy.csv")
    settings = CONFIG["thermal_accuracy"]
    metrics = {
        "D1": "physical_two_site_trace_norm_max",
        "ZZ": "physical_ZZ_error_max",
    }
    rows: list[dict[str, Any]] = []
    for metric_name, metric_column in metrics.items():
        for method in ("bare", "first_order"):
            selected = data[data.method == method]
            for row in grouped_power_law_fit(
                selected,
                "lambda",
                metric_column,
                "beta",
                x_min=settings["fit_lambda_min"],
                x_max=settings["fit_lambda_max"],
            ):
                rows.append(
                    {
                        "metric": metric_name,
                        "metric_column": metric_column,
                        "method": method,
                        "beta": row.pop("group"),
                        **row,
                    }
                )
    return pd.DataFrame(rows)


def nominal_exponent(fits: pd.DataFrame, label: str) -> float:
    return float(fits[(fits.label == label) & (fits.trim == "none")].iloc[0].exponent)


def fit_sensitivity(fits: pd.DataFrame, label: str) -> float:
    values = fits[fits.label == label].exponent.to_numpy(dtype=float)
    nominal = values[0]
    return float(np.max(np.abs(values[1:] - nominal)))


def optimizer_robustness() -> pd.DataFrame:
    runs = load_combined("optimizer_runs.csv")
    settings = COMBINED_CONFIG["variational_scans"]
    absolute_tolerance = float(settings["objective_comparison_atol"])
    relative_tolerance = float(settings["objective_comparison_rtol"])
    rows = []
    for point_id, group in runs.groupby("point_id"):
        best = float(group.objective_density.min())
        tolerance = max(absolute_tolerance, relative_tolerance * abs(best))
        near_best = np.abs(group.objective_density - best) <= tolerance
        rows.append(
            {
                "point_id": point_id,
                "delta": float(group.delta.iloc[0]),
                "lambda": float(group["lambda"].iloc[0]),
                "runs": len(group),
                "success_fraction": float(group.success.astype(bool).mean()),
                "near_best_fraction": float(near_best.mean()),
                "objective_spread": float(group.objective_density.max() - best),
                "best_gradient_norm": float(
                    group.loc[group.objective_density.idxmin(), "gradient_norm"]
                ),
            }
        )
    return pd.DataFrame(rows)


def crossover_summary(
    data: pd.DataFrame,
    *,
    scan: str,
    x: str,
    metric: str,
    descending: bool,
) -> dict[str, Any]:
    """Bracket the first analytic-to-bare crossover along a declared scan."""

    selected = data[data.scan == scan].pivot(
        index=x, columns="method", values=metric
    )
    selected = selected.sort_index(ascending=not descending)
    ratio = selected.analytic_first_order / selected.bare
    values = list(ratio.items())
    for (x0, r0), (x1, r1) in zip(values[:-1], values[1:]):
        if r0 < 1.0 <= r1:
            use_log = x0 > 0.0 and x1 > 0.0
            coordinate0 = np.log(x0) if use_log else x0
            coordinate1 = np.log(x1) if use_log else x1
            fraction = (1.0 - r0) / (r1 - r0) if r1 != r0 else 0.5
            coordinate = coordinate0 + fraction * (coordinate1 - coordinate0)
            estimate = float(np.exp(coordinate) if use_log else coordinate)
            return {
                "found": True,
                "safe_value": float(x0),
                "failed_value": float(x1),
                "bracket_min": float(min(x0, x1)),
                "bracket_max": float(max(x0, x1)),
                "estimate": estimate,
                "safe_ratio": float(r0),
                "failed_ratio": float(r1),
            }
    return {
        "found": False,
        "safe_value": None,
        "failed_value": None,
        "bracket_min": None,
        "bracket_max": None,
        "estimate": None,
        "safe_ratio": None,
        "failed_ratio": None,
    }


def build_verification(
    fits: pd.DataFrame,
    grouped_fits: pd.DataFrame,
    robustness: pd.DataFrame,
    completeness_checks: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    linear_low, linear_high = CONFIG["numerics"]["fit_linear_band"]
    quad_low, quad_high = CONFIG["numerics"]["fit_quadratic_band"]
    checks: list[dict[str, Any]] = list(completeness_checks)

    def add(aspect: str, check: str, value: float, criterion: str, passed: bool) -> None:
        checks.append(
            {
                "aspect": aspect,
                "check": check,
                "value": float(value),
                "criterion": criterion,
                "passed": bool(passed),
            }
        )

    bare_slope = nominal_exponent(fits, "residual_bare")
    corrected_slope = nominal_exponent(fits, "residual_first_order")
    add(
        "residual",
        "bare exponent",
        bare_slope,
        f"{linear_low:g} <= p <= {linear_high:g}",
        linear_low <= bare_slope <= linear_high,
    )
    add(
        "residual",
        "corrected exponent",
        corrected_slope,
        f"{quad_low:g} <= p <= {quad_high:g}",
        quad_low <= corrected_slope <= quad_high,
    )
    add(
        "residual",
        "corrected fit-window sensitivity",
        fit_sensitivity(fits, "residual_first_order"),
        "<= 0.1",
        fit_sensitivity(fits, "residual_first_order") <= 0.1,
    )
    algebra = load("algebra_checks.csv")
    algebra_max = float(
        algebra[
            [
                "relative_pauli_reconstruction_error",
                "residual_hermiticity_error",
                "unitarity_error",
            ]
        ].to_numpy().max()
    )
    algebra_tolerance = float(CONFIG["numerics"]["algebra_tolerance"])
    add(
        "residual",
        "algebraic reconstruction",
        algebra_max,
        f"< {algebra_tolerance:g}",
        algebra_max < algebra_tolerance,
    )
    tolerance_sweep = load("pauli_tolerance_sweep.csv")
    tolerance_spreads = []
    for _, group in tolerance_sweep.groupby("lambda"):
        values = group.incident_strength.to_numpy(dtype=float)
        tolerance_spreads.append((values.max() - values.min()) / values.mean())
    tolerance_spread = float(max(tolerance_spreads))
    tolerance_limit = float(CONFIG["numerics"]["tolerance_relative_spread_max"])
    add(
        "residual",
        "Pauli pruning-tolerance stability",
        tolerance_spread,
        f"< {tolerance_limit:g}",
        tolerance_spread < tolerance_limit,
    )
    residual = load("residual_scaling.csv")
    size = residual[
        (residual.method == "first_order_size_scan") & (residual.N >= 12)
    ].sort_values("N")
    displayed_values = size.incident_strength.round(7)
    displayed_spread = float(displayed_values.max() - displayed_values.min())
    add(
        "residual",
        "N>=12 displayed-precision saturation",
        displayed_spread,
        "= 0 after rounding epsilon_P to 7 decimal places",
        len(size) >= 2 and displayed_spread == 0.0,
    )
    scaling = residual[residual.method.isin(["bare", "first_order"])].pivot(
        index="lambda", columns="method", values="incident_strength"
    )
    small_improvement = float(
        (scaling.first_order.iloc[:3] < scaling.bare.iloc[:3]).mean()
    )
    add(
        "residual",
        "small-lambda improvement fraction",
        small_improvement,
        "= 1",
        small_improvement == 1.0,
    )

    thermal = load("thermal_accuracy.csv")
    for beta in CONFIG["thermal_accuracy"]["beta_values"]:
        b = nominal_exponent(fits, f"thermal_beta_{beta:g}_bare")
        c = nominal_exponent(fits, f"thermal_beta_{beta:g}_first_order")
        ok_b = linear_low <= b <= linear_high
        ok_c = quad_low <= c <= quad_high
        add(
            "thermal",
            f"beta={beta:g} bare exponent",
            b,
            f"{linear_low:g} <= p <= {linear_high:g}",
            ok_b,
        )
        add(
            "thermal",
            f"beta={beta:g} corrected exponent",
            c,
            f"{quad_low:g} <= p <= {quad_high:g}",
            ok_c,
        )

    grouped_specs = (
        ("D1", "bare", linear_low, linear_high),
        ("D1", "first_order", quad_low, quad_high),
        ("ZZ", "bare", linear_low, linear_high),
        ("ZZ", "first_order", quad_low + 1.0, quad_high + 1.0),
    )
    for metric, method, lower, upper in grouped_specs:
        selected = grouped_fits[
            (grouped_fits.metric == metric) & (grouped_fits.method == method)
        ]
        exponent = float(selected.exponent.iloc[0])
        consistent = bool(
            np.allclose(
                selected.exponent.to_numpy(dtype=float),
                exponent,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        add(
            "thermal",
            f"grouped {metric} {method} exponent",
            exponent,
            f"{lower:g} <= p <= {upper:g} and identical across prefactor rows",
            consistent and lower <= exponent <= upper,
        )
    covariance_max = float(thermal.gibbs_covariance_trace_error.max())
    add(
        "thermal",
        "Gibbs covariance identity",
        covariance_max,
        "< 1e-10",
        covariance_max < 1.0e-10,
    )
    state_error = float(
        max(
            thermal.trace_error.max(),
            thermal.hermiticity_error.max(),
            max(0.0, -thermal.min_eigenvalue.min()),
        )
    )
    add("thermal", "state diagnostics", state_error, "< 1e-10", state_error < 1.0e-10)

    combined = load_combined("scan_results.csv")
    plotted_combined = combined_plot_data(combined)
    unique = combined.drop_duplicates(["point_id", "method"])
    runs = load_combined("optimizer_runs.csv")
    settings = COMBINED_CONFIG["variational_scans"]
    expected_points = int(COMBINED_METADATA["total_unique_points"])
    expected_starts = 2 + int(settings["random_starts"])
    add(
        "combined",
        "embedded metadata matches live combined configuration",
        1.0,
        "= 1",
        COMBINED_METADATA.get("effective_config") == LIVE_COMBINED_CONFIG,
    )
    add(
        "combined",
        "unique physical parameter points",
        float(unique.point_id.nunique()),
        f"= {expected_points}",
        unique.point_id.nunique() == expected_points,
    )
    add(
        "combined",
        "total optimizer runs",
        float(len(runs)),
        f"= {expected_points * expected_starts}",
        len(runs) == expected_points * expected_starts,
    )
    current_schema = {
        "core_alpha_odd",
        "core_alpha_even",
    }.issubset(combined.columns) and not {"mu_A", "mu_B"}.intersection(combined.columns)
    add(
        "combined",
        "current loader-angle result schema",
        float(current_schema),
        "= 1",
        current_schema,
    )
    finite_objectives = bool(np.isfinite(runs.objective_density.to_numpy()).all())
    add(
        "combined",
        "finite optimizer objectives",
        float(finite_objectives),
        "= 1",
        finite_objectives,
    )

    complete_methods = unique.groupby("point_id").method.nunique()
    complete_method_fraction = float((complete_methods == 3).mean())
    add(
        "combined",
        "three methods at every unique point",
        complete_method_fraction,
        "= 1",
        len(complete_methods) == expected_points and complete_method_fraction == 1.0,
    )
    runs_per_point = runs.groupby("point_id").size()
    complete_start_fraction = float((runs_per_point == expected_starts).mean())
    add(
        "combined",
        "ten optimizer starts at every unique point",
        complete_start_fraction,
        "= 1",
        len(runs_per_point) == expected_points and complete_start_fraction == 1.0,
    )
    add(
        "combined",
        "minimum optimizer success fraction",
        float(robustness.success_fraction.min()),
        "= 1",
        float(robustness.success_fraction.min()) == 1.0,
    )
    state_error = float(
        max(
            unique.trace_error.max(),
            unique.hermiticity_error.max(),
            max(0.0, -unique.min_eigenvalue.min()),
        )
    )
    add(
        "combined",
        "state diagnostics",
        state_error,
        "< 1e-10",
        state_error < 1.0e-10,
    )
    residual_imaginary = float(unique.residual_max_imaginary_coefficient.max())
    add(
        "combined",
        "residual Hermiticity",
        residual_imaginary,
        "< 1e-10",
        residual_imaginary < 1.0e-10,
    )
    minimum_gap = float(unique.free_energy_gap_density.min())
    add(
        "combined",
        "nonnegative free-energy gap",
        minimum_gap,
        ">= -1e-12",
        minimum_gap >= -1.0e-12,
    )
    gap_pivot = unique.pivot(
        index="point_id", columns="method", values="free_energy_gap_density"
    )
    variational_excess = float(
        (
            gap_pivot.variational
            - gap_pivot[["bare", "analytic_first_order"]].min(axis=1)
        ).max()
    )
    add(
        "combined",
        "variational objective no worse than deterministic comparisons",
        variational_excess,
        "<= 1e-10",
        variational_excess <= 1.0e-10,
    )
    variational_rows = unique[unique.method == "variational"]
    alpha_lower, alpha_upper = map(float, settings["core_angle_bounds"])
    loader_bound_margin = float(
        min(
            variational_rows[["core_alpha_odd", "core_alpha_even"]].min().min()
            - alpha_lower,
            alpha_upper
            - variational_rows[["core_alpha_odd", "core_alpha_even"]].max().max(),
        )
    )
    add(
        "combined",
        "minimum retained loader-angle bound margin",
        loader_bound_margin,
        "> 1e-8",
        loader_bound_margin > 1.0e-8,
    )
    theta_lower, theta_upper = map(float, settings["angle_bounds"])
    correction_bound_margin = float(
        min(
            variational_rows[["theta_pair", "theta_exchange"]].min().min()
            - theta_lower,
            theta_upper
            - variational_rows[["theta_pair", "theta_exchange"]].max().max(),
        )
    )
    add(
        "combined",
        "minimum retained correction-angle bound margin",
        correction_bound_margin,
        "> 1e-8",
        correction_bound_margin > 1.0e-8,
    )

    state_crossovers = {
        "resonance": crossover_summary(
            plotted_combined,
            scan="resonance",
            x="delta",
            metric="physical_two_site_trace_norm_max",
            descending=True,
        ),
        "coupling": crossover_summary(
            plotted_combined,
            scan="coupling",
            x="lambda",
            metric="physical_two_site_trace_norm_max",
            descending=False,
        ),
    }
    rescue_rows: list[dict[str, Any]] = []
    for scan, x in (("resonance", "delta"), ("coupling", "lambda")):
        for value, group in plotted_combined[
            plotted_combined.scan == scan
        ].groupby(x):
            indexed = group.set_index("method")
            variational_error = float(
                indexed.loc["variational", "physical_two_site_trace_norm_max"]
            )
            comparison = min(
                float(indexed.loc["bare", "physical_two_site_trace_norm_max"]),
                float(
                    indexed.loc[
                        "analytic_first_order", "physical_two_site_trace_norm_max"
                    ]
                ),
            )
            rescue_rows.append(
                {
                    "scan": scan,
                    "parameter": x,
                    "value": float(value),
                    "variational_rescue": bool(variational_error < comparison),
                }
            )
    rescue = pd.DataFrame(rescue_rows)

    checks_frame = pd.DataFrame(checks)
    aspect_status = {
        aspect: bool(group.passed.all()) for aspect, group in checks_frame.groupby("aspect")
    }
    verification = {
        "all_checks_passed": bool(checks_frame.passed.all()),
        "aspect_status": aspect_status,
        "failure_count": int((~checks_frame.passed).sum()),
        "state_crossovers": state_crossovers,
        "variational_rescue_counts": {
            scan: int(group.variational_rescue.sum())
            for scan, group in rescue.groupby("scan")
        },
        "variational_rescue_totals": {
            scan: int(len(group)) for scan, group in rescue.groupby("scan")
        },
        "checks": checks,
    }
    return verification, checks_frame


def write_hashes() -> pd.DataFrame:
    source_items = [
        ("source/benchmark_config.json", BENCHMARK_CONFIG_PATH),
        ("source/variational_config.json", VARIATIONAL_CONFIG_PATH),
        *[
            (f"source/{path.relative_to(HERE).as_posix()}", path)
            for path in [
                HERE / "README.md",
                HERE / "requirements.txt",
                HERE / "run_benchmarks.py",
                HERE / "run_variational.py",
                HERE / "analyze_results.py",
                *sorted((HERE / "scta_numerics").glob("*.py")),
                *sorted((HERE / "tests").glob("*.py")),
            ]
        ],
    ]
    groups = (
        ("raw/benchmarks", RAW, sorted(RAW.glob("*"))),
        ("raw/variational", COMBINED_RAW, sorted(COMBINED_RAW.glob("*"))),
        ("processed", PROCESSED, sorted(PROCESSED.glob("*"))),
        ("figures", FIGURES, sorted(FIGURES.glob("*"))),
    )
    rows = [
        {"path": label, "sha256": sha256(path), "bytes": path.stat().st_size}
        for label, path in source_items
        if path.is_file() and not path.name.startswith(".")
    ]
    for prefix, root, paths in groups:
        for path in paths:
            if (
                not path.is_file()
                or path.name == "artifact_hashes.csv"
                or path.name.startswith(".")
            ):
                continue
            try:
                relative = path.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Cannot label {path} relative to its declared artifact root."
                ) from exc
            rows.append(
                {
                    "path": f"{prefix}/{relative.as_posix()}",
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    frame = pd.DataFrame(rows)
    frame.to_csv(PROCESSED / "artifact_hashes.csv", index=False)
    return frame


def parse_args() -> argparse.Namespace:
    """Parse configurable input and output roots for a non-destructive analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=HERE / "results",
        help="Root containing benchmarks/raw and variational/raw.",
    )
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=HERE / "results" / "processed",
        help="Directory for fitted tables, checks, and artifact hashes.",
    )
    parser.add_argument(
        "--figures-output",
        type=Path,
        default=HERE / "figures",
        help="Directory for the three manuscript PDF figures.",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=HERE / "benchmark_config.json",
    )
    parser.add_argument(
        "--variational-config",
        type=Path,
        default=HERE / "variational_config.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(
        args.results_root.resolve(),
        args.processed_output.resolve(),
        args.figures_output.resolve(),
        args.benchmark_config.resolve(),
        args.variational_config.resolve(),
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    residual = load("residual_scaling.csv")
    algebra = load("algebra_checks.csv")
    tolerance_sweep = load("pauli_tolerance_sweep.csv")
    thermal = load("thermal_accuracy.csv")
    combined = load_combined("scan_results.csv")
    optimizer_runs = load_combined("optimizer_runs.csv")
    completeness_checks = configured_completeness_checks(
        residual=residual,
        algebra=algebra,
        tolerance_sweep=tolerance_sweep,
        thermal=thermal,
        combined=combined,
        optimizer_runs=optimizer_runs,
        benchmark_config=CONFIG,
        variational_config=COMBINED_CONFIG,
    )
    failed_completeness = [check for check in completeness_checks if not check["passed"]]
    if failed_completeness:
        frame = pd.DataFrame(completeness_checks)
        frame.to_csv(PROCESSED / "verification_checks.csv", index=False)
        raise SystemExit(
            "Raw input completeness checks failed:\n"
            + frame[~frame.passed].to_string(index=False)
        )

    fits = fit_table()
    fits.to_csv(PROCESSED / "power_law_fits.csv", index=False)
    grouped_fits = thermal_grouped_fit_table()
    grouped_fits.to_csv(PROCESSED / "thermal_grouped_power_law_fits.csv", index=False)
    robustness = optimizer_robustness()
    robustness.to_csv(PROCESSED / "optimizer_robustness.csv", index=False)
    verification, checks = build_verification(
        fits,
        grouped_fits,
        robustness,
        completeness_checks,
    )
    checks.to_csv(PROCESSED / "verification_checks.csv", index=False)
    (PROCESSED / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generate_manuscript_figures(
        residual=residual,
        thermal=thermal,
        combined_plot=combined_plot_data(combined),
        fits=fits,
        grouped_fits=grouped_fits,
        state_crossovers=verification["state_crossovers"],
        beta_values=[float(value) for value in CONFIG["thermal_accuracy"]["beta_values"]],
        output_dir=FIGURES,
    )
    write_hashes()
    if not verification["all_checks_passed"]:
        failed = checks[~checks.passed]
        raise SystemExit(
            "One or more verification checks failed:\n" + failed.to_string(index=False)
        )


if __name__ == "__main__":
    main()
