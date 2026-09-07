"""Final manuscript figure generation for the SCTA numerical benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator


def plotting_style() -> None:
    """Apply the typography and dimensions used by the manuscript figures."""

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.latex.preamble": (
                r"\usepackage[T1]{fontenc}"
                r"\usepackage{lmodern}"
                r"\usepackage{amsmath,amssymb}"
            ),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_label(axis: plt.Axes, label: str, *, y: float = 1.035) -> None:
    axis.text(
        -0.12,
        y,
        rf"\textbf{{({label})}}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="black",
        clip_on=False,
    )


def save_figure(fig: plt.Figure, name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_residual(
    data: pd.DataFrame,
    fits: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.40))
    colors = {"bare": "#555555", "first_order": "#0072B2"}
    markers = {"bare": "s", "first_order": "o"}
    fit_rows: dict[str, pd.Series] = {}
    for method in ("bare", "first_order"):
        selected = data[data.method == method].sort_values("lambda")
        fit = fits[
            (fits.label == f"residual_{method}") & (fits.trim == "none")
        ].iloc[0]
        fit_rows[method] = fit
        fit_lambda = np.geomspace(
            selected["lambda"].min(), selected["lambda"].max(), 200
        )
        axes[0].loglog(
            selected["lambda"],
            selected.incident_strength,
            linestyle="none",
            marker=markers[method],
            ms=4.2,
            color=colors[method],
        )
        axes[0].loglog(
            fit_lambda,
            fit.prefactor * fit_lambda ** fit.exponent,
            linestyle="--",
            lw=1.4,
            color=colors[method],
        )
    label_positions = {"bare": 0.00562341325, "first_order": 0.035}
    label_offsets = {"bare": (0, 18), "first_order": (0, -28)}
    label_vertical_alignment = {"bare": "bottom", "first_order": "top"}
    for method in ("bare", "first_order"):
        fit = fit_rows[method]
        x_position = label_positions[method]
        y_position = fit.prefactor * x_position ** fit.exponent
        exponent_label = (
            rf"$p_{{\mathrm{{bare}}}}={fit.exponent:.3f}$"
            if method == "bare"
            else rf"$p_{{\mathrm{{corr}}}}={fit.exponent:.3f}\pm{fit.exponent_standard_error:.3f}$"
        )
        axes[0].annotate(
            exponent_label,
            xy=(x_position, y_position),
            xytext=label_offsets[method],
            textcoords="offset points",
            ha="center",
            va=label_vertical_alignment[method],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        )
    axes[0].set_xlabel("$|\\lambda|$")
    axes[0].set_ylabel("$\\varepsilon_P$")
    panel_label(axes[0], "a")

    for method, marker in (("bare_size_scan", "s"), ("first_order_size_scan", "o")):
        selected = data[data.method == method].sort_values("N")
        axes[1].plot(
            selected.N,
            selected.incident_strength,
            marker + "-",
            ms=3.5,
            lw=1.2,
            color=colors["bare" if method.startswith("bare") else "first_order"],
        )
    axes[1].set_xlabel("$N$")
    axes[1].set_ylabel("$\\varepsilon_P$")
    panel_label(axes[1], "b")
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=colors[method],
                linestyle="-",
                marker=markers[method],
                lw=1.0,
                ms=4.2,
                label="bare" if method == "bare" else "first order",
            )
            for method in ("bare", "first_order")
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.4,
        handlelength=1.3,
    )
    fig.tight_layout(rect=[0.0, 0.09, 1.0, 1.0], h_pad=1.25)
    save_figure(fig, "residual_scaling_vertical", output_dir)


def plot_thermal(
    data: pd.DataFrame,
    grouped_fits: pd.DataFrame,
    beta_values: list[float],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 5.00))
    beta_colors = {0.5: "#D55E00", 1.0: "#56B4E9", 2.0: "#0072B2"}
    metrics = (
        "physical_two_site_trace_norm_max",
        "physical_ZZ_error_max",
    )
    fit_curves: dict[tuple[str, float, str], pd.Series] = {}
    for beta in beta_values:
        for method, marker in (("bare", "s"), ("first_order", "o")):
            selected = data[(data.beta == beta) & (data.method == method)].sort_values("lambda")
            fit_lambda = np.geomspace(
                selected["lambda"].min(), selected["lambda"].max(), 200
            )
            for axis, metric in zip(axes, metrics):
                axis.loglog(
                    selected["lambda"],
                    selected[metric],
                    linestyle="none",
                    marker=marker,
                    ms=3.4,
                    color=beta_colors[beta],
                )
                metric_name = "D1" if metric == metrics[0] else "ZZ"
                fit = grouped_fits[
                    (grouped_fits.metric == metric_name)
                    & (grouped_fits.method == method)
                    & np.isclose(grouped_fits.beta, beta)
                ].iloc[0]
                fit_curves[(metric, beta, method)] = fit
                axis.loglog(
                    fit_lambda,
                    fit.prefactor * fit_lambda ** fit.exponent,
                    linestyle="--",
                    lw=1.15,
                    color=beta_colors[beta],
                )
    axes[0].set_xlabel("$|\\lambda|$")
    axes[0].set_ylabel("$D_1$")
    panel_label(axes[0], "a")
    label_x_bare = 0.016875
    label_x_corrected = 0.018

    def fitted_value(metric: str, method: str, x_value: float, reducer: Any) -> float:
        values = [
            fit_curves[(metric, beta, method)]["prefactor"]
            * x_value ** fit_curves[(metric, beta, method)]["exponent"]
            for beta in beta_values
        ]
        return float(reducer(values))

    def grouped_label(metric_name: str, method: str) -> str:
        fit = grouped_fits[
            (grouped_fits.metric == metric_name)
            & (grouped_fits.method == method)
        ].iloc[0]
        name = "bare" if method == "bare" else "corr"
        return (
            rf"$p_{{\mathrm{{{name}}}}}="
            rf"{fit.exponent:.4f}\pm{fit.exponent_standard_error:.4f}$"
        )

    axes[0].annotate(
        grouped_label("D1", "bare"),
        xy=(label_x_bare, fitted_value(metrics[0], "bare", label_x_bare, max)),
        xytext=(0, 17),
        textcoords="offset points",
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )
    axes[0].annotate(
        grouped_label("D1", "first_order"),
        xy=(
            label_x_corrected,
            fitted_value(metrics[0], "first_order", label_x_corrected, min),
        ),
        xytext=(5, -7),
        textcoords="offset points",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )
    axes[1].set_xlabel("$|\\lambda|$")
    axes[1].set_ylabel("$\\Delta_{ZZ}$")
    axes[1].yaxis.set_minor_locator(NullLocator())
    panel_label(axes[1], "b")
    axes[1].annotate(
        grouped_label("ZZ", "bare"),
        xy=(label_x_bare, fitted_value(metrics[1], "bare", label_x_bare, max)),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )
    axes[1].annotate(
        grouped_label("ZZ", "first_order"),
        xy=(
            label_x_corrected,
            fitted_value(metrics[1], "first_order", label_x_corrected, min),
        ),
        xytext=(5, -7),
        textcoords="offset points",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )

    beta_handles = [
        Line2D(
            [0],
            [0],
            color=beta_colors[beta],
            linestyle="--",
            lw=1.5,
            label=rf"$\beta={beta:g}$",
        )
        for beta in beta_values
    ]
    method_handles = [
        Line2D([0], [0], color="0.35", linestyle="-", marker="s", ms=3.4, lw=1.0, label="bare"),
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle="-",
            marker="o",
            ms=3.4,
            lw=1.0,
            label="first order",
        ),
    ]
    fig.legend(
        handles=beta_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.0,
        handlelength=1.6,
    )
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.2,
        handlelength=1.3,
    )
    fig.tight_layout(rect=[0.0, 0.15, 1.0, 1.0], h_pad=1.35)
    save_figure(fig, "thermal_accuracy_vertical", output_dir)


def plot_combined(
    data: pd.DataFrame,
    state_crossovers: dict[str, Any],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.60))
    colors = {
        "bare": "#555555",
        "analytic_first_order": "#0072B2",
        "variational": "#009E73",
    }
    labels = {
        "bare": "bare",
        "analytic_first_order": "first order",
        "variational": "variational",
    }
    markers = {"bare": "s", "analytic_first_order": "o", "variational": "D"}

    resonance = data[data.scan == "resonance"]
    coupling = data[data.scan == "coupling"]
    for method in colors:
        selected = resonance[resonance.method == method].sort_values(
            "delta", ascending=False
        )
        axes[0].plot(
            selected.delta,
            selected.physical_two_site_trace_norm_max,
            marker=markers[method],
            ms=3.6,
            lw=1.15,
            color=colors[method],
            label=labels[method],
        )
        selected = coupling[coupling.method == method].sort_values("lambda")
        axes[1].loglog(
            selected["lambda"],
            selected.physical_two_site_trace_norm_max,
            marker=markers[method],
            ms=3.6,
            lw=1.15,
            color=colors[method],
        )

    resonance_crossover = state_crossovers["resonance"]
    coupling_crossover = state_crossovers["coupling"]
    if resonance_crossover["found"]:
        axes[0].axvspan(
            resonance_crossover["bracket_min"],
            resonance_crossover["bracket_max"],
            color="#E69F00",
            alpha=0.12,
            lw=0,
        )
        axes[0].axvline(
            resonance_crossover["estimate"], color="#E69F00", ls=":", lw=1.0
        )
    if coupling_crossover["found"]:
        axes[1].axvspan(
            coupling_crossover["bracket_min"],
            coupling_crossover["bracket_max"],
            color="#E69F00",
            alpha=0.12,
            lw=0,
        )
        axes[1].axvline(
            coupling_crossover["estimate"], color="#E69F00", ls=":", lw=1.0
        )
        axes[1].annotate(
            rf"$\lambda_\times\approx{coupling_crossover['estimate']:.3f}$",
            xy=(coupling_crossover["estimate"], 1.015),
            xycoords=("data", "axes fraction"),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#9A6700",
        )

    axes[0].set_yscale("log")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([0.001953125, 0.0078125, 0.03125, 0.125, 0.5])
    axes[0].set_xticklabels(
        [r"$2^{-9}$", r"$2^{-7}$", r"$2^{-5}$", r"$2^{-3}$", r"$2^{-1}$"]
    )
    axes[0].set_xlabel(r"$\delta$")
    axes[0].set_ylabel("$D_1$")
    panel_label(axes[0], "a", y=1.085)
    axes[1].set_xlabel(r"$|\lambda|$")
    axes[1].set_ylabel("$D_1$")
    panel_label(axes[1], "b", y=1.085)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=colors[method],
                marker=markers[method],
                lw=1.15,
                ms=3.6,
                label=labels[method],
            )
            for method in colors
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=8.2,
        columnspacing=1.2,
        handlelength=1.3,
    )
    fig.tight_layout(rect=[0.0, 0.09, 1.0, 1.0], h_pad=1.25)
    save_figure(fig, "combined_variational_failure_vertical", output_dir)


def generate_manuscript_figures(
    *,
    residual: pd.DataFrame,
    thermal: pd.DataFrame,
    combined_plot: pd.DataFrame,
    fits: pd.DataFrame,
    grouped_fits: pd.DataFrame,
    state_crossovers: dict[str, Any],
    beta_values: list[float],
    output_dir: Path,
) -> None:
    """Generate the three finalized vertical PDFs used by the manuscript."""

    plotting_style()
    plot_residual(residual, fits, output_dir)
    plot_thermal(thermal, grouped_fits, beta_values, output_dir)
    plot_combined(combined_plot, state_crossovers, output_dir)
