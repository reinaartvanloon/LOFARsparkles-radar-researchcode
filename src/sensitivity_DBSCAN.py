#!/usr/bin/env python3
"""
OAT sensitivity study for DBSCAN clustering parameters.

Library module — contains config dataclasses, all computational functions,
plot functions, and main() for use by operational scripts.
"""
from __future__ import annotations

import os
import csv
import gc
import json
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from dataclasses import dataclass
from typing import List, Optional
from scipy.stats import ks_2samp

from general import ConfigPlot, outpath_gen
from read_RAD import (
    ConfigDataRAD, ConfigMaskRADnearVHF,
    get_data_RADandLOFAR, add_mask_RADnearVHF,
)
from plot_LOFAR import SparkleParams, ConfigLOFAR
from stats_sparklesRAD import cliffs_delta


# ── OAT design constants ───────────────────────────────────────────────────────

BASELINE = {
    "large_d":   1000,
    "large_t":   150,
    "large_n":   30,
    "sparkle_d": 200,
    "sparkle_t": 5,
    "sparkle_n": 2,
}

PARAM_LEVELS = {
    "large_d":   [500, 800, 1000, 1500, 2000],
    "large_t":   [50, 100,   150,  250, 400],
    "large_n":   [10, 20,  30, 40,  60],
    "sparkle_d": [100, 150, 200, 300, 500],
    "sparkle_t": [1, 3, 5, 10, 20],
    "sparkle_n": [2, 3, 5, 10],
}

PARAM_LABELS = {
    "large_d":   r"$d_\mathrm{large}$ [m]",
    "large_t":   r"$\tau_\mathrm{large}$ [ms]",
    "large_n":   r"$n_\mathrm{large}$",
    "sparkle_d": r"$d_\mathrm{sparkle}$ [m]",
    "sparkle_t": r"$\tau_\mathrm{sparkle}$ [ms]",
    "sparkle_n": r"$n_\mathrm{sparkle}$",
}

PARAM_GROUPS = {
    "large":   ["large_d",   "large_t",   "large_n"],
    "sparkle": ["sparkle_d", "sparkle_t", "sparkle_n"],
}

PARAMS = list(PARAM_LEVELS.keys())

METRICS = [
    "n_sparkle_dbzh",
    "n_other_dbzh",
    "mean_sparkle_dbzh",  "mean_other_dbzh",
    "p25_sparkle_dbzh",   "p75_sparkle_dbzh",
    "p25_other_dbzh",     "p75_other_dbzh",
    "mean_sparkle_wradh", "mean_other_wradh",
    "p25_sparkle_wradh",  "p75_sparkle_wradh",
    "p25_other_wradh",    "p75_other_wradh",
    "cliffs_delta_dbzh",  "cliffs_delta_wradh",
    "cliffs_delta_ci_lo_dbzh",  "cliffs_delta_ci_hi_dbzh",
    "cliffs_delta_ci_lo_wradh", "cliffs_delta_ci_hi_wradh",
    "ks_stat_dbzh",       "ks_stat_wradh",
    "ks_stat_ci_lo_dbzh", "ks_stat_ci_hi_dbzh",
    "ks_stat_ci_lo_wradh","ks_stat_ci_hi_wradh",
]

METRIC_LABELS = {
    "n_sparkle_dbzh":           "Nr. radar bins\n(sparkles)",
    "n_other_dbzh":             "Nr. radar bins\n(other VHF)",
    "mean_sparkle_dbzh":        r"Mean $Z_h$ sparkles [dBZ]",
    "mean_other_dbzh":          r"Mean $Z_h$ other VHF [dBZ]",
    "mean_sparkle_wradh":       r"Mean $W_{rad}$ sparkles [m s$^{-1}$]",
    "mean_other_wradh":         r"Mean $W_{rad}$ other VHF [m s$^{-1}$]",
    "cliffs_delta_dbzh":        r"Cliff's $\delta$, $Z_h$",
    "cliffs_delta_wradh":       r"Cliff's $\delta$, $W_{rad}$",
    "ks_stat_dbzh":             r"KS statistic, $Z_h$",
    "ks_stat_wradh":            r"KS statistic, $W_{rad}$",
}

N_BOOTSTRAP     = 2000
MAX_BOOT_SAMPLE = 50_000

# ── Plot styling ───────────────────────────────────────────────────────────────

C_SPARK  = "crimson"
C_OTHER  = "steelblue"
C_ZH     = "forestgreen"
C_W      = "goldenrod"
EDGE     = "black"
FONTSIZE = 14


# ── Config dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ConfigSensitivityDBSCANData:
    """Input data configuration for the DBSCAN OAT sensitivity study."""
    rad_data_dirpath: str
    lofar_data_dirpath: str
    advection_reference_filepath: str
    hmc_msf_filepath: str
    lofar_file_list: List[str]
    varlist: List[str]
    config_mask_rad: ConfigMaskRADnearVHF
    stormcode: str = "21C"
    rad_station: str = "asb"
    max_distance: float = 100e3


@dataclass
class ConfigSensitivityDBSCANPlot(ConfigPlot):
    """Output and run-control configuration for the DBSCAN sensitivity study."""
    outdir: str = None
    csv_path: str = None
    plot_only: bool = False
    n_bootstrap: int = N_BOOTSTRAP
    confidence_interval: bool = False


# ── Private helpers ────────────────────────────────────────────────────────────

def _make_config_data_rad(cfg: ConfigSensitivityDBSCANData) -> ConfigDataRAD:
    return ConfigDataRAD(
        cfg.rad_data_dirpath,
        stormcode=cfg.stormcode,
        VHFtype="sparkles&otherVHF",
        RADstation=cfg.rad_station,
        RADvars=cfg.varlist,
        advection_reference_filepath=cfg.advection_reference_filepath,
        temp_reference_filepath=cfg.advection_reference_filepath,
        hmc_msf_filepath=cfg.hmc_msf_filepath,
        epsg=28992,
    )


def _label_panel(ax, letter):
    ax.set_title(f"({letter})", loc="left", fontsize=FONTSIZE + 1, fontweight="bold")


# ── Bootstrap helpers ──────────────────────────────────────────────────────────

def _ks_stat(x, y):
    return ks_2samp(x, y, method='asymp').statistic


def _cliffs_delta_func(x, y):
    y_sorted    = np.sort(y)
    n_x_gt_y    = np.searchsorted(y_sorted, x, side="left")
    equal_count = np.searchsorted(y_sorted, x, side="right") - n_x_gt_y
    n_x_lt_y    = len(y) - n_x_gt_y - equal_count
    return (np.sum(n_x_gt_y) - np.sum(n_x_lt_y)) / (len(x) * len(y))


def _bootstrap_ci(func, x, y, n_boot=N_BOOTSTRAP, ci=0.95, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if len(x) > MAX_BOOT_SAMPLE:
        x = x[rng.integers(0, len(x), size=MAX_BOOT_SAMPLE)]
    if len(y) > MAX_BOOT_SAMPLE:
        y = y[rng.integers(0, len(y), size=MAX_BOOT_SAMPLE)]
    nx, ny = len(x), len(y)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        xb = x[rng.integers(0, nx, size=nx)]
        yb = y[rng.integers(0, ny, size=ny)]
        vals[i] = func(xb, yb)
    half = (1.0 - ci) / 2.0
    return float(np.percentile(vals, 100 * half)), float(np.percentile(vals, 100 * (1 - half)))


# ── Model evaluation ───────────────────────────────────────────────────────────

def run_one(params: dict, vardata: dict, config_data: ConfigSensitivityDBSCANData,
            n_bootstrap: int = N_BOOTSTRAP, do_ci: bool = True) -> dict:
    """Evaluate all metrics for one parameter combination."""
    large_cluster   = {k[len("large_"):]:   params[k] for k in ("large_d",   "large_t",   "large_n")}
    sparkle_cluster = {k[len("sparkle_"):]: params[k] for k in ("sparkle_d", "sparkle_t", "sparkle_n")}

    sparkle_params = SparkleParams(
        large_cluster=large_cluster,
        sparkle_cluster=sparkle_cluster,
        alt_windows=[[8000, None]],
    )
    config_LOFAR = ConfigLOFAR(
        stormcode=config_data.stormcode,
        datapath=config_data.lofar_data_dirpath,
        sparkle_params=sparkle_params,
        max_distance=config_data.max_distance,
    )
    config_rad = _make_config_data_rad(config_data)

    per_var = {var: {"sparkles": [], "other": []} for var in config_data.varlist}

    for file in config_data.lofar_file_list:
        print(file)
        config_LOFAR.LOFAR_file = file
        data = get_data_RADandLOFAR(config_rad, config_LOFAR=config_LOFAR)
        data.RAD = add_mask_RADnearVHF(data.RAD, data.LOFAR, config=config_data.config_mask_rad)

        ds      = data.RAD.ds
        mask_sp = ds.mask_sparkles.values.ravel()
        mask_ot = ds.mask_otherVHF.values.ravel() & ~mask_sp

        for var in config_data.varlist:
            odim_key = vardata[var]["ODIM"]
            vals = ds[odim_key].values.ravel()
            per_var[var]["sparkles"].append(vals[mask_sp])
            per_var[var]["other"].append(vals[mask_ot])

        del data, ds, mask_sp, mask_ot
        gc.collect()

    rng    = np.random.default_rng(42)
    result = dict(params)

    for var in config_data.varlist:
        v_spark = np.concatenate(per_var[var]["sparkles"])
        v_spark = v_spark[np.isfinite(v_spark)]
        v_other = np.concatenate(per_var[var]["other"])
        v_other = v_other[np.isfinite(v_other)]

        result[f"n_sparkle_{var}"] = len(v_spark)
        result[f"n_other_{var}"]   = len(v_other)
        result[f"mean_sparkle_{var}"] = np.mean(v_spark) if len(v_spark) > 0 else np.nan
        result[f"mean_other_{var}"]   = np.mean(v_other) if len(v_other) > 0 else np.nan
        result[f"p25_sparkle_{var}"]  = np.percentile(v_spark, 25) if len(v_spark) > 0 else np.nan
        result[f"p75_sparkle_{var}"]  = np.percentile(v_spark, 75) if len(v_spark) > 0 else np.nan
        result[f"p25_other_{var}"]    = np.percentile(v_other, 25) if len(v_other) > 0 else np.nan
        result[f"p75_other_{var}"]    = np.percentile(v_other, 75) if len(v_other) > 0 else np.nan

        if len(v_spark) > 1 and len(v_other) > 1:
            result[f"cliffs_delta_{var}"] = cliffs_delta(v_spark, v_other)
            result[f"ks_stat_{var}"]      = _ks_stat(v_spark, v_other)
            if do_ci:
                cd_lo, cd_hi = _bootstrap_ci(_cliffs_delta_func, v_spark, v_other,
                                             n_boot=n_bootstrap, rng=rng)
                result[f"cliffs_delta_ci_lo_{var}"] = cd_lo
                result[f"cliffs_delta_ci_hi_{var}"] = cd_hi
                ks_lo, ks_hi = _bootstrap_ci(_ks_stat, v_spark, v_other,
                                             n_boot=n_bootstrap, rng=rng)
                result[f"ks_stat_ci_lo_{var}"]      = ks_lo
                result[f"ks_stat_ci_hi_{var}"]      = ks_hi
            else:
                for suffix in ("cliffs_delta_ci_lo", "cliffs_delta_ci_hi",
                               "ks_stat_ci_lo", "ks_stat_ci_hi"):
                    result[f"{suffix}_{var}"] = np.nan
        else:
            for suffix in ("cliffs_delta", "cliffs_delta_ci_lo", "cliffs_delta_ci_hi",
                           "ks_stat", "ks_stat_ci_lo", "ks_stat_ci_hi"):
                result[f"{suffix}_{var}"] = np.nan

    return result


def build_oat_runs() -> list[dict]:
    """Return list of parameter dicts: baseline + one-at-a-time perturbations."""
    runs = [dict(BASELINE)]
    for param, levels in PARAM_LEVELS.items():
        for val in levels:
            if val == BASELINE[param]:
                continue
            perturbed = dict(BASELINE)
            perturbed[param] = val
            runs.append(perturbed)
    return runs


def _params_key(params: dict) -> tuple:
    return tuple(sorted(params.items()))


def load_results_from_csv(csv_path: str) -> tuple[list[dict], dict]:
    """Load OAT results from a previously saved CSV."""
    param_keys = list(BASELINE.keys())
    results    = []
    with open(csv_path, newline="") as f:
        reader     = csv.DictReader(f)
        csv_columns = reader.fieldnames or []
        for row in reader:
            result = {}
            for k in csv_columns:
                v = row[k]
                try:
                    result[k] = int(v) if k in param_keys else float(v)
                except (ValueError, TypeError):
                    result[k] = np.nan
            for m in METRICS:
                result.setdefault(m, np.nan)
            results.append(result)

    baseline_result = next(
        (r for r in results if all(r.get(p) == BASELINE[p] for p in param_keys)),
        None,
    )
    if baseline_result is None:
        raise ValueError(f"No baseline row found in {csv_path}.\nExpected params: {BASELINE}")

    print(f"Loaded {len(results)} rows from {csv_path}")
    return results, baseline_result


# ── Per-run subprocess worker ──────────────────────────────────────────────────

def _run_worker(params, vardata, config_data, n_bootstrap, do_ci, result_queue):
    """Runs in a child process so its memory is fully released on exit."""
    try:
        result = run_one(params, vardata, config_data, n_bootstrap=n_bootstrap, do_ci=do_ci)
        result_queue.put(("ok", result))
    except Exception as e:
        import traceback
        result_queue.put(("error", str(e), traceback.format_exc()))


# ── Sensitivity matrix plot ────────────────────────────────────────────────────

def sensitivity_matrix_plot(oat_results: list[dict], baseline_result: dict,
                             group_name: str, params_in_group: list[str],
                             outdir: str):
    """
    Sensitivity matrix: 5 rows (metric panels) × len(params_in_group) columns.

    Row 0 — Nr. radar bins: sparkles left y, other VHF right twinx
    Row 1 — Mean Zh + IQR shading
    Row 2 — Mean W  + IQR shading
    Row 3 — KS statistic D: Zh and W on same axis
    Row 4 — Cliff's delta: Zh and W on same axis
    """
    n_rows   = 5
    n_params = len(params_in_group)

    fig, axes = plt.subplots(
        n_rows, n_params,
        figsize=(4 * n_params, 3.2 * n_rows),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )

    if n_params == 1:
        axes = axes[:, np.newaxis]

    for col, param in enumerate(params_in_group):
        baseline_val = BASELINE[param]

        runs_for_param = [r for r in oat_results
                          if all(r[p] == BASELINE[p] for p in PARAMS if p != param)]
        runs_for_param.sort(key=lambda r: r[param])
        xv = [r[param] for r in runs_for_param]

        def _letter(row, _col=col):
            return chr(ord('a') + row * n_params + _col)

        # ── Row 0: Nr. radar bins ─────────────────────────────────────────────
        ax   = axes[0, col]
        ax_r = ax.twinx()

        n_sp = [r["n_sparkle_dbzh"] for r in runs_for_param]
        n_ot = [r["n_other_dbzh"]   for r in runs_for_param]

        ax.plot(xv, n_sp, color=C_SPARK, linestyle="-",  marker="o", linewidth=1.5,
                markeredgecolor=EDGE, markeredgewidth=0.5)
        ax_r.plot(xv, n_ot, color=C_OTHER, linestyle="--", marker="s", linewidth=1.5,
                  markeredgecolor=EDGE, markeredgewidth=0.5)
        ax.plot(baseline_val, baseline_result["n_sparkle_dbzh"],
                marker="o", color=C_SPARK, markersize=8, zorder=5,
                markeredgecolor="k", markeredgewidth=1.2)
        ax_r.plot(baseline_val, baseline_result["n_other_dbzh"],
                  marker="s", color=C_OTHER, markersize=8, zorder=5,
                  markeredgecolor="k", markeredgewidth=1.2)

        ax.axvline(baseline_val, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(xv)
        ax.tick_params(labelsize=FONTSIZE - 1)
        ax.tick_params(axis="y", colors=C_SPARK, labelsize=FONTSIZE - 1)
        ax_r.tick_params(axis="y", colors=C_OTHER, labelsize=FONTSIZE - 1)

        if col == 0:
            ax.set_ylabel("Nr. radar bins\n(sparkles)", fontsize=FONTSIZE, color=C_SPARK)
        if col == n_params - 1:
            ax_r.set_ylabel("Nr. radar bins\n(other VHF)", fontsize=FONTSIZE, color=C_OTHER)

        ax.set_title(PARAM_LABELS[param], fontsize=FONTSIZE)
        _label_panel(ax, _letter(0))

        # ── Row 1: Mean Zh + IQR ──────────────────────────────────────────────
        ax = axes[1, col]
        for group, color, marker, ls in [
            ("sparkle", C_SPARK, "o", "-"),
            ("other",   C_OTHER, "s", "--"),
        ]:
            means = [r[f"mean_{group}_dbzh"] for r in runs_for_param]
            p25   = [r[f"p25_{group}_dbzh"]  for r in runs_for_param]
            p75   = [r[f"p75_{group}_dbzh"]  for r in runs_for_param]
            ax.fill_between(xv, p25, p75, alpha=0.25, color=color, zorder=1)
            ax.plot(xv, means, color=color, linestyle=ls, marker=marker,
                    linewidth=1.5, markeredgecolor=EDGE, markeredgewidth=0.5, zorder=3)
            b_mean = baseline_result[f"mean_{group}_dbzh"]
            ax.plot(baseline_val, b_mean, marker=marker, color=color, markersize=8, zorder=5,
                    markeredgecolor="k", markeredgewidth=1.2)
        ax.axvline(baseline_val, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(xv)
        ax.tick_params(labelsize=FONTSIZE - 1)
        if col == 0:
            ax.set_ylabel(r"$Z_h$ [dBZ]", fontsize=FONTSIZE)
        _label_panel(ax, _letter(1))

        # ── Row 2: Mean W + IQR ───────────────────────────────────────────────
        ax = axes[2, col]
        for group, color, marker, ls in [
            ("sparkle", C_SPARK, "o", "-"),
            ("other",   C_OTHER, "s", "--"),
        ]:
            means = [r[f"mean_{group}_wradh"] for r in runs_for_param]
            p25   = [r[f"p25_{group}_wradh"]  for r in runs_for_param]
            p75   = [r[f"p75_{group}_wradh"]  for r in runs_for_param]
            ax.fill_between(xv, p25, p75, alpha=0.25, color=color, zorder=1)
            ax.plot(xv, means, color=color, linestyle=ls, marker=marker,
                    linewidth=1.5, markeredgecolor=EDGE, markeredgewidth=0.5, zorder=3)
            b_mean = baseline_result[f"mean_{group}_wradh"]
            ax.plot(baseline_val, b_mean, marker=marker, color=color, markersize=8, zorder=5,
                    markeredgecolor="k", markeredgewidth=1.2)
        ax.axvline(baseline_val, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(xv)
        ax.tick_params(labelsize=FONTSIZE - 1)
        if col == 0:
            ax.set_ylabel(r"$W_{rad}$ [m s$^{-1}$]", fontsize=FONTSIZE)
        _label_panel(ax, _letter(2))

        # ── Row 3: KS statistic D ─────────────────────────────────────────────
        ax = axes[3, col]
        for var, color, marker in [("dbzh", C_ZH, "o"), ("wradh", C_W, "s")]:
            ks_vals = [r[f"ks_stat_{var}"]       for r in runs_for_param]
            ci_lo   = [r[f"ks_stat_ci_lo_{var}"] for r in runs_for_param]
            ci_hi   = [r[f"ks_stat_ci_hi_{var}"] for r in runs_for_param]
            if not all(np.isnan(v) for v in ci_lo):
                ax.fill_between(xv, ci_lo, ci_hi, alpha=0.25, color=color, zorder=1)
            ax.plot(xv, ks_vals, color=color, marker=marker, linestyle="-",
                    linewidth=1.5, markeredgecolor=EDGE, markeredgewidth=0.5, zorder=3)
            b_ks = baseline_result[f"ks_stat_{var}"]
            ax.plot(baseline_val, b_ks, marker=marker, color=color, markersize=8, zorder=5,
                    markeredgecolor="k", markeredgewidth=1.2)
        ax.axvline(baseline_val, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(xv)
        ax.tick_params(labelsize=FONTSIZE - 1)
        if col == 0:
            ax.set_ylabel("KS statistic", fontsize=FONTSIZE)
        _label_panel(ax, _letter(3))

        # ── Row 4: Cliff's delta ──────────────────────────────────────────────
        ax = axes[4, col]
        for var, color, marker in [("dbzh", C_ZH, "o"), ("wradh", C_W, "s")]:
            cds   = [r[f"cliffs_delta_{var}"]       for r in runs_for_param]
            ci_lo = [r[f"cliffs_delta_ci_lo_{var}"] for r in runs_for_param]
            ci_hi = [r[f"cliffs_delta_ci_hi_{var}"] for r in runs_for_param]
            if not all(np.isnan(v) for v in ci_lo):
                ax.fill_between(xv, ci_lo, ci_hi, alpha=0.25, color=color, zorder=1)
            ax.plot(xv, cds, color=color, marker=marker, linestyle="-",
                    linewidth=1.5, markeredgecolor=EDGE, markeredgewidth=0.5, zorder=3)
            b_cd = baseline_result[f"cliffs_delta_{var}"]
            ax.plot(baseline_val, b_cd, marker=marker, color=color, markersize=8, zorder=5,
                    markeredgecolor="k", markeredgewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axvline(baseline_val, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(xv)
        ax.tick_params(labelsize=FONTSIZE - 1)
        if col == 0:
            ax.set_ylabel(r"Cliff's $\delta$", fontsize=FONTSIZE)
        ax.set_xlabel(PARAM_LABELS[param], fontsize=FONTSIZE)
        _label_panel(ax, _letter(4))

    # ── Figure legend ─────────────────────────────────────────────────────────
    handles = [
        mlines.Line2D([], [], color=C_SPARK, linestyle="-",  marker="o", linewidth=1.5,
                      markeredgecolor=EDGE, label="Sparkles"),
        mlines.Line2D([], [], color=C_OTHER, linestyle="--", marker="s", linewidth=1.5,
                      markeredgecolor=EDGE, label="Other VHF"),
        mlines.Line2D([], [], color=C_ZH, marker="o", linewidth=1.5,
                      markeredgecolor=EDGE, label=r"$Z_h$"),
        mlines.Line2D([], [], color=C_W,  marker="s", linewidth=1.5,
                      markeredgecolor=EDGE, label=r"$W_{rad}$"),
        mlines.Line2D([], [], linestyle=':', color="k", marker="none",
                      markeredgecolor="none", markersize=7, label="Baseline"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=len(handles),
               fontsize=FONTSIZE, framealpha=0.85,
               bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.96))

    outpath = os.path.join(outdir, f"sensitivity_matrix_{group_name}.png")
    fig.show()
    return fig
    # fig.savefig(outpath, dpi=150, bbox_inches="tight")


def elasticity_heatmap(oat_results: list[dict], baseline_result: dict, outdir: str):
    """Normalised elasticity heatmap: E = (Δy/y_base) / (Δx/x_base)."""
    heatmap_metrics = [
        m for m in METRICS
        if not any(s in m for s in ("_ci_lo", "_ci_hi", "p25_", "p75_"))
    ]

    E = np.full((len(PARAMS), len(heatmap_metrics)), np.nan)

    for i, param in enumerate(PARAMS):
        x_base = BASELINE[param]
        runs_for_param = [r for r in oat_results
                          if all(r[p] == BASELINE[p] for p in PARAMS if p != param)
                          and r[param] != x_base]

        for j, metric in enumerate(heatmap_metrics):
            y_base = baseline_result[metric]
            if y_base == 0 or not np.isfinite(y_base):
                continue
            elasticities = []
            for r in runs_for_param:
                dx = (r[param] - x_base) / x_base
                dy = (r[metric] - y_base) / y_base
                if dx != 0:
                    elasticities.append(dy / dx)
            if elasticities:
                E[i, j] = np.mean(elasticities)

    absmax = np.nanmax(np.abs(E))
    norm   = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)

    fig, ax = plt.subplots(figsize=(max(7, len(heatmap_metrics) * 0.9), 5),
                           constrained_layout=True)
    im = ax.imshow(E, cmap="RdBu_r", norm=norm, aspect="auto")

    xlabels = [METRIC_LABELS.get(m, m) for m in heatmap_metrics]
    ax.set_xticks(range(len(heatmap_metrics)))
    ax.set_xticklabels(xlabels, fontsize=FONTSIZE - 1, rotation=30, ha="right")
    ax.set_yticks(range(len(PARAMS)))
    ax.set_yticklabels([PARAM_LABELS[p] for p in PARAMS], fontsize=FONTSIZE)

    for i in range(len(PARAMS)):
        for j in range(len(heatmap_metrics)):
            val = E[i, j]
            txt = f"{val:.2f}" if np.isfinite(val) else "–"
            ax.text(j, i, txt, ha="center", va="center", fontsize=FONTSIZE - 1,
                    color="white" if abs(val) > 0.5 * absmax else "black")

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Normalised elasticity\n" r"$(\Delta y/y_0) \,/\, (\Delta x/x_0)$",
                 fontsize=FONTSIZE - 1)
    ax.set_title("OAT sensitivity: normalised elasticity", fontsize=FONTSIZE + 1)

    outpath = os.path.join(outdir, "sensitivity_elasticity_heatmap.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Elasticity heatmap saved to: {outpath}")
    # plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(config_data: ConfigSensitivityDBSCANData, config_plot: ConfigSensitivityDBSCANPlot):
    os.makedirs(config_plot.outdir, exist_ok=True)

    csv_path    = config_plot.csv_path or os.path.join(config_plot.outdir, "sensitivity_DBSCAN_results.csv")
    n_bootstrap = config_plot.n_bootstrap
    do_ci       = config_plot.confidence_interval

    if config_plot.plot_only:
        print(f"Plot-only mode: loading results from {csv_path}")
        results, baseline_result = load_results_from_csv(csv_path)

    else:
        with open(os.path.join(config_data.rad_data_dirpath, "variables_metadata.json")) as f:
            vardata = json.load(f)

        runs   = build_oat_runs()
        n_runs = len(runs)
        print(f"OAT design: {n_runs} model evaluations "
              f"(1 baseline + {n_runs - 1} perturbations across {len(PARAMS)} parameters)")
        print(f"Bootstrap CI: {'enabled (' + str(n_bootstrap) + ' iterations)' if do_ci else 'disabled'}\n")

        fieldnames = list(BASELINE.keys()) + METRICS

        completed_keys  = set()
        results         = []
        baseline_result = None
        if os.path.exists(csv_path):
            try:
                results, baseline_result = load_results_from_csv(csv_path)
                completed_keys = {
                    _params_key({p: r[p] for p in BASELINE}) for r in results
                }
                print(f"Resuming: {len(completed_keys)} runs already done.")
            except Exception as exc:
                print(f"Could not read existing CSV ({exc}); starting fresh.")
                results         = []
                baseline_result = None

        csv_mode = "a" if completed_keys else "w"
        with open(csv_path, csv_mode, newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csv_mode == "w":
                writer.writeheader()

            ctx = mp.get_context("spawn")
            for i, params in enumerate(runs, start=1):
                if _params_key(params) in completed_keys:
                    is_baseline = (params == BASELINE)
                    label = "baseline" if is_baseline else (
                        next(p for p in PARAMS if params[p] != BASELINE[p])
                        + f"={params[next(p for p in PARAMS if params[p] != BASELINE[p])]}"
                    )
                    print(f"[{i}/{n_runs}] {label} — already done, skipping")
                    continue

                is_baseline = (params == BASELINE)
                label = "baseline" if is_baseline else (
                    next(p for p in PARAMS if params[p] != BASELINE[p])
                    + f"={params[next(p for p in PARAMS if params[p] != BASELINE[p])]}"
                )
                print(f"[{i}/{n_runs}] {label}")

                q = ctx.Queue()
                p = ctx.Process(
                    target=_run_worker,
                    args=(params, vardata, config_data, n_bootstrap, do_ci, q),
                )
                p.start()
                p.join()

                if p.exitcode == 0:
                    msg = q.get_nowait()
                else:
                    msg = ("error", f"process killed (exitcode={p.exitcode})", "")

                if msg[0] == "ok":
                    result = msg[1]
                else:
                    print(f"  ERROR: {msg[1]}")
                    if msg[2]:
                        print(msg[2])
                    result = dict(params)
                    for m in METRICS:
                        result[m] = np.nan

                results.append(result)
                if is_baseline and baseline_result is None:
                    baseline_result = result

                writer.writerow({k: result[k] for k in fieldnames})
                csvfile.flush()

        print(f"\nResults written to: {csv_path}")

    for group_name, params_in_group in PARAM_GROUPS.items():
        fig = sensitivity_matrix_plot(results, baseline_result, group_name, params_in_group,
                                config_plot.outdir)
    
        if config_plot.save:
            os.makedirs(config_plot.outdir, exist_ok=True)
            outpath = outpath_gen("/tmp", config_plot.outdir, f"sensititymatrix_{group_name}")
            fig.savefig(outpath + ".png", dpi=200)
            print(f"Sensitivity matrix saved to {outpath}.png")

    # elasticity_heatmap(results, baseline_result, config_plot.outdir)
