#!/usr/bin/env python3
"""
Sensitivity study for the RADnearVHF_radius parameter in ConfigMaskRADnearVHF.

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
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import List, Optional
from scipy.stats import ks_2samp

from general import ConfigPlot
from read_RAD import (
    ConfigDataRAD, ConfigMaskRADnearVHF,
    get_data_RADandLOFAR, add_mask_RADnearVHF,
)
from plot_LOFAR import SparkleParams, ConfigLOFAR
from stats_sparklesRAD import cliffs_delta


# ── Sensitivity sweep constants ────────────────────────────────────────────────

BASELINE_RADIUS = 2000  # [m]
RADIUS_LEVELS   = [500, 1000, 1500, 2000, 3000, 4000]  # [m]

VAR_LABELS = {
    "dbzh":  r"$Z_h$ [dBZ]",
    "wradh": r"$W_{rad}$ [m s$^{-1}$]",
}

N_BOOTSTRAP     = 2000
CI_ALPHA        = 0.95
MAX_BOOT_SAMPLE = 50_000

_DEFAULT_VARLIST = ["dbzh", "wradh"]

METRIC_FIELDS = [
    f"{stat}_{var}"
    for var in _DEFAULT_VARLIST
    for stat in (
        "mean_sparkle", "mean_other",
        "p25_sparkle",  "p75_sparkle",
        "p25_other",    "p75_other",
        "n_sparkle",    "n_other",
        "cliffs_delta",
        "cliffs_delta_ci_lo", "cliffs_delta_ci_hi",
        "ks_stat",
        "ks_stat_ci_lo", "ks_stat_ci_hi",
    )
]
FIELDNAMES = ["radius"] + METRIC_FIELDS

HMC_COLORS = {
    "LR": "lemonchiffon",
    "MR": "khaki",
    "HR": "gold",
    "LD": "goldenrod",
    "HL": "coral",
    "RH": "chocolate",
    "GH": "brown",
    "DS": "lightblue",
    "WS": "steelblue",
    "HC": "lightpink",
    "VC": "violet",
    "NP": "silver",
}

C_SPARK = "crimson"
C_OTHER = "steelblue"
C_ZH    = "forestgreen"
C_W     = "goldenrod"
EDGE    = "black"


# ── Config dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ConfigSensitivityRadiusData:
    """Input data configuration for the RADnearVHF_radius sensitivity study."""
    rad_data_dirpath: str
    lofar_data_dirpath: str
    advection_reference_filepath: str
    hmc_msf_filepath: str
    lofar_file_list: List[str]
    varlist: List[str]
    sparkle_params: SparkleParams
    stormcode: str = "21C"
    rad_station: str = "asb"
    max_distance: float = 100e3


@dataclass
class ConfigSensitivityRadiusPlot(ConfigPlot):
    """Output and run-control configuration for the radius sensitivity study."""
    outdir: str = None
    csv_path: str = None
    hmc_json_path: str = None
    plot_only: bool = False
    n_bootstrap: int = N_BOOTSTRAP
    confidence_interval: bool = False


# ── Private helpers ────────────────────────────────────────────────────────────

def _make_config_data_rad(cfg: ConfigSensitivityRadiusData) -> ConfigDataRAD:
    return ConfigDataRAD(
        cfg.rad_data_dirpath,
        stormcode=cfg.stormcode,
        VHFtype="sparkles&otherVHF",
        RADstation=cfg.rad_station,
        RADvars=cfg.varlist + ["hmc"],
        advection_reference_filepath=cfg.advection_reference_filepath,
        temp_reference_filepath=cfg.advection_reference_filepath,
        hmc_msf_filepath=cfg.hmc_msf_filepath,
        epsg=28992,
    )


def _ks_stat(x, y):
    return ks_2samp(x, y, method='asymp').statistic


def _cliffs_delta_func(x, y):
    y_sorted      = np.sort(y)
    n_x_gt_y      = np.searchsorted(y_sorted, x, side="left")
    equal_count   = np.searchsorted(y_sorted, x, side="right") - n_x_gt_y
    n_x_lt_y      = len(y) - n_x_gt_y - equal_count
    return (np.sum(n_x_gt_y) - np.sum(n_x_lt_y)) / (len(x) * len(y))


def _bootstrap_ci(func, x, y, n_boot=N_BOOTSTRAP, ci=CI_ALPHA, rng=None):
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


def _vline_and_baseline(ax, radii, baseline_val):
    ax.axvline(baseline_val, color="k", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_xticks(radii)
    ax.tick_params(labelsize=9)


def _label_panel(ax, letter):
    ax.set_title(f"({letter})", loc="left", fontsize=12, fontweight="bold")


# ── Model evaluation ───────────────────────────────────────────────────────────

def run_one_combined(
    radius: float,
    config_data_RAD: ConfigDataRAD,
    config_LOFAR: ConfigLOFAR,
    varlist: list,
    vardata: dict,
    n_bootstrap: int = N_BOOTSTRAP,
    do_ci: bool = True,
) -> tuple[dict, dict]:
    """Load each file once and compute both regular-var and HMC metrics."""
    config_mask = ConfigMaskRADnearVHF(
        RADnearVHF_radius=radius,
        RADalt_threshold=8e3,
        RADdbzh_threshold=0,
    )

    per_var = {var: {"sparkles": [], "other": []} for var in varlist}
    hmc_types      = None
    class_sparkles = []
    class_other    = []

    for file in config_LOFAR.LOFAR_file_list:
        print(file)
        config_LOFAR.LOFAR_file = file
        data = get_data_RADandLOFAR(config_data_RAD, config_LOFAR=config_LOFAR)
        data.RAD = add_mask_RADnearVHF(data.RAD, data.LOFAR, config=config_mask)

        ds = data.RAD.ds
        if hmc_types is None:
            hmc_types = [str(t) for t in ds.hmc.values]

        mask_sp = ds.mask_sparkles.values.ravel()
        mask_ot = ds.mask_otherVHF.values.ravel() & ~mask_sp

        for var in varlist:
            odim_key = vardata[var]["ODIM"]
            vals = ds[odim_key].values.ravel()
            per_var[var]["sparkles"].append(vals[mask_sp])
            per_var[var]["other"].append(vals[mask_ot])

        hmc_class = ds["HMC"].fillna(0).argmax("hmc").values.ravel()
        class_sparkles.append(hmc_class[mask_sp])
        class_other.append(hmc_class[mask_ot])

        del data, ds, mask_sp, mask_ot, hmc_class
        gc.collect()

    result = {"radius": radius}
    rng = np.random.default_rng(42)

    for var in varlist:
        v_spark = np.concatenate(per_var[var]["sparkles"])
        v_spark = v_spark[np.isfinite(v_spark)]
        v_other = np.concatenate(per_var[var]["other"])
        v_other = v_other[np.isfinite(v_other)]

        result[f"n_sparkle_{var}"]    = len(v_spark)
        result[f"n_other_{var}"]      = len(v_other)
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
                result[f"ks_stat_ci_lo_{var}"] = ks_lo
                result[f"ks_stat_ci_hi_{var}"] = ks_hi
            else:
                for suffix in ("cliffs_delta_ci_lo", "cliffs_delta_ci_hi",
                               "ks_stat_ci_lo", "ks_stat_ci_hi"):
                    result[f"{suffix}_{var}"] = np.nan
        else:
            for suffix in ("cliffs_delta", "cliffs_delta_ci_lo", "cliffs_delta_ci_hi",
                           "ks_stat", "ks_stat_ci_lo", "ks_stat_ci_hi"):
                result[f"{suffix}_{var}"] = np.nan

    n_types = len(hmc_types)

    def _hist_pct(arrays):
        combined = np.concatenate(arrays)
        n = len(combined)
        if n == 0:
            return [np.nan] * n_types
        return (np.bincount(combined, minlength=n_types) / n * 100).tolist()

    hmc_result = {
        "radius":        radius,
        "hmc_types":     hmc_types,
        "hist_sparkles": _hist_pct(class_sparkles),
        "hist_other":    _hist_pct(class_other),
    }

    return result, hmc_result


# ── CSV I/O ────────────────────────────────────────────────────────────────────

def load_results_from_csv(csv_path: str) -> list[dict]:
    results = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        csv_columns = set(reader.fieldnames or [])
        for row in reader:
            result = {}
            for k in csv_columns:
                try:
                    result[k] = float(row[k])
                except (ValueError, TypeError):
                    result[k] = np.nan
            for m in METRIC_FIELDS:
                result.setdefault(m, np.nan)
            results.append(result)
    print(f"Loaded {len(results)} rows from {csv_path}")
    return results


# ── HMC JSON I/O ───────────────────────────────────────────────────────────────

def save_hmc_results(hmc_results: list[dict], json_path: str) -> None:
    with open(json_path, "w") as f:
        json.dump(hmc_results, f, indent=2)
    print(f"HMC results written to: {json_path}")


def load_hmc_results(json_path: str) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} HMC rows from {json_path}")
    return data


# ── Plotting ───────────────────────────────────────────────────────────────────

def sensitivity_plot(results: list[dict], outdir: str, hmc_results: list[dict] | None = None):
    """
    3-row × 2-column figure for the radius sensitivity:
      (a) Nr. radar bins — sparkles left y, other VHF right y (twinx)
      (b) HMC type fractions
      (c) Mean W  with IQR shading
      (d) Mean Zh with IQR shading
      (e) Cliff's delta for Zh and W with 95% bootstrap CI
      (f) KS statistic for Zh and W with 95% bootstrap CI
    """
    radii = [r["radius"] for r in results]
    base  = next((r for r in results if r["radius"] == BASELINE_RADIUS), None)

    has_hmc = bool(hmc_results)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

    # ── Panel (a): Nr. radar bins ─────────────────────────────────────────────
    ax   = axes[0, 0]
    ax_r = ax.twinx()

    n_sp = [r["n_sparkle_dbzh"] for r in results]
    n_ot = [r["n_other_dbzh"]   for r in results]

    l1, = ax.plot(radii, n_sp, color=C_SPARK, linestyle="-", marker="o", linewidth=1.8,
                  markeredgecolor=EDGE, markeredgewidth=0.6, label="Sparkles")
    l2, = ax_r.plot(radii, n_ot, color=C_OTHER, linestyle="--", marker="s", linewidth=1.8,
                    markeredgecolor=EDGE, markeredgewidth=0.6, label="Other VHF")
    if base is not None:
        ax.plot(BASELINE_RADIUS, base["n_sparkle_dbzh"],
                marker="o", color=C_SPARK, markersize=10, zorder=5,
                markeredgecolor="k", markeredgewidth=1.2)
        ax_r.plot(BASELINE_RADIUS, base["n_other_dbzh"],
                  marker="s", color=C_OTHER, markersize=10, zorder=5,
                  markeredgecolor="k", markeredgewidth=1.2)

    _vline_and_baseline(ax, radii, BASELINE_RADIUS)
    ax.set_ylabel("Nr. radar bins (sparkles)",  color=C_SPARK, fontsize=10)
    ax_r.set_ylabel("Nr. radar bins (other VHF)", color=C_OTHER, fontsize=10)
    ax.tick_params(axis="y", colors=C_SPARK)
    ax_r.tick_params(axis="y", colors=C_OTHER)
    ax.legend(handles=[l1, l2], fontsize=9, loc="upper left")
    _label_panel(ax, "a")

    # ── Panels (c) and (d): mean + IQR shading ───────────────────────────────
    def _plot_iqr(ax, var, ylabel, letter):
        for group, color, marker, ls in [
            ("sparkle", C_SPARK, "o", "-"),
            ("other",   C_OTHER, "s", "--"),
        ]:
            means = [r[f"mean_{group}_{var}"] for r in results]
            p25   = [r[f"p25_{group}_{var}"]  for r in results]
            p75   = [r[f"p75_{group}_{var}"]  for r in results]
            lbl   = "Sparkles" if group == "sparkle" else "Other VHF"
            ax.fill_between(radii, p25, p75, alpha=0.25, color=color, zorder=1)
            ax.plot(radii, means, color=color, linestyle=ls, marker=marker,
                    linewidth=1.8, markeredgecolor=EDGE, markeredgewidth=0.6, zorder=3, label=lbl)
            if base is not None:
                ax.plot(BASELINE_RADIUS, base[f"mean_{group}_{var}"],
                        marker=marker, color=color, markersize=10, zorder=5,
                        markeredgecolor="k", markeredgewidth=1.2)
        _vline_and_baseline(ax, radii, BASELINE_RADIUS)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        _label_panel(ax, letter)

    _plot_iqr(axes[1, 1], "dbzh",  r"$Z_h$ [dBZ]",           "d")
    _plot_iqr(axes[1, 0], "wradh", r"$W_{rad}$ [m s$^{-1}$]", "c")

    # ── Panel (f): KS statistic D ─────────────────────────────────────────────
    ax = axes[2, 1]
    for var, color, marker, lbl in [
        ("dbzh",  C_ZH, "o", r"$Z_h$"),
        ("wradh", C_W,  "s", r"$W_{rad}$"),
    ]:
        ks_vals = [r[f"ks_stat_{var}"]        for r in results]
        ci_lo   = [r[f"ks_stat_ci_lo_{var}"]  for r in results]
        ci_hi   = [r[f"ks_stat_ci_hi_{var}"]  for r in results]
        if not all(np.isnan(v) for v in ci_lo):
            ax.fill_between(radii, ci_lo, ci_hi, alpha=0.25, color=color)
        ax.plot(radii, ks_vals, color=color, marker=marker, linestyle="-",
                linewidth=1.8, markeredgecolor=EDGE, markeredgewidth=0.6, label=lbl)
        if base is not None:
            ax.plot(BASELINE_RADIUS, base[f"ks_stat_{var}"],
                    marker=marker, color=color, markersize=10, zorder=5,
                    markeredgecolor="k", markeredgewidth=1.2)
    _vline_and_baseline(ax, radii, BASELINE_RADIUS)
    ax.set_ylabel("KS statistic", fontsize=10)
    ax.set_xlabel("r (m)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    _label_panel(ax, "f")

    # ── Panel (e): Cliff's delta ──────────────────────────────────────────────
    ax = axes[2, 0]
    for var, color, marker, lbl in [
        ("dbzh",  C_ZH, "o", r"$Z_h$"),
        ("wradh", C_W,  "s", r"$W_{rad}$"),
    ]:
        cds   = [r[f"cliffs_delta_{var}"]        for r in results]
        ci_lo = [r[f"cliffs_delta_ci_lo_{var}"]  for r in results]
        ci_hi = [r[f"cliffs_delta_ci_hi_{var}"]  for r in results]
        if not all(np.isnan(v) for v in ci_lo):
            ax.fill_between(radii, ci_lo, ci_hi, alpha=0.25, color=color)
        ax.plot(radii, cds, color=color, marker=marker, linestyle="-",
                linewidth=1.8, markeredgecolor=EDGE, markeredgewidth=0.6, label=lbl)
        if base is not None:
            ax.plot(BASELINE_RADIUS, base[f"cliffs_delta_{var}"],
                    marker=marker, color=color, markersize=10, zorder=5,
                    markeredgecolor="k", markeredgewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.9, linestyle="--", alpha=0.7)
    _vline_and_baseline(ax, radii, BASELINE_RADIUS)
    ax.set_ylabel(r"Cliff's $\delta$", fontsize=10)
    ax.set_xlabel("r (m)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    _label_panel(ax, "e")

    # ── Panel (b): HMC combined ───────────────────────────────────────────────
    ax = axes[0, 1]
    if has_hmc:
        hmc_radii = [r["radius"] for r in hmc_results]
        hmc_types = hmc_results[0]["hmc_types"]

        for i, hmc_type in enumerate(hmc_types):
            color = HMC_COLORS.get(hmc_type, f"C{i}")
            spark_pcts = [r["hist_sparkles"][i] for r in hmc_results]
            other_pcts = [r["hist_other"][i]    for r in hmc_results]
            ax.plot(hmc_radii, spark_pcts, color=color, linestyle="-",  marker="o",
                    linewidth=1.5)
            ax.plot(hmc_radii, other_pcts, color=color, linestyle="--", marker="s",
                    linewidth=1.5)

        _vline_and_baseline(ax, hmc_radii, BASELINE_RADIUS)
        ax.set_ylabel("HMC fraction [%]", fontsize=10)

        hmc_handles = [
            mlines.Line2D([], [], color=HMC_COLORS.get(t, f"C{i}"), label=t)
            for i, t in enumerate(hmc_types)
        ]
        style_handles = [
            mlines.Line2D([], [], color="black", linestyle="-",  marker="o", label="Sparkles"),
            mlines.Line2D([], [], color="black", linestyle="--", marker="s", label="Other VHF"),
        ]
        ax.legend(handles=hmc_handles + style_handles, fontsize=8, ncol=2, loc="upper right")
        _label_panel(ax, "b")
    else:
        ax.set_visible(False)

    fig.suptitle(
        "Sensitivity to RADnearVHF_radius  (DBSCAN parameters fixed at baseline)",
        fontsize=11,
    )

    outpath = os.path.join(outdir, "sensitivity_r-near-RAD.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Sensitivity plot saved to: {outpath}")
    plt.show()


# ── Per-radius subprocess worker ───────────────────────────────────────────────

def _radius_worker(radius, config_data_RAD, config_LOFAR, varlist,
                   vardata, n_bootstrap, do_ci, result_queue):
    """Runs in a child process so its memory is fully released on exit."""
    try:
        result, hmc_result = run_one_combined(
            radius, config_data_RAD, config_LOFAR, varlist,
            vardata, n_bootstrap=n_bootstrap, do_ci=do_ci,
        )
        result_queue.put(("ok", result, hmc_result))
    except Exception as e:
        import traceback
        result_queue.put(("error", str(e), traceback.format_exc()))


# ── Main ───────────────────────────────────────────────────────────────────────

def main(config_data: ConfigSensitivityRadiusData, config_plot: ConfigSensitivityRadiusPlot):
    os.makedirs(config_plot.outdir, exist_ok=True)

    csv_path    = config_plot.csv_path      or os.path.join(config_plot.outdir, "sensitivity_r-near-RAD_results.csv")
    hmc_path    = config_plot.hmc_json_path or os.path.join(config_plot.outdir, "sensitivity_r-near-RAD_hmc.json")
    n_bootstrap = config_plot.n_bootstrap
    do_ci       = config_plot.confidence_interval

    if config_plot.plot_only:
        print(f"Plot-only mode: loading results from {csv_path}")
        results = load_results_from_csv(csv_path)

        hmc_results = None
        if os.path.exists(hmc_path):
            hmc_results = load_hmc_results(hmc_path)
        else:
            print(f"HMC JSON not found ({hmc_path}); skipping HMC panel.")

    else:
        with open(os.path.join(config_data.rad_data_dirpath, "variables_metadata.json")) as f:
            vardata = json.load(f)

        n_runs = len(RADIUS_LEVELS)
        print(f"Radius sweep: {n_runs} evaluations over {RADIUS_LEVELS} m")
        print(f"Baseline radius: {BASELINE_RADIUS} m")
        print(f"Bootstrap iterations: {n_bootstrap if do_ci else 'disabled (--confidence-interval not set)'}\n")

        config_data_RAD = _make_config_data_rad(config_data)
        config_LOFAR = ConfigLOFAR(
            stormcode      = config_data.stormcode,
            datapath       = config_data.lofar_data_dirpath,
            sparkle_params = config_data.sparkle_params,
            max_distance   = config_data.max_distance,
        )
        config_LOFAR.LOFAR_file_list = config_data.lofar_file_list

        completed_csv_radii = set()
        results = []
        if os.path.exists(csv_path):
            try:
                results = load_results_from_csv(csv_path)
                completed_csv_radii = {float(r["radius"]) for r in results}
                print(f"CSV: {len(completed_csv_radii)} radii done "
                      f"({sorted(completed_csv_radii)} m).")
            except Exception as exc:
                print(f"Could not read existing CSV ({exc}); starting fresh.")
                results = []

        hmc_results = []
        completed_hmc_radii = set()
        if os.path.exists(hmc_path):
            try:
                raw = load_hmc_results(hmc_path)
                seen = {}
                for entry in raw:
                    seen[float(entry["radius"])] = entry
                hmc_results = list(seen.values())
                completed_hmc_radii = set(seen.keys())
                if len(raw) != len(hmc_results):
                    print(f"HMC JSON: removed {len(raw) - len(hmc_results)} duplicate "
                          f"entries; {len(hmc_results)} unique radii kept.")
                else:
                    print(f"HMC JSON: {len(hmc_results)} radii loaded.")
            except Exception:
                hmc_results = []

        completed_radii = completed_csv_radii & completed_hmc_radii
        if completed_radii:
            print(f"Resuming from: skipping {sorted(completed_radii)} m.")

        csv_mode = "a" if completed_radii else "w"
        with open(csv_path, csv_mode, newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            if csv_mode == "w":
                writer.writeheader()

            ctx = mp.get_context("spawn")
            for i, radius in enumerate(RADIUS_LEVELS, start=1):
                if float(radius) in completed_radii:
                    print(f"[{i}/{n_runs}] radius = {radius} m — already done, skipping")
                    continue
                print(f"[{i}/{n_runs}] radius = {radius} m")
                q = ctx.Queue()
                p = ctx.Process(
                    target=_radius_worker,
                    args=(radius, config_data_RAD, config_LOFAR,
                          config_data.varlist, vardata, n_bootstrap, do_ci, q),
                )
                p.start()
                p.join()

                if p.exitcode == 0:
                    msg = q.get_nowait()
                else:
                    msg = ("error", f"process killed (exitcode={p.exitcode})", "")

                if msg[0] == "ok":
                    result, hmc_result = msg[1], msg[2]
                else:
                    print(f"  ERROR: {msg[1]}")
                    if msg[2]:
                        print(msg[2])
                    result = {"radius": radius}
                    for m in METRIC_FIELDS:
                        result[m] = np.nan
                    hmc_result = {"radius": radius, "hmc_types": [],
                                  "hist_sparkles": [], "hist_other": []}

                results.append(result)
                writer.writerow({k: result[k] for k in FIELDNAMES})
                csvfile.flush()
                hmc_results.append(hmc_result)

        print(f"\nResults written to: {csv_path}")
        save_hmc_results(hmc_results, hmc_path)

    sensitivity_plot(results, config_plot.outdir, hmc_results=hmc_results)
