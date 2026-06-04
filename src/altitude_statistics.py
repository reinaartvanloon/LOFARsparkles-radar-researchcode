#!/usr/bin/env python3
"""
Compare radar statistics (Zh, W, HMC) between sparkles and other VHF sources
as a function of altitude layer.

Library module — contains config dataclasses, all computational functions,
plot functions, and main() for use by operational scripts.
"""
from __future__ import annotations

import os
import csv
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import xarray as xr
from dataclasses import dataclass
from typing import List, Optional
from scipy.stats import ks_2samp

from general import ConfigPlot
from read_RAD import ConfigDataRAD, ConfigMaskRADnearVHF
from plot_LOFAR import SparkleParams, ConfigLOFAR
from stats_sparklesRAD import statistics_sparkles_vs_other


# ── Constants ──────────────────────────────────────────────────────────────────

ALT_BANDS = [
    (8_000,  9_000,  "8–9 km"),
    (9_000,  10_000, "9–10 km"),
    (10_000, 11_000, "10–11 km"),
    (12_000, 13_000, "12–13 km"),
    (13_000, None,   ">13 km"),
]

N_BOOTSTRAP     = 2000
MAX_BOOT_SAMPLE = 50_000

_DEFAULT_RAD_VARS = ["dbzh", "wradh"]

METRIC_FIELDS = [
    f"{stat}_{var}"
    for var in _DEFAULT_RAD_VARS
    for stat in (
        "n_sparkle", "n_other",
        "mean_sparkle", "mean_other",
        "p25_sparkle",  "p75_sparkle",
        "p25_other",    "p75_other",
        "ks_stat",      "ks_stat_ci_lo",      "ks_stat_ci_hi",
        "cliffs_delta", "cliffs_delta_ci_lo", "cliffs_delta_ci_hi",
    )
]
FIELDNAMES = ["z_lo", "z_hi", "label"] + METRIC_FIELDS

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

C_SPARK  = "crimson"
C_OTHER  = "steelblue"
C_ZH     = "forestgreen"
C_W      = "goldenrod"
EDGE     = "black"
FONTSIZE = 10


# ── Config dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ConfigAltitudeStatsData:
    """Input data configuration for the altitude-band statistics."""
    rad_data_dirpath: str
    lofar_data_dirpath: str
    advection_reference_filepath: str
    hmc_msf_filepath: str
    lofar_file_list: List[str]
    varlist: List[str]
    sparkle_params: SparkleParams
    config_mask_rad: ConfigMaskRADnearVHF
    stormcode: str = "21C"
    rad_station: str = "asb"
    max_distance: float = 100e3


@dataclass
class ConfigAltitudeStatsPlot(ConfigPlot):
    """Output and run-control configuration for the altitude-band statistics."""
    outdir: str = None
    csv_path: str = None
    hmc_json_path: str = None
    plot_only: bool = False
    n_bootstrap: int = N_BOOTSTRAP
    confidence_interval: bool = False


# ── Private helpers ────────────────────────────────────────────────────────────

def _make_config_data_rad(cfg: ConfigAltitudeStatsData, varlist: list) -> ConfigDataRAD:
    return ConfigDataRAD(
        cfg.rad_data_dirpath,
        stormcode=cfg.stormcode,
        VHFtype="sparkles&otherVHF",
        RADstation=cfg.rad_station,
        RADvars=varlist,
        advection_reference_filepath=cfg.advection_reference_filepath,
        temp_reference_filepath=cfg.advection_reference_filepath,
        hmc_msf_filepath=cfg.hmc_msf_filepath,
        epsg=28992,
    )


def _label_panel(ax, letter):
    ax.set_title(f"({letter})", loc="left", fontsize=FONTSIZE + 1, fontweight="bold")


def _stats(vals: np.ndarray) -> dict:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return dict(n=0, mean=np.nan, p25=np.nan, p75=np.nan)
    return dict(n=len(vals), mean=np.mean(vals),
                p25=np.percentile(vals, 25), p75=np.percentile(vals, 75))


def _extract(ds, z_lo, z_hi, varkey, mask):
    z    = ds["z"]
    band = (z >= z_lo) if z_hi is None else ((z >= z_lo) & (z < z_hi))
    vals = ds[varkey].values[(mask & band).values]
    return vals[np.isfinite(vals)]


def _extract_hmc(ds_hmc, hmc_class, z_lo, z_hi, mask):
    z    = ds_hmc["z"]
    band = (z >= z_lo) if z_hi is None else ((z >= z_lo) & (z < z_hi))
    return hmc_class[(mask & band).values]


def _cliffs_delta(x, y):
    y_sorted  = np.sort(y)
    n_x_gt_y  = np.sum(np.searchsorted(y_sorted, x, side="left"))
    n_x_lt_y  = np.sum(len(y) - np.searchsorted(y_sorted, x, side="right"))
    return (n_x_gt_y - n_x_lt_y) / (len(x) * len(y))


def _ks_stat(x, y):
    return ks_2samp(x, y, method='asymp').statistic


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


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_ds(rad_vars: list, config_rad: ConfigDataRAD,
             config_data: ConfigAltitudeStatsData) -> xr.Dataset:
    datasets = []
    for lofar_file in config_data.lofar_file_list:
        print(lofar_file)
        config_LOFAR = ConfigLOFAR(
            stormcode      = config_data.stormcode,
            LOFAR_file     = lofar_file,
            datapath       = config_data.lofar_data_dirpath,
            sparkle_params = config_data.sparkle_params,
            max_distance   = config_data.max_distance,
        )
        ds = statistics_sparkles_vs_other(
            rad_vars,
            config_rad,
            config_LOFAR,
            config_data.config_mask_rad,
        )
        datasets.append(ds)
        gc.collect()
    ds_all = xr.concat(datasets, dim="radar_vol")
    ds_all.load()
    del datasets
    gc.collect()
    return ds_all


# ── Computation ────────────────────────────────────────────────────────────────

def compute_stats(
    vardata: dict,
    config_data: ConfigAltitudeStatsData,
    config_data_RAD: ConfigDataRAD,
    n_bootstrap: int = N_BOOTSTRAP,
    do_ci: bool = True,
) -> list[dict]:
    print("Loading radar data (dbzh, wradh)…")
    ds_all = _load_ds(config_data.varlist, config_data_RAD, config_data)

    dbzh_key  = vardata["dbzh"]["ODIM"]
    wradh_key = vardata["wradh"]["ODIM"]

    mask_sp = ds_all.mask_sparkles
    mask_ot = ds_all.mask_otherVHF & (~ds_all.mask_sparkles)
    rng     = np.random.default_rng(42)

    results = []
    for z_lo, z_hi, label in ALT_BANDS:
        print(f"  {label}: extracting…", flush=True)
        sp_zh = _extract(ds_all, z_lo, z_hi, dbzh_key,  mask_sp)
        ot_zh = _extract(ds_all, z_lo, z_hi, dbzh_key,  mask_ot)
        sp_wr = _extract(ds_all, z_lo, z_hi, wradh_key, mask_sp)
        ot_wr = _extract(ds_all, z_lo, z_hi, wradh_key, mask_ot)

        entry = {
            "z_lo":  float(z_lo),
            "z_hi":  float(z_hi) if z_hi is not None else np.nan,
            "label": label,
        }

        for var, sp_v, ot_v in [("dbzh", sp_zh, ot_zh), ("wradh", sp_wr, ot_wr)]:
            s, o = _stats(sp_v), _stats(ot_v)
            entry.update({
                f"n_sparkle_{var}":    s["n"],    f"n_other_{var}":      o["n"],
                f"mean_sparkle_{var}": s["mean"], f"mean_other_{var}":   o["mean"],
                f"p25_sparkle_{var}":  s["p25"],  f"p75_sparkle_{var}":  s["p75"],
                f"p25_other_{var}":    o["p25"],  f"p75_other_{var}":    o["p75"],
            })
            if len(sp_v) > 1 and len(ot_v) > 1:
                entry[f"ks_stat_{var}"]      = _ks_stat(sp_v, ot_v)
                entry[f"cliffs_delta_{var}"] = _cliffs_delta(sp_v, ot_v)
                if do_ci:
                    print(f"    bootstrap CI {var} KS…", flush=True)
                    kl, kh = _bootstrap_ci(_ks_stat,      sp_v, ot_v, n_boot=n_bootstrap, rng=rng)
                    entry[f"ks_stat_ci_lo_{var}"]      = kl
                    entry[f"ks_stat_ci_hi_{var}"]      = kh
                    print(f"    bootstrap CI {var} Cliff's δ…", flush=True)
                    cl, ch = _bootstrap_ci(_cliffs_delta, sp_v, ot_v, n_boot=n_bootstrap, rng=rng)
                    entry[f"cliffs_delta_ci_lo_{var}"] = cl
                    entry[f"cliffs_delta_ci_hi_{var}"] = ch
                else:
                    for suf in ("ks_stat_ci_lo", "ks_stat_ci_hi",
                                "cliffs_delta_ci_lo", "cliffs_delta_ci_hi"):
                        entry[f"{suf}_{var}"] = np.nan
            else:
                for suf in ("ks_stat", "ks_stat_ci_lo", "ks_stat_ci_hi",
                            "cliffs_delta", "cliffs_delta_ci_lo", "cliffs_delta_ci_hi"):
                    entry[f"{suf}_{var}"] = np.nan

        results.append(entry)
        print(f"  {label}: N_sp={entry['n_sparkle_dbzh']}, N_ot={entry['n_other_dbzh']}, "
              f"Zh_sp={entry['mean_sparkle_dbzh']:.1f} dBZ, "
              f"W_sp={entry['mean_sparkle_wradh']:.2f} m/s", flush=True)

    return results


def compute_hmc_stats(
    config_data: ConfigAltitudeStatsData,
    config_data_RAD_hmc: ConfigDataRAD,
) -> list[dict]:
    print("\nLoading radar data (HMC)…")
    ds_hmc = _load_ds(["hmc"], config_data_RAD_hmc, config_data)

    hmc_types = [str(t) for t in ds_hmc.hmc.values]
    n_types   = len(hmc_types)
    hmc_class = ds_hmc["HMC"].fillna(0).argmax("hmc").values
    mask_sp   = ds_hmc.mask_sparkles
    mask_ot   = ds_hmc.mask_otherVHF & (~ds_hmc.mask_sparkles)

    def _pct(cls):
        n = len(cls)
        if n == 0:
            return [np.nan] * n_types
        return (np.bincount(cls, minlength=n_types) / n * 100).tolist()

    hmc_results = []
    for z_lo, z_hi, label in ALT_BANDS:
        sp_cls = _extract_hmc(ds_hmc, hmc_class, z_lo, z_hi, mask_sp)
        ot_cls = _extract_hmc(ds_hmc, hmc_class, z_lo, z_hi, mask_ot)
        hmc_results.append({
            "z_lo":          float(z_lo),
            "z_hi":          float(z_hi) if z_hi is not None else np.nan,
            "label":         label,
            "hmc_types":     hmc_types,
            "hist_sparkles": _pct(sp_cls),
            "hist_other":    _pct(ot_cls),
        })

    return hmc_results


# ── CSV / JSON I/O ─────────────────────────────────────────────────────────────

def save_to_csv(results: list[dict], csv_path: str) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, np.nan) for k in FIELDNAMES})
    print(f"Results written to: {csv_path}")


def load_from_csv(csv_path: str) -> list[dict]:
    results = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {}
            for k, v in row.items():
                if k == "label":
                    entry[k] = v
                else:
                    try:
                        entry[k] = float(v)
                    except (ValueError, TypeError):
                        entry[k] = np.nan
            for m in METRIC_FIELDS:
                entry.setdefault(m, np.nan)
            results.append(entry)
    print(f"Loaded {len(results)} rows from {csv_path}")
    return results


def save_hmc_json(hmc_results: list[dict], json_path: str) -> None:
    with open(json_path, "w") as f:
        json.dump(hmc_results, f, indent=2)
    print(f"HMC results written to: {json_path}")


def load_hmc_json(json_path: str) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} HMC rows from {json_path}")
    return data


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(results: list[dict], outdir: str, hmc_results: list[dict] | None = None) -> None:
    """2 columns × 3 rows: (a) Nr. bins, (b) HMC, (c) W, (d) Zh, (e) Cliff's δ, (f) KS."""
    has_hmc = bool(hmc_results)
    x       = np.arange(len(results))
    xlabels = [r["label"] for r in results]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

    def _xticks(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=15, ha="right", fontsize=FONTSIZE)

    # ── (a) Nr. radar bins ────────────────────────────────────────────────────
    ax   = axes[0, 0]
    ax_r = ax.twinx()

    n_sp = [r["n_sparkle_dbzh"] for r in results]
    n_ot = [r["n_other_dbzh"]   for r in results]

    l1, = ax.plot(x, n_sp, color=C_SPARK, linestyle="-",  marker="o", linewidth=1.8,
                  markeredgecolor=EDGE, markeredgewidth=0.6, label="Sparkles")
    l2, = ax_r.plot(x, n_ot, color=C_OTHER, linestyle="--", marker="s", linewidth=1.8,
                    markeredgecolor=EDGE, markeredgewidth=0.6, label="Other VHF")
    _xticks(ax)
    ax.set_ylabel("Nr. radar bins (sparkles)",    color=C_SPARK, fontsize=FONTSIZE)
    ax_r.set_ylabel("Nr. radar bins (other VHF)", color=C_OTHER, fontsize=FONTSIZE)
    ax.tick_params(axis="y", colors=C_SPARK,  labelsize=FONTSIZE)
    ax_r.tick_params(axis="y", colors=C_OTHER, labelsize=FONTSIZE)
    ax.legend(handles=[l1, l2], fontsize=FONTSIZE, loc="upper right")
    _label_panel(ax, "a")

    # ── (c) and (d): mean + IQR shading ──────────────────────────────────────
    def _plot_iqr(ax, var, ylabel, letter):
        for group, color, marker, ls in [
            ("sparkle", C_SPARK, "o", "-"),
            ("other",   C_OTHER, "s", "--"),
        ]:
            means = np.array([r[f"mean_{group}_{var}"] for r in results])
            p25   = np.array([r[f"p25_{group}_{var}"]  for r in results])
            p75   = np.array([r[f"p75_{group}_{var}"]  for r in results])
            lbl   = "Sparkles" if group == "sparkle" else "Other VHF"
            ax.fill_between(x, p25, p75, alpha=0.25, color=color, zorder=1)
            ax.plot(x, means, color=color, linestyle=ls, marker=marker,
                    linewidth=1.8, markeredgecolor=EDGE, markeredgewidth=0.6, zorder=3, label=lbl)
        _xticks(ax)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE)
        ax.legend(fontsize=FONTSIZE, loc="upper right")
        _label_panel(ax, letter)

    _plot_iqr(axes[1, 1], "dbzh",  r"Z_h [dBZ]",               "d")
    _plot_iqr(axes[1, 0], "wradh", r"$W_{rad}$ [m s$^{-1}$]",  "c")

    # ── (b) HMC fractions ─────────────────────────────────────────────────────
    ax = axes[0, 1]
    if has_hmc:
        hmc_types = hmc_results[0]["hmc_types"]
        for i, hmc_type in enumerate(hmc_types):
            color      = HMC_COLORS.get(hmc_type, f"C{i}")
            spark_pcts = [r["hist_sparkles"][i] for r in hmc_results]
            other_pcts = [r["hist_other"][i]    for r in hmc_results]
            ax.plot(x, spark_pcts, color=color, linestyle="-",  marker="o", linewidth=1.5)
            ax.plot(x, other_pcts, color=color, linestyle="--", marker="s", linewidth=1.5)
        _xticks(ax)
        ax.set_ylabel("Fraction [%]", fontsize=FONTSIZE)
        hmc_handles = [
            mlines.Line2D([], [], color=HMC_COLORS.get(t, f"C{i}"), label=t)
            for i, t in enumerate(hmc_types)
        ]
        style_handles = [
            mlines.Line2D([], [], color="black", linestyle="-",  marker="o", label="Sparkles"),
            mlines.Line2D([], [], color="black", linestyle="--", marker="s", label="Other VHF"),
        ]
        ax.legend(handles=hmc_handles + style_handles, fontsize=FONTSIZE - 1,
                  ncol=2, loc="upper right")
        _label_panel(ax, "b")
    else:
        ax.set_visible(False)

    # ── (e) Cliff's delta ─────────────────────────────────────────────────────
    ax = axes[2, 0]
    for var, color, marker, lbl in [
        ("dbzh",  C_ZH, "o", r"Z_h"),
        ("wradh", C_W,  "s", r"$W_{rad}$"),
    ]:
        cds   = np.array([r[f"cliffs_delta_{var}"]        for r in results])
        ci_lo = np.array([r[f"cliffs_delta_ci_lo_{var}"]  for r in results])
        ci_hi = np.array([r[f"cliffs_delta_ci_hi_{var}"]  for r in results])
        if not all(np.isnan(v) for v in ci_lo):
            ax.fill_between(x, ci_lo, ci_hi, alpha=0.25, color=color)
        ax.plot(x, cds, color=color, marker=marker, linestyle="-",
                linewidth=1.8, markeredgecolor=EDGE, markeredgewidth=0.6, label=lbl)
    ax.axhline(0, color="gray", linewidth=0.9, linestyle="--", alpha=0.7)
    _xticks(ax)
    ax.set_xlabel("Altitude layer", fontsize=FONTSIZE)
    ax.set_ylabel(r"Cliff's $\delta$", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE, loc="upper right")
    _label_panel(ax, "e")

    # ── (f) KS statistic D ───────────────────────────────────────────────────
    ax = axes[2, 1]
    for var, color, marker, lbl in [
        ("dbzh",  C_ZH, "o", r"Z_h"),
        ("wradh", C_W,  "s", r"$W_{rad}$"),
    ]:
        ks_vals = np.array([r[f"ks_stat_{var}"]        for r in results])
        ci_lo   = np.array([r[f"ks_stat_ci_lo_{var}"]  for r in results])
        ci_hi   = np.array([r[f"ks_stat_ci_hi_{var}"]  for r in results])
        if not all(np.isnan(v) for v in ci_lo):
            ax.fill_between(x, ci_lo, ci_hi, alpha=0.25, color=color)
        ax.plot(x, ks_vals, color=color, marker=marker, linestyle="-",
                linewidth=1.8, markeredgecolor=EDGE, markeredgewidth=0.6, label=lbl)
    _xticks(ax)
    ax.set_xlabel("Altitude layer", fontsize=FONTSIZE)
    ax.set_ylabel("KS statistic", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE, loc="upper right")
    _label_panel(ax, "f")

    outpath = os.path.join(outdir, "altitude_statistics.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {outpath}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(config_data: ConfigAltitudeStatsData, config_plot: ConfigAltitudeStatsPlot):
    os.makedirs(config_plot.outdir, exist_ok=True)

    csv_path = config_plot.csv_path      or os.path.join(config_plot.outdir, "altitude_statistics.csv")
    hmc_path = config_plot.hmc_json_path or os.path.join(config_plot.outdir, "altitude_statistics_hmc.json")

    config_rad     = _make_config_data_rad(config_data, config_data.varlist)
    config_rad_hmc = _make_config_data_rad(config_data, ["hmc"])

    if config_plot.plot_only:
        print(f"Plot-only mode: loading results from {csv_path}")
        results = load_from_csv(csv_path)
        hmc_results = None
        if os.path.exists(hmc_path):
            hmc_results = load_hmc_json(hmc_path)
        else:
            print(f"HMC JSON not found ({hmc_path}); skipping HMC panel.")
    else:
        with open(os.path.join(config_data.rad_data_dirpath, "variables_metadata.json")) as f:
            vardata = json.load(f)
        results = compute_stats(
            vardata, config_data, config_rad,
            n_bootstrap=config_plot.n_bootstrap,
            do_ci=config_plot.confidence_interval,
        )
        save_to_csv(results, csv_path)
        hmc_results = compute_hmc_stats(config_data, config_rad_hmc)
        save_hmc_json(hmc_results, hmc_path)

    plot(results, config_plot.outdir, hmc_results)
