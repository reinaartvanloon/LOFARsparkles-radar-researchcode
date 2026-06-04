#!/usr/bin/env python3
"""
Compare ERA5 and pySTEPS advection in the sparkle statistics pipeline.

Runs stats_sparklesRAD.main() twice (once per advection source) and saves
all figures under {outdir_base}/era5/ and {outdir_base}/pysteps/ respectively.

Then loads each LOFAR event's radar scan independently, applies both
advection methods, and produces a histogram of the per-bin advection-distance
difference (ERA5 − pySTEPS) saved as {outdir_base}/{outname}.png.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional

from general import ConfigPlot, outpath_gen, open_reference_file
from read_RAD import (
    ConfigDataRAD, ConfigMaskRADnearVHF,
    load_radar_data, get_metadata_RADvars,
)
from plot_LOFAR import SparkleParams, ConfigLOFAR, get_datetime_LOFAR_from_metadata
import stats_sparklesRAD
from stats_sparklesRAD import ConfigPlotSparkleStats


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConfigAdvectionData:
    """Configuration for input data and processing parameters."""
    rad_data_dirpath: str
    lofar_data_dirpath: str
    era5_filepath: str
    pysteps_motion_filepath: Optional[str]
    hmc_msf_filepath: str
    lofar_file_list: List[str]
    varlist: List[str]
    sparkle_params: SparkleParams
    config_mask_rad: ConfigMaskRADnearVHF
    stormcode: str = "21C"
    rad_station: str = "asb"
    max_distance: float = 100e3

    def __post_init__(self):
        if self.pysteps_motion_filepath is None or not os.path.isfile(self.pysteps_motion_filepath):
            raise FileNotFoundError(
                f"pySTEPS motion field file not found: {self.pysteps_motion_filepath!r}\n"
                "Compute it first with: scripts/compute_pystep_motion_fields.py"
            )


@dataclass
class ConfigAdvectionPlot(ConfigPlot):
    """Configuration for the advection comparison output."""
    outdir_base: str = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_config_lofar(cfg: ConfigAdvectionData) -> ConfigLOFAR:
    return ConfigLOFAR(
        stormcode      = cfg.stormcode,
        datapath       = cfg.lofar_data_dirpath,
        sparkle_params = cfg.sparkle_params,
        max_distance   = cfg.max_distance,
    )


def _make_config_data_rad(
    cfg: ConfigAdvectionData,
    adv_filepath: Optional[str],
    adv_kind: str,
) -> ConfigDataRAD:
    return ConfigDataRAD(
        data_dirpath                 = cfg.rad_data_dirpath,
        stormcode                    = cfg.stormcode,
        VHFtype                      = "sparkles&otherVHF",
        RADstation                   = cfg.rad_station,
        RADvars                      = cfg.varlist,
        advection_reference_filepath = adv_filepath,
        advection_reference_kind     = adv_kind,
        temp_reference_filepath      = cfg.era5_filepath,
        hmc_msf_filepath             = cfg.hmc_msf_filepath,
        epsg                         = 28992,
    )


def _make_config_plots(
    cfg_plot: ConfigAdvectionPlot,
    cfg_data: ConfigAdvectionData,
    outdir: str,
) -> ConfigPlotSparkleStats:
    return ConfigPlotSparkleStats(
        LOFARfile_list = cfg_data.lofar_file_list,
        varlist        = cfg_data.varlist,
        together       = True,
        save           = True,
        outdir         = outdir,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_data: ConfigAdvectionData, config_plot: ConfigAdvectionPlot):
    # ── 1. ERA5 statistics run ─────────────────────────────────────────────
    print("=" * 60)
    print("Running stats_sparklesRAD.main() with ERA5 advection")
    print("=" * 60)
    outdir_era5 = os.path.join(config_plot.outdir_base, "era5")
    stats_sparklesRAD.main(
        _make_config_plots(config_plot, config_data, outdir_era5),
        _make_config_data_rad(config_data, config_data.era5_filepath, "era5"),
        config_data.sparkle_params,
        _make_config_lofar(config_data),
        config_data.config_mask_rad,
    )

    # ── 2. pySTEPS statistics run ──────────────────────────────────────────
    print("=" * 60)
    print("Running stats_sparklesRAD.main() with pySTEPS advection")
    print("=" * 60)
    outdir_pysteps = os.path.join(config_plot.outdir_base, "pysteps")
    stats_sparklesRAD.main(
        _make_config_plots(config_plot, config_data, outdir_pysteps),
        _make_config_data_rad(config_data, config_data.pysteps_motion_filepath, "pysteps"),
        config_data.sparkle_params,
        _make_config_lofar(config_data),
        config_data.config_mask_rad,
    )

    # ── 3. Advection distance comparison ───────────────────────────────────
    print("=" * 60)
    print("Computing advection distance differences")
    print("=" * 60)

    ds_era5    = open_reference_file(config_data.era5_filepath,    kind="era5")
    ds_pysteps = open_reference_file(config_data.pysteps_motion_filepath, kind="pysteps")

    variable_data = get_metadata_RADvars(config_data.rad_data_dirpath)

    _load_config = _make_config_data_rad(config_data, adv_filepath=None, adv_kind="era5")

    all_diffs_m  = []
    all_dx_era5  = []
    all_dy_era5  = []
    all_dx_pys   = []
    all_dy_pys   = []

    for lofar_file in config_data.lofar_file_list:
        print(f"  Processing {lofar_file}")
        cfg_lofar = _make_config_lofar(config_data)
        cfg_lofar.LOFAR_file = lofar_file

        try:
            dt_target, dt_range = get_datetime_LOFAR_from_metadata(cfg_lofar)
        except Exception as exc:
            print(f"    Skipped – cannot read metadata: {exc}")
            continue

        try:
            radar = load_radar_data(_load_config, variable_data, dt_target, dt_range)
        except Exception as exc:
            print(f"    Skipped – radar load failed: {exc}")
            continue

        x_orig = radar.ds["x"].copy()
        y_orig = radar.ds["y"].copy()

        try:
            radar.advect(dt_target, ds_era5)
            dx_era5_vals = radar.ds["dx_adv"].values.ravel()
            dy_era5_vals = radar.ds["dy_adv"].values.ravel()
        except Exception as exc:
            print(f"    ERA5 advect failed: {exc}")
            continue
        finally:
            radar.ds["x"] = x_orig
            radar.ds["y"] = y_orig

        try:
            radar.advect(dt_target, ds_pysteps)
            dx_pys_vals = radar.ds["dx_adv"].values.ravel()
            dy_pys_vals = radar.ds["dy_adv"].values.ravel()
        except Exception as exc:
            print(f"    pySTEPS advect failed: {exc}")
            continue

        dist_era5 = np.sqrt(dx_era5_vals**2 + dy_era5_vals**2)
        dist_pys  = np.sqrt(dx_pys_vals**2  + dy_pys_vals**2)
        diff      = dist_era5 - dist_pys

        finite = np.isfinite(diff)
        all_diffs_m.append(diff[finite])
        all_dx_era5.append(dx_era5_vals[finite])
        all_dy_era5.append(dy_era5_vals[finite])
        all_dx_pys.append(dx_pys_vals[finite])
        all_dy_pys.append(dy_pys_vals[finite])

        n = finite.sum()
        print(
            f"    n={n:,}  mean_diff={diff[finite].mean()/1e3:.3f} km  "
            f"std={diff[finite].std()/1e3:.3f} km"
        )

    if not all_diffs_m:
        print("No data collected – histogram skipped.")
        return

    all_diffs_flat   = np.concatenate(all_diffs_m)  / 1e3   # → km
    all_dx_era5_flat = np.concatenate(all_dx_era5)  / 1e3
    all_dy_era5_flat = np.concatenate(all_dy_era5)  / 1e3
    all_dx_pys_flat  = np.concatenate(all_dx_pys)   / 1e3
    all_dy_pys_flat  = np.concatenate(all_dy_pys)   / 1e3

    n_bins    = len(all_diffs_m)
    n_total   = all_diffs_flat.size
    mean_diff = all_diffs_flat.mean()
    std_diff  = all_diffs_flat.std()
    print(
        f"\nAggregated: n={n_total:,}  "
        f"mean(ERA5−pySTEPS)={mean_diff:.3f} km  "
        f"std={std_diff:.3f} km"
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), dpi=200, sharey=True)

    ax = axes[0]
    clip = np.percentile(np.abs(all_diffs_flat), 99)
    bins = np.linspace(-clip, clip, 80)
    ax.hist(all_diffs_flat, bins=bins, color="steelblue", edgecolor="none")
    ax.axvline(0,          color="k",   linewidth=0.8, linestyle="--")
    ax.axvline(mean_diff,  color="red", linewidth=1.0, linestyle="-",
               label=f"mean = {mean_diff:.3f} km")
    ax.set_xlabel("Distance difference (ERA5 − pySTEPS) [km]")
    ax.set_ylabel("Radar bin count")
    ax.set_title("Advection distance difference")
    ax.legend(fontsize=8)

    ax = axes[1]
    dx_diff = all_dx_era5_flat - all_dx_pys_flat
    clip2 = np.percentile(np.abs(dx_diff), 99)
    bins2 = np.linspace(-clip2, clip2, 80)
    ax.hist(dx_diff, bins=bins2, color="coral", edgecolor="none")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.axvline(dx_diff.mean(), color="red", linewidth=1.0,
               label=f"mean = {dx_diff.mean():.3f} km")
    ax.set_xlabel("Δdx (ERA5 − pySTEPS) [km]")
    ax.set_title("East component difference")
    ax.legend(fontsize=8)

    ax = axes[2]
    dy_diff = all_dy_era5_flat - all_dy_pys_flat
    clip3 = np.percentile(np.abs(dy_diff), 99)
    bins3 = np.linspace(-clip3, clip3, 80)
    ax.hist(dy_diff, bins=bins3, color="mediumseagreen", edgecolor="none")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.axvline(dy_diff.mean(), color="red", linewidth=1.0,
               label=f"mean = {dy_diff.mean():.3f} km")
    ax.set_xlabel("Δdy (ERA5 − pySTEPS) [km]")
    ax.set_title("North component difference")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"ERA5 vs pySTEPS advection — {n_bins} events, n={n_total:,} bins\n"
        f"|distance diff|: mean={mean_diff:.3f} km, std={std_diff:.3f} km",
        fontsize=10,
    )
    fig.tight_layout()

    if config_plot.save:
        os.makedirs(config_plot.outdir_base, exist_ok=True)
        outpath = outpath_gen("/tmp", config_plot.outdir_base, config_plot.outname)
        fig.savefig(outpath + ".png", dpi=200)
        print(f"Histogram saved to {outpath}.png")
        
    plt.show()
