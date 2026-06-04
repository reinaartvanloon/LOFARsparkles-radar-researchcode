#!/usr/bin/env python3
"""Run radius sensitivity study for sparkle identification parameters."""

import os
import sys
import argparse

here = os.path.abspath(os.path.dirname(__file__))
src  = os.path.abspath(os.path.join(here, "..", "src"))
for p in (here, src):
    if p not in sys.path:
        sys.path.insert(0, p)

from sensitivity_r_near_RAD import (
    main, ConfigSensitivityRadiusData, ConfigSensitivityRadiusPlot, N_BOOTSTRAP,
)
from plot_LOFAR import SparkleParams

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Sensitivity study for RADnearVHF_radius."
)
parser.add_argument(
    "--plot-only", action="store_true",
    help="Skip computation and load results from an existing CSV file.",
)
parser.add_argument(
    "--csv", default=None, metavar="PATH",
    help=(
        "CSV file to read from (--plot-only) or write to (default run). "
        "Defaults to <outdir>/sensitivity_r-near-RAD_results.csv."
    ),
)
parser.add_argument(
    "--hmc-json", default=None, metavar="PATH",
    help=(
        "JSON file to read HMC results from (--plot-only) or write to (default run). "
        "Defaults to <outdir>/sensitivity_r-near-RAD_hmc.json."
    ),
)
parser.add_argument(
    "--n-bootstrap", type=int, default=None, metavar="N",
    help=f"Number of bootstrap iterations (default: {N_BOOTSTRAP}).",
)
parser.add_argument(
    "--confidence-interval", action="store_true", default=False,
    help="Compute and plot bootstrap confidence intervals (slow; off by default).",
)
args = parser.parse_args()

# ── Config ─────────────────────────────────────────────────────────────────────

config_data = ConfigSensitivityRadiusData(
    rad_data_dirpath             = "/home/reinaart/sparkles/data_zenodo/borkum_radar",
    lofar_data_dirpath           = "/home/reinaart/sparkles/data_zenodo/LOFAR",
    advection_reference_filepath = "/home/reinaart/sparkles/data_additional/ERA5_20210618.grib",
    hmc_msf_filepath             = "/home/reinaart/sparkles/data_additional/msf_cband_v2.nc",
    lofar_file_list = [
        "21C1eCt-all.dat",
        "21C2en-all.dat",
        "21C3e-all.dat",
        "21C4e-all.dat",
        "21C5e-all.dat",
        "21C6er-all.dat",
        "21C7-all.dat",
        "21C8-all.dat",
        "21C9-all.dat",
    ],
    varlist        = ["dbzh", "wradh"],
    sparkle_params = SparkleParams(
        large_cluster   = {"d": 1000, "t": 150, "n": 30},
        sparkle_cluster = {"d": 200,  "t": 5,   "n": 2},
        alt_windows     = [[8000, None]],
    ),
)

outdir = "/home/reinaart/sparkles/temp_figures/sensitivity_r-near-RAD"


config_plot = ConfigSensitivityRadiusPlot(
    outdir        = outdir,
    csv_path      = args.csv,
    hmc_json_path = args.hmc_json,
    plot_only     = True,
    n_bootstrap   = 2000,
    confidence_interval = args.confidence_interval,
    save          = True,
)

main(config_data, config_plot)
