#!/usr/bin/env python3
"""Run DBSCAN OAT sensitivity study for sparkle identification parameters."""

import os
import sys
import argparse

here = os.path.abspath(os.path.dirname(__file__))
src  = os.path.abspath(os.path.join(here, "..", "src"))
for p in (here, src):
    if p not in sys.path:
        sys.path.insert(0, p)

from sensitivity_DBSCAN import (
    main, ConfigSensitivityDBSCANData, ConfigSensitivityDBSCANPlot, N_BOOTSTRAP,
)
from read_RAD import ConfigMaskRADnearVHF

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="OAT sensitivity study for DBSCAN clustering parameters."
)
parser.add_argument(
    "--plot-only", action="store_true",
    help="Skip computation and load results from an existing CSV file.",
)
parser.add_argument(
    "--csv", default=None, metavar="PATH",
    help=(
        "CSV file to read from (--plot-only) or write to (default run). "
        "Defaults to <outdir>/sensitivity_DBSCAN_results.csv."
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

config_data = ConfigSensitivityDBSCANData(
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
    varlist         = ["dbzh", "wradh"],
    config_mask_rad = ConfigMaskRADnearVHF(
        RADnearVHF_radius = 2000,
        RADalt_threshold  = 8e3,
        RADdbzh_threshold = 0,
    ),
    stormcode    = "21C",
    rad_station  = "asb",
    max_distance = 100e3,
)

config_plot = ConfigSensitivityDBSCANPlot(
    outdir              = "/home/reinaart/sparkles/temp_figures/sensitivity_DBSCAN",
    csv_path            = "/home/reinaart/sparkles/temp_figures/sensitivity_DBSCAN/sensitivity_DBSCAN_results.csv",
    plot_only           = True,
    n_bootstrap         = args.n_bootstrap if args.n_bootstrap is not None else N_BOOTSTRAP,
    confidence_interval = True,
    save                = True,
)

main(config_data, config_plot)
