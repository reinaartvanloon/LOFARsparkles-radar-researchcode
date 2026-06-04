#!/usr/bin/env python3
"""Compare radar statistics by altitude band (sparkles vs. other VHF)."""

import os
import sys
import argparse

here = os.path.abspath(os.path.dirname(__file__))
src  = os.path.abspath(os.path.join(here, "..", "src"))
for p in (here, src):
    if p not in sys.path:
        sys.path.insert(0, p)

from altitude_statistics import (
    main, ConfigAltitudeStatsData, ConfigAltitudeStatsPlot, N_BOOTSTRAP,
)
from read_RAD import ConfigMaskRADnearVHF
from plot_LOFAR import SparkleParams

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Compare radar statistics by altitude band (sparkles vs. other VHF)."
)
parser.add_argument(
    "--plot-only", action="store_true",
    help="Skip computation and load results from existing files.",
)
parser.add_argument(
    "--csv", default=None, metavar="PATH",
    help=(
        "CSV file to read from (--plot-only) or write to (default run). "
        "Defaults to <outdir>/altitude_statistics.csv."
    ),
)
parser.add_argument(
    "--hmc-json", default=None, metavar="PATH",
    help=(
        "JSON file for HMC results. "
        "Defaults to <outdir>/altitude_statistics_hmc.json."
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

# ── Configure paths before running ──────────────────────────────────────────
rad_data_dirpath             = "/path/to/borkum_radar"           # HDF5 volume files (Zenodo)
lofar_data_dirpath           = "/path/to/LOFAR"                  # LOFAR CSV files (Zenodo)
advection_reference_filepath = "/path/to/ERA5_20210618.grib"     # ERA5 GRIB (Copernicus CDS)
hmc_msf_filepath             = "lib/msf_cband_v2.nc"             # bundled in repo lib/
output_dir                   = "/path/to/output"
# ─────────────────────────────────────────────────────────────────────────────

config_data = ConfigAltitudeStatsData(
    rad_data_dirpath             = rad_data_dirpath,
    lofar_data_dirpath           = lofar_data_dirpath,
    advection_reference_filepath = advection_reference_filepath,
    hmc_msf_filepath             = hmc_msf_filepath,
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
    config_mask_rad = ConfigMaskRADnearVHF(
        RADnearVHF_radius            = 2000,
        RADalt_threshold             = 8e3,
        RADdbzh_threshold            = 0,
        sparkle_selection_dimension  = "3D",
        otherVHF_selection_dimension = "horizontal",
    ),
)

outdir        = os.path.join(output_dir, "altitude_statistics")
csv_path      = os.path.join(outdir, "altitude_statistics.csv")
hmc_json_path = os.path.join(outdir, "altitude_statistics_hmc.json")


config_plot = ConfigAltitudeStatsPlot(
    outdir        = outdir,
    csv_path      = csv_path,
    hmc_json_path = hmc_json_path,
    plot_only     = True,
    n_bootstrap   = 2000,
    confidence_interval = True,
    save          = True,
)

main(config_data, config_plot)
