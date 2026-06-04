#!/usr/bin/env python3
"""Run ERA5 vs. pySTEPS advection comparison across LOFAR events."""

import os, sys
import matplotlib
matplotlib.use("Agg")

here = os.path.abspath(os.path.dirname(__file__))
src  = os.path.abspath(os.path.join(here, "..", "src"))
for p in (here, src):
    if p not in sys.path:
        sys.path.insert(0, p)

from advection_comparison import main, ConfigAdvectionData, ConfigAdvectionPlot
from plot_LOFAR import SparkleParams
from read_RAD import ConfigMaskRADnearVHF

# ── Configure paths before running ──────────────────────────────────────────
rad_data_dirpath        = "/path/to/borkum_radar"                           # HDF5 volume files (Zenodo)
lofar_data_dirpath      = "/path/to/LOFAR"                                  # LOFAR CSV files (Zenodo)
era5_filepath           = "/path/to/ERA5_20210618.grib"                     # ERA5 GRIB (Copernicus CDS)
pysteps_motion_filepath = "/path/to/motion_20210618T1700_20210618T2100.nc"  # pySTEPS motion NetCDF
hmc_msf_filepath        = "lib/msf_cband_v2.nc"                             # bundled in repo lib/
output_dir              = "/path/to/output"
# ─────────────────────────────────────────────────────────────────────────────

config_data = ConfigAdvectionData(
    rad_data_dirpath        = rad_data_dirpath,
    lofar_data_dirpath      = lofar_data_dirpath,
    era5_filepath           = era5_filepath,
    pysteps_motion_filepath = pysteps_motion_filepath,
    hmc_msf_filepath        = hmc_msf_filepath,
    lofar_file_list    = [
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
        sparkle_selection_dimension  = "horizontal",
        otherVHF_selection_dimension = "horizontal",
    ),
    stormcode    = "21C",
    rad_station  = "asb",
    max_distance = 100e3,
)

config_plot = ConfigAdvectionPlot(
    outdir_base = os.path.join(output_dir, "advection_comparison"),
    outname     = "advection_distance_diff",
    save        = True,
)

main(config_data, config_plot)
