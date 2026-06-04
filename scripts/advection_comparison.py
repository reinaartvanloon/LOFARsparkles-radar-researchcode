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

config_data = ConfigAdvectionData(
    rad_data_dirpath   = "/home/reinaart/sparkles/data_zenodo/borkum_radar",
    lofar_data_dirpath = "/home/reinaart/sparkles/data_zenodo/LOFAR",
    era5_filepath      = "/home/reinaart/sparkles/data_additional/ERA5_20210618.grib",
    pysteps_motion_filepath   = "/home/reinaart/sparkles/data_additional/motion_20210618T1700_20210618T2100.nc",
    hmc_msf_filepath   = "/home/reinaart/sparkles/data_additional/msf_cband_v2.nc",
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
    outdir_base = "/home/reinaart/sparkles/temp_figures/advection_comparison",
    outname     = "advection_distance_diff",
    save        = True,
)

main(config_data, config_plot)
