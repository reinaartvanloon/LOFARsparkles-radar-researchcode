#!/usr/bin/env python3
"""Generate vertical radar cross-sections with LOFAR VHF source overlay."""

import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
src  = os.path.abspath(os.path.join(here, "..", "src"))
for p in (here, src):
    if p not in sys.path:
        sys.path.insert(0, p)

import plot_RAD_crosssect
from plot_RAD_crosssect import ConfigPlotRADcrossSection
from read_RAD import ConfigDataRAD, ConfigMaskRADnearVHF
from plot_LOFAR import ConfigLOFAR, SparkleParams
from general import WindowExtent

# ── Configure paths before running ──────────────────────────────────────────
rad_data_dirpath             = "/path/to/borkum_radar"        # HDF5 volume files (Zenodo)
lofar_data_dirpath           = "/path/to/LOFAR"               # LOFAR CSV files (Zenodo)
advection_reference_filepath = "/path/to/ERA5_20210618.grib"  # ERA5 GRIB (Copernicus CDS)
hmc_msf_filepath             = "lib/msf_cband_v2.nc"          # bundled in repo lib/
output_dir                   = "/path/to/output"
dirpath_shapefiles_borders   = "/path/to/shapefiles"          # optional, GADM
# ─────────────────────────────────────────────────────────────────────────────

LOFAR_file = "21C6er-all.dat"
outname    = f"cellA_{LOFAR_file[:4]}_crosssect"

varlist_RAD = ["dbzh", "vradh", "wradh", "hmc"]

plot_extent = WindowExtent(
    x_range=[6.8,  7.2],
    y_range=[53.22, 53.6],
    z_range=[0,    14.5e3],
)

sparkle_params = SparkleParams(
    large_cluster   = {"d": 1000, "t": 150, "n": 30},
    sparkle_cluster = {"d": 200,  "t": 5,   "n": 2},
    alt_windows     = [[8000, None]],
)

config_LOFAR = ConfigLOFAR(
    LOFAR_file     = LOFAR_file,
    stormcode      = "21C",
    datapath       = lofar_data_dirpath,
    sparkle_params = sparkle_params,
    max_distance   = 100e3,
    window_extent  = plot_extent,
)

config_data_RAD = ConfigDataRAD(
    data_dirpath                 = rad_data_dirpath,
    stormcode                    = "21C",
    RADstation                   = "asb",
    RADvars                      = varlist_RAD,
    advection_reference_filepath = advection_reference_filepath,
    temp_reference_filepath      = advection_reference_filepath,
    hmc_msf_filepath             = hmc_msf_filepath,
)

config_plot = ConfigPlotRADcrossSection(
    sweep_angle_list              = [5.5],
    pointA                        = [6.85, 53.4],
    pointB                        = [7.1,  53.272],
    varlist_RAD                   = varlist_RAD,
    VHFprojection_dist_to_Vplane  = 50e3,
    plot_extent                   = plot_extent,
    VHF_type                      = "sparkles",
    markersize                    = 3,
    live_plot                     = True,
    save                          = True,
    outdir                        = output_dir,
    outname                       = outname,
    dirpath_shapefiles_borders    = dirpath_shapefiles_borders,
)

plot_RAD_crosssect.main(config_plot, config_data_RAD, config_LOFAR)
