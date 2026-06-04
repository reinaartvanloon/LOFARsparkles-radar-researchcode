#!/usr/bin/env python3
"""Generate LOFAR VHF visualisations (sparkles vs. other VHF sources)."""

import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
src  = os.path.abspath(os.path.join(here, "..", "src"))
for p in (here, src):
    if p not in sys.path:
        sys.path.insert(0, p)

import plot_LOFAR
from plot_LOFAR import ConfigLOFAR, ConfigPlotLOFAR, SparkleParams
from general import WindowExtent

# ── Configure paths before running ──────────────────────────────────────────
lofar_data_dirpath         = "/path/to/LOFAR"       # LOFAR CSV files (Zenodo)
output_dir                 = "/path/to/output"
dirpath_shapefiles_borders = "/path/to/shapefiles"  # optional, GADM
# ─────────────────────────────────────────────────────────────────────────────

LOFAR_file = "21C8-all.dat"
outname    = "21C8_cellB"

plot_extent = WindowExtent(
    x_range=[6.81, 7.15],
    y_range=[53.1,  53.293],
    z_range=[2500,  14000],
    t_range=[None,  None],
)

sparkle_params = SparkleParams(
    large_cluster   = {"d": 1000, "t": 150, "n": 30},
    sparkle_cluster = {"d": 200,  "t": 5,   "n": 2},
    alt_windows     = [[8000, None]],
)

config_LOFAR = ConfigLOFAR(
    LOFAR_file    = LOFAR_file,
    stormcode     = "21C",
    datapath      = lofar_data_dirpath,
    sparkle_params = sparkle_params,
    max_distance  = 100e3,
    window_extent = plot_extent,
)

config_plot = ConfigPlotLOFAR(
    VHF_type                  = "all",
    plot_extent               = plot_extent,
    outdir                    = output_dir,
    outname                   = outname,
    save                      = True,
    live_plot                 = True,
    markersize                = 0.5,
    dirpath_shapefiles_borders = dirpath_shapefiles_borders,
)

plot_LOFAR.main(config_LOFAR, config_plot)
