#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:29:04 2024

@author: rvloon
"""

from radar.RAD_crosssection_plot import main as plot_crossSection
from radar.RAD_crosssection_plot import ConfigPlotRADcrossSection
from radar.ReadRAD import ConfigDataRAD, ConfigMaskRADnearVHF
from LOFAR.LOFAR_plot import ConfigLOFAR, SparkleParams
from general import WindowExtent, PlottingObject
import matplotlib.patheffects as pe

LOFAR_file = "21C6er-all.dat"
image_category = f"cellA_{LOFAR_file[:4]}"
outname = f"{image_category}_multi-var"
save_config = f"thesis_sparkles/plotting_config/radar_multivar/{image_category}/{outname}.json"

datapath = "/media/reinaart/KINGSTON/backup_knmi/data"
# path_configfile = "/media/reinaart/KINGSTON/backup_knmi/plotting_config/radar_multivar/cellB_21C8/cellB_21C8_multi-var.json"
advection_reference_filepath = "/media/reinaart/KINGSTON/backup_knmi/data/ERA5/21C/ERA5_20210618.grib"
hmc_msf_filepath = "/media/reinaart/KINGSTON/backup_knmi/data/wradlib_data_main/misc/msf_cband_v2.nc"
outdir = f"/media/reinaart/KINGSTON/backup_knmi/figures/21C/radar/crosssection/{image_category}/"

varlist_RAD=[
    "dbzh",
    # "vradh",
    # "wradh",
    # "hmc",
    ]

plot_extent = WindowExtent(
    x_range=[6.8,7.2],
    y_range=[53.22,53.6],
    z_range=[0,14.5e3]
    )

config_plot = ConfigPlotRADcrossSection(
    sweep_angle_list=[
        # 0.5,
        # 1.5, 
        # 2.5, 
        # 3.5, 
        # 4.5,
        5.5,
        # 8, 
        # 12, 
        # 17, 
        # 25,
        ],
    pointA = [6.85,53.4],
    pointB = [7.1,53.272],
    VHFprojection_dist_to_Vplane=50e3,
    varlist_RAD=varlist_RAD,
    plot_extent=plot_extent,
    live_plot=True,
    save=True,
    outdir=f"{outdir}",
    outname=f"{image_category}",
    VHF_type="sparkles",
    markersize=3
    )

config_data_RAD = ConfigDataRAD(
    datapath, 
    stormcode="21C", 
    RADstation="asb", 
    RADvars=varlist_RAD,
    advection_reference_filepath=advection_reference_filepath, 
    temp_reference_filepath=advection_reference_filepath,
    hmc_msf_filepath=hmc_msf_filepath,
    epsg=28992,
    )

sparkle_params = SparkleParams(
    large_cluster = {"d": 1000, "t": 150, "n": 30},
    sparkle_cluster = {"d": 200, "t": 5, "n": 2},
    alt_windows = [[8000,None]],
    )

config_LOFAR = ConfigLOFAR(
    stormcode="21C", 
    LOFAR_file=LOFAR_file,
    datapath = datapath,
    sparkle_params = sparkle_params,
    max_distance = 100e3, # radial distance from LOFAR superterm [meter]
    )

config_mask_RADnearVHF = ConfigMaskRADnearVHF(
    RADnearVHF_radius = 2000, # Max distance between radar and VHF data to be matched
    RADalt_threshold=8e3, # Minimum altitude [meter] for radar data to be considered
    RADdbzh_threshold=0, # Minimum dbzh value for radar data to be considered
    )

## Add the following objects to the plot
arrow = PlottingObject(
    obj_type="arrow",
    ax_name = "ax2",
    x = 18e3,
    y = 1e3,
    dx = -2e3,
    dy = 1.2e3,
    kwargs = {
        "facecolor":"white",
        "width":200,
        "edgecolor":"k",
        "linewidth":1,
        }
    )
annotation = PlottingObject(
    obj_type ="annotation",
    ax_name="ax2",
    text="BWER",
    xy = [18.2e3,0.8e3],
    kwargs = {
        "color":"white",
        "weight":"semibold",
        },
    path_effect = [
        pe.Stroke(linewidth=1, foreground="black"),
        pe.Normal()
        ]
    )
config_plot.list_plotting_objects = [arrow,annotation]

#%% Run

plot_crossSection(
    config_plot = config_plot, 
    config_data_RAD = config_data_RAD,
    config_LOFAR = config_LOFAR,
    )