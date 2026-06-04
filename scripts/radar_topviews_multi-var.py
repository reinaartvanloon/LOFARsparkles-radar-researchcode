#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:15:56 2024

@author: rvloon
"""

from radar.RAD_multivar_plot import main as plot_RAD
from radar.RAD_multivar_plot import ConfigPlotMultiVarRAD
from radar.ReadRAD import ConfigDataRAD, ConfigMaskRADnearVHF
from LOFAR.LOFAR_plot import ConfigLOFAR, SparkleParams
from general import load_config, WindowExtent


LOFAR_file = "21C8-all.dat"
image_category = f"cellA_{LOFAR_file[:4]}"
outname = f"{image_category}_multi-var"
save_config = f"thesis_sparkles/plotting_config/radar_multivar/{image_category}/{outname}.json"

datapath = "/media/reinaart/KINGSTON/backup_knmi/data"
path_configfile = "/media/reinaart/KINGSTON/backup_knmi/plotting_config/radar_multivar/cellB_21C8/cellB_21C8_multi-var.json"
advection_reference_filepath = "/media/reinaart/KINGSTON/backup_knmi/data/ERA5/21C/ERA5_20210618.grib"
hmc_msf_filepath = "/media/reinaart/KINGSTON/backup_knmi/data/wradlib_data_main/misc/msf_cband_v2.nc"
outdir = f"/media/reinaart/KINGSTON/backup_knmi/figures/21C/radar/multi-var/{image_category}/"

vars_RAD=[
    "dbzh",
    "vradh",
    "wradh",
    "hmc"]

plot_extent = WindowExtent(
    x_range=[6.35,7.2],
    y_range=[52.95,53.55],
    )

config_plot = ConfigPlotMultiVarRAD(
    sweep_angle_list=[
        # 0.5
        # 1.5, 
        # 2.5, 
        # 3.5, 
        # 4.5,
        # 5.5,
        8, 
        12, 
        # 17, 
        # 25,
        ],
    vars_RAD=vars_RAD,
    plot_extent=plot_extent,
    live_plot=True,
    save=True,
    outdir=f"{outdir}",
    outname="topview_dbzh_vrad_wrad_hmc"
    )

config_data_RAD = ConfigDataRAD(
    datapath, 
    stormcode="21C", 
    VHFtype="sparkles&otherVHF", 
    RADstation="asb", 
    RADvars=vars_RAD,
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
#%%

config_, im = plot_RAD(
    config_plot = config_plot,
    config_data_RAD = config_data_RAD,
    config_LOFAR = config_LOFAR,
    config_mask_RADnearVHF = config_mask_RADnearVHF,
    )
