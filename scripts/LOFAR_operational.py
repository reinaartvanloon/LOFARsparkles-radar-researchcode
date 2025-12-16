#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:15:56 2024

@author: rvloon
"""

from LOFAR.LOFAR_plot import main as plot_LOFAR
from LOFAR.LOFAR_plot import ConfigLOFAR, ConfigPlotLOFAR, SparkleParams
from general import  WindowExtent, PlottingObject, save_configs_json, load_configs_json
from cartopy.crs import PlateCarree 
outdir = "/media/reinaart/KINGSTON/backup_knmi/figures/21C/LOFAR/article"
datapath = "/media/reinaart/KINGSTON/backup_knmi/data"
# path_configfile = "/media/reinaart/KINGSTON/backup_knmi/plotting_config/LOFAR/21C6_all.json" 
outname = "21C8_cellB"

plot_extent_large = WindowExtent(
    x_range=[6.81, 7.15],
    y_range=[53.1, 53.293],
    z_range=[2500,14000],
    t_range=[None,None],
    )
plot_extent_zoom = WindowExtent(
    x_range=[7, 7.1],
    y_range=[53.14, 53.2],
    z_range=[8000,13700],
    t_range=[None,None],
    )

# plot_extent_zoom2 = WindowExtent(
#     x_range=[7.042, 7.05],
#     y_range=[53.175, 53.181],
#     z_range=[8000,13700],
#     t_range=[None,None],
#     )

plot_extent_zoom2 = WindowExtent(
    x_range=[7.058, 7.07],
    y_range=[53.165, 53.172],
    z_range=[12400,13000],
    t_range=[None,None],
    )

sparkle_params = SparkleParams(
    large_cluster = {"d": 1000, "t": 150, "n": 30},
    sparkle_cluster = {"d": 200, "t": 5, "n": 2},
    alt_windows = [[8000,None]],
    )

config_LOFAR = ConfigLOFAR(
    LOFAR_file="21C8-all.dat",
    stormcode="21C",
    # sparkle_params = sparkle_params,
    datapath = datapath,
    crs='epsg:28992',
    window_extent=WindowExtent(
        ),
    )

config_plot_LOFAR = ConfigPlotLOFAR(
    VHF_type="all",
    plot_extent = plot_extent_large,
    outdir=outdir,
    outname=outname,
    save=True,
    live_plot= True,
    )

#%%
# --- annotation1 ---
annotation1_tz = PlottingObject(
    type='annotation',
    ax_name='axs.tz',
    xy=(120, 10.4),
    text='1',
)

annotation1_xy = PlottingObject(
    type='annotation',
    ax_name='axs.xy',
    xy=(7.06, 53.255),
    text='1',
    kwargs={'transform': PlateCarree()},
)

annotation1_zy = PlottingObject(
    type='annotation',
    ax_name='axs.zy',
    xy=(0.8, 0.88),
    text='1',
    kwargs={'xycoords': 'axes fraction'},
)

# --- annotation2 ---
annotation2_tz = PlottingObject(
    type='annotation',
    ax_name='axs.tz',
    xy=(240, 5.7),
    text='2',
)

annotation2_xy = PlottingObject(
    type='annotation',
    ax_name='axs.xy',
    xy=(6.945, 53.178),
    text='2',
    kwargs={'transform': PlateCarree()},
)

annotation2_xz = PlottingObject(
    type='annotation',
    ax_name='axs.xz',
    xy=(0.4, 0.15),
    text='2',
    kwargs={'xycoords': 'axes fraction'},
)

annotation2_zy = PlottingObject(
    type='annotation',
    ax_name='axs.zy',
    xy=(0.12, 0.4),
    text='2',
    kwargs={'xycoords': 'axes fraction'},
)

# --- annotation3 ---
annotation3_tz = PlottingObject(
    type='annotation',
    ax_name='axs.tz',
    xy=(1100, 6),
    text='3',
)

annotation3_xy = PlottingObject(
    type='annotation',
    ax_name='axs.xy',
    xy=(7.01, 53.125),
    text='3',
    kwargs={'transform': PlateCarree()},
)

annotation3_xz = PlottingObject(
    type='annotation',
    ax_name='axs.xz',
    xy=(0.65, 0.25),
    text='3',
    kwargs={'xycoords': 'axes fraction'},
)

annotation3_zy = PlottingObject(
    type='annotation',
    ax_name='axs.zy',
    xy=(0.3, 0.15),
    text='3',
    kwargs={'xycoords': 'axes fraction'},
)

# --- annotation4 ---
annotation4_tz = PlottingObject(
    type='annotation',
    ax_name='axs.tz',
    xy=(1255, 5),
    text='4',
)

annotation4_xy = PlottingObject(
    type='annotation',
    ax_name='axs.xy',
    xy=(7.04, 53.283),
    text='4',
    kwargs={'transform': PlateCarree()},
)

annotation4_xz = PlottingObject(
    type='annotation',
    ax_name='axs.xz',
    xy=(0.75, 0.15),
    text='4',
    kwargs={'xycoords': 'axes fraction'},
)

annotation4_zy = PlottingObject(
    type='annotation',
    ax_name='axs.zy',
    xy=(0.33, 0.92),
    text='4',
    kwargs={'xycoords': 'axes fraction'},
)

# Update the plotting list
config_plot_LOFAR.list_plotting_objects = [
    annotation1_tz, annotation1_xy, annotation1_zy,
    annotation2_tz, annotation2_xy, annotation2_xz, annotation2_zy,
    annotation3_tz, annotation3_xy, annotation3_xz, annotation3_zy,
    annotation4_tz, annotation4_xy, annotation4_xz, annotation4_zy,
]

#%%
    
LOFARdata = plot_LOFAR(
    config_LOFAR = config_LOFAR, 
    config_plot = config_plot_LOFAR)

#%% 
configs_map = {
    "config_LOFAR":config_LOFAR,
    # "config_plot":config_plot_LOFAR,
    }
save_configs_json(configs_map, outdir+outname+".json")

configs_map2 = {
    "config_LOFAR":ConfigLOFAR,
    "config_plot":ConfigPlotLOFAR,
    }
kwargs = load_configs_json(configs_map2, outdir+outname+".json")