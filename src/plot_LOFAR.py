#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:06:47 2023

@author: rvloon
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import datetime
import cartopy.crs as ccrs
from pyproj import Transformer
import matplotlib.ticker as mticker
import copy
from matplotlib.patches import Polygon
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pandas import to_datetime

from read_LOFAR_data import DataLOFAR, VHFclustering
from general import WindowExtent, outpath_gen, select_points_in_windows, select_points_in_polygons, add_plotting_objects, ProjectCustomExtent, PlottingObject, ConfigSpatialPlot, plot_borders
import gc
#%% Define convenient dataclasses

@dataclass
class ClusteringParams:
    d: float  # Distance threshold in meters
    t: float  # Time threshold in milliseconds
    n: int    # Minimum points for clustering

@dataclass
class SparkleParams:
    large_cluster: dict = None
    sparkle_cluster: dict = None
    alt_windows: List[Tuple[float, float]] = field(default_factory=list)
    time_windows: List[Tuple[float, float]] = field(default_factory=list)
    xy_polygons: List[List[Tuple[float, float]]] = field(default_factory=list)

@dataclass
class ConfigLOFAR:
    LOFAR_file: str = None
    window_extent: WindowExtent
    datapath: Optional[str] = None
    stormcode: Optional[str] = None
    LOFAR_info: Optional[Dict[str, Any]] = None
    LOFARpath: Optional[str] = None
    trange: Optional[Any] = None
    dt_range = None
    sparkle_params: Optional[SparkleParams] = None
    epsg: str = 28992
    window_extent: WindowExtent = WindowExtent()
    crs: str = "LOFAR"
    max_distance: Optional[float] = None
    
# @dataclass_json
@dataclass 
class ConfigPlotLOFAR(ConfigSpatialPlot):
    VHF_type: str = "sparkles&otherVHF",
    crs: str = "epsg:28992"
    interpolation_grid: Optional[str] = None
    contour_binsize: Optional[float] = None
    list_plotting_objects: List[PlottingObject] = field(default_factory=list)
    live_plot: Optional[bool] = None
    title: Optional[str] = ""
    xy_ratio: Optional[str] ='equal'
    markersize: float = 1
    elaborate: Optional[bool] = True
    outname: str = None
    save: bool = False
    outdir: str = ""
    

#%% Plot function

def plotter(
    LOFARdata, 
    config_LOFAR: ConfigLOFAR,
    config_plot: ConfigPlotLOFAR,
    SATimage=None, 
    raddata=None, 
    ):
    
    # Plotting layout
    cmaps = {"LOFAR": gen_olaf_cmap(), "SAT": 'CMRmap_r', "PCP": 'HomeyerRainbow'}
    LOFAR_cmap_lims = (LOFARdata.df.t.min(), LOFARdata.df.t.max())
    markersize = config_plot.markersize
    markersize_sparkles = 15*markersize
    markersize_nosparkles = 10*markersize
    markeredgewidth = markersize/2
    figwidth = 6
    plt.rcParams['font.size'] = 10
    dpi = 400

    if config_LOFAR.sparkle_params is not None:
        VHFmask1, VHFmask2, VHFmask3 = masking_VHF_types(
            LOFARdata.sparklemask, 
            LOFARdata.sparklemask_filtered, 
            config_plot.VHF_type)  # 1 for regular VHF points, 2 for accented VHF points
    else: 
        VHFmask1 = np.ones(len(LOFARdata.df), dtype=bool)
        VHFmask2 = VHFmask3 = np.zeros(len(LOFARdata.df), dtype=bool)

    # window_extent = WindowExtent(x_range=window_extent.x, y_range=window_extent.y) #Update window extent in order for projection
    # map_proj = projection(window_extent.x_central, window_extent.y[0], window_extent.y[1])
    plot_extent = config_plot.plot_extent
    
    axs = ax_builder(
        figwidth, 
        config_plot.projection, 
        plot_extent,
        )

    if config_LOFAR.sparkle_params is not None:
        if config_LOFAR.sparkle_params.xy_polygons:
            for vertices in config_LOFAR.sparkle_params.xy_polygons:
                polygon = Polygon(
                    vertices, 
                    closed=True, 
                    edgecolor='grey', 
                    facecolor='none', 
                    linestyle="--")
                axs.xy.add_patch(polygon)
    
        if config_LOFAR.sparkle_params.alt_windows:
            for alt_window in config_LOFAR.sparkle_params.alt_windows:
                for alt in alt_window:
                    if alt:
                        axs.xz.axhline(y=alt / 1e3, color='grey', linestyle='--')
                        axs.zy.axvline(x=alt / 1e3, color='grey', linestyle='--')
    
        if config_LOFAR.sparkle_params.time_windows:
            for time_window in config_LOFAR.sparkle_params.time_windows:
                for time in time_window:
                    if time:
                        axs.tz.axvline(x=time, color='grey', linestyle='--')

    im = axs.xy.scatter( # XY-topview LOFAR data
        LOFARdata.df.x[VHFmask1], LOFARdata.df.y[VHFmask1], 
        c=LOFARdata.df.t[VHFmask1], 
        cmap=cmaps["LOFAR"], 
        s=markersize, 
        vmin=LOFAR_cmap_lims[0], 
        vmax=LOFAR_cmap_lims[1],
        )  
    axs.xy.scatter( # XY-topview LOFAR data
        LOFARdata.df.x[VHFmask2], LOFARdata.df.y[VHFmask2], 
        c=LOFARdata.df.t[VHFmask2], 
        s=markersize_nosparkles, 
        edgecolor='k', 
        linewidth=markeredgewidth,
        cmap=cmaps["LOFAR"], 
        vmin=LOFAR_cmap_lims[0], 
        vmax=LOFAR_cmap_lims[1],
        )  
    axs.xy.scatter( # XY-topview LOFAR data
        LOFARdata.df.x[VHFmask3], LOFARdata.df.y[VHFmask3], 
        c=LOFARdata.df.t[VHFmask3], 
        s=markersize_sparkles, 
        edgecolor='k', 
        cmap=cmaps["LOFAR"], 
        marker='v', 
        linewidth=markeredgewidth,
        vmin=LOFAR_cmap_lims[0], 
        vmax=LOFAR_cmap_lims[1],
        )  

    plt.colorbar(im, cax=axs.cbar, label='Time [ms]', extend=None)  # Add colorbar for satellite data
    # axs.xy.set_extent(
    #     [
    #         config_plot.plot_extent.x[0], 
    #         config_plot.plot_extent.x[1], 
    #         config_plot.plot_extent.y[0], 
    #         config_plot.plot_extent.y[1]
    #     ], 
    #     crs=config_plot.projection,
    #     )

    #### Plotting the other VHF sources ####
    axs.tz.scatter( # Top time axes
        LOFARdata.df.t[VHFmask1], LOFARdata.df.z[VHFmask1] / 1e3, 
        c=LOFARdata.df.t[VHFmask1], 
        s=markersize, 
        cmap=cmaps["LOFAR"], 
        vmin=LOFAR_cmap_lims[0], 
        vmax=LOFAR_cmap_lims[1],
        )  
    axs.xz.scatter( # xz-sideview
        LOFARdata.df.x[VHFmask1], LOFARdata.df.z[VHFmask1] / 1e3, 
        c=LOFARdata.df.t[VHFmask1], 
        s=markersize, 
        cmap=cmaps["LOFAR"], 
        vmin=LOFAR_cmap_lims[0], 
        vmax=LOFAR_cmap_lims[1],
        )  
    axs.zy.scatter(
        LOFARdata.df.z[VHFmask1] / 1e3, LOFARdata.df.y[VHFmask1], 
        c=LOFARdata.df.t[VHFmask1], 
        s=markersize, cmap=cmaps["LOFAR"], 
        vmin=LOFAR_cmap_lims[0], 
        vmax=LOFAR_cmap_lims[1])  # zy-sideview

    #### Plotting the sparkles, but outside filter criteria ####
    axs.tz.scatter( # Top time axes
        LOFARdata.df.t[VHFmask2],
        LOFARdata.df.z[VHFmask2] / 1e3, 
        c=LOFARdata.df.t[VHFmask2], 
        s=markersize_nosparkles, 
        linewidth=markeredgewidth,
        edgecolor='k', 
        cmap=cmaps["LOFAR"], 
        vmin=LOFAR_cmap_lims[0],
        vmax=LOFAR_cmap_lims[1],
        )  
    axs.xz.scatter( # xz-sideview
        LOFARdata.df.x[VHFmask2], LOFARdata.df.z[VHFmask2] / 1e3, 
        c=LOFARdata.df.t[VHFmask2], 
        s=markersize_nosparkles, 
        linewidth=markeredgewidth, 
        edgecolor='k', 
        cmap=cmaps["LOFAR"], 
        vmin=LOFAR_cmap_lims[0], 
        vmax=LOFAR_cmap_lims[1])  
    axs.zy.scatter( # zy-sideview
        LOFARdata.df.z[VHFmask2] / 1e3, LOFARdata.df.y[VHFmask2], 
        c=LOFARdata.df.t[VHFmask2], 
        s=markersize_nosparkles, 
        linewidth=markeredgewidth,
        edgecolor='k', 
        cmap=cmaps["LOFAR"], 
        vmin=LOFAR_cmap_lims[0], 
        vmax=LOFAR_cmap_lims[1],
        )  

    #### Plotting the sparkles ####
    axs.tz.scatter( # Top time axes
        LOFARdata.df.t[VHFmask3], LOFARdata.df.z[VHFmask3] / 1e3, 
        c=LOFARdata.df.t[VHFmask3],
        s=markersize_sparkles, 
        linewidth=markeredgewidth,
        edgecolor='k', 
        cmap=cmaps["LOFAR"], 
        marker='v', 
        vmin=LOFAR_cmap_lims[0], 
        vmax=LOFAR_cmap_lims[1],
        )  
    axs.xz.scatter( # xz-sideview
        LOFARdata.df.x[VHFmask3], LOFARdata.df.z[VHFmask3] / 1e3, 
        c=LOFARdata.df.t[VHFmask3], 
        s=markersize_sparkles, 
        linewidth=markeredgewidth,
        edgecolor='k', 
        cmap=cmaps["LOFAR"], 
        marker='v', 
        vmin=LOFAR_cmap_lims[0],
        vmax=LOFAR_cmap_lims[1],
        )  
    axs.zy.scatter( # zy-sideview
        LOFARdata.df.z[VHFmask3] / 1e3, LOFARdata.df.y[VHFmask3], 
        c=LOFARdata.df.t[VHFmask3], 
        s=markersize_sparkles, 
        linewidth=markeredgewidth,
        edgecolor='k', 
        cmap=cmaps["LOFAR"], 
        marker='v', 
        vmin=LOFAR_cmap_lims[0],
        vmax=LOFAR_cmap_lims[1],
        )

    # Aesthetics
    axs.zy.set_xlim(plot_extent.z[0] / 1e3, plot_extent.z[1] / 1e3)
    axs.zy.set_ylim(plot_extent.y[0], plot_extent.y[1])

    axs.xz.set_xlim(plot_extent.x[0], plot_extent.x[1])
    axs.xz.set_ylim(plot_extent.z[0] / 1e3, plot_extent.z[1] / 1e3)

    axs.tz.set_xlim(plot_extent.t[0], plot_extent.t[1])
    axs.tz.set_ylim(plot_extent.z[0] / 1e3, plot_extent.z[1] / 1e3)

    plot_borders( # With shapefiles in the dirpath
        config_plot.dirpath_shapefiles_borders,
        axs.xy,
        config_LOFAR.crs
        )
    
    # Plot terp
    if config_plot.terp == True:
        axs.xy.plot(config_plot.terp['longitude'], config_plot.terp['latitude'], marker='x', ms=3, color='k')

    # axs.tz.set_title("{} UTC".format(LOFARdata.dt.time()))
    axs.fig.suptitle(config_plot.title, horizontalalignment='left', x=0.05)

    add_plotting_objects(
        config_plot.list_plotting_objects, 
        axes_mapping={
            "axs.tz":axs.tz,
            "axs.xy":axs.xy,
            "axs.xz":axs.xz,
            "axs.zy":axs.zy,
            },
        )

    if config_plot.save == True:
        outpath = outpath_gen(config_LOFAR.LOFARpath, config_plot.outdir, config_plot.outname)
        axs.fig.savefig(outpath, dpi=dpi)
        print("File is saved to: " + outpath + ".png")


    if config_plot.live_plot:
        plt.show()


class ax_builder:
    def __init__(
            self, 
            figwidth, 
            map_proj, 
            plot_extent,
            ):

        # #Make dummy plot to find axis ratio of xy-axis
        # fig, ax = plt.subplots(visible=False)
        # ax.set_xlim([plot_extent.x[0], plot_extent.x[1]])
        # ax.set_ylim([plot_extent.y[0], plot_extent.y[1]])

        # # Now retrieve the axis limits
        # x_limits = ax.get_xlim()
        # y_limits = ax.get_ylim()

        # Calculate the ratio of the axes
        yx_ratio = (plot_extent.y[1] - plot_extent.y[0]) / (plot_extent.x[1] - plot_extent.x[0])

        # Calculate axes sizes and spacing
        D_x = 1 / 10  # Spacing unit, relative to width
        D = figwidth * D_x  # Absolute spacing size
        W_x = 1 - 2.5 * D_x  # Width of the temporal plot, relative to x
        W = W_x * figwidth  # Absolute Width of temporal plot
        H = 0.3 * W  # Absolute height of temporal plot (and xz-plot and zy-plot)

        lon = W - H  # Absolute size of longitude axis
        lat = yx_ratio * lon  # Absolute size of latitude axis
        X = figwidth  # Absolute width of figure
        Y = 4 * D + lat + 2 * H  # Absolute height of figure

        lon_x = lon / X  # Size of longitude axis wrt figure width
        lat_y = lat / Y  # Size of latitude axis wrt figure height
        H_x = H / X  # Size of altitude axis wrt figure width
        H_y = H / Y  # Size of altitude axis wrt figure height
        D_y = D / Y  # Size of spacing wrt figure height

        self.fig = plt.figure(figsize=(X, Y))

        # xy-topview axes
        rect_xy = [1.5 * D_x, D_y, lon_x, lat_y]
        self.xy = self.fig.add_axes(rect_xy, projection=map_proj)
        self.xy.set_xlim(plot_extent.x)
        self.xy.set_ylim(plot_extent.y)
        gl = self.xy.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='dotted')
        gl.xlocator = mticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
        gl.ylocator = mticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
        gl.top_labels = False
        gl.right_labels = False

        # xz-sideview
        rect_xz = [1.5 * D_x, D_y + lat_y, lon_x, H_y]
        self.xz = self.fig.add_axes(rect_xz)
        self.xz.set_xlim(plot_extent.x)
        abs_distance_xticks(self.xz, plot_extent.x)
        self.xz.xaxis.set_ticks_position('top')  # Set ticks to bottom
        self.xz.xaxis.set_label_position('top')  # Set labels to bottom
        self.xz.set_xlabel('Distance [km]')
        self.xz.set_ylabel("Altitude [km]")
        self.xz.grid(linestyle='dotted')
        
        # Hide the first y-tick label
        ytl = self.xz.get_yticklabels()
        if ytl:
            ytl[0].set_visible(False)
            

        # zy-sideview
        rect_zy = [1.5 * D_x + lon_x, D_y, H_x, lat_y]
        self.zy = self.fig.add_axes(rect_zy)
        self.zy.set_ylim(plot_extent.y)
        abs_distance_yticks(self.zy, plot_extent.y)
        self.zy.yaxis.set_ticks_position('right')
        self.zy.yaxis.set_label_position('right')
        self.zy.set_xlabel('Altitude [km]')
        self.zy.set_ylabel("Distance [km]")
        self.zy.grid(linestyle='dotted')
        
        # Hide the first x-tick label
        xtl = self.zy.get_xticklabels()
        if xtl:
            xtl[0].set_visible(False)

        # tz-temporal plot
        rect_tz = [1.5 * D_x, 2.5 * D_y + lat_y + H_y, W_x, H_y]
        self.tz = self.fig.add_axes(rect_tz)
        self.tz.set_xlabel('Time [ms]')
        self.tz.set_ylabel('Altitude [km]')
        self.tz.grid(linestyle='dotted')

        # Colorbar
        rect_cbar = [2 * D_x + lon_x, 1.5 * D_y + lat_y, 0.5 * D_x, H_y - 0.5 * D_y]
        self.cbar = self.fig.add_axes(rect_cbar)
        
        for label, ax in zip(
            ["a", "b", "c", "d"],
            [self.tz, self.xz, self.xy, self.zy]
            ):
            
            ax.annotate(
                f"({label})",
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(1, -1),
                textcoords="offset points",
                ha="left",
                va="top",
            )


def remove_first_ticklabel(ax,dim):
    if dim == "x":
        xtl = ax.get_xticklabels()
        if xtl:
            xtl[0].set_visible(False) 
            
    if dim =="y":
        ytl = ax.get_yticklabels()
        if ytl:
            ytl[0].set_visible(False) 
    
def abs_distance_xticks(ax, x_limits):
    locator = mticker.MaxNLocator(nbins=6, steps=[1, 2, 4, 5, 10])
    x_ticks = locator.tick_values(0, x_limits[1] - x_limits[0])
    ax.set_xticks(np.array(x_ticks) + x_limits[0])
    ax.set_xticklabels([f"{tick/1000}" for tick in x_ticks])


def abs_distance_yticks(ax, y_limits):
    locator = mticker.MaxNLocator(nbins=6, steps=[1, 2, 4, 5, 10])
    y_ticks = locator.tick_values(0, y_limits[1] - y_limits[0])
    ax.set_yticks(np.array(y_ticks) + y_limits[0])
    ax.set_yticklabels([f"{tick/1000}" for tick in y_ticks])


def format_xdistance(value, tick_number, x_limits):
    # Convert value relative to the lower bound of each axis
    return f"{(value - x_limits[0])/1000:.1f}"  # Convert meters to km


def format_ydistance(value, tick_number, y_limits):
    # Convert value relative to the lower bound of each axis
    return f"{(value - y_limits[0])/1000:.1f}"  # Convert meters to km


class fig_xy_ratios:  # Convert the horizontal figure ratio to a vertical one such that the absolute distance is the equal.
    def __init__(self, y, figsize):
        self.y = y
        self.x = y * figsize[1] / figsize[0]


def gen_olaf_cmap(num=256):
    def RGB(V):
        cs = 0.9 + V * 0.9
        R = 0.5 + 0.5 * np.cos(2 * 3.14 * cs)
        G = 0.5 + 0.5 * np.cos(2 * 3.14 * (cs + 0.25))
        B = 0.5 + 0.5 * np.cos(2 * 3.14 * (cs + 0.5))
        return R, G, B

    RGBlist = [RGB(x) for x in np.linspace(0.05, 0.95, num)]
    return colors.LinearSegmentedColormap.from_list('OlafOfManyColors', RGBlist, N=num)


def projection(lon_central, lat_min, lat_max):
    map_proj = ccrs.Mercator(central_longitude=lon_central,
                             min_latitude=lat_min,
                             max_latitude=lat_max,
                             latitude_true_scale=lat_max)
    return map_proj


def check_ConfigLOFAR(config: ConfigLOFAR):
    if config.LOFAR_file == None and config.dt_range == None:
        raise(ValueError("Cannot open LOFAR file if neither LOFAR_file or dt_range are provided."))
    

    
#%% new functions 

def load_data_LOFAR(config: ConfigLOFAR):
    """Load LOFAR data and update configuration."""
    config.stormcode = config.LOFAR_file[:3]
    config.timecode = config.LOFAR_file[3]

    # Load LOFAR image metadata
    lofar_info_path = os.path.join(
        config.datapath, 
        'LOFAR_image-info.json',
        )
    with open(lofar_info_path) as file:
        LOFARinfo = json.load(file)
        config.LOFAR_info = LOFARinfo[config.LOFAR_file.split(".")[0]]

    # Set LOFAR path
    config.LOFARpath = os.path.join(
        config.datapath, 
        config.LOFAR_file,
        )
    LOFARdata = DataLOFAR(
        config.LOFARpath, 
        copy.deepcopy(config.window_extent), 
        crs=config.crs,
        max_distance=config.max_distance,
        )  # LOAD LOFAR VHF data
    LOFARdata.dt = datetime.datetime.strptime(
        config.LOFAR_info['time'], 
        '%Y-%m-%dT%H:%M:%S.')  # LOFAR datetime
    LOFARdata.ext = config.LOFAR_file[4:-4]  # Info of storm and LOFAR file
    
    return config, LOFARdata



def update_with_LOFAR_info(config: ConfigLOFAR) -> ConfigLOFAR:
    """Update config with LOFAR info attributes."""
    for key, value in config.LOFAR_info.items():
        if getattr(config, key, None) is None:  # Only update if not already set
            setattr(config, key, value)
    return config

def masking_VHF_types(
    sparklemask: np.ndarray, 
    subset_sparklemask: np.ndarray, 
    VHF_types: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate masks for VHF data."""
    shape = sparklemask.shape
    mask1 = np.zeros(shape, dtype=bool)
    mask2 = np.zeros(shape, dtype=bool)
    mask3 = np.zeros(shape, dtype=bool)

    if VHF_types == 'sparkles':
        mask2 = sparklemask & ~subset_sparklemask
        mask3 = subset_sparklemask
    elif VHF_types == "otherVHF":
        mask1 = ~sparklemask
    elif VHF_types == "sparkles&otherVHF":
        mask1 = ~sparklemask
        mask2 = sparklemask & ~subset_sparklemask
        mask3 = subset_sparklemask
    elif VHF_types == "all":
        mask1 = np.ones(shape, dtype=bool)
    else:
        raise ValueError(f"Unsupported VHF type: {VHF_types}")

    return mask1, mask2, mask3

def get_data_LOFAR(
    config: ConfigLOFAR
    ):
    """Retrieve matching LOFAR data based on configuration and time range."""
    if config.LOFAR_file:  # Explicitly provided LOFAR file
        return load_data_LOFAR(config)

    # Load LOFAR image metadata
    lofar_info_path = os.path.join(config.datapath,'LOFAR_image-info.json')
    with open(lofar_info_path) as file:
        LOFARinfo = json.load(file)

    # Iterate over available LOFAR files
    for LOFAR_file, info in LOFARinfo.items():
        dt_LOFAR = to_datetime(info['time'], format='%Y-%m-%dT%H:%M:%S.')
        if config.dt_start <= dt_LOFAR <= config.dt_end:  # Match found within range
            config.LOFAR_file = LOFAR_file
            config.LOFAR_info = info
            return load_data_LOFAR(config)

    return config, None


def cluster_LOFARsparkles(
        config: ConfigLOFAR, 
        data_LOFAR: DataLOFAR, 
        sparkle_params: SparkleParams
        ) -> Tuple[np.ndarray, np.ndarray]:

    # Large-scale clustering
    print(f"Large-scale clustering with parameters: d={sparkle_params.large_cluster['d']} m, "
          f"t={sparkle_params.large_cluster['t']} ms, n={sparkle_params.large_cluster['n']}")
    LOFAR_df = VHFclustering(
        data_LOFAR.df,
        d=sparkle_params.large_cluster['d'],
        t=sparkle_params.large_cluster['t'],
        n=sparkle_params.large_cluster['n'],
    )

    # Filter out large lightning structures and mark the remainder with Trues
    mask_not_large_clusters = LOFAR_df['structure_label'].values == -1

    if not sparkle_params.sparkle_cluster:
        print("No clustering on sparkle-scale (sparkle cluster params not provided).")
        return mask_not_large_clusters, mask_not_large_clusters

    # Sparkle-scale clustering
    sparkle_params = sparkle_params.sparkle_cluster
    print(f"Sparkle-scale clustering with parameters: d={sparkle_params['d']} m, "
          f"t={sparkle_params['t']} ms, n={sparkle_params['n']}")
    df_subset = LOFAR_df[mask_not_large_clusters]
    df_subset = VHFclustering(
        df_subset,
        d=sparkle_params['d'],
        t=sparkle_params['t'],
        n=sparkle_params['n'],
    )
    # Keep all the groups and mark with Trues
    sparklemask_subset = df_subset['structure_label'].values != -1

    # Define sparklemask
    sparklemask = np.zeros(len(LOFAR_df), dtype=bool)
    sparklemask[mask_not_large_clusters] = sparklemask_subset

    return mask_not_large_clusters, sparklemask

def filter_mask_LOFAR( 
    mask,
    data_LOFAR: DataLOFAR,
    sparkle_params: SparkleParams,
    crs_data
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sparkle mask considering various filters."""

    transformer_lonlat_to_crs = Transformer.from_crs(
        'epsg:4326',
        crs_data,
        always_xy=True,
        )
    # Transform sparkle polygons to LOFAR-native CRS
    if sparkle_params.xy_polygons:
        sparkle_params.xy_polygons = [
            [transformer_lonlat_to_crs.transform(lon, lat) for lon, lat in polygon]
            for polygon in sparkle_params.xy_polygons
        ]

    # Apply altitude and time filters
    sparklemask_in_alt_windows = select_points_in_windows(
        data_LOFAR.df[mask].z, 
        sparkle_params.alt_windows, 
        var="altitude"
        )
    sparklemask_in_time_windows = select_points_in_windows(
        data_LOFAR.df[mask].t, 
        sparkle_params.time_windows, 
        var="time"
        )
    
    # Check sparkles in polygons
    sparklemask_inpolygons = select_points_in_polygons(
        data_LOFAR.df['x'][mask],
        data_LOFAR.df['y'][mask],
        xy_polygon_lst=sparkle_params.xy_polygons,
        )

    # Combine filters
    mask_filtered = np.zeros(np.shape(mask), dtype=bool)
    mask_filtered[mask] = (
        sparklemask_in_alt_windows & 
        sparklemask_in_time_windows & 
        sparklemask_inpolygons 
        )

    return mask_filtered

def get_datetime_LOFAR_from_metadata(
    config: ConfigLOFAR
    ):

    path_LOFAR_info = os.path.join(config.datapath, 'LOFAR_image-info.json')
    with open(path_LOFAR_info) as file:
        LOFARinfo = json.load(file)
        LOFAR_info = LOFARinfo[config.LOFAR_file.split(".")[0]]
        dt_target = LOFAR_info['time'][:-1]  # LOFAR datetime
        dt_target = datetime.datetime.strptime(dt_target, '%Y-%m-%dT%H:%M:%S')
        dt_range = [
            dt_target - datetime.timedelta(seconds=150),
            dt_target + datetime.timedelta(seconds=150)
            ]
    return dt_target, dt_range


# Define the dataclass structure
@dataclass
class ConfigMain:
    datapath: str
    LOFAR_file: str
    configfile_path: Optional[str] = None
    save: bool = False
    outdir: Optional[str] = None
    outname: Optional[str] = None

    # VHF options
    VHFtype: Optional[str] = None
    VHF_largecluster_params: Optional[Dict[str, Any]] = field(default=None, metadata={"parser": json.loads})
    VHF_sparklecluster_params: Optional[Dict[str, Any]] = field(default=None, metadata={"parser": json.loads})
    sparkle_xy_polygons: Optional[List[Any]] = field(default=None, metadata={"parser": "parse_polygons"})
    sparkle_time_windows: Optional[List[float]] = field(default=None, metadata={"parser": "parse_floa_list"})
    sparkle_alt_windows: Optional[List[float]] = field(default=None, metadata={"parser": "parse_loat_list"})

    # Plotting window
    xrange: Optional[List[float]] = None
    yrange: Optional[List[float]] = None
    zrange: Optional[List[float]] = None
    trange: Optional[List[float]] = None

    # Plotting options
    epsg: int = 4326
    xy_ratio: str = "equal"
    plot_margins: List[float] = field(default_factory=lambda: [0, 0])
    interpolation_grid: Optional[str] = None
    contour_binsize: Optional[float] = None
    live_plot: Optional[bool] = None

    # Logger configuration
    console: bool = True
    logfile: bool = True
    verbose: bool = False

    # Satellite options
    SATvar: Optional[str] = None
    SAT_timeoffset: int = 700
    SAT_which_im: str = "nearest"
    SATrange: Optional[List[float]] = None

    # Radar options
    RADvar: Optional[str] = None
    RAD_which_im: str = "nearest"
    RADrange: Optional[List[float]] = None
    RADstation: Optional[str] = None
    SATadvection: Optional[str] = None
    v_data_dir: Optional[str] = None

# The main function
def main(
        config_LOFAR,
        config_plot,
        ):
    # try:
    if not config_plot.outname:
        config_plot.outname = config_LOFAR.LOFAR_file[:-4]

    # # Define logger
    # logger = customLOG(
    #     f"LOFAR-plotter_{config.outname}",
    #     print_console=config.console,
    #     logfile=config.logfile,
    #     outdir=config.outdir,
    #     outname=config.outname,
    #     verbose=config.verbose
    # )
    
    # Load and process data
    config_LOFAR, LOFARdata = get_data_LOFAR(config_LOFAR)

    if config_LOFAR.sparkle_params is not None:
        # Want to distinguish between sparkles and other VHF sources
        mask_not_large_clusters, mask_sparkles = cluster_LOFARsparkles(
            config = config_LOFAR,
            data_LOFAR = LOFARdata,
            sparkle_params = config_LOFAR.sparkle_params,
            )
        
        # Want to impose some more restrictions for something to be a sparkle
        mask_sparkles_filtered = filter_mask_LOFAR(
            mask_sparkles, 
            data_LOFAR = LOFARdata, 
            sparkle_params = config_LOFAR.sparkle_params, 
            crs_data = config_LOFAR.crs,
            )
        
        LOFARdata, [LOFARdata.sparklemask, LOFARdata.sparklemask_filtered] = LOFARdata.select_2_windowextent(
            window_extent = config_plot.plot_extent, 
            crs_target = config_LOFAR.crs, 
            mask_list = [mask_sparkles, mask_sparkles_filtered],
            )  
    
    else:
        LOFARdata, [] = LOFARdata.select_2_windowextent(
            window_extent = config_plot.plot_extent, 
            crs_target = config_LOFAR.crs, 
            mask_list = [],
            ) 
    
    #Plotting extent needs to be in same crs as plot crs
    config_plot.plot_extent = config_plot.plot_extent.transform_crs(
        crs_target=config_plot.crs,
        )
    
    # Already build the projection from the crs code, with custom horizontal extent
    config_plot.projection = ProjectCustomExtent(
        epsg=config_plot.crs[5:], 
        window_extent=config_plot.plot_extent,
        )

    if getattr(config_plot, "title") is not None:
        config_plot.title = f"""Storm {config_LOFAR.stormcode} ({LOFARdata.dt.date()}) \nLOFAR: {config_LOFAR.LOFAR_file} ({LOFARdata.dt.time()} UTC)"""

    plotter(
        LOFARdata = LOFARdata, 
        config_LOFAR = config_LOFAR, 
        config_plot = config_plot)
    
    # # Additional processing and plotting logic here
    # logger.info("Plotting...")
    # plotter(
    #     LOFARdata,
    #     config.LOFARpath,
    #     config,
    #     SATimage=None,
    #     raddata=None,
    #     title=title,
    #     xy_ratio=config.xy_ratio,
    #     window_extent=window_extent
    # )
    
    del LOFARdata
    gc.collect()
    # return LOFARdata

    # except Exception as e:
    #     tb_str = traceback.format_exc()
    #     print(f"An exception occurred: {tb_str}")

# %%Run the main function with argparse functionality when called from the command line

# if __name__ == "__main__":
#     RunFromCL(kwargs_config)(main)()
