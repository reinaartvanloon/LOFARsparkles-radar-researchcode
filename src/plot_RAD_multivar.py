#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:28:14 2024

@author: rvloon
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec
from typing import Optional
import matplotlib.ticker as mticker
import warnings
from dataclasses import dataclass, field
from typing import List
import string

# Custom funtions
from read_RAD import get_data_RADandLOFAR, add_mask_RADnearVHF, ConfigDataRAD, ConfigMaskRADnearVHF, DataRADandLOFAR
from general import  outpath_gen, CRSfromPROJ4, getCmap, ConfigSpatialPlot, add_plotting_objects, PlottingObject, plot_borders
from plot_LOFAR import ConfigLOFAR

# Suppress UserWarnings specifically
warnings.filterwarnings("ignore", category=UserWarning, message="The input coordinates to pcolormesh are interpreted as cell centers, but are not monotonically increasing or decreasing.")


# %% plotting function

    
@dataclass
class ConfigPlotMultiVarRAD(ConfigSpatialPlot):
    sweep_angle_list: list =  field(default_factory=list)
    vars_RAD : list = field(default_factory=list)
    VHF_type: str = "sparkles&otherVHF"
    interpolation_grid: Optional[str] = None
    contour_binsize: Optional[float] = None
    list_plotting_objects:  List[PlottingObject] = field(default_factory=list)
    max_nr_panel_cols: Optional[int] = None
    merge_sweep_angles: bool = False
    
    vardata = None
    data_dirpath:str = None
    save:bool = False
    live_plot:bool = True
    outdir:str = None
    outname:str = "topview_multivar"
    

def draw_radar_image(
    ax,
    sweep_data,
    sweep_angle,
    var,
    cmap,
    cmap_norm,
    varlib,
    varkey,
    config,
    linewidth = 1,
    color_otherVHFcontour='midnightblue',
    color_sparklecontour='black'
    ):
    
    if varkey == "HMC":
        sweep_var = sweep_data["HMC"].argmax("hmc")
    elif var == "HMC_prob":
        sweep_var = sweep_data['HMC'].max("hmc")
    else:
        sweep_var = sweep_data[varkey]

    im = ax.pcolormesh(
        sweep_var.x, 
        sweep_var.y, 
        sweep_var, 
        norm=cmap_norm, 
        cmap=cmap, 
        transform=config.crs
        )

    # Use MaxNLocator to determine logical contour levels
    if sweep_angle < 11:
        contour_step = 2
    else:
        contour_step = 4

    contour_levels = np.arange(
        0, sweep_var.z.max() / 1000, contour_step,
        )
    alt_contour = ax.contour(
        sweep_var.x, sweep_var.y, 
        sweep_var.z / 1000, 
        levels=contour_levels, 
        linewidths=linewidth,
        colors='gold')

    ax.clabel(alt_contour, inline=True, fmt='%1.1f km')
    
    if "mask_otherVHF" in sweep_data.coords:
        # White contours for emphasis
        ax.contour(
            sweep_var.x, sweep_var.y, 
            np.where(sweep_data['mask_otherVHF'], 1, 0), 
            colors="white",
            alpha=1,
            linewidths=2*linewidth, 
            levels=[0.5], 
            transform=config.crs,
            zorder=5
            )
        
        # Different contour colors for disctintion
        ax.contour(
            sweep_var.x, sweep_var.y, 
            np.where(sweep_data['mask_otherVHF'], 1, 0), 
            colors=color_otherVHFcontour, 
            linewidths=linewidth, 
            linestyles=[(0, (4, 0.5))],
            levels=[0.5],
            transform=config.crs,
            zorder=6,
            )
        
    if "mask_sparkles" in sweep_data.coords:
        ax.contour( 
            sweep_var.x, sweep_var.y, 
            np.where(sweep_data['mask_sparkles'], 1, 0), 
            colors="white",
            alpha=1,
            linewidths=2*linewidth, 
            levels=[0.5], 
            transform=config.crs,
            zorder=5
            )
    
        ax.contour(
            sweep_var.x, sweep_var.y, 
            np.where(sweep_data['mask_sparkles'], 1, 0), 
            colors=color_sparklecontour, 
            linewidths=linewidth, 
            levels=[0.5], 
            linestyles=[(0, (1, 0.5))],
            transform=config.crs,
            zorder=6,
            )
        
    plot_borders( # With shapefiles in the dirpath
        config.dirpath_shapefiles_borders,
        ax,
        config.crs
        )
    # Add gridlines
    gridl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=0.5)
  
    return im, gridl

def plotter_multivar(
        sweep_angle: float,
        config: ConfigPlotMultiVarRAD,
        data_RAD: DataRADandLOFAR,
        outname: Optional[str]=None,
        outdir: Optional[str]=None,
        ):
    
    # Need to get the plot extent from somewhere if 
    if config.plot_extent is None:
        raise(ValueError("Need to define plot_extent in the config_plot!"))
    
    config.crs = CRSfromPROJ4(
        f"+proj=aeqd +lat_0={data_RAD.ds.latitude.values} +lon_0={data_RAD.ds.longitude.values} +x_0=0 +y_0=0 +datum=WGS84",
        extent=config.plot_extent,
        )

    linewidth = 1
    alphabet = string.ascii_letters

    # setting the nr of rows and columns of figure panels
    def calc_nrows_ncols(n, ncols_max):
        ncols = n // ncols_max + (n%ncols_max>0)
        nrows = n // ncols + (n%ncols>0)
        return nrows,ncols
    
    if config.max_nr_panel_cols is None:
        nrows, ncols = calc_nrows_ncols(
            len(config.vars_RAD),
            3,
            )
    else: 
        nrows, ncols = calc_nrows_ncols(
            len(config.vars_RAD),
            config.max_nr_panel_cols,
            )

    # # Set height ratios: plots get more space than colorbars
    # height_ratios = []
    # for row in range(nrows):
    #     if row % 2 == 0:  # Plot rows
    #         height_ratios.append(5)  # Larger ratio for plot rows
    #     else:  # Colorbar rows
    #         height_ratios.append(1)  # Smaller ratio for colorbar rows


    plt.rcParams['font.size'] = 8
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(
        nrows, ncols, 
        hspace=0.03,
        wspace=0.03,
        figure=fig,
        )

    i = 0

    # Select data sweep and calculate masks
    sweep = data_RAD.ds.sel(
        sweep_fixed_angle=sweep_angle, 
        method='nearest',
        )

    axes_mapping = {}
    for row in range(nrows):
        if i >= len(config.vars_RAD):
            break

        for col in range(ncols):
            var = config.vars_RAD[i]
            # var = plot_dict['var']
            varlib = config.vardata[var]
            varkey = varlib['ODIM']
            cmap, cmap_norm = getCmap(varlib)
            
            if varkey == "HMC":
                bounds = np.arange(-0.5, data_RAD.ds.hmc.size, 1)
                ticks = np.arange(0, data_RAD.ds.hmc.size)
                cmap = cmap
                cmap_norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # If it's the first plot, create it without sharing axes
            if i == 0:
                ax = fig.add_subplot(gs[row, col], projection=config.crs)
                shared_ax = ax  # Set this as the axis to share with others
            else:
                # Subsequent plots share x and y axes with the first plot (shared_ax)
                ax = fig.add_subplot(
                    gs[row, col], 
                    projection=config.crs, 
                    sharex=shared_ax, 
                    sharey=shared_ax)

            ax.set_extent(
                [
                    config.plot_extent.x[0], 
                    config.plot_extent.x[1], 
                    config.plot_extent.y[0], 
                    config.plot_extent.y[1]
                    ], 
                crs=config.crs)
            
            im, gridl = draw_radar_image(
                ax, 
                sweep, 
                sweep_angle, 
                var, 
                cmap, 
                cmap_norm, 
                varlib, 
                varkey, 
                config, 
                linewidth,
                )

            gridl.top_labels = False
            gridl.right_labels = False
            gridl.bottom_labels = False if row + 1 != nrows else True
            gridl.left_labels = False if col != 0 else True
            
            # Add colorbar based on the type of plot
            if varkey == "HMC":
                labels = [hm_type for hm_type in data_RAD.ds.hmc.values]
                cbar = fig.colorbar(
                    im, ax=ax, shrink=0.8, orientation="horizontal", 
                    ticks=ticks, norm=cmap_norm, pad=0.05, fraction=0.1)
                cbar.ax.set_xticklabels(labels, rotation=-90)
            else:
                cbar = fig.colorbar(
                    im, ax=ax, shrink=0.8, orientation="horizontal", 
                    norm=cmap_norm, extend='both', pad=0.05, fraction=0.1)
                cbar.ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
                cbar.ax.xaxis.set_minor_locator(mticker.NullLocator())
                cbar.set_label('{} [{}]'.format(varlib['symbol'], varlib['units']))

            ax.set_title(f"({alphabet[i]})",loc="left")
            
            axes_mapping[f'ax{i}'] = ax
            i += 1

    aspect_ratio_subplot = (
        config.plot_extent.y[1] - config.plot_extent.y[0]) / (config.plot_extent.x[1] - config.plot_extent.x[0])
    fig_width = ncols * 3 + 0.6
    fig_height = (nrows) *3.5 * aspect_ratio_subplot
    fig.set_size_inches(fig_width, fig_height)
    
    add_plotting_objects(config.list_plotting_objects, axes_mapping)

    if config.live_plot:
        plt.show()
    if config.save:
        if outname is None:
            outname = config.outname
        if outdir is None:
            outdir = config.outdir
        outpath = outpath_gen(config.data_dirpath, outdir, outname)
        fig.savefig(outpath, dpi=200)
        print("File is saved to: " + outpath)

    return config, fig
    
    

    return config, fig

def plotter_multivar_multielev(
        config: ConfigPlotMultiVarRAD,
        data_RAD,
        ):
    
    # Need to get the plot extent from somewhere if 
    if config.plot_extent is None:
        raise(ValueError("Need to define plot_extent in the config_plot!"))
    
    config.crs = CRSfromPROJ4(
        f"+proj=aeqd +lat_0={data_RAD.ds.latitude.values} +lon_0={data_RAD.ds.longitude.values} +x_0=0 +y_0=0 +datum=WGS84",
        extent=config.plot_extent,
        )
    
    linewidth = 1
    alphabet = string.ascii_letters
       
    # Need to get the plot extent from somewhere if 
    if config.plot_extent is None:
        raise(ValueError("Need to define plot_extent in the config_plot!"))
    
    config.crs = CRSfromPROJ4(
        f"+proj=aeqd +lat_0={data_RAD.ds.latitude.values} +lon_0={data_RAD.ds.longitude.values} +x_0=0 +y_0=0 +datum=WGS84",
        extent=config.plot_extent,
        )

    linewidth = 1
    plt.rcParams['font.size'] = 12
    alphabet = string.ascii_letters
    
    nrows = len(config.sweep_angle_list)
    ncols = len(config.vars_RAD)

    
    fig = plt.figure(constrained_layout=True)
    subplot_width = 3 #inch
    height_cbar = 0.2 #inch
    gs_height_ratios = [subplot_width for panel in range(nrows)]
    gs_height_ratios.append(height_cbar)
    gs = GridSpec(
        nrows+1, ncols, 
        height_ratios=gs_height_ratios,
        hspace=0.1/nrows,
        wspace=0.5/ncols,
        figure=fig,
        )

    i = 0

    axes_mapping = {}
    for row,sweep_angle in enumerate(config.sweep_angle_list):
                
        # Select radar data
        sweep = data_RAD.ds.sel(
            sweep_fixed_angle=sweep_angle, 
            method='nearest',
            )

        for col,var in enumerate(config.vars_RAD):
            # var = plot_dict['var']
            varlib = config.vardata[var]
            varkey = varlib['ODIM']
            cmap, cmap_norm = getCmap(varlib)
            
            if varkey == "HMC":
                bounds = np.arange(-0.5, data_RAD.ds.hmc.size, 1)
                ticks = np.arange(0, data_RAD.ds.hmc.size)
                cmap = cmap
                cmap_norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # If it's the first plot, create it without sharing axes
            if i == 0:
                ax = fig.add_subplot(gs[row, col], projection=config.crs)
                shared_ax = ax  # Set this as the axis to share with others
            else:
                # Subsequent plots share x and y axes with the first plot (shared_ax)
                ax = fig.add_subplot(
                    gs[row, col], 
                    projection=config.crs, 
                    sharex=shared_ax, 
                    sharey=shared_ax)

            ax.set_extent(
                [
                    config.plot_extent.x[0], 
                    config.plot_extent.x[1], 
                    config.plot_extent.y[0], 
                    config.plot_extent.y[1]
                    ], 
                crs=config.crs)
            
            im, gridl = draw_radar_image(
                ax, 
                sweep, 
                sweep_angle, 
                var, 
                cmap, 
                cmap_norm, 
                varlib, 
                varkey, 
                config, 
                linewidth,
                )

            gridl.top_labels = False
            gridl.right_labels = False
            gridl.bottom_labels = False if row + 1 != nrows else True
            gridl.left_labels = False if col != 0 else True
            
            ax.set_title(f"({alphabet[i]})",loc="center")
            axes_mapping[f'ax{i}'] = ax
        
            # Add colorbar below each column of panels
            if row+1 == nrows:
                cax = fig.add_subplot(gs[-1, col])   # Subplot occupying the GridSpec cell
                # cax.axis('off')                      # hide axes spine/ticks if desired
                if varkey == "HMC":
                    labels = [hm_type for hm_type in data_RAD.ds.hmc.values]
                    cbar = fig.colorbar(
                        im, cax=cax, orientation="horizontal", 
                        ticks=ticks, norm=cmap_norm, pad=0.05, fraction=0.1)
                    cbar.ax.set_xticklabels(labels, rotation=-90)
                else:
                    cbar = fig.colorbar(
                        im, cax=cax, orientation="horizontal", 
                        norm=cmap_norm, extend='both', pad=0.05, fraction=0.1)
                    cbar.ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
                    cbar.ax.xaxis.set_minor_locator(mticker.NullLocator())
                    cbar.set_label('{} [{}]'.format(varlib['symbol'], varlib['units']))
                   
            i += 1
    
    # Definite size of the figure
    aspect_ratio_subplot = (
        (config.plot_extent.y[1] - config.plot_extent.y[0]) / 
        (config.plot_extent.x[1] - config.plot_extent.x[0])
        ) # Need this for nicely shaped subpanels
    fig_width = ncols * subplot_width + 0.6
    fig_height = (nrows) *subplot_width*1.2 * aspect_ratio_subplot+height_cbar
    # hspace = 0.2/fig_height
    # wspace = 0.1/fig_width
    fig.set_size_inches(fig_width, fig_height)
    # fig.subplots_adjust(hspace=hspace,wspace=wspace)


    # Add titles with angle above each row
    fig.canvas.draw()  # ensure positions are up to date
    axs = fig.axes[:-ncols] #remove the axes of the cbars
    axs = np.reshape(axs, (nrows,ncols))
    for i in range(nrows):
        pos_ax_i = axs[i,0].get_position() 
        ax_y_top = pos_ax_i.y1
        offset_y = 0.1/fig_height # height of title above subplot
        fig.text(
            0.52, 
            ax_y_top + offset_y, 
            f"Elevation angle = {config.sweep_angle_list[i]}$^\circ$", 
            ha='center', 
            va='bottom', 
            fontweight='bold')
            
    # plt.subplots_adjust(top=0.95,
    #                     bottom=0.1,
    #                     left=0.05,
    #                     right=0.95,
    #                     hspace=0,
    #                     wspace=0)
    
    add_plotting_objects(config.list_plotting_objects, axes_mapping)

    if config.live_plot: #Show the figure, during the run
        plt.show()
    if config.save: # Save the figure
        outdir = config.outdir
        outpath = outpath_gen(config.data_dirpath, outdir, config.outname)
        fig.savefig(outpath, dpi=300)
        print("File is saved to: " + outpath)
    
    return config, fig
   
# %% Main function

# @SetKwargs(kwargs_main)
def main(
    config_plot: ConfigPlotMultiVarRAD,
    config_data_RAD: ConfigDataRAD,
    config_LOFAR: ConfigLOFAR = None,
    config_mask_RADnearVHF: ConfigMaskRADnearVHF = None,
    ):
    
    if (
            config_LOFAR is not None
        ) and (
            config_mask_RADnearVHF is None
        ):
        raise(ValueError("If plotting LOFAR, must also configure the settings to collect data near LOFAR, i.e. config_mask_RADnearVHF"))

    config_data_RAD.window_extent = config_plot.plot_extent
    
    # logger = customLOG(kwargs.outname,
    #                    print_console=kwargs.console,
    #                    logfile=kwargs.logfile,
    #                    outdir=kwargs.outdir,
    #                    outname=kwargs.outname,
    #                    verbose=kwargs.verbose)

    data = get_data_RADandLOFAR(
        config = config_data_RAD,
        config_LOFAR = config_LOFAR)
    
    # Collected radar data is used for the plotting options
    config_plot.vardata = data.vardata
    config_plot.data_dirpath = config_data_RAD.data_dirpath
    config_plot.plot_extent.transformer_lonlat_to_crs = data.window_extent.transformer_lonlat_to_crs
    config_plot.plot_extent = config_plot.plot_extent.transform_crs(crs_target=data.crs)   
    
    # Add LOFAR data ifLOFAR configuration is provided
    if config_LOFAR is not None:
        data.RAD = add_mask_RADnearVHF(
            data.RAD,
            data.LOFAR,
            config_mask_RADnearVHF,
            )
        
    # Multiple elevations: Individual figures or merged into one figure?
    if config_plot.merge_sweep_angles == True:
        config_plot_, fig = plotter_multivar_multielev(
            config = config_plot,
            data_RAD = data.RAD,
            )
    else:
        # Create individual figures for each elevation angle
        for elev in config_plot.sweep_angle_list:
            outname = f"{config_plot.outname}_elev{elev}.png"
            print(f"Plotting {outname}")
    
            config_plot_, fig = plotter_multivar(
                sweep_angle = elev,
                config = config_plot,
                data_RAD = data.RAD,
                outname = outname,
                )
        

    return config_plot_, fig

# %% If called from the Comman Line (CL)


# if __name__ == "__main__":
#     RunFromCL(kwargs_main)(main)()
