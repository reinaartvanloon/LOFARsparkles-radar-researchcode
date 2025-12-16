#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:50:51 2024

@author: rvloon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:56:39 2024

@author: rvloon
"""

import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pandas import to_datetime
import matplotlib.gridspec as gridspec
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from pyproj.transformer import Transformer
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter

#Custom funtions
from read_RAD import get_data_RADandLOFAR, ConfigDataRAD
from general import outpath_gen, lineProjection, plot_parallel_lines, getCmap,  ConfigSpatialPlot, add_plotting_objects, PlottingObject, plot_borders
from plot_LOFAR import masking_VHF_types, ConfigLOFAR, filter_mask_LOFAR, cluster_LOFARsparkles


#%% Functions
@dataclass
class ConfigPlotRADcrossSection(ConfigSpatialPlot):
    sweep_angle_list: list = field(default_factory=list)
    pointA: list = field(default_factory=list)
    pointB: list = field(default_factory=list)
    varlist_RAD : list = field(default_factory=list)
    crossxz_resolution: float = None
    interpolation_method: str = "nearest"
    VHF_type: str = "sparkles&otherVHF"
    VHFprojection_dist_to_Vplane: float = 0.0 
    interpolation_grid: Optional[str] = None
    contour_binsize: Optional[float] = None
    markersize: float = 1
    vardata = None
    datapath:str = None
    save:bool = False
    live_plot:bool = True
    outdir:str = None
    outname:str = "crosssect"
    list_plotting_objects:  List[PlottingObject] = field(default_factory=list)
    

def plotter(
        elev:float,
        config: ConfigPlotRADcrossSection,
        data,
        var,
        config_LOFAR:ConfigLOFAR = None,
        VHF_masks = None,
        outname = None,
        ):
    
    if outname is None: outname = config.outname
    varlib = config.vardata[var]
    varkey = varlib['ODIM']
    cmap, cmap_norm = getCmap(varlib)
    ds = data.RAD.ds
    ds['sweep_mode'] = data.RAD.ds['sweep_mode']
    
    config.crs = ccrs.AzimuthalEquidistant(
        central_longitude=ds.longitude.values, 
        central_latitude=ds.latitude.values,
        )
    
    # Hard coded Aestethics
    dpi = 400
    figsize = (8, 5)
    plt.rcParams['font.size'] = 9
    markersize=config.markersize
    linewidth = 0.5
    color_otherVHF = 'grey'
    color_sparkles = 'lightgrey'
    
    extent_width_ratios = [1, 2]
    extent_height_ratios = [1, 0.05]

    # Commence the plotting!
    # Create figure with a gridspec defined layout
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(
        2, 2, 
        width_ratios=extent_width_ratios, 
        height_ratios=extent_height_ratios,
        hspace=0.01,
        figure=fig,
        )
    
    title = "{} at {} UTC".format(varlib['symbol'],data.LOFAR.dt.strftime("%H:%M:%S"))

    # Subplot creation
    ax1 = fig.add_subplot(gs[0, 0], projection=config.crs)
    ax2 = fig.add_subplot(gs[0, 1])

    sweep = ds.sel(sweep_fixed_angle=elev, method='nearest')[varkey]

    if varkey == "HMC":
        sweep = sweep.argmax("hmc")
            
    im = ax1.pcolormesh(sweep.x,
                        sweep.y,
                        sweep,
                        norm=cmap_norm,
                        cmap=cmap)
    
    ax1.set_extent([
        config.plot_extent.x[0], 
        config.plot_extent.x[1],
        config.plot_extent.y[0], 
        config.plot_extent.y[1],
        ], 
        crs=config.crs)

    #Plot cross_sect section on horizontal plane.
    ax1.plot(
        (config.pointA[0], config.pointB[0]), 
        (config.pointA[1],config.pointB[1]),
        color='k', linestyle='--',linewidth=linewidth*2,
        marker='.',
        markersize=markersize,
        ) 
    
    ax1.annotate(
        "A", xy=config.pointA, xytext=(-5,5), 
        textcoords='offset points', weight="semibold")
    ax1.annotate(
        "B", xy=config.pointB, xytext=(-5,5), 
        textcoords='offset points', weight="semibold")
    
    plot_borders( # With shapefiles in the dirpath
        config.dirpath_shapefiles_borders,
        ax1,
        config.crs
        )
    
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', linewidth=linewidth/2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {
        "rotation":40,
        "ha": "center",
        "va":"bottom",
        }
    
    # Add major locators at every 0.1 degrees
    gl.xlocator = mticker.MultipleLocator(0.1)
    gl.ylocator = mticker.MultipleLocator(0.1)
    
    # (Optional) Format the labels to show fixed decimal places
    gl.xformatter = mticker.FormatStrFormatter('%.1f°E')
    gl.yformatter = mticker.FormatStrFormatter('%.1f°N')
    
    ax1.set_xlabel("")
     
    xy = data.RAD.cross_sect.xy.broadcast_like(data.RAD.cross_sect.z)
    
    if varkey == "HMC":
        crosssect_values = data.RAD.cross_sect["HMC"].argmax("hmc")
    elif varkey == "HMC_prob":
        crosssect_values = data.RAD.cross_sect["HMC"].max("hmc")
    else:
        crosssect_values = data.RAD.cross_sect[varkey]
        
    im_cross = ax2.pcolormesh(xy.values,
                              data.RAD.cross_sect.z.values,
                              crosssect_values,
                              norm=cmap_norm,
                              cmap=cmap)

    # Iterate through each sweep_fixed_angle
    nearest_sweep_angle = data.RAD.ds.sweep_fixed_angle.sel(sweep_fixed_angle= elev, method="nearest")
    for angle in data.RAD.cross_sect.sweep_fixed_angle.values:
        cross_sect_angle = data.RAD.cross_sect.sel(sweep_fixed_angle=angle)
        xy_values = cross_sect_angle.xy.values
        z_values = cross_sect_angle.z.values
        time = to_datetime(data.RAD.cross_sect.time.sel(sweep_fixed_angle=angle).mean('xy').values).strftime("%H:%M:%S")

        if angle == nearest_sweep_angle: color='k'
        else: color = 'lightgrey'
        ax2.plot(xy_values, z_values, color=color, linewidth=linewidth)
        
        i_max = max(enumerate(xy_values),key=lambda x: x[1])[0]
        z_rhs = z_values[i_max]/(config.plot_extent.z[1]-config.plot_extent.z[0])
        # if z_rhs<0.95:
        #     ax2.text(1.0, z_values[i_max]/(config.plot_extent.z[1]-config.plot_extent.z[0]), time, transform=ax2.transAxes, va='center')
            
    #Plot Lofar sources if provided
    if config_LOFAR is not None:      
        mask1 = VHF_masks[0]
        mask2 = VHF_masks[1]
        mask3 = VHF_masks[2]
        #Horizontal cross section
        ax1.scatter(data.LOFAR.df.x[mask1], data.LOFAR.df.y[mask1], c=color_otherVHF, marker='.', s=markersize*0.5)  #XY-topview LOFAR data
        ax1.scatter(data.LOFAR.df.x[mask2], data.LOFAR.df.y[mask2], c=color_sparkles, s=markersize*5, marker = 'v')  #XY-topview LOFAR data
        ax1.scatter(data.LOFAR.df.x[mask3], data.LOFAR.df.y[mask3], c=color_sparkles, s=markersize*5, marker='v', edgecolor='k', linewidths=0.1*markersize)
        
        #Vertical cross section
        if config.VHFprojection_dist_to_Vplane:
            LOFARdf, mask_proj = lineProjection(
                config.pointA, 
                config.pointB, 
                data.LOFAR.df, 
                perp_range= config.VHFprojection_dist_to_Vplane,
                )
            mask1 = mask1[mask_proj]
            mask2 = mask2[mask_proj]
            mask3 = mask3[mask_proj]
            ax2.scatter(LOFARdf['projection'][mask1], LOFARdf['z'][mask1], c=color_otherVHF, marker='.', s=markersize*0.5)
            ax2.scatter(LOFARdf['projection'][mask2], LOFARdf['z'][mask2], c=color_sparkles, s=markersize*5,  marker = 'v')  #XY-topview LOFAR data
            ax2.scatter(LOFARdf['projection'][mask3], LOFARdf['z'][mask3], c=color_sparkles, s=markersize*5,  marker = 'v', edgecolor='k', linewidths=markersize*0.1)  #XY-topview LOFAR data

            if config. VHFprojection_dist_to_Vplane is not None:
                parallel_lines = plot_parallel_lines(
                    ax1, 
                    config.pointA, 
                    config.pointB, 
                    config.VHFprojection_dist_to_Vplane,
                    )
                [line.set(linestyle=':', color='k',linewidth=linewidth*2) for line in parallel_lines]
        
    ax2.set_ylim(config.plot_extent.z[0], config.plot_extent.z[1])
    
    # Define a formatting function: divide by 1000 and format as int
    def in_km(x, pos):
        return f"{int(x/1000)}"
    
    # Attach it as the major‐axis formatter
    ax2.xaxis.set_major_formatter(FuncFormatter(in_km))
    ax2.yaxis.set_major_formatter(FuncFormatter(in_km))
   
    # ax2.axhline(y=kwargs.altitude, linestyle=':', color='k', linewidth=markersize)
    ax2.annotate("A", xy=(-0.01,1.01), xycoords="axes fraction", weight="semibold")
    ax2.annotate("B", xy=(0.99,1.01), xycoords="axes fraction", weight="semibold")
    ax2.set_ylabel("Altitude [km]")
    ax2.set_xlabel("Distance along cross section [km]")

    # Colorbar
    cbar_ax = fig.add_subplot(gs[1, :])
    if varkey == "HMC":
        ticks = np.arange(0, ds.hmc.size)
        cbar = plt.colorbar(im, cax=cbar_ax, 
                     label='{}'.format(varlib['symbol']), 
                     orientation="horizontal",
                     ticks=ticks,
                     fraction=0.046, 
                     norm=cmap_norm, 
                     pad=0.05)
        labels = [hm_type for hm_type in data.RAD.ds.hmc.values]
        cbar.ax.set_xticklabels(labels)
    else:
        plt.colorbar(im, cax=cbar_ax, 
                     label='{} [{}]'.format(varlib['symbol'], 
                                            varlib['units']), 
                     extend='both', orientation="horizontal",
                     norm=cmap_norm)

    # Set titles and labels
    ax1.set_title(
        "(a) elevation angle = {:.1f}$^\circ$".format(
            data.RAD.ds.sweep_fixed_angle.sel(
                sweep_fixed_angle= elev, 
                method='nearest'
                ).item()
            )
        )
    ax2.set_title("(b) Vertical cross section from A to B")
        
    ## add the objects from the _list_plotting_objects
    add_plotting_objects(
        config.list_plotting_objects, 
        axes_mapping = {
            "ax1":ax1, 
            "ax2":ax2,
            }
        )
    
    fig.suptitle(title)

    #Saving
    if config.save == True:
        outname = outname+".png"
        outdir = os.path.join(config.outdir, varkey)
        outpath = outpath_gen(config.datapath, outdir,outname)
        fig.savefig(outpath, dpi=dpi)
        print("File is saved to: "+outpath)
        plt.show()
    else:
        print("File is not saved")
        plt.show()
        
#%%

def main(
    config_plot: ConfigPlotRADcrossSection,
    config_data_RAD: ConfigDataRAD,
    config_LOFAR: ConfigLOFAR = None,
    ):
    
    config_data_RAD.window_extent = config_plot.plot_extent

    data = get_data_RADandLOFAR(
        config = config_data_RAD,
        config_LOFAR = config_LOFAR,
        )
    
    data.stormcode = config_data_RAD.stormcode
    
    # Collected radar data is used for the plotting options
    config_plot.vardata = data.vardata
    config_plot.datapath = config_data_RAD.data_dirpath
    
    # We retrieve a transormed WindowExtent object in the data crs
    # and store it in the plotting configuration
    config_plot.plot_extent = data.window_extent
    
    #Transforming lon-lat coordinates to appropriate coord systems
    crs = f"+proj=aeqd +lat_0={data.RAD.ds.latitude.values} +lon_0={data.RAD.ds.longitude.values} +x_0=0 +y_0=0 +datum=WGS84"
    transform2lonlat = Transformer.from_crs(f"{crs}", "EPSG:4326", always_xy=True)

    config_plot.pointA = transform2lonlat.transform(*config_plot.pointA, direction="INVERSE")
    config_plot.pointB = transform2lonlat.transform(*config_plot.pointB, direction="INVERSE")
    
    #Make vertical cross_section
    # First convert the varlist to the ODIM variable keys
    varkeys = [data.vardata[var]['ODIM'] for var in config_plot.varlist_RAD]
    # Now get a vertical crosssection
    data.RAD.cross_sect = data.RAD.CrossSection(
        config_plot.pointA, 
        config_plot.pointB, 
        variables=varkeys, 
        resolution=config_plot.crossxz_resolution,
        method=config_plot.interpolation_method,
        )
    
    VHF_masks = None
    if config_LOFAR is not None:
        mask_not_large_clusters, mask_sparkles = cluster_LOFARsparkles(
            config = config_LOFAR,
            data_LOFAR = data.LOFAR,
            sparkle_params = config_LOFAR.sparkle_params,
            )
    
        # Want to impose some more restrictions for something to be a sparkle
        mask_sparkles_filtered = filter_mask_LOFAR(
            mask_sparkles, 
            data_LOFAR = data.LOFAR, 
            sparkle_params = config_LOFAR.sparkle_params, 
            crs_data = crs,
            )
        
        data.LOFAR, [mask_sparkles, mask_sparkles_filtered] = data.LOFAR.select_2_windowextent(
            window_extent = config_plot.plot_extent, 
            crs_target = config_LOFAR.crs, 
            mask_list = [mask_sparkles, mask_sparkles_filtered],
            )    
        
        VHF_masks = masking_VHF_types(
            mask_sparkles, 
            mask_sparkles_filtered, 
            config_plot.VHF_type,
            )


    for elev in config_plot.sweep_angle_list:
        for var in config_plot.varlist_RAD:
            outname = f"{config_plot.outname}_{data.LOFAR.dt.strftime('%H%M%S')}_var-{var}_elev{elev}"
            print(f"Plotting {outname}")
            plotter(
                elev,
                config_plot,
                data,
                var,
                config_LOFAR = config_LOFAR,
                VHF_masks = VHF_masks,
                outname = outname,
                )
   