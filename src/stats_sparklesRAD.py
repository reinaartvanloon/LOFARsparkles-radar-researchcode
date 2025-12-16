#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:33:49 2024

@author: rvloon
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import numpy as np
import wradlib as wrl
import json
import psutil
import xarray as xr
import matplotlib.ticker as mticker
from dataclasses import dataclass
from typing import Optional

# Custom funtions
from read_RAD import get_data_RADandLOFAR, ConfigDataRAD, add_mask_RADnearVHF, ConfigMaskRADnearVHF #, add_RAD_near_sparkles_mask_multiradii, kwargs_add_RAD_near_sparkles_mask_multiradii
from general import outpath_gen
from plot_LOFAR import SparkleParams, ConfigLOFAR
from plot_RAD_multivar import ConfigPlotMultiVarRAD
from plot_RAD_multivar import plotter_multivar as topview_plotter

# %% Function to colect radar data near LOFAR sources, sparkles vs non-sparkle VHF

def statistics_sparkles_vs_other(
        RAD_vars,
        config_data_RAD: ConfigDataRAD,
        config_LOFAR: ConfigLOFAR,
        config_mask_RADnearVHF,
        config_topview_plots = None,
        ):
    """Collect radar data around sparkles and other VHF sources."""

    LOFARcode = config_LOFAR.LOFAR_file[:4]
    
    data = get_data_RADandLOFAR(
        config_data_RAD, 
        config_LOFAR = config_LOFAR)
    
    # Want to select the radar data that are "close" in in distance to lighting data
    # Mark with dataset coordinates: "mask_otherVHF" and "mask_sparkles"
    data.RAD = add_mask_RADnearVHF(
        data.RAD,
        data.LOFAR,
        config = config_mask_RADnearVHF
        ) 
    
    # Plots to verify that the radar data selection is going alright
    if config_topview_plots is not None:
        # Collected radar data is used for the plotting options
        config_topview_plots.vardata = data.vardata
        config_topview_plots.datapath = config_data_RAD.datapath
        # config_topview_plots.plot_extent.transformer_lonlat_to_crs = data.window_extent.transformer_lonlat_to_crs
        config_topview_plots.plot_extent = config_topview_plots.plot_extent.transform_crs(crs_target=data.crs)   
        
        print("Plotting topviews!")
        for elev in config_topview_plots.sweep_angle_list:
            outname=f"{config_topview_plots.outname}_elev{elev}.png"
            outdir=f"{config_topview_plots.outdir}/{LOFARcode}/"
            
            topview_plotter(
                sweep_angle = elev,
                config = config_topview_plots,
                data_RAD = data.RAD,
                outname=outname,
                outdir=outdir,
                )

    # Can only store so much data in memory
    # So stack and keep only the radar data
    # marked with the dataset coordinates: "mask_otherVHF" and "mask_sparkles"
    data.RAD.ds = data.RAD.ds.stack(radar_vol=("sweep_fixed_angle", "azimuth", "range"))
    mask_total = data.RAD.ds.mask_otherVHF | data.RAD.ds.mask_sparkles

    ds_subset = data.RAD.ds.isel(radar_vol=mask_total.values.ravel(), drop=True)

    # Save memory by only keeping relevant coordinates
    relevant_coords = ['z', 'hmc', 'radar_vol', 'mask_otherVHF', 
                       'mask_sparkles', "sweep_fixed_angle", 
                       "azimuth", "range"]
    RADvarkeys = [data.vardata[var]['ODIM'] for var in RAD_vars]
    vars_redundant = [var for var in ds_subset.variables if var not in RADvarkeys + relevant_coords]
    ds_subset = ds_subset.drop_vars(vars_redundant)

    del data
    return ds_subset

import logging

def setup_logger(console: bool, logfile: bool, verbose: bool):
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    if console:
        logging.getLogger().addHandler(logging.StreamHandler())

# %%Get configfiles

@dataclass
class ConfigPlotSparkleStats:
    LOFARfile_list : list
    varlist : list
    together : bool
    histogram_1D: bool = True 
    histogram_2D: bool = True
    topview_plots: Optional[ConfigPlotMultiVarRAD] = False
    save: bool = False
    outname: str = None
    outdir: str = None
#%% Run

def make_plots(
        config_plots,
        ds,
        vardata, 
        outname, 
        outdir,
        save=False,
        ):
    
    plt.rcParams.update({'font.size': 11})

    var_range = {
        'dbzh': [0, 70],
        'vradh': [-35, 15],
        'wradh': [0, 10],
        "zdr": [-2, 1],
        "rhohv": [0.95, 1],
        "kdp": [-2, 2]
    }

    binsize = {
        'dbzh': 5,
        'vradh': 2.5,
        'wradh': 0.5,
        "zdr": 0.2,
        "rhohv": 0.005,
        "kdp": 0.2}
    
    colors = ['lightcoral', 'lightsteelblue']
    labels = ['Sparkles', 'Other VHF sources']
    
    if config_plots.histogram_1D ==True:
        plot_histogram(
            ds, 
            config_plots.varlist,
            vardata, 
            var_range, 
            binsize, 
            colors, 
            labels, 
            outname=outname+"_Zh-Wrad", 
            outdir=outdir+"/histograms_Zh-Wrad_1D",
            save=config_plots.save,
            )
    
    if config_plots.histogram_2D == True:
        plot_2D_histograms(
            ds,
            vardata,
            var_range,
            binsize,
            outname=outname+"_Zh-Wrad_2D",
            outdir=outdir+"/histograms_Zh-Wrad_2D",
            save=config_plots.save,
            )
    
    if "hmc" in config_plots.varlist:
        hmc_types = ds.hmc.values # ['LR', 'MR', 'HR', etc.]
        HMC_data = prep_HMC_data(ds,hmc_types,config_plots,vardata)
        plot_histograms_HMC(
            HMC_data,
            hmc_types,
            outname=outname+"_hmc",
            outdir=outdir+"/histograms__hmc",
            save=config_plots.save,
            )
        
def main(
    config_plots : ConfigPlotSparkleStats,
    config_data_RAD: ConfigDataRAD,
    sparkle_params: SparkleParams,
    config_LOFAR: ConfigLOFAR,
    config_mask_RADnearVHF: ConfigMaskRADnearVHF,
    config_topview_plots: Optional[ConfigPlotMultiVarRAD] = None
    ):
    
    # Load variable data
    with open(os.path.join(config_data_RAD.data_dirpath, "variables_metadata.json")) as file:
        vardata = json.load(file)

    print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

    
    if config_plots.together: #Results of different LOFAR images added togehter
        for i, file in enumerate(config_plots.LOFARfile_list):
            print(file)
            LOFARcode = file[:4]
            config_LOFAR.LOFAR_file = file

            ds = statistics_sparkles_vs_other(
                config_plots.varlist,
                config_data_RAD,
                config_LOFAR,
                config_mask_RADnearVHF,
                config_topview_plots = config_topview_plots,
                )

            ds = ds.assign_coords(dataset=LOFARcode)
            
            nr_sparklebins = np.sum(ds.where(ds.dataset == LOFARcode).mask_sparkles.values)
            nr_otherbins = np.sum(ds.where(ds.dataset == LOFARcode).mask_otherVHF.values & 
                  (~ds.where(ds.dataset == LOFARcode).mask_sparkles.values))

            print(f"nr of sparkle radar bins: {nr_sparklebins}")
            print(f"nr of other lightning radar bins: {nr_otherbins}")
            print("")
            
            if i == 0:
                ds_all = ds.copy()
            else:
                ds_all = xr.concat([ds_all, ds], dim='radar_vol')

            # Print memory usage in MB
            print(
                f"Memory usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

        ds = ds_all
                
        make_plots(
            config_plots, 
            ds, 
            vardata, 
            outname="aggregatedstats", 
            outdir=config_plots.outdir,
            save=config_plots.save,
            )
                   
        
    else: #Results of different LOFAR images in different graphs
        for i, file in enumerate(config_plots.LOFARfile_list):
            print(file)
            LOFARcode = file[:4]
            config_LOFAR.LOFAR_file = file

            ds = statistics_sparkles_vs_other(
                config_plots.varlist,
                config_data_RAD,
                config_LOFAR,
                config_mask_RADnearVHF,
                config_topview_plots = config_topview_plots,
                )

            ds = ds.assign_coords(dataset=LOFARcode)
            
            nr_sparklebins = np.sum(ds.where(ds.dataset == LOFARcode).mask_sparkles.values)
            nr_otherbins = np.sum(ds.where(ds.dataset == LOFARcode).mask_otherVHF.values & 
                  (~ds.where(ds.dataset == LOFARcode).mask_sparkles.values))

            print(f"nr of sparkle radar bins: {nr_sparklebins}")
            print(f"nr of other lightning radar bins: {nr_otherbins}")
            print("")

            # Print memory usage in MB
            print(
                f"Memory usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")
            
            make_plots(
                config_plots, 
                ds, 
                vardata, 
                outname=LOFARcode, 
                outdir=config_plots.outdir,
                save=config_plots.save,
                )
            
# %% Plotting variables

def prep_HMC_data(
        ds,
        hmc_types: list,
        config_plots : ConfigPlotSparkleStats,
        vardata,
        ):
    for var in config_plots.varlist:
        VAR = vardata[var]['ODIM']
        if var == "hmc":
            ds = ds.assign_coords({
                'HMC_class': ds[VAR].argmax('hmc')})
            hmc_mapping = {i: label for i, label in enumerate(hmc_types)}
    
            data_sparkles = ds.HMC_class.where(
                ds.mask_sparkles, 
                drop=True
                ).astype("int64")
            n_sparkles = data_sparkles.size
            hist_sparkles = (
                np.bincount(
                    data_sparkles, 
                    minlength=ds.hmc.size
                    )
                /n_sparkles*100
                )
    
            data_other = ds.HMC_class.where(
                ds.mask_otherVHF & (~ds.mask_sparkles)
                , drop=True
                ).astype("int64")
            n_other = data_other.size
            hist_other = (
                np.bincount(
                    data_other, 
                    minlength=ds.hmc.size
                    )
                /n_other*100
                )
    
            # Probability matrix for heatmapHMC_sparkles
    
            # Averaging over all points with same HMC_class and same hmc type
            prob_matrix_sparkles = ds.HMC.where(
                ds.mask_sparkles, drop=True).groupby(
                    'HMC_class').mean()
            prob_matrix_other = ds.HMC.where(
                ds.mask_otherVHF, drop=True).groupby(
                    'HMC_class').mean()
    
            # Change the HMC_class coordinate back to strings
            prob_matrix_sparkles = prob_matrix_sparkles.assign_coords(
                HMC_class=[hmc_mapping[i] for i in prob_matrix_sparkles['HMC_class'].values]
            )
            prob_matrix_other = prob_matrix_other.assign_coords(
                HMC_class=[hmc_mapping[i] for i in prob_matrix_other['HMC_class'].values]
            )
    
            # Reindex to fill hmc types that were not present in HMC_class
            prob_matrix_sparkles = prob_matrix_sparkles.reindex({
                "hmc": hmc_types,
                "HMC_class": hmc_types, })
            prob_matrix_other = prob_matrix_other.reindex({
                "hmc": hmc_types,
                "HMC_class": hmc_types, })
            
            HMC_aggr_sparkles=None
            HMC_aggr_other=None
    
        elif var == 'hmc_aggregate':
            nr_sparkles = np.sum(ds.mask_sparkles)
            nr_other = np.sum(ds.mask_otherVHF & (
                ~ds.mask_sparkles))
    
            prob_sparkles = ds[VAR].where((ds.mask_sparkles)).sum(dim=[
                'radar_vol'])
            prob_other = ds[VAR].where(ds.mask_otherVHF & (
                ~ds.mask_sparkles)).sum(dim=['radar_vol'])
    
            HMC_aggr_sparkles = prob_sparkles.values / nr_sparkles.values
            HMC_aggr_other = prob_other.values / nr_other.values
    
    return {
        "hist_sparkles":hist_sparkles,
        "hist_other":hist_other,
        "matrix_sparkles":prob_matrix_sparkles,
        "matrix_other":prob_matrix_other,
        "aggr_sparkles":HMC_aggr_sparkles,
        "aggr_other":HMC_aggr_other,
        "n_sparkles":n_sparkles,
        "n_other":n_other,
        }
            

    # else:
    #     data_sparkles = ds[VAR].where(ds.mask_sparkles).values()
    #     data_other = ds[VAR].where(ds.mask_otherVHF & (
    #         ~ds.mask_sparkles)).values()


# %% Histograms for variables (not HMC)

def plot_histogram(
        ds, 
        varlist,
        vardata, 
        var_range, 
        binsize, 
        colors, 
        labels, 
        outname, 
        outdir, 
        save=False):
    
    
    varkeys = [vardata[var]['ODIM'] for var in varlist]
    ncols = 2
    nrows = 1
    
    fig = plt.figure(
        figsize=(5 * ncols, 4 * nrows), 
        dpi=400, 
        constrained_layout=True,
        )
    gs0 = gridspec.GridSpec(
        nrows, ncols, 
        top=0.82
        )
    
    i = 0
    for row in range(nrows):
        if i == len(varlist):
            break
        for col in range(ncols):
            var = varlist[i]
            # Plot histograms for each variable
            ax = fig.add_subplot(gs0[row, col])
            ax.ticklabel_format(axis='x', scilimits=[0, 0])
            ax.set_yticks([])
            ax.set_xticks([])
    
            gs = gridspec.GridSpecFromSubplotSpec(
                1, 2, 
                wspace=0, 
                subplot_spec=gs0[row, col],
                )
    
            VAR = varkeys[i]
    
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
    
            data_sparkles = ds[VAR].where(
                ds.mask_sparkles, 
                drop=True,
                ).values
            
            data_other = ds[VAR].where(
                ds.mask_otherVHF & (~ds.mask_sparkles),
                drop=True,
                ).values
        
            bins = np.arange(var_range[var][0],
                             var_range[var][1], binsize[var])
            
            # counts_sparkles, _ = np.histogram(
            #     data_sparkles, 
            #     bins=bins,
            #     weights=-np.ones_like(data_sparkles)/data_sparkles.size,
            #     )
            # counts_other, _ = np.histogram(
            #     data_other, 
            #     bins=bins,
            #     weights=np.ones_like(data_other)/data_other.size,
            #     )
    
            counts_sparkles, _, _ = ax1.hist(
                data_sparkles, 
                weights=-np.ones_like(data_sparkles)/data_sparkles.size,
                bins=bins, 
                orientation='horizontal', 
                color=colors[0], 
                edgecolor='tab:red', 
                label=labels[0],
                )
            counts_other, _, _ = ax2.hist(
                data_other, 
                weights=np.ones_like(data_other)/data_other.size,
                bins=bins, 
                orientation='horizontal',
                color=colors[1], 
                edgecolor='tab:blue', 
                label=labels[1])
    
            # Add horizontal lines at mean values
            ax1.axhline(np.nanmean(data_sparkles), color='k',
                        linestyle='--', linewidth=2)
            ax2.axhline(np.nanmean(data_other), color='k',
                        linestyle='--', linewidth=2)
    
            ax1.set_yticks(bins[::2])
            ax2.set_yticks([])
    
            ax1.set_ylim(var_range[var][0], var_range[var][1])
            ax2.set_ylim(var_range[var][0], var_range[var][1])
    
            # Set x-limits to reduce redundant space
            xlim = [
                0,
                1.1*max(-1*counts_sparkles.min(), counts_other.max()),
                ]
            ax1.set_xlim(-xlim[1], xlim[0])
            ax2.set_xlim(xlim[0], xlim[1])
    
            ax1ticklabels = [
                "{}%".format(int(abs(tick*100))) for tick in ax1.get_xticks()[1:-1]
                ] + ['']
            ax2ticklabels = [''] + [
                "{}%".format(int(tick*100)) for tick in ax2.get_xticks()[1:-1]
                ] 
    
            ax1.set_xticks(
                ax1.get_xticks()[1:], 
                labels=ax1ticklabels, 
                ha='center',
                )
            ax2.set_xticks(
                ax2.get_xticks()[:-1], 
                labels = ax2ticklabels, 
                ha='center',
                )
           
            ax1.set_ylabel(
                f"{vardata[var]['symbol']} [{vardata[var]['units']}]")
    
            print(f"For {var} variable: ")
            print(f"sparkle mean: {np.nanmean(data_sparkles)}")
            print(f"other mean: {np.nanmean(data_other)}")
    
            ax.set_title(f"{vardata[var]['longname']} [{vardata[var]['units']}]")
            
            ax1.spines['left'].set_visible(True)
            # ax1.spines['left'].set_linestyle(':')
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
    
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_visible(True)
            # ax2.spines['right'].set_linestyle(':')
            ax2.spines['top'].set_visible(False)
    
            ax2.set_yticklabels([])
    
            # Add a legend at the top
            handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
            fig.legend(handles, labels, loc='upper center',
                       bbox_to_anchor=(0.5, 1), ncol=2)
            
            ax1.annotate(
                text = r"n = {}".format(data_sparkles.size), 
                xy = (0.3,0.8),
                xycoords = 'axes fraction',
                color="tab:red",
                weight="semibold"
                )
            ax2.annotate(
                text = r"n = {}".format(data_other.size), 
                xy = (0.3,0.8),
                xycoords = 'axes fraction',
                color="tab:blue",
                weight="semibold"
                )
    
            # fig.tight_layout(rect=[0, 0, 1, 0.95])
    
            i += 1
    
    if save==True:
        if outdir == None:
            raise(ValueError(outdir), "If save=True, needs a directry to save")
        outpath = outpath_gen(outdir, outdir, outname)
        fig.savefig(outpath, dpi=400)
        print("File is saved to: "+outpath+".png")

# %% Histograms for HMC

# Function to add spaces (gaps) between the rows by inserting NaNs
# Function to create a grid for pcolormesh with gaps between rows


def create_grid_with_v_gaps(data, gap_size=0.1):
    n_rows, n_cols = data.shape
    dy = 1 / 2 * gap_size
    # Create the grid, with extra space between rows
    y0 = np.arange(n_rows + 1)
    y = [[y_i - dy, y_i + dy] for y_i in y0]
    y = sum(y, [])[1:-1]
    x = np.arange(n_cols + 1)

    new_shape = (n_rows + n_rows - 1, n_cols)
    data_with_nans = np.full(new_shape, np.nan)
    data_with_nans[::2] = data

    return x, y, data_with_nans

def plot_histograms_HMC(data, hmc_types, outname, outdir, save=False):
    pr_types = wrl.classify.pr_types
    hm_longnames = {}
    for k, v in pr_types.items():
        hm_longnames[v[0]] = v[1]
        
    # Reversed order of rows feels more natural in plot
    data['hist_sparkles'] = data['hist_sparkles'][::-1]
    data['hist_other'] = data['hist_other'][::-1]
    hmc_types = hmc_types[::-1]
    prob_matrix_sparkles = data["matrix_sparkles"][:,::-1]
    prob_matrix_other = data["matrix_other"][:,::-1]
    
    # Transposing because info per hmc_type are on the colums, and not on rows
    x, y, prob_matrix_sparkles_withgaps = create_grid_with_v_gaps(prob_matrix_sparkles.transpose('HMC_class', 'hmc').values, gap_size=0.2)
    x, y, prob_matrix_other_withgaps = create_grid_with_v_gaps(prob_matrix_other.transpose('HMC_class', 'hmc').values, gap_size=0.2)

    fontsize=8
    ncols = 3
    nrows = 1
    colors = ['lightcoral', 'lightsteelblue']
    
    # Panel a
    # Histograms
    fig = plt.figure(figsize=(
        4 * ncols, 3 * nrows), 
        dpi=400, 
        constrained_layout=True,
        )
    gs0 = gridspec.GridSpec(
        nrows, ncols, 
        width_ratios= [0.8,1,1],
        wspace=0.05,
        )
    
    # Subplot for max HMC classification
    ax_max = fig.add_subplot(gs0[0, 0])
    ax_max.ticklabel_format(axis='x', scilimits=[0, 0])
    ax_max.set_yticks([])
    ax_max.set_xticks([])
    
    gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, wspace=0.005, subplot_spec=gs0[0, 0])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # ax1.grid(axis="x", which="both", zorder=1)
    # ax2.grid(axis="x", which="both", zorder=1)
    
    ax1.barh(
        range(len(hmc_types)), -data['hist_sparkles'], 
        align='center',
        height=0.8, 
        color=colors[0], 
        edgecolor='tab:blue',
        zorder=2
        )
    ax2.barh(
        range(len(hmc_types)), 
        data['hist_other'] , 
        align='center',
        height=0.8, 
        color=colors[1], 
        edgecolor='tab:blue',
        zorder=2
        )
    
    ax1.set_yticks(range(len(hmc_types)))
    ax1.set_yticklabels([hm_longnames[hm_type]
                        for hm_type in hmc_types])
    ax2.set_yticks([])
    
    ax1.set_ylim(-0.5, len(hmc_types) - 0.5)
    ax2.set_ylim(-0.5, len(hmc_types) - 0.5)
    
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax1.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax2.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    
    # Set x-limits to reduce redundant space
    xlim = np.max(np.concatenate((data["hist_sparkles"], data["hist_other"])))
    ax1.set_xlim(-xlim * 1.05, 0)
    ax2.set_xlim(0, xlim * 1.05)
        
    ax1ticks = [tick for tick in ax1.get_xticks() if ((tick<=0) and (tick>-0.9*xlim))]
    ax1ticklabels = [
        "{}%".format(int(abs(tick))) for tick in ax1ticks[:-1] 
        ] + ['']
    ax2ticks = [tick for tick in ax2.get_xticks() if ((tick>=0) and (tick<0.9*xlim))]
    ax2ticklabels = [''] + [
        "{}%".format(int(tick)) for tick in ax2ticks[1:]
        ] 
    
    print("ax2ticks: ", ax2ticks)
    ax1.set_xticks(ax1ticks, labels=ax1ticklabels)
    ax2.set_xticks(ax2ticks, labels=ax2ticklabels)
    
    ax1.spines['left'].set_visible(True)
    # ax1.spines['left'].set_linestyle(':')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    # ax2.spines['right'].set_linestyle(':')
    ax2.spines['top'].set_visible(False)
    
    ax_max.set_title("HMC")
    
    ax1.annotate(
        text = r"n = {}".format(data["n_sparkles"]), 
        xy = (0.3,0.8),
        color="tab:red",
        xycoords = 'axes fraction',
        weight="semibold",
        zorder=3,
        )
    ax2.annotate(
        text = r"n = {}".format(data["n_other"]), 
        color="tab:blue",
        xy = (0.3,0.8),
        xycoords = 'axes fraction',
        weight="semibold",
        zorder=3
        )
    
    # Panel b    
    # Subplot for probability matrix sparkles
    ax_heatmap_sparkles = fig.add_subplot(gs0[0, 1])
    heatmap_sparkles = ax_heatmap_sparkles.pcolormesh(
        x, y, prob_matrix_sparkles_withgaps*100,
        cmap='Reds', edgecolors='none',
        vmax=100,
        )
    
    # Add colorbar for the heatmap
    cbar_sparkles = fig.colorbar(heatmap_sparkles, ax=ax_heatmap_sparkles)
    cbar_sparkles.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    
    # Set labels for axs
    ax_heatmap_sparkles.set_xticks(np.arange(len(hmc_types)) + 0.5)
    ax_heatmap_sparkles.set_yticks(np.arange(len(hmc_types)) + 0.5)
    ax_heatmap_sparkles.set_xticklabels(hmc_types[::-1], rotation=45, ha='center')
    ax_heatmap_sparkles.set_yticklabels([])
    ax_heatmap_sparkles.set_ylim((0, len(hmc_types)))
    
    # Annotate the squares with the values from prob_matrix_sparkles
    for i in range(len(hmc_types)):
        for j in range(len(hmc_types)):
            value = prob_matrix_sparkles.transpose('HMC_class', 'hmc').values[i, j]
            if not np.isnan(value) and value>=0.01:  # Only annotate non-NaN values
                ax_heatmap_sparkles.text(
                    j + 0.5, i + 0.5, 
                    f'{value*100:.0f}', 
                    ha='center', va='center', color='black', size=fontsize,
                    )
    
    ax_heatmap_sparkles.set_title("Mean probability (Sparkles)")
    
    # Panel c
    # Subplot for probability matrix other VHF
    ax_heatmap_other = fig.add_subplot(gs0[0, 2])
    heatmap_other = ax_heatmap_other.pcolormesh(
        x, y, prob_matrix_other_withgaps*100, 
        cmap='Blues', edgecolors='none',
        vmax=100,
        )
    
    # Add colorbar for the heatmap
    cbar_other = fig.colorbar(heatmap_other, ax=ax_heatmap_other)
    cbar_other.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    
    # Set labels for axs
    ax_heatmap_other.set_xticks(np.arange(len(hmc_types)) + 0.5)
    ax_heatmap_other.set_yticks(np.arange(len(hmc_types)) + 0.5)
    ax_heatmap_other.set_xticklabels(hmc_types[::-1], rotation=45, ha='center')
    ax_heatmap_other.set_yticklabels([])
    ax_heatmap_other.set_ylim((0, len(hmc_types)))
    
    # Annotate the squares with the values from prob_matrix_other
    for i in range(len(hmc_types)):
        for j in range(len(hmc_types)):
            value = prob_matrix_other.transpose('HMC_class', 'hmc').values[i, j]
            if not np.isnan(value) and value>=0.01:  # Only annotate non-NaN values
                ax_heatmap_other.text(
                    j + 0.5, i + 0.5, 
                    f'{value*100:.0f}', 
                    ha='center', va='center', color='black', size=fontsize,
                    )
    
    ax_heatmap_other.set_title("Mean probability (other VHF)")
    
    plt.show()
    
    if save==True:
        if outdir == None:
            raise(ValueError(outdir), "If save=True, needs a directry to save")
        outpath = outpath_gen(outdir, outdir, outname)
        fig.savefig(outpath, dpi=400)
        print("File is saved to: "+outpath+".png")


# %% Scatter plots

def plot_scatter(ds,colors,vardata):
    dbzh_sparkles = ds.DBZH.where(ds.mask_sparklesu)
    dbzh_other = ds.DBZH.where(ds.mask_otherVHF & ~ds.mask_sparkles)
    
    wrad_sparkles = ds.WRADH.where(ds.mask_sparkles)
    wrad_other = ds.WRADH.where(ds.mask_otherVHF & ~ds.mask_sparkles)
    
    
    # Example: Scatter plot of dbzh vs wradh
    plt.figure(figsize=(6, 6), dpi=150)
    
    # Scatter for sparkles
    plt.scatter(dbzh_sparkles, wrad_sparkles,
                color=colors[0], label='Sparkles', alpha=0.6, edgecolor='k')
    
    # Scatter for other
    plt.scatter(dbzh_other, wrad_other,
                color=colors[1], label='Other', alpha=0.6, edgecolor='k')
    
    # Correlation line
    # plt.plot(np.unique(dbzh_spa), np.poly1d(np.polyfit(total_data_dbzh, total_data_wradh, 1))(np.unique(total_data_dbzh)), color='k', linestyle='--')
    
    plt.xlabel(f"{vardata['dbzh']['symbol']} [{vardata['dbzh']['units']}]")
    plt.ylabel(f"{vardata['wradh']['symbol']} [{vardata['wradh']['units']}]")
    plt.legend(loc='upper right')
    
    plt.title('Scatter Plot of DBZH vs WRADH')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% 2D histograms - preprocessing

def plot_2D_histograms(
        ds, 
        vardata,
        var_range,
        binsize,
        outname,
        outdir,
        save=False
        ):
    
    var_range['dbzh'][1]=60
    fontsize=8
    # Calculate the number of bins based on the range and binsize
    num_bins_dbzh = int((var_range['dbzh'][1] - var_range['dbzh'][0]) / binsize['dbzh'])
    num_bins_wradh = int((var_range['wradh'][1] - var_range['wradh'][0]) / binsize['wradh'])
    
    # Define bin edges for both variables
    bins_dbzh = np.linspace(var_range['dbzh'][0], var_range['dbzh'][1], num_bins_dbzh + 1)
    bins_wradh = np.linspace(var_range['wradh'][0], var_range['wradh'][1], num_bins_wradh + 1)
    
    labels_dbzh_bins = ["{:.0f} - {:.0f}".format(bins_dbzh[i], bins_dbzh[i+1]) for i in range(num_bins_dbzh)]
    labels_wradh_bins = ["{:.0f} - {:.0f}".format(bins_wradh[i], bins_wradh[i+1]) for i in range(num_bins_wradh)]
    
    dbzh_sparkles = ds.DBZH.where(ds.mask_sparkles)
    dbzh_other = ds.DBZH.where(ds.mask_otherVHF & ~ds.mask_sparkles)
    
    wrad_sparkles = ds.WRADH.where(ds.mask_sparkles)
    wrad_other = ds.WRADH.where(ds.mask_otherVHF & ~ds.mask_sparkles)
    
    # Compute 2D histograms
    hist_sparkles, _, _ = np.histogram2d(
        wrad_sparkles, dbzh_sparkles, bins=[bins_wradh, bins_dbzh])
    
    hist_other, _, _ = np.histogram2d(
        wrad_other, dbzh_other, bins=[bins_wradh, bins_dbzh])
    
    # Normalize the histograms to get probability densities
    hist_sparkles_norm = hist_sparkles / np.sum(hist_sparkles)
    hist_other_norm = hist_other / np.sum(hist_other)
    
    # Subtract the normalized histograms
    hist_diff = hist_sparkles_norm - hist_other_norm
    
    # Group the dataset by DBZH bins
    # count_sparkles_per_dbzhbin = dbzh_sparkles.groupby_bins('DBZH', bins_dbzh).count()
    # count_other_per_dbzhbin  = dbzh_other.groupby_bins('DBZH', bins_dbzh).count()
    hist_sparkles_norm_dbzhbin = hist_sparkles / np.sum(hist_sparkles, axis=0, keepdims=True)
    hist_other_norm_dbzhbin = hist_other / np.sum(hist_other, axis=0, keepdims=True)
    
    hist_diff_dbzhbin = hist_sparkles_norm_dbzhbin - hist_other_norm_dbzhbin
    
    
    def create_grid_with_gaps(data, axis=None, gap_size=0.1):
        n_rows, n_cols = data.shape
        dx = 1 / 2 * gap_size
        # Create the grid, with extra space between rows
        if axis == 'both':
            x0 = np.arange(n_cols + 1)
            x = [[x_i - dx, x_i + dx] for x_i in x0]
            x = sum(x, [])[1:-1]
    
            # Create the grid, with extra space between rows
            y0 = np.arange(n_rows + 1)
            y = [[y_i - dx, y_i + dx] for y_i in y0]
            y = sum(y, [])[1:-1]
    
            new_shape = (2 * n_rows - 1, 2 * n_cols - 1)
            data_with_nans = np.full(new_shape, np.nan)
            data_with_nans[::2, ::2] = data
    
            return x, y, data_with_nans
    
        elif axis == 1:
            x0 = np.arange(n_cols + 1)
            x = [[x_i - dx, x_i + dx] for x_i in x0]
            x = sum(x, [])[1:-1]
    
            new_shape = (n_rows, 2 * n_cols - 1)
            data_with_nans = np.full(new_shape, np.nan)
            data_with_nans[:, ::2] = data
    
            return x, data_with_nans
    
        elif axis == 0:
            y0 = np.arange(n_rows + 1)
            y = [[y_i - dx, y_i + dx] for y_i in y0]
            y = sum(y, [])[1:-1]
    
            new_shape = (2 * n_rows - 1, n_cols)
            data_with_nans = np.full(new_shape, np.nan)
            data_with_nans[::2, :] = data
    
            return y, data_with_nans
    
    
    x, hist_sparkles_norm_dbzhbin_gaps = create_grid_with_gaps(hist_sparkles_norm_dbzhbin, axis=1, gap_size=0.2)
    _, hist_other_norm_dbzhbin_gaps = create_grid_with_gaps(hist_other_norm_dbzhbin, axis=1, gap_size=0.2)
    _, hist_diff_dbzhbin_gaps = create_grid_with_gaps(hist_diff_dbzhbin, axis=1, gap_size=0.2)
    
    # Compute histograms normalized by WRADH bins
    hist_sparkles_norm_wradhbin = hist_sparkles / np.sum(hist_sparkles, axis=1, keepdims=True)
    hist_other_norm_wradhbin = hist_other / np.sum(hist_other, axis=1, keepdims=True)
    hist_diff_wradhbin = hist_sparkles_norm_wradhbin - hist_other_norm_wradhbin
    
    # Create grids with gaps for the new data
    y, hist_sparkles_norm_wradhbin_gaps = create_grid_with_gaps(hist_sparkles_norm_wradhbin, axis=0, gap_size=0.2)
    _, hist_other_norm_wradhbin_gaps = create_grid_with_gaps(hist_other_norm_wradhbin, axis=0, gap_size=0.2)
    _, hist_diff_wradhbin_gaps = create_grid_with_gaps(hist_diff_wradhbin, axis=0, gap_size=0.2)
    
    ## The real plotting:
    
    # Sum over the WRADH axis to get the total counts per DBZH bin
    counts_sparkles_per_dbzhbin = np.sum(hist_sparkles, axis=0)
    counts_other_per_dbzhbin = np.sum(hist_other, axis=0)
    
    # Sum over the DBZH axis to get the total counts per WRADH bin
    counts_sparkles_per_wradhbin = np.sum(hist_sparkles, axis=1)
    counts_other_per_wradhbin = np.sum(hist_other, axis=1)
    
    
    # Adjust figure size to accommodate three rows, 2 columns
    nrows = 3
    ncols = 2
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=400, 
        sharey=True, sharex="col")
    
    # First column
    # Panel a: Sparkles 2D histogram
    im1 = axs[0, 0].imshow(
        hist_sparkles, origin='lower', cmap='Reds',
        extent=[bins_dbzh[0], bins_dbzh[-1], bins_wradh[0], bins_wradh[-1]], 
        aspect='auto',
        )
    axs[0, 0].set_title('(a) Sparkles: 2D Histogram')
    # axs[0, 0].set_xlabel(f"{vardata['dbzh']['symbol']} [{vardata['dbzh']['units']}]")
    axs[0, 0].set_ylabel(f"{vardata['wradh']['symbol']} [{vardata['wradh']['units']}]")
    fig.colorbar(im1, ax=axs[0, 0], label='Counts')
    
    # Annotate the total counts per DBZH bin in Panel 1
    for i, count in enumerate(counts_sparkles_per_dbzhbin):
        axs[0, 0].text(
            bins_dbzh[i] + binsize['dbzh'] / 2, 
            bins_wradh[-2] - binsize['wradh']/2, 
            f'{int(count)}',
            ha='center', va='bottom', fontsize=fontsize, color='black',
            rotation=15,
            )
    
    axs[0, 0].text(
        bins_dbzh[0] + binsize['dbzh'] * 0.2, 
        bins_wradh[-5] + 0 * binsize['wradh'],
        r'$\downarrow$',
        ha='center', va='bottom', fontsize=25, color='black',
        )
    axs[0, 0].text(
        bins_dbzh[0] + binsize['dbzh'] * 0.8, 
        bins_wradh[-4] + 0 * binsize['wradh'],
        r'$\sum$',
        ha='center', va='bottom', fontsize=8, color='black',
        )
    
    # Panel c: Other 2D histogram
    im2 = axs[1, 0].imshow(
        hist_other, origin='lower', cmap='Blues',
        extent=[bins_dbzh[0], bins_dbzh[-1], bins_wradh[0], bins_wradh[-1]], 
        aspect='auto')
    axs[1,0].set_title('(c) Other VHF: 2D Histogram')
    # axs[1,0].set_xlabel(f"{vardata['dbzh']['symbol']} [{vardata['dbzh']['units']}]")
    axs[1,0].set_ylabel(f"{vardata['wradh']['symbol']} [{vardata['wradh']['units']}]")
    fig.colorbar(im2, ax=axs[1,0], label='Counts')
    
    # Annotate the total counts per DBZH bin in Panel 1
    for i, count in enumerate(counts_other_per_dbzhbin):
        axs[1,0].text(
            bins_dbzh[i] + binsize['dbzh'] / 2, 
            bins_wradh[-2] - binsize['wradh']/2, 
            f'{int(count)}',
            ha='center', 
            va='bottom', 
            fontsize=fontsize, 
            color='black',
            rotation=15,
            )
        
    axs[1,0].text(
        bins_dbzh[0] + binsize['dbzh'] * 0.2, 
        bins_wradh[-5] + 0 * binsize['wradh'],
        r'$\downarrow$',
        ha='center', va='bottom', fontsize=25, color='black')
    axs[1,0].text(
        bins_dbzh[0] + binsize['dbzh'] * 0.8, 
        bins_wradh[-4] + 0 * binsize['wradh'],
        r'$\sum$',
        ha='center', va='bottom', fontsize=8, color='black')
    
    # Panel e: Difference in normalized histograms
    im3 = axs[2, 0].imshow(
        hist_diff * 100, origin='lower', cmap='coolwarm',
        extent=[bins_dbzh[0], bins_dbzh[-1], bins_wradh[0], bins_wradh[-1]], 
        aspect='auto', vmin=-1.5, vmax=1.5,
        )
    axs[2,0].set_title('(e) Difference: Sparkles - Other VHF')
    axs[2,0].set_xlabel(f"{vardata['dbzh']['symbol']} [{vardata['dbzh']['units']}]")
    axs[2,0].set_ylabel(f"{vardata['wradh']['symbol']} [{vardata['wradh']['units']}]")
    cbar3 = fig.colorbar(
        im3, ax=axs[2,0], 
        label='Difference in Normalized Counts [%]', extend='both',
        )
    # cbar3.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    
    # Annotate the squares with the values from prob_matrix_other
    for i in range(num_bins_wradh):
        for j in range(num_bins_dbzh):
            value = hist_diff[i, j]
            if not np.isnan(value) and abs(value)>=0.001:  # Only annotate non-NaN values
                axs[2,0].text(
                    bins_dbzh[j] + binsize['dbzh'] / 2, 
                    bins_wradh[i] + binsize['wradh'] / 2, 
                    f'{value*100:.1f}', 
                    ha='center', 
                    va='center', 
                    color='black', 
                    size=fontsize,
                    )
    
    # Second column: Normalized histograms per DBZH bin
    # Panel b: Sparkles normalized within each DBZH bin
    im4 = axs[0, 1].pcolormesh(
        x, bins_wradh, 
        hist_sparkles_norm_dbzhbin_gaps * 100, 
        cmap='Reds',
        )
    axs[0,1].set_title('(b) Sparkles: normalized per $Z_h$ bin')
    # axs[0,1].set_xlabel(f"{vardata['dbzh']['symbol']} [{vardata['dbzh']['units']}]")
    # axs[0,1].set_ylabel(f"{vardata['wradh']['symbol']} [{vardata['wradh']['units']}]")
    cbar4 = fig.colorbar(im4, ax=axs[0,1], label='Normalized counts per $Z_h$ bin [%]')
    
    # Panel d: Other normalized within each DBZH bin
    im5 = axs[1, 1].pcolormesh(
        x, bins_wradh, 
        hist_other_norm_dbzhbin_gaps * 100, 
        cmap='Blues',
        )
    axs[1, 1].set_title('(d) Other VHF: normalized per $Z_h$ bin')
    # axs[1, 1].set_xlabel(f"{vardata['dbzh']['symbol']} [{vardata['dbzh']['units']}]")
    # axs[1, 1].set_ylabel(f"{vardata['wradh']['symbol']} [{vardata['wradh']['units']}]")
    cbar5 = fig.colorbar(im5, ax=axs[1, 1], label='Normalized counts per $Z_h$ bin [%]')
    
    # Panel f: Difference in normalized histograms per DBZH bin
    im6 = axs[2, 1].pcolormesh(
        x, bins_wradh, 
        hist_diff_dbzhbin_gaps * 100, 
        cmap='coolwarm', 
        vmin=-10, vmax=10,
        )
    axs[2,1].set_title('(f) Difference: Sparkles - Other VHF')
    axs[2,1].set_xlabel(f"{vardata['dbzh']['symbol']} [{vardata['dbzh']['units']}]")
    # axs[2,1].set_ylabel(f"{vardata['wradh']['symbol']} [{vardata['wradh']['units']}]")
    cbar6 = fig.colorbar(im6, ax=axs[2,1], label='Difference in normalized counts per $Z_h$ bin [%]', extend='both')
    
    # Annotate the squares with the values from prob_matrix_other
    for i in range(num_bins_wradh):
        for j in range(num_bins_dbzh):
            value = hist_diff_dbzhbin[i, j]
            if not np.isnan(value) and abs(value)>=0.01:  # Only annotate non-NaN values
                txt = axs[2,1].text(
                    j + 0.5, bins_wradh[i] + binsize['wradh'] / 2,
                    f'{value*100:.1f}', 
                    ha='center', 
                    va='center', 
                    color='black', 
                    size=fontsize,
                    )
    
    # Set tick labels for the DBZH axis
    # for ax in [axs[0, 1], axs[1, 1], axs[2,1]]:
    axs[2,1].set_xticks(np.arange(num_bins_dbzh) + 0.5)
    axs[2,1].set_xticklabels(
        [label for label in labels_dbzh_bins], 
        rotation=-55, ha='left', va="top")
    
    formatter = mticker.FormatStrFormatter('%.i')
    for cbar in [cbar4, cbar5, cbar6]:
        cbar.ax.yaxis.set_major_formatter(formatter)
    
    # Final adjustments
    plt.tight_layout()
    plt.show()

    if save==True:
        if outdir == None:
            raise(ValueError(outdir), "If save=True, needs a directry to save")
        outpath = outpath_gen(outdir, outdir, outname)
        fig.savefig(outpath, dpi=400)
        print("File is saved to: "+outpath+".png")