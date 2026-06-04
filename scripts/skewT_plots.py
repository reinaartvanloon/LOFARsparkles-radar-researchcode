#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:46:03 2026

@author: reinaart
"""

import os, sys
here = os.path.abspath(os.path.dirname(__file__))
src = os.path.join(here, "src")
for p in (here, src):
    if p not in sys.path:
        sys.path.insert(0, p)

from meteo_analysis_test import plot_skewt, ConfigSounding, ConfigERA5, ConfigSkewT

#%%
datadir = "/home/reinaart/sparkles/meteo_data/"
# filename = "2021061812-soundingMeppen.csv"
# filename = "2021061812-Nordeney10113.csv"
# filename = "2021061821-Nordeney10113.csv"
filename = "2021061811-deBilt06260.csv"
config_sounding = ConfigSounding(
    filepath=datadir+filename
)

outdir = "/home/reinaart/sparkles/temp_figures/skewt_diagrams/loc_deBilt"
outname = "locDeBilt_11UTC"

config_skewt_sounding = ConfigSkewT(
    max_altitude_km=15,
    xlims_skewt=(-30, 30),
    outdir=outdir,
    outname=outname,
    show=True,
    save=True,
)

plot = plot_skewt(config_sounding, config_skewt_sounding)

#%%

# Meppen location
# lat=52.715
# lon=7.318
# time = "2021-06-18 19:00:00"
# outname = f"locMeppen_19UTC"

# Some other location
lat=53
lon=7
hh = 18
time=f"2021-06-18 {hh}:00:00"
outname = f"lon{lon}_lat{lat}_{hh}UTC"
outdir = f"/home/reinaart/sparkles/temp_figures/skewt_diagrams/lon{lon}_lat{lat}"


config_era5 = ConfigERA5(
    filepath="/home/reinaart/sparkles/meteo_data/ERA5_volume_20210618.grib",
    time=time,
    latitude=lat,
    longitude=lon,
)

config_skewt_era5 = ConfigSkewT(
    max_altitude_km=15,
    xlims_skewt=(-30, 30),
    outdir=outdir,
    outname=outname,
    show=True,
    save=True,
)

plot = plot_skewt(config_era5, config_skewt_era5)

#%%

import xarray as xr
import numpy as np
path = '/home/reinaart/sparkles/meteo_data/ERA5_volume_20210618.grib'
ds = xr.open_dataset(path, engine='cfgrib')
lat = ds['latitude'].values
lon = ds['longitude'].values
print(f'latitude min={np.nanmin(lat):.3f}, max={np.nanmax(lat):.3f}, n={lat.size}')
print(f'longitude min={np.nanmin(lon):.3f}, max={np.nanmax(lon):.3f}, n={lon.size}')
print(f'lat first={lat[0]:.3f}, last={lat[-1]:.3f}')
print(f'lon first={lon[0]:.3f}, last={lon[-1]:.3f}')
print()
