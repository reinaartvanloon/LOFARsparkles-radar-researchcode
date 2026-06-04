# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research code for studying polarimetric radar signatures near "sparkles" — lightning-induced VHF signals detected by LOFAR (Low-Frequency Array). The goal is reproducing figures for a scientific paper comparing radar data near sparkles vs. other VHF sources.

## Setup & Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The main entry point for figure reproduction is `article_figures.ipynb` (Jupyter notebook).

## Required Data (not in repo)

- **LOFAR data**: CSV files from Zenodo
- **Radar data**: Borkum Island HDF5 volume files + NL25/NL61/NL62 composites
- **ERA5 meteorological data**: GRIB files (geopotential, temperature, wind) for June 18, 2021
- **Hydrometeor classification**: `lib/msf_cband_v2.nc` (wradlib membership functions, already in repo)
- **Optional shapefiles**: GADM country/province boundaries for cartopy maps

## Code Architecture

**Data flow**: Read → Window/filter → Transform coordinates → Cluster VHF → Match radar → Plot

### Core modules (`src/`)

- **`general.py`**: Shared utilities. `WindowExtent` is the central configuration object — it defines the spatial bounding box, time window, and altitude range used by all readers and plotters. `ConfigPlot` controls figure saving.

- **`read_LOFAR_data.py`**: `DataLOFAR` class reads LOFAR VHF CSV files and applies `WindowExtent` filtering. Distinguishes "sparkles" from other VHF types via `data_key` metadata.

- **`read_RAD.py`**: `RADdata` class loads radar data (composite or volume) via `GetComposite()`, `GetVolume()`, `GetBorkumVolume2()`. Handles gridding, interpolation, and masking radar pixels near VHF sources. Supports: dbzh, vradh, wradh, HMC (hydrometeor classification via wradlib).

- **`plot_LOFAR.py`**: Multi-panel LOFAR visualizations. `SparkleParams` and `ClusteringParams` control sparkle identification; clustering groups VHF sources by distance/time/count thresholds.

- **`plot_RAD_crosssect.py`**: Vertical cross-sections through radar volumes with LOFAR VHF projections overlaid.

- **`plot_RAD_multivar.py`**: Top-down multi-variable radar maps (`draw_radar_image()`). Renders dbzh, vradh, wradh, and HMC panels with altitude contours and VHF overlays.

- **`stats_sparklesRAD.py`**: Statistical comparison — `statistics_sparkles_vs_other()` collects radar data around sparkle vs. non-sparkle VHF sources and computes distributions. Produces 1D and 2D histograms with statistical tests.

- **`meteo_analysis.py`** (root): Skew-T/Hodograph plots from sounding CSV or ERA5 GRIB data using MetPy.

### Operational scripts (`scripts/`)

Three example scripts showing typical workflows — use these as templates for new analyses:
- `LOFAR_plots.py` — LOFAR VHF visualization
- `radar_cross_section.py` — vertical radar slices
- `radar_topviews_multi-var.py` — multi-variable top-view radar

## Key Concepts

- **Sparkles**: Intense VHF radio emissions near strong radar reflectivity, caused by lightning-related processes. The VHF type field in LOFAR data distinguishes sparkles from regular VHF.
- **WindowExtent**: Nearly all functions accept a `WindowExtent` instance that gates which data is loaded/plotted. Changing the window is the primary way to switch between events or regions.
- **Coordinate systems**: Data lives in geographic (lat/lon) and projected (e.g., Dutch RD New / EPSG:28992) CRS. `general.py` handles conversions.
- **HMC**: Hydrometeor classification uses fuzzy-logic membership functions from wradlib (`lib/msf_cband_v2.nc`).
