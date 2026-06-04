# LOFARsparkles-radar-researchcode

This repository contains the research code used to study polarimetric radar signatures near *sparkles* — intense, small-scale VHF radio emissions near the tops of deep convective storms, measured with the LOFAR radio telescope. The analysis compares dual-polarisation radar observations (reflectivity, radial velocity, spectrum width, hydrometeor classification) around sparkle sources versus other VHF lightning sources, to investigate the microphysical and dynamical conditions that produce them.

**Associated publication:**
> Reinaart van Loon et al., *"Graupel and increased turbulence observed near small-scale intermittent lightning discharges at the top of intense thunderstorms"*, [DOI to be added upon publication]

---

## Repository structure

### Source library (`src/`)

| Module                            | Purpose                                                                                           |
|---                                |---                                                                                                |
| `general.py`                      | Shared utilities; `WindowExtent` (spatial/temporal bounding box) and `ConfigPlot` (figure saving) |
| `read_LOFAR_data.py`              | Reads LOFAR VHF CSV files; distinguishes sparkles from other VHF sources                          |
| `read_RAD.py`                     | Loads radar volumes and composites; gridding, interpolation, masking, advection                   |
| `plot_LOFAR.py`                   | Multi-panel LOFAR VHF visualisations with sparkle clustering                                      |
| `plot_RAD_crosssect.py`           | Vertical radar cross-sections with LOFAR VHF overlay and temperature isotherms                    |
| `plot_RAD_multivar.py`            | Top-down multi-variable radar maps (Zh, vrad, W, HMC)                                             |
| `stats_sparklesRAD.py`            | Statistical comparison of radar properties near sparkles vs. other VHF sources                    |
| `meteo_analysis.py`               | Skew-T and hodograph plots from radiosonde or ERA5 data (MetPy)                                   |
| `altitude_statistics.py`          | Altitude-stratified radar statistics with bootstrap confidence intervals                          |
| `advection_comparison.py`         | Comparison of ERA5 vs. pySTEPS-derived advection in the statistics pipeline                       |
| `compute_pystep_motion_fields.py` | Lucas-Kanade optical-flow motion fields from KNMI radar composites (pySTEPS)                      |
| `sensitivity_DBSCAN.py`           | One-at-a-time sensitivity study for DBSCAN sparkle clustering parameters                          |
| `sensitivity_r_near_RAD.py`       | Sensitivity study for the radar masking radius around VHF sources                                 |

### Example scripts (`scripts/`)

| Script                            | Purpose                                                  |
|---                                |---                                                       |
| `LOFAR_plots.py`                  | Generate LOFAR VHF visualisations                        |
| `radar_cross_section.py`          | Generate vertical radar cross-sections                   |
| `radar_topviews_multi-var.py`     | Generate top-down multi-variable radar maps              |
| `skewT_plots.py`                  | Generate Skew-T / hodograph plots from soundings or ERA5 |
| `altitude_statistics.py`          | Run altitude-stratified radar statistics                 |
| `advection_comparison.py`         | Run ERA5 vs. pySTEPS advection comparison                |
| `compute_pystep_motion_fields.py` | Compute and save pySTEPS motion fields                   |
| `sensitivity_DBSCAN.py`           | Run DBSCAN parameter sensitivity study                   |
| `sensitivity_r-near-RAD.py`       | Run radar masking radius sensitivity study               |

---

## Installation

Requirements: **Python >= 3.8**. All dependencies are listed in `setup.py`.

```bash
# 1. Clone the repository
git clone https://github.com/reinaartvanloon/LOFARsparkles-radar-researchcode.git
cd LOFARsparkles-radar-researchcode

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the package and its dependencies
pip install -e .
```

This installs all required packages and adds `src/` to the Python path so the library modules can be imported directly.

---

## Required data

The datasets below are needed to run the code. Download them and set the corresponding paths in `article_figures.ipynb` before running.

| Dataset                                         | Source                                                                          | Notes                                                                                                         |
|---                                              |---                                                                              |---                                                                                                            |
| LOFAR VHF data (CSV)                            | [Zenodo — doi:10.5281/zenodo.17778996](https://doi.org/10.5281/zenodo.17778996) | Lightning VHF source locations detected by LOFAR                                                              |
| Borkum radar volumes (HDF5)                     | [Zenodo — doi:10.5281/zenodo.17778996](https://doi.org/10.5281/zenodo.17778996) | Dual-polarisation C-band radar data (Borkum Island, Germany)                                                  |
| KNMI NL25/NL61/NL62 composites (HDF5)           | [KNMI Data Platform](https://dataplatform.knmi.nl)                              | Dutch national radar network composites (1.5 km, 5-min)                                                       |
| ERA5 reanalysis (GRIB)                          | [Copernicus CDS](https://cds.climate.copernicus.eu)                             | Variables: geopotential, temperature, u/v wind; all pressure levels; June 18, 2021; domain: 2°–10°E, 51°–56°N |
| Hydrometeor classification membership functions | Bundled — `lib/msf_cband_v2.nc`                                                 | C-band fuzzy-logic model from [wradlib-data](https://github.com/wradlib/wradlib-data)                         |
| Country/province shapefiles (optional)          | [GADM](https://gadm.org)                                                        | Only required for map border overlays                                                                         |

---

## Reproducing the figures

Open `article_figures.ipynb` in a Jupyter environment and follow the instructions in the notebook. The notebook imports the library modules from `src/` and calls the operational scripts in `scripts/`. All file paths to the datasets listed above must be set in the first cells of the notebook.

```bash
jupyter notebook article_figures.ipynb
```

---

## Troubleshooting

- If package installation fails, try installing from `requirements.txt`: `pip install -r requirements.txt`
- Make sure that all file and directory paths to the downloaded datasets are set correctly in `article_figures.ipynb`
- For questions or issues, please open an issue on the [GitHub page](https://github.com/reinaartvanloon/LOFARsparkles-radar-researchcode)
