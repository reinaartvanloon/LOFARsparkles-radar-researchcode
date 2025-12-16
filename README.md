# LOFARsparkles-radar-researchcode
This directory provides the research code that was used to study polarimetric radar data near sparkles. The research is intended for publication. 

Everyone is encouraged to produce their own figures, making use of my code. Use the article_figures.ipynb file in an iPython environment to produce all figures.

# Installation
The following steps are necessary to reproduce the 
- Download this github repository.
- Create a python virtual environment and activate it. In the command line: python3 -m venv .venv && .venv/bin/activate
- Navigate to the directory you dowloaded and run on the command line: pip install -e .
This:
	- the necessary python version and modules from setup.py
	- adds the "src" directory to the PYTHONPATH
- Download the required datasets (see next section)
- In a iPython notebook environment, open the article_figures.ipynb and follow the instructions in the file to make the figures of interest.

# Downloading the data
The data that is necessary to run the python code and reproduce figures are the following
- LOFAR data. Download from https://doi.org/10.5281/zenodo.17778996
- Radar data (Borkum Island, Germany). Download from https://doi.org/10.5281/zenodo.17778996 . This directory includes two files with respectively, metadata about different radars, and metadata about plotting of different radar variables.
- ERA5 volume data. Download from the Copernicus' Climate Data Store.
  The following variables and ranges are required
	- Variables: Geopotential, Temperature, meridional wind speed (v), zonal wind speed (u)
	- All pressure levels (all vertical levels)
	- June 18, 2021
	- Horizontal extent of minimal 2$^\circ$  to 10$^\circ$ latitude, and 51$^\circ$ to 56$^\circ$. 
- Wradlib's membership functions for hydrometeor classification. In the github repository as "lib/hmc_msf_cband_v2.nc". Can also be found on the Wradlib data page: https://github.com/wradlib/wradlib-data/tree/main
- Optional: Shapefiles with the borders of countries or provinces. This is necessary for plotting of borders in geospatial plots. Dowload on www.gadm.org

# Troubleshooting
- Consider using the requirements.txt file to install python packages
- Make shure that in the article_figures.ipynb file, you fill out the right file and directory paths to the data.
- Leave a comment in the github page: https://github.com/reinaartvanloon/LOFARsparkles-radar-researchcode
