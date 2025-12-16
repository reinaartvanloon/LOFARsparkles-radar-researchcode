#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:53:50 2024

@author: rvloon
"""


import numpy as np
import xarray as xr
import os
import datetime
from general import round_time, trim_2D_unstructured_data, WindowExtent, open_reference_file, split_dim_to_variables
import logging
import glob
import wradlib
from pyproj import Transformer
from scipy.interpolate import griddata
from pandas import read_csv
import io
import h5py
import json
import wradlib as wrl
import gc
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any

from plot_LOFAR import get_data_LOFAR, cluster_LOFARsparkles, filter_mask_LOFAR, get_datetime_LOFAR_from_metadata, ConfigLOFAR, update_with_LOFAR_info
from read_LOFAR_data import surrounding_data

logger = logging.getLogger(__name__)


class RADdata:
    def __init__(self, data_dirpath, stormcode, station, varkeys):
        self.stormcode = stormcode
        self.data_dirpath = data_dirpath
        self.station = station
        self.varkeys = varkeys

        if station == 'NL25':
            self.datatype = self.varkeys
        else:
            self.datatype = "VOL"

        return

    def get_RAD_data(self, dt_target, which_image='nearest', **kwargs):
        round_to = {"before": "down",
                    "after": "up",
                    "nearest": "average"}

        self.dt, self.delta_t = round_time(
            dt_target, datetime.timedelta(minutes=5), to=round_to[which_image])

        # Safely extract values from kwargs
        window_extent = kwargs.get('window_extent', None)
        grid = kwargs.get('grid', None)  # Default to None if not provided
        dt_start = kwargs.get('dt_start')
        dt_end = kwargs.get('dt_end')

        if self.station == "NL25":
            self.GetComposite(window_extent, grid)
        elif self.station == "NL61" or self.station == "NL62":
            self.GetVolume(window_extent)
        elif self.station == "asb":
            self.GetBorkumVolume2(dt_start, dt_end)

    def GetComposite(self, window_extent, grid=None):
        self.fid = '*' + self.station + "_" + self.datatype + \
            "*" + self.dt.strftime('%Y%m%d%H%M') + '*'  # File ID
        filepath = glob.glob(os.path.join(
            self.data_dirpath, "**", self.fid))  # Path to file
        # Logging the file being processed
        logger.info("Read file: {}".format(filepath[0].split("/")[-1]))
        self.h5 = wradlib.io.read_opera_hdf5(filepath[0])  # open dataset

        lon_mesh, lat_mesh = NL25_grid()  # Grid corresponding to NL25 composite data

        if window_extent:
            values, lon_mesh, lat_mesh = trim_2D_unstructured_data(data_calibration(
                self.h5['image1/image_data'], self.varkey), lon_mesh, lat_mesh, window_extent)

        if grid:  # Interpolate to structured grid
            logger.info(
                "Interpolate satellite data to {}".format(grid.gridname))
            values = griddata(np.column_stack((lon_mesh, lat_mesh)),
                              values, (grid.xmesh, grid.ymesh), method='linear')

            self.ds = xr.Dataset(
                {f"{self.varkey}": (("latitude", "longitude"), values)
                 },
                coords={
                    "longitude": grid.x,
                    "latitude": grid.y
                }
            )

        else:  # Save unstructured grid
            self.ds = xr.Dataset({'lon_mesh': (('x_points', ' y_points'), lon_mesh),
                                  'lat_mesh': (('x_points', ' y_points'), lat_mesh)})
            self.ds[self.varkey] = xr.DataArray(
                values, dims=('x_points', ' y_points'))

        logger.info("Retrieved {} variable".format(self.varkey))

    # def __getattr__(self, name):
    #     """Delegate attribute access to the data_object."""
    #     try:
    #         return getattr(self.ds, name)
    #     except AttributeError:
    #         raise AttributeError(f"'Radar_data' object has no attribute '{name}'")

    def GetBorkumVolume2(self, dt_start, dt_end):
        # varkey logic to select only appropriate varkeys
        Borkum_varkeys = ["dbzh", "dbzv", "zdr", "rhohv",
                          "uphidp", "vradh", "wradh", "vradv", "wradv"]
        uphidp_derivatives = ["UPHIDP_1", "UPHIDP_unfolded", "phidp",
                              "KDP_1", "KDP_unfolded", "KDP_filtered", "kdp", "deltaHV"]

        # If there is a varkey that is a uphidp derivative
        if any([item in uphidp_derivatives for item in self.varkeys]):
            self.varkeys = self.varkeys + ['dbzh', 'dbzv', 'rhohv', 'uphidp']

        self.varkeys = list(
            set([key for key in self.varkeys if key in Borkum_varkeys]))

        logger.info("Open radar variables: ", *[key for key in self.varkeys])
        radar_volumes = []
        for i, varkey in enumerate(self.varkeys):
            logger.info(f"Open radar variable: {varkey}")
            if varkey is None:
                continue
            # directory with borkum files
            datadir = glob.glob(
                os.path.join(
                    self.data_dirpath, "**", f"vol*_{varkey}*"
                    ),
                recursive=True
                )
            try:
                # All scan files within datadir
                radar_files = os.listdir(datadir[0])
            except IndexError:
                logger.info(f"Could not find {varkey} variable")
            # Algorithm to select the files wihtin the timerange
            selected_files = []
            sweeps = []
            for file in radar_files:
                file_dt = datetime.datetime.strptime(
                    file.split("-")[2], '%Y%m%d%H%M%S%f')
                if dt_start <= file_dt < dt_end:
                    selected_files.append(os.path.join(datadir[0], file))
                    filepath = os.path.join(datadir[0], file)
                    with xr.open_dataset(filepath, engine="odim") as sweep:
                        # round the azimuths to avoid slight differences
                        sweep.coords["azimuth"] = (
                            sweep.coords["azimuth"].round(1))
                        sweep = sweep.drop_duplicates("azimuth")
                        sweep = sweep.set_coords('sweep_fixed_angle')
                        sweeps.append(sweep)

            radar_vol = xr.concat(
                sweeps, 
                dim="sweep_fixed_angle",
                join="outer", 
                coords="different", 
                compat="equals",
                ).sortby("sweep_fixed_angle")

            # # Reduce coordinates so the georeferencing works
            radar_vol["elevation"] = radar_vol["elevation"].median("azimuth")
            radar_vol["sweep_mode"] = radar_vol["sweep_mode"].min()

            radar_volumes.append(radar_vol)
        if len(radar_volumes) > 1:
            radar_vol = xr.merge(radar_volumes, compat='override')
        elif len(radar_volumes) == 1:
            radar_vol = radar_volumes[0]
        else:
            # Handle the case where no radar volumes are found
            radar_vol = None
            logger.warning("No radar volumes found for the specified time range and variables.")
            return None  # Or handle this gracefully as needed

        self.ds = radar_vol

        return radar_vol

    def CrossSection(self, p1, p2, variables, resolution=None, method="nearest"):
        # from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
        from scipy.spatial import KDTree

        # Check if dataset is georeferenced by checking for x,y,z, coordinates
        if not all(map(lambda v: v in self.ds.coords, ['x', 'y', 'z'])):
            raise AttributeError(
                "The dataset needs to be georeferenced (x, y, and, z) in metres. E.g. with wradlib.georef")

        # Make list of variables that need to be interpolated
        varlist = ['z', 'time']
        for var in variables:
            if var not in self.ds.variables:
                raise AttributeError(f"Variable {var} is not in the dataset.")
            else:
                varlist.append(var)

        self.ds['time'] = self.ds.time.broadcast_like(self.ds.z)

        # Determine the horizontal resolution of the interpolation
        if resolution is None:
            range_resolution = np.unique(
                np.diff(self.ds.range.values).round(1))
            if len(range_resolution) != 1:
                raise ValueError(
                    "Bin ranges are not evenly spaced. Please set the resolution variable")
            else:
                resolution = 1 / 2 * range_resolution[0]

        p1, p2 = np.array(p1), np.array(p2)

        # Check that the two points given are not the same
        try:
            if (p1 == p2).all():
                raise ValueError(
                    "p1=p2. The two points given are the same. Please give different points."
                )
        except AttributeError as err:
            if p1 == p2:
                raise ValueError(
                    "p1=p2. The two points given are the same. Please give different points."
                ) from err

        if "HMC" in varlist:
            HMC = True
            self.ds, varlist = unpack_hmc(self.ds, varlist)
        else:
            HMC = False

        # Make array of target points --> locations to interpolate to
        xy_total = np.linalg.norm(p2 - p1)
        nr_points = int(np.ceil(xy_total / resolution))
        target_points = np.linspace(p1, p2, nr_points)
        xy = np.linspace(0, xy_total, nr_points)

        # #select points close to line (2 km?)
        # D =  2000 #Distance in m
        # nr_dims = len(self.ds.x.dims)
        # points = np.stack((self.ds.x.values, self.ds.y.values), axis=nr_dims+1)
        # distance = np.abs(np.cross(p2-p1, points-p1))/xy_total

        # self.ds = self.ds.assign_coords(distance_to_crosssect=(self.ds.x.dims,
        #                                                        distance))
        interpolated_values = {var: [] for var in varlist}

        for elev in self.ds.sweep_fixed_angle.values:  # iterate over different sweep_angles
            sweep = self.ds.sel(sweep_fixed_angle=elev)

            # mask = sweep.distance_to_crosssect<D
            x = sweep.x.values.ravel()
            y = sweep.y.values.ravel()

            points = np.stack((x, y), axis=1)

            tree = KDTree(points)

            if method == 'nearest':
                dd, ii = tree.query(target_points)
                for var in varlist:
                    interpolated_values[var].append(
                        sweep[var].values.ravel()[ii]
                    )

        data_arrays = {}
        for i, var in enumerate(varlist):
            var_values = np.array(interpolated_values[var])
            
            da = xr.DataArray(
                var_values,
                coords={
                    "sweep_fixed_angle": ("sweep_fixed_angle", self.ds.sweep_fixed_angle.values),
                    "xy": ("xy", xy)
                    },
                dims=["sweep_fixed_angle", "xy"],
                )
            
            data_arrays[var] = da
        ds = xr.Dataset(data_arrays)

        if HMC:
            ds, varlist = repack_hmc(ds, self.ds.hmc.values, varlist)

        return ds

    def get_virtices(self, vert_beamwidth):
        ranges = self.ds.range.values.round(1)          # ranges of radar bins
        azimuths = self.ds.azimuth.values.round(1)  # Azimuths of radar beams
        elevs = self.ds.sweep_fixed_angle.values.round(
            1)  # Sweep elevation angles

        radar_site = (self.ds.longitude,
                      self.ds.latitude,
                      self.ds.altitude)

        # Range resolution
        range_res = np.unique(np.diff(ranges))
        if len(range_res) > 1:
            raise ValueError(
                "Bin ranges are not evenly spaced. Please resample for even spacing.")

        # Azimuth resolution
        azi_res = range_res = np.unique(np.diff(azimuths))
        if len(azi_res) > 1:
            raise ValueError(
                "Azimuths are not evenly spaced. Please resample for even spacing.")

        # Construct virtices in spherical coordinates
        r_virt = np.arange(ranges[0] - 1 / 2 * range_res,
                           ranges[-1] + range_res,
                           range_res)
        azi_virt = np.arange(azimuths[0] - 1 / 2 * azi_res,
                             azimuths[-1] + azi_res,
                             azi_res)

        elev_virt = []
        for elev in elevs:
            elev_virt.append(elev - 1 / 2 * vert_beamwidth)
            elev_virt.append(elev + 1 / 2 * vert_beamwidth)
        elev_virt = np.array(elev_virt)

        # Create meshgrid before georeferencing
        r_mesh, azi_mesh, elev_mesh = np.meshgrid(
            r_virt,
            azi_virt,
            elev_virt)

        coords, aeqd = wradlib.georef.polar.spherical_to_xyz(
            r_mesh, azi_mesh, elev_mesh, radar_site, squeeze=True, strict_dims=True)

    def advect(self, dt_target, ds_ref):
        '''
        Advect the data coordinates (x and y) of self.ds
        using the horizontal velocities of ds_ref
        given a certain time to advect to (target time).

        Parameters
        ----------
        dt_target : string
            The time, to which the "self" dataset is advected to
        ds_ref : xarray.Dataset
            The dataset with the horizontal velocities "u" and "v"

        Returns
        -------
        self.ds
            With the spatial coordinates, matching dt_target

        '''

        time_da, _ = xr.broadcast(self.ds.time, self.ds.longitudes)
        u_interp = ds_ref.u.interp(
            time=self.ds.time, longitude=self.ds.longitudes, latitude=self.ds.latitudes, H=self.ds.z)
        v_interp = ds_ref.v.interp(
            time=self.ds.time, longitude=self.ds.longitudes, latitude=self.ds.latitudes, H=self.ds.z)

        dt_target = np.array(dt_target).astype(
            dtype='datetime64[ns]')  # Convert to numpy datetime
        delta_dt = (dt_target - time_da.values) / np.timedelta64(1, 's')
        self.ds['dx_adv'] = delta_dt * u_interp
        self.ds['dy_adv'] = delta_dt * v_interp

        # x,y,z-values outside ds_ref give nan for u_interp and v_interp
        # Make nanmask for advection
        mask = ~np.isnan(u_interp.values)

        # Using the mask to update values in x and y DataArrays
        self.ds['x'] = xr.where(mask, self.ds['x'] + self.ds['dx_adv'], self.ds['x'])
        self.ds['y'] = xr.where(mask, self.ds['y'] + self.ds['dy_adv'], self.ds['y'])

        # If virtices of grids are given. Advect as well.
        if "virtices_x" in self.ds.variables and "virtices_y" in self.ds.variables:
            self.ds["virtices_x_adv"] = self.ds['virtices_x'] + self.ds['dx_adv']
            self.ds["virtices_y_adv"] = self.ds['virtices_y'] + self.ds['dx_adv']

        return self.ds

    def lonlat_georeference(self):
        '''
        Return itself, but with longitude and latitude coordinates
        matching the x- and y- coordinates
        '''
        
        radar_location = {'lon': self.ds.longitude.values,
                          'lat': self.ds.latitude.values}
        crs = f"+proj=aeqd +lat_0={radar_location['lat']} +lon_0={radar_location['lon']} +x_0=0 +y_0=0 +datum=WGS84"
        transform2lonlat = Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True)

        lons, lats = transform2lonlat.transform(self.ds.x, self.ds.y)
        self.ds['longitudes'] = xr.DataArray(
            lons, dims=self.ds.x.dims, coords=self.ds.x.coords)
        self.ds['latitudes'] = xr.DataArray(
            lats, dims=self.ds.y.dims, coords=self.ds.x.coords)
        return self.ds

    def temp_reference(self, ds_ref):
        '''
        Interpolate the data of ds_ref towards the time, 
        longitude and latitude of the self.ds

        Parameters
        ----------
        ds_ref : xarray.Dataset
            Dataset with the "t" (temperature) variable in Kelvin
            with coordinates time, longitude, latitude, and H (height).

        Returns
        -------
        self
            With TEMP values on the coordinates. 

        '''
        self.ds['TEMP'] = ds_ref.t.interp(
            time=self.ds.time, 
            longitude=self.ds.longitudes, 
            latitude=self.ds.latitudes, 
            H=self.ds.z
            ) - 273.15
        return self.ds

    def hmc(self, hm_classification_filepath):
        with xr.open_dataset(hm_classification_filepath, engine='h5netcdf') as msf:

            # Initialize an empty list to collect classification results
            cl_res_list = []
            
            #Need to not get duplication error in wrl.classify.msf_index_indep
            if "obs" in msf.dims:
                msf = split_dim_to_variables(msf, 'obs')
            
            # Need to do this separately per elevation angle
            for i, elev in enumerate(self.ds.sweep_fixed_angle.values):
                mask_nonsense = (self.ds.DBZH.sel(sweep_fixed_angle=elev) < 0).values | (
                    self.ds.DBZV.sel(sweep_fixed_angle=elev) < 0).values | (
                        self.ds.DBZH.sel(sweep_fixed_angle=elev).isnull().values)
                
                DBZH = self.ds.DBZH.isel(sweep_fixed_angle=i)
                # without renaming, there is an error for dimension "obs" duplicate
                # print(msf)
                
                msf_val = msf.wrl.classify.msf_index_indep(DBZH)
                
                # Define the observation mapping
                obs_mapping = dict(ZH="DBZH", ZDR="ZDR",
                                   RHO="RHOHV", KDP="KDP", T="TEMP")

                # Call the fuzzyfi function on the msf object
                ds = self.ds.isel(sweep_fixed_angle=i)
            
                fu = msf_val.wrl.classify.fuzzyfi(hmc_ds=ds, msf_obs_mapping=obs_mapping)

                w = xr.Dataset(dict(ZH=1.0, ZDR=1.0, RHO=1.0, KDP=1.0, T=1.0))
                prob = fu.wrl.classify.probability(w).compute()
                cl_res = prob.wrl.classify.classify(threshold=0.0)
                cl_res = cl_res.compute()

                cl_res.loc[{'hmc': 'NP'}] = xr.where(
                    mask_nonsense, 1, cl_res.loc[{'hmc': 'NP'}])
                cl_res_list.append(cl_res)
            cl_res_combined = xr.concat(
                cl_res_list, 
                dim='sweep_fixed_angle',
                join="outer", 
                coords="different", 
                compat="equals",
                )
            del cl_res_list
            gc.collect()

        # Add the combined classification results to the original dataset
        self.ds['HMC'] = cl_res_combined

        return self.ds

    def uphidp_processing(self, varlist, winlen=11, th1=-5, th2=20, th3=-40, ndespeckle=5, niter=2):
        import wradlib as wrl
        from scipy import integrate
        dims_phidp = self.ds['UPHIDP'].dims

        # self.ds['PHIDP'], self.ds['KDP'] = self.ds['UPHIDP'].wrl.dp.phidp_kdp_vulpiani(winlen=winlen,th1=th1, th2=th2, th3=th3, copy=True, ndescpeckle=ndespeckle)
        # Ignore phase shift for DBZH or DBZV below 0
        mask = (self.ds['DBZH'] < 0) | (self.ds['DBZV'] < 0)
        uphidp = self.ds['UPHIDP'].where(~mask).copy()
        uphidp = uphidp.ffill(dim='range')

        uphidp_despeckled = uphidp.wrl.util.despeckle(n=ndespeckle, copy=True)
        kdp1 = uphidp_despeckled.wrl.dp.kdp_from_phidp(
            winlen=winlen, copy=True, skipna=False)
        # self.ds["UPHIDP_unfolded"] = (self.ds['UPHIDP'].dims,
        #                                 wrl.dp.unfold_phi(self.ds['UPHIDP_despeckled'].values, self.ds['RHOHV'].values, width=5, copy=True))

        # self.ds["UPHIDP_unfolded"] = (self.ds['UPHIDP'].dims,
        #                                wrl.dp.unfold_phi_vulpiani(self.ds['UPHIDP_despeckled'].values, self.ds['KDP_1'].values, th=th3, winlen=winlen, copy=True))

        uphidp_unfolded = unfold_phi(uphidp_despeckled.values,
                                     kdp1.values,
                                     self.ds['RHOHV'].values,
                                     w=winlen,
                                     th_std=0.8,
                                     th_kdp=(th1, th2),
                                     th_phi=th3,
                                     copy=True)

        # self.ds['UPHIDP_clean'] =  self.ds["UPHIDP_unfolded"].where(self.ds["UPHIDP_unfolded"] <360).copy()

        # use given (fast) derivation methods
        kdp2 = wrl.dp.kdp_from_phidp(
            uphidp_unfolded, winlen=winlen, dr=0.25, copy=True)

        # find kdp values with no physical meaning like noise, backscatter differential
        # phase, nonuniform beamfilling or residual artifacts using th1 and th2
        mask = (kdp2 <= th1) | (kdp2 >= th2)
        kdp_filtered = np.where(~mask, kdp2, 0)
        # fill remaining NaN with zeros
        kdp_filtered = np.where(~np.isnan(kdp_filtered), kdp_filtered, 0)

        self.ds['KDP'] = (self.ds['UPHIDP'].dims, kdp_filtered)

        for _i in range(niter):
            # phidp from kdp through integration
            self.ds['PHIDP'] = (self.ds['UPHIDP'].dims,
                                2 * integrate.cumulative_trapezoid(self.ds['KDP'], dx=0.25, initial=0.0, axis=-1))
            # kdp from phidp by convolution
            self.ds['KDP'] = self.ds['PHIDP'].wrl.dp.kdp_from_phidp(
                winlen=winlen)

        # Save intermediate variables as DataArrays for later use
        da_dict = {
            "deltaHV": (dims_phidp, kdp2 - self.ds['KDP'].values),
            "UPHIDP_1": uphidp_despeckled,
            "KDP_1": kdp1,
            "UPHIDP_unfolded": (dims_phidp, uphidp_unfolded),
            "KDP_unfolded": (dims_phidp, kdp2),
            "KDP_filtered": (dims_phidp, kdp_filtered)
        }

        # Add variables to radar ds if in varlist
        for key, da in da_dict.items():
            if key in varlist:
                self.ds[key] = da

    def removeLowDBZ(self, varlist, vardata):
        for var in varlist:
            varlib = vardata[var]
            if 'dbz_filter' in varlib:
                # Low dbz values is probably rubbish
                dbz_threshold = varlib['dbz_filter']
                try:
                    dbzh_mask = self.ds.DBZH >= dbz_threshold
                    dbzv_mask = self.ds.DBZV >= dbz_threshold
                except AttributeError:
                    if 'DBZH' in self.ds.keys():
                        dbzh_mask = self.ds.DBZH >= dbz_threshold
                        dbzv_mask = np.ones(dbzh_mask.shape) == 1
                    elif 'DBZV' in self.ds.keys():
                        dbzv_mask = self.ds.DBZV >= dbz_threshold
                        dbzh_mask = np.ones(dbzv_mask.shape) == 1
                    else:
                        raise AttributeError(
                            "No dbzh, or dbzv data loaded, so cannot filter upon these variables.")
                dbz_mask = dbzh_mask & dbzv_mask
                self.ds[vardata[var]['ODIM']] = self.ds[vardata[var]
                                                        ["ODIM"]].where(dbz_mask)


@dataclass
class ConfigDataRAD:
    # Variables for main function
    data_dirpath: str  
    stormcode: str
    RADstation: str
    RADvars: list  
    # station_metadata_filepath: str
    # metadata_vars_filepath: str 
    advection_reference_filepath: str
    temp_reference_filepath: str
    hmc_msf_filepath: str
    
    VHFtype: str = "sparkles&otherVHF"

    dt_target: Optional[str] = None
    dt_range: Optional[List] = None
    window_extent: Optional[WindowExtent] = None
    epsg: int = 4326


@dataclass
class DataRADandLOFAR:
    station: str
    varkeys: List[str]
    RAD: Any  # Replace `Any` with the actual type of RADdata
    vardata: Dict[str, Any]
    window_extent: WindowExtent  # Replace `Any` with the actual type
    crs: str
    LOFAR: Optional[Any] = None  # Replace `Any` with the actual type
    
def get_metadata_RADstation(data_dirpath: str, station_key: str) -> Tuple[dict, dict]:
    """Load metadata from station and variable JSON files."""
    with open(os.path.join(data_dirpath,"stations_metadata.json")) as file:
        station_data = json.load(file)
    station = station_data[station_key]
    return station

def get_metadata_RADvars(data_dirpath: str) -> Tuple[dict, dict]:
    """Load metadata from station and variable JSON files."""
    with open(os.path.join(data_dirpath,"variables_metadata.json")) as file:
        variable_data = json.load(file)
    return variable_data


def convert_dt_str_to_datetime(dt_target: str, dt_range: Optional[list]=None, delta_seconds: int = 300) -> Tuple[datetime.datetime, datetime.datetime, datetime.datetime]:
    """Determine the target, start, and end times."""
    dt_delta = datetime.timedelta(seconds=delta_seconds)
    dt_target = datetime.datetime.strptime(dt_target, '%Y-%m-%dT%H:%M:%S')
    
    if dt_range is None:
        try:
            dt_start = datetime.datetime.strptime(dt_range[0], '%Y-%m-%dT%H:%M:%S')
            dt_end = datetime.datetime.strptime(dt_range[1], '%Y-%m-%dT%H:%M:%S')
        except (TypeError, ValueError):
            dt_start = dt_target - 0.5 * dt_delta
            dt_end = dt_target + 0.5 * dt_delta
        dt_range = [dt_start, dt_end]
        
    return dt_target, dt_range


def load_radar_data(
        config: ConfigDataRAD, 
        variable_data: dict,
        dt_target,
        dt_range,
        ) -> RADdata:
    """Load and preprocess radar data."""
    
    # Determine variables to load
    if "hmc" in config.RADvars:
        varlist = list(set(config.RADvars + ["dbzh", "dbzv", "zdr", "rhohv", "uphidp"]))
    else:
        varlist = config.RADvars

    radar_data = RADdata(
        config.data_dirpath, 
        config.stormcode, 
        config.RADstation, 
        varkeys=varlist
        )
    radar_data.get_RAD_data(
        dt_target=dt_target, 
        dt_start=dt_range[0], 
        dt_end=dt_range[1]
        )

    if 'UPHIDP' in radar_data.ds.variables:
        radar_data.uphidp_processing(config.RADvars)

    # Remove low DBZH values for some variables
    radar_data.removeLowDBZ(varlist, variable_data)
    
    # Make a cartesian grid of the range, azimuth, elevation angle, given radar location
    radar_data.ds = wrl.georef.georeference(radar_data.ds)
    
    # Need longitude lattitude coordinates to match with ERA5 grid
    radar_data.ds = radar_data.lonlat_georeference()

    return radar_data


def setup_crs_and_window(
        window_extent: WindowExtent, 
        radar_data: RADdata
        ):
    """Setup CRS and define the plotting window."""
    crs = (
        f"+proj=aeqd +lat_0={radar_data.ds.latitude.values} "
        f"+lon_0={radar_data.ds.longitude.values} +x_0=0 +y_0=0 +datum=WGS84"
    )
    transformer_to_lonlat = Transformer.from_crs(f"{crs}", "EPSG:4326", always_xy=True)

    if window_extent.x[0] and window_extent.y[0]:
        window_extent.x[0], window_extent.y[0] = transformer_to_lonlat.transform(
            window_extent.x_range[0], 
            window_extent.y_range[0], 
            direction="INVERSE"
            )
    if window_extent.x[1] and window_extent.y[1]:
        window_extent.x[1], window_extent.y[1] = transformer_to_lonlat.transform(
            window_extent.x_range[1], 
            window_extent.y_range[1], 
            direction="INVERSE"
            )
        
    return crs, window_extent, transformer_to_lonlat


def process_LOFAR_data( 
        config_LOFAR : ConfigLOFAR,
        # transformer_lonlat_to_crs,
        ) -> ConfigDataRAD:
    """Retrieve and process LOFAR data if applicable."""
    
    config_LOFAR, LOFAR_data = get_data_LOFAR(config_LOFAR)
    config_LOFAR = update_with_LOFAR_info(config_LOFAR)
    if hasattr(config_LOFAR, "sparkle_params"):
        print("Clustering LOFAR sources to get sparkles")
        # We want to find the LOFAR VHF that correspond to Sparkles
        _, mask_sparkles = cluster_LOFARsparkles(
            config = config_LOFAR, 
            data_LOFAR = LOFAR_data, 
            sparkle_params = config_LOFAR.sparkle_params,
            )
        # Impose filters found in the sparkle_params
        mask_sparkles = filter_mask_LOFAR(
            mask_sparkles,
            data_LOFAR = LOFAR_data,
            sparkle_params = config_LOFAR.sparkle_params,
            crs_data = config_LOFAR.crs
            )
        LOFAR_data.sparklemask = mask_sparkles
    
    return config_LOFAR, LOFAR_data


def get_data_RADandLOFAR(
        config: ConfigDataRAD,
        config_LOFAR: ConfigLOFAR = None,
        ) -> DataRADandLOFAR:
    """Main function to collect radar and LOFAR data based on configuration."""
    config.station = get_metadata_RADstation(
        config.data_dirpath, config.RADstation)
    variable_data = get_metadata_RADvars(
        config.data_dirpath)
    
    # Either choose radar data based on dt_target, or on the config_LOFAR --> LOFAR data
    if config.dt_target is not None:
        dt_target, dt_range = convert_dt_str_to_datetime(
            config.dt_target, 
            config.dt_range,
            )
    elif type(config_LOFAR) == ConfigLOFAR:
        dt_target, dt_range = get_datetime_LOFAR_from_metadata(config_LOFAR)
    else:
        raise(TypeError("Either need to give dt_target or define the ConfigLOFAR properly"))
        
    radar_data = load_radar_data(
        config, 
        variable_data, 
        dt_target, 
        dt_range,
        )
    
    if any(
            key in config.RADvars for key in ["hmc", "TEMP"]
        ) or (
            config.advection_reference_filepath
        ):
        with open_reference_file(config.temp_reference_filepath) as ds_ref:
            if "hmc" in config.RADvars:
                radar_data.ds = radar_data.temp_reference(ds_ref)
                radar_data.ds = radar_data.hmc(config.hmc_msf_filepath)

            if config.advection_reference_filepath:
                ds_ref = open_reference_file(config.advection_reference_filepath)
                radar_data.ds = radar_data.advect(dt_target, ds_ref)

            if "TEMP" in config.RADvars:
                radar_data.ds = radar_data.temp_reference(ds_ref)
    
    # We want the window extent in the crs of the radar data
    crs = (
        f"+proj=aeqd +lat_0={radar_data.ds.latitude.values} "
        f"+lon_0={radar_data.ds.longitude.values} +x_0=0 +y_0=0 +datum=WGS84"
        )
    
    if type(config.window_extent) == WindowExtent:
        config.window_extent = config.window_extent.transform_crs(crs_target=crs)
    
    # Create and return a data result object
    result = DataRADandLOFAR(
        station=config.station,
        varkeys=[variable_data[var]['ODIM'] for var in config.RADvars],
        RAD=radar_data,
        vardata=variable_data,
        window_extent=config.window_extent,
        crs=crs,
        )

    # If config_LOFAR is provided
    if config_LOFAR is not None:
        # Want to have the LOFAR data in same crs as radar data
        config_LOFAR.crs = crs
        _config_LOFAR, data_LOFAR = process_LOFAR_data(
            config_LOFAR = config_LOFAR,
            # transformer_lonlat_to_crs = config.window_extent.transformer_lonlat_to_crs,
            )
        result.LOFAR=data_LOFAR

    del radar_data
    gc.collect()
    
    return result


def get_mask_RADnearVHF(
        raddata_ds, 
        VHF_df, 
        radius, 
        alt_th, 
        dbzh_th, 
        dimension,
        ):
    raddata_shape = raddata_ds['DBZH'].shape
    
    i_near = surrounding_data(
        VHF_df, 
        raddata_ds, 
        d=radius, 
        dimension=dimension,
        )
    mask = np.zeros(raddata_shape).ravel()
    mask[i_near] = 1
    mask = mask.reshape(raddata_shape)
    mask = (mask == 1)

    if alt_th:
        mask = mask & (raddata_ds['z'] > alt_th).values

    if dbzh_th is not None:
        mask = mask & (raddata_ds['DBZH'] > dbzh_th).values

    return mask

@dataclass
class ConfigMaskRADnearVHF:
    RADnearVHF_radius: float = 2000
    RADalt_threshold: float = 0
    RADdbzh_threshold: Optional[float] = None
    sparkle_selection_dimension: str = '3D'
    otherVHF_selection_dimension: str = 'horizontal'
    
# @SetKwargs(kwargs_addRADnearVHFmask)
# @profile
def add_mask_RADnearVHF(
        data_RAD, 
        data_LOFAR,
        config: ConfigMaskRADnearVHF,
        ):
    mask_otherVHF = get_mask_RADnearVHF(
        data_RAD.ds,
        data_LOFAR.df[~data_LOFAR.sparklemask],
        config.RADnearVHF_radius,
        config.RADalt_threshold,
        config.RADdbzh_threshold,
        dimension=config.otherVHF_selection_dimension,
        )
    mask_sparkles = get_mask_RADnearVHF(
        data_RAD.ds,
        data_LOFAR.df[data_LOFAR.sparklemask],
        config.RADnearVHF_radius,
        config.RADalt_threshold,
        config.RADdbzh_threshold,
        dimension=config.sparkle_selection_dimension,
        )

    data_RAD.ds = data_RAD.ds.assign_coords({
        "mask_otherVHF": (data_RAD.ds.x.dims, mask_otherVHF),
        "mask_sparkles": (data_RAD.ds.x.dims, mask_sparkles)})
    
    return data_RAD

'''
kwargs_add_RAD_near_sparkles_mask_multiradii = {
    "RADnearSparkle_radiuslist": KwargDict(type_=Optional[list], default=None, description="The radius that is used to collect radar data near VHF sources"),
    "RADalt_threshold": KwargDict(type_=float, default=0, description="Minimum altide threshold for the radar data."),
    "RADdbzh_threshold": KwargDict(type_=Optional[float], default=None, description="Minimum dbzh threshold for the radar data."),
}


@SetKwargs(kwargs_add_RAD_near_sparkles_mask_multiradii)
def add_RAD_near_sparkles_mask_multiradii(kwargs):
    for R in kwargs.RADnearSparkle_radiuslist:
        mask = get_mask_RADnearVHF(kwargs.raddata.ds,
                                 kwargs.LOFARdata.df[~kwargs.sparklemask],
                                 R,
                                 kwargs.RADalt_threshold,
                                 kwargs.RADdbzh_threshold,
                                 dimension='3D')

        kwargs.raddata.ds = kwargs.raddata.ds.assign_coords({
            f"mask_R-{R}km": (kwargs.raddata.ds.x.dims, mask)})

'''


def unfold_phi(phidp, kdp, rho, *, w=5, copy=False, th_phi=-80, th_kdp=(-5, 20), th_std=1):
    """Unfolds differential phase by adjusting values that exceeded maximum ambiguous range.

    Accepts arbitrarily dimensioned arrays, but THE LAST DIMENSION MUST BE THE RANGE.

    The algorithm is based on the paper of :cite:`Wang2009`.

    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        array of shape (...,nr) with nr being the number of range bins
    rho : :class:`numpy:numpy.ndarray`
        array of same shape as ``phidp``
    w : int, optional
       Width of the analysis window, defaults to 5.
    copy : bool, optional
       Leaves original `phidp` array unchanged if set to True (default: False)
    th_phi : float, optional
       Threshold for phase shift, defaults to -80.
    th_kdp : tuple, optional
       Threshold range for gradphi, defaults to (-5, 20).
    th_std : float, optional
       Threshold for standard deviation, defaults to 5.

    Returns
    -------
    phidp : :class:`numpy:numpy.ndarray`
        array of shape (..., n azimuth angles, n range gates) reconstructed :math:`Phi_{DP}`
    """
    shape = phidp.shape
    if rho.shape != shape:
        raise ValueError(
            f"`rho` ({rho.shape}) and `phidp` ({shape}) must have the same shape."
        )

    if copy is True:
        phidp = np.reshape(phidp, (-1, shape[-1])).copy()
    else:
        phidp = np.reshape(phidp, (-1, shape[-1]))
    kdp = np.reshape(kdp, (-1, shape[-1]))
    rho = np.reshape(rho, (-1, shape[-1]))

    beams, rs = phidp.shape

    # Compute the standard deviation within windows of 9 range bins
    stdarr = np.zeros(phidp.shape, dtype=np.float32)
    for r in range(rs - 9):
        stdarr[..., r] = np.std(phidp[..., r: r + w], axis=-1)

    for beam in range(beams):
        if np.all(phidp[beam] == 0):
            continue

        # step 1: determine location where meaningful PhiDP profile begins
        for j in range(0, rs - w):
            if (np.sum(stdarr[beam, j: j + w] < w * th_std) == w) and (
                np.sum(rho[beam, j: j + w] > 0.9) == w
            ):
                break

        ref = np.mean(phidp[beam, j: j + w])
        for k in range(j + w, rs):
            if np.sum(stdarr[beam, k - w: k] < w * th_std) == w and np.logical_and(
                    kdp[beam, k] > th_kdp[0], kdp[beam, k] < th_kdp[1]):
                ref += kdp[beam, k] * 0.5
                r += 1
                if (phidp[beam, k] - ref < th_phi):
                    if phidp[beam, k] < 0:
                        phidp[beam, k] += 360
            elif (phidp[beam, k] - ref < th_phi):
                if phidp[beam, k] < 0:
                    phidp[beam, k] += 360
            # elif (phidp[beam, k] - ref > 2* th_phi):
            #     if phidp[beam, k] > 0:
            #         phidp[beam, k] -= 360

    return phidp.reshape(shape)


def rolling_window(arr, window_size):
    """
    Creates a rolling window view of the input array.

    Parameters:
    arr (numpy.ndarray): Input 1D array.
    window_size (int): Size of the rolling window.

    Returns:
    numpy.ndarray: 2D array where each row is a rolling window view of the input array.
    """
    shape = (arr.size - window_size + 1, window_size)
    strides = (arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def unpack_hmc(ds, varlist):

    hmc_categories = ds.hmc.values
    for hmc_type in hmc_categories:
        ds[hmc_type] = ds.HMC.sel(hmc=hmc_type)
        varlist.append(str(hmc_type))
    varlist.remove("HMC")
    return ds, varlist


def repack_hmc(ds, hm_list, varlist):
    da_list = []
    for hm_type in hm_list:
        da_list.append(ds[hm_type])
        ds.drop_vars(hm_type)
        varlist.remove(hm_type)
    varlist.append("HMC")

    da = xr.concat(
        da_list, 
        dim='hmc',
        join="outer", 
        coords="different", 
        compat="equals",
        ).assign_coords(hmc=hm_list)
    ds["HMC"] = da
    return ds, varlist

def get_CRS_from_WTK(crs_wtk, loc_longitude, loc_latitude):
    return f"+proj=aeqd +lat_0={loc_latitude} +lon_0={loc_longitude} +x_0=0 +y_0=0 +datum=WGS84"


def data_calibration(data, variable):
    if variable == "PCP":
        data_corr = data * 0.5 - 32
    return data_corr


def NL25_grid():
    # Radar Grid
    # Code -- Eva van der Kooij (RDWK)
    gt = [0.0, 1.0, 0.0, -3650.0, 0.0, -1.0]
    # x 1000 to transfrom from kilometers to meters
    gt = [val * 1000 for val in gt]
    xsize, ysize = (700, 765)

    # upper left coord
    xul = gt[0]
    yul = gt[3]

    # stepsize
    xres = gt[1]
    yres = gt[5]

    # get the edge coordinates and add half the resolution
    # to go to center coordinates
    xmin = xul + xres * 0.5
    xmax = xul + (xres * xsize) - xres * 0.5
    ymin = yul + (yres * ysize) - yres * 0.5
    ymax = yul + yres * 0.5

    # define x and y values
    xs = np.linspace(xmin, xmax, xsize)
    ys = np.linspace(ymax, ymin, ysize)

    xmesh, ymesh = np.meshgrid(xs, ys)

    transformer = Transformer.from_crs("+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378137 +b=6356752 +x_0=0 +y_0=0 +type=crs",
                                       "+proj=latlon")

    # empty meshgrid to store lat,lon coordinates in
    Rlon, Rlat = np.empty([ysize, xsize]), np.empty([ysize, xsize])

    for i in range(len(xs)):
        for j in range(len(ys)):
            x_old = xmesh[j, i]
            y_old = ymesh[j, i]
            lon_, lat_ = transformer.transform(x_old, y_old)
            Rlon[j, i] = lon_
            Rlat[j, i] = lat_

    return Rlon, Rlat


def combineBorkumHDF5(input_files):
    """Combine multiple HDF5 files into a single in-memory HDF5 file."""
    combined_file = io.BytesIO()

    with h5py.File(combined_file, 'w') as outfile:

        with h5py.File(input_files[0], 'r') as first_file:
            conventions = first_file.attrs.get("Conventions")
            if conventions:
                outfile.attrs["Conventions"] = conventions

        for i, file_path in enumerate(input_files):
            with h5py.File(file_path, 'r') as infile:
                for key in infile.keys():
                    if key.startswith("dataset"):
                        infile.copy(
                            key, outfile, name=f"{key[:-1]}{i+1}", expand_soft=True, expand_refs=True)
                    else:
                        if key in outfile:
                            continue  # Skip if object already exists
                        else:
                            # Copy each dataset/group from input file to output file
                            infile.copy(key, outfile, name=key,
                                        expand_soft=True, expand_refs=True)
    combined_file.seek(0)  # Reset file pointer to the beginning
    return combined_file


def coord_tranform_ranges(xrange, yrange, lon0, lat0):
    from pyproj import Transformer

    proj_cart = "epsg:28992"
    proj_coord = "epsg:4326"
    transformer = Transformer.from_crs(proj_coord, proj_cart, always_xy=True)

    x0, y0 = transformer.transform(lon0, lat0)
    for i in [0, 1]:
        x, y = transformer.transform(xrange[i], yrange[i])
        xrange[i] = x - x0[0]
        yrange[i] = y - y0[0]

    return xrange, yrange