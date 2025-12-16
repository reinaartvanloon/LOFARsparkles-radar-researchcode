#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:35:49 2023

@author: rvloon
"""

from pandas import read_csv
import logging
from general import WindowExtent
from pyproj import Transformer

logger = logging.getLogger(__name__)


class data_key:
    """ class to store information on the dataset."""

    def __init__(self, value=None, longname="", unit="-"):
        self.value = value
        self.longname = longname
        self.unit = unit


class DataLOFAR:
    """Class to open data in the format as provided by Olaf."""

    def __init__(
            self, 
            filepath, 
            window_extent:WindowExtent=None, 
            crs=None,
            max_distance=None,
            ):
        with open(filepath, "r") as frame:
            plot_range, data_info, dataframe = frame.read().split("!")  # Different parts of information of the data

        # Save the different plotting variables
        plot_vars = plot_range.split()
        self.x_range = data_key([plot_vars[0], plot_vars[1]], "Longitudinal plotting range", "km")
        self.y_range = data_key([plot_vars[2], plot_vars[3]], "Lattitudinal plotting range", "km")
        self.z_range = data_key([plot_vars[4], plot_vars[5]], "Vertical plotting range", "km")
        self.t_range = data_key([plot_vars[6], plot_vars[7]], "Temporal plotting range", "ms")
        self.unknown_1 = data_key(plot_vars[8])
        self.t_shift = data_key(plot_vars[9], "Time shift", "ms")

        # Save the information on data
        data_info_lst = data_info.split()
        self.unknown_2 = data_key(data_info_lst[0])
        self.nr_sources = data_key(data_info_lst[1], "Number of VHf sources")
        self.unknown_3 = data_key(data_info_lst[2])
        self.t_shift = data_key(data_info_lst[3], "Time shift", "s")
        self.data_code = data_key(data_info_lst[4].split("/")[-1], "The coded name of the dataset")
        self.unknown_4 = data_key(data_info_lst[5])
        self.unknown_5 = data_key(data_info_lst[6])
        self.unknown_6 = data_key(data_info_lst[7])
        self.unknown_7 = data_key(data_info_lst[8])
        self.unknown_8 = data_key(data_info_lst[9])
        self.unknown_9 = data_key(data_info_lst[10])
        self.unknown_9 = data_key(data_info_lst[10])

        self.crs = 'LOFAR'
        self.max_distance = max_distance
        # Load and save data
        cols = ['id', 't', 'x', 'y', 'z', 'I']

        df = read_csv(filepath,
                      skiprows=2,
                      usecols=[0, 1, 2, 3, 4, 5],
                      names=cols,
                      delimiter=r'\s+',
                      engine='c')

        df['x'], df['y'], df['z'] = 1e3 * df['x'], 1e3 * df['y'], 1e3 * df['z']

        # Emtpy WindowExtent allows all LOFAR data to be loaded
        if window_extent is None:
            window_extent = WindowExtent()
            
        if max_distance is not None:
            print(f"Filtering LOFAR data for max distance of {max_distance} meter from superterp.")
            r2 = df['x']**2 + df['y']**2 #radius squared
            mask = r2<max_distance**2
            df = df[mask]

        self.df = df

        # Besides trimming LOFAR data to window_extent,
        # also tranform data to the target_crs
        self,terp = self.terp_correction()
        self,_ = self.select_2_windowextent(window_extent, crs)

        if df.shape[0] == 0:
            logger.info("No datapoints in your selection window")
            

    def select_2_windowextent(self, window_extent, crs_target, mask_list=[]):
        self.window_extent = WindowExtent(
            t_range=window_extent.t, 
            y_range=window_extent.y, 
            z_range=window_extent.z, 
            x_range=window_extent.x,
            crs=window_extent.crs,
            )  # Set the extent of the window we want to display
        
        # Select temporal and vertical window for data output
        self.df, self.window_extent.t, mask_list = data_window(self.df, 't', self.window_extent.t, mask_list=mask_list)
        self.df, self.window_extent.z, mask_list = data_window(self.df, 'z', self.window_extent.z, mask_list=mask_list)
        
        # Align the coordinate reference systems (crs)
        self = self.coord_transformer(crs_target)
        if ( 
                self.window_extent.x is not None
            ) & (
                self.window_extent.y is not None
            ):
                self.window_extent = self.window_extent.transform_crs(crs_target)
        # Select horizontal windows for data output
        self.df, self.window_extent.x, mask_list = data_window(self.df, 'x', self.window_extent.x, mask_list=mask_list)
        self.df, self.window_extent.y, mask_list = data_window(self.df, 'y', self.window_extent.y, mask_list=mask_list)
        
        return self, mask_list

    def print_keys(self):
        '''Print all the data information'''

        for item in vars(self).items():
            if isinstance(item[1], data_key):
                key, _ = item[0], item[1]
                print("{}: {} ({} [{}])".format(key, _.value, _.longname, _.unit))

    def terp_correction(self):
        """
        In the LOFAR data, the superterp is the origin
        Subtract the terp location from data to get epsg 28992 crs.

        Returns
        -------
        self: Object
            LOFAR data
        terp : Dictionary
            Longitude, latitude coordinates of the superterp

        """
        data_proj = "epsg:28992"
        terp = dict(latitude=52.915007, longitude=6.869711)  # Reference location of LOFAR data --> Terp in Exloo

        transformer_lonlat_to_crs = Transformer.from_crs("epsg:4326", data_proj, always_xy=True)  # Meter to degrees longitude and latitude
        terp['x'], terp['y'] = transformer_lonlat_to_crs.transform(
            terp['longitude'], 
            terp['latitude'],
            )
        self.df['x'], self.df['y'] = self.df['x'] + terp['x'], self.df['y'] + terp['y']
        
        self.crs = "epsg:28992"
        
        return self, terp
        
    def coord_transformer(self, crs):
        """
        Change the LOFAR crs to another one.

        Returns
        -------
        self: Object
            LOFAR data
        """
        if crs != "epsg:28992":
            print("Transforming LOFAR to target crs.")
            transformer_LOFAR_to_crs = Transformer.from_crs(
                self.crs, 
                crs,
                always_xy=True,
                )  # Meter to degrees longitude and latitude
            self.df['x'], self.df['y'] = transformer_LOFAR_to_crs.transform(
                self.df['x'], 
                self.df['y'],
                )
        else: 
            print("LOFAR data already in the right crs.")
            
        self.crs = crs
        
        return self


def VHFclustering(df, d, t, n, method='DBSCAN'):
    """Function that separates the LOFAR sources into lightning structures.
    It compares the distance in time and space of each source, to previous sources.
    If the time and space distance gives a velocity smaller than the v_max,
    it adds the source to the lightning structure.
    Each lightning structure is numbered. 

    Parameters
    ----------

    Returns
    -------
    None.
    """
    from sklearn.cluster import DBSCAN
    import numpy as np

    if method != "DBSCAN":
        raise AttributeError("No other methods supported yet")

     # Ensure df is a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Prepare data for clustering
    XYZ = np.stack((df.x.values, df.y.values, df.z.values), axis=1) / d
    T = df.t.values[:, np.newaxis] / t
    XYZT = np.concatenate((XYZ, T), axis=1)

    # Apply DBSCAN clustering
    db = DBSCAN(eps=1.0, min_samples=n).fit(XYZT)

    # Safely add or update the 'structure_label' column
    df.loc[:, 'structure_label'] = db.labels_

    return df


def surrounding_data(df_VHF, ds_ref, d, D=None, method="nearest_distance", dimension='3D'):
    from scipy.spatial import KDTree
    import numpy as np

    if method != "nearest_distance":
        raise AttributeError("Method '{}' not supported. Choose from: {}".format(method, 'nearest_distance'))

    # mask = sweep.distance_to_crosssect<D
    x_RAD = ds_ref.x.values.ravel()
    y_RAD = ds_ref.y.values.ravel()
    z_RAD = ds_ref.z.values.ravel()

    if dimension == 'horizontal':
        points = np.stack((x_RAD, y_RAD), axis=1)
    elif dimension == '3D':
        points = np.stack((x_RAD, y_RAD, z_RAD), axis=1)
    else:
        raise AttributeError(f"dimension {dimension} not supported")

    tree = KDTree(points)

    x_LO = df_VHF.x
    y_LO = df_VHF.y
    z_LO = df_VHF.z

    if dimension == 'horizontal':
        target_points = np.stack((x_LO, y_LO), axis=1)
    elif dimension == '3D':
        target_points = np.stack((x_LO, y_LO, z_LO), axis=1)
    else:
        raise AttributeError(f"dimension kwarg {dimension} not supported")

    ii_d = tree.query_ball_point(target_points, r=d)
    if len(ii_d) == 0:
        ii_near = []
    elif len(ii_d) == 1:
        ii_near = ii_d
    else:
        ii_near = np.unique(np.concatenate(ii_d)).astype(np.int64)
    # time = ds_ref.time.broadcast_like(ds_ref.x)
    # d_t = time.values.ravel()[ii_near]

    if D:
        ii_D = tree.query_ball_point(target_points, r=D)
        ii_D = np.unique(np.concatenate(ii_D))
        ii_surround = ii_D[~np.isin(ii_D, ii_near)].astype(np.int64)

        return ii_near, ii_surround
    else:
        return ii_near


def data_window(df, key, window_lims=None, mask_list=[]):
    if window_lims:
        if (not window_lims[0]) and (not window_lims[1]):  # Both window limits undefined
            x_min, x_max = df[key].min(), df[key].max()  # Maximum and minimum value
            dx = x_max - x_min  # Datarange
            window_lims = [x_min - 1 / 20 * dx, x_max + 1 / 20 * dx]  # PLotting range with extra range of 10% on each side

        elif not window_lims[0]:  # Lower limit not defined
            mask_list = [mask[df[key] >= window_lims[0]] for mask in mask_list]
            df = df[df[key] <= window_lims[1]]  # Filter with maximum value
            x_min = df[key].min()
            dx = window_lims[1] - x_min  # Datarange
            window_lims = [x_min - 1 / 20 * dx, window_lims[1]]  # PLotting range with extra range of 10% on each side
            mask_list = [mask[df[key] <= window_lims[1]] for mask in mask_list]

        elif not window_lims[1]:  # Upper limit not defined
            mask_list = [mask[df[key] >= window_lims[0]] for mask in mask_list]
            df = df[df[key] >= window_lims[0]]  # Filter with minimum value
            x_max = df[key].max()
            dx = x_max - window_lims[0]  # Datarange
            window_lims = [window_lims[0], window_lims[1]]  # PLotting range with extra range of 10% on each side

        else:  # Upper and lower window limit defined
            mask_list = [mask[df[key] >= window_lims[0]] for mask in mask_list]
            df = df[df[key] >= window_lims[0]]  # Filter with minimum value

            mask_list = [mask[df[key] <= window_lims[1]] for mask in mask_list]
            df = df[df[key] <= window_lims[1]]  # Filter with maximum value

        print(f"{len(df)} VHF sources in {window_lims[0]}<{key}<{window_lims[1]}")
        
    else:  # No data limit, so make it based on data for the sake of plotting
        x_min, x_max = df[key].min(), df[key].max()  # Maximum and minimum value
        dx = x_max - x_min  # Datarange
        window_lims = [x_min - 1 / 20 * dx, x_max + 1 / 20 * dx]  # PLotting range with extra range of 10% on each side

    
    return df, window_lims, mask_list
