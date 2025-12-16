#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:09:31 2023

Python functions for general usage
@author: rvloon
"""
from cartopy.crs import Projection
from pyproj import Transformer
import cartopy.crs as ccrs
from geopy.distance import distance
import numpy as np
import xarray as xr
import datetime
from scipy.interpolate import griddata
import os
import logging
import sys
import functools
import json
import argparse
from typing import Optional, List, TypeVar, Tuple
import copy
import geopandas as gpd
#from pydantic import ValidationError
from dataclasses import dataclass, field, is_dataclass
from matplotlib.patches import Rectangle, Ellipse

logger = logging.getLogger()

@dataclass
class ConfigPlot:
    outname: str = None
    save: bool = False
    live_plot: Optional[bool] = None
    title: Optional[str] = ""

class WindowExtent:
    def __init__(
            self, 
            y_range=None, 
            t_range=None,
            z_range=None, 
            x_range=None, 
            transformer_lonlat_to_crs=None,
            crs='epsg:4326'
            ):
        
        if t_range:
            self.t = t_range
        else:
            self.t = None

        if z_range:
            self.z = z_range
        else:
            self.z = None

        if x_range:
            self.x = x_range
        else:
            self.x = None

        if y_range:
            self.y = y_range
        else:
            self.y = None
        
        self.transformer_lonlat_to_crs = transformer_lonlat_to_crs         
        self.crs = crs
    
    def transform_crs(self, crs_target):
        # No transformation needed if the target crs is the same as the current crs
        if crs_target == self.crs:
            return self
        else:            
            print(f"Transforming window_extent crs: {self.crs} to {crs_target}")
            self.transformer_lonlat_to_crs = Transformer.from_crs(
                self.crs,
                crs_target, 
                always_xy=True,
                )
            
            # Can only transform (x,y) coordinate. So must have corner points
            if (    
                    (
                        self.x is not None
                    ) or ( 
                        self.y is not None 
                    )
                ) and not (
                    (
                        self.x[0] and self.y[0] # Bottom left corner
                    ) or (
                        self.x[1] and self.y[1] # Top right corner
                    )
                ):
                raise(ValueError("Must have x range and y range both explicilty defined"))
               
            x_min, y_min = self.transformer_lonlat_to_crs.transform(
                self.x[0], self.y[0]
            )
       
            x_max, y_max = self.transformer_lonlat_to_crs.transform(
                self.x[1], self.y[1],
            )
            
            window_extent = WindowExtent(
                x_range=[x_min,x_max],
                y_range=[y_min,y_max],
                z_range=self.z,
                t_range=self.t,
                crs=crs_target,
                )
            window_extent.transformer_lonlat_to_crs = self.transformer_lonlat_to_crs
            
            return window_extent
            
@dataclass
class ConfigSpatialPlot(ConfigPlot):
    epsg: int = 4326
    plot_extent: WindowExtent = None
    plot_margins: List[float] = field(default_factory=lambda: [0, 0])
    xy_ratio: str = "equal"
    terp: Optional[bool] = False
    dirpath_shapefiles_borders: Optional[str] = None  
    
def plot_borders(dirpath_shapefiles_borders, ax, crs):
    n = 0
    for file in os.listdir(dirpath_shapefiles_borders):
        if file[-4:] == ".shp":
            # print(f"Plotting borders of {file}")
            borders = gpd.read_file(dirpath_shapefiles_borders + "/"+file)
            borders['geometry'] = borders['geometry'].to_crs(crs)
            ax.add_geometries(
                borders['geometry'], 
                crs=crs, 
                linewidth=0.5, 
                edgecolor='black', 
                facecolor='none',
                )
            n=n+1
            
    if n==0:
        print("No shape files found to plot borders.")
            
    return 1

def lineProjection(pointA, pointB, df, perp_range=None):
    AB_x = abs(pointA[0] - pointB[0])
    AB_y = abs(pointA[1] - pointB[1])
    AB = (AB_x**2 + AB_y**2)**0.5

    AC_x = abs(pointA[0] - df['x'].values)
    AC_y = abs(pointA[1] - df['y'].values)
    AC_2 = AC_x**2 + AC_y**2

    BC_x = abs(pointB[0] - df['x'].values)
    BC_y = abs(pointB[1] - df['y'].values)
    BC_2 = BC_x**2 + BC_y**2

    AD = (AC_2 - BC_2 + AB**2) / (2 * AB)

    if perp_range is None:
        mask = (AD >= 0) & (AD <= AB)
    else:
        CD_2 = AC_2 - AD**2
        mask = (AD >= 0) & (AD <= AB) & (CD_2 <= perp_range**2)

    df_ = df[mask].copy()
    df_['projection'] = AD[mask]

    return df_, mask


def v_2_lonlat(lon, lat, vx, vy):
    v_lat = np.array([distance(meters=vy_i).destination((lat_i, lon_i), bearing=0)[0] - lat_i for lon_i, lat_i, vy_i in zip(lon, lat, vy)])
    v_lon = np.array([distance(meters=vx_i).destination((lat_i, lon_i), bearing=90)[1] - lon_i for lon_i, lat_i, vx_i in zip(lon, lat, vx)])
    return v_lon, v_lat


def round_time(dt=None, date_delta=datetime.timedelta(minutes=1), to='average'):
    """
    Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    from:  http://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
    """
    round_to = date_delta.total_seconds()
    if dt is None:
        dt = datetime.now()

    seconds = (dt - dt.min).seconds

    if seconds % round_to == 0 and dt.microsecond == 0:
        rounding = (seconds + round_to / 2) // round_to * round_to
    else:
        if to == 'up':
            # // is a floor division, not a comment on following line (like in javascript):
            rounding = (seconds + dt.microsecond / 1000000 + round_to) // round_to * round_to
        elif to == 'down':
            rounding = seconds // round_to * round_to
        else:
            rounding = (seconds + round_to / 2) // round_to * round_to

    delta = datetime.timedelta(0, rounding - seconds, - dt.microsecond)
    dt_round = dt + delta

    return dt_round, delta


def advect_data(ds, varkey, delta_t, gridtype="unstructured", target_grid=None):
    dt_s = delta_t.total_seconds()  # convert time to seconds

    # Interpolatie v_field to location of values
    # if v_field.manual:
    #     pass

    # elif v_field.grid.gridtype=='regular' and not v_field.grid.z: # v_field regular grid using meshgrid
    #     interp_vx =  RegularGridInterpolator((v_field.grid.x, v_field.grid.y), v_field.x)
    #     interp_vy =  RegularGridInterpolator((v_field.grid.x, v_field.grid.y), v_field.x)

    #     vx = interp_vx((lon_pre, lat_pre))
    #     vy = interp_vy((lon_pre, lat_pre))
    #     #vx = griddata((v_field.grid.xmesh.flatten(), v_field.grid.ymesh.flatten()), v_field.x.flatten(), (lon_pre, lat_pre))
    #     #vy = griddata((v_field.grid.xmesh.flatten(), v_field.grid.ymesh.flatten()), v_field.y.flatten(), (lon_pre, lat_pre))

    # elif v_field.grid.gridtype=='regular_xy' and v_field.grid.z is not None: #Irregular z-dimension, but regular xy-grid
    #     vx = griddata((v_field.grid.xmesh.flatten(), v_field.grid.ymesh.flatten(), v_field.grid.z.flatten()), v_field.x.values.flatten(), (lon_pre, lat_pre, alt_pre))
    #     vy = griddata((v_field.grid.xmesh.flatten(), v_field.grid.ymesh.flatten(), v_field.grid.z.flatten()), v_field.y.values.flatten(), (lon_pre, lat_pre, alt_pre))

    # elif not v_field.grid.z: # v_field with irregular grid
    #     vx = griddata((v_field.grid.x, v_field.grid.y), v_field.x, (lon_pre, lat_pre))
    #     vy = griddata((v_field.grid.x, v_field.grid.y), v_field.y, (lon_pre, lat_pre))

    # Convert velocities to longitude & latitude
    v_lon, v_lat = v_2_lonlat(
        ds.longitudes.values, 
        ds.latitudes.values, 
        ds.u.values, 
        ds.v.values,
        )

    # Advect
    logger.info(f"Calculating advected values for a time of {delta_t}.")
    ds.longitudes_advected = xr.DataArray(ds.longitudes - dt_s * v_lon, dims='points')
    ds.longitudes_advected = xr.DataArray(ds.latitudes - dt_s * v_lat, dims='points')

    if target_grid:  # Interpolate to target_grid and return, grid is the input grid
        logger.info("Interpolate advected values to target grid.")
        values_advected = griddata((ds.longitudes_advected, ds.longitudes_advected),
                                   ds[varkey].values, (target_grid.xmesh, target_grid.ymesh),
                                   method='linear', fill_value=0)
        return values_advected


class GRID:
    def __init__(self, x, y, z=None, gridtype="regular", gridname="grid"):
        self.gridtype = gridtype
        self.gridname = gridname
        self.x, self.y, self.z = x, y, z
        if gridtype == "regular" and z is not None:
            self.xmesh, self.ymesh, self.zmesh = np.meshgrid(x, y, z)
        elif gridtype == "regular_xy" and z is not None:
            self.xmesh, self.ymesh = np.meshgrid(x, y)
            self.xmesh = np.tile(self.xmesh, (z.shape[0], 1, 1))
            self.ymesh = np.tile(self.ymesh, (z.shape[0], 1, 1))
        elif gridtype == "regular":
            self.xmesh, self.ymesh = np.meshgrid(x, y)


def distance_to_lon(d_x, latitude):
    # Starting point (longitude doesn't matter as we're only interested in the latitude)
    start_point = (latitude, 0)
    # Use the .destination function to calculate the end point Eastward
    end_point = distance(kilometers=d_x).destination(start_point, bearing=90)
    return end_point[1]


class Vfield:  # in m/s
    def __init__(self, vx, vy, grid):
        self.x = vx
        self.y = vy
        self.grid = grid


def outpath_gen(inpath, outdir, outname):
    if outdir == None:  # When no directory is given to save the figure in (outpath)
        outdir = os.path.dirname(inpath)
    if outname == None:  # When no filename is given to figure
        outname = os.path.basename(os.path.splitext(inpath)[0])
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, outname)


def time_to_sec(dateobj):
    total = int(dateobj.strftime('%S'))
    total += int(dateobj.strftime('%M')) * 60
    total += int(dateobj.strftime('%H')) * 60 * 60
    total += (int(dateobj.strftime('%j')) - 1) * 60 * 60 * 24
    total += (int(dateobj.strftime('%Y')) - 1970) * 60 * 60 * 24 * 365
    return total


def tRange_2list(t_range, delta_dt, dt_format_in):
    dt_range = [datetime.datetime.strptime(dt_str, dt_format_in) for dt_str in t_range]
    num_dt_list = np.arange(dt_range[0].timestamp(), dt_range[1].timestamp() + 1,
                            step=delta_dt.total_seconds())
    dt_list = [datetime.datetime.fromtimestamp(num_dt) for num_dt in num_dt_list]
    return dt_list


def customLOG(logger_name, print_console=True, logfile=True, verbose=True, outdir=None, outname=None):
    import logging
    import sys

    # Determine the logging level based on the verbose flag
    if verbose == "DEBUG":
        verbosity = logging.DEBUG
    elif verbose:
        verbosity = logging.INFO
    else:
        verbosity = logging.WARNING

    logger = logging.getLogger(logger_name)
    logger.setLevel(verbosity)

    if logfile:
        logdir = os.path.join(outdir, 'logfiles')
        log_outpath = os.path.join(logdir, outname + '.log')
        os.makedirs(logdir, exist_ok=True)  # Make directory if it doesn't exist

        file_handler = logging.FileHandler(log_outpath, mode='w')
        file_format = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
        file_handler.setFormatter(file_format)
        file_handler.setLevel(verbosity)
        logger.addHandler(file_handler)

    # Define a Handler which writes messages to the sys.stderr if print_console is True
    if print_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console_handler.setFormatter(console_format)
        console_handler.setLevel(verbosity)
        logger.addHandler(console_handler)

    logging.basicConfig()
    return logger


def match_dsets(ds_A, ds_B):
    import pandas as pd

    # Convert datasets to pandas DataFrame for easier manipulation
    df_A = ds_A.to_dataframe()
    df_B = ds_B.to_dataframe()

    # Assuming 'longitude' and 'latitude' are the coordinate names
    merged_df = pd.merge(df_A, df_B, on=['longitude', 'latitude'], how='left')

    # Handle cases where there are no matches
    if merged_df.isnull().any().any():
        logger.error("Some coordinates in the target dataset donnot have corresponding values in the other dataset. \n Consider expanding the other dataset to the acquired range for the target ds.")
        logger.error("Target dataset latitude range: {}-{} and longitude range: {}-{}".format(ds_A.longitude.min(), ds_A.longitude.max(), ds_A.latitude.min(), ds_A.latitude.max()))
        logger.error("Other dataset latitude range: {}-{} and longitude range: {}-{}".format(ds_B.longitude.min(), ds_B.longitude.max(), ds_B.latitude.min(), ds_B.latitude.max()))

        sys.exit(0)
        # Handle this case as needed, e.g., drop, fill with default values, or exit

    return merged_df


def trim_2D_unstructured_data(data, lon, lat, extent):
    # Make mask to remove part of data
    mask_x = (lon >= extent.x[0]) & (lon <= extent.x[1])
    mask_y = (lat >= extent.y[0]) & (lat <= extent.y[1])
    mask = mask_x & mask_y

    rows, cols = np.where(mask)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    data = data[row_min:row_max + 1, col_min:col_max + 1]
    lon = lon[row_min:row_max + 1, col_min:col_max + 1]
    lat = lat[row_min:row_max + 1, col_min:col_max + 1]

    return data, lon, lat


def round_to_multiple(number, multiple, direction='nearest'):
    from math import ceil, floor

    if direction == 'nearest':
        return multiple * round(number / multiple)
    elif direction == 'up':
        return multiple * ceil(number / multiple)
    elif direction == 'down':
        return multiple * floor(number / multiple)
    else:
        return multiple * round(number / multiple)


def plotrange_2_contourlevels(range_lst, contour_binsize, data):
    if not range_lst:
        if contour_binsize:
            range_lst = [round_to_multiple(data.min(), contour_binsize, 'down'),
                         round_to_multiple(data.max(), contour_binsize, 'up')]
        else:
            range_lst = [None, None]
            levels = None
            return levels, range_lst

    elif range_lst[0] == 'None':
        range_lst[0] = round_to_multiple(data.min(), contour_binsize, 'down')
    elif range_lst[1] == 'None':
        range_lst[1] = round_to_multiple(data.max(), contour_binsize, 'down')

    levels = np.arange(float(range_lst[0]), float(range_lst[1]) + 0.1, contour_binsize)  # levels for contourplot
    return levels, range_lst


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def plot_parallel_lines(ax, pointA, pointB, distance):
    A = np.array(pointA)
    B = np.array(pointB)
    AB = B - A
    AB_dist = np.sqrt(AB[0]**2 + AB[1]**2)

    # calculate perpendicular unit vector
    u_vec_perp = np.array([-AB[1], AB[0]]) / AB_dist

    offset = u_vec_perp * distance
    A1 = A + offset
    A2 = A - offset
    B1 = B + offset
    B2 = B - offset

    line1, = ax.plot([A1[0], B1[0]], [A1[1], B1[1]])
    line2, = ax.plot([A2[0], B2[0]], [A2[1], B2[1]])

    return [line1, line2]


def parse_polygons(polygon_str):
    '''
    Read the trans

    Parameters
    ----------
    polygon_str : string
        Strings that describe polygons. ';' delimiters used to distinguish between polygons and 
        ',' delimiter to ditinguish between x and y coordinates.

    Returns
    -------
    parsed_polygons : TYPE
        DESCRIPTION.

    '''
    # Split the string by semicolon to get each polygon
    polygons = polygon_str.split(';')

    # For each polygon, split by spaces and convert coordinates to tuples
    parsed_polygons = []
    for poly in polygons:
        vertices = [tuple(map(float, coord.split(','))) for coord in poly.strip().split()]
        parsed_polygons.append(vertices)

    return parsed_polygons


def parse_float_list(floatlist_str):
    '''
    Read the trans

    Parameters
    ----------
    floatlist_str : string
        Strings that describes a list of floats ';' delimiters used to distinguish between polygons and 
        ',' delimiter to ditinguish between x and y coordinates.

    Returns
    -------
    parsed_polygons : TYPE
        DESCRIPTION.

    '''
    return [float(item) for item in floatlist_str[1:-1].split(',')]


def parse_string_list(stringlist_str):
    return [str(item) for item in stringlist_str[1:-1].split(',')]


def select_points_in_windows(arr, window_lst=None, var: Optional[str] = None):
    # Check if datapoints in df are within time windows

    if window_lst and len(window_lst) > 0:  # Not none and not an empty list

        mask = np.ones(arr.shape[0]) == 0  # Assume no points within window
        mask_true = mask == 0  # List of only True, for convenience

        for window in window_lst:  # Add points to mask if inside window
            print(f"Filtering {var} variable on {window}.")

            if window[0]:  # If lower boundary defined, so not None
                mask1 = (arr > window[0])
            else:
                mask1 = mask_true

            if window[1]:  # If lower boundary defined, so not None
                mask2 = (arr < window[1])
            else:
                mask2 = mask_true

            mask = (mask1 & mask2) | mask

    else:  # No time windows specified, so everythnig passes
        print(f"No {var} window specified, so not filtering on this variable.")
        mask = np.ones(arr.shape) == 1

    return mask


def select_points_in_polygons(x_arr, y_arr, xy_polygon_lst=None):
    # Check if datapoints in df are within polygons

    # Check if x_arr and y_arr have the same shape
    if x_arr.shape == y_arr.shape:
        pass
    else:
        raise (Exception, "x_arr and y_arr should have the same shape.")

    if xy_polygon_lst and len(xy_polygon_lst) > 0:
        import matplotlib.path as mpltPath

        mask = np.ones(x_arr.shape[0]) == 0  # Assume no points in polygon

        for polygon in xy_polygon_lst:
            path = mpltPath.Path(polygon)
            # Add points to mask if in polygon
            mask = (path.contains_points(np.array([x_arr, y_arr]).T)) | (mask)
    else:
        print("No xy_polygons specified, so not filtering for polygons.")

        mask = np.ones(x_arr.shape[0]) == 1

    return mask  # mask of points in df that are within polygons

class KwargConfiguration:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return self.__dict__.copy()  # Return a copy of the instance's attributes as a dictionary

# Class to represent each kwarg, including its type, default value, and description


class KwargDict:
    def __init__(self, type_: type,
                 *,  # Forces all following arguments to be key-word only
                 description: str = "",
                 required: bool = False,
                 default: any = None,
                 CLparser: callable = None):
        self.type = type_                 # Type of the kwarg (e.g., int, float)
        self.description = description    # Description of the kwarg for documentation
        self.default = default            # Default value if the kwarg is not provided
        self.required = required          # Determines if it is a required kwarg
        self.CLparser = CLparser   # Optional custom parser function for bash integration

T = TypeVar("T")

from dataclasses import asdict
        
class DataclassOnlyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        # If it's a dataclass, turn it into a dict (recursively).
        if is_dataclass(o):
            return asdict(o)
        # Otherwise, let json raise a TypeError—
        # this prevents accidentally serializing objects as strings.
        return super().default(o)

def save_configs_json(configs: dict, filename: str):
    """
    Save multiple dataclass instances to JSON.
    Nested dataclasses all become dicts; unknown objects cause a TypeError.
    """
    # First, turn the top‐level into plain dicts too
    plain = {name: asdict(obj) for name, obj in configs.items()}

    with open(filename, 'w') as f:
        json.dump(
            plain,
            f,
            indent=2,
            cls=DataclassOnlyJSONEncoder
        )

def RunFromCL(kwarg_dicts: dict[str, KwargDict]):
    def decorator(func: callable):
        @functools.wraps(func)
        def wrapper_RunFromCL(**kwargs):
            parser = argparse.ArgumentParser(description=func.__doc__)
            for name, kwarg in kwarg_dicts.items():
                # Use custom parse function for comman line arguments
                arg_type = kwarg.CLparser if kwarg.CLparser else kwarg.type

                parser.add_argument(
                    f"--{name}",
                    type=arg_type,
                    default=argparse.SUPPRESS,
                    help=f"{kwarg.description} (type: {kwarg.type.__name__})",
                )
            # Parse arguments from the command line
            args = parser.parse_args()
            parsed_kwargs = vars(args)
            # Call the decorated function with parsed arguments
            # [print(key,value) for key, value in parsed_kwargs.items()]
            # print(f"RunFromCL kwargs out: {parsed_kwargs}")
            return func(**parsed_kwargs)

        return wrapper_RunFromCL
    return decorator


def deepcopyKwargs(func):
    """Deepcopy kwargs object before passing them to the function."""
    @functools.wraps(func)
    def wrapper_deepcopyKwargs(inargs, *args, **kwargs):
        # Make a deepcopy of kwargs to avoid modifying the original object
        copied_inargs = copy.deepcopy(inargs)
        return func(copied_inargs, *args, **kwargs)
    return wrapper_deepcopyKwargs

def save_config(filepath, kwargs):
    """
    Saves the attributes of a kwargs object to a JSON file.

    Args:
        filepath (str): The path where the JSON file will be saved.
        kwargs (object): An object containing various attributes to save.
    """

    # #Remove variables that represent personal information, e.g. when it is a directory
    # # Create a list of keys to remove
    # keys_to_remove = [key for key in kwargs.keys() if any(str_ in key for str_ in ['dir', 'path'])]

    # # Remove the keys from kwargs
    # for key in keys_to_remove:
    #     kwargs.pop(key)

    # Serialize and save the dictionary as a JSON file
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as json_file:
            json.dump(kwargs, json_file, indent=4, default=str)  # default=str handles non-serializable attributes
        print(f"Configuration saved successfully to {filepath}.")
    except Exception as e:
        print(f"An error occurred while saving the configuration: {e}")


def BoolParser(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

from dataclasses import dataclass, field
from typing import Optional,  List, Literal, Dict

@dataclass
class PlottingObject:
    obj_type: Literal['annotation', 'arrow', 'rectangle', 'ellipse']
    ax_name: str

    # --- annotation fields ---
    xy: Optional[Tuple[float, float]] = None # also for rectangle and ellipse
    text: Optional[str]                = None
    path_effect: Optional[List]        = None

    # --- arrow fields ---
    x:  Optional[float] = None
    y:  Optional[float] = None
    dx: Optional[float] = None
    dy: Optional[float] = None

    # --- rectangle fields ---
    width:  Optional[float] = None # also for ellipse
    height: Optional[float] = None # also for ellipse

    # --- shared styling ---
    kwargs: Dict = field(default_factory=dict)
    crs:    Optional[str] = None

    def __post_init__(self):
        if self.obj_type == 'annotation':
            if self.xy is None or self.text is None:
                raise ValueError("annotation requires xy and text")
        elif self.obj_type == 'arrow':
            if None in (self.x, self.y, self.dx, self.dy):
                raise ValueError("arrow requires x, y, dx, dy")
        elif self.obj_type == 'rectangle':
            if self.xy is None or self.width is None or self.height is None:
                raise ValueError("rectangle requires xy, width, height")
        elif self.obj_type == 'ellipse':
            if self.xy is None or self.width is None or self.height is None:
                raise ValueError("ellipse requires xy, width, height")
        else:
            raise ValueError(f"Unknown obj_type {self.obj_type}")

def _draw_on_axis(ax, obj):
    """Draw a single PlottingObject on one Axes."""
    if obj.obj_type == 'arrow':
        ax.arrow(obj.x, obj.y, obj.dx, obj.dy, **obj.kwargs)

    elif obj.obj_type == 'annotation':
        txt = ax.annotate(obj.text, xy=obj.xy, **obj.kwargs)
        if obj.path_effect:
            txt.set_path_effects(obj.path_effect)

    elif obj.obj_type == 'rectangle':
        rect = Rectangle(obj.xy, obj.width, obj.height, **obj.kwargs)
        ax.add_patch(rect)
    
    elif obj.obj_type == 'ellipse':
        ellipse = Ellipse(obj.xy, obj.width, obj.height, **obj.kwargs)
        ax.add_patch(ellipse)

def add_plotting_objects(objs: List[PlottingObject], axes_mapping: Dict[str, any]):
    """
    Place each PlottingObject on its specified axis.
    If obj.ax_name == "all", draw on every axis in axes_mapping.
    """
    for obj in objs:
        # Determine target axes
        if obj.ax_name == "all":
            targets = axes_mapping.values()
        else:
            # Will KeyError if name is invalid
            targets = [axes_mapping[obj.ax_name]]

        for ax in targets:
            print(f"Plotting {obj.obj_type} in axis {obj.ax_name}.")
            _draw_on_axis(ax, obj)
            

# %% Modify the limits of a projection beyond default


class ProjectCustomExtent(Projection):

    def __init__(
            self, 
            epsg, 
            window_extent:WindowExtent=None,
            ):
        self.epsg = epsg
        try:
            # Directly create the CRS using the EPSG code
            base_crs = ccrs.epsg(epsg)
            # Initialize the Projection base class with the projection string
            super().__init__(proj4_params=base_crs.proj4_init)

        except Exception as e:
            raise ValueError(f"Error initializing CRS from epsg {epsg}: {e}")

        # Retrieve default x and y limits from the Cartopy CRS object
        default_xmin, default_xmax = base_crs.x_limits
        default_ymin, default_ymax = base_crs.y_limits

        # Set initial limits to default values
        self.xmin = default_xmin
        self.xmax = default_xmax
        self.ymin = default_ymin
        self.ymax = default_ymax

        if window_extent is not None:
            # Update the extent only if it extends beyond the defaults
            if window_extent.x[0] is not None:
                self.xmin = min(default_xmin, window_extent.x[0])
            if window_extent.x[1] is not None:
                self.xmax = max(default_xmax, window_extent.x[1])
            if window_extent.y[0] is not None:
                self.ymin = min(default_ymin, window_extent.y[0])
            if window_extent.y[1] is not None:
                self.ymax = max(default_ymax, window_extent.y[1])

    @Projection.boundary.getter
    def boundary(self):
        coords = ((self.x_limits[0], self.y_limits[0]),
                  (self.x_limits[0], self.y_limits[1]),
                  (self.x_limits[1], self.y_limits[1]),
                  (self.x_limits[1], self.y_limits[0]))
        return ccrs.sgeom.LineString(coords)

    @Projection.x_limits.getter
    def x_limits(self):
        return self.xmin, self.xmax

    @Projection.y_limits.getter
    def y_limits(self):
        return self.ymin, self.ymax

    #   # Implement custom copy method
    # def __copy__(self):
    #     # Create a new instance with the same parameters
    #     return type(self)(
    #         epsg=self.epsg,
    #         window_extent=WindowExtent(
    #             x=[self.xmin,self.xmax],
    #             y=[self.ymin,self.ymax],
    #             )
    #       )

    # # Implement custom deepcopy method
    # def __deepcopy__(self, memo):
    #     # Create a deepcopy of each attribute and return a new instance
    #     new_extent = copy.deepcopy([self.xmin, self.xmax, self.ymin, self.ymax], memo)
    #     new_obj = type(self)(
    #         epsg=self.epsg,
    #         new_extent=new_extent
    #     )
    #     memo[id(self)] = new_obj
    #     return new_obj


class CRSfromPROJ4(Projection):

    def __init__(self, PROJ4_str, extent):
        base_crs = ccrs.Projection(PROJ4_str)
        super().__init__(base_crs.proj4_init)

        # Update the extent only if it extends beyond the defaults
        self.xmin = extent.x[0]
        self.xmax = extent.x[1]
        self.ymin = extent.y[0]
        self.ymax = extent.y[1]

    @Projection.boundary.getter
    def boundary(self):
        coords = ((self.x_limits[0], self.y_limits[0]),
                  (self.x_limits[0], self.y_limits[1]),
                  (self.x_limits[1], self.y_limits[1]),
                  (self.x_limits[1], self.y_limits[0]))
        return ccrs.sgeom.LineString(coords)

    @Projection.x_limits.getter
    def x_limits(self):
        return self.xmin, self.xmax

    @Projection.y_limits.getter
    def y_limits(self):
        return self.ymin, self.ymax


# %% wradlib georef to crs
def wradlibCRS2pyproj(ds):
    # Manually define the azimuthal equidistant projection
    radar_location = {'lon': ds.longitude.values, 'lat': ds.latitude.values}
    return f"+proj=aeqd +lat_0={radar_location['lat']} +lon_0={radar_location['lon']} +x_0=0 +y_0=0 +datum=WGS84"

# %%

def getCmapNorm(varlib):
    import matplotlib.colors as mcolors
    try:
        cmap_norm = varlib['colormap_norm']
        if isinstance(cmap_norm, list):
            norm = mcolors.Normalize(vmin=cmap_norm[0], vmax=cmap_norm[1])

        elif isinstance(cmap_norm, dict):
            norm_type = cmap_norm.get('type', 'linear')
            vmin = cmap_norm.get('vmin', None)
            vmax = cmap_norm.get('vmax', None)

            if norm_type == 'linear':
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            elif norm_type == 'log':
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            elif norm_type == 'symlog':
                linthresh = cmap_norm.get('linthresh', 1.0)
                linscale = cmap_norm.get('linscale', 1.0)
                norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax)
            elif norm_type == 'power':
                gamma = cmap_norm.get('gamma', 1.0)
                norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
            else:
                raise ValueError(f"Unsupported normalization type: {norm_type}")
    except KeyError:
        norm = None
    return norm


def getCmap(varlib):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    try:
        colormap = varlib['colormap']
        if isinstance(colormap, str):
            cmap = colormap

            norm = getCmapNorm(varlib)

        elif isinstance(colormap, dict):

            try:
                scale = colormap["scale"]
                colorlist = list(colormap)
                colorlist.remove('scale')

                c_ranges = {}
                for c in colorlist:
                    c_ranges[c] = np.arange(colormap[c][0],
                                            colormap[c][1] + scale / 2,
                                            scale)
                c_palette = np.concatenate([plt.get_cmap(c)(np.linspace(0., 1., len(c_ranges[c]))) for c in colorlist])
                boundaries = [c_ranges[c] for c in colorlist]
                boundaries = np.unique(np.concatenate(boundaries))
            except KeyError:
                c_palette = list(colormap.values())
                boundaries = np.arange(-0.5, len(c_palette))
            try:
                cmap = mcolors.ListedColormap(c_palette)
                norm = mcolors.BoundaryNorm(boundaries, cmap.N)
            except:
                raise ValueError("Colormap error. Provide valid colormap dictionary")
        else:
            raise ValueError(f"Unsupported colormap type {type(colormap)}.")

    except KeyError:
        cmap = None

    return cmap, norm

# %%#%% open an ERA5 reference file


def open_reference_file(path_reference_file):
    import metpy.calc
    with xr.open_dataset(path_reference_file) as ds:
        ds["h"] = metpy.calc.geopotential_to_height(ds.z)
        ds = ds.assign_coords(H=ds.h.mean(["longitude", "latitude", "time"]), dims='isobaricInhPa')
        return ds.swap_dims({'isobaricInhPa': 'H'})


def split_dim_to_variables(ds: xr.Dataset, dim: str = "obs") -> xr.Dataset:
    """
    Splits a dimension (e.g., 'obs') into separate variables in the dataset.

    Parameters:
    - ds (xr.Dataset): Input dataset with the dimension to split.
    - dim (str): The dimension to split into variables. Default is "obs".

    Returns:
    - xr.Dataset: Dataset with separate variables for each value in the specified dimension.
    """
    if dim not in ds.dims:
        raise ValueError(f"The dimension '{dim}' does not exist in the dataset.")
    
    # Create a new dataset with variables for each value in the dimension
    new_ds = ds.drop_dims(dim)
    for value in ds[dim].values:
        # Extract the slice corresponding to the current value
        new_ds[value] = ds.sel({dim: value}).drop_vars(dim).to_array().squeeze()
    
    return new_ds