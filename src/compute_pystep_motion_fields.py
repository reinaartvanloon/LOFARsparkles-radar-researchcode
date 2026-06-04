#!/usr/bin/env python3
"""Compute motion fields from KNMI NL25 1.5 km reflectivity composites with pySTEPS.

Reads a sequence of 5-min HDF5 composite files (product RAD_NL25_PCP_H1.5_NA),
crops to a lat/lon bounding box on the native polar-stereographic grid,
preprocesses per the standard pySTEPS recipe (dBZ -> rain rate -> dB), and
runs Lucas-Kanade optical flow over a rolling 3-frame window. All motion
fields are written to a single NetCDF file.

CRS note: the NL25 composites are in KNMI polar stereographic
    +proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378137 +b=6356752
(NOT EPSG:28992, which is Amersfoort/RD). Motion is computed on the native
1 km regular grid, so u and v are along the stereographic x- and y-axes.
Over the Netherlands these are within a few degrees of true east/north.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from pysteps import motion
from pysteps.utils import conversion, transformation

logger = logging.getLogger("motion_fields")

KNMI_STERE_PROJ = (
    "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 "
    "+a=6378137 +b=6356752 +x_0=0 +y_0=0 +type=crs"
)
NX, NY = 700, 765
DX_KM, DY_KM = 1.0, -1.0  # row 0 is north, so dy is negative
X_UL_KM, Y_UL_KM = 0.0, -3650.0

FILENAME_RE = re.compile(r"RAD_NL25_PCP_NA_(\d{12})\.h5$")


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class ConfigPystepsMotion:
    """Configuration for pySTEPS Lucas-Kanade motion field computation."""
    data_dir: str
    start: str                # ISO datetime string, e.g. "2021-06-18T17:00"
    end: str                  # ISO datetime string, e.g. "2021-06-18T21:00"
    outdir: str
    outname: Optional[str] = None  # if None → motion_{start}_{end}.nc
    step_minutes: int = 5
    lon_min: float = 4.0
    lon_max: float = 8.0
    lat_min: float = 52.0
    lat_max: float = 54.0
    window_size: int = 3
    verbose: bool = False


# ── Utility functions ──────────────────────────────────────────────────────────

def native_grid_km() -> tuple[np.ndarray, np.ndarray]:
    x = X_UL_KM + DX_KM * (np.arange(NX) + 0.5)
    y = Y_UL_KM + DY_KM * (np.arange(NY) + 0.5)
    return x, y


def parse_time(path: Path) -> dt.datetime:
    m = FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse timestamp from {path.name}")
    return dt.datetime.strptime(m.group(1), "%Y%m%d%H%M")


def read_composite_dbz(path: Path) -> np.ndarray:
    """Return calibrated reflectivity in dBZ; PV=0 (missing) and PV=255
    (out-of-image) become NaN."""
    with h5py.File(path, "r") as f:
        pv = f["image1/image_data"][:]
    dbz = 0.5 * pv.astype(np.float32) - 32.0
    dbz[(pv == 0) | (pv == 255)] = np.nan
    return dbz


def collect_files(
    data_dir: Path,
    start: dt.datetime,
    end: dt.datetime,
    step_minutes: int,
) -> list[tuple[dt.datetime, Path]]:
    out: list[tuple[dt.datetime, Path]] = []
    t = start
    while t <= end:
        p = data_dir / f"RAD_NL25_PCP_NA_{t:%Y%m%d%H%M}.h5"
        if p.exists():
            out.append((t, p))
        else:
            logger.warning("Missing composite for %s", t)
        t += dt.timedelta(minutes=step_minutes)
    return out


def crop_indices(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float
) -> tuple[slice, slice]:
    """Map a lat/lon bbox onto native stere row/col slices (tight outer bound)."""
    tr = Transformer.from_crs("EPSG:4326", KNMI_STERE_PROJ, always_xy=True)
    xs_m, ys_m = tr.transform(
        [lon_min, lon_max, lon_min, lon_max],
        [lat_min, lat_min, lat_max, lat_max],
    )
    xs_km = np.asarray(xs_m) / 1000.0
    ys_km = np.asarray(ys_m) / 1000.0
    x_c, y_c = native_grid_km()

    i0 = int(np.clip(np.searchsorted(x_c, xs_km.min()) - 1, 0, NX - 1))
    i1 = int(np.clip(np.searchsorted(x_c, xs_km.max()) + 1, 1, NX))
    y_desc = -y_c  # y_c is decreasing, so -y_c is ascending
    j0 = int(np.clip(np.searchsorted(y_desc, -ys_km.max()) - 1, 0, NY - 1))
    j1 = int(np.clip(np.searchsorted(y_desc, -ys_km.min()) + 1, 1, NY))
    return slice(j0, j1), slice(i0, i1)


def build_dataset(
    uv_list: list[np.ndarray],
    times_list: list[dt.datetime],
    x_km: np.ndarray,
    y_km: np.ndarray,
    dt_seconds: float,
    config: ConfigPystepsMotion,
) -> xr.Dataset:
    dx_m = DX_KM * 1000.0
    dy_m = DY_KM * 1000.0  # negative: converts pix/step -> stere y m/s with correct sign
    u_stack = np.stack([uv[0] * dx_m / dt_seconds for uv in uv_list], axis=0)
    v_stack = np.stack([uv[1] * dy_m / dt_seconds for uv in uv_list], axis=0)

    # Flip y to ascending (north-up CF convention)
    y_km_asc = y_km[::-1]
    u_stack  = u_stack[:, ::-1, :]
    v_stack  = v_stack[:, ::-1, :]

    tr = Transformer.from_crs(KNMI_STERE_PROJ, "EPSG:4326", always_xy=True)
    xm, ym = np.meshgrid(x_km * 1000.0, y_km_asc * 1000.0)
    lon2d, lat2d = tr.transform(xm, ym)

    crs = CRS.from_proj4(KNMI_STERE_PROJ)
    ds = xr.Dataset(
        data_vars=dict(
            u=(
                ("time", "y", "x"),
                u_stack.astype(np.float32),
                {
                    "long_name": "velocity along stereographic x-axis",
                    "units": "m s-1",
                    "grid_mapping": "stereographic",
                    "coordinates": "lon lat",
                },
            ),
            v=(
                ("time", "y", "x"),
                v_stack.astype(np.float32),
                {
                    "long_name": "velocity along stereographic y-axis",
                    "units": "m s-1",
                    "grid_mapping": "stereographic",
                    "coordinates": "lon lat",
                },
            ),
            lon=(
                ("y", "x"),
                lon2d.astype(np.float64),
                {"long_name": "longitude", "units": "degrees_east"},
            ),
            lat=(
                ("y", "x"),
                lat2d.astype(np.float64),
                {"long_name": "latitude", "units": "degrees_north"},
            ),
        ),
        coords=dict(
            time=(
                "time",
                np.array(times_list, dtype="datetime64[ns]"),
                {"long_name": "valid time (latest input frame in OF window)"},
            ),
            x=(
                "x",
                (x_km * 1000.0).astype(np.float64),
                {
                    "long_name": "stereographic x",
                    "units": "m",
                    "standard_name": "projection_x_coordinate",
                    "axis": "X",
                },
            ),
            y=(
                "y",
                (y_km_asc * 1000.0).astype(np.float64),
                {
                    "long_name": "stereographic y",
                    "units": "m",
                    "standard_name": "projection_y_coordinate",
                    "axis": "Y",
                },
            ),
        ),
    )
    ds["stereographic"] = xr.DataArray(
        np.int8(0),
        attrs={
            "grid_mapping_name": "polar_stereographic",
            "latitude_of_projection_origin": 90.0,
            "straight_vertical_longitude_from_pole": 0.0,
            "standard_parallel": 60.0,
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.0,
            "false_easting": 0.0,
            "false_northing": 0.0,
            "crs_wkt": crs.to_wkt(),
            "proj4": KNMI_STERE_PROJ,
        },
    )
    ds.attrs.update(
        {
            "title": "pySTEPS motion fields from KNMI NL25 1.5 km CAPPI composites",
            "source": "RAD_NL25_PCP_H1.5_NA reflectivity composites (KNMI)",
            "pysteps_method": "dense_lucaskanade",
            "pysteps_preprocessing": (
                "dBZ -> rain rate (Marshall-Palmer Z=200*R**1.6) -> dB transform "
                "(threshold 0.1 mm/h, zerovalue -15 dB); NaN filled with zerovalue "
                "before optical flow."
            ),
            "motion_field_definition": (
                f"Lucas-Kanade OF over a rolling window of the {config.window_size} "
                "most recent input frames. Valid time is the latest frame in the "
                "window. Units are m/s along the stereographic x- and y-axes."
            ),
            "bbox_lonlat": (
                f"{config.lon_min},{config.lat_min},{config.lon_max},{config.lat_max}"
            ),
            "note_on_crs": (
                "Native grid is KNMI polar stereographic, NOT EPSG:28992. "
                "The y-axis in this file is ascending (south -> north)."
            ),
            "Conventions": "CF-1.8",
            "created": dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
        }
    )
    return ds


# ── Main ───────────────────────────────────────────────────────────────────────

def main(config: ConfigPystepsMotion) -> None:
    logging.basicConfig(
        level=logging.INFO if config.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    
    if config.outdir==None:
        raise ValueError("Provide an outdir directory, where the results can be written.")

    start    = dt.datetime.fromisoformat(config.start)
    end      = dt.datetime.fromisoformat(config.end)
    data_dir = Path(config.data_dir)

    files = collect_files(data_dir, start, end, config.step_minutes)
    if len(files) < config.window_size:
        raise SystemExit(
            f"Only {len(files)} files found, need >= {config.window_size}"
        )
    logger.info("Found %d files between %s and %s", len(files), start, end)

    yslice, xslice = crop_indices(
        config.lon_min, config.lon_max, config.lat_min, config.lat_max
    )
    x_c, y_c = native_grid_km()
    x_km = x_c[xslice]
    y_km = y_c[yslice]
    logger.info(
        "Cropped grid: %d rows x %d cols",
        yslice.stop - yslice.start,
        xslice.stop - xslice.start,
    )

    dt_seconds = config.step_minutes * 60.0
    base_meta  = dict(
        unit="dBZ",
        transform=None,
        accutime=config.step_minutes,
        threshold=-10.0,
        zerovalue=np.nan,
        projection=KNMI_STERE_PROJ,
        xpixelsize=DX_KM * 1000.0,
        ypixelsize=abs(DY_KM) * 1000.0,
        yorigin="upper",
    )

    oflow: list[np.ndarray]    = []
    uv_list: list[np.ndarray]  = []
    times_list: list[dt.datetime] = []
    oflow_fn = motion.get_method("lucaskanade")
    buffer: list[np.ndarray]   = []

    for t, path in files:
        try:
            dbz = read_composite_dbz(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", path.name, exc)
            buffer.clear()
            continue

        dbz_crop = dbz[yslice, xslice]
        R, meta  = conversion.to_rainrate(dbz_crop, dict(base_meta))
        R, meta  = transformation.dB_transform(R, meta, threshold=0.1, zerovalue=-15.0)

        buffer.append(R)
        if len(buffer) > config.window_size:
            buffer.pop(0)
        if len(buffer) < config.window_size:
            continue

        frames = np.stack(buffer, axis=0)
        frames = np.where(np.isnan(frames), -15.0, frames)
        uv     = oflow_fn(frames, verbose=False)  # pixels per timestep
        uv_list.append(uv)
        times_list.append(t)

    if not uv_list:
        raise SystemExit("No motion fields computed (check window size and gaps)")
    logger.info("Computed %d motion fields", len(uv_list))

    ds = build_dataset(uv_list, times_list, x_km, y_km, dt_seconds, config)

    if config.output is None:
        out = str(config.out_dir / f"motion_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.nc")
    else:
        out = str(config.outdir / config.outname)

    encoding = {v: {"zlib": True, "complevel": 4} for v in ("u", "v", "lon", "lat")}
    ds.to_netcdf(out, engine="h5netcdf", encoding=encoding)
    print(f"Wrote {out} with {len(uv_list)} motion fields.")
