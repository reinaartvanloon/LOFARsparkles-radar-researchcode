#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create Skew-T/Hodograph diagnostics from sounding and ERA5 profiles."""

from __future__ import annotations
"""Code mostly reused from Metpy cookbook example: Advanced Sounding Plot with Complex Layout"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.transforms import Bbox
import matplotlib as mpl

import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT, add_metpy_logo
from metpy.units import units

from general import ConfigPlot, outpath_gen

mpl.rcParams["font.size"] = 16


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConfigSounding:
    """Configuration for loading a radiosonde sounding CSV."""
    filepath: str


@dataclass
class ConfigERA5:
    """Configuration for loading an ERA5 GRIB profile."""
    filepath: str
    time: Any  # str or pd.Timestamp
    latitude: float
    longitude: float


@dataclass
class ConfigSkewT(ConfigPlot):
    """Configuration for the Skew-T / Hodograph figure."""
    max_altitude_km: float = 12.0
    xlims_skewt: Optional[Tuple[float, float]] = None
    title: Optional[str] = None
    show: bool = True
    outdir: str = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sorted_by_pressure_desc(*arrays: Any) -> list[Any]:
    """Return arrays sorted with highest pressure first."""
    p = arrays[0]
    order = np.argsort(p.magnitude)[::-1]
    return [arr[order] for arr in arrays]


def _altitude_km_to_pressure_hpa(altitude_km: float) -> float:
    """Convert altitude (km) to pressure (hPa) using standard atmosphere."""
    alt = altitude_km * units.km
    press = mpcalc.height_to_pressure_std(alt)
    return press.to("hPa").magnitude


def _clean_profile(p: Any, z: Any, t: Any, td: Any, u: Any, v: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Remove non-finite values across profile fields."""
    mask = (
        np.isfinite(p.magnitude)
        & np.isfinite(z.magnitude)
        & np.isfinite(t.magnitude)
        & np.isfinite(td.magnitude)
        & np.isfinite(u.magnitude)
        & np.isfinite(v.magnitude)
    )
    return p[mask], z[mask], t[mask], td[mask], u[mask], v[mask]


def _subset_for_max_pressure(
    p: Any,
    z: Any,
    t: Any,
    td: Any,
    u: Any,
    v: Any,
    max_pressure_hpa: float,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Subset profile fields to points above a given pressure (lower pressure = higher altitude)."""
    max_pressure = max_pressure_hpa * units.hPa
    mask = p >= max_pressure
    p_sub, z_sub, t_sub, td_sub, u_sub, v_sub = p[mask], z[mask], t[mask], td[mask], u[mask], v[mask]
    if p_sub.size < 6:
        raise ValueError(f"Too few profile levels above {max_pressure_hpa} hPa for plotting.")
    return p_sub, z_sub, t_sub, td_sub, u_sub, v_sub

def to_floats(data):
    return pd.to_numeric(data, errors='coerce')


def _build_sounding_profile(config: ConfigSounding) -> dict[str, Any]:
    """Load the sounding CSV and prepare profile arrays."""
    col_names = [
        "time",
        "longitude",
        "latitude",
        "pressure",
        "geopotential_height",
        "temperature",
        "dewpoint",
        "temperature_ice",
        "relative_humidity",
        "humidity_ice",
        "q",
        "direction",
        "speed",
    ]

    sounding_data = pd.read_csv(config.filepath, skiprows=1, names=col_names)
    sounding_data = sounding_data.dropna(
        subset=("pressure", "geopotential_height", "temperature", "dewpoint", "direction", "speed"),
        how="any",
    ).reset_index(drop=True)

    
    p = to_floats(sounding_data["pressure"]).values * units.hPa
    z = to_floats(sounding_data["geopotential_height"]).values * units.m
    t = to_floats(sounding_data["temperature"]).values * units.degC
    td = to_floats(sounding_data["dewpoint"]).values * units.degC
    wind_speed = to_floats(sounding_data["speed"]).values * units.m / units.s
    wind_dir = to_floats(sounding_data["direction"]).values * units.degrees
    u, v = mpcalc.wind_components(wind_speed, wind_dir)

    p, z, t, td, u, v = _clean_profile(p, z, t, td, u, v)
    p, z, t, td, u, v = _sorted_by_pressure_desc(p, z, t, td, u, v)

    sounding_time = sounding_data["time"].dropna().iloc[0] if not sounding_data["time"].dropna().empty else None
    return {"p": p, "z": z, "t": t, "td": td, "u": u, "v": v, "time": sounding_time}


def _build_era5_profile(config: ConfigERA5) -> dict[str, Any]:
    """Interpolate ERA5 profile and derive dewpoint and profile heights."""
    time = pd.to_datetime(config.time)
    with xr.open_dataset(config.filepath) as ds:
        p = ds.isobaricInhPa.values * units.hPa

        t = ds.t.interp(time=time, longitude=config.longitude, latitude=config.latitude).values * units.kelvin
        hum = ds.r.interp(time=time, longitude=config.longitude, latitude=config.latitude).values * units.percent
        td = mpcalc.dewpoint_from_relative_humidity(t, hum)
        u = ds.u.interp(time=time, longitude=config.longitude, latitude=config.latitude).values * units.m / units.s
        v = ds.v.interp(time=time, longitude=config.longitude, latitude=config.latitude).values * units.m / units.s

        print("ERA5 variables:" + ", ".join(ds.data_vars))
        if "z" in ds:
            geopotential = ds.z.interp(time=time, longitude=config.longitude, latitude=config.latitude).values * (
                units.meter ** 2 / units.second ** 2
            )
            z = mpcalc.geopotential_to_height(geopotential)
        else:
            z = mpcalc.pressure_to_height_std(p)

    p, z, t, td, u, v = _clean_profile(p, z, t.to("degC"), td.to("degC"), u, v)
    p, z, t, td, u, v = _sorted_by_pressure_desc(p, z, t, td, u, v)
    return {"p": p, "z": z, "t": t, "td": td, "u": u, "v": v, "time": time}


def _format_param(value: Any) -> str:
    """Format diagnostic values while handling NaNs gracefully."""
    try:
        if hasattr(value, "magnitude"):
            if np.ndim(value.magnitude) > 0:
                val = value[0]
            else:
                val = value
            if np.isfinite(val.m):
                return f"{val:.0f~P}"
            return "n/a"
        if np.isfinite(float(value)):
            return f"{float(value):.0f}"
    except Exception:
        return "n/a"
    return "n/a"


def _format_time_for_title(value: Any) -> str:
    """Format different time representations into a compact title string."""
    if value is None:
        return "TIME UNKNOWN"
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def _safe_calc(calc_fn: Any, default: Any, *args: Any, **kwargs: Any) -> Any:
    """Run diagnostics while preserving plotting when one metric fails."""
    try:
        return calc_fn(*args, **kwargs)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def create_skewt_figure(
    profile: dict[str, Any],
    config: ConfigSkewT,
) -> plt.Figure:
    """Build a diagnostics-style Skew-T and Hodograph figure."""
    p_all = profile["p"]
    z_all = profile["z"]
    t_all = profile["t"]
    td_all = profile["td"]
    u_all = profile["u"]
    v_all = profile["v"]

    max_pressure_hpa = _altitude_km_to_pressure_hpa(config.max_altitude_km)
    p, z, t, td, u, v = _subset_for_max_pressure(p_all, z_all, t_all, td_all, u_all, v_all, max_pressure_hpa)
    p_all, z_all, t_all, td_all, u_all, v_all = p, z, t, td, u, v

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(
        nrows=12,
        ncols=40,
        left=0.05,
        right=0.98,
        bottom=0.05,
        top=0.95,
        wspace=0.1,
        hspace=0.050,
    )
    skew = SkewT(fig, rotation=45, subplot=gs[:, 1:25])
    skew.ax.set_ylim(p.to("hPa").magnitude.max(), max_pressure_hpa)
    ax2 = fig.add_subplot(gs[:, 26], sharey=skew.ax)
    hodo_ax = fig.add_subplot(gs[0:6, 30:40])
    ax_diag = fig.add_subplot(gs[6:11, 30:40])

    add_metpy_logo(fig, 200, 200, size="large")

    if config.xlims_skewt is not None:
        skew.ax.set_xlim(config.xlims_skewt[0], config.xlims_skewt[1])

    skew.ax.set_xlabel(f"Temperature [{t.units:~P}]", weight="bold")
    skew.ax.set_ylabel(f"Pressure [{p.units:~P}]", weight="bold")

    fig.set_facecolor("#ffffff")
    skew.ax.set_facecolor("#ffffff")

    x1 = np.linspace(-100, 40, 8)
    x2 = np.linspace(-90, 50, 8)
    y = [1100, 50]
    for i in range(8):
        skew.shade_area(y=y, x1=x1[i], x2=x2[i], color="gray", alpha=0.02, zorder=1)

    skew.plot(p, t, "r", lw=4, label="Temperature")
    skew.plot(p, td, "g", lw=4, label="Dewpoint")

    p_low_hpa = np.nanmin(p.magnitude)
    p_high_hpa = np.nanmax(p.magnitude)
    interval = np.arange(
        np.floor(p_low_hpa / 50.0) * 50.0,
        np.ceil(p_high_hpa / 50.0) * 50.0 + 1.0,
        50.0,
    ) * units.hPa
    idx = mpcalc.resample_nn_1d(p, interval)
    skew.plot_barbs(
        pressure=p[idx],
        u=u[idx],
        v=v[idx],
        x_clip_radius=0.25,
        length=7,
        linewidth=1.6,
    )

    skew.ax.axvline(0 * units.degC, linestyle="--", color="blue", alpha=0.3)
    skew.plot_dry_adiabats(lw=1, alpha=0.3)
    skew.plot_mixing_lines(lw=1, alpha=0.3)

    lcl_pressure, lcl_temperature = mpcalc.lcl(p_all[0], t_all[0], td_all[0])
    skew.plot(lcl_pressure, lcl_temperature, "ko", markerfacecolor="black")

    prof_all = mpcalc.parcel_profile(p_all, t_all[0], td_all[0]).to("degC")
    prof = prof_all[np.isin(p_all.magnitude, p.magnitude)]
    skew.plot(p, prof, "k", linewidth=2, label="SB parcel path")

    # skew.shade_cin(p, t, prof, td, alpha=0.2, label="SBCIN")
    skew.shade_cape(p, t, prof, alpha=0.2, label="SB cape")

    wind_mag = np.sqrt(u.magnitude**2 + v.magnitude**2)
    max_wind = float(np.nanmax(wind_mag)) if wind_mag.size else 20.0
    component_range = max(20.0, np.ceil((max_wind * 1.15) / 10.0) * 10.0)
    x_range = max(20.0, np.ceil((float(np.nanmax(u.magnitude) * 1.15) / 10.0) * 10.0))
    h = Hodograph(hodo_ax, component_range=component_range)
    h.add_grid(increment=10, ls="-", lw=1.5, alpha=0.5)
    h.add_grid(increment=5, ls="--", lw=1, alpha=0.2)

    # Show only the top half of the hodograph (positive v / northward component)
    h.ax.set_xlim(-x_range, x_range)
    h.ax.set_ylim(-0.1*component_range, component_range)
    h.ax.set_aspect("equal")
    h.ax.set_facecolor("white")
    h.ax.set_yticklabels([])
    h.ax.set_xticklabels([])
    h.ax.set_xticks([])
    h.ax.set_yticks([])
    h.ax.set_xlabel(" ")
    h.ax.set_ylabel(" ")

    ring_max = int(np.ceil(component_range / 10.0) * 10)
    for i in range(10, ring_max + 1, 10):
        h.ax.annotate(
            str(i),
            (i, 0),
            xytext=(0, 2),
            textcoords="offset pixels",
            clip_on=True,
            fontsize=10,
            weight="bold",
            alpha=0.3,
            zorder=0,
        )
    for i in range(10, ring_max + 1, 10):
        h.ax.annotate(
            str(i),
            (0, i),
            xytext=(0, 2),
            textcoords="offset pixels",
            clip_on=True,
            fontsize=10,
            weight="bold",
            alpha=0.3,
            zorder=0,
        )

    z_km = z.to("km").magnitude
    zmin = float(np.nanmin(z_km))
    zmax = float(np.nanmax(z_km))
    h.plot_colormapped(
        u,
        v,
        c=z_km,
        linewidth=6,
        cmap="viridis",
        norm=plt.Normalize(zmin, zmax),
        label=f"0-{config.max_altitude_km:g}km WIND",
    )


    color_strip = np.linspace(zmin, zmax, 256)[:, np.newaxis]
    ax2.imshow(
        color_strip,
        cmap="viridis",
        vmin=zmin,
        vmax=zmax,
        aspect="auto",
        extent=(0, 1, p.to("hPa").magnitude.max(), max_pressure_hpa),
        origin="lower",
    )

    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax2_alt = ax2.twinx()
    ax2_alt.set_yscale("log")
    ax2.set_xticks([])
    ax2.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)

    altitude_ticks = np.arange(0, z.to("km").magnitude.max(), 1)
    pressure_ticks = np.exp(np.interp(altitude_ticks * 1000, z.to("m").magnitude, np.log(p.to("hPa").magnitude)))
    ax2_alt.set_ylim(skew.ax.get_ylim())
    ax2_alt.set_yticks(pressure_ticks)
    ax2_alt.set_yticklabels([f"{zz:.0f}" for zz in altitude_ticks])
    ax2_alt.yaxis.set_minor_locator(plt.NullLocator())
    ax2_alt.set_ylabel("Altitude [km]", weight="bold")
    ax2_alt.tick_params(axis="y", which="both", direction="out")
    ax2_alt.spines["left"].set_visible(False)

    skew_pos = skew.ax.get_position()
    strip_pos = ax2.get_position()
    aligned_pos = Bbox.from_extents(strip_pos.x0, skew_pos.y0, strip_pos.x1, skew_pos.y1)
    ax2.set_position(aligned_pos)
    ax2_alt.set_position(aligned_pos)

    kindex = _safe_calc(mpcalc.k_index, np.nan * units.delta_degree_Celsius, p_all, t_all, td_all)
    total_totals = _safe_calc(mpcalc.total_totals_index, np.nan * units.dimensionless, p_all, t_all, td_all)

    mlcape, mlcin = _safe_calc(
        mpcalc.mixed_layer_cape_cin,
        (np.nan * units.joule / units.kilogram, np.nan * units.joule / units.kilogram),
        p_all,
        t_all,
        td_all,
        depth=50 * units.hPa,
    )

    mucape, mucin = _safe_calc(
        mpcalc.most_unstable_cape_cin,
        (np.nan * units.joule / units.kilogram, np.nan * units.joule / units.kilogram),
        p_all,
        t_all,
        td_all,
        depth=50 * units.hPa,
    )

    new_p = np.append(p_all[p_all > lcl_pressure], lcl_pressure)
    new_t = np.append(t_all[p_all > lcl_pressure], lcl_temperature)
    lcl_height = _safe_calc(mpcalc.thickness_hydrostatic, np.nan * units.m, new_p, new_t)

    sbcape, sbcin = _safe_calc(
        mpcalc.surface_based_cape_cin,
        (np.nan * units.joule / units.kilogram, np.nan * units.joule / units.kilogram),
        p_all,
        t_all,
        td_all,
    )

    (u_storm, v_storm), *_ = _safe_calc(
        mpcalc.bunkers_storm_motion,
        ((np.nan * units.meter / units.second, np.nan * units.meter / units.second), None, None),
        p_all,
        u_all,
        v_all,
        z_all,
    )
    *_, total_helicity1 = _safe_calc(
        mpcalc.storm_relative_helicity,
        (None, None, np.nan * units.meter ** 2 / units.second ** 2),
        z_all,
        u_all,
        v_all,
        depth=1 * units.km,
        storm_u=u_storm,
        storm_v=v_storm,
    )
    *_, total_helicity3 = _safe_calc(
        mpcalc.storm_relative_helicity,
        (None, None, np.nan * units.meter ** 2 / units.second ** 2),
        z_all,
        u_all,
        v_all,
        depth=3 * units.km,
        storm_u=u_storm,
        storm_v=v_storm,
    )
    *_, total_helicity6 = _safe_calc(
        mpcalc.storm_relative_helicity,
        (None, None, np.nan * units.meter ** 2 / units.second ** 2),
        z_all,
        u_all,
        v_all,
        depth=6 * units.km,
        storm_u=u_storm,
        storm_v=v_storm,
    )

    ubshr1, vbshr1 = _safe_calc(
        mpcalc.bulk_shear,
        (np.nan * units.meter / units.second, np.nan * units.meter / units.second),
        p_all,
        u_all,
        v_all,
        height=z_all,
        depth=1 * units.km,
    )
    bshear1 = mpcalc.wind_speed(ubshr1, vbshr1)
    ubshr3, vbshr3 = _safe_calc(
        mpcalc.bulk_shear,
        (np.nan * units.meter / units.second, np.nan * units.meter / units.second),
        p_all,
        u_all,
        v_all,
        height=z_all,
        depth=3 * units.km,
    )
    bshear3 = mpcalc.wind_speed(ubshr3, vbshr3)
    ubshr6, vbshr6 = _safe_calc(
        mpcalc.bulk_shear,
        (np.nan * units.meter / units.second, np.nan * units.meter / units.second),
        p_all,
        u_all,
        v_all,
        height=z_all,
        depth=6 * units.km,
    )
    bshear6 = mpcalc.wind_speed(ubshr6, vbshr6)

    sig_tor = _safe_calc(
        mpcalc.significant_tornado,
        np.array([np.nan]) * units.dimensionless,
        sbcape,
        lcl_height,
        total_helicity3,
        bshear3,
    ).to_base_units()
    super_comp = _safe_calc(
        mpcalc.supercell_composite,
        np.array([np.nan]) * units.dimensionless,
        mucape,
        total_helicity3,
        bshear3,
    )

    ax_diag.set_facecolor("white")
    ax_diag.set_xticks([])
    ax_diag.set_yticks([])
    for spine in ax_diag.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    metrics = [
        ("SB CAPE", _format_param(sbcape), "orangered"),
        ("SB CIN", _format_param(sbcin), "lightblue"),
        ("0-1km SRH", _format_param(total_helicity1), "navy"),
        ("0-1km SHEAR", _format_param(bshear1), "blue"),
        ("0-3km SRH", _format_param(total_helicity3), "navy"),
        ("0-3km SHEAR", _format_param(bshear3), "blue"),
        ("0-6km SRH", _format_param(total_helicity6), "navy"),
        ("0-6km SHEAR", _format_param(bshear6), "blue"),
    ]

    fontsize = 16
    y_positions = np.linspace(0.92, 0.05, len(metrics))
    for y_pos, (name, value, color) in zip(y_positions, metrics):
        ax_diag.text(0.05, y_pos, f"{name}:", transform=ax_diag.transAxes,
                     ha="left", va="center", fontsize=fontsize, weight="bold")
        ax_diag.text(0.95, y_pos, value, transform=ax_diag.transAxes,
                     ha="right", va="center", fontsize=fontsize, weight="bold", color=color)

    skew.ax.legend(loc="center left")

    skew.ax.set_title("(a)", fontweight="bold", loc="left")
    hodo_ax.set_title(f"(b) Hodograph 0-{config.max_altitude_km:g} km", fontweight="bold", loc="left")
    ax_diag.set_title("(c)", fontweight="bold", loc="left")

    title = config.title if config.title is not None else _format_time_for_title(profile.get("time"))
    fig.suptitle(f"{title} | MAX ALT {config.max_altitude_km:.1f} KM", fontsize=18, fontweight="bold", y=0.985)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_skewt(
    config_data: ConfigSounding | ConfigERA5,
    config_skewt: ConfigSkewT,
) -> plt.Figure:
    """Load profile data and produce a Skew-T / Hodograph figure.

    Parameters
    ----------
    config_data:
        Either a :class:`ConfigSounding` or :class:`ConfigERA5` instance
        describing the data source.
    config_skewt:
        A :class:`ConfigSkewT` instance with figure settings.

    Returns
    -------
    plt.Figure
    """
    if isinstance(config_data, ConfigSounding):
        profile = _build_sounding_profile(config_data)
        if config_skewt.title is None:
            config_skewt = ConfigSkewT(
                max_altitude_km=config_skewt.max_altitude_km,
                xlims_skewt=config_skewt.xlims_skewt,
                title=f"SOUNDING PROFILE | {_format_time_for_title(profile['time'])}",
                outdir=config_skewt.outdir,
                outname=config_skewt.outname,
                save=config_skewt.save,
                show=config_skewt.show,
            )
    elif isinstance(config_data, ConfigERA5):
        profile = _build_era5_profile(config_data)
        if config_skewt.title is None:
            time_str = _format_time_for_title(config_data.time)
            config_skewt = ConfigSkewT(
                max_altitude_km=config_skewt.max_altitude_km,
                xlims_skewt=config_skewt.xlims_skewt,
                title=f"ERA5 PROFILE | {time_str} | {config_data.latitude:.2f}N, {config_data.longitude:.2f}E",
                outdir=config_skewt.outdir,
                outname=config_skewt.outname,
                save=config_skewt.save,
                show=config_skewt.show,
            )
    else:
        raise TypeError(f"config_data must be ConfigSounding or ConfigERA5, got {type(config_data)}")

    fig = create_skewt_figure(profile, config_skewt)

    if config_skewt.save == True:
        outpath = outpath_gen("/tmp",config_skewt.outdir, config_skewt.outname)
        fig.savefig(outpath)
        print("File is saved to: " + outpath + ".png")
        
    if config_skewt.show:
        plt.show()
    else:
        plt.close(fig)

    return fig
