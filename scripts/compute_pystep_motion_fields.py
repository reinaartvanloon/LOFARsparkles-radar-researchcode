#!/usr/bin/env python3
"""Compute pySTEPS Lucas-Kanade motion fields from KNMI NL25 composites."""

import os
import sys
import argparse

here = os.path.abspath(os.path.dirname(__file__))
src  = os.path.abspath(os.path.join(here, "..", "src"))
for p in (here, src):
    if p not in sys.path:
        sys.path.insert(0, p)

from compute_pystep_motion_fields import main, ConfigPystepsMotion

ap = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
ap.add_argument(
    "--data-dir",
    default="/media/reinaart/KINGSTON/backup_knmi/data/radar/21C/"
    "reflectivity_1500m_composite",
    help="Directory containing RAD_NL25_PCP_NA_*.h5 files.",
)
ap.add_argument("--start",        default="2021-06-18T17:00", help="ISO start datetime.")
ap.add_argument("--end",          default="2021-06-18T21:00", help="ISO end datetime.")
ap.add_argument("--step-minutes", type=int,   default=5)
ap.add_argument("--lon-min",      type=float, default=4.0)
ap.add_argument("--lon-max",      type=float, default=8.0)
ap.add_argument("--lat-min",      type=float, default=52.0)
ap.add_argument("--lat-max",      type=float, default=54.0)
ap.add_argument("--window-size",  type=int,   default=3)
ap.add_argument(
    "--output", default=None,
    help="Output NetCDF path. Defaults to "
         "data_additional/motion_{start}_{end}.nc",
)
ap.add_argument("-v", "--verbose", action="store_true")
args = ap.parse_args()

config = ConfigPystepsMotion(
    data_dir     = args.data_dir,
    start        = args.start,
    end          = args.end,
    output       = args.output,
    step_minutes = args.step_minutes,
    lon_min      = args.lon_min,
    lon_max      = args.lon_max,
    lat_min      = args.lat_min,
    lat_max      = args.lat_max,
    window_size  = args.window_size,
    verbose      = args.verbose,
)

main(config)
