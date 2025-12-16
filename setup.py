#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 13:22:55 2025

@author: rvloon
"""

from setuptools import setup, find_packages

setup(
    name="LOFAR_sparkles",                      # change to your chosen package name
    version="0.1.0",
    description="LOFAR Sparkles radar analysis tools (supporting ACP paper)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Reinaart van Loon",
    author_email="reinaart.vanloon@wur.nl",
    url="https://github.com/reinaartvanloon/LOFARsparkles-radar-researchcode",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "xarray",
        "matplotlib",
        "netCDF4",
        "wradlib",            
        "cartopy",
        "geopy",
        "geopandas",
        "dataclasses_json",
        "scikit-learn",
        "metpy",
        "cfgrib",
        "datetime",
        "bottleneck"
    ],
    extras_require={
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    entry_points={
    },
)