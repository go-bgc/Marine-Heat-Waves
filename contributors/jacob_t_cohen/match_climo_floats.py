"""
Script for matching the interpolated climatologies to interpolated float data based
on latitude, longitude, and month of year.

Jacob T. Cohen, August 19, 2026
"""

import numpy as np
import pandas as pd
import xarray as xr
import datetime


def match_climo_floats(da_floats, da_climo):
        
