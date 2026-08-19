# Data Manipulation Imports
import numpy as np
import pandas as pd
import xarray as xr
import scipy
from datetime import datetime, date, time, timedelta

def round_to_year(dt):
    # If month is 7 or higher, round up; otherwise, round down
    new_year = dt.year + 1 if dt.month >= 7 else dt.year
    return datetime(new_year, 1, 1)
def find_bloom_phenological_indices(bloom_slice, var_names):
    PRES, TEMP, PSAL, DOXY, CHLA, BBP700, LATITUDE, LONGITUDE, JULD = var_names
    if len(bloom_slice.CHLA) == 0:
        phenology_dict = {
            # "REGION": region,
            "YEAR": np.nan,
            # "WMOS": bloom_slice["WMO_ID"].unique(),
            "BLOOM_MAXIMUM": np.nan,
            "BLOOM_PEAKS_VALUES": np.nan,
            "BLOOM_PEAKS_JULD": np.nan,
            "BLOOM_SLICE": np.nan,
            "BLOOM_INITIATION_TS": np.nan,
            "BLOOM_TERMINATION_TS": np.nan,
            "BLOOM_DURATION_TS": np.nan,
            "BLOOM_INITIATION_TS_DATE": np.nan,
            "BLOOM_TERMINATION_TS_DATE": np.nan,
            "BLOOM_INITIATION_CS": np.nan,
            "BLOOM_TERMINATION_CS": np.nan,
            "BLOOM_DURATION_CS": np.nan,
            "BLOOM_INITIATION_CS_DATE": np.nan,
            "BLOOM_TERMINATION_CS_DATE": np.nan,
            "BLOOM_INITIATION_RC": np.nan,
            "BLOOM_TERMINATION_RC": np.nan,
            "BLOOM_DURATION_RC": np.nan,
            "BLOOM_INITIATION_RC_DATE": np.nan,
            "BLOOM_TERMINATION_RC_DATE": np.nan,
        return pd.DataFrame([phenology_dict]).reset_index(drop=True)
    bloom_slice["RATE_OF_CHANGE"] = bloom_slice[CHLA].diff() / bloom_slice[JULD].diff().dt.days
    bloom_max_idx = bloom_slice[CHLA].idxmax()
    bloom_max_value = bloom_slice[CHLA].max()
    pre_slice = bloom_slice.loc[:bloom_max_idx]
    post_slice = bloom_slice.loc[bloom_max_idx:]
    bloom_min_idx = pre_slice[CHLA].idxmin()
    bloom_min_value = pre_slice[CHLA].min()
    bloom_range = bloom_slice[CHLA].max() - bloom_slice[CHLA].min()
    # Bloom Initiations
    bloom_initiation_TS_threshold = bloom_min_value + (0.05 * bloom_range)
    mini_slice = bloom_slice.loc[bloom_min_idx:bloom_max_idx]
    try:
        bloom_initiation_TS = mini_slice.where(mini_slice[CHLA] > bloom_initiation_TS_threshold).dropna(how="all").iloc[0]
    except IndexError:
        bloom_initiation_TS = None
    try:
        bloom_termination_TS = post_slice.where(post_slice[CHLA] < bloom_initiation_TS_threshold).dropna(how="all").iloc[0]
    except IndexError:
        bloom_termination_TS = None
    # mins = scipy.signal.argrelextrema(bloom_slice["CHLA_ADJUSTED_BGCArgoPlus"].values, np.less, order=2)
    mini_slice = bloom_slice.loc[bloom_min_idx:]
    mini_slice_filtered = mini_slice[mini_slice[CHLA] < 3*np.median(bloom_slice[CHLA])]
    chla_cumsum = np.cumsum(mini_slice_filtered[CHLA]).values[-1]
    try:
        bloom_initiation_CS = mini_slice.where(mini_slice[CHLA] > (chla_cumsum * 0.1)).dropna(how="all").iloc[0]
    except IndexError:
        bloom_initiation_CS = None
    try:
        bloom_termination_CS = post_slice.where(post_slice[CHLA] < (chla_cumsum * 0.1))[1:].dropna(how="all").iloc[0]
    except IndexError:
        bloom_termination_CS = None
    rate_of_change_threshold = 0.15 * np.nanmedian(np.abs(bloom_slice["RATE_OF_CHANGE"]))
    try:
        bloom_initiation_RC = pre_slice.where(pre_slice["RATE_OF_CHANGE"] > rate_of_change_threshold).dropna(how="all").iloc[0]
    except IndexError:
        bloom_initiation_RC = None
    try:
        bloom_termination_RC = post_slice.where(post_slice["RATE_OF_CHANGE"] > -rate_of_change_threshold)[1:].dropna(how="all").iloc[0]
    except IndexError:
        bloom_termination_RC = None
    peaks, _ = scipy.signal.find_peaks(bloom_slice[CHLA], height=bloom_max_value * 0.75, distance = 3)

    phenology_dict = {
        # "REGION": region,
        "YEAR": round_to_year(bloom_slice[JULD].loc[bloom_max_idx]).year,
        # "WMOS": bloom_slice["WMO_ID"].unique(),
        "BLOOM_MAXIMUM": bloom_max_value,
        "BLOOM_PEAKS_VALUES": bloom_slice[CHLA].iloc[peaks].values,
        "BLOOM_PEAKS_JULD": bloom_slice[JULD].iloc[peaks].values,
        "BLOOM_SLICE": [bloom_slice],
        "BLOOM_INITIATION_TS": bloom_initiation_TS[CHLA] if bloom_initiation_TS is not None else None,
        "BLOOM_TERMINATION_TS": bloom_termination_TS[CHLA] if bloom_termination_TS is not None else None,
        "BLOOM_DURATION_TS": (bloom_termination_TS[JULD] - bloom_initiation_TS["JULD"]).days if bloom_initiation_TS is not None and bloom_termination_TS is not None else None,
        "BLOOM_INITIATION_TS_DATE": bloom_initiation_TS[JULD] if bloom_initiation_TS is not None else None,
        "BLOOM_TERMINATION_TS_DATE": bloom_termination_TS[JULD] if bloom_termination_TS is not None else None,
        "BLOOM_INITIATION_CS": bloom_initiation_CS[CHLA] if bloom_initiation_CS is not None else None,
        "BLOOM_TERMINATION_CS": bloom_termination_CS[CHLA] if bloom_termination_CS is not None else None,
        "BLOOM_DURATION_CS": (bloom_termination_CS[JULD] - bloom_initiation_CS["JULD"]).days if bloom_initiation_CS is not None and bloom_termination_CS is not None else None,
        "BLOOM_INITIATION_CS_DATE": bloom_initiation_CS[JULD] if bloom_initiation_CS is not None else None,
        "BLOOM_TERMINATION_CS_DATE": bloom_termination_CS[JULD] if bloom_termination_CS is not None else None,
        "BLOOM_INITIATION_RC": bloom_initiation_RC[CHLA] if bloom_initiation_RC is not None else None,
        "BLOOM_TERMINATION_RC": bloom_termination_RC[CHLA] if bloom_termination_RC is not None else None,
        "BLOOM_DURATION_RC": (bloom_termination_RC[JULD] - bloom_initiation_RC["JULD"]).days if bloom_initiation_RC is not None and bloom_termination_RC is not None else None,
        "BLOOM_INITIATION_RC_DATE": bloom_initiation_RC[JULD] if bloom_initiation_RC is not None else None,
        "BLOOM_TERMINATION_RC_DATE": bloom_termination_RC[JULD] if bloom_termination_RC is not None else None,
    }
    
    return pd.DataFrame([phenology_dict]).reset_index(drop=True)