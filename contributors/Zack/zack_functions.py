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
def find_bloom_phenological_indices(bloom_slice, var_names, what_type):
    PRES, TEMP, PSAL, DOXY, CHLA, BBP700, LATITUDE, LONGITUDE, JULD = var_names
    # 1 - chla
    # 2 - bbp700
    if what_type == 1:
        desired_var = CHLA
    elif what_type == 2:
        desired_var = BBP700
    if len(bloom_slice.desired_var) == 0:
        phenology_dict = {
            # "REGION": region,
            f"YEAR_{desired_var}": np.nan,
            # "WMOS": bloom_slice["WMO_ID"].unique(),
            f"BLOOM_MAXIMUM_{desired_var}": np.nan,
            f"BLOOM_PEAKS_VALUES_{desired_var}": np.nan,
            f"BLOOM_PEAKS_JULD_{desired_var}": np.nan,
            f"BLOOM_SLICE_{desired_var}": np.nan,
            f"BLOOM_INITIATION_TS_{desired_var}": np.nan,
            f"BLOOM_TERMINATION_TS_{desired_var}": np.nan,
            f"BLOOM_DURATION_TS_{desired_var}": np.nan,
            f"BLOOM_INITIATION_TS_DATE_{desired_var}": np.nan,
            f"BLOOM_TERMINATION_TS_DATE_{desired_var}": np.nan,
            f"BLOOM_INITIATION_CS_{desired_var}": np.nan,
            f"BLOOM_TERMINATION_CS_{desired_var}": np.nan,
            f"BLOOM_DURATION_CS_{desired_var}": np.nan,
            f"BLOOM_INITIATION_CS_DATE_{desired_var}": np.nan,
            f"BLOOM_TERMINATION_CS_DATE_{desired_var}": np.nan,
            f"BLOOM_INITIATION_RC_{desired_var}": np.nan,
            f"BLOOM_TERMINATION_RC_{desired_var}": np.nan,
            f"BLOOM_DURATION_RC_{desired_var}": np.nan,
            f"BLOOM_INITIATION_RC_DATE_{desired_var}": np.nan,
            f"BLOOM_TERMINATION_RC_DATE_{desired_var}": np.nan,
            f"INTEGRATED_BLOOM_TS_{desired_var}": np.nan,
            f"INTEGRATED_BLOOM_CS_{desired_var}": np.nan,
            f"INTEGRATED_BLOOM_RC_{desired_var}": np.nan,
            f"MEAN_BLOOM_TS_{desired_var}": np.nan,
            f"MEAN_BLOOM_CS_{desired_var}": np.nan,
            f"MEAN_BLOOM_RC_{desired_var}": np.nan,
        }
        return pd.DataFrame([phenology_dict]).reset_index(drop=True)
    bloom_slice["RATE_OF_CHANGE"] = bloom_slice[desired_var].diff() / bloom_slice[JULD].diff().dt.days
    bloom_max_idx = bloom_slice[desired_var].idxmax()
    bloom_max_value = bloom_slice[desired_var].max()
    pre_slice = bloom_slice.loc[:bloom_max_idx]
    post_slice = bloom_slice.loc[bloom_max_idx:]
    bloom_min_idx = pre_slice[desired_var].idxmin()
    bloom_min_value = pre_slice[desired_var].min()
    bloom_range = bloom_slice[desired_var].max() - bloom_slice[desired_var].min()
    # Bloom Initiations
    bloom_initiation_TS_threshold = bloom_min_value + (0.05 * bloom_range)
    mini_slice = bloom_slice.loc[bloom_min_idx:bloom_max_idx]
    try:
        bloom_initiation_TS = mini_slice.where(mini_slice[desired_var] > bloom_initiation_TS_threshold).dropna(how="all").iloc[0]
    except IndexError:
        bloom_initiation_TS = None
    try:
        bloom_termination_TS = post_slice.where(post_slice[desired_var] < bloom_initiation_TS_threshold).dropna(how="all").iloc[0]
    except IndexError:
        bloom_termination_TS = None
    try:
        start_idx = bloom_slice.index[bloom_slice[JULD] == bloom_initiation_TS[JULD]][0]
        end_idx = bloom_slice.index[bloom_slice[JULD] == bloom_termination_TS[JULD]][0]
        TS_slice = bloom_slice.loc[start_idx:end_idx]
        mean_bloom_TS = np.nanmean(TS_slice[CHLA])
        integrated_bloom_TS = np.trapezoid(y=TS_slice[CHLA].values, x=TS_slice[JULD].astype("int64")/8.64e+13) # days
    except Exception as e:
        mean_bloom_TS = None
        integrated_bloom_TS = None
    # mins = scipy.signal.argrelextrema(bloom_slice["CHLA_ADJUSTED_BGCArgoPlus"].values, np.less, order=2)
    mini_slice = bloom_slice.loc[bloom_min_idx:]
    mini_slice_filtered = mini_slice[mini_slice[desired_var] < 3*np.median(bloom_slice[desired_var])]
    chla_cumsum = np.cumsum(mini_slice_filtered[desired_var]).values[-1]
    try:
        bloom_initiation_CS = mini_slice.where(mini_slice[desired_var] > (chla_cumsum * 0.1)).dropna(how="all").iloc[0]
    except IndexError:
        bloom_initiation_CS = None
    try:
        bloom_termination_CS = post_slice.where(post_slice[desired_var] < (chla_cumsum * 0.1))[1:].dropna(how="all").iloc[0]
    except IndexError:
        bloom_termination_CS = None
    try:
        start_idx = bloom_slice.index[bloom_slice[JULD] == bloom_initiation_CS[JULD]][0]
        end_idx = bloom_slice.index[bloom_slice[JULD] == bloom_termination_CS[JULD]][0]
        CS_slice = bloom_slice.loc[start_idx:end_idx]
        mean_bloom_CS = np.nanmean(CS_slice[CHLA])
        integrated_bloom_CS = np.trapezoid(y=CS_slice[CHLA].values, x=CS_slice[JULD].astype("int64")/8.64e+13) # days
    except Exception as e:
        mean_bloom_CS = None
        integrated_bloom_CS = None
        
    rate_of_change_threshold = 0.15 * np.nanmedian(np.abs(bloom_slice["RATE_OF_CHANGE"]))
    try:
        bloom_initiation_RC = pre_slice.where(pre_slice["RATE_OF_CHANGE"] > rate_of_change_threshold).dropna(how="all").iloc[0]
    except IndexError:
        bloom_initiation_RC = None
    try:
        bloom_termination_RC = post_slice.where(post_slice["RATE_OF_CHANGE"] > -rate_of_change_threshold)[1:].dropna(how="all").iloc[0]
    except IndexError:
        bloom_termination_RC = None
        df1.loc['a'] > 0
    try:
        start_idx = bloom_slice.index[bloom_slice[JULD] == bloom_initiation_RC[JULD]][0]
        end_idx = bloom_slice.index[bloom_slice[JULD] == bloom_termination_RC[JULD]][0]
        RC_slice = bloom_slice.loc[start_idx:end_idx]
        mean_bloom_RC = np.nanmean(RC_slice[CHLA])
        integrated_bloom_RC = np.trapezoid(y=RC_slice[CHLA].values, x=RC_slice[JULD].astype("int64")/8.64e+13) # days
    except Exception as e:
        mean_bloom_RC = None
        integrated_bloom_RC = None

    peaks, _ = scipy.signal.find_peaks(bloom_slice[desired_var], height=bloom_max_value * 0.75, distance = 3)
    phenology_dict = {
        # "REGION": region,
        f"YEAR_{desired_var}": round_to_year(bloom_slice[JULD].loc[bloom_max_idx]).year,
        # "WMOS": bloom_slice["WMO_ID"].unique(),
        f"BLOOM_MAXIMUM_{desired_var}": bloom_max_value,
        f"BLOOM_PEAKS_VALUES_{desired_var}": bloom_slice[desired_var].iloc[peaks].values,
        f"BLOOM_PEAKS_JULD_{desired_var}": bloom_slice[JULD].iloc[peaks].values,
        f"BLOOM_SLICE_{desired_var}": [bloom_slice],
        f"BLOOM_INITIATION_TS_{desired_var}": bloom_initiation_TS[desired_var] if bloom_initiation_TS is not None else None,
        f"BLOOM_TERMINATION_TS_{desired_var}": bloom_termination_TS[desired_var] if bloom_termination_TS is not None else None,
        f"BLOOM_DURATION_TS_{desired_var}": (bloom_termination_TS[JULD] - bloom_initiation_TS[JULD]).days if bloom_initiation_TS is not None and bloom_termination_TS is not None else None,
        f"BLOOM_INITIATION_TS_DATE_{desired_var}": bloom_initiation_TS[JULD] if bloom_initiation_TS is not None else None,
        f"BLOOM_TERMINATION_TS_DATE_{desired_var}": bloom_termination_TS[JULD] if bloom_termination_TS is not None else None,
        f"BLOOM_INITIATION_CS_{desired_var}": bloom_initiation_CS[desired_var] if bloom_initiation_CS is not None else None,
        f"BLOOM_TERMINATION_CS_{desired_var}": bloom_termination_CS[desired_var] if bloom_termination_CS is not None else None,
        f"BLOOM_DURATION_CS_{desired_var}": (bloom_termination_CS[JULD] - bloom_initiation_CS[JULD]).days if bloom_initiation_CS is not None and bloom_termination_CS is not None else None,
        f"BLOOM_INITIATION_CS_DATE_{desired_var}": bloom_initiation_CS[JULD] if bloom_initiation_CS is not None else None,
        f"BLOOM_TERMINATION_CS_DATE_{desired_var}": bloom_termination_CS[JULD] if bloom_termination_CS is not None else None,
        f"BLOOM_INITIATION_RC_{desired_var}": bloom_initiation_RC[desired_var] if bloom_initiation_RC is not None else None,
        f"BLOOM_TERMINATION_RC_{desired_var}": bloom_termination_RC[desired_var] if bloom_termination_RC is not None else None,
        f"BLOOM_DURATION_RC_{desired_var}": (bloom_termination_RC[JULD] - bloom_initiation_RC[JULD]).days if bloom_initiation_RC is not None and bloom_termination_RC is not None else None,
        f"BLOOM_INITIATION_RC_DATE_{desired_var}": bloom_initiation_RC[JULD] if bloom_initiation_RC is not None else None,
        f"BLOOM_TERMINATION_RC_DATE_{desired_var}": bloom_termination_RC[JULD] if bloom_termination_RC is not None else None,
        f"INTEGRATED_BLOOM_TS_{desired_var}": integrated_bloom_TS if integrated_bloom_TS is not None else None,
        f"INTEGRATED_BLOOM_CS_{desired_var}": integrated_bloom_CS if integrated_bloom_CS is not None else None,
        f"INTEGRATED_BLOOM_RC_{desired_var}": integrated_bloom_RC if integrated_bloom_RC is not None else None,
        f"MEAN_BLOOM_TS_{desired_var}": mean_bloom_TS if mean_bloom_TS is not None else None,
        f"MEAN_BLOOM_CS_{desired_var}": mean_bloom_CS if mean_bloom_CS is not None else None,
        f"MEAN_BLOOM_RC_{desired_var}": mean_bloom_RC if mean_bloom_RC is not None else None,
    }
    
    return pd.DataFrame([phenology_dict]).reset_index(drop=True)