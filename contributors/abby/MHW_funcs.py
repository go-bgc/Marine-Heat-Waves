import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
from cmocean import cm as cmo
from cartopy import crs as ccrs
from cartopy import feature as cfeature 

def get_oisst_flag(all_float_data, oisst_path = 's3://uw-escience-scratch-prod/oisst/mhw_mask_north_atlantic_final.zarr',
                   time_name = 'time', lon_name = 'lon', lat_name = 'lat', flag_name = 'mhw_mask'):

    """
    Function to match up BGC-Argo float profile to OISST MWH
    based on KDTree (nearest neighbor)

    float_data: Position data must structured as a pandas DataFrame with the following
    column names:
    - LON
    - LAT
    - JULD 
    
    """

    # Get all locations 
    float_data = all_float_data.loc[:,['PLATFORM_NUMBER','CYCLE_NUMBER', \
                                       'LONGITUDE','LATITUDE','JULD']].groupby(by = ['PLATFORM_NUMBER','CYCLE_NUMBER']).mean()

    float_wmo_cycle = float_data.index.values
    
    # Load OISST data
    # zarr
    print('Openining : '+oisst_path)
    if oisst_path.split('.')[-1] == 'zarr':
        mhw_data = xr.open_zarr(oisst_path)
    elif oisst_path.split('.')[-1] == 'nc':
        mhw_data = xr.open_dataset(oisst_path)
    
    mhw_data.close()
    
    # Make KDTree for nearest neighbor matchups based on MWH mask
    XX, YY = np.meshgrid(mhw_data[lon_name].values, mhw_data[lat_name].values)
    # xx: lon; yy: lat
    tree = KDTree(np.c_[XX.ravel(),YY.ravel()])
    
    all_mhw_flags = np.zeros(all_float_data.shape[0])*-999
    
    # Convert all float date-times to just date
    dates = np.array([pd.Timestamp(xi).date() for xi in float_data.loc[:,'JULD'].values], dtype='datetime64[D]')
    
    # Get unique dates
    unique_dates = np.unique(dates)
    
    # For each date...
    for di, match_date in enumerate(unique_dates):
    
        if di%100 == 0:
            print(di, ' out of', unique_dates.shape[0])
            
        # 1. Get MWH slice for that date
        mhw_i = np.where(mhw_data[time_name].values.astype('datetime64[D]') == match_date)[0]
    
        # If the date is in the data range,
        if mhw_i.shape[0] > 0:
    
            # Subset array
            # mhw_mask = mhw_data[flag_name].isel(**{time_name: mhw_i})
            
            # 2. Get all profile locations
            prof_inds = np.where(unique_dates == match_date)[0]
            prof_inds = np.where(dates == match_date)[0]

            if prof_inds.shape[0]==0:
                    print(match_date)

            # Convert longitude to 360
            lons = (((360 + (float_data.loc[:,'LONGITUDE'].values[prof_inds] % 360)) % 360))
            lats = float_data.loc[:,'LATITUDE'].values[prof_inds]

            # 3. Use nearest neighbor to get flag
            for ni in np.arange(prof_inds.shape[0]):
            
                # print(ni, lons[ni], lats[ni])
                dd, ii = tree.query([lons[ni], lats[ni]])
                ri, ci = np.unravel_index(ii, XX.shape)
    
                # Get index for expanded data
                wmo, cycle = float_wmo_cycle[prof_inds][ni]
                all_inds = all_float_data.loc[(all_float_data.loc[:,'PLATFORM_NUMBER']==wmo) & \
                    (all_float_data.loc[:,'CYCLE_NUMBER']==cycle),:].index.values
                
                # 4. Assign flags back to original shape
                output_value = mhw_data[flag_name].isel(**{time_name: mhw_i,lat_name: ri, lon_name: ci}).values[0]
                all_mhw_flags[all_inds] = output_value

        else:
            print(di)
    # Add flags to pandas data frame
    all_float_data = all_float_data.assign(**{flag_name: all_mhw_flags})

    return all_float_data

def map_study_region(ax=None, ax_lims=[-127, -121, 34, 38], gridlabel=True, figsize=(8,4), PROJ=None):

    if PROJ is None:
        PROJ = ccrs.PlateCarree()

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=PROJ)

    ax.set_extent(
        ax_lims,
        crs=ccrs.PlateCarree()
    )

    ax.add_feature(
        cfeature.LAND,
        zorder=5,
        linewidth=1,
        edgecolor='k',
        facecolor='linen'
    )

    ax.coastlines(
        resolution="50m",
        zorder=6,
        linewidth=1
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False
    )

    return ax

def get_oisst_flag_spatialONLY(all_float_data, 
                               oisst_path='s3://uw-escience-scratch-prod/oisst/mhw_mask_north_atlantic_final.zarr', 
                               lon_name='lon',
                               lat_name='lat',
                               flag_name='mhw_mask'):

    """
    Match BGC-Argo float profiles to a spatial OISST mask
    using nearest-neighbor matching with a KDTree.

    Parameters
    ----------
    all_float_data : pandas.DataFrame
        Must contain:
        - PLATFORM_NUMBER
        - CYCLE_NUMBER
        - LONGITUDE
        - LATITUDE

    oisst_path : str
        Path to spatial OISST mask.

    lon_name : str
        Longitude coordinate name in OISST dataset.

    lat_name : str
        Latitude coordinate name in OISST dataset.

    flag_name : str
        Name of spatial mask variable.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with spatial mask flag added.
    """

    # ---------------------------------------------------
    # 1. Get one location for each float profile
    # ---------------------------------------------------

    float_data = (
        all_float_data[
            ['PLATFORM_NUMBER', 'CYCLE_NUMBER',
             'LONGITUDE', 'LATITUDE']
        ]
        .groupby(['PLATFORM_NUMBER', 'CYCLE_NUMBER'])
        .mean()
    )

    float_wmo_cycle = float_data.index.values

    # ---------------------------------------------------
    # 2. Load OISST spatial mask
    # ---------------------------------------------------

    print('Opening:', oisst_path)

    if oisst_path.endswith('.zarr'):
        mhw_data = xr.open_zarr(oisst_path)

    elif oisst_path.endswith('.nc'):
        mhw_data = xr.open_dataset(oisst_path)

    else:
        raise ValueError('OISST file must be .zarr or .nc')

    # ---------------------------------------------------
    # 3. Build KDTree from OISST grid
    # ---------------------------------------------------

    XX, YY = np.meshgrid(
        mhw_data[lon_name].values,
        mhw_data[lat_name].values
    )

    tree = KDTree(
        np.c_[XX.ravel(), YY.ravel()]
    )

    # Default value for profiles outside / unavailable
    all_mhw_flags = np.full(
        all_float_data.shape[0],
        -999.0
    )

    # ---------------------------------------------------
    # 4. Match each float profile spatially
    # ---------------------------------------------------

    lons = float_data['LONGITUDE'].values % 360
    lats = float_data['LATITUDE'].values

    for ni in range(float_data.shape[0]):

        if ni % 100 == 0:
            print(ni, 'out of', float_data.shape[0])

        # Find nearest OISST grid point
        distance, ii = tree.query(
            [lons[ni], lats[ni]]
        )

        ri, ci = np.unravel_index(
            ii,
            XX.shape
        )

        # Get WMO and cycle
        wmo, cycle = float_wmo_cycle[ni]

        # Find all rows belonging to this profile
        all_inds = all_float_data.index[
            (all_float_data['PLATFORM_NUMBER'] == wmo) &
            (all_float_data['CYCLE_NUMBER'] == cycle)
        ].values

        # Get spatial mask value
        output_value = mhw_data[flag_name].isel(
            **{
                lat_name: ri,
                lon_name: ci
            }
        ).values

        # Assign back to every measurement from this profile
        all_mhw_flags[all_inds] = output_value

    # ---------------------------------------------------
    # 5. Add flag to original dataframe
    # ---------------------------------------------------

    all_float_data = all_float_data.assign(
        **{flag_name: all_mhw_flags}
    )

    mhw_data.close()

    return all_float_data
