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

def get_oisst_flag_spatialONLY(
        all_float_data,
        start_date,
        end_date,
        oisst_path='s3://uw-escience-scratch-prod/oisst/mhw_mask_north_atlantic_final.zarr',
        time_name='time',
        lon_name='lon',
        lat_name='lat',
        flag_name='mhw_mask'):

    """
    Assign a spatial MHW flag to BGC-Argo profiles for a specified
    OISST time range.

    An OISST grid cell is flagged True if mhw_mask == 1 at least
    once between start_date and end_date.

    Each float profile is matched to its nearest OISST grid cell.
    If that grid cell is True in the spatial mask, the profile
    receives mhw_mask = 1. Otherwise it receives mhw_mask = 0.

    Parameters
    ----------
    all_float_data : pandas.DataFrame
        Must contain:
            PLATFORM_NUMBER
            CYCLE_NUMBER
            LONGITUDE
            LATITUDE

    start_date : str or datetime-like
        Beginning of OISST time range.
        Example: '2018-01-01'

    end_date : str or datetime-like
        End of OISST time range.
        Example: '2020-12-31'

    oisst_path : str
        Path to OISST dataset (.zarr or .nc).

    time_name : str
        Name of OISST time dimension.

    lon_name : str
        Name of OISST longitude coordinate.

    lat_name : str
        Name of OISST latitude coordinate.

    flag_name : str
        Name of OISST MHW mask variable.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with spatial MHW flag added.
    """


    # ---------------------------------------------------
    # 1. Get one location for each float profile
    # ---------------------------------------------------

    float_data = (
        all_float_data[
            [
                'PLATFORM_NUMBER',
                'CYCLE_NUMBER',
                'LONGITUDE',
                'LATITUDE'
            ]
        ]
        .groupby(
            ['PLATFORM_NUMBER', 'CYCLE_NUMBER']
        )
        .mean()
    )


    # ---------------------------------------------------
    # 2. Convert dates
    # ---------------------------------------------------

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if start_date > end_date:
        raise ValueError(
            'start_date must be before end_date'
        )

    print(
        'Requested MHW time range:',
        start_date,
        'to',
        end_date
    )


    # ---------------------------------------------------
    # 3. Open OISST dataset
    # ---------------------------------------------------

    print('Opening:', oisst_path)

    if oisst_path.endswith('.zarr'):

        mhw_data = xr.open_zarr(oisst_path)

    elif oisst_path.endswith('.nc'):

        mhw_data = xr.open_dataset(oisst_path)

    else:

        raise ValueError(
            'OISST file must be either .zarr or .nc'
        )


    # ---------------------------------------------------
    # 4. Restrict OISST to requested time period
    # ---------------------------------------------------

    print('Subsetting OISST time...')

    mask_subset = mhw_data[flag_name].sel(
        {
            time_name: slice(
                start_date,
                end_date
            )
        }
    )

    if mask_subset.sizes[time_name] == 0:
        mhw_data.close()

        raise ValueError(
            'No OISST data found between '
            f'{start_date} and {end_date}'
        )

    print(
        'Actual OISST subset:',
        mask_subset[time_name].min().values,
        'to',
        mask_subset[time_name].max().values
    )

    print(
        'Number of OISST time steps:',
        mask_subset.sizes[time_name]
    )


    # ---------------------------------------------------
    # 5. Collapse ONLY selected dates into spatial mask
    # ---------------------------------------------------

    spatial_mask = (
        (mask_subset == 1)
        .any(dim=time_name)
        .compute()
    )


    # ---------------------------------------------------
    # 6. Diagnostics
    # ---------------------------------------------------

    n_total = spatial_mask.size
    n_mhw = spatial_mask.sum().item()

    print('Total grid cells:', n_total)
    print('MHW grid cells:', n_mhw)

    print(
        'Percent selected:',
        n_mhw / n_total * 100
    )


    # ---------------------------------------------------
    # 7. Get coordinates that are actually True
    # ---------------------------------------------------

    lat_idx, lon_idx = np.where(
        spatial_mask.values
    )

    oisst_lats = mhw_data[lat_name].values
    oisst_lons = mhw_data[lon_name].values

    mhw_lats = oisst_lats[lat_idx]
    mhw_lons = oisst_lons[lon_idx]

    print(
        'Number of selected MHW coordinates:',
        len(mhw_lats)
    )


    # ---------------------------------------------------
    # 8. Prepare float coordinates
    # ---------------------------------------------------

    float_lons = (
        float_data['LONGITUDE']
        .values
        .copy()
    )

    float_lats = (
        float_data['LATITUDE']
        .values
        .copy()
    )


    # Make longitude convention match OISST
    if np.nanmax(oisst_lons) > 180:

        # OISST uses 0--360
        float_lons = float_lons % 360

    else:

        # OISST uses -180--180
        float_lons = (
            (float_lons + 180) % 360
        ) - 180


    # ---------------------------------------------------
    # 9. Initialize output
    # ---------------------------------------------------

    all_mhw_flags = np.full(
        len(all_float_data),
        -999.0
    )


    # ---------------------------------------------------
    # 10. Match each float profile to nearest OISST cell
    # ---------------------------------------------------

    for ni, ((wmo, cycle), row) in enumerate(
        float_data.iterrows()
    ):

        if ni % 100 == 0:

            print(
                ni,
                'out of',
                len(float_data)
            )


        lon = float_lons[ni]
        lat = float_lats[ni]


        # Skip missing positions
        if np.isnan(lon) or np.isnan(lat):
            continue


        # ------------------------------------------------
        # Find nearest OISST grid cell and get its flag
        # ------------------------------------------------

        output_value = spatial_mask.sel(
            {
                lon_name: lon,
                lat_name: lat
            },
            method='nearest'
        ).item()

        output_value = int(output_value)


        # ------------------------------------------------
        # Find all dataframe rows belonging to profile
        # ------------------------------------------------

        all_inds = np.where(
            (
                all_float_data[
                    'PLATFORM_NUMBER'
                ].values == wmo
            )
            &
            (
                all_float_data[
                    'CYCLE_NUMBER'
                ].values == cycle
            )
        )[0]


        # ------------------------------------------------
        # Assign spatial MHW flag
        # ------------------------------------------------

        all_mhw_flags[
            all_inds
        ] = output_value


    # ---------------------------------------------------
    # 11. Add flag to dataframe
    # ---------------------------------------------------

    all_float_data = all_float_data.assign(
        **{
            flag_name: all_mhw_flags
        }
    )


    # ---------------------------------------------------
    # 12. Close dataset
    # ---------------------------------------------------

    mhw_data.close()


    return all_float_data