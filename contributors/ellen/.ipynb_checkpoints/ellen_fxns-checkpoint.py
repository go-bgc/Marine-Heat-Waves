import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import xarray as xr
from scipy.interpolate import griddata
import glob

# Add path to Zack's fxns
import sys
sys.path.append('../Zack/')
from zack_functions import find_bloom_phenological_indices

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
    elif oisst_path.split('.')[-1] == '.nc':
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

def get_soca_pheno(yr_range, lon0, lon1, lat0, lat1, soca_dir = '/Volumes/FATDATABABY/copernicus/bgc-chl-poc/',
                   soca_version = 'P202511', save=False, outdir = ''):

    # Load biome flags
    biomes = xr.open_dataset('biomes_2010_2025.nc')
    biomes.close()
    
    XX_biome, YY_biome = np.meshgrid(biomes.lon.values, biomes.lat.values)
    points = (XX_biome.flatten(), YY_biome.flatten())
    
    # pheno columns
    var_names = ['PRES', 'TEMP', 'PSAL', 'DOXY', 'CHLA', 'BBP700', 'LATITUDE', 'LONGITUDE', 'JULD']
    
    columns = ['BLOOM_MAXIMUM', 'BLOOM_INITIATION_TS', 'BLOOM_TERMINATION_TS', 'BLOOM_DURATION_TS',
           'BLOOM_INITIATION_TS_DATE', 'BLOOM_TERMINATION_TS_DATE',
           'BLOOM_INITIATION_CS', 'BLOOM_TERMINATION_CS', 'BLOOM_DURATION_CS',
           'BLOOM_INITIATION_CS_DATE', 'BLOOM_TERMINATION_CS_DATE',
           'BLOOM_INITIATION_RC', 'BLOOM_TERMINATION_RC', 'BLOOM_DURATION_RC',
           'BLOOM_INITIATION_RC_DATE', 'BLOOM_TERMINATION_RC_DATE',
           'INTEGRATED_BLOOM_TS','INTEGRATED_BLOOM_CS', 'INTEGRATED_BLOOM_RC',
               'MEAN_BLOOM_TS', 'MEAN_BLOOM_CS','MEAN_BLOOM_RC']
    
    for yi, year in enumerate(yr_range):
    
        print('\n',year)
        # Get all SOCA-files
        flist = sorted(glob.glob(soca_dir+'cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg_P7D-m_'+str(int(year))+'*_'+soca_version+'.nc'))
        data = xr.open_mfdataset(flist)
        data.close()
        
        # Drop extra variables
        data = data.drop_vars(["ED380", "ED380_error","ED412","ED412_error","ED490", "ED490_error","chl_error",
                        "poc_error","bbp_error","PAR_error"])
        # Add new coordinate of year
        data = data.assign_coords(year=("year", [year]))
    
        # Crop to target region
        lon_inds = np.where((data.longitude.values>=lon0) & (data.longitude.values<=lon1))[0]
        lat_inds = np.where((data.latitude.values>=lat0) & (data.latitude.values<=lat1))[0]
        
        # Subset just to surface
        # This will need to be updated if we want to keep depth resolved
        data = data.isel(longitude = lon_inds, latitude = lat_inds, depth = 0)
        
        XX_SOCA, YY_SOCA = np.meshgrid(data.longitude.values, data.latitude.values)
    
        # Re-grid biome flags
        biome_flags = griddata(points, 
                               biomes.Biomes.values[np.where(biomes.year.values==year)[0],:,:].flatten(), 
                       (XX_SOCA, YY_SOCA), method='nearest').reshape(XX_SOCA.shape)
        biome_flags = biome_flags.reshape(XX_SOCA.shape)
        
        # Add biome
        data['BIOME'] = (['year','latitude','longitude'],np.expand_dims(biome_flags.reshape(XX_SOCA.shape), axis=0))
    
        # Calculate bloom phenology
        chlorophyll = data.chl.values
        backscatter = data.bbp.values
        # plt.plot(data.time.values, bloom_slice)
    
        olabels = ['CHLA','BBP700']
        for oi, oparam in enumerate(['chl','bbp']):

            print(oparam)
            odata = data[oparam].values
            for ri in np.arange(data.latitude.shape[0]):

                print(ri, data.latitude.shape[0])
                for ci in np.arange(data.longitude.shape[0]):
            
                    df = pd.DataFrame({olabels[oi]: odata[:,ri,ci],
                                       'JULD': data.time.values})
                    df = df.loc[(df.loc[:,olabels[oi]].isna()==False), :]
            
                    results =  find_bloom_phenological_indices(df, var_names,int(oi+1))
                    results = results.drop(columns=["BLOOM_SLICE_"+olabels[oi]])
            
                    # print('start')
                    if (ri == 0) and (ci == 0):
                        df_all = results
                    else:
                        df_all = pd.concat((df_all, results))
        
            # Add bloom pheonology to data
        
            for pheno_name in columns:
    
                name = pheno_name+'_'+olabels[oi]
                if 'DATE' not in name:
                    bloom_values = df_all.loc[:,name].values.astype('float64')
                else:
                    # Convert to date time
                    bloom_values = np.array([np.datetime64("NaT") if type(ti) == float else np.datetime64(ti) for ti in
                        df_all.loc[:,name].values])
        
                    # Convert to day of year instead
                    # bloom_values = np.array([np.nan if type(ti) == float else pd.Timestamp(ti).dayofyear for ti in 
                    # df_all.loc[:,name].values])
                    
                bloom_values = np.expand_dims(bloom_values.reshape(XX_SOCA.shape), axis=0)
                data[name] = (['year','latitude','longitude'], bloom_values)
    

        # Update time to day of year
        data['time'] = [pd.Timestamp(ti).dayofyear for ti in data.time.values]

        if save:
            data.to_netcdf(outdir+str(int(year))+'.nc')
        # if yi == 0:
        #     all_data = data
        # else:
        #     all_data = xr.concat([all_data, data],dim = 'year')

    return