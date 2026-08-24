from ellen_fxns import get_soca_pheno
import pandas as pd
import numpy as np


# Load MHW data
mhw_dates = pd.read_csv('../../MHW_list.csv')
mhw = 'NA 2023'
lon0, lon1, lat0, lat1 = mhw_dates.loc[(mhw_dates.loc[:,'name']==mhw), ['lon0','lon1','lat0','lat1']].values[0,:]

# Calculate pheonology for full time range
yr_range = np.arange(2010, 2024)
output = get_soca_pheno(yr_range, lon0, lon1, lat0, lat1)

# Add MHW mask
XX_SOCA, YY_SOCA = np.meshgrid(output.longitude.values, output.latitude.values)

all_mhw = np.zeros((yr_range.shape[0], XX_SOCA.shape[0], XX_SOCA.shape[1]))*np.nan

XX_MHW, YY_MHW = np.meshgrid(mhw_mask.lon.values, mhw_mask.lat.values)
points = (XX_MHW.flatten(), YY_MHW.flatten())
for yi, year in enumerate(yr_range):

    # Down sample to match SOCA grid
    mwh_flags = griddata(points, 
                         mhw_mask.mhw_counts.values[np.where(mhw_mask.year.values == year)[0], :,:].flatten(), 
                           (XX_SOCA, YY_SOCA), method='nearest').reshape(XX_SOCA.shape)
    all_mhw[yi,:] = mwh_flags.reshape(XX_SOCA.shape)

output['mhw_count'] = (['year','latitude','longitude'], all_mhw)

output.to_netcdf('../../../gobgc26_example/data/soca_bloom_pheno_all.nc')