import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import netCDF4 as nc
import datetime
import pandas as pd
import glob
import multiprocessing
import scipy.interpolate.ndgriddata as ndgriddata
import global_land_mask
import cftime
import numpy.ma as ma
import dateutil.parser
from sklearn.linear_model import LinearRegression

data_plot = nc.Dataset('./plot_mean_k_sig_land.nc')['value'][:]
#0:mean_annual_total_precp_obs,1:mean_annual_total_precp_e3sm, 2:e3sm-obs,3. |e3sm-obs|  4:k_bia, 5:sig_bia (if >0.1 give 1)
lat = nc.Dataset('./plot_mean_k_sig_land.nc')['lat'][:]
lon = nc.Dataset('./plot_mean_k_sig_land.nc')['lon'][:]
lon_grid, lat_grid = np.meshgrid(lon,lat)
land_ocean = global_land_mask.is_land(lat_grid, lon_grid)+1-1
df = pd.DataFrame(land_ocean)
land_1 = np.array(df.replace(0,np.nan))
#m = Basemap(projection='cyl')
#lon, lat = np.meshgrid(lons, lats)
#xi, yi = m(lon, lat)
fig = plt.figure(figsize = (16*2*1.5,10*3))
i=0
ax=fig.add_subplot(3,2,int(i+1))
value = data_plot[i,:,:].data
value_sort = np.sort(np.array(pd.DataFrame(value).replace(np.nan,0)).reshape(-1,))
levels = np.linspace (np.nanmin(value),value_sort[int(0.98*len(value_sort)),],100)
plt.contourf(lon,lat,value,cmap='BrBG',levels=levels,extend='both')
plt.colorbar()
plt.title('Precp obs (mm/day)')
plt.ylabel('latitude')



i=1
ax=fig.add_subplot(3,2,int(i+1))
value = data_plot[i,:,:].data
value_sort = np.sort(np.array(pd.DataFrame(value).replace(np.nan,0)).reshape(-1,))
levels = np.linspace (np.nanmin(value),value_sort[int(0.98*len(value_sort)),],100)
plt.contourf(lon,lat,value,cmap='BrBG',levels=levels,extend='both')
plt.colorbar()
plt.title('Precp E3SM (mm/day)')
plt.ylabel('latitude')

i=2
ax=fig.add_subplot(3,2,int(i+1))
value = data_plot[i,:,:].data
value_sort = np.sort(np.array(pd.DataFrame(value).replace(np.nan,0)).reshape(-1,))
levels = np.linspace (-6.9,6.9,100)
plt.contourf(lon,lat,value,cmap='seismic',levels=levels,extend='both')
plt.colorbar()
plt.title('Precp Bias E3SM-OBS  (mm/day)')
plt.ylabel('latitude')

i=3
ax=fig.add_subplot(3,2,int(i+1))
value = data_plot[i,:,:].data
value_sort = np.sort(np.array(pd.DataFrame(value).replace(np.nan,0)).reshape(-1,))
levels = np.linspace (np.nanmin(value),value_sort[int(0.98*len(value_sort)),],100)
plt.contourf(lon,lat,value,cmap='Reds',levels=levels,extend='both')
plt.colorbar()
plt.title('Precp Bias |E3SM-OBS|  (mm/day)')
plt.ylabel('latitude')

i=4
ax=fig.add_subplot(3,2,int(i+1))
value = data_plot[i,:,:].data
value_sort = np.sort(np.array(pd.DataFrame(value).replace(np.nan,0)).reshape(-1,))
levels = np.linspace (-0.1,0.1,100)
plt.contourf(lon,lat,value,cmap='seismic',levels=levels,extend='both')
plt.colorbar()
plt.title('Trend of precp Bias |E3SM-OBS|  (mm/day)*y-1')
plt.xlabel('longitude')
plt.ylabel('latitude')

i=4
ax=fig.add_subplot(3,2,6)
value = data_plot[i,:,:].data
value_sort = np.sort(np.array(pd.DataFrame(value).replace(np.nan,0)).reshape(-1,))
levels = np.linspace (-0.1,0.1,100)
plt.contourf(lon,lat,value,cmap='seismic',levels=levels,extend='both')
plt.colorbar()
plt.title('Trend of precp Bias |E3SM-OBS| (p<0.1) (mm/day)*y-1')
plt.xlabel('longitude')
plt.ylabel('latitude')
sig = data_plot[i+1,:,:].data
df = pd.DataFrame(sig)
df_5 = df.replace(np.nan,5)
df_new = np.array(df_5.replace(1,np.nan))*land_1#this part I just want to make the no-significant region as 5 and significant region as np.nan
ax.contourf(lon, lat, df_new, hatches=['++',None],levels=levels,colors='white', extend='both', alpha=0.1)


plt.show()

