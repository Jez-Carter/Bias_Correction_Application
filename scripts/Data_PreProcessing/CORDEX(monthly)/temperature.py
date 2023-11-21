# %% Importing packages
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from pyproj import Transformer
import cartopy.crs as ccrs
import geopandas as gpd

# %% Loading CORDEX data and reformatting
base_path = '/home/jez/'
cordex_temperature_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/ensemble_temperature.npy'
grid_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/MetUM_011_ERA_INT_tas_2010.nc'
da_grid = xr.open_dataset(grid_path)

cordex_array = np.load(cordex_temperature_path)

models = ['ERAI',
          'ERA5',
          'MetUM(044)',
          'MetUM(011)',
          'MAR(ERAI)',
          'MAR(ERA5)',
          'RACMO(ERAI)',
          'RACMO(ERA5)']
cordex_da = xr.DataArray(cordex_array,
                       coords={'Model': models,
                               'Time':pd.date_range(start='1981-01-15', periods=456, freq='M'),
                               'grid_latitude':da_grid.grid_latitude,
                               'grid_longitude':da_grid.grid_longitude,
                               'latitude':(['grid_latitude','grid_longitude'],da_grid.latitude.data),
                               'longitude':(['grid_latitude','grid_longitude'],da_grid.longitude.data)},
                       dims=["Model",'Time','grid_latitude','grid_longitude'])
cordex_ds = cordex_da.to_dataset(name='Temperature')

mask_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/metum011_grid_land_filter.npy'
land_only_mask = np.load(mask_path)
cordex_ds['LSM'] = (('grid_latitude','grid_longitude'), land_only_mask)

cordex_ds = cordex_ds.assign_coords({"month": (cordex_ds.Time.dt.month)})
cordex_ds = cordex_ds.assign_coords({"year": (cordex_ds.Time.dt.year)})

# %% Including Specific Rotated Coordinate System (glon,glat)
cordex_ds_stacked = cordex_ds.stack(X=(('grid_longitude','grid_latitude')))
climate_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=0.0,
    globe=None,
)
rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
grid_lon=cordex_ds_stacked['grid_longitude']
grid_lat=cordex_ds_stacked['grid_latitude']
transformer = Transformer.from_crs(climate_coord_system, rotated_coord_system)

glon,glat = transformer.transform(grid_lon,grid_lat)

cordex_ds_stacked = cordex_ds_stacked.assign_coords(glon=("X", glon))
cordex_ds_stacked = cordex_ds_stacked.assign_coords(glat=("X", glat))
cordex_ds = cordex_ds_stacked.unstack()

# %% Saving Reformatted CORDEX data
CORDEX_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX.nc'
cordex_ds.to_netcdf(CORDEX_outpath)

# %% Importing ice sheet shapefile for plotting purposes
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
gdf_icesheet = gpd.read_file(icehsheet_shapefile)

rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)

# %% Figure Showing MetUM(011) DEM
fig, ax = plt.subplots(1, 1, figsize=(10, 8),dpi=600)

jan_cordex_ds = cordex_ds.where(cordex_ds.month==1,drop=True)
jan_mean_cordex_ds = jan_cordex_ds.mean('Time')
jan_mean_cordex_ds_landonly = jan_mean_cordex_ds.where(jan_mean_cordex_ds.LSM)#,drop=True)
da = jan_mean_cordex_ds_landonly['Temperature'].sel(Model='MetUM(011)')

da.plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.8,
    linewidths=0.01)
    # cmap = 'viridis')

gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

# %%
# cordex_ds_2010_2018 = cordex_ds.where(cordex_ds.year>=2010,drop=True)
# jan_cordex_ds_2010_2018 = cordex_ds_2010_2018.where(cordex_ds_2010_2018.month==1,drop=True)
# mean_jan_cordex_ds_2010_2018 = jan_cordex_ds_2010_2018.mean('Time')
# Jan_Mean_2010_2018_CORDEX_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_janmean_20102018.nc'
# mean_jan_cordex_ds_2010_2018.to_netcdf(Jan_Mean_2010_2018_CORDEX_outpath)