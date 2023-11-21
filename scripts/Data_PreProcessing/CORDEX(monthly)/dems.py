# %%
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from pyproj import Transformer
import cartopy.crs as ccrs
import geopandas as gpd

# %% Loading DEM data and Reformatting
base_path = '/home/jez/'
dems_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/ensemble_elevations.npy'
grid_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/MetUM_011_ERA_INT_tas_2010.nc'
da_grid = xr.open_dataset(grid_path)

dems_array = np.load(dems_path)
models = ['ERAI',
          'ERA5',
          'MetUM(044)',
          'MetUM(011)',
          'MAR',
          'RACMO']
dems_da = xr.DataArray(dems_array,
                       coords={'Model': models,
                               'grid_latitude':da_grid.grid_latitude,
                               'grid_longitude':da_grid.grid_longitude,
                               'latitude':(['grid_latitude','grid_longitude'],da_grid.latitude.data),
                               'longitude':(['grid_latitude','grid_longitude'],da_grid.longitude.data)},
                       dims=["Model", "grid_latitude", "grid_longitude"])
dems_ds = dems_da.to_dataset(name='Elevation')

mask_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/metum011_grid_land_filter.npy'
land_only_mask = np.load(mask_path)
dems_ds['LSM'] = (('grid_latitude','grid_longitude'), land_only_mask)

# %% Including Specific Rotated Coordinate System (glon,glat)

dems_ds_stacked = dems_ds.stack(X=(('grid_longitude','grid_latitude')))
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
grid_lon=dems_ds_stacked['grid_longitude']
grid_lat=dems_ds_stacked['grid_latitude']
transformer = Transformer.from_crs(climate_coord_system, rotated_coord_system)

glon,glat = transformer.transform(grid_lon,grid_lat)

dems_ds_stacked = dems_ds_stacked.assign_coords(glon=("X", glon))
dems_ds_stacked = dems_ds_stacked.assign_coords(glat=("X", glat))
dems_ds = dems_ds_stacked.unstack()

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

dems_ds_landonly = dems_ds.where(dems_ds.LSM)#,drop=True)
da = dems_ds_landonly['Elevation'].sel(Model='MetUM(011)')

da.plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.8,
    vmin=0,
    # edgecolors='k',
    linewidths=0.01)
    # cmap = 'viridis')

gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

# %% Saving Reformatted DEMs data
dems_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/dems.nc'
dems_ds.to_netcdf(dems_outpath)
