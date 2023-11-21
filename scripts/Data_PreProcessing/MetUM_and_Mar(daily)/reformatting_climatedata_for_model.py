
# %% Importing Packages

import xarray as xr
import cartopy.crs as ccrs
from pyproj import Transformer

# %% Merging Temperature, Elevation and Masks Data

base_path = '/home/jez/'
nst_climate_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/ProcessedData/MetUM_Daily_TAS.nc'
ele_climate_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/Antarctic_CORDEX_MetUM_0p44deg_orog.nc'
mask_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/ProcessedData/MetUM_044_Masks.nc'
out_folder = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/ProcessedData/'

ds_nst_climate = xr.open_dataset(nst_climate_path)
ds_ele_climate = xr.open_dataset(ele_climate_path)
ds_mask = xr.open_dataset(mask_path)
ds_climate = xr.merge([ds_nst_climate,ds_ele_climate,ds_mask])

# %% Adjusting temperature from Kelvin to Degrees
ds_climate['tas']=ds_climate['tas']-273.15

# %% Including Specific Rotated Coordinate System (glon,glat)
# ds_climate = ds_climate.isel(time=(ds_climate.time.dt.month == 1))
# ds_climate = ds_climate.where(ds_climate.ross_mask)
ds_climate_stacked = ds_climate.stack(X=(('grid_longitude','grid_latitude')))#.dropna('X','all')

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
grid_lon=ds_climate_stacked['grid_longitude']
grid_lat=ds_climate_stacked['grid_latitude']

transformer = Transformer.from_crs(climate_coord_system, rotated_coord_system)

glon,glat = transformer.transform(grid_lon,grid_lat)

ds_climate_stacked = ds_climate_stacked.assign_coords(glon=("X", glon))
ds_climate_stacked = ds_climate_stacked.assign_coords(glat=("X", glat))
ds_climate = ds_climate_stacked.unstack()

# %% Saving
ds_climate.to_netcdf(f'{out_folder}MetUM_Reformatted.nc')

