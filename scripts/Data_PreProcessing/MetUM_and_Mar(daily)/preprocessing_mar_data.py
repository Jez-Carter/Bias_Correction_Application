# %%
import xarray as xr
from src.loader_functions import regrid
import iris
import numpy as np

# %% Loading Data
base_path = '/home/jez/'
metum_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/ProcessedData/MetUM_Reformatted.nc'
# data_folder = f'{base_path}DSNE_ice_sheets/Jez/MAR_CORDEX_Data/'
# mar_path = f'{data_folder}daily-TT-MAR_ERA5-2010-2019.nc'
mar_path = f'{base_path}Bias_Correction_Application/MAR3.11_TT_2010_3h.nc4'
mar_mask_path = f'{base_path}Bias_Correction_Application/mar_land_sea_mask.nc'

ds_climate = xr.open_dataset(mar_path)
ds_climate = ds_climate.resample(time='D').mean()

ds_mask = xr.open_dataset(mar_mask_path)
ds_climate['LSM']=ds_mask.ICE
ds_climate['TT'].attrs["units"] = 'degree_Celsius'
ds_climate['TT'].attrs["standard_name"] = 'air_temperature'
cube_climate = ds_climate['TT'].to_iris()

# %% Regridding MAR
grid_cube = iris.load(metum_path)[0]
cubelist = []
for i in np.arange(0,cube_climate.shape[0],1): #Regridding each time-coordinate in turn. 
    cube_regrid = regrid(cube_climate[i],grid_cube,'cubic')
    cube_regrid.add_aux_coord(cube_climate[i].coord('time'))
    cubelist.append(cube_regrid)   
cubelist = iris.cube.CubeList(cubelist)
cube = cubelist.merge_cube() 
da_climate = xr.DataArray.from_iris(cube)
da_climate = da_climate.rename({'dim_1': 'grid_longitude','dim_2': 'grid_latitude'})
ds_climate = da_climate.to_dataset()

# %% Reformatting
ds_metum = xr.open_dataset(metum_path)
ds_climate['lsm'] = ds_metum['lsm']
ds_climate['orog'] = ds_metum['orog']
ds_climate.reset_coords(names=['lsm','orog'])
ds_climate = ds_climate.rename({'cube_regridded': 'tas'})

# %% Saving Processed Data
mar_outpath = f'{base_path}Bias_Correction_Application/MAR3.11_TT_2010_daily_reformatted.nc'
da_climate.to_netcdf(mar_outpath)

# %%
