# %% Importing packages
import xarray as xr

# %% Loading preprocessed data
base_path = '/home/jez/'
CORDEX_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX.nc'
dems_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/dems.nc'

cordex_ds = xr.open_dataset(CORDEX_path)
dems_ds = xr.open_dataset(dems_path)

# %% Adjusting DEMs DS to match CORDEX Models
mar_dem_ds = dems_ds.sel(Model='MAR')
mar_dem_ds['Model'] = 'MAR(ERAI)'
dems_ds = xr.concat([dems_ds,mar_dem_ds], 'Model')
mar_dem_ds['Model'] = 'MAR(ERA5)'
dems_ds = xr.concat([dems_ds,mar_dem_ds], 'Model')
mar_dem_ds = dems_ds.sel(Model='RACMO')
mar_dem_ds['Model'] = 'RACMO(ERAI)'
dems_ds = xr.concat([dems_ds,mar_dem_ds], 'Model')
mar_dem_ds['Model'] = 'RACMO(ERA5)'
dems_ds = xr.concat([dems_ds,mar_dem_ds], 'Model')
dems_ds = dems_ds.drop(labels=['MAR','RACMO'], dim='Model')

# %% Including new variable in CORDEX dataset
cordex_ds['Elevation']=dems_ds['Elevation']

# %% Saving Reformatted CORDEX data
CORDEX_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'
cordex_ds.to_netcdf(CORDEX_outpath)
