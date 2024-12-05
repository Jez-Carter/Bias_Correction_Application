# %% Importing Packages

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.slide_functions import background_map_rotatedcoords, plot_hex_grid, rotated_coord_system, markersize_legend

base_path = '/home/jez/'
repo_path = f'{base_path}Bias_Correction_Application/'
internal_datapath = f'{repo_path}Slide_Images/Data/'
external_datapath = f'{base_path}DSNE_ice_sheets/Jez/Slides/'
scenario_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'

# %% Loading Data
ds_aws_stacked = xr.open_dataset(f'{external_datapath}ds_aws_stacked.nc')
ds_climate = xr.open_dataset(f'{external_datapath}ds_climate.nc')

scenario = np.load(
    scenario_path, allow_pickle="TRUE",fix_imports=True,
).item()

ds_climate_coarse_june_stacked = scenario['ds_climate_coarse_june_stacked']
ds_climate_coarse_june_stacked_landonly = scenario['ds_climate_coarse_june_stacked_landonly'].rename({'Temperature':'LandOnly Temperature'})

ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked_landonly,ds_climate_coarse_june_stacked])
ds_climate_coarse_june = ds_climate_coarse_june_stacked.unstack()

# %% 