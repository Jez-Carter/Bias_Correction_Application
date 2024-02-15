# %% Importing Packages
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gpd
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1

legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm

from src.helper_functions import create_mask

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.hierarchical.prediction_functions import (
    generate_posterior_predictive_realisations_hierarchical_mean,
    generate_posterior_predictive_realisations_hierarchical_std,
    generate_meanfunction_posterior_predictions
)

# %% Loading data
inpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real_posterior.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

inpath_scaler = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario_for_scaler = np.load(
    inpath_scaler, allow_pickle="TRUE"
).item()

# %% Printing Scaling Values
ele_scaler = scenario_for_scaler['ele_scaler']
lat_scaler = scenario_for_scaler['lat_scaler']

print(f'Elevation Scalar Mean = {ele_scaler.mean_[0]:.2f} \n',
      f'Elevation Scalar Standard Dev = {ele_scaler.scale_[0]:.2f} \n',
      f'Latitude Scalar Mean = {lat_scaler.mean_[0]:.2f} \n',
      f'Latitude Scalar Standard Dev = {lat_scaler.scale_[0]:.2f}'
      )


# %% Loading climate model data
base_path = '/home/jez/'
cordex_inpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'
ds_climate = xr.open_dataset(cordex_inpath)

# %% Reformatting data for Predictions
months = [12,1,2]
ds_climate_coarse = ds_climate.coarsen(grid_latitude=7,grid_longitude=7).mean()
ds_climate_coarse = ds_climate_coarse.sel(Model='MAR(ERA5)')
ds_climate_stacked = ds_climate_coarse.stack(X=('grid_latitude', 'grid_longitude'))

nx = jnp.array(np.dstack([ds_climate_stacked['glon'],ds_climate_stacked['glat']])[0])
nlat = jnp.array(ds_climate_stacked['latitude'].values)
nele = jnp.array(ds_climate_stacked['Elevation'].values)

ele_scaler = scenario_for_scaler['ele_scaler']
lat_scaler = scenario_for_scaler['lat_scaler']
nlat_scaled = lat_scaler.transform(nlat.reshape(-1,1))[:,0]
nele_scaled = ele_scaler.transform(nele.reshape(-1,1))[:,0]

# %% Sanity Check
print('Data Shapes: \n',
      f'nx.shape:{nx.shape} \n',
      f'nele_scaled.shape:{nele_scaled.shape} \n',
      f'nlat_scaled.shape:{nlat_scaled.shape} \n',
      )

# %% Generating posterior predictive realisations 1D
generate_posterior_predictive_realisations_hierarchical_mean(
    nx, nele_scaled, nlat_scaled, scenario, 400, 1
)
generate_posterior_predictive_realisations_hierarchical_std(
    nx, nele_scaled, nlat_scaled, scenario, 400, 1  
)

# %% Generate Mean Function Estimates
generate_meanfunction_posterior_predictions(
    nele_scaled, nlat_scaled, scenario, 400, 1  
)


# %% Reformatting predictions

ds_predictions = xr.DataArray(
    data=scenario['mean_truth_posterior_predictive_realisations'],
    dims=["Realisations", "X"]
).to_dataset(name='Mean_Predictions')
ds_predictions['Mean_Bias_Predictions'] = (['Realisations','X'],  scenario['mean_bias_posterior_predictive_realisations'])
ds_predictions['StdDev_Predictions'] = (['Realisations','X'],  scenario['std_truth_posterior_predictive_realisations'])
ds_predictions['StdDev_Bias_Predictions'] = (['Realisations','X'],  scenario['std_bias_posterior_predictive_realisations'])

ds_climate_stacked_reformatted= ds_climate_stacked.where(ds_climate_stacked.month.isin(months),drop=True)
ds_climate_stacked_reformatted['Temperature'] = ds_climate_stacked_reformatted['Temperature']-273.15

ds_predictions = xr.merge([ds_climate_stacked_reformatted,ds_predictions])
ds_predictions = ds_predictions.unstack()
scenario["predictions"] = ds_predictions

# %% Saving the output
outpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real_posterior.npy"
np.save(outpath, scenario)


# %%
