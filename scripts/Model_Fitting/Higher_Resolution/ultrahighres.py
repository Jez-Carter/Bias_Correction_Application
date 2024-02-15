import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import timeit

import jax
import jax.numpy as jnp
rng_key = jax.random.PRNGKey(1)
jax.config.update("jax_enable_x64", True)

plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1

legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm

from src.spatial.dualprocess_prediction_functions import (
    generate_posterior_predictive_realisations_dualprocess_mean,
    generate_posterior_predictive_realisations_dualprocess_logvar,
)

from src.spatial.singleprocess_model_functions import (
    generate_posterior_predictive_realisations_singleprocess_mean,
    generate_posterior_predictive_realisations_singleprocess_logvar,
)

base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
cordex_inpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'

scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

ds_climate = xr.open_dataset(cordex_inpath)
ele_scaler = scenario['ele_scaler']
lat_scaler = scenario['lat_scaler']

posterior = scenario['Mean_Function_Posterior']
posterior_climate = scenario['Mean_Function_Posterior_Climate']

Station = scenario['ds_aws_june_filtered']['Station']
X = scenario['ds_climate_coarse_june_stacked_landonly']['X']

ds_climate = ds_climate.set_coords(("LSM", "Elevation"))
ds_climate = ds_climate.sel(Model='MAR(ERA5)')
ds_climate['Temperature'] = ds_climate['Temperature']-273.15
ds_climate_june = ds_climate.where(ds_climate.month==6,drop=True)
ds_climate_june_stacked = ds_climate_june.stack(X=('grid_latitude', 'grid_longitude'))
ds_climate_june_stacked_landonly = ds_climate_june_stacked.where(ds_climate_june_stacked.LSM>0.8).dropna('X')

ds_climate_coarse = ds_climate.coarsen(grid_latitude=4,grid_longitude=4).mean()
ds_climate_coarse_june = ds_climate_coarse.where(ds_climate_coarse.month==6,drop=True)
ds_climate_coarse_june_stacked = ds_climate_coarse_june.stack(X=('grid_latitude', 'grid_longitude'))
ds_climate_coarse_june_stacked_landonly = ds_climate_coarse_june_stacked.where(ds_climate_coarse_june_stacked.LSM>0.8).dropna('X')

clat = jnp.array(ds_climate_coarse_june_stacked_landonly['latitude'].values)
cele = jnp.array(ds_climate_coarse_june_stacked_landonly['Elevation'].values)

cele_scaled = ele_scaler.transform(cele.reshape(-1,1))[:,0]
clat_scaled = lat_scaler.transform(clat.reshape(-1,1))[:,0]

ds_climate_coarse_june_stacked_landonly = ds_climate_coarse_june_stacked_landonly.assign_coords(Elevation_Scaled=("X", cele_scaled))
ds_climate_coarse_june_stacked_landonly = ds_climate_coarse_june_stacked_landonly.assign_coords(Latitude_Scaled=("X", clat_scaled))

posterior_climate_uhr = posterior_climate.drop_dims('X')
posterior_climate_uhr = posterior_climate_uhr.assign_coords(Elevation_Scaled=("X", cele_scaled))
posterior_climate_uhr = posterior_climate_uhr.assign_coords(Latitude_Scaled=("X", clat_scaled))

posterior_climate_uhr['meanfunc_prediction'] = (posterior_climate_uhr['mean_b0']
                                + posterior_climate_uhr['mean_b1']*posterior_climate_uhr['Elevation_Scaled']
                                + posterior_climate_uhr['mean_b2']*posterior_climate_uhr['Latitude_Scaled'])
posterior_climate_uhr['logvarfunc_prediction'] = (posterior_climate_uhr['logvar_b0'])

posterior_climate_uhr['meanfunc_prediction_unbiased'] = (posterior['mean_b0']
                                + posterior['mean_b1']*posterior_climate_uhr['Elevation_Scaled']
                                + posterior['mean_b2']*posterior_climate_uhr['Latitude_Scaled'])
posterior_climate_uhr['logvarfunc_prediction_unbiased'] = (posterior['logvar_b0'])

posterior_climate_uhr['meanfunc_prediction_bias'] = (posterior_climate_uhr['meanfunc_prediction'] - 
                                                 posterior_climate_uhr['meanfunc_prediction_unbiased'])
posterior_climate_uhr['logvarfunc_prediction_bias'] = (posterior_climate_uhr['logvarfunc_prediction'] - 
                                                   posterior_climate_uhr['logvarfunc_prediction_unbiased'])

scenario['Mean_Function_Posterior_Climate_uhr'] = posterior_climate_uhr

cx = jnp.array(np.dstack([ds_climate_coarse_june_stacked_landonly['glon'],
                          ds_climate_coarse_june_stacked_landonly['glat']])[0])

starttime = timeit.default_timer()
t_realisations = generate_posterior_predictive_realisations_singleprocess_mean(
        cx,
        scenario,
        40,
        1,
        rng_key
)
print("Time Taken:", timeit.default_timer() - starttime)

scenario["posterior_predictive_realisations_singleprocess_mean_uhr"] = t_realisations

starttime = timeit.default_timer()
t_realisations = generate_posterior_predictive_realisations_singleprocess_logvar(
        cx,
        scenario,
        40,
        1,
        rng_key
)
print("Time Taken:", timeit.default_timer() - starttime)

scenario["posterior_predictive_realisations_singleprocess_logvar_uhr"] = t_realisations


starttime = timeit.default_timer()
t_realisations, b_realisations = generate_posterior_predictive_realisations_dualprocess_mean(
        cx,
        scenario,
        40,
        1,
        rng_key
)
print("Time Taken:", timeit.default_timer() - starttime)

scenario["truth_posterior_predictive_realisations_dualprocess_mean_uhr"] = t_realisations
scenario["bias_posterior_predictive_realisations_dualprocess_mean_uhr"] = b_realisations

starttime = timeit.default_timer()
t_realisations, b_realisations = generate_posterior_predictive_realisations_dualprocess_logvar(
        cx,
        scenario,
        40,
        1,
        rng_key
)
print("Time Taken:", timeit.default_timer() - starttime)

scenario["truth_posterior_predictive_realisations_dualprocess_logvar_uhr"] = t_realisations
scenario["bias_posterior_predictive_realisations_dualprocess_logvar_uhr"] = b_realisations

ds_climate_coarse_june_stacked_landonly['exp_truth_posterior_predictive_realisations_dualprocess_mean_uhr'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_mean_uhr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_truth_posterior_predictive_realisations_dualprocess_mean_uhr'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_mean_uhr'].std(axis=0))
ds_climate_coarse_june_stacked_landonly['exp_bias_posterior_predictive_realisations_dualprocess_mean_uhr'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_mean_uhr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_bias_posterior_predictive_realisations_dualprocess_mean_uhr'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_mean_uhr'].std(axis=0))

ds_climate_coarse_june_stacked_landonly['exp_truth_posterior_predictive_realisations_singleprocess_mean_uhr'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_mean_uhr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_truth_posterior_predictive_realisations_singleprocess_mean_uhr'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_mean_uhr'].std(axis=0))
ds_climate_coarse_june_stacked_landonly['exp_truth_posterior_predictive_realisations_singleprocess_logvar_uhr'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_logvar_uhr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_truth_posterior_predictive_realisations_singleprocess_logvar_uhr'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_logvar_uhr'].std(axis=0))

ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

scenario['ds_climate_coarse_june_stacked_uhr'] = ds_climate_coarse_june_stacked
scenario['ds_climate_coarse_june_stacked_landonly_uhr'] = ds_climate_coarse_june_stacked_landonly

scenario_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'
np.save(scenario_outpath, scenario)