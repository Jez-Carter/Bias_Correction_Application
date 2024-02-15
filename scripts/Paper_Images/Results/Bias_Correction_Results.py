# %% Importing Packages
import numpy as np
import jax
from jax import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
from scipy.spatial import distance

plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 1

legend_fontsize = 6
cm = 1 / 2.54
text_width = 17.68 * cm
page_width = 21.6 * cm

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

results_path = '/home/jez/Bias_Correction_Application/results/Paper_Images/'

# %% Loading data
base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

ds_climate_coarse_june_stacked = scenario['ds_climate_coarse_june_stacked_hr']
ds_climate_coarse_june_stacked_landonly = scenario['ds_climate_coarse_june_stacked_landonly_hr']

posterior_climate = scenario['Mean_Function_Posterior_Climate_hr']

####### Ice Sheet Shapefile ######
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

# %% Computing empirical mean and log variance for climate model

ds_climate_coarse_june_stacked_landonly['Mean Temperature'] = (
    ds_climate_coarse_june_stacked_landonly['Temperature'].mean('Time')
    )

ds_climate_coarse_june_stacked_landonly['Log Var Temperature'] = (
    np.log(ds_climate_coarse_june_stacked_landonly['Temperature'].var('Time'))
    )

# %% Computing the unbiased posterior predictive estimates for the mean and log variance at the climate model locations
exp_meanfunc_unbiased = posterior_climate['meanfunc_prediction_unbiased'].mean(['chain','draw']).data
std_meanfunc_unbiased = posterior_climate['meanfunc_prediction_unbiased'].std(['chain','draw']).data
exp_logvarfunc_unbiased = posterior_climate['logvarfunc_prediction_unbiased'].mean(['chain','draw']).data
std_logvarfunc_unbiased = posterior_climate['logvarfunc_prediction_unbiased'].std(['chain','draw']).data

residual_mean_unbiased = scenario['truth_posterior_predictive_realisations_dualprocess_mean_hr']
residual_logvar_unbiased = scenario['truth_posterior_predictive_realisations_dualprocess_logvar_hr']
exp_residual_mean_unbiased = residual_mean_unbiased.mean(axis=0)
std_residual_mean_unbiased = residual_mean_unbiased.std(axis=0)
exp_residual_logvar_unbiased = residual_logvar_unbiased.mean(axis=0)
std_residual_logvar_unbiased = residual_logvar_unbiased.std(axis=0)

exp_mean_unbiased = exp_meanfunc_unbiased + exp_residual_mean_unbiased
std_mean_unbiased = np.sqrt(std_meanfunc_unbiased**2 + std_residual_mean_unbiased**2)
exp_logvar_unbiased = exp_logvarfunc_unbiased + exp_residual_logvar_unbiased
std_logvar_unbiased = np.sqrt(std_logvarfunc_unbiased**2 + std_residual_logvar_unbiased**2)

ds_climate_coarse_june_stacked_landonly['exp_mean_unbiased'] = (('X'),exp_mean_unbiased)
ds_climate_coarse_june_stacked_landonly['std_mean_unbiased'] = (('X'),std_mean_unbiased)
ds_climate_coarse_june_stacked_landonly['exp_logvar_unbiased'] = (('X'),exp_logvar_unbiased)
ds_climate_coarse_june_stacked_landonly['std_logvar_unbiased'] = (('X'),std_logvar_unbiased)

# %% Nearest Neighbours
ds_aws_june_filtered = scenario['ds_aws_june_filtered']

cx = np.array(np.dstack([ds_climate_coarse_june_stacked_landonly['glon'],
                         ds_climate_coarse_june_stacked_landonly['glat']])[0])

nn_indecies = []
for point in scenario['ox']:
    nn_indecies.append(distance.cdist([point], cx).argmin())

ds_climate_coarse_june_stacked_landonly_nn = ds_climate_coarse_june_stacked_landonly.isel({'X':nn_indecies})
ds_climate_coarse_june_stacked_landonly_nn = ds_climate_coarse_june_stacked_landonly_nn.assign_coords({"Nearest_Station": ("X", ds_aws_june_filtered.Station.data)})
ds_climate_coarse_june_stacked_landonly_nn = ds_climate_coarse_june_stacked_landonly_nn.swap_dims({"X": "Nearest_Station"})

# %% Computing empirical mean and log variance for climate model

ds_climate_coarse_june_stacked_landonly_nn['Mean Temperature'] = (
    ds_climate_coarse_june_stacked_landonly_nn['Temperature'].mean('Time')
    )

ds_climate_coarse_june_stacked_landonly_nn['Log Var Temperature'] = (
    np.log(ds_climate_coarse_june_stacked_landonly_nn['Temperature'].var('Time'))
    )
# %%
ds_climate_coarse_june_stacked_landonly_nn

# %%
ds_climate_coarse_june_stacked_landonly_nn


# %% Loading predictions and creating dataset
posterior_meanfunc = scenario['Mean_Function_Posterior_Climate']

exp_mean_climate = posterior_meanfunc['mean'].mean(['chain','draw']).data
std_mean_climate = posterior_meanfunc['mean'].std(['chain','draw']).data
exp_logvar_climate = posterior_meanfunc['logvar'].mean(['chain','draw']).data
std_logvar_climate = posterior_meanfunc['logvar'].std(['chain','draw']).data

exp_meanfunc_unbiased = posterior_meanfunc['meanfunc_prediction_unbiased'].mean(['chain','draw']).data
std_meanfunc_unbiased = posterior_meanfunc['meanfunc_prediction_unbiased'].std(['chain','draw']).data
exp_logvarfunc_unbiased = posterior_meanfunc['logvarfunc_prediction_unbiased'].mean(['chain','draw']).data
std_logvarfunc_unbiased = posterior_meanfunc['logvarfunc_prediction_unbiased'].std(['chain','draw']).data

posterior_meanfunc['meanfunc_prediction_bias'] = posterior_meanfunc['meanfunc_prediction'] - posterior_meanfunc['meanfunc_prediction_unbiased']
posterior_meanfunc['logvarfunc_prediction_bias'] = posterior_meanfunc['logvarfunc_prediction'] - posterior_meanfunc['logvarfunc_prediction_unbiased']
exp_meanfunc_bias = posterior_meanfunc['meanfunc_prediction_bias'].mean(['chain','draw']).data
std_meanfunc_bias = posterior_meanfunc['meanfunc_prediction_bias'].std(['chain','draw']).data
exp_logvarfunc_bias = posterior_meanfunc['logvarfunc_prediction_bias'].mean(['chain','draw']).data
std_logvarfunc_bias = posterior_meanfunc['logvarfunc_prediction_bias'].std(['chain','draw']).data

residual_mean_unbiased = scenario['truth_posterior_predictive_realisations_dualprocess_mean']
residual_logvar_unbiased = scenario['truth_posterior_predictive_realisations_dualprocess_logvar']
exp_residual_mean_unbiased = residual_mean_unbiased.mean(axis=0)
std_residual_mean_unbiased = residual_mean_unbiased.std(axis=0)
exp_residual_logvar_unbiased = residual_logvar_unbiased.mean(axis=0)
std_residual_logvar_unbiased = residual_logvar_unbiased.std(axis=0)

residual_mean_bias = scenario['bias_posterior_predictive_realisations_dualprocess_mean']
residual_logvar_bias = scenario['bias_posterior_predictive_realisations_dualprocess_logvar']
exp_residual_mean_bias = residual_mean_bias.mean(axis=0)
std_residual_mean_bias = residual_mean_bias.std(axis=0)
exp_residual_logvar_bias = residual_logvar_bias.mean(axis=0)
std_residual_logvar_bias = residual_logvar_bias.std(axis=0)

exp_mean_unbiased = exp_meanfunc_unbiased + exp_residual_mean_unbiased
std_mean_unbiased = np.sqrt(std_meanfunc_unbiased**2 + std_residual_mean_unbiased**2)
exp_logvar_unbiased = exp_logvarfunc_unbiased + exp_residual_logvar_unbiased
std_logvar_unbiased = np.sqrt(std_logvarfunc_unbiased**2 + std_residual_logvar_unbiased**2)

exp_mean_bias = exp_meanfunc_bias + exp_residual_mean_bias
std_mean_bias = np.sqrt(std_meanfunc_bias**2 + std_residual_mean_bias**2)
exp_logvar_bias = exp_logvarfunc_bias + exp_residual_logvar_bias
std_logvar_bias = np.sqrt(std_logvarfunc_bias**2 + std_residual_logvar_bias**2)

ds_climate_coarse_june_stacked_landonly['exp_mean_climate'] = (('X'),exp_mean_climate)
ds_climate_coarse_june_stacked_landonly['std_mean_climate'] = (('X'),std_mean_climate)
ds_climate_coarse_june_stacked_landonly['exp_logvar_climate'] = (('X'),exp_logvar_climate)
ds_climate_coarse_june_stacked_landonly['std_logvar_climate'] = (('X'),std_logvar_climate)

ds_climate_coarse_june_stacked_landonly['exp_mean_unbiased'] = (('X'),exp_mean_unbiased)
ds_climate_coarse_june_stacked_landonly['std_mean_unbiased'] = (('X'),std_mean_unbiased)
ds_climate_coarse_june_stacked_landonly['exp_logvar_unbiased'] = (('X'),exp_logvar_unbiased)
ds_climate_coarse_june_stacked_landonly['std_logvar_unbiased'] = (('X'),std_logvar_unbiased)

ds_climate_coarse_june_stacked_landonly['exp_mean_bias'] = (('X'),exp_mean_bias)
ds_climate_coarse_june_stacked_landonly['std_mean_bias'] = (('X'),std_mean_bias)
ds_climate_coarse_june_stacked_landonly['exp_logvar_bias'] = (('X'),exp_logvar_bias)
ds_climate_coarse_june_stacked_landonly['std_logvar_bias'] = (('X'),std_logvar_bias)

# ds_climate_coarse_june_stacked_landonly['exp_mean_bias'] = (ds_climate_coarse_june_stacked_landonly['exp_mean_climate']-
#                                                             ds_climate_coarse_june_stacked_landonly['exp_mean_unbiased'])
# ds_climate_coarse_june_stacked_landonly['std_mean_bias'] = np.sqrt(ds_climate_coarse_june_stacked_landonly['std_mean_climate']**2-
#                                                                    ds_climate_coarse_june_stacked_landonly['std_mean_unbiased']**2)
# ds_climate_coarse_june_stacked_landonly['exp_logvar_bias'] = (ds_climate_coarse_june_stacked_landonly['exp_logvar_climate']-
#                                                               ds_climate_coarse_june_stacked_landonly['exp_logvar_unbiased'])
# ds_climate_coarse_june_stacked_landonly['std_logvar_bias'] = np.sqrt(ds_climate_coarse_june_stacked_landonly['std_logvar_climate']**2-
#                                                                      ds_climate_coarse_june_stacked_landonly['std_logvar_unbiased']**2)

###### Merging for Coordinate Reasons ######
ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

# %%
ds_climate_coarse_june_stacked

# %% Examining Predictions Spatially

fig, axs = plt.subplots(2,3, figsize=(text_width, text_width*0.4),dpi=300)#,frameon=False)

vars = ['exp_mean_climate',
        'exp_mean_unbiased',
        'exp_mean_bias',
        'exp_logvar_climate',
        'exp_logvar_unbiased',
        'exp_logvar_bias']

cmap = 'jet'

for ax,var in zip(axs.ravel(),vars):
    ds_climate_coarse_june_stacked[var].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap=cmap,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,}
    )

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

# %%
        vmin=vmin_max[0],
        vmax=vmin_max[1],
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':cbar_label}
    )

# %%

# vmins_maxs = [[-65,-15],
#               [0.2,1.2],
#               [1.2,3.0],
#               [0.15,0.4]]

# cmaps = ['jet',
#          'viridis',
#          'jet',
#          'viridis']

cbar_labels = ['Mean Expectation',
               'Mean 1$\sigma$ Uncertainty',
               'Log Variance Expectation',
               'Log Variance 1$\sigma$ Uncertainty']
labels = ['a.','b.','c.','d.']

# %%

###### Climate Model Mean Function Residual Output ######
ds_climate_coarse_june_stacked_landonly['exp_meanfunc_residual_climate'] = (
    ('X'),
    scenario['exp_meanfunc_residual_climate'])
ds_climate_coarse_june_stacked_landonly['var_meanfunc_residual_climate'] = (
    ('X'),
    scenario['var_meanfunc_residual_climate'])
ds_climate_coarse_june_stacked_landonly['exp_logvarfunc_residual_climate'] = (
    ('X'),
    scenario['exp_logvarfunc_residual_climate'])
ds_climate_coarse_june_stacked_landonly['var_logvarfunc_residual_climate'] = (
    ('X'),
    scenario['var_logvarfunc_residual_climate'])

###### Single Process Output ######
ds_climate_coarse_june_stacked_landonly['exp_posterior_predictive_realisations_singleprocess_mean'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_mean'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_posterior_predictive_realisations_singleprocess_mean'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_mean'].std(axis=0))
ds_climate_coarse_june_stacked_landonly['exp_posterior_predictive_realisations_singleprocess_logvar'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_logvar'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_posterior_predictive_realisations_singleprocess_logvar'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_logvar'].std(axis=0))

###### Dual Process Output ######
ds_climate_coarse_june_stacked_landonly['exp_truth_posterior_predictive_realisations_dualprocess_mean'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_mean'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_truth_posterior_predictive_realisations_dualprocess_mean'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_mean'].std(axis=0))
ds_climate_coarse_june_stacked_landonly['exp_bias_posterior_predictive_realisations_dualprocess_mean'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_mean'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_bias_posterior_predictive_realisations_dualprocess_mean'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_mean'].std(axis=0))

ds_climate_coarse_june_stacked_landonly['exp_truth_posterior_predictive_realisations_dualprocess_logvar'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_logvar'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_truth_posterior_predictive_realisations_dualprocess_logvar'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_logvar'].std(axis=0))
ds_climate_coarse_june_stacked_landonly['exp_bias_posterior_predictive_realisations_dualprocess_logvar'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_logvar'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_bias_posterior_predictive_realisations_dualprocess_logvar'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_logvar'].std(axis=0))

###### Merging for Coordinate Reasons ######
ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

####### Nearest Neighbors ######
nn_indecies = []
for point in scenario['ox']:
    nn_indecies.append(distance.cdist([point], scenario['cx']).argmin())

exp_meanfunc_residual_bias_nn = (scenario['exp_meanfunc_residual_climate'][nn_indecies]-
                                 scenario['exp_meanfunc_residual_obs'])
exp_logvarfunc_residual_bias_nn = (scenario['exp_logvarfunc_residual_climate'][nn_indecies]-
                                 scenario['exp_logvarfunc_residual_obs'])

####### Ice Sheet Shapefile ######
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
