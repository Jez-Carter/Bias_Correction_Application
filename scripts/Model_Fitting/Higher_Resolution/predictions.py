# %% Importing Packages
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import arviz as az
import xarray as xr
import timeit
import geopandas as gpd
import cartopy.crs as ccrs

from src.helper_functions import run_inference

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

# %% Loading data
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


# %%
scenario['ds_climate_coarse_june_stacked_landonly_uhr']


# %% Adjusting resolution of climate data
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

# %% Generating mean function predictions at the climate model grid locations

posterior_climate_hr = posterior_climate.drop_dims('X')
posterior_climate_hr = posterior_climate_hr.assign_coords(Elevation_Scaled=("X", cele_scaled))
posterior_climate_hr = posterior_climate_hr.assign_coords(Latitude_Scaled=("X", clat_scaled))

posterior_climate_hr['meanfunc_prediction'] = (posterior_climate_hr['mean_b0']
                                + posterior_climate_hr['mean_b1']*posterior_climate_hr['Elevation_Scaled']
                                + posterior_climate_hr['mean_b2']*posterior_climate_hr['Latitude_Scaled'])
posterior_climate_hr['logvarfunc_prediction'] = (posterior_climate_hr['logvar_b0'])

posterior_climate_hr['meanfunc_prediction_unbiased'] = (posterior['mean_b0']
                                + posterior['mean_b1']*posterior_climate_hr['Elevation_Scaled']
                                + posterior['mean_b2']*posterior_climate_hr['Latitude_Scaled'])
posterior_climate_hr['logvarfunc_prediction_unbiased'] = (posterior['logvar_b0'])

posterior_climate_hr['meanfunc_prediction_bias'] = (posterior_climate_hr['meanfunc_prediction'] - 
                                                 posterior_climate_hr['meanfunc_prediction_unbiased'])
posterior_climate_hr['logvarfunc_prediction_bias'] = (posterior_climate_hr['logvarfunc_prediction'] - 
                                                   posterior_climate_hr['logvarfunc_prediction_unbiased'])

scenario['Mean_Function_Posterior_Climate_hr'] = posterior_climate_hr

# %% Generating residual predictions at the climate model grid locations

cx = jnp.array(np.dstack([ds_climate_coarse_june_stacked_landonly['glon'],
                          ds_climate_coarse_june_stacked_landonly['glat']])[0])

starttime = timeit.default_timer()
t_realisations, b_realisations = generate_posterior_predictive_realisations_dualprocess_mean(
        cx,
        scenario,
        100,
        10,
        rng_key
)
print("Time Taken:", timeit.default_timer() - starttime)

scenario["truth_posterior_predictive_realisations_dualprocess_mean_hr"] = t_realisations
scenario["bias_posterior_predictive_realisations_dualprocess_mean_hr"] = b_realisations

starttime = timeit.default_timer()
t_realisations, b_realisations = generate_posterior_predictive_realisations_dualprocess_logvar(
        cx,
        scenario,
        100,
        10,
        rng_key
)
print("Time Taken:", timeit.default_timer() - starttime)

scenario["truth_posterior_predictive_realisations_dualprocess_logvar_hr"] = t_realisations
scenario["bias_posterior_predictive_realisations_dualprocess_logvar_hr"] = b_realisations


# %% Merging data for plotting

ds_climate_coarse_june_stacked_landonly['exp_truth_posterior_predictive_realisations_dualprocess_mean_hr'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_mean_hr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_truth_posterior_predictive_realisations_dualprocess_mean_hr'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_mean_hr'].std(axis=0))
ds_climate_coarse_june_stacked_landonly['exp_bias_posterior_predictive_realisations_dualprocess_mean_hr'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_mean_hr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly['std_bias_posterior_predictive_realisations_dualprocess_mean_hr'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_mean_hr'].std(axis=0))

ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])




# %% Saving Output

scenario['ds_climate_coarse_june_stacked_hr'] = ds_climate_coarse_june_stacked
scenario['ds_climate_coarse_june_stacked_landonly_hr'] = ds_climate_coarse_june_stacked_landonly

scenario_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'
np.save(scenario_outpath, scenario)


# %%
# %% Loading data
base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
cordex_inpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'

scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

ds_climate_coarse_june_stacked = scenario['ds_climate_coarse_june_stacked_hr']


# %% MeanFunction Prediction 

from src.slide_functions import background_map_rotatedcoords, rotated_coord_system, markersize_legend
ds_climate_coarse_june_stacked_landonly['exp_meanfunc_prediction_unbiased'] = (
    ('X'),
    posterior_climate_hr['meanfunc_prediction_unbiased'].mean(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['std_meanfunc_prediction_unbiased'] = (
    ('X'),
    posterior_climate_hr['meanfunc_prediction_unbiased'].std(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['exp_meanfunc_prediction_bias'] = (
    ('X'),
    posterior_climate_hr['meanfunc_prediction_bias'].mean(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['std_meanfunc_prediction_bias'] = (
    ('X'),
    posterior_climate_hr['meanfunc_prediction_bias'].std(['chain','draw']).data)
ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

# %% Meanfunc Prediction Unbiased Expectation
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)

posterior_climate_hr

# ds_climate_coarse_june_stacked['exp_meanfunc_prediction_unbiased'].unstack().plot.pcolormesh(
ds_climate_coarse_june_stacked['exp_meanfunc_prediction_bias'].unstack().plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    # vmin=-55,
    # vmax=-10,
    cmap='jet',
    # add_colorbar=False,
    cbar_kwargs = {'fraction':0.030,
                'pad':0.02,
                'label':'Mean Function Prediction'}
)

ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')
                       
plt.tight_layout()

# %% Meanfunc Prediction Unbiased Uncertainty
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)

posterior_climate_hr

# ds_climate_coarse_june_stacked['std_meanfunc_prediction_unbiased'].unstack().plot.pcolormesh(
ds_climate_coarse_june_stacked['std_meanfunc_prediction_bias'].unstack().plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    vmin=0.8,
    vmax=2.0,
    cmap='viridis',
    # add_colorbar=False,
    cbar_kwargs = {'fraction':0.030,
                'pad':0.02,
                'label':'Mean Function Prediction'}
)

ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')
                       
plt.tight_layout()






# %% Spatial Plot of Unbiased and Biased Mean Residual Predictions
fig, axs = plt.subplots(2,2, figsize=(text_width, text_width*0.7),dpi=300)#,frameon=False)

kwargs = {'x':'glon',
          'y':'glat'}
exp_kwargs = {'vmin':-15,
              'vmax':15,
              'cmap':'RdBu'}
cbar_kwargs = {'fraction':0.030,
               'pad':0.01,
               'label':''}

ds_climate_coarse_june_stacked[f'exp_truth_posterior_predictive_realisations_dualprocess_mean_hr'].unstack().plot.pcolormesh(
        ax=axs[0,0],
        **kwargs,
        vmin=-15,
        vmax=15,
        cmap='RdBu',
        cbar_kwargs = dict(cbar_kwargs, label='Shared Process Exp. $r_{\mu_Y}$')
    )

ds_climate_coarse_june_stacked[f'exp_bias_posterior_predictive_realisations_dualprocess_mean_hr'].unstack().plot.pcolormesh(
        ax=axs[0,1],
        **kwargs,
        vmin=-3,
        vmax=3,
        cmap='RdBu',
        cbar_kwargs = dict(cbar_kwargs, label='Shared Process Exp. $r_{\mu_B}$')
    )

ds_climate_coarse_june_stacked[f'std_truth_posterior_predictive_realisations_dualprocess_mean_hr'].unstack().plot.pcolormesh(
        ax=axs[1,0],
        **kwargs,
        vmin=1,
        vmax=4,
        cmap='viridis',
        cbar_kwargs = dict(cbar_kwargs, label='Shared Process Exp. $r_{\mu_Y}$')
    )

ds_climate_coarse_june_stacked[f'std_bias_posterior_predictive_realisations_dualprocess_mean_hr'].unstack().plot.pcolormesh(
        ax=axs[1,1],
        **kwargs,
        vmin=0.8,
        vmax=2,
        cmap='viridis',
        cbar_kwargs = dict(cbar_kwargs, label='Shared Process Exp. $r_{\mu_B}$')
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
