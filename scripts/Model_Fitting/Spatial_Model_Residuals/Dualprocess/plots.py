# %% Importing Packages
import numpy as np
import jax
from jax import random
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
from scipy.spatial import distance

plt.rcParams["lines.markersize"] = 3
plt.rcParams["lines.linewidth"] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

# %% Loading data
base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

ds_climate_coarse_june_stacked = scenario['ds_climate_coarse_june_stacked']
ds_climate_coarse_june_stacked_landonly = scenario['ds_climate_coarse_june_stacked_landonly']

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

ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

# Nearest Neighbors
nn_indecies = []
for point in scenario['ox']:
    nn_indecies.append(distance.cdist([point], scenario['cx']).argmin())

exp_meanfunc_residual_bias_nn = (scenario['exp_meanfunc_residual_climate'][nn_indecies]-
                                 scenario['exp_meanfunc_residual_obs'])
exp_logvarfunc_residual_bias_nn = (scenario['exp_logvarfunc_residual_climate'][nn_indecies]-
                                 scenario['exp_logvarfunc_residual_obs'])


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

# %% Spatial Plot of Predictions and Uncertainty (Mean Truth and Bias)
fig, axes = plt.subplots(2,2, figsize=(10, 8),dpi=300)#,frameon=False)

components = ['truth','bias']
exp_vminmaxs = [[-10,10],[-5,5]]
std_vminmaxs = [[2,5],[1,3.5]]

for axs,component,exp_vminmax,std_vminmax in zip(axes,components,exp_vminmaxs,std_vminmaxs):
    ax=axs[0]
    ds_climate_coarse_june_stacked[f'exp_{component}_posterior_predictive_realisations_dualprocess_mean'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='RdBu',
        vmin=exp_vminmax[0],
        vmax=exp_vminmax[1],
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Temperature Monthly Mean Residual'}
    )

    if component == 'truth':
        obs = scenario['exp_meanfunc_residual_obs']
    elif component == 'bias':
        obs = exp_meanfunc_residual_bias_nn

    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
        marker="o",
        c=obs,
        cmap='RdBu',
        edgecolor="w",
        linewidth=0.6,
        vmin=exp_vminmax[0],
        vmax=exp_vminmax[1],
    )

    ax=axs[1]
    (ds_climate_coarse_june_stacked[f'std_{component}_posterior_predictive_realisations_dualprocess_mean']*2).unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='viridis',
        vmin=std_vminmax[0],
        vmax=std_vminmax[1],
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Temperature Monthly Mean Residual'}
    )

    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
        marker="o",
        facecolor="none",
        edgecolor="w",
        linewidth=0.6,
    )

    for ax in axs:
        gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)

plt.tight_layout()

# %% Spatial Plot of Predictions and Uncertainty (LogVar Truth and Bias)
fig, axes = plt.subplots(2,2, figsize=(10, 8),dpi=300)#,frameon=False)

components = ['truth','bias']
exp_vminmaxs = [[-1,1],[-1,1]]
std_vminmaxs = [[0,0.3],[0,0.3]]

for axs,component,exp_vminmax,std_vminmax in zip(axes,components,exp_vminmaxs,std_vminmaxs):
    ax=axs[0]
    ds_climate_coarse_june_stacked[f'exp_{component}_posterior_predictive_realisations_dualprocess_logvar'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='RdBu',
        vmin=exp_vminmax[0],
        vmax=exp_vminmax[1],
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Temperature Monthly Mean Residual'}
    )

    if component == 'truth':
        obs = scenario['exp_logvarfunc_residual_obs']
    elif component == 'bias':
        obs = exp_logvarfunc_residual_bias_nn

    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
        marker="o",
        c=obs,
        cmap='RdBu',
        edgecolor="w",
        linewidth=0.6,
        vmin=exp_vminmax[0],
        vmax=exp_vminmax[1],
    )

    ax=axs[1]
    (ds_climate_coarse_june_stacked[f'std_{component}_posterior_predictive_realisations_dualprocess_logvar']*2).unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='viridis',
        vmin=std_vminmax[0],
        vmax=std_vminmax[1],
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Temperature Monthly Mean Residual'}
    )

    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
        marker="o",
        facecolor="none",
        edgecolor="w",
        linewidth=0.6,
    )

    for ax in axs:
        gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)

plt.tight_layout()

# %% Spatial Plot of Expectation for Truth Residual
fig, axs = plt.subplots(1,2, figsize=(10, 8),dpi=300)#,frameon=False)

ax=axs[0]
ds_climate_coarse_june_stacked['exp_truth_posterior_predictive_realisations_dualprocess_mean'].unstack().plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    cmap='RdBu',
    vmin=-10,
    vmax=10,
    cbar_kwargs = {'fraction':0.030,
                'pad':0.04,
                'label':'Temperature Monthly Mean Residual'}
)

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
    marker="o",
    c=scenario['exp_meanfunc_residual_obs'],
    cmap='RdBu',
    edgecolor="w",
    linewidth=0.6,
    vmin=-10,
    vmax=10,
)

ax=axs[1]
(ds_climate_coarse_june_stacked['std_truth_posterior_predictive_realisations_dualprocess_mean']*2).unstack().plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    cmap='viridis',
    vmin=0,
    vmax=12,
    cbar_kwargs = {'fraction':0.030,
                'pad':0.04,
                'label':'Temperature Monthly Mean Residual'}
)

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
    marker="o",
    # c=scenario['exp_meanfunc_residual_obs'],
    # cmap='RdBu',
    # s=10,
    facecolor="none",
    edgecolor="w",
    linewidth=0.6,
    # alpha=0,
    vmin=-10,
    vmax=10,
)

for ax in axs:
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)

plt.tight_layout()


# %% Spatial Plot of Expectation for Bias Residual
fig, axs = plt.subplots(1,2, figsize=(10, 8),dpi=300)#,frameon=False)

ax=axs[0]
ds_climate_coarse_june_stacked['exp_bias_posterior_predictive_realisations_dualprocess_mean'].unstack().plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    cmap='RdBu',
    vmin=-5,
    vmax=5,
    cbar_kwargs = {'fraction':0.030,
                'pad':0.04,
                'label':'Temperature Monthly Mean Residual'}
)

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
    marker="o",
    c=exp_meanfunc_residual_bias_nn,
    cmap='RdBu',
    edgecolor="w",
    linewidth=0.6,
    vmin=-5,
    vmax=5,
)

ax=axs[1]
(ds_climate_coarse_june_stacked['std_bias_posterior_predictive_realisations_dualprocess_mean']*2).unstack().plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    cmap='viridis',
    vmin=1,
    vmax=3.5,
    cbar_kwargs = {'fraction':0.030,
                'pad':0.04,
                'label':'Temperature Monthly Mean Residual'}
)

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
    marker="o",
    # c=scenario['exp_meanfunc_residual_obs'],
    # cmap='RdBu',
    # s=10,
    facecolor="none",
    edgecolor="w",
    linewidth=0.6,
    # alpha=0,
    vmin=-10,
    vmax=10,
)

for ax in axs:
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)

plt.tight_layout()

# %% Spatial Plot of Expectation for Truth Residual
fig, axs = plt.subplots(1,2, figsize=(10, 8),dpi=300)#,frameon=False)

ax=axs[0]
ds_climate_coarse_june_stacked['exp_posterior_predictive_realisations_singleprocess_logvar'].unstack().plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    cmap='RdBu',
    vmin=-1,
    vmax=1,
    cbar_kwargs = {'fraction':0.030,
                'pad':0.04,
                'label':'Temperature Monthly Mean Residual'}
)

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
    marker="o",
    c=scenario['exp_logvarfunc_residual_obs'],
    cmap='RdBu',
    edgecolor="w",
    linewidth=0.6,
    vmin=-1,
    vmax=1,
)

ax=axs[1]
(ds_climate_coarse_june_stacked['std_posterior_predictive_realisations_singleprocess_logvar']*2).unstack().plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    cmap='viridis',
    vmin=0,
    vmax=1,
    cbar_kwargs = {'fraction':0.030,
                'pad':0.04,
                'label':'Temperature Monthly Mean Residual'}
)

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
    marker="o",
    # c=scenario['exp_meanfunc_residual_obs'],
    # cmap='RdBu',
    # s=10,
    facecolor="none",
    edgecolor="w",
    linewidth=0.6,
    # alpha=0,
    vmin=-10,
    vmax=10,
)

for ax in axs:
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)

plt.tight_layout()

# %%
