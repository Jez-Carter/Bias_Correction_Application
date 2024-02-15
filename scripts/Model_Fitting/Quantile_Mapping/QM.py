# %% Importing Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
from scipy.spatial import distance
import numpyro.distributions as dist
from scipy.stats import norm

plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1
legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm

import jax
rng_key = jax.random.PRNGKey(1)
jax.config.update("jax_enable_x64", True)

# %% Loading data
base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

posterior_meanfunc = scenario['Mean_Function_Posterior_Climate']
posterior_meanfunc_station_locations = scenario['Mean_Function_Posterior']

ds_aws_june_filtered = scenario['ds_aws_june_filtered']
ds_climate_coarse_june_stacked_landonly = scenario['ds_climate_coarse_june_stacked_landonly']

###### Ice Sheet Shapefile ######
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
icehsheet_main_shapefile = f'{shapefiles_path}/cst10_polygon.shp'

gdf_icesheet = gpd.read_file(icehsheet_shapefile)
gdf_icesheet_main = gpd.read_file(icehsheet_main_shapefile)
gdf_icesheet_main = gdf_icesheet_main.explode().iloc[[61]]
gdf_icesheet_main = gdf_icesheet_main.reset_index().drop(columns=['level_0','level_1'])

rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)


# %% Loading predictions and creating dataset
exp_mean_climate = posterior_meanfunc['mean'].mean(['chain','draw']).data
std_mean_climate = posterior_meanfunc['mean'].std(['chain','draw']).data
exp_logvar_climate = posterior_meanfunc['logvar'].mean(['chain','draw']).data
std_logvar_climate = posterior_meanfunc['logvar'].std(['chain','draw']).data

exp_meanfunc_unbiased = posterior_meanfunc['meanfunc_prediction_unbiased'].mean(['chain','draw']).data
std_meanfunc_unbiased = posterior_meanfunc['meanfunc_prediction_unbiased'].std(['chain','draw']).data
exp_logvarfunc_unbiased = posterior_meanfunc['logvarfunc_prediction_unbiased'].mean(['chain','draw']).data
std_logvarfunc_unbiased = posterior_meanfunc['logvarfunc_prediction_unbiased'].std(['chain','draw']).data

residual_mean_unbiased = scenario['truth_posterior_predictive_realisations_dualprocess_mean']
residual_logvar_unbiased = scenario['truth_posterior_predictive_realisations_dualprocess_logvar']
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
ds_climate_coarse_june_stacked_landonly['exp_mean_climate'] = (('X'),exp_mean_climate)
ds_climate_coarse_june_stacked_landonly['exp_logvar_climate'] = (('X'),exp_logvar_climate)

# Nearest Neighbours
nn_indecies = []
for point in scenario['ox']:
    nn_indecies.append(distance.cdist([point], scenario['cx']).argmin())

ds_climate_coarse_june_stacked_landonly_nn = ds_climate_coarse_june_stacked_landonly.isel({'X':nn_indecies})
ds_climate_coarse_june_stacked_landonly_nn = ds_climate_coarse_june_stacked_landonly_nn.assign_coords({"Nearest_Station": ("X", ds_aws_june_filtered.Station.data)})
ds_climate_coarse_june_stacked_landonly_nn = ds_climate_coarse_june_stacked_landonly_nn.swap_dims({"X": "Nearest_Station"})

# %% Station Locations Test
exp_meanfunc_unbiased_station_locations = posterior_meanfunc_station_locations['meanfunc_prediction'].mean(['chain','draw']).data
std_meanfunc_unbiased_station_locations = posterior_meanfunc_station_locations['meanfunc_prediction'].std(['chain','draw']).data
exp_logvarfunc_unbiased_station_locations = posterior_meanfunc_station_locations['logvarfunc_prediction'].mean(['chain','draw']).data
std_logvarfunc_unbiased_station_locations = posterior_meanfunc_station_locations['logvarfunc_prediction'].std(['chain','draw']).data

residual_mean_unbiased_station_locations = scenario['truth_posterior_predictive_realisations_dualprocess_mean_station_locations']
residual_logvar_unbiased_station_locations = scenario['truth_posterior_predictive_realisations_dualprocess_logvar_station_locations']
exp_residual_mean_unbiased_station_locations = residual_mean_unbiased_station_locations.mean(axis=0)
std_residual_mean_unbiased_station_locations = residual_mean_unbiased_station_locations.std(axis=0)
exp_residual_logvar_unbiased_station_locations = residual_logvar_unbiased_station_locations.mean(axis=0)
std_residual_logvar_unbiased_station_locations = residual_logvar_unbiased_station_locations.std(axis=0)

exp_mean_unbiased_station_locations = exp_meanfunc_unbiased_station_locations + exp_residual_mean_unbiased_station_locations
std_mean_unbiased_station_locations = np.sqrt(std_meanfunc_unbiased_station_locations**2 + std_residual_mean_unbiased_station_locations**2)
exp_logvar_unbiased_station_locations = exp_logvarfunc_unbiased_station_locations + exp_residual_logvar_unbiased_station_locations
std_logvar_unbiased_station_locations = np.sqrt(std_logvarfunc_unbiased_station_locations**2 + std_residual_logvar_unbiased_station_locations**2)

ds_climate_coarse_june_stacked_landonly_nn['exp_mean_unbiased_station_locations'] = (('Nearest_Station'),exp_mean_unbiased_station_locations)
ds_climate_coarse_june_stacked_landonly_nn['std_mean_unbiased_station_locations'] = (('Nearest_Station'),std_mean_unbiased_station_locations)
ds_climate_coarse_june_stacked_landonly_nn['exp_logvar_unbiased_station_locations'] = (('Nearest_Station'),exp_logvar_unbiased_station_locations)
ds_climate_coarse_june_stacked_landonly_nn['std_logvar_unbiased_station_locations'] = (('Nearest_Station'),std_logvar_unbiased_station_locations)

# %% Station Locations
fig, ax = plt.subplots(1, 1, figsize=(text_width/2, text_width/2),dpi=300)#,frameon=False)

stations = ['Henry','Manuela','Butler Island','Byrd','Relay Station','Dome C']
ds = ds_aws_june_filtered.sel(Station = stations)
ds_isolated = ds_climate_coarse_june_stacked_landonly.isel(X=318)

ax.scatter(
    ds.glon,
    ds.glat,
    s=10,
    marker='*',
    edgecolor='k',
    linewidths=0.5,
)

ax.scatter(
    ds_isolated.glon,
    ds_isolated.glat,
    s=10,
    marker='*',
    edgecolor='k',
    linewidths=0.5,
)

gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)

ax.set_title('')
ax.set_axis_off()

offsets = [[10, 5],[10, 5],[-8, 10],[-8, 10],[-20, -15],[0,10]]
for station,offset in zip(stations,offsets):
    ds_station = ds.sel(Station = station)
    ax.annotate(
        f'{ds_station.Station.data}',
        xy=(ds_station.glon, ds_station.glat), xycoords='data',
        xytext=(offset[0],offset[1]), textcoords='offset points',
        arrowprops=dict(arrowstyle="->"),
        fontsize=4)
    
ax.annotate(
    f'Isolated Location',
    xy=(ds_isolated.glon, ds_isolated.glat), xycoords='data',
    xytext=(-25,10), textcoords='offset points',
    arrowprops=dict(arrowstyle="->"),
    fontsize=4)
    
plt.tight_layout()

# %%


# %% Bias Corrected Time Series
stations = ['Henry','Manuela','Butler Island','Byrd','Relay Station','Dome C','Isolated Location']
labels = ['a.','b.','c.','d.','e.','f.','g.']
fig, axs = plt.subplots(7, 1, figsize=(text_width, text_width),dpi=300)#,frameon=False)

for ax,station,label in zip(axs[:-1],stations[:-1],labels[:-1]):
    ds_climate = ds_climate_coarse_june_stacked_landonly_nn.sel(Nearest_Station=station)
    ds_aws = ds_aws_june_filtered.sel(Station=station)

    cdf_c = norm.cdf(
        ds_climate['Temperature'],
        ds_climate['exp_mean_climate'],
        np.sqrt(np.exp(ds_climate['exp_logvar_climate'])),
    )

    mean_unbiased_dist = dist.Normal(ds_climate['exp_mean_unbiased_station_locations'].data,ds_climate['std_mean_unbiased_station_locations'].data)
    logvar_unbiased_dist = dist.Normal(ds_climate['exp_logvar_unbiased_station_locations'].data,ds_climate['std_logvar_unbiased_station_locations'].data)

    mean_unbiased_samples = mean_unbiased_dist.sample(rng_key,(1000,))
    logvar_unbiased_samples = logvar_unbiased_dist.sample(rng_key,(1000,))

    c_corrected = norm.ppf(
        cdf_c,
        mean_unbiased_samples.reshape(-1, 1),
        np.sqrt(np.exp(logvar_unbiased_samples)).reshape(-1, 1),
    )

    ax.annotate(label+station,xy=(0.01,1.02),xycoords='axes fraction')
    ds_climate['Temperature'].plot(x='year',
                               ax=ax,
                               marker='+',
                               linestyle="-",
                               linewidth=0.8,
                               zorder=2,
                               label='Climate Model Ouput')
    ds_aws['Temperature'].plot(x='Year',
                            ax=ax,
                            marker='x',
                            linestyle="-",
                            linewidth=0.8,
                            zorder=2,
                            label='Nearest AWS Ouput')
    ax.plot(ds_climate['year'],
        c_corrected.mean(axis=0),
        color='k',
        marker='+',
        linestyle="-",
        linewidth=0.8,
        zorder=1,
        label='Bias Corrected Ouput Expectation')
    ax.fill_between(
        ds_climate['year'],
        c_corrected.mean(axis=0)
        + 3 * c_corrected.std(axis=0),
        c_corrected.mean(axis=0)
        - 3 * c_corrected.std(axis=0),
        interpolate=True,
        color="k",
        alpha=0.5,
        label="Bias Corrected Output Uncertainty 3$\sigma$",
        linewidth=0.5,
        facecolor="none",
        edgecolor="k",
        linestyle=(0, (5, 2)),
    )
    for corrected_timeseries in c_corrected[::10]:
        ax.plot(
            ds_climate['year'],
            corrected_timeseries,
            color="k",
            alpha=0.2,
            linestyle="-",
            linewidth=0.2,
            zorder=1,
        )

ax=axs[-1]
ds_isolated = ds_climate_coarse_june_stacked_landonly.isel(X=318)
cdf_c = norm.cdf(
    ds_isolated['Temperature'],
    ds_isolated['exp_mean_climate'],
    np.sqrt(np.exp(ds_isolated['exp_logvar_climate'])),
)

mean_unbiased_dist = dist.Normal(ds_isolated['exp_mean_unbiased'].data,ds_isolated['std_mean_unbiased'].data)
logvar_unbiased_dist = dist.Normal(ds_isolated['exp_logvar_unbiased'].data,ds_isolated['std_logvar_unbiased'].data)

mean_unbiased_samples = mean_unbiased_dist.sample(rng_key,(1000,))
logvar_unbiased_samples = logvar_unbiased_dist.sample(rng_key,(1000,))

c_corrected = norm.ppf(
    cdf_c,
    mean_unbiased_samples.reshape(-1, 1),
    np.sqrt(np.exp(logvar_unbiased_samples)).reshape(-1, 1),
)

ax.annotate('g. Isolated Location',xy=(0.01,1.02),xycoords='axes fraction')

ds_isolated['Temperature'].plot(x='year',
                               ax=ax,
                               marker='+',
                               linestyle="-",
                               linewidth=0.8,
                               zorder=2,
                               label='Climate Model Ouput')

ax.plot(ds_isolated['year'],
    c_corrected.mean(axis=0),
    color='k',
    marker='+',
    linestyle="-",
    linewidth=0.8,
    zorder=1,
    label='Bias Corrected Ouput Expectation')
ax.fill_between(
    ds_isolated['year'],
    c_corrected.mean(axis=0)
    + 3 * c_corrected.std(axis=0),
    c_corrected.mean(axis=0)
    - 3 * c_corrected.std(axis=0),
    interpolate=True,
    color="k",
    alpha=0.5,
    label="Bias Corrected Output Uncertainty 3$\sigma$",
    linewidth=0.5,
    facecolor="none",
    edgecolor="k",
    linestyle=(0, (5, 2)),
)
for corrected_timeseries in c_corrected[::10]:
    ax.plot(
        ds_isolated['year'],
        corrected_timeseries,
        color="k",
        alpha=0.2,
        linestyle="-",
        linewidth=0.2,
        zorder=1,
    )

for ax in axs:
    ax.set_title('')
    ax.set_xlim([1978,2022])

for ax in axs[:-1]:
    ax.set_xlabel('')
    ax.set_xticklabels('')

handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    fontsize=legend_fontsize,
    bbox_to_anchor=(0.5, -0.02),
    ncols=4,
    loc=10,
)
plt.tight_layout()

# %%
