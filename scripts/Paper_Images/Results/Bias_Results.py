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
import numpyro.distributions as dist
from scipy.stats import norm

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
posterior_obs = scenario['Mean_Function_Posterior']

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

# Computing the bias posterior predictive estimates for the mean and log variance at the climate model locations
exp_meanfunc_bias = posterior_climate['meanfunc_prediction_bias'].mean(['chain','draw']).data
std_meanfunc_bias = posterior_climate['meanfunc_prediction_bias'].std(['chain','draw']).data
exp_logvarfunc_bias = posterior_climate['logvarfunc_prediction_bias'].mean(['chain','draw']).data
std_logvarfunc_bias = posterior_climate['logvarfunc_prediction_bias'].std(['chain','draw']).data

residual_mean_bias = scenario['bias_posterior_predictive_realisations_dualprocess_mean_hr']
residual_logvar_bias = scenario['bias_posterior_predictive_realisations_dualprocess_logvar_hr']
exp_residual_mean_bias = residual_mean_bias.mean(axis=0)
std_residual_mean_bias = residual_mean_bias.std(axis=0)
exp_residual_logvar_bias = residual_logvar_bias.mean(axis=0)
std_residual_logvar_bias = residual_logvar_bias.std(axis=0)

exp_mean_bias = exp_meanfunc_bias + exp_residual_mean_bias
std_mean_bias = np.sqrt(std_meanfunc_bias**2 + std_residual_mean_bias**2)
exp_logvar_bias = exp_logvarfunc_bias + exp_residual_logvar_bias
std_logvar_bias = np.sqrt(std_logvarfunc_bias**2 + std_residual_logvar_bias**2)

ds_climate_coarse_june_stacked_landonly['exp_mean_bias'] = (('X'),exp_mean_bias)
ds_climate_coarse_june_stacked_landonly['std_mean_bias'] = (('X'),std_mean_bias)
ds_climate_coarse_june_stacked_landonly['exp_logvar_bias'] = (('X'),exp_logvar_bias)
ds_climate_coarse_june_stacked_landonly['std_logvar_bias'] = (('X'),std_logvar_bias)

ds_climate_coarse_june_stacked_landonly['exp_meanfunc_bias'] = (('X'),exp_meanfunc_bias)
ds_climate_coarse_june_stacked_landonly['exp_residual_mean_bias'] = (('X'),exp_residual_mean_bias)

ds_climate_coarse_june_stacked_landonly['meanfunc_prediction_bias_single'] = (('X'),posterior_climate['meanfunc_prediction_bias'].isel(chain=0,draw=0).data)
ds_climate_coarse_june_stacked_landonly['residual_mean_bias_single'] = (('X'),residual_mean_bias[0])


ds_climate_coarse_june_stacked_landonly['exp_residual_mean_bias'] = (('X'),exp_residual_mean_bias)


###### Merging for Coordinate Reasons ######
ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

# %%

residual_mean_bias.shape
# %%
fig, axs = plt.subplots(1,2, figsize=(text_width, text_width*0.6),dpi=300)#,frameon=False)

ds_climate_coarse_june_stacked['meanfunc_prediction_bias_single'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0],
        cmap='RdBu',
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

ds_climate_coarse_june_stacked['residual_mean_bias_single'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1],
        cmap='RdBu',
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

# %% Examining Truth Prediction

fig, axs = plt.subplots(1,2, figsize=(text_width, text_width*0.6),dpi=300)#,frameon=False)

ds_climate_coarse_june_stacked['exp_mean_unbiased'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0],
        cmap='jet',
        vmin=-55,
        vmax=-10,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

np.sqrt(np.exp(ds_climate_coarse_june_stacked['exp_logvar_unbiased'])).unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1],
        cmap='jet',
        vmin=1,
        vmax=5,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()


# %% Test

ds_climate_coarse_june_stacked['Test Bias Mean'] = (ds_climate_coarse_june_stacked['Mean Temperature'] -
                                                    ds_climate_coarse_june_stacked['exp_mean_unbiased'])

fig, axs = plt.subplots(1,2, figsize=(text_width, text_width*0.6),dpi=300)#,frameon=False)

ds_climate_coarse_june_stacked['Test Bias Mean'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0],
        cmap='RdBu',
        vmin=-5,
        vmax=5,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

ds_climate_coarse_june_stacked['exp_residual_mean_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1],
        cmap='RdBu',
        vmin=-5,
        vmax=5,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

# %% Test 3

ds_climate_coarse_june_stacked['exp_mean_climate'] = (ds_climate_coarse_june_stacked['exp_mean_unbiased'] -
                                                    ds_climate_coarse_june_stacked['exp_mean_bias'])

fig, axs = plt.subplots(1,2, figsize=(text_width, text_width*0.6),dpi=300)#,frameon=False)

ds_climate_coarse_june_stacked['Mean Temperature'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0],
        cmap='RdBu',
        # vmin=-5,
        # vmax=5,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

ds_climate_coarse_june_stacked['exp_mean_climate'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1],
        cmap='RdBu',
        # vmin=-5,
        # vmax=5,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

# %% Examining Bias Predictions Spatially

fig, axs = plt.subplots(2,2, figsize=(text_width, text_width*0.6),dpi=300)#,frameon=False)

ds_climate_coarse_june_stacked['exp_mean_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0,0],
        cmap='RdBu',
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

ds_climate_coarse_june_stacked['exp_logvar_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0,1],
        cmap='RdBu',
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Log-Var. Parameter Bias'}
    )

ds_climate_coarse_june_stacked['std_mean_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1,0],
        cmap='viridis',
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'1$\sigma$ Unc. Mean Parameter Bias'}
    )

ds_climate_coarse_june_stacked['std_logvar_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1,1],
        cmap='viridis',
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'1$\sigma$ Unc. Log-Var. Parameter Bias'}
    )

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

for ax,label in zip(axs.ravel(),['a.','b.','c.','d.']):
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')

plt.tight_layout()

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

# %%

test = (ds_climate_coarse_june_stacked_landonly_nn['Mean Temperature'] -
        ds_climate_coarse_june_stacked_landonly_nn['exp_mean_unbiased'])

stations = ['Henry','Manuela','Butler Island','Byrd','Relay Station','Dome C']

for station in stations:
    print(f'''{station}:{test.sel(Nearest_Station=station).data}:{ds_climate_coarse_june_stacked_landonly_nn['exp_mean_bias'].sel(Nearest_Station=station).data}''')

fig, ax = plt.subplots(1, 1, figsize=(10, 6),dpi=300)#,frameon=False)
ax.hist(test.data,bins=40,alpha=0.6)

# %% Station Locations
fig, ax = plt.subplots(1, 1, figsize=(text_width*0.6, text_width*0.6),dpi=300)#,frameon=False)

all_stations = ds_aws_june_filtered['Station'].data
stations = ['Henry','Manuela','Butler Island','Byrd','Relay Station','Dome C']
other_stations = all_stations[(np.isin(all_stations,stations)==False)]
ds = ds_aws_june_filtered.sel(Station = stations)
ds_others = ds_aws_june_filtered.sel(Station = other_stations)
ds_isolated = ds_climate_coarse_june_stacked_landonly.isel(X=4218)

gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)

ax.scatter(
    ds_others.glon,
    ds_others.glat,
    s=10,
    marker='.',
    edgecolor='k',
    linewidths=0.5,
    alpha=0.7,
)

ax.scatter(
    ds.glon,
    ds.glat,
    s=10,
    marker='*',
    edgecolor='k',
    linewidths=0.5,
    zorder=2
)

ax.scatter(
    ds_isolated.glon,
    ds_isolated.glat,
    s=10,
    marker='+',
    linewidths=0.5,
)

ax.set_title('')
ax.set_axis_off()

offsets = [[10, 5],[-27, -2],[-8, 10],[-8, 10],[-20, 10],[10,-5]]
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

# %% Bias Corrected Time Series
stations = ['Henry','Manuela','Butler Island','Byrd','Relay Station','Dome C','Isolated Location']
labels = ['a.','b.','c.','d.','e.','f.','g.']
fig, axs = plt.subplots(7, 1, figsize=(text_width, text_width),dpi=300)#,frameon=False)

for ax,station,label in zip(axs[:-1],stations[:-1],labels[:-1]):
    ds_climate = ds_climate_coarse_june_stacked_landonly_nn.sel(Nearest_Station=station)
    ds_aws = ds_aws_june_filtered.sel(Station=station)

    cdf_c = norm.cdf(
        ds_climate['Temperature'],
        ds_climate['Mean Temperature'],
        np.sqrt(np.exp(ds_climate['Log Var Temperature'])),
    )

    mean_unbiased_dist = dist.Normal(ds_climate['exp_mean_unbiased'].data,ds_climate['std_mean_unbiased'].data)
    logvar_unbiased_dist = dist.Normal(ds_climate['exp_logvar_unbiased'].data,ds_climate['std_logvar_unbiased'].data)

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
ds_isolated = ds_climate_coarse_june_stacked_landonly.isel(X=4218)
cdf_c = norm.cdf(
    ds_isolated['Temperature'],
    ds_isolated['Mean Temperature'],
    np.sqrt(np.exp(ds_isolated['Log Var Temperature'])),
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


# %% Additional
fig, ax = plt.subplots(1,1, figsize=(text_width, text_width*0.6),dpi=300)#,frameon=False)

ax.scatter(
    ds_aws_june_filtered['Temperature'].mean('Year').data,
    ds_climate_coarse_june_stacked_landonly_nn['exp_mean_unbiased'].data,
    alpha=0.7,
    marker='x',
)

ax.scatter(
    ds_aws_june_filtered['Temperature'].mean('Year').data,
    ds_climate_coarse_june_stacked_landonly_nn['Mean Temperature'].data,
    alpha=0.7,
    marker='+',
)

ax.plot(np.arange(-65,-10,1),
        np.arange(-65,-10,1))

# %%

ds_climate_coarse_june_stacked_landonly_nn['exp_mean_unbiased']