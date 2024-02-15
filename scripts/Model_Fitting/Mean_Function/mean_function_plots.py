# %% Importing Packages
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gpd
import seaborn as sns
from scipy.stats import pearsonr
from numpyro.diagnostics import hpdi
from scipy.spatial import distance

plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1

legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w"))

results_path = '/home/jez/Bias_Correction_Application/results/Paper_Images/'

# %% Loading data
base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

posterior_meanfunc_obs = scenario['Mean_Function_Posterior']
posterior_meanfunc_climate = scenario['Mean_Function_Posterior_Climate']

empirical_mean_obs = np.nanmean(scenario['odata'],axis=0)
empirical_logvar_obs = np.log(np.nanvar(scenario['odata'],axis=0))
empirical_mean_climate = np.nanmean(scenario['cdata'],axis=0)
empirical_logvar_climate = np.log(np.nanvar(scenario['cdata'],axis=0))

ds_climate_coarse_june_stacked = scenario['ds_climate_coarse_june_stacked']
ds_climate_coarse_june_stacked_landonly = scenario['ds_climate_coarse_june_stacked_landonly']
ds_climate_coarse_june_stacked_landonly['exp_meanfunc_residual'] = (
    ('X'),
    posterior_meanfunc_climate['meanfunc_residual'].mean(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['exp_logvarfunc_residual'] = (
    ('X'),
    posterior_meanfunc_climate['logvarfunc_residual'].mean(['chain','draw']).data)
ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

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

# Nearest Neighbors
nn_indecies = []
for point in scenario['ox']:
    nn_indecies.append(distance.cdist([point], scenario['cx']).argmin())

posterior_meanfunc_climate_nn = posterior_meanfunc_climate.isel({'X':nn_indecies})

# %% Summary Mean Function Parameter Estimates Observations
az.summary(posterior_meanfunc_obs[['mean_b0',
                                   'mean_b1',
                                   'mean_b2',
                                   'mean_noise',
                                   'logvar_b0',
                                   'logvar_noise'
                                   ]],hdi_prob=0.95)

# %% Summary Mean Function Parameter Estimates Climate Model
az.summary(posterior_meanfunc_climate[['mean_b0',
                                   'mean_b1',
                                   'mean_b2',
                                   'mean_noise',
                                   'logvar_b0',
                                   'logvar_noise'
                                   ]],hdi_prob=0.95)

# %% R2 Scores
print(f'''Observations:
    Mean:
{az.r2_score(posterior_meanfunc_obs['mean'].data[0],
             posterior_meanfunc_obs['meanfunc_prediction'].data[0])}
    LogVar:
{az.r2_score(posterior_meanfunc_obs['logvar'].data[0],
             posterior_meanfunc_obs['logvarfunc_prediction'].broadcast_like(posterior_meanfunc_obs).data[0])}
''')

print(f'''Climate Model Output:
    Mean:
{az.r2_score(posterior_meanfunc_climate['mean'].data[0],
             posterior_meanfunc_climate['meanfunc_prediction'].data[0])}
    LogVar:
{az.r2_score(posterior_meanfunc_climate['logvar'].data[0],
             posterior_meanfunc_climate['logvarfunc_prediction'].broadcast_like(posterior_meanfunc_climate).data[0])}
''')


# %% Mean Function Prediction against Mean Prediction and Residuals
fig, axes = plt.subplots(2, 2, figsize=(text_width, text_width/2),dpi=1000)#,frameon=False)

for axs,posterior in zip(axes,[posterior_meanfunc_obs,posterior_meanfunc_climate]):

    exp_mean_prediction = posterior['mean'].mean(['chain','draw'])
    hdpi_mean_prediction = hpdi(posterior['mean'],0.95,axis=1)[0]
    err_mean_prediction = hdpi_mean_prediction[1]-exp_mean_prediction

    exp_meanfunc_prediction = posterior['meanfunc_prediction'].mean(['chain','draw'])
    hdpi_meanfunc_prediction = hpdi(posterior['meanfunc_prediction'],0.95,axis=1)[0]
    err_meanfunc_prediction = hdpi_meanfunc_prediction[1]-exp_meanfunc_prediction

    exp_meanfunc_residual = posterior['meanfunc_residual'].mean(['chain','draw'])
    hdpi_meanfunc_residual = hpdi(posterior['meanfunc_residual'],0.95,axis=1)[0]
    err_meanfunc_residual = hdpi_meanfunc_residual[1]-exp_meanfunc_residual

    linspace_mean = np.linspace(exp_mean_prediction.min(),exp_mean_prediction.max(),10)

    ax=axs[0]
    ax.errorbar(x = exp_mean_prediction,
                y = exp_meanfunc_prediction,
                xerr = err_mean_prediction,
                yerr = err_meanfunc_prediction,
                marker='o',
                ms=1,
                ls="none",
                alpha=0.5,
                linewidth=0.4,
                )
    ax.plot(linspace_mean,
        linspace_mean,
        ls='dotted',
        c='k')
    ax.set_xlabel('Mean Temperature Prediction')
    ax.set_ylabel('Mean Function')

    ax=axs[1]
    ax.errorbar(x = exp_mean_prediction,
                y = exp_meanfunc_residual,
                xerr = err_mean_prediction,
                yerr = err_meanfunc_residual,
                marker='o',
                ms=1,
                ls="none",
                linewidth=0.4,
                alpha=0.5
                )
    ax.plot(linspace_mean,
        np.zeros(linspace_mean.shape),
        ls='dotted',
        c='k')
    
    ax.set_xlabel('Mean Temperature Prediction')
    ax.set_ylabel('Mean Function Residual')

for ax,label in zip(axes.ravel(),['a.','b.','c.','d.']):
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')

plt.tight_layout()

# %% LogVar Function Prediction against Mean Prediction and Residuals
fig, axes = plt.subplots(2, 2, figsize=(text_width, text_width/2),dpi=1000)#,frameon=False)

for axs,posterior in zip(axes,[posterior_meanfunc_obs,posterior_meanfunc_climate]):

    exp_logvar_prediction = posterior['logvar'].mean(['chain','draw'])
    hdpi_logvar_prediction = hpdi(posterior['logvar'],0.95,axis=1)[0]
    err_logvar_prediction = hdpi_logvar_prediction[1]-exp_logvar_prediction

    exp_logvarfunc_prediction = posterior['logvarfunc_prediction'].mean(['chain','draw'])
    hdpi_logvarfunc_prediction = hpdi(posterior['logvarfunc_prediction'],0.95,axis=1)[0]
    err_logvarfunc_prediction = hdpi_logvarfunc_prediction[1]-exp_logvarfunc_prediction

    exp_logvarfunc_prediction = np.broadcast_arrays(exp_logvarfunc_prediction,
                                                    exp_logvar_prediction)[0]
    err_logvarfunc_prediction = np.broadcast_arrays(err_logvarfunc_prediction,
                                                    exp_logvar_prediction)[0]

    exp_logvarfunc_residual = posterior['logvarfunc_residual'].mean(['chain','draw'])
    hdpi_logvarfunc_residual = hpdi(posterior['logvarfunc_residual'],0.95,axis=1)[0]
    err_logvarfunc_residual = hdpi_logvarfunc_residual[1]-exp_logvarfunc_residual

    linspace_logvar = np.linspace(exp_logvar_prediction.min(),exp_logvar_prediction.max(),10)

    ax=axs[0]
    ax.errorbar(x = exp_logvar_prediction,
                y = exp_logvarfunc_prediction,
                xerr = err_logvar_prediction,
                yerr = err_logvarfunc_prediction,
                marker='o',
                ms=1,
                ls="none",
                alpha=0.5,
                linewidth=0.4,
                )
    ax.plot(linspace_logvar,
        linspace_logvar,
        ls='dotted',
        c='k')
    ax.set_xlabel('Log Variance Temperature Prediction')
    ax.set_ylabel('Log Variance Function')

    ax=axs[1]
    ax.errorbar(x = exp_logvar_prediction,
                y = exp_logvarfunc_residual,
                xerr = err_logvar_prediction,
                yerr = err_logvarfunc_residual,
                marker='o',
                ms=1,
                ls="none",
                linewidth=0.4,
                alpha=0.5
                )
    ax.plot(linspace_logvar,
        np.zeros(linspace_logvar.shape),
        ls='dotted',
        c='k')
    
    ax.set_xlabel('Log Variance Temperature Prediction')
    ax.set_ylabel('Log Variance Function Residual')

for ax,label in zip(axes.ravel(),['a.','b.','c.','d.']):
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')

plt.tight_layout()

# %% Residuals Correlation between Observations and Nearest Neighbour Climate Model
fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=1000)#,frameon=False)

exp_meanfunc_residual_obs = posterior_meanfunc_obs['meanfunc_residual'].mean(['chain','draw'])
hdpi_meanfunc_residual_obs = hpdi(posterior_meanfunc_obs['meanfunc_residual'],0.95,axis=1)[0]
err_meanfunc_residual_obs = hdpi_meanfunc_residual_obs[1]-exp_meanfunc_residual_obs

exp_meanfunc_residual_climate_nn = posterior_meanfunc_climate_nn['meanfunc_residual'].mean(['chain','draw'])
hdpi_meanfunc_residual_climate_nn = hpdi(posterior_meanfunc_climate_nn['meanfunc_residual'],0.95,axis=1)[0]
err_meanfunc_residual_climate_nn = hdpi_meanfunc_residual_climate_nn[1]-exp_meanfunc_residual_climate_nn

exp_logvarfunc_residual_obs = posterior_meanfunc_obs['logvarfunc_residual'].mean(['chain','draw'])
hdpi_logvarfunc_residual_obs = hpdi(posterior_meanfunc_obs['logvarfunc_residual'],0.95,axis=1)[0]
err_logvarfunc_residual_obs = hdpi_logvarfunc_residual_obs[1]-exp_logvarfunc_residual_obs

exp_logvarfunc_residual_climate_nn = posterior_meanfunc_climate_nn['logvarfunc_residual'].mean(['chain','draw'])
hdpi_logvarfunc_residual_climate_nn = hpdi(posterior_meanfunc_climate_nn['logvarfunc_residual'],0.95,axis=1)[0]
err_logvarfunc_residual_climate_nn = hdpi_logvarfunc_residual_climate_nn[1]-exp_logvarfunc_residual_climate_nn

linspace_mean_residual = np.linspace(exp_meanfunc_residual_obs.min(),exp_meanfunc_residual_obs.max(),10)
linspace_logvar_residual = np.linspace(exp_logvarfunc_residual_obs.min(),exp_logvarfunc_residual_obs.max(),10)

ax=axs[0]
ax.plot(linspace_mean_residual,
    linspace_mean_residual,
    ls='dotted',
    c='k')
ax.errorbar(x = exp_meanfunc_residual_obs,
            y = exp_meanfunc_residual_climate_nn,
            xerr = err_meanfunc_residual_obs,
            yerr = err_meanfunc_residual_climate_nn,
            marker='o',
            ms=1,
            ls="none",
            linewidth=0.4,
            alpha=0.5
            )
ax.set_xlabel('Mean Temperature Residual AWS')
ax.set_ylabel('Mean Temperature Residual Climate Model NN')

ax=axs[1]
ax.plot(linspace_logvar_residual,
    linspace_logvar_residual,
    ls='dotted',
    c='k')
ax.errorbar(x = exp_logvarfunc_residual_obs,
            y = exp_logvarfunc_residual_climate_nn,
            xerr = err_logvarfunc_residual_obs,
            yerr = err_logvarfunc_residual_climate_nn,
            marker='o',
            ms=1,
            ls="none",
            linewidth=0.4,
            alpha=0.5
            )
ax.set_xlabel('Log Variance Temperature Residual AWS')
ax.set_ylabel('Log Variance Temperature Residual Climate Model NN')

for ax,label in zip(axs.ravel(),['a.','b.']):
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')

# %% Spatial Plot of Residuals
fig, axs = plt.subplots(2,2, figsize=(10, 8),dpi=300)#,frameon=False)

for ax in axs[:,0]:
    ds_climate_coarse_june_stacked['exp_meanfunc_residual'].unstack().plot.pcolormesh(
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
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*5,
        marker="o",
        c=posterior_meanfunc_obs['meanfunc_residual'].mean(['chain','draw']).data,
        cmap='RdBu',
        edgecolor="w",
        linewidth=0.4,
        vmin=-10,
        vmax=10,
    )

for ax in axs[:,1]:
    ds_climate_coarse_june_stacked['exp_logvarfunc_residual'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='RdBu',
        vmin=-1,
        vmax=1,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Temperature Monthly LogVar Residual'}
    )

    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*5,
        marker="o",
        c=posterior_meanfunc_obs['logvarfunc_residual'].mean(['chain','draw']).data,
        cmap='RdBu',
        edgecolor="w",
        linewidth=0.4,
        vmin=-1,
        vmax=1,
    )

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)

for ax in axs[1]:
    ax.set_xlim([0,10])
    ax.set_ylim([-20,-10])

plt.tight_layout()

# %%
hdpi_meanfunc_residual_bias_nn.shape
# %%
meanfunc_residual_bias_nn = (posterior_meanfunc_climate_nn['meanfunc_residual'].data
                             -posterior_meanfunc_obs['meanfunc_residual'].data)
exp_meanfunc_residual_bias_nn = meanfunc_residual_bias_nn.mean(axis=(0,1))
hdpi_meanfunc_residual_bias_nn = hpdi(meanfunc_residual_bias_nn,0.95,axis=1)[0]
err_meanfunc_residual_bias_nn = hdpi_meanfunc_residual_bias_nn[1]-exp_meanfunc_residual_bias_nn

# %% Residuals Correlation between Observations and Bias with Nearest Neighbour Climate Model
fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=1000)#,frameon=False)

exp_meanfunc_residual_obs = posterior_meanfunc_obs['meanfunc_residual'].mean(['chain','draw'])
hdpi_meanfunc_residual_obs = hpdi(posterior_meanfunc_obs['meanfunc_residual'],0.95,axis=1)[0]
err_meanfunc_residual_obs = hdpi_meanfunc_residual_obs[1]-exp_meanfunc_residual_obs

exp_meanfunc_residual_climate_nn = posterior_meanfunc_climate_nn['meanfunc_residual'].mean(['chain','draw'])
hdpi_meanfunc_residual_climate_nn = hpdi(posterior_meanfunc_climate_nn['meanfunc_residual'],0.95,axis=1)[0]
err_meanfunc_residual_climate_nn = hdpi_meanfunc_residual_climate_nn[1]-exp_meanfunc_residual_climate_nn

meanfunc_residual_bias_nn = (posterior_meanfunc_climate_nn['meanfunc_residual'].data
                             -posterior_meanfunc_obs['meanfunc_residual'].data)
exp_meanfunc_residual_bias_nn = meanfunc_residual_bias_nn.mean(axis=(0,1))
hdpi_meanfunc_residual_bias_nn = hpdi(meanfunc_residual_bias_nn,0.95,axis=1)[0]
err_meanfunc_residual_bias_nn = hdpi_meanfunc_residual_bias_nn[1]-exp_meanfunc_residual_bias_nn


exp_logvarfunc_residual_obs = posterior_meanfunc_obs['logvarfunc_residual'].mean(['chain','draw'])
hdpi_logvarfunc_residual_obs = hpdi(posterior_meanfunc_obs['logvarfunc_residual'],0.95,axis=1)[0]
err_logvarfunc_residual_obs = hdpi_logvarfunc_residual_obs[1]-exp_logvarfunc_residual_obs

exp_logvarfunc_residual_climate_nn = posterior_meanfunc_climate_nn['logvarfunc_residual'].mean(['chain','draw'])
hdpi_logvarfunc_residual_climate_nn = hpdi(posterior_meanfunc_climate_nn['logvarfunc_residual'],0.95,axis=1)[0]
err_logvarfunc_residual_climate_nn = hdpi_logvarfunc_residual_climate_nn[1]-exp_logvarfunc_residual_climate_nn

logvarfunc_residual_bias_nn = (posterior_meanfunc_climate_nn['logvarfunc_residual'].data
                             -posterior_meanfunc_obs['logvarfunc_residual'].data)
exp_logvarfunc_residual_bias_nn = logvarfunc_residual_bias_nn.mean(axis=(0,1))
hdpi_logvarfunc_residual_bias_nn = hpdi(logvarfunc_residual_bias_nn,0.95,axis=1)[0]
err_logvarfunc_residual_bias_nn = hdpi_logvarfunc_residual_bias_nn[1]-exp_logvarfunc_residual_bias_nn


linspace_mean_residual = np.linspace(exp_meanfunc_residual_obs.min(),exp_meanfunc_residual_obs.max(),10)
linspace_logvar_residual = np.linspace(exp_logvarfunc_residual_obs.min(),exp_logvarfunc_residual_obs.max(),10)

ax=axs[0]
# ax.plot(linspace_mean_residual,
#     linspace_mean_residual,
#     ls='dotted',
#     c='k')
ax.errorbar(x = exp_meanfunc_residual_obs,
            y = exp_meanfunc_residual_bias_nn,
            xerr = err_meanfunc_residual_obs,
            yerr = err_meanfunc_residual_bias_nn,
            marker='o',
            ms=1,
            ls="none",
            linewidth=0.4,
            alpha=0.5
            )
ax.set_xlabel('Mean Temperature Residual AWS')
ax.set_ylabel('Mean Temperature Residual Bias NN')

ax=axs[1]
# ax.plot(linspace_logvar_residual,
#     linspace_logvar_residual,
#     ls='dotted',
#     c='k')
ax.errorbar(x = exp_logvarfunc_residual_obs,
            y = exp_logvarfunc_residual_bias_nn,
            xerr = err_logvarfunc_residual_obs,
            yerr = err_logvarfunc_residual_bias_nn,
            marker='o',
            ms=1,
            ls="none",
            linewidth=0.4,
            alpha=0.5
            )
ax.set_xlabel('Log Variance Temperature Residual AWS')
ax.set_ylabel('Log Variance Temperature Residual Climate Model NN')

for ax,label in zip(axs.ravel(),['a.','b.']):
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')

plt.tight_layout()

# %% 

posterior_meanfunc_climate_nn['meanfunc_residual'].mean(['chain','draw']) - posterior_meanfunc_climate_nn['meanfunc_residual'].mean(['chain','draw'])

# %% Nearest Neighbors
nn_indecies = []
for point in scenario['ox']:
    nn_indecies.append(distance.cdist([point], scenario['cx']).argmin())

posterior_nn = posterior_climate.isel({'CM Grid Cell':nn_indecies})

vars = [i for i in list(posterior_nn) if 'residual' in i]
posterior_nn = posterior_nn[vars]

posterior_nn['mean_prediction_exp_residual'] = (
    ('CM Grid Cell'),
    posterior['mean_prediction_exp_residual'].data
    )

posterior_nn['logvar_prediction_exp_residual'] = (
    ('CM Grid Cell'),
    posterior['logvar_prediction_exp_residual'].data
    )

posterior_nn['mean_prediction_exp_residual_standardised'] = (
    ('CM Grid Cell'),
    posterior['mean_prediction_exp_residual_standardised'].data
    )

posterior_nn['logvar_prediction_exp_residual_standardised'] = (
    ('CM Grid Cell'),
    posterior['logvar_prediction_exp_residual_standardised'].data
    )

posterior_nn = posterior_nn.assign_coords(Stations=("CM Grid Cell", ds_aws_stacked_jan_filtered.Station.values))


# %% Correlation Between Residuals
# posterior_meanfunc_obs['meanfunc_residual']#.mean(['chain','draw'])

posterior_meanfunc_climate['meanfunc_residual']

# %% Mean Prediction against Elevation and Latitude

fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=1000)#,frameon=False)

exp_mean_prediction = posterior_meanfunc_obs['mean'].mean(['chain','draw'])
hdpi_mean_prediction = hpdi(posterior_meanfunc_obs['mean'],0.95,axis=1)[0]
err_mean_prediction = hdpi_mean_prediction[1]-exp_mean_prediction

ele_obs = scenario['oele']
ele_obs_scaled = scenario['oele_scaled']
linspace_ele = np.linspace(ele_obs.min(),ele_obs.max(),10)
linspace_ele_scaled = scenario['ele_scaler'].transform(linspace_ele.reshape(-1,1))[:,0]

lat_obs = scenario['olat']
lat_obs_scaled = scenario['olat_scaled']
linspace_lat = np.linspace(lat_obs.min(),lat_obs.max(),10)
linspace_lat_scaled = scenario['lat_scaler'].transform(linspace_lat.reshape(-1,1))[:,0]

meanfunc_ele_prediction = (posterior_meanfunc_obs['mean_b0'] + 
                               posterior_meanfunc_obs['mean_b1']*
                               xr.DataArray(linspace_ele_scaled))
exp_meanfunc_ele_prediction = meanfunc_ele_prediction.mean(['chain','draw'])
hdpi_meanfunc_ele_prediction = hpdi(meanfunc_ele_prediction,0.95,axis=1)[0]

meanfunc_lat_prediction = (posterior_meanfunc_obs['mean_b0'] + 
                               posterior_meanfunc_obs['mean_b2']*
                               xr.DataArray(linspace_lat_scaled))
exp_meanfunc_lat_prediction = meanfunc_lat_prediction.mean(['chain','draw'])
hdpi_meanfunc_lat_prediction = hpdi(meanfunc_lat_prediction,0.95,axis=1)[0]

ax=axs[0]

ax.scatter(ele_obs,
        exp_mean_prediction,
        marker='+',
        # ms=2,
)

ax.plot(linspace_ele,
        exp_meanfunc_ele_prediction,
        color = 'k',
        linestyle = 'dotted')

ax.fill_between(x = linspace_ele,
                y1 = hdpi_meanfunc_ele_prediction[0],
                y2 = hdpi_meanfunc_ele_prediction[1],
                alpha = 0.3,
                color = 'k',
                linestyle = 'None',
)

ax=axs[1]

ax.scatter(lat_obs,
        exp_mean_prediction,
        marker='+',
        # ms=2,
)

ax.plot(linspace_lat,
        exp_meanfunc_lat_prediction,
        color = 'k',
        linestyle = 'dotted')

ax.fill_between(x = linspace_lat,
                y1 = hdpi_meanfunc_lat_prediction[0],
                y2 = hdpi_meanfunc_lat_prediction[1],
                alpha = 0.3,
                color = 'k',
                linestyle = 'None',
)

# %% Mean Prediction against Elevation and Latitude

fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=1000)#,frameon=False)

exp_mean_prediction = posterior_meanfunc_climate['mean'].mean(['chain','draw'])
hdpi_mean_prediction = hpdi(posterior_meanfunc_climate['mean'],0.95,axis=1)[0]
err_mean_prediction = hdpi_mean_prediction[1]-exp_mean_prediction

ele_climate = scenario['cele']
ele_climate_scaled = scenario['cele_scaled']
linspace_ele = np.linspace(ele_climate.min(),ele_climate.max(),10)
linspace_ele_scaled = scenario['ele_scaler'].transform(linspace_ele.reshape(-1,1))[:,0]

lat_climate = scenario['clat']
lat_climate_scaled = scenario['clat_scaled']
linspace_lat = np.linspace(lat_climate.min(),lat_climate.max(),10)
linspace_lat_scaled = scenario['lat_scaler'].transform(linspace_lat.reshape(-1,1))[:,0]

meanfunc_ele_prediction = (posterior_meanfunc_climate['mean_b0'] + 
                               posterior_meanfunc_climate['mean_b1']*
                               xr.DataArray(linspace_ele_scaled))
exp_meanfunc_ele_prediction = meanfunc_ele_prediction.mean(['chain','draw'])
hdpi_meanfunc_ele_prediction = hpdi(meanfunc_ele_prediction,0.95,axis=1)[0]

meanfunc_lat_prediction = (posterior_meanfunc_climate['mean_b0'] + 
                               posterior_meanfunc_climate['mean_b2']*
                               xr.DataArray(linspace_lat_scaled))
exp_meanfunc_lat_prediction = meanfunc_lat_prediction.mean(['chain','draw'])
hdpi_meanfunc_lat_prediction = hpdi(meanfunc_lat_prediction,0.95,axis=1)[0]

ax=axs[0]

ax.scatter(ele_climate,
        exp_mean_prediction,
        marker='+',
        # ms=2,
)

ax.plot(linspace_ele,
        exp_meanfunc_ele_prediction,
        color = 'k',
        linestyle = 'dotted')

ax.fill_between(x = linspace_ele,
                y1 = hdpi_meanfunc_ele_prediction[0],
                y2 = hdpi_meanfunc_ele_prediction[1],
                alpha = 0.3,
                color = 'k',
                linestyle = 'None',
)

ax=axs[1]

ax.scatter(lat_climate,
        exp_mean_prediction,
        marker='+',
        # ms=2,
)

ax.plot(linspace_lat,
        exp_meanfunc_lat_prediction,
        color = 'k',
        linestyle = 'dotted')

ax.fill_between(x = linspace_lat,
                y1 = hdpi_meanfunc_lat_prediction[0],
                y2 = hdpi_meanfunc_lat_prediction[1],
                alpha = 0.3,
                color = 'k',
                linestyle = 'None',
)

# %%

fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=1000)#,frameon=False)

ax = axs[0]
ax.scatter(lat_climate,
           ele_climate)
ax = axs[1]
ax.scatter(lat_climate,
           exp_mean_prediction)

# %%
meanfunc_ele_prediction = (posterior_meanfunc_obs['mean_b0'] + 
                               posterior_meanfunc_obs['mean_b1']*
                               xr.DataArray(linspace_ele_scaled))
exp_meanfunc_ele_prediction = meanfunc_ele_prediction.mean(['chain','draw'])
hdpi_meanfunc_ele_prediction = hpdi(meanfunc_ele_prediction,0.95,axis=1)[0]

ax.fill_between(x = linspace_ele,
                y1 = hdpi_meanfunc_ele_prediction[0],
                y2 = hdpi_meanfunc_ele_prediction[1],
)

# %%
hdpi_meanfunc_ele_prediction[1]


# %%
xr.DataArray(linspace_ele_scaled)

# %%
posterior_meanfunc_obs

# %%
scenario['ele_scaler'].transform(linspace_ele.reshape(-1,1))[:,0]

# %%
ele_obs_scaled.shape

# %%
ele_scaler = scenario['ele_scaler']

# %%
scenario['ele_scaler'].transform()

# %% 
np.linspace(ele_obs.min(),ele_obs.max(),10)

# %%
oele_scaled = ele_scaler.fit_transform(oele.reshape(-1,1))[:,0]


# %%
posterior_meanfunc_obs['oele_scaled']*posterior_meanfunc_obs['mean_b1']



# %%
posterior_meanfunc_obs


# %%


    residuals_mean[idx], y, xerr=err[idx], marker="o", ms=5, mew=4, ls="none", alpha=0.8
)
# %%
exp_meanfunc_prediction = posterior_meanfunc_obs['meanfunc_prediction'].mean(['chain','draw'])
hdpi_meanfunc_prediction = hpdi(posterior_meanfunc_obs['meanfunc_prediction'],0.95,axis=1)[0]
err_meanfunc_prediction = hdpi_meanfunc_prediction[1]-exp_meanfunc_prediction


# %%
hdpi_meanfunc_prediction[1]-exp_meanfunc_prediction


# %%
err_meanfunc_prediction = hdpi_meanfunc_prediction[1]-


- residuals_mean

# %%
posterior_meanfunc_obs['meanfunc_prediction']


# %%

ax.scatter(x=empirical_mean_obs,
           y = posterior_meanfunc_obs['meanfunc_prediction'].mean(['chain','draw']).data,
           marker='+',
           alpha=1.0,
           label='AWS',
           )

# %%
posterior_meanfunc_obs['meanfunc_prediction']


# %%
az.summary(posterior_meanfunc_obs, hdi_prob=0.95)


# %%
    df = az.summary(scenario["mcmc"].posterior, hdi_prob=0.95)


# %%

# %%

empirical_mean_obs

# %% Plotting the Expectation against Empirical Values 
fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=1000)#,frameon=False)

ax=axs[0]
ax.scatter(x=posterior['means'],
           y = posterior['mean_prediction'].mean(['chain','draw']).data,
           marker='+',
           alpha=1.0,
           label='AWS',
           )
ax.scatter(x=posterior_climate['cmeans'],
           y = posterior_climate['cmean_prediction'].mean(['chain','draw']).data,
           marker='o',
           alpha=0.1,
           label='Climate Model',
)
mean_range = np.linspace(posterior['means'].min(),posterior['means'].max(),20)
ax.plot(mean_range,mean_range,linestyle='dotted',color='k',alpha=0.7)
ax.set_xlabel('Mean Function Mean Scaled Prediction')
ax.set_ylabel('Mean Scaled Empirical Value')

ax=axs[1]
ax.scatter(x=posterior['logvars'],
           y = posterior['logvar_prediction'].mean(['chain','draw']).data,
           marker='+',
           alpha=1.0,
           label='AWS',
           )
ax.scatter(x=posterior_climate['clogvars'],
           y = posterior_climate['clogvar_prediction'].mean(['chain','draw']).data,
           marker='o',
           alpha=0.1,
           label='Climate Model',
           )
logvar_range = np.linspace(posterior['logvars'].min(),posterior['logvars'].max(),20)
ax.plot(logvar_range,logvar_range,linestyle='dotted',color='k',alpha=0.7)
ax.set_xlabel('Mean Function LogVar Scaled Prediction')
ax.set_ylabel('LogVar Scaled Empirical Value')

for ax in axs:
    ax.legend()