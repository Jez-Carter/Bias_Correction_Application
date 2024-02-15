# %% Importing Packages
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
import arviz as az
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gpd
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
ds_climate_coarse_june_stacked_landonly['var_meanfunc_residual'] = (
    ('X'),
    posterior_meanfunc_climate['meanfunc_residual'].var(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['var_logvarfunc_residual'] = (
    ('X'),
    posterior_meanfunc_climate['logvarfunc_residual'].var(['chain','draw']).data)

ds_climate_coarse_june_stacked_landonly['exp_mean'] = (
    ('X'),
    posterior_meanfunc_climate['mean'].mean(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['exp_logvar'] = (
    ('X'),
    np.full(
        ds_climate_coarse_june_stacked_landonly.X.shape,
        posterior_meanfunc_climate['logvar'].mean(['chain','draw']).data
        ))
ds_climate_coarse_june_stacked_landonly['std_mean'] = (
    ('X'),
    posterior_meanfunc_climate['mean'].std(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['std_logvar'] = (
    ('X'),
    np.full(
        ds_climate_coarse_june_stacked_landonly.X.shape,
        posterior_meanfunc_climate['logvar'].std(['chain','draw']).data
        ))
ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

posterior_meanfunc_obs['exp_mean'] = posterior_meanfunc_obs['mean'].mean(['chain','draw'])
posterior_meanfunc_obs['std_mean'] = posterior_meanfunc_obs['mean'].std(['chain','draw'])
posterior_meanfunc_obs['exp_logvar'] = posterior_meanfunc_obs['logvar'].mean(['chain','draw'])
posterior_meanfunc_obs['std_logvar'] = posterior_meanfunc_obs['logvar'].std(['chain','draw'])

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

# Nearest Neighbor difference in residuals
posterior_meanfunc_climate_nn['meanfunc_residual_difference'] = (
    ('chain','draw','X'),
    (posterior_meanfunc_obs['meanfunc_residual'].data-posterior_meanfunc_climate_nn['meanfunc_residual'].data))

posterior_meanfunc_climate_nn['logvarfunc_residual_difference'] = (
    ('chain','draw','X'),
    (posterior_meanfunc_obs['logvarfunc_residual'].data-posterior_meanfunc_climate_nn['logvarfunc_residual'].data))

# %% Spatial Plot of Mean and Log Variance Predictions

fig, axs = plt.subplots(2,2, figsize=(text_width, text_width*0.7),dpi=300)#,frameon=False)

vars = ['exp_mean',
        'exp_logvar',
        'std_mean',
        'std_logvar']
vmins_maxs = [[-65,-15],
              [1.2,3.0],
              [0.4,2.5],
              [0.3,0.8]]
cmaps = ['jet',
         'jet',
         'viridis',
         'viridis']
cbar_labels = ['Mean Expectation',
               'Log Variance Expectation',
               'Mean 2$\sigma$ Uncertainty',
               'Log Variance 2$\sigma$ Uncertainty']
labels = ['a.','b.','c.','d.']

for ax,var,vmin_max,cmap,cbar_label in zip(axs.ravel(),vars,vmins_maxs,cmaps,cbar_labels):

    if '2$\sigma$' in cbar_label:
        ds = 2*ds_climate_coarse_june_stacked[var]
        obs = 2*posterior_meanfunc_obs[var]
    else:
        ds = ds_climate_coarse_june_stacked[var]
        obs = posterior_meanfunc_obs[var]

    ds.unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap=cmap,
        vmin=vmin_max[0],
        vmax=vmin_max[1],
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':cbar_label}
    )
    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0),
        marker="o",
        c=obs.data,
        cmap=cmap,
        vmin=vmin_max[0],
        vmax=vmin_max[1],
        edgecolor="w",
        linewidth=0.4,
    )

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

for ax,title in zip(axs[0,:], ['Mean Parameter','Log-Var. Parameter']):
    ax.set_title(title,fontsize=8)

for ax,label in zip(axs[:,0], ['Expectation',r'$2\sigma$ Uncertainty']):
    ax.set_ylabel(label,fontsize=8)

for ax,label in zip(axs.ravel(),['a.','b.','c.','d.']):
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')

plt.tight_layout()
fig.savefig(f"{results_path}fig15.pdf", dpi=300, bbox_inches="tight")

# %% Spatial Plot of Mean and Log Variance Prediction Variances

fig, axs = plt.subplots(1,2, figsize=(text_width, text_width*0.7),dpi=300)#,frameon=False)

vars = ['std_mean',
        'std_logvar']
cmaps = ['viridis',
         'viridis']
cbar_labels = ['Residual Variance',
               'Log Variance Expectation',
               'Mean 2$\sigma$ Uncertainty',
               'Log Variance 2$\sigma$ Uncertainty']
labels = ['a.','b.','c.','d.']

for ax,var,vmin_max,cmap,cbar_label in zip(axs.ravel(),vars,vmins_maxs,cmaps,cbar_labels):

    if '2$\sigma$' in cbar_label:
        ds = 2*ds_climate_coarse_june_stacked[var]
        obs = 2*posterior_meanfunc_obs[var]
    else:
        ds = ds_climate_coarse_june_stacked[var]
        obs = posterior_meanfunc_obs[var]

    ds.unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap=cmap,
        vmin=vmin_max[0],
        vmax=vmin_max[1],
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':cbar_label}
    )
    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0),
        marker="o",
        c=obs.data,
        cmap=cmap,
        vmin=vmin_max[0],
        vmax=vmin_max[1],
        edgecolor="w",
        linewidth=0.4,
    )

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

for ax,title in zip(axs[0,:], ['Mean Parameter','Log-Var. Parameter']):
    ax.set_title(title,fontsize=8)

for ax,label in zip(axs[:,0], ['Expectation',r'$2\sigma$ Uncertainty']):
    ax.set_ylabel(label,fontsize=8)

for ax,label in zip(axs.ravel(),['a.','b.','c.','d.']):
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')

plt.tight_layout()
# fig.savefig(f"{results_path}fig15.pdf", dpi=300, bbox_inches="tight")

# %% Correlation between expectations for AWS sites and NN climate model output

print(
pearsonr(posterior_meanfunc_obs['mean'].mean(['chain','draw']).data,
         posterior_meanfunc_climate_nn['mean'].mean(['chain','draw']).data)
)

print(
pearsonr(posterior_meanfunc_obs['logvar'].mean(['chain','draw']).data,
         posterior_meanfunc_climate_nn['logvar'].mean(['chain','draw']).data)
)

# %% R2 Scores
print(f'''Observations:
    Mean:
{az.r2_score(posterior_meanfunc_obs['mean'].data[0],
             posterior_meanfunc_obs['meanfunc_prediction'].data[0])}
''')

print(f'''Climate Model Output:
    Mean:
{az.r2_score(posterior_meanfunc_climate['mean'].data[0],
             posterior_meanfunc_climate['meanfunc_prediction'].data[0])}
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
fig, axs = plt.subplots(2,2, figsize=(text_width, text_width*0.7),dpi=300)#,frameon=False)

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
                    'label':'Mean Parameter Residual Expectation'}
    )

    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0),
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
                    'label':'Log-Var. Parameter Residual Expectation'}
    )

    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0),
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
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

for ax in axs[1]:
    ax.set_xlim([0,10])
    ax.set_ylim([-18,-10])

for ax in axs[0]:
    ax.add_patch(Rectangle((0, -18), 10, 8, fc='none',ec='k',ls='dotted'))

con1 = ConnectionPatch(xyA=(0, -18), xyB=(0, -10), coordsA="data", coordsB="data",
                      axesA=axs[0,0], axesB=axs[1,0], color="k",ls='dotted')
con2 = ConnectionPatch(xyA=(10, -18), xyB=(10, -10), coordsA="data", coordsB="data",
                      axesA=axs[0,0], axesB=axs[1,0], color="k",ls='dotted')
con3 = ConnectionPatch(xyA=(0, -18), xyB=(0, -10), coordsA="data", coordsB="data",
                      axesA=axs[0,1], axesB=axs[1,1], color="k",ls='dotted')
con4 = ConnectionPatch(xyA=(10, -18), xyB=(10, -10), coordsA="data", coordsB="data",
                      axesA=axs[0,1], axesB=axs[1,1], color="k",ls='dotted')
axs[1,0].add_artist(con1)
axs[1,0].add_artist(con2)
axs[1,1].add_artist(con3)
axs[1,1].add_artist(con4)

for ax,label in zip(axs.ravel(),['a.','b.','c.','d.']):
    ax.annotate(label,xy=(0.01,1.03),xycoords='axes fraction')

for ax,title in zip(axs[0,:], ['Mean Parameter Residual','Log-Var. Parameter Residual']):
    ax.set_title(title,fontsize=8)

plt.subplots_adjust(hspace=0.1,wspace=0.3)

# plt.tight_layout()
fig.savefig(f"{results_path}fig11.pdf", dpi=300, bbox_inches="tight")

# %% Spatial Plot of Residuals Variances
fig, axs = plt.subplots(2,2, figsize=(text_width, text_width*0.7),dpi=300)#,frameon=False)

for ax in axs[:,0]:
    ds_climate_coarse_june_stacked['var_meanfunc_residual'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='viridis',
        vmin=0,
        vmax=3,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Mean Parameter Residual Expectation'}
    )

    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0),
        marker="o",
        c=posterior_meanfunc_obs['meanfunc_residual'].var(['chain','draw']).data,
        cmap='viridis',
        edgecolor="w",
        linewidth=0.4,
        vmin=0,
        vmax=3,
    )

for ax in axs[:,1]:
    ds_climate_coarse_june_stacked['var_logvarfunc_residual'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='viridis',
        vmin=0,
        vmax=0.2,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Log-Var. Parameter Residual Expectation'}
    )

    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0),
        marker="o",
        c=posterior_meanfunc_obs['logvarfunc_residual'].var(['chain','draw']).data,
        cmap='viridis',
        edgecolor="w",
        linewidth=0.4,
        vmin=0,
        vmax=0.2,
    )

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

for ax in axs[1]:
    ax.set_xlim([0,10])
    ax.set_ylim([-18,-10])

for ax in axs[0]:
    ax.add_patch(Rectangle((0, -18), 10, 8, fc='none',ec='k',ls='dotted'))

con1 = ConnectionPatch(xyA=(0, -18), xyB=(0, -10), coordsA="data", coordsB="data",
                      axesA=axs[0,0], axesB=axs[1,0], color="k",ls='dotted')
con2 = ConnectionPatch(xyA=(10, -18), xyB=(10, -10), coordsA="data", coordsB="data",
                      axesA=axs[0,0], axesB=axs[1,0], color="k",ls='dotted')
con3 = ConnectionPatch(xyA=(0, -18), xyB=(0, -10), coordsA="data", coordsB="data",
                      axesA=axs[0,1], axesB=axs[1,1], color="k",ls='dotted')
con4 = ConnectionPatch(xyA=(10, -18), xyB=(10, -10), coordsA="data", coordsB="data",
                      axesA=axs[0,1], axesB=axs[1,1], color="k",ls='dotted')
axs[1,0].add_artist(con1)
axs[1,0].add_artist(con2)
axs[1,1].add_artist(con3)
axs[1,1].add_artist(con4)

for ax,label in zip(axs.ravel(),['a.','b.','c.','d.']):
    ax.annotate(label,xy=(0.01,1.03),xycoords='axes fraction')

for ax,title in zip(axs[0,:], ['Mean Parameter Residual','Log-Var. Parameter Residual']):
    ax.set_title(title,fontsize=8)

plt.subplots_adjust(hspace=0.1,wspace=0.3)

# plt.tight_layout()
fig.savefig(f"{results_path}fig11.pdf", dpi=300, bbox_inches="tight")

# %%
# %% Spatial Plot of Difference in Residuals
fig, axs = plt.subplots(1,2, figsize=(text_width, text_width*0.35),dpi=300)#,frameon=False)

vars = ['meanfunc_residual_difference',
        'logvarfunc_residual_difference'
        ]
vmins_maxs = [[-10,10],
              [-1.0,1.0]
              ]
cbar_labels = ['Mean Expectation',
               'Mean 1$\sigma$ Uncertainty',
               'Log Variance Expectation',
               'Log Variance 1$\sigma$ Uncertainty']
labels = ['a.','b.','c.','d.']

for ax,var,vmins_max,cbar_label in zip(axs.ravel(),vars,vmins_maxs,cbar_labels):
    plot = ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0),
        marker="o",
        c=posterior_meanfunc_climate_nn[var].mean(['chain','draw']).data,
        cmap='RdBu',
        edgecolor="w",
        linewidth=0.4,
        vmin=vmins_max[0],
        vmax=vmins_max[1],
    )
    fig.colorbar(plot,ax=ax,fraction=0.030,pad=0.04)

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

for ax,label in zip(axs.ravel(),['a.','b.']):
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')

plt.tight_layout()

# %% Test

print(
    posterior_meanfunc_obs['meanfunc_residual'].mean(['chain','draw','Station']).data,
    posterior_meanfunc_obs['logvarfunc_residual'].mean(['chain','draw','Station']).data
)
print(
    posterior_meanfunc_climate['meanfunc_residual'].mean(['chain','draw','X']).data,
    posterior_meanfunc_climate['logvarfunc_residual'].mean(['chain','draw','X']).data
)
print(
    posterior_meanfunc_climate_nn['meanfunc_residual_difference'].mean(['chain','draw','X']).data,
    posterior_meanfunc_climate_nn['logvarfunc_residual_difference'].mean(['chain','draw','X']).data
)
print(
    posterior_meanfunc_climate_nn['meanfunc_residual'].mean(['chain','draw','X']).data,
    posterior_meanfunc_climate_nn['logvarfunc_residual'].mean(['chain','draw','X']).data
)
# posterior_meanfunc_climate_nn['meanfunc_residual_difference'].mean(['chain','draw','X'])
# posterior_meanfunc_climate_nn['logvarfunc_residual_difference'].mean(['chain','draw','X'])