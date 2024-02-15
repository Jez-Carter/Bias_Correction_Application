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

ds_climate_coarse_june_stacked = scenario['ds_climate_coarse_june_stacked']
ds_climate_coarse_june_stacked_landonly = scenario['ds_climate_coarse_june_stacked_landonly']

ds_climate_coarse_june_stacked_uhr = scenario['ds_climate_coarse_june_stacked_uhr']
ds_climate_coarse_june_stacked_landonly_uhr = scenario['ds_climate_coarse_june_stacked_landonly_uhr']

# %%
###### Single Process Output ######
ds_climate_coarse_june_stacked_landonly_uhr['exp_posterior_predictive_realisations_singleprocess_mean'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_mean_uhr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly_uhr['std_posterior_predictive_realisations_singleprocess_mean'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_mean_uhr'].std(axis=0))
ds_climate_coarse_june_stacked_landonly_uhr['exp_posterior_predictive_realisations_singleprocess_logvar'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_logvar_uhr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly_uhr['std_posterior_predictive_realisations_singleprocess_logvar'] = (
    ('X'),
    scenario['posterior_predictive_realisations_singleprocess_logvar_uhr'].std(axis=0))

###### Dual Process Output ######
ds_climate_coarse_june_stacked_landonly_uhr['exp_truth_posterior_predictive_realisations_dualprocess_mean'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_mean_uhr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly_uhr['std_truth_posterior_predictive_realisations_dualprocess_mean'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_mean_uhr'].std(axis=0))
ds_climate_coarse_june_stacked_landonly_uhr['exp_bias_posterior_predictive_realisations_dualprocess_mean'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_mean_uhr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly_uhr['std_bias_posterior_predictive_realisations_dualprocess_mean'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_mean_uhr'].std(axis=0))

ds_climate_coarse_june_stacked_landonly_uhr['exp_truth_posterior_predictive_realisations_dualprocess_logvar'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_logvar_uhr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly_uhr['std_truth_posterior_predictive_realisations_dualprocess_logvar'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_logvar_uhr'].std(axis=0))
ds_climate_coarse_june_stacked_landonly_uhr['exp_bias_posterior_predictive_realisations_dualprocess_logvar'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_logvar_uhr'].mean(axis=0))
ds_climate_coarse_june_stacked_landonly_uhr['std_bias_posterior_predictive_realisations_dualprocess_logvar'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_logvar_uhr'].std(axis=0))

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


# %%

###### Merging for Coordinate Reasons ######
ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])
# ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked_uhr,ds_climate_coarse_june_stacked_landonly_uhr])


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

# %%
ds_climate_coarse_june_stacked

# %% Comparison of Mean Residual Predictions
fig, axs = plt.subplots(2, 2, figsize=(text_width, text_width/1.5),dpi=300)#,frameon=False)

kwargs = {'x':'glon',
          'y':'glat'}
exp_kwargs = {'vmin':-10,
              'vmax':10,
              'cmap':'RdBu'}
std_kwargs = {'vmin':4,
              'vmax':15,
              'cmap':'viridis'}
cbar_kwargs = {'fraction':0.030,
               'pad':0.02}

ds_climate_coarse_june_stacked[f'exp_posterior_predictive_realisations_singleprocess_mean'].unstack().plot.pcolormesh(
        ax=axs[0,0],
        **kwargs,
        **exp_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Single Process Exp. $r_{\mu_Y}$')
    )

ds_climate_coarse_june_stacked[f'exp_truth_posterior_predictive_realisations_dualprocess_mean'].unstack().plot.pcolormesh(
        ax=axs[0,1],
        **kwargs,
        **exp_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Shared Process Exp. $r_{\mu_Y}$')
    )

(2*ds_climate_coarse_june_stacked[f'std_posterior_predictive_realisations_singleprocess_mean']).unstack().plot.pcolormesh(
        ax=axs[1,0],
        **kwargs,
        **std_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Single Process Uncert. $r_{\mu_Y}$')
    )

(2*ds_climate_coarse_june_stacked[f'std_truth_posterior_predictive_realisations_dualprocess_mean']).unstack().plot.pcolormesh(
        ax=axs[1,1],
        **kwargs,
        **std_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Shared Process Uncert. $r_{\mu_Y}$')
    )

for ax in axs.ravel()[:2]:
    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
        marker="o",
        c=scenario['exp_meanfunc_residual_obs'],
        edgecolor="w",
        linewidth=0.6,
        **exp_kwargs,
    )

for ax,label in zip(axs.ravel(),['a.','b.','c.','d.','e.']):
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    # ax.axis("off")
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')
    ax.set_xlim([-26.5,26.5])

plt.tight_layout()
fig.savefig(f"{results_path}fig12.pdf", dpi=300, bbox_inches="tight")

# %% Comparison of Log Var. Residual Predictions
fig = plt.figure(figsize=(text_width, text_width),dpi=300) 
gs = gridspec.GridSpec(3, 4,
                       hspace=0.08,
                       wspace=0.45)

axs = [plt.subplot(gs[0,1:3]),
       plt.subplot(gs[1,0:2]),
       plt.subplot(gs[1,2:]),
       plt.subplot(gs[2,0:2]),
       plt.subplot(gs[2,2:])
       ]
kwargs = {'x':'glon',
          'y':'glat'}
exp_kwargs = {'vmin':-1,
              'vmax':1,
              'cmap':'RdBu'}
std_kwargs = {'vmin':0.1,
              'vmax':0.5,
              'cmap':'viridis'}
cbar_kwargs = {'fraction':0.030,
               'pad':0.01,
               'label':''}

ds_climate_coarse_june_stacked[f'exp_logvarfunc_residual_climate'].unstack().plot.pcolormesh(
        ax=axs[0],
        **kwargs,
        **exp_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Residuals $r_{log(\sigma_Y^2)}$,$r_{log(\sigma_Z^2)}$')
    )

ds_climate_coarse_june_stacked[f'exp_posterior_predictive_realisations_singleprocess_logvar'].unstack().plot.pcolormesh(
        ax=axs[1],
        **kwargs,
        **exp_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Single Process Exp. $r_{log(\sigma_Y^2)}$')
    )

ds_climate_coarse_june_stacked[f'exp_truth_posterior_predictive_realisations_dualprocess_logvar'].unstack().plot.pcolormesh(
        ax=axs[2],
        **kwargs,
        **exp_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Shared Process Exp. $r_{log(\sigma_Y^2)}$')
    )

(2*ds_climate_coarse_june_stacked[f'std_posterior_predictive_realisations_singleprocess_logvar']).unstack().plot.pcolormesh(
        ax=axs[3],
        **kwargs,
        **std_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Single Process Uncert. $r_{log(\sigma_Y^2)}$')
    )

(2*ds_climate_coarse_june_stacked[f'std_truth_posterior_predictive_realisations_dualprocess_logvar']).unstack().plot.pcolormesh(
        ax=axs[4],
        **kwargs,
        **std_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Shared Process Uncert. $r_{log(\sigma_Y^2)}$')
    )

for ax in axs[:3]:
    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
        marker="o",
        c=scenario['exp_logvarfunc_residual_obs'],
        edgecolor="w",
        linewidth=0.6,
        **exp_kwargs,
    )

for ax,label in zip(axs,['a.','b.','c.','d.','e.']):
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    # ax.axis("off")
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')
    ax.set_xlim([-26.5,26.5])

plt.tight_layout()
fig.savefig(f"{results_path}fig13.pdf", dpi=300, bbox_inches="tight")

# %% Comparison of Log Var. Residual Predictions
fig = plt.figure(figsize=(text_width, text_width),dpi=300) 
gs = gridspec.GridSpec(3, 4,
                       hspace=0.08,
                       wspace=0.45)

axs = [plt.subplot(gs[0,1:3]),
       plt.subplot(gs[1,0:2]),
       plt.subplot(gs[1,2:]),
       plt.subplot(gs[2,0:2]),
       plt.subplot(gs[2,2:])
       ]
kwargs = {'x':'glon',
          'y':'glat'}
exp_kwargs = {'vmin':-1,
              'vmax':1,
              'cmap':'RdBu'}
std_kwargs = {'vmin':0.1,
              'vmax':0.5,
              'cmap':'viridis'}
cbar_kwargs = {'fraction':0.030,
               'pad':0.01,
               'label':''}

ds_climate_coarse_june_stacked[f'exp_logvarfunc_residual_climate'].unstack().plot.pcolormesh(
        ax=axs[0],
        **kwargs,
        **exp_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Residuals $r_{log(\sigma_Y^2)}$,$r_{log(\sigma_Z^2)}$')
    )

ds_climate_coarse_june_stacked[f'exp_posterior_predictive_realisations_singleprocess_logvar'].unstack().plot.pcolormesh(
        ax=axs[1],
        **kwargs,
        **exp_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Single Process Exp. $r_{log(\sigma_Y^2)}$')
    )

ds_climate_coarse_june_stacked[f'exp_truth_posterior_predictive_realisations_dualprocess_logvar'].unstack().plot.pcolormesh(
        ax=axs[2],
        **kwargs,
        **exp_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Shared Process Exp. $r_{log(\sigma_Y^2)}$')
    )

(2*ds_climate_coarse_june_stacked[f'std_posterior_predictive_realisations_singleprocess_logvar']).unstack().plot.pcolormesh(
        ax=axs[3],
        **kwargs,
        **std_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Single Process Uncert. $r_{log(\sigma_Y^2)}$')
    )

(2*ds_climate_coarse_june_stacked[f'std_truth_posterior_predictive_realisations_dualprocess_logvar']).unstack().plot.pcolormesh(
        ax=axs[4],
        **kwargs,
        **std_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Shared Process Uncert. $r_{log(\sigma_Y^2)}$')
    )

for ax in axs[:3]:
    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
        marker="o",
        c=scenario['exp_logvarfunc_residual_obs'],
        edgecolor="w",
        linewidth=0.6,
        **exp_kwargs,
    )

for ax,label in zip(axs,['a.','b.','c.','d.','e.']):
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    # ax.axis("off")
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')
    ax.set_xlim([-26.5,26.5])

plt.tight_layout()
fig.savefig(f"{results_path}fig13.pdf", dpi=300, bbox_inches="tight")

# %% Examining the Bias

fig = plt.figure(figsize=(text_width, text_width*0.7)) 
gs = gridspec.GridSpec(2, 2,
                       hspace=0.0,
                       wspace=0.3)

axs = [plt.subplot(gs[0,0]),
       plt.subplot(gs[1,0]),
       plt.subplot(gs[0,1:]),
       plt.subplot(gs[1,1]),
       ]
kwargs = {'x':'glon',
          'y':'glat'}
exp_kwargs = {'cmap':'RdBu'}
std_kwargs = {'cmap':'viridis'}
cbar_kwargs = {'fraction':0.030,
               'pad':0.02,
               'label':''}

ds_climate_coarse_june_stacked[f'exp_bias_posterior_predictive_realisations_dualprocess_mean'].unstack().plot.pcolormesh(
        ax=axs[0],
        **kwargs,
        **exp_kwargs,
        vmin=-3,
        vmax=3,
        cbar_kwargs = dict(cbar_kwargs, label='Exp. Bias Mean Residual')
    )

(2*ds_climate_coarse_june_stacked[f'std_bias_posterior_predictive_realisations_dualprocess_mean']).unstack().plot.pcolormesh(
        ax=axs[1],
        **kwargs,
        **std_kwargs,
        vmin=1,
        vmax=4,
        cbar_kwargs = dict(cbar_kwargs, label='2$\sigma$ Bias Mean Residual')
    )

ds_climate_coarse_june_stacked[f'exp_bias_posterior_predictive_realisations_dualprocess_logvar'].unstack().plot.pcolormesh(
        ax=axs[2],
        **kwargs,
        **exp_kwargs,
        vmin=-0.25,
        vmax=0.25,
        cbar_kwargs = dict(cbar_kwargs, label='Exp. Bias Log Var. Residual')
    )

(2*ds_climate_coarse_june_stacked[f'std_bias_posterior_predictive_realisations_dualprocess_logvar']).unstack().plot.pcolormesh(
        ax=axs[3],
        **kwargs,
        **std_kwargs,
        vmin=0.1,
        vmax=0.3,
        cbar_kwargs = dict(cbar_kwargs, label='2$\sigma$ Bias Log Var. Residual')
    )

for ax,label in zip(axs,['a.','b.','c.','d.']):
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')

plt.tight_layout()

fig.savefig(f"{results_path}fig16.pdf", dpi=300, bbox_inches="tight")


# %%
