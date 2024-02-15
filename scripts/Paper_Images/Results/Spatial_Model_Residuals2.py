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

# %% Incorporating predictions into xarray dataset
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

###### Merging for Coordinate Reasons ######
ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])
ds_climate_coarse_june_stacked_uhr = xr.merge([ds_climate_coarse_june_stacked_uhr,ds_climate_coarse_june_stacked_landonly_uhr])

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
fig, ax = plt.subplots(1, 1, figsize=(text_width, text_width/2),dpi=300)#,frameon=False)

gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
ax.set_yticks(np.arange(-20,25,5))
ax.set_xticks(np.arange(-25,30,5))
ax.set_ylabel('Grid Latitude')
ax.set_xlabel('Grid Longitude')
ax.grid(which='major', alpha=0.2)

def annotation_line( ax, xmin, xmax, y, text, ytext=0, linecolor='black', linewidth=1, fontsize=12 ):
    ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '|-|', 'color':linecolor, 'linewidth':linewidth, 'mutation_scale':1})
    ax.annotate( text, xy=(xmin,ytext), ha='left', va='center', fontsize=fontsize)

annotation_line( ax=ax, text='$l_{\mu_Y}$ = 4.67', xmin=0, xmax=4.67, \
                    y=8, ytext=9.5, linewidth=0.4, linecolor='k', fontsize=8 )
annotation_line( ax=ax, text='$l_{\mu_B}$ = 13.34', xmin=0, xmax=13.34, \
                    y=4, ytext=5.5, linewidth=0.4, linecolor='k', fontsize=8 )
annotation_line( ax=ax, text='$l_{log(\sigma^2_Y)}$ = 5.02', xmin=0, xmax=5.02, \
                    y=0, ytext=1.5, linewidth=0.4, linecolor='k', fontsize=8 )
annotation_line( ax=ax, text='$l_{log(\sigma^2_B)}$ = 13.72', xmin=0, xmax=13.72, \
                    y=-4, ytext=-3.0, linewidth=0.4, linecolor='k', fontsize=8 )

plt.show()
plt.tight_layout()
fig.savefig(f"{results_path}figa06.pdf", dpi=300, bbox_inches="tight")

# %% Residual Predictions Expecations
fig, axs = plt.subplots(3, 2, figsize=(text_width, text_width*0.9),dpi=300)

kwargs = {'x':'glon',
          'y':'glat'}
mean_kwargs = {'vmin':-10,
              'vmax':10,
              'cmap':'RdBu'}
logvar_kwargs = {'vmin':-1,
              'vmax':1,
              'cmap':'RdBu'}
cbar_kwargs = {'fraction':0.030,
               'pad':0.02}

ds_climate_coarse_june_stacked[f'exp_meanfunc_residual_climate'].unstack().plot.pcolormesh(
        ax=axs[0,0],
        **kwargs,
        **mean_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Res. Expecations $E[r_{\mu_Y}],E[r_{\mu_Z}]$')
    )

ds_climate_coarse_june_stacked[f'exp_logvarfunc_residual_climate'].unstack().plot.pcolormesh(
        ax=axs[0,1],
        **kwargs,
        **logvar_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label=r'Res. Expecations $E[r_{\tilde{\sigma}_Y}],E[r_{\tilde{\sigma}_Z}]$')
    )

ds_climate_coarse_june_stacked_uhr[f'exp_truth_posterior_predictive_realisations_dualprocess_mean'].unstack().plot.pcolormesh(
        ax=axs[1,0],
        **kwargs,
        **mean_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Unbiased Res. Expecation $E[r_{\mu_Y}]$')
    )

ds_climate_coarse_june_stacked_uhr[f'exp_truth_posterior_predictive_realisations_dualprocess_logvar'].unstack().plot.pcolormesh(
        ax=axs[1,1],
        **kwargs,
        **logvar_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label=r'Unbiased Res. Expecation $E[r_{\tilde{\sigma}_Y}]$')
    )

ds_climate_coarse_june_stacked_uhr[f'exp_posterior_predictive_realisations_singleprocess_mean'].unstack().plot.pcolormesh(
        ax=axs[2,0],
        **kwargs,
        **mean_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Unbiased Res. Expecation $E[r_{\mu_Y}]$')
    )

ds_climate_coarse_june_stacked_uhr[f'exp_posterior_predictive_realisations_singleprocess_logvar'].unstack().plot.pcolormesh(
        ax=axs[2,1],
        **kwargs,
        **logvar_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label=r'Unbiased Res. Expecation $E[r_{\tilde{\sigma}_Y}]$')
    )

axs[0,0].scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
        marker="o",
        c=scenario['exp_meanfunc_residual_obs'],
        edgecolor="w",
        linewidth=0.6,
        vmin = -10,
        vmax = 10,
        cmap = 'RdBu',
    )

axs[0,1].scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*1,
    marker="o",
    c=scenario['exp_logvarfunc_residual_obs'],
    edgecolor="w",
    linewidth=0.6,
    vmin = -0.7,
    vmax = 0.7,
    cmap = 'RdBu',
)

for ax,label in zip(axs.ravel(),['a.','b.','c.','d.','e.','f.']):
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


for ax,title in zip(axs[0,:], ['Mean Parameter Residual','Log-Var. Parameter Residual']):
    ax.set_title(title,fontsize=8)

for ax,label in zip(axs[:,0], ['Input Data','Dual Process Model','Single Process Model']):
    ax.set_ylabel(label,fontsize=8)

plt.tight_layout()
fig.savefig(f"{results_path}fig12.pdf", dpi=300, bbox_inches="tight")

# %% Residual Predictions Uncertainties
fig, axs = plt.subplots(2, 2, figsize=(text_width, text_width/1.5),dpi=300)#,frameon=False)

kwargs = {'x':'glon',
          'y':'glat'}
mean_kwargs = {
              'cmap':'viridis'}
logvar_kwargs = {
              'cmap':'viridis'}
cbar_kwargs = {'fraction':0.030,
               'pad':0.02}

(2*ds_climate_coarse_june_stacked_uhr[f'std_truth_posterior_predictive_realisations_dualprocess_mean']).unstack().plot.pcolormesh(
        ax=axs[0,0],
        **kwargs,
        **mean_kwargs,
        # vmin=3.8,
        vmax=7,
        cbar_kwargs = dict(cbar_kwargs, label='Unbiased Res. $2\sigma$ Uncertainty $r_{\mu_Y}$')
    )

(2*ds_climate_coarse_june_stacked_uhr[f'std_truth_posterior_predictive_realisations_dualprocess_logvar']).unstack().plot.pcolormesh(
        ax=axs[0,1],
        **kwargs,
        **logvar_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label=r'Unbiased Res. $2\sigma$ Uncertainty $r_{\tilde{\sigma}_Y}$')
    )

(2*ds_climate_coarse_june_stacked_uhr[f'std_posterior_predictive_realisations_singleprocess_mean']).unstack().plot.pcolormesh(
        ax=axs[1,0],
        **kwargs,
        **mean_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label='Unbiased Res. $2\sigma$ Uncertainty $r_{\mu_Y}$')
    )

(2*ds_climate_coarse_june_stacked_uhr[f'std_posterior_predictive_realisations_singleprocess_logvar']).unstack().plot.pcolormesh(
        ax=axs[1,1],
        **kwargs,
        **logvar_kwargs,
        cbar_kwargs = dict(cbar_kwargs, label=r'Unbiased Res. $2\sigma$ Uncertainty $r_{\tilde{\sigma}_Y}$')
    )

for ax in axs.ravel():
    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=1,
        marker="o",
        c=scenario['exp_meanfunc_residual_obs'],
        edgecolor="w",
        linewidth=0.6,
    )

for ax,label in zip(axs.ravel(),['a.','b.','c.','d.','e.','f.']):
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


for ax,title in zip(axs[0,:], ['Mean Parameter Residual','Log-Var. Parameter Residual']):
    ax.set_title(title,fontsize=8)

for ax,label in zip(axs[:,0], ['Dual Process Model','Single Process Model']):
    ax.set_ylabel(label,fontsize=8)

plt.tight_layout()

fig.savefig(f"{results_path}fig13.pdf", dpi=300, bbox_inches="tight")


# %% Residual Predictions Uncertainties Variances
fig, axs = plt.subplots(2, 2, figsize=(text_width, text_width/1.5),dpi=300)#,frameon=False)

kwargs = {'x':'glon',
          'y':'glat'}
mean_kwargs = {
              'cmap':'BrBG'}
logvar_kwargs = {
              'cmap':'BrBG'}
cbar_kwargs = {'fraction':0.030,
               'pad':0.02}

np.square(ds_climate_coarse_june_stacked_uhr[f'std_truth_posterior_predictive_realisations_dualprocess_mean']).unstack().plot.pcolormesh(
        ax=axs[0,0],
        **kwargs,
        **mean_kwargs,
        vmin=0,
        vmax=12,
        cbar_kwargs = dict(cbar_kwargs, label='Unbiased Res. Variance $V[r_{\mu_Y}]$')
    )

np.square(ds_climate_coarse_june_stacked_uhr[f'std_truth_posterior_predictive_realisations_dualprocess_logvar']).unstack().plot.pcolormesh(
        ax=axs[0,1],
        **kwargs,
        **logvar_kwargs,
        vmin=0,
        vmax=0.014,
        cbar_kwargs = dict(cbar_kwargs, label=r'Unbiased Res. Variance $V[r_{\tilde{\sigma}_Y}]$')
    )

np.square(ds_climate_coarse_june_stacked_uhr[f'std_posterior_predictive_realisations_singleprocess_mean']).unstack().plot.pcolormesh(
        ax=axs[1,0],
        **kwargs,
        **mean_kwargs,
        vmin=0.5,
        vmax=18.5,
        cbar_kwargs = dict(cbar_kwargs, label='Unbiased Res. Variance $V[r_{\mu_Y}]$')
    )

np.square(ds_climate_coarse_june_stacked_uhr[f'std_posterior_predictive_realisations_singleprocess_logvar']).unstack().plot.pcolormesh(
        ax=axs[1,1],
        **kwargs,
        **logvar_kwargs,
        vmin=0,
        vmax=0.02,
        cbar_kwargs = dict(cbar_kwargs, label=r'Unbiased Res. Variance $V[r_{\tilde{\sigma}_Y}]$')
    )

for ax in axs.ravel():
    ax.scatter(
        scenario['ox'][:, 0],
        scenario['ox'][:, 1],
        s=1,
        marker="o",
        c=scenario['exp_meanfunc_residual_obs'],
        edgecolor="w",
        linewidth=0.6,
    )

for ax,label in zip(axs.ravel(),['a.','b.','c.','d.','e.','f.']):
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


for ax,title in zip(axs[0,:], ['Mean Parameter Residual','Log-Var. Parameter Residual']):
    ax.set_title(title,fontsize=8)

for ax,label in zip(axs[:,0], ['Dual Process Model','Single Process Model']):
    ax.set_ylabel(label,fontsize=8)

plt.tight_layout()

fig.savefig(f"{results_path}figa07.pdf", dpi=300, bbox_inches="tight")
