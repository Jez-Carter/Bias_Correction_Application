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

ds_climate_coarse_june_stacked = scenario['ds_climate_coarse_june_stacked_uhr']
ds_climate_coarse_june_stacked_landonly = scenario['ds_climate_coarse_june_stacked_landonly_uhr']

posterior_climate = scenario['Mean_Function_Posterior_Climate_uhr']
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

noise_residual_mean_unbiased = scenario[f"mcmc_dualprocess_mean_residual"].posterior['noise'].mean().data
noise_residual_logvar_unbiased = scenario[f"mcmc_dualprocess_logvar_residual"].posterior['noise'].mean().data

exp_mean_unbiased = exp_meanfunc_unbiased + exp_residual_mean_unbiased
std_mean_unbiased = np.sqrt(std_meanfunc_unbiased**2 + std_residual_mean_unbiased**2+noise_residual_mean_unbiased)
exp_logvar_unbiased = exp_logvarfunc_unbiased + exp_residual_logvar_unbiased
std_logvar_unbiased = np.sqrt(std_logvarfunc_unbiased**2 + std_residual_logvar_unbiased**2+noise_residual_logvar_unbiased)

ds_climate_coarse_june_stacked_landonly['exp_mean_unbiased'] = (('X'),exp_mean_unbiased)
ds_climate_coarse_june_stacked_landonly['std_mean_unbiased'] = (('X'),std_mean_unbiased)
ds_climate_coarse_june_stacked_landonly['exp_logvar_unbiased'] = (('X'),exp_logvar_unbiased)
ds_climate_coarse_june_stacked_landonly['std_logvar_unbiased'] = (('X'),std_logvar_unbiased)

ds_climate_coarse_june_stacked_landonly['exp_meanfunc_unbiased'] = (('X'),exp_meanfunc_unbiased)
ds_climate_coarse_june_stacked_landonly['exp_residual_mean_unbiased'] = (('X'),exp_residual_mean_unbiased)
ds_climate_coarse_june_stacked_landonly['exp_logvarfunc_unbiased'] = (('X'),np.full(ds_climate_coarse_june_stacked_landonly['X'].shape,exp_logvarfunc_unbiased))
ds_climate_coarse_june_stacked_landonly['exp_residual_logvar_unbiased'] = (('X'),exp_residual_logvar_unbiased)

# ds_climate_coarse_june_stacked_landonly['std_meanfunc_unbiased'] = (('X'),std_meanfunc_unbiased)
ds_climate_coarse_june_stacked_landonly['var_residual_mean_unbiased'] = (('X'),std_residual_mean_unbiased**2)
# ds_climate_coarse_june_stacked_landonly['std_logvarfunc_unbiased'] = (('X'),np.full(ds_climate_coarse_june_stacked_landonly['X'].shape,std_logvarfunc_unbiased))
ds_climate_coarse_june_stacked_landonly['var_residual_logvar_unbiased'] = (('X'),std_residual_logvar_unbiased**2)

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

noise_residual_mean_bias = scenario[f"mcmc_dualprocess_mean_residual"].posterior['bnoise'].mean().data
noise_residual_logvar_bias = scenario[f"mcmc_dualprocess_logvar_residual"].posterior['bnoise'].mean().data

exp_mean_bias = exp_meanfunc_bias + exp_residual_mean_bias
std_mean_bias = np.sqrt(std_meanfunc_bias**2 + std_residual_mean_bias**2 + noise_residual_mean_bias)
exp_logvar_bias = exp_logvarfunc_bias + exp_residual_logvar_bias
std_logvar_bias = np.sqrt(std_logvarfunc_bias**2 + std_residual_logvar_bias**2 + noise_residual_logvar_bias)

ds_climate_coarse_june_stacked_landonly['exp_mean_bias'] = (('X'),exp_mean_bias)
ds_climate_coarse_june_stacked_landonly['std_mean_bias'] = (('X'),std_mean_bias)
ds_climate_coarse_june_stacked_landonly['exp_logvar_bias'] = (('X'),exp_logvar_bias)
ds_climate_coarse_june_stacked_landonly['std_logvar_bias'] = (('X'),std_logvar_bias)

ds_climate_coarse_june_stacked_landonly['exp_meanfunc_bias'] = (('X'),exp_meanfunc_bias)
ds_climate_coarse_june_stacked_landonly['exp_residual_mean_bias'] = (('X'),exp_residual_mean_bias)
ds_climate_coarse_june_stacked_landonly['exp_logvarfunc_bias'] = (('X'),np.full(ds_climate_coarse_june_stacked_landonly['X'].shape,exp_logvarfunc_bias))
ds_climate_coarse_june_stacked_landonly['exp_residual_logvar_bias'] = (('X'),exp_residual_logvar_bias)

ds_climate_coarse_june_stacked_landonly['meanfunc_prediction_bias_single'] = (('X'),posterior_climate['meanfunc_prediction_bias'].isel(chain=0,draw=0).data)
ds_climate_coarse_june_stacked_landonly['residual_mean_bias_single'] = (('X'),residual_mean_bias[0])


ds_climate_coarse_june_stacked_landonly['exp_residual_mean_bias'] = (('X'),exp_residual_mean_bias)


###### Merging for Coordinate Reasons ######
ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

# %% Overall Unbiased Parameter Predictions Plot

fig, axs = plt.subplots(2,2, figsize=(text_width, text_width/1.5),dpi=300)#,frameon=False)

cbar_kwargs = {'fraction':0.030,
            'pad':0.04,}

ds_climate_coarse_june_stacked['exp_mean_unbiased'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0,0],
        cmap='jet',
        # vmin=-55,
        # vmax=-10,
        cbar_kwargs = dict(cbar_kwargs, label='Mean Expecation $E[{\mu_Y}]$')
    )

ds_climate_coarse_june_stacked['exp_logvar_unbiased'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0,1],
        cmap='jet',
        # vmin=1,
        # vmax=5,
        cbar_kwargs = dict(cbar_kwargs, label='Log-Var. Expecation $E[{\mu_Y}]$')

    )

(2*ds_climate_coarse_june_stacked['std_mean_unbiased']).unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1,0],
        cmap='viridis',
        vmin=5,
        vmax=10,
        cbar_kwargs = dict(cbar_kwargs, label='Mean $2\sigma$ Uncertainty')
    )

(2*ds_climate_coarse_june_stacked['std_logvar_unbiased']).unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1,1],
        cmap='viridis',
        # vmin=-0.3,
        # vmax=0.3,
        cbar_kwargs = dict(cbar_kwargs, label='Log-Var. $2\sigma$ Uncertainty')
    )

for ax,label in zip(axs.ravel(),['a.','b.','c.','d.']):
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')

for ax,title in zip(axs[0,:], ['Mean Parameter','Log-Var. Parameter']):
    ax.set_title(title,fontsize=8)

for ax,label in zip(axs[:,0], ['Expectation',r'$2\sigma$ Uncertainty']):
    ax.set_ylabel(label,fontsize=8)

plt.tight_layout()
fig.savefig(f"{results_path}fig17.pdf", dpi=300, bbox_inches="tight")

# %% Overall Bias Predictions Plot

fig, axs = plt.subplots(2,2, figsize=(text_width, text_width/1.5),dpi=300)#,frameon=False)

cbar_kwargs = {'fraction':0.030,
            'pad':0.04,}

ds_climate_coarse_june_stacked['exp_mean_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0,0],
        cmap='RdBu',
        vmin=-5.5,
        vmax=5.5,
        cbar_kwargs = dict(cbar_kwargs, label='Bias in Mean Expecation $E[{\mu_B}]$')
    )

ds_climate_coarse_june_stacked['exp_logvar_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0,1],
        cmap='RdBu',
        vmin=-0.2,
        vmax=0.2,
        cbar_kwargs = dict(cbar_kwargs, label='Bias in Log-Var. Expecation $E[{\mu_B}]$')

    )

(2*ds_climate_coarse_june_stacked['std_mean_bias']).unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1,0],
        cmap='viridis',
        # vmin=-0.3,
        # vmax=0.3,
        cbar_kwargs = dict(cbar_kwargs, label='Bias in Mean $2\sigma$ Uncertainty')
    )

(2*ds_climate_coarse_june_stacked['std_logvar_bias']).unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1,1],
        cmap='viridis',
        vmin=0.25,
        vmax=0.45,
        cbar_kwargs = dict(cbar_kwargs, label='Bias in Log-Var. $2\sigma$ Uncertainty')
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

plt.tight_layout()
fig.savefig(f"{results_path}fig18.pdf", dpi=300, bbox_inches="tight")

# %% Unbiased Components Uncertainties

fig, axs = plt.subplots(1,2, figsize=(text_width, text_width/1.5),dpi=300)#,frameon=False)

kwargs_mean = {'vmin':0.8,
               'vmax':7}
kwargs_logv = {'vmin':0.2,
               'vmax':0.06}

ds_climate_coarse_june_stacked['var_residual_mean_unbiased'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0],
        cmap='viridis',
        # **kwargs_mean,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

ds_climate_coarse_june_stacked['var_residual_logvar_unbiased'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1],
        cmap='viridis',
        # **kwargs_logv,
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
# fig.savefig(f"{results_path}figa07.pdf", dpi=300, bbox_inches="tight")

# %% Bias Components Plot

fig, axs = plt.subplots(2,2, figsize=(text_width, text_width/1.5),dpi=300)#,frameon=False)

print(f'''
      Meanfunc Bias:
Min:{ds_climate_coarse_june_stacked['exp_meanfunc_bias'].min().data}
Max:{ds_climate_coarse_june_stacked['exp_meanfunc_bias'].max().data}
      Residual Bias:
Min:{ds_climate_coarse_june_stacked['exp_residual_mean_bias'].min().data}
Max:{ds_climate_coarse_june_stacked['exp_residual_mean_bias'].max().data}
''')

ds_climate_coarse_june_stacked['exp_meanfunc_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0,0],
        cmap='RdBu',
        vmin=-3,
        vmax=3,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

ds_climate_coarse_june_stacked['exp_residual_mean_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1,0],
        cmap='RdBu',
        vmin=-3,
        vmax=3,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

ds_climate_coarse_june_stacked['exp_logvarfunc_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[0,1],
        cmap='RdBu',
        vmin=-0.3,
        vmax=0.3,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

ds_climate_coarse_june_stacked['exp_residual_logvar_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=axs[1,1],
        cmap='RdBu',
        vmin=-0.3,
        vmax=0.3,
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Exp. Mean Parameter Bias'}
    )

for ax,label in zip(axs.ravel(),['a.','b.','c.','d.']):
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(label,xy=(0.02,0.93),xycoords='axes fraction')

for ax,title in zip(axs[0,:], ['Mean Parameter','Log-Var. Parameter']):
    ax.set_title(title,fontsize=8)

for ax,label in zip(axs[:,0], ['Mean Function Bias','Spatial Model Residual Bias']):
    ax.set_ylabel(label,fontsize=8)

plt.tight_layout()
fig.savefig(f"{results_path}figa04.pdf", dpi=300, bbox_inches="tight")

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