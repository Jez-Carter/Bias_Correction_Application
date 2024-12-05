# %% Importing Packages

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import distance
from src.slide_functions import background_map_rotatedcoords, plot_hex_grid, rotated_coord_system, markersize_legend

base_path = '/home/jez/'
repo_path = f'{base_path}Bias_Correction_Application/'
internal_datapath = f'{repo_path}Slide_Images/Data/'
external_datapath = f'{base_path}DSNE_ice_sheets/Jez/Slides/'
scenario_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'

# %% Loading Data
ds_aws_stacked = xr.open_dataset(f'{external_datapath}ds_aws_stacked.nc')
ds_climate = xr.open_dataset(f'{external_datapath}ds_climate.nc')

scenario = np.load(
    scenario_path, allow_pickle="TRUE",fix_imports=True,
).item()

ds_climate_coarse_june_stacked = scenario['ds_climate_coarse_june_stacked']
ds_climate_coarse_june_stacked_landonly = scenario['ds_climate_coarse_june_stacked_landonly'].rename({'Temperature':'LandOnly Temperature'})

ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked_landonly,ds_climate_coarse_june_stacked])
ds_climate_coarse_june = ds_climate_coarse_june_stacked.unstack()

# %% Plotting Background Map with AWS Sites & Hexgrid
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)

ax.scatter(
    ds_aws_stacked.glon,
    ds_aws_stacked.glat,
    s=20,
    edgecolor='w',
    linewidths=0.5,
)

ax.set_ylim([-26,26])
ax.set_axis_off()

plot_hex_grid(ax,rotated_coord_system,18)

# highlighting individual station
station = 'Manuela'
ax.annotate(
    f'{station}',
    xy=(ds_aws_stacked.sel(Station = station).glon, ds_aws_stacked.sel(Station = station).glat), xycoords='data',
    xytext=(-120,-30), textcoords='offset points',
    arrowprops=dict(arrowstyle="->"),
    fontsize=14)

plt.tight_layout()

# %% Plotting Mean Temperature Map AWS Sites
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)

ax.scatter(
    ds_aws_stacked.glon,
    ds_aws_stacked.glat,
    s=ds_aws_stacked['June Temperature Records']*2,
    c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
)

ax.set_ylim([-26,26])
ax.set_axis_off()
markersize_legend(ax, [1,5,10,15,20,25,30,35,40], scale_multipler=2, legend_fontsize=10,loc=8,ncols=9,columnspacing=0.3,handletextpad=-0.4,bbox=(0.4,0.15))

plt.tight_layout()

# %% Plotting Mean Temperature Map High-Resolution Climate Model & AWS Sites

fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)

ds_climate.where(ds_climate.LSM)['Mean June Temperature'].plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    vmin=-55,
    vmax=-10,
    cmap='jet',
    add_colorbar=False,
    # cbar_kwargs = {'fraction':0.030,
    #             'pad':0.02,
    #             'label':'Mean January Temperature'}
)

ax.scatter(
    ds_aws_stacked.glon,
    ds_aws_stacked.glat,
    s=ds_aws_stacked['June Temperature Records']*2,
    c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
)

ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')
                       
plt.tight_layout()


# %% Plotting Mean Temperature Map Coarse-Resolution Climate Model & AWS Sites

fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)

ds_climate_coarse_june['LandOnly Temperature'].mean('Time').plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    vmin=-55,
    vmax=-10,
    cmap='jet',
    add_colorbar=False,
    # cbar_kwargs = {'fraction':0.030,
    #             'pad':0.02,
    #             'label':'Mean January Temperature'}
)

ax.scatter(
    ds_aws_stacked.glon,
    ds_aws_stacked.glat,
    s=ds_aws_stacked['June Temperature Records']*2,
    c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
)

ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')
                       
plt.tight_layout()


############################################ MeanFunc Results ############################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# %% MeanFunc Results: Loading Data
posterior_meanfunc_obs = scenario['Mean_Function_Posterior']
posterior_meanfunc_climate = scenario['Mean_Function_Posterior_Climate']

# Nearest Neighbors
nn_indecies = []
for point in scenario['ox']:
    nn_indecies.append(distance.cdist([point], scenario['cx']).argmin())

posterior_meanfunc_climate_nn = posterior_meanfunc_climate.isel({'X':nn_indecies})

diff_mean = posterior_meanfunc_obs['mean'].mean(['chain','draw']).data - posterior_meanfunc_climate_nn['mean'].mean(['chain','draw']).data
posterior_meanfunc_climate_nn['mean_diff']= (
    ('X'),
    diff_mean)
diff_residual = posterior_meanfunc_obs['meanfunc_residual'].mean(['chain','draw']).data - posterior_meanfunc_climate_nn['meanfunc_residual'].mean(['chain','draw']).data
posterior_meanfunc_climate_nn['meanfunc_residual_diff']= (
    ('X'),
    diff_residual)


ds_climate_coarse_june_stacked_landonly['exp_meanfunc_residual'] = (
    ('X'),
    posterior_meanfunc_climate['meanfunc_residual'].mean(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['exp_meanfunc_prediction'] = (
    ('X'),
    posterior_meanfunc_climate['meanfunc_prediction'].mean(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['exp_meanfunc_prediction_unbiased'] = (
    ('X'),
    posterior_meanfunc_climate['meanfunc_prediction_unbiased'].mean(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['exp_meanfunc_prediction_bias'] = (
    ('X'),
    posterior_meanfunc_climate['meanfunc_prediction'].mean(['chain','draw']).data-
    posterior_meanfunc_climate['meanfunc_prediction_unbiased'].mean(['chain','draw']).data)

ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

# %% MeanFunction Prediction 

fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)

ds_climate_coarse_june_stacked['exp_meanfunc_prediction'].unstack().plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    vmin=-55,
    vmax=-10,
    cmap='jet',
    add_colorbar=False,
    # cbar_kwargs = {'fraction':0.030,
    #             'pad':0.02,
    #             'label':'Mean Function Prediction'}
)

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*5,
    marker="o",
    c=posterior_meanfunc_obs['meanfunc_prediction'].mean(['chain','draw']).data,
    cmap='jet',
    edgecolor="w",
    linewidth=1.3,
    vmin=-55,
    vmax=-10,
)

ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')
                       
plt.tight_layout()

# %% MeanFunction Residual 

fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)

ds_climate_coarse_june_stacked['exp_meanfunc_residual'].unstack().plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    vmin=-10,
    vmax=10,
    cmap='RdBu',
    # add_colorbar=False,
    cbar_kwargs = {'fraction':0.030,
                'pad':0.02,
                'label':'Residual Prediction'}
)

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*5,
    marker="o",
    c=posterior_meanfunc_obs['meanfunc_residual'].mean(['chain','draw']).data,
    cmap='RdBu',
    edgecolor="w",
    linewidth=1.3,
    vmin=-10,
    vmax=10,
)

ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')
                       
plt.tight_layout()



#  %% Truth Prediction and Residual

fig, axs = plt.subplots(1, 2, figsize=(10, 20),dpi=300)#,frameon=False)

for ax in axs:
    background_map_rotatedcoords(ax)

ax=axs[0]
# ds_climate_coarse_june_stacked['exp_meanfunc_prediction_unbiased'].unstack().plot.pcolormesh(
ds_climate_coarse_june_stacked['exp_meanfunc_prediction'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        alpha=0.9,
        vmin=-55,
        vmax=-10,
        cmap='jet',
        cbar_kwargs = {'fraction':0.030,
                    'pad':-0.04,
                    'anchor':(0.5,0.48),
                    'label':'Mean Function Prediction'}
    )

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*2,
    marker="o",
    c=posterior_meanfunc_obs['meanfunc_prediction'].mean(['chain','draw']).data,
    cmap='jet',
    edgecolor="w",
    linewidth=0.4,
    vmin=-55,
    vmax=-10,
)

ax=axs[1]

ds_climate_coarse_june_stacked['exp_meanfunc_residual'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        alpha=0.9,
        vmin=-10,
        vmax=10,
        cmap='RdBu',
        cbar_kwargs = {'fraction':0.030,
                    'pad':-0.04,
                    'label':'Mean Function Residual'}
    )

aws_plot = ax.scatter(
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

# fig.colorbar(aws_plot, ax=ax, orientation='vertical',fraction=0.030,pad=0.04,label='Temperature Monthly Mean Residual')

for ax in axs:
    # ax.set_ylim([-26,26])
    ax.set_axis_off()
    ax.set_title('')

plt.tight_layout()

#  %% Bias Prediction and Residual

fig, axs = plt.subplots(1, 2, figsize=(10, 10),dpi=300)#,frameon=False)

for ax in axs:
    background_map_rotatedcoords(ax)

ax=axs[0]
ds_climate_coarse_june_stacked['exp_meanfunc_prediction_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        alpha=0.9,
        vmin=-5,
        vmax=5,
        cmap='jet',
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'label':'Temperature Monthly Mean Residual'}
    )


ax=axs[1]
ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*5,
    marker="o",
    c=posterior_meanfunc_climate_nn['meanfunc_residual_diff'].data,
    cmap='RdBu',
    edgecolor="w",
    linewidth=0.4,
    vmin=-10,
    vmax=10,
)

fig.colorbar(aws_plot, ax=ax, orientation='vertical',fraction=0.030,pad=0.04,label='Temperature Monthly Mean Residual')

for ax in axs:
    # ax.set_ylim([-26,26])
    ax.set_axis_off()
    ax.set_title('')

plt.tight_layout()


#  %% NN Bias in Mean and Residual

fig, axs = plt.subplots(1, 2, figsize=(10, 10),dpi=300)#,frameon=False)

for ax in axs:
    background_map_rotatedcoords(ax)

ax=axs[0]
ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*5,
    marker="o",
    c=posterior_meanfunc_climate_nn['mean_diff'].data,
    cmap='RdBu',
    edgecolor="w",
    linewidth=0.4,
    vmin=-10,
    vmax=10,
)


ax=axs[1]
ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*5,
    marker="o",
    c=posterior_meanfunc_climate_nn['meanfunc_residual_diff'].data,
    cmap='RdBu',
    edgecolor="w",
    linewidth=0.4,
    vmin=-10,
    vmax=10,
)

fig.colorbar(aws_plot, ax=ax, orientation='vertical',fraction=0.030,pad=0.04,label='Temperature Monthly Mean Residual')

for ax in axs:
    # ax.set_ylim([-26,26])
    ax.set_axis_off()
    ax.set_title('')

plt.tight_layout()


# %% Histograms
df = posterior_meanfunc_climate_nn[['mean_diff','meanfunc_residual_diff']].to_dataframe()

fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

df[['mean_diff']].hist(bins=20,
                    ax=ax,
                    edgecolor='k',
                    linewidth=0.2,
                    grid=False,
                    density=1,
                    alpha=0.7,
                    label = 'Bias',
                    )
df[['meanfunc_residual_diff']].hist(bins=20,
                    ax=ax,
                    edgecolor='k',
                    linewidth=0.2,
                    grid=False,
                    density=1,
                    alpha=0.7,
                    label = 'Residual Bias',
                    )

ax.set_title('')
ax.set_xlabel('Temperature Difference')
ax.set_ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

############################################ Spatial Results #############################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# %% Spatial Model Results: Loading Data
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

i=6
ds_climate_coarse_june_stacked_landonly[f'{i}random_truth_posterior_predictive_realisations_dualprocess_mean'] = (
    ('X'),
    scenario['truth_posterior_predictive_realisations_dualprocess_mean'][i,:])
ds_climate_coarse_june_stacked_landonly[f'{i}random_bias_posterior_predictive_realisations_dualprocess_mean'] = (
    ('X'),
    scenario['bias_posterior_predictive_realisations_dualprocess_mean'][i,:])



ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])#,compat='override')


exp_meanfunc_residual_bias_nn = (scenario['exp_meanfunc_residual_climate'][nn_indecies]-
                                 scenario['exp_meanfunc_residual_obs'])


# %%
scenario[
        "truth_posterior_predictive_realisations_dualprocess_mean"
    ].mean(axis=0)


# %%
fig, ax = plt.subplots(1,1, figsize=(10, 8),dpi=300)#,frameon=False)

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


# %% Spatial Plot of Predictions and Uncertainty (Mean Truth and Bias)
fig, axes = plt.subplots(2,2, figsize=(10, 8),dpi=300)#,frameon=False)

for ax in axes.ravel():
    background_map_rotatedcoords(ax)

components = ['truth','bias']
exp_vminmaxs = [[-10,10],[-5,5]]
std_vminmaxs = [[2,4.5],[1,3.5]]

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

for ax in axes.ravel():
    # ax.set_ylim([-26,26])
    ax.set_axis_off()
    ax.set_title('')

plt.tight_layout()

# %%


# %% Plotting Random Realisations
fig, axs = plt.subplots(1,2, figsize=(10, 5),dpi=300)#,frameon=False)
# i=500
for ax in axs.ravel():
    background_map_rotatedcoords(ax)

components = ['truth','bias']
exp_vminmaxs = [[-10,10],[-5,5]]
std_vminmaxs = [[2,5],[1,3.5]]

for ax,component,exp_vminmax,std_vminmax in zip(axs,components,exp_vminmaxs,std_vminmaxs):
    # ax=axs[0]
    ds_climate_coarse_june_stacked[f'{i}random_{component}_posterior_predictive_realisations_dualprocess_mean'].unstack().plot.pcolormesh(
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

for ax in axs.ravel():
    # ax.set_ylim([-26,26])
    ax.set_axis_off()
    ax.set_title('')

plt.tight_layout()

# %%
fig, axs = plt.subplots(1,2, figsize=(10, 5),dpi=300)#,frameon=False)

for ax in axs.ravel():
    background_map_rotatedcoords(ax)

components = ['truth','bias']
exp_vminmaxs = [[-10,10],[-5,5]]
std_vminmaxs = [[2,5],[1,3.5]]

ds_climate_coarse_june_stacked[f'diffrandom_truth_posterior_predictive_realisations_dualprocess_mean'] = (
    ds_climate_coarse_june_stacked[f'100random_{component}_posterior_predictive_realisations_dualprocess_mean']
)

for ax,component,exp_vminmax,std_vminmax in zip(axs,components,exp_vminmaxs,std_vminmaxs):
    # ax=axs[0]
    ds_climate_coarse_june_stacked[f'{i}random_{component}_posterior_predictive_realisations_dualprocess_mean'].unstack().plot.pcolormesh(
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

for ax in axs.ravel():
    # ax.set_ylim([-26,26])
    ax.set_axis_off()
    ax.set_title('')

plt.tight_layout()



############################################# Joint Results ##############################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# %% Combining MeanFunc and Spatial Results (hr)

ds_climate_coarse_june_stacked = scenario['ds_climate_coarse_june_stacked_hr'] 
ds_climate_coarse_june_stacked_landonly = scenario['ds_climate_coarse_june_stacked_landonly_hr']

posterior_climate = scenario['Mean_Function_Posterior_Climate_hr']

exp_meanfunc_unbiased = posterior_climate['meanfunc_prediction_unbiased'].mean(['chain','draw']).data
std_meanfunc_unbiased = posterior_climate['meanfunc_prediction_unbiased'].std(['chain','draw']).data

residual_mean_unbiased = scenario['truth_posterior_predictive_realisations_dualprocess_mean_hr']
exp_residual_mean_unbiased = residual_mean_unbiased.mean(axis=0)
std_residual_mean_unbiased = residual_mean_unbiased.std(axis=0)

noise_residual_mean_unbiased = scenario[f"mcmc_dualprocess_mean_residual"].posterior['noise'].mean().data

exp_mean_unbiased = exp_meanfunc_unbiased + exp_residual_mean_unbiased
std_mean_unbiased = np.sqrt(std_meanfunc_unbiased**2 + std_residual_mean_unbiased**2+noise_residual_mean_unbiased)

ds_climate_coarse_june_stacked_landonly['exp_mean_unbiased'] = (('X'),exp_mean_unbiased)
ds_climate_coarse_june_stacked_landonly['std_mean_unbiased'] = (('X'),std_mean_unbiased)
ds_climate_coarse_june_stacked_landonly['exp_meanfunc_unbiased'] = (('X'),exp_meanfunc_unbiased)
ds_climate_coarse_june_stacked_landonly['exp_residual_mean_unbiased'] = (('X'),exp_residual_mean_unbiased)
ds_climate_coarse_june_stacked_landonly['var_residual_mean_unbiased'] = (('X'),std_residual_mean_unbiased**2)



exp_meanfunc_bias = posterior_climate['meanfunc_prediction_bias'].mean(['chain','draw']).data
std_meanfunc_bias = posterior_climate['meanfunc_prediction_bias'].std(['chain','draw']).data

residual_mean_bias = scenario['bias_posterior_predictive_realisations_dualprocess_mean_hr']
exp_residual_mean_bias = residual_mean_bias.mean(axis=0)
std_residual_mean_bias = residual_mean_bias.std(axis=0)

noise_residual_mean_bias = scenario[f"mcmc_dualprocess_mean_residual"].posterior['bnoise'].mean().data

exp_mean_bias = exp_meanfunc_bias + exp_residual_mean_bias
std_mean_bias = np.sqrt(std_meanfunc_bias**2 + std_residual_mean_bias**2+noise_residual_mean_bias)

ds_climate_coarse_june_stacked_landonly['exp_mean_bias'] = (('X'),exp_mean_bias)
ds_climate_coarse_june_stacked_landonly['std_mean_bias'] = (('X'),std_mean_bias)
ds_climate_coarse_june_stacked_landonly['exp_meanfunc_bias'] = (('X'),exp_meanfunc_bias)
ds_climate_coarse_june_stacked_landonly['exp_residual_mean_bias'] = (('X'),exp_residual_mean_bias)
ds_climate_coarse_june_stacked_landonly['var_residual_mean_bias'] = (('X'),std_residual_mean_bias**2)


###### Merging for Coordinate Reasons ######
# vars = ['exp_mean_unbiased','std_mean_unbiased','exp_meanfunc_unbiased','exp_residual_mean_unbiased','var_residual_mean_unbiased']
ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked['exp_meanfunc_prediction_unbiased'],ds_climate_coarse_june_stacked_landonly])

# %% Mean Prediction Expectation
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)
cbar_kwargs = {'fraction':0.030,
            'pad':0.04,}

ds_climate_coarse_june_stacked['exp_mean_unbiased'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='jet',
        vmin=-55,
        vmax=-10,
        add_colorbar=False,
        # cbar_kwargs = dict(cbar_kwargs, label='Mean Expecation $E[{\mu_Y}]$')
    )

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*5,
    marker="o",
    c=np.nanmean(scenario['odata'],axis=0),
    cmap='jet',
    edgecolor="w",
    linewidth=1.5,
    vmin=-55,
    vmax=-10,
)

ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')

plt.tight_layout()

# %% Mean Prediction Uncertainty
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)
cbar_kwargs = {'fraction':0.030,
            'pad':0.04,}

(2*ds_climate_coarse_june_stacked['std_mean_unbiased']).unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='viridis',
        vmin=5,
        vmax=10,
        add_colorbar=False,
        # cbar_kwargs = dict(cbar_kwargs, label='Mean $2\sigma$ Uncertainty')
    )

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=5,
    marker="o",
    edgecolor="w",
    linewidth=0.6,
)

ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')

plt.tight_layout()
# %% Mean Prediction Expectation Bias
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)
cbar_kwargs = {'fraction':0.030,
            'pad':0.04,}

ds_climate_coarse_june_stacked['exp_mean_bias'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='RdBu',
        vmin=-4,
        vmax=4,
        add_colorbar=False,
        # cbar_kwargs = dict(cbar_kwargs, label='Bias Mean Expecation $E[{\mu_B}]$')
    )

ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')

plt.tight_layout()

# %% Mean Prediction Uncertainty Bias
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)
cbar_kwargs = {'fraction':0.030,
            'pad':0.04,}

(2*ds_climate_coarse_june_stacked['std_mean_bias']).unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='viridis',
        # vmin=5,
        # vmax=10,
        cbar_kwargs = dict(cbar_kwargs, label='Mean $2\sigma$ Uncertainty')
    )

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=5,
    marker="o",
    edgecolor="w",
    linewidth=0.6,
)

ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')

plt.tight_layout()

# %% MeanFunction Prediction Expectation

fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)
cbar_kwargs = {'fraction':0.030,
            'pad':0.04,}

ds_climate_coarse_june_stacked['exp_meanfunc_unbiased'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='jet',
        vmin=-55,
        vmax=-10,
        # add_colorbar=False,
        cbar_kwargs = dict(cbar_kwargs, label='Mean Function Expecation $E[{m_{\mu_Y}}]$')
    )


ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')

plt.tight_layout()

# %% Mean Residual Prediction Expectation

fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)
cbar_kwargs = {'fraction':0.030,
            'pad':0.04,}

ds_climate_coarse_june_stacked['exp_residual_mean_unbiased'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        cmap='RdBu',
        # vmin=-55,
        # vmax=-10,
        add_colorbar=False,
        # cbar_kwargs = dict(cbar_kwargs, label='Mean Expecation Residual $E[{r_{\mu_Y}}]$')
    )

ax.scatter(
    scenario['ox'][:, 0],
    scenario['ox'][:, 1],
    s=np.count_nonzero(~np.isnan(scenario['odata']),axis=0)*5,
    marker="o",
    c=scenario['exp_meanfunc_residual_obs'],
    cmap='RdBu',
    edgecolor="w",
    linewidth=1.5,
    vmin=-15,
    vmax=15,
)

ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')

plt.tight_layout()



























# %%
posterior_meanfunc_climate_nn[['meanfunc_residual']].to_dataframe().describe()

# %%
posterior_meanfunc_climate_nn[['meanfunc_residual']].to_dataframe().describe()
# posterior_meanfunc_obs['meanfunc_residual'].mean(['chain','draw']).to_dataframe().describe()

# %%
df.describe()


















# %%
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)

im = ax.imshow(data, cmap='bone')

fig.colorbar(im, cax=cax, orientation='vertical')


ds_climate_coarse_june_stacked['exp_meanfunc_prediction_unbiased'].unstack().plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        alpha=0.,
        vmin=-10,
        vmax=10,
        cmap='jet',
        cbar_kwargs = {'fraction':0.030,
                    'pad':0.04,
                    'alpha':1,
                    'label':'Temperature Monthly Mean Residual'}
    )

# %%

ax.scatter(
    posterior_meanfunc_obs.glon,
    posterior_meanfunc_obs.glat,
    s=ds_aws_stacked['June Temperature Records']*2,
    c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
)

# %%
posterior_meanfunc_climate['meanfunc_residual'].mean(['chain','draw']).to_dataframe().describe()

# %%

ds_climate_coarse_june_stacked_landonly['exp_meanfunc_residual'] = (
    ('X'),
    posterior_meanfunc_climate['meanfunc_residual'].mean(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['exp_meanfunc_prediction'] = (
    ('X'),
    posterior_meanfunc_climate['meanfunc_prediction'].mean(['chain','draw']).data)
ds_climate_coarse_june_stacked_landonly['exp_meanfunc_prediction_unbiased'] = (
    ('X'),
    posterior_meanfunc_climate['meanfunc_prediction_unbiased'].mean(['chain','draw']).data)

ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

# %%
scenario['var_meanfunc_residual_obs'].shape