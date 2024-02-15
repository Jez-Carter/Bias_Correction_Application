# %% Importing Packages
import numpy as np
import pandas as pd
import xarray as xr
import numpyro
import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess
import arviz as az
from tqdm import tqdm
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.helper_functions import run_inference

import jax
jax.config.update("jax_enable_x64", True)
from jax import random
rng_key = random.PRNGKey(1)

# %% Loading AWS Data
base_path = '/home/jez/'
aws_inpath = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Processed/df_all_combined_75_monthly.csv'
cordex_inpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'

df_aws = pd.read_csv(aws_inpath)
df_aws_group = df_aws.set_index(['Station','Year','Month'])
df_aws_group_coords = df_aws.set_index(['Station'])
da_temp = df_aws_group[~df_aws_group.index.duplicated()]['Temperature'].to_xarray()
ds_coords = df_aws_group_coords[~df_aws_group_coords.index.duplicated()][['Lat(℃)','Lon(℃)','Elevation(m)','glat','glon','grid_latitude', 'grid_longitude']].to_xarray()
ds_aws = xr.merge([ds_coords,da_temp])
ds_aws_stacked = ds_aws.stack(X=('Year','Month'))

# %% Selecting Stations and Plotting Data
stations = ['Larsen Ice Shelf','Henry']
ds_stacked_stations_filtered = ds_aws_stacked.sel(Station=stations).reset_index('X')
ds_stacked_single_station = ds_aws_stacked.sel(Station=stations[0]).reset_index('X')

fig, ax = plt.subplots(1, 1, figsize=(10, 6),dpi=300)#,frameon=False)
ds_stacked_stations_filtered['Temperature'].plot(ax=ax,hue='Station')
ax.set_xlabel('Month Since 1980')

# %% Reformatting Data for Model

ot = jnp.array(ds_stacked_single_station.X.values).astype(jnp.float64)
odata = jnp.array(ds_stacked_single_station['Temperature'].values)
obs_mask = (jnp.isnan(odata)==False)
ot = ot[obs_mask]
odata = odata[obs_mask]

print(f'ot shape = {ot.shape} \n',
      f'odata shape = {odata.shape} \n',
      f'obs_mask shape = {obs_mask.shape} \n')


# %% Defining Simple Time Series Model
def tinygp_model(t,jdata=None):
    mean = numpyro.sample("mean", dist.Normal(jnp.nanmean(jdata), 5.0))
    kern_var = numpyro.sample("kern_var", dist.Uniform(1,2000))
    noise = numpyro.sample("noise", dist.HalfNormal(20))

    kernel = kern_var * kernels.ExpSineSquared(scale=12.0,gamma=12.0) * kernels.ExpSquared(scale=60.0)
    
    gp = GaussianProcess(kernel, t, diag=noise+1e-5, mean=mean)
    numpyro.sample("temperature", gp.numpyro_dist(),obs=jdata)

# %% Running Inference
mcmc = run_inference(tinygp_model, rng_key, 1000, 2000,1, ot,jdata=odata)
idata = az.from_numpyro(mcmc)

# %% Defining Function to Compute Posterior Predictives
def posterior_predictive_realisations(
        rng_key,ot,odata,nt,idata,num_parameter_realisations):
    
    posterior = idata.posterior
    kern_vars = posterior['kern_var'].data[0,:]
    means = posterior['mean'].data[0,:]
    noises = posterior['noise'].data[0,:]

    realisations_list = []
    for kern_var,mean,noise in tqdm(list(zip(kern_vars,means,noises))[:num_parameter_realisations]):
        kernel = kern_var * kernels.ExpSineSquared(scale=jnp.array(12.0),gamma=jnp.array(12.0)) * kernels.ExpSquared(scale=60.0)
        gp = GaussianProcess(kernel, ot, diag=noise+1e-5, mean=mean)
        gp_cond = gp.condition(odata,nt).gp
        rng_key, rng_key_ = random.split(rng_key)
        realisations_list.append(gp_cond.sample(rng_key))
    return(np.array(realisations_list))

# %% Computing Predictions
nt = jnp.arange(ot.min(),ot.max()+1,1)
predictions = posterior_predictive_realisations(rng_key,ot,odata,nt,idata,20)

# %% Plotting the predictions
fig, ax = plt.subplots(1,1,figsize=(12, 5.0),dpi= 300)

ax.plot(nt,predictions.mean(axis=0),label='Prediction Expectation',alpha=0.6)
ax.fill_between(nt,
                predictions.mean(axis=0)+3*predictions.std(axis=0),
                predictions.mean(axis=0)-3*predictions.std(axis=0),
                interpolate=True, color='k',alpha=0.2,
                label='Prediction Uncertainty 3$\sigma$')
ax.scatter(ot,odata,label='Observations',alpha=0.6)

results_path = '/home/jez/Bias_Correction_Application/results/Paper_Images/'
# fig.savefig(f"{results_path}fig13.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{results_path}fig13.pdf", dpi=300, bbox_inches="tight")

# %%
