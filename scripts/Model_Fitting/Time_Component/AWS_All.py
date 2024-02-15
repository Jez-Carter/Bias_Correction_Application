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

df_aws_group = df_aws_group[~df_aws_group.index.duplicated()]
df_aws_group_coords = df_aws_group_coords[~df_aws_group_coords.index.duplicated()]

da_temp = df_aws_group['Temperature'].to_xarray()
ds_coords = df_aws_group_coords[['Lat(℃)','Lon(℃)','Elevation(m)','glat','glon','grid_latitude', 'grid_longitude']].to_xarray()

ds_aws = xr.merge([ds_coords,da_temp])
ds_aws_stacked = ds_aws.stack(X=('Year','Month'))

ds_aws_stacked = ds_aws_stacked.where(ds_aws_stacked['Temperature'].count('X')>5,drop=True)
ds_aws_stacked = ds_aws_stacked.reset_index('X')

# %% Reformatting Data
ot = jnp.array(ds_aws_stacked.X.values).astype(jnp.float64)
odata = jnp.array(ds_aws_stacked['Temperature'].values)
obs_mask = (jnp.isnan(odata)==False)

otlist = [ot[mask] for mask in obs_mask]
odatalist = [data[mask] for data,mask in zip(odata,obs_mask)]
meansarray = jnp.array([odata.mean() for odata in odatalist])
varsarray = jnp.array([odata.var() for odata in odatalist])

print(f'ot shape = {ot.shape} \n',
      f'odata shape = {odata.shape} \n',
      f'obs_mask shape = {obs_mask.shape} \n',
      f'otlist length = {len(otlist)} \n',
      f'odatalist length = {len(odatalist)} \n',
      f'meansarray shape = {meansarray.shape} \n',
      f'varsarray shape = {varsarray.shape} \n')

# %% Plotting the mean and variance histograms
fig, ax = plt.subplots(1, 1, figsize=(10, 6),dpi=300)#,frameon=False)
ax.hist(meansarray,bins=20,alpha=0.6)
ax.hist(varsarray,bins=20,alpha=0.6)

# %% Defining Simple Time Series Model
def tinygp_model_multiple(tlist,jdatalist=None,means=None,vars=None):
    means = numpyro.sample(f"mean", dist.Normal(means, 10.0))
    kern_vars = numpyro.sample(f"kern_var", dist.TruncatedNormal(vars, 50.0,low=1,high=300))
    noises = numpyro.sample(f"noise", dist.HalfNormal(20),sample_shape=(len(tlist),))                           
    for t,jdata,mean,kern_var,noise,i in zip(tlist,jdatalist,means,kern_vars,noises,range(len(tlist))):
        kernel = kern_var * kernels.ExpSineSquared(scale=12.0,gamma=12.0) * kernels.ExpSquared(scale=60.0)    
        gp = GaussianProcess(kernel, t, diag=noise+1e-5, mean=mean)
        numpyro.sample(f"temperature_{i}", gp.numpyro_dist(),obs=jdata)

# %% Running Inference
mcmc = run_inference(tinygp_model_multiple,
                    rng_key,
                    1000,
                    2000,
                    1,
                    otlist[:10],
                    jdatalist=odatalist[:10],
                    means=meansarray[:10],
                    vars=varsarray[:10],
                    )
idata = az.from_numpyro(mcmc)

# %%
def posterior_predictive_realisations(
        rng_key,ot,odata,nt,idata,num_parameter_realisations):
    
    posterior = idata.posterior
    kern_vars_all = posterior['kern_var'].data[0,:]
    means_all = posterior['mean'].data[0,:]
    noises_all = posterior['noise'].data[0,:]

    realisations_list_all = []
    shape = kern_vars_all.shape
    
    for i in range(shape[1]):
        kern_vars = kern_vars_all[:,i]
        means = means_all[:,i]
        noises = noises_all[:,i]
        odata=odatalist[i]
        ot = otlist[i] 

        realisations_list = []
        for kern_var,mean,noise in tqdm(list(zip(kern_vars,means,noises))[:num_parameter_realisations]):
            kernel = kern_var * kernels.ExpSineSquared(scale=jnp.array(12.0),gamma=jnp.array(12.0)) * kernels.ExpSquared(scale=60.0) 
            gp = GaussianProcess(kernel, ot, diag=noise+1e-5, mean=mean)
            gp_cond = gp.condition(odata,nt).gp
            rng_key, rng_key_ = random.split(rng_key)
            realisations_list.append(gp_cond.sample(rng_key))
        realisations_list_all.append(realisations_list)
    return(np.array(realisations_list_all))

# %% Computing Predictions
nt = jnp.arange(1,504+1,1)
predictions = posterior_predictive_realisations(rng_key,ot,odata,nt,idata,20)

# %% Plotting the predictions
fig, axs = plt.subplots(10,1,figsize=(10, 50.0),dpi= 300)

for i in range(10):
    ax=axs[i]
    ds_station = ds_aws_stacked.isel(Station=i)
    station_name = ds_station.Station.values

    ax.plot(nt,predictions[i].mean(axis=0),label='Prediction Expectation',alpha=0.6)
    ax.fill_between(nt,
                    predictions[i].mean(axis=0)+3*predictions[i].std(axis=0),
                    predictions[i].mean(axis=0)-3*predictions[i].std(axis=0),
                    interpolate=True, color='k',alpha=0.2,
                    label='Prediction Uncertainty 3$\sigma$')
    ax.scatter(otlist[i],odatalist[i],label='Observations',alpha=0.6)

