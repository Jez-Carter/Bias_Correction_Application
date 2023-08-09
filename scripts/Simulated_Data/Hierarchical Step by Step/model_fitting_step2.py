# %% Importing Packages
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import arviz as az
import numpyro
import pandas as pd
from tinygp import kernels, GaussianProcess

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4
plt.rcParams.update({'font.size': 8})

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.simulated_data_functions import run_inference
from src.simulated_data_functions_hierarchical import generate_mt_conditional_mc_dist

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %% Specifications

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)

pd.options.display.max_colwidth = 100

plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1

legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm

out_path = '/home/jez/Bias_Correction/results/Paper_Images/'

jax.config.update("jax_enable_x64", True)

# %% Loading Data
scenario_base = np.load(f'{inpath}scenario_base_hierarchical_step_by_step.npy',allow_pickle='TRUE').item()

# %% Model Helper Functions
def diagonal_noise(coord,noise):
    return(jnp.diag(jnp.full(coord.shape[0],noise)))

def generate_mt_conditional_mc_dist(scenario,
                                    mc_kernel,
                                    mc_mean,
                                    mc_noise,
                                    mt_kernel,
                                    mt_mean,
                                    mt_noise,
                                    mc):
    ox = scenario['ox']
    cx = scenario['cx']
    y2 = mc
    u1 = jnp.full(ox.shape[0], mt_mean)
    u2 = jnp.full(cx.shape[0], mc_mean)
    k11 = mt_kernel(ox,ox) + diagonal_noise(ox,mt_noise)
    k12 = mt_kernel(ox,cx)
    k21 = mt_kernel(cx,ox) 
    k22 = mc_kernel(cx,cx) + diagonal_noise(cx,mc_noise)
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)
    mvn_dist = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn_dist)

# %% Defining Model
def spatial_model(scenario):
    mt_kern_var = numpyro.sample("mt_kern_var", scenario['MEAN_T_variance_prior'])
    mt_lengthscale = numpyro.sample("mt_lengthscale", scenario['MEAN_T_lengthscale_prior'])
    mt_kernel = mt_kern_var * kernels.ExpSquared(mt_lengthscale)
    mt_mean = numpyro.sample("mt_mean", scenario['MEAN_T_mean_prior'])
    mb_kern_var = numpyro.sample("mb_kern_var", scenario['MEAN_B_variance_prior'])
    mb_lengthscale = numpyro.sample("mb_lengthscale", scenario['MEAN_B_lengthscale_prior'])
    mb_kernel = mb_kern_var * kernels.ExpSquared(mb_lengthscale)
    mb_mean = numpyro.sample("mb_mean", scenario['MEAN_B_mean_prior'])

    mc_kernel = mt_kernel+mb_kernel
    mc_mean = mt_mean+mb_mean

    mt_samples = scenario['mcmc_step1_samples']['mt']
    mc_samples = scenario['mcmc_step1_samples']['mc_t']+scenario['mcmc_step1_samples']['mc_b']

    mc_gp = GaussianProcess(mc_kernel, scenario['cx'], diag=mc_samples.std(axis=0), mean=mc_mean)
    mc = numpyro.sample("mc", mc_gp.numpyro_dist(),obs=mc_samples.mean(axis=0))

    mt_conditional_mc_dist = generate_mt_conditional_mc_dist(scenario,
                                                             mc_kernel,
                                                             mc_mean,
                                                             mc_samples.std(axis=0),
                                                             mt_kernel,
                                                             mt_mean,
                                                             mt_samples.std(axis=0),
                                                             mc)

    numpyro.sample("mt", mt_conditional_mc_dist,obs=mt_samples.mean(axis=0))

# %%
rng_key = random.PRNGKey(0)
mcmc_spatial = run_inference(
        spatial_model, rng_key, 1000, 2000,1,scenario_base)
idata_spatial = az.from_numpyro(mcmc_spatial)
scenario_base['mcmc_step2'] = idata_spatial
scenario_base['mcmc_step2_samples']=mcmc_spatial.get_samples()

# %%
outpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
np.save(f'{outpath}scenario_base_hierarchical_step_by_step.npy', scenario_base) 
