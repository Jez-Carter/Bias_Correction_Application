# %% Importing Packages
import numpy as np
import jax
from jax import random
import matplotlib.pyplot as plt
import arviz as az
import numpyro.distributions as dist

plt.rcParams["lines.markersize"] = 3
plt.rcParams["lines.linewidth"] = 0.4
plt.rcParams.update({"font.size": 8})

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.spatial.singleprocess_model_functions import generate_posterior_singleprocess_mean
from src.spatial.singleprocess_model_functions import generate_posterior_singleprocess_logvar

# %% Loading data
base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

# %% Useful Metrics for Priors
print('Useful Metrics for Priors: \n',
      f"""
      Mean:
      min={scenario['exp_meanfunc_residual_obs'].min():.1f},
      mean={scenario['exp_meanfunc_residual_obs'].mean():.1f},
      max={scenario['exp_meanfunc_residual_obs'].max():.1f},
      var={scenario['exp_meanfunc_residual_obs'].var():.1f},
      \n,
      Log Variance,
      min={scenario['exp_logvarfunc_residual_obs'].min():.1f},
      mean={scenario['exp_logvarfunc_residual_obs'].mean():.1f},
      max={scenario['exp_logvarfunc_residual_obs'].max():.1f},
      var={scenario['exp_logvarfunc_residual_obs'].var():.1f},
      """
)

# %% Choosing Priors

lengthscale_max = ((scenario['ox'].max(axis=0)-scenario['ox'].min(axis=0))/2).max()

scenario['mean_residual_obs_kvprior'] = dist.Uniform(0.1,100.0)
scenario['mean_residual_obs_klprior'] = dist.Uniform(1,lengthscale_max/2)
scenario['mean_residual_obs_nprior'] = dist.Uniform(0.1,20.0)

scenario['logvar_residual_obs_kvprior'] = dist.Uniform(1e-3,0.75)
scenario['logvar_residual_obs_klprior'] = dist.Uniform(1,lengthscale_max)
scenario['logvar_residual_obs_nprior'] = dist.Uniform(1e-4,0.025)

# %% Fitting the model
generate_posterior_singleprocess_mean(scenario, rng_key, 1000, 2000, 1)
generate_posterior_singleprocess_logvar(scenario, rng_key, 1000, 2000, 1)

# %% Summary statistics from MCMC
az.summary(scenario["mcmc_singleprocess_mean_residual"].posterior, hdi_prob=0.95)
az.summary(scenario["mcmc_singleprocess_logvar_residual"].posterior, hdi_prob=0.95)

# %% Saving Output
scenario_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'
np.save(scenario_outpath, scenario)

# %%
