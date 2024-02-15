import numpy as np
import numpyro
from tinygp import kernels, GaussianProcess
from tinygp.kernels.distance import L2Distance
import jax
from jax import random
import jax.numpy as jnp
import arviz as az
from tqdm import tqdm

from src.helper_functions import run_inference

jax.config.update("jax_enable_x64", True)

def singleprocess_model_mean(scenario):
    """
    Example model where the truth is modelled just using the 
    observational data, which is generated from a GP
    """
    kern_var = numpyro.sample("kern_var", scenario['mean_residual_obs_kvprior'])
    lengthscale = numpyro.sample("lengthscale", scenario['mean_residual_obs_klprior'])
    kernel = kern_var * kernels.Matern32(lengthscale,L2Distance())
    noise = numpyro.sample("noise", scenario['mean_residual_obs_nprior'])
    data = scenario['exp_meanfunc_residual_obs']
    diag = scenario['var_meanfunc_residual_obs']+noise

    gp = GaussianProcess(kernel, scenario['ox'], diag=diag, mean=0)
    numpyro.sample("observations", gp.numpyro_dist(),obs=data)

def singleprocess_model_logvar(scenario):
    """
    Example model where the truth is modelled just using the 
    observational data, which is generated from a GP
    """
    kern_var = numpyro.sample("kern_var", scenario['logvar_residual_obs_kvprior'])
    lengthscale = numpyro.sample("lengthscale", scenario['logvar_residual_obs_klprior'])
    kernel = kern_var * kernels.Matern32(lengthscale,L2Distance())
    noise = numpyro.sample("noise", scenario['logvar_residual_obs_nprior'])
    data = scenario['exp_logvarfunc_residual_obs']
    diag = scenario['var_logvarfunc_residual_obs']+noise

    gp = GaussianProcess(kernel, scenario['ox'], diag=diag, mean=0)
    numpyro.sample("observations", gp.numpyro_dist(),obs=data)

def generate_posterior_singleprocess_mean(scenario,rng_key,num_warmup,num_samples,num_chains):    
    mcmc = run_inference(
        singleprocess_model_mean, rng_key, num_warmup, num_samples,num_chains,scenario)
    idata = az.from_numpyro(mcmc)
    scenario[f'mcmc_singleprocess_mean_residual'] = idata

def generate_posterior_singleprocess_logvar(scenario,rng_key,num_warmup,num_samples,num_chains):    
    mcmc = run_inference(
        singleprocess_model_logvar, rng_key, num_warmup, num_samples,num_chains,scenario)
    idata = az.from_numpyro(mcmc)
    scenario[f'mcmc_singleprocess_logvar_residual'] = idata
    # scenario['mcmc_singleprocess_samples'] = mcmc.get_samples()

def posterior_predictive_dist_singleprocess_mean(nx, scenario, posterior_param_realisation):
    kern_var_realisation = posterior_param_realisation["kern_var_realisation"]
    lengthscale_realisation = posterior_param_realisation["lengthscale_realisation"]
    noise_realisation = posterior_param_realisation["noise_realisation"]
    data = scenario['exp_meanfunc_residual_obs']
    diag = scenario['var_meanfunc_residual_obs']+noise_realisation

    kernel = kern_var_realisation * kernels.Matern32(lengthscale_realisation,L2Distance())
    gp = GaussianProcess(kernel, scenario['ox'], diag=diag, mean=0)
    gp_cond = gp.condition(data, nx, diag=1e-5).gp
    return gp_cond.numpyro_dist()

def posterior_predictive_dist_singleprocess_logvar(nx, scenario, posterior_param_realisation):
    kern_var_realisation = posterior_param_realisation["kern_var_realisation"]
    lengthscale_realisation = posterior_param_realisation["lengthscale_realisation"]
    noise_realisation = posterior_param_realisation["noise_realisation"]
    data = scenario['exp_logvarfunc_residual_obs']
    diag = scenario['var_logvarfunc_residual_obs']+noise_realisation

    kernel = kern_var_realisation * kernels.Matern32(lengthscale_realisation,L2Distance())
    gp = GaussianProcess(kernel, scenario['ox'], diag=diag, mean=0)
    gp_cond = gp.condition(data, nx, diag=1e-5).gp
    return gp_cond.numpyro_dist()

def generate_posterior_predictive_realisations_singleprocess_mean(
    nx, scenario, num_parameter_realisations, num_posterior_pred_realisations, rng_key):
    posterior = scenario["mcmc_singleprocess_mean_residual"].posterior
    posterior_predictive_realisations = []
    for i in tqdm(np.random.randint(posterior.draw.shape, size=num_parameter_realisations)):
        posterior_param_realisation = {
            "kern_var_realisation": posterior["kern_var"].data[0, :][i],
            "lengthscale_realisation": posterior["lengthscale"].data[0, :][i],
            "noise_realisation": posterior["noise"].data[0, :][i],
        }
        predictive_dist = posterior_predictive_dist_singleprocess_mean(
            nx, scenario, posterior_param_realisation
        )

        predictive_realisations = predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)
        posterior_predictive_realisations.append(predictive_realisations)

    posterior_predictive_realisations = jnp.array(
        posterior_predictive_realisations
    )
    posterior_predictive_realisations = (
        posterior_predictive_realisations.reshape(
            -1, posterior_predictive_realisations.shape[-1]
        )
    )
    # scenario[
    #     "posterior_predictive_realisations_singleprocess_mean"
    # ] = posterior_predictive_realisations

    return(posterior_predictive_realisations)

def generate_posterior_predictive_realisations_singleprocess_logvar(
    nx, scenario, num_parameter_realisations, num_posterior_pred_realisations, rng_key):
    posterior = scenario["mcmc_singleprocess_logvar_residual"].posterior
    posterior_predictive_realisations = []
    for i in tqdm(np.random.randint(posterior.draw.shape, size=num_parameter_realisations)):
        posterior_param_realisation = {
            "kern_var_realisation": posterior["kern_var"].data[0, :][i],
            "lengthscale_realisation": posterior["lengthscale"].data[0, :][i],
            "noise_realisation": posterior["noise"].data[0, :][i],
        }
        predictive_dist = posterior_predictive_dist_singleprocess_logvar(
            nx, scenario, posterior_param_realisation
        )

        predictive_realisations = predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)
        posterior_predictive_realisations.append(predictive_realisations)

    posterior_predictive_realisations = jnp.array(
        posterior_predictive_realisations
    )
    posterior_predictive_realisations = (
        posterior_predictive_realisations.reshape(
            -1, posterior_predictive_realisations.shape[-1]
        )
    )
    # scenario[
    #     "posterior_predictive_realisations_singleprocess_logvar"
    # ] = posterior_predictive_realisations

    return(posterior_predictive_realisations)