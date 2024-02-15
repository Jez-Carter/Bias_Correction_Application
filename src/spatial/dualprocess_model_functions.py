import numpyro
import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess
import jax.numpy as jnp
import arviz as az
from tinygp.kernels.distance import L2Distance

from src.helper_functions import diagonal_noise
from src.helper_functions import run_inference


def generate_obs_conditional_climate_dist(
    ox, cx, cdata, ckernel, cdiag, okernel, odiag
):
    y2 = cdata
    u1 = jnp.full(ox.shape[0], 0)
    u2 = jnp.full(cx.shape[0], 0)
    k11 = okernel(ox, ox) + diagonal_noise(ox, odiag)
    k12 = okernel(ox, cx)
    k21 = okernel(cx, ox)
    k22 = ckernel(cx, cx) + diagonal_noise(cx, cdiag)
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn_dist = dist.MultivariateNormal(u1g2, k1g2)
    return mvn_dist


def tinygp_2process_model_mean(scenario,cindecies):
    """
    Example model where the climate data is generated from 2 GPs,
    one of which also generates the observations and one of
    which generates bias in the climate model.
    """
    kern_var = numpyro.sample("kern_var", scenario['mean_residual_obs_kvprior'])
    lengthscale = numpyro.sample("lengthscale", scenario['mean_residual_obs_klprior'])
    kernel = kern_var * kernels.Matern32(lengthscale,L2Distance())
    noise = numpyro.sample("noise", scenario['mean_residual_obs_nprior'])
    var_obs = scenario['var_meanfunc_residual_obs']

    bkern_var = numpyro.sample("bkern_var", scenario['mean_residual_bias_kvprior'])
    blengthscale = numpyro.sample("blengthscale", scenario['mean_residual_bias_klprior'])
    bkernel = bkern_var * kernels.Matern32(blengthscale,L2Distance())
    bnoise = numpyro.sample("bnoise", scenario['mean_residual_bias_nprior'])

    ckernel = kernel + bkernel
    cnoise = noise + bnoise 
    cgp = GaussianProcess(ckernel, scenario["cx"][cindecies], diag=cnoise, mean=0)
    numpyro.sample("climate_temperature",
                   cgp.numpyro_dist(),
                   obs=scenario["exp_meanfunc_residual_climate"][cindecies])

    obs_conditional_climate_dist = generate_obs_conditional_climate_dist(
        scenario["ox"],
        scenario["cx"][cindecies],
        scenario['exp_meanfunc_residual_climate'][cindecies],
        ckernel,
        cnoise,
        kernel,
        var_obs+noise
    )
    numpyro.sample(
        "obs_temperature",
        obs_conditional_climate_dist,
        obs=scenario["exp_meanfunc_residual_obs"]
    )

def tinygp_2process_model_logvar(scenario,cindecies):
    """
    Example model where the climate data is generated from 2 GPs,
    one of which also generates the observations and one of
    which generates bias in the climate model.
    """
    kern_var = numpyro.sample("kern_var", scenario['logvar_residual_obs_kvprior'])
    lengthscale = numpyro.sample("lengthscale", scenario['logvar_residual_obs_klprior'])
    kernel = kern_var * kernels.Matern32(lengthscale,L2Distance())
    noise = numpyro.sample("noise", scenario['logvar_residual_obs_nprior'])
    var_obs = scenario['var_logvarfunc_residual_obs']

    bkern_var = numpyro.sample("bkern_var", scenario['logvar_residual_bias_kvprior'])
    blengthscale = numpyro.sample("blengthscale", scenario['logvar_residual_bias_klprior'])
    bkernel = bkern_var * kernels.Matern32(blengthscale,L2Distance())
    bnoise = numpyro.sample("bnoise", scenario['logvar_residual_bias_nprior'])

    ckernel = kernel + bkernel
    cnoise = noise + bnoise 
    cgp = GaussianProcess(ckernel, scenario["cx"][cindecies], diag=cnoise, mean=0)
    numpyro.sample("climate_temperature",
                   cgp.numpyro_dist(),
                   obs=scenario["exp_logvarfunc_residual_climate"][cindecies])

    obs_conditional_climate_dist = generate_obs_conditional_climate_dist(
        scenario["ox"],
        scenario["cx"][cindecies],
        scenario['exp_logvarfunc_residual_climate'][cindecies],
        ckernel,
        cnoise,
        kernel,
        var_obs+noise
    )
    numpyro.sample(
        "obs_temperature",
        obs_conditional_climate_dist,
        obs=scenario["exp_logvarfunc_residual_obs"]
    )


def generate_posterior_dualprocess_mean(scenario,
                                        cindecies,
                                        rng_key,
                                        num_warmup,
                                        num_samples,
                                        num_chains):
    mcmc_2process = run_inference(
        tinygp_2process_model_mean,
        rng_key,
        num_warmup,
        num_samples,
        num_chains,
        scenario,
        cindecies
    )
    idata_2process = az.from_numpyro(mcmc_2process)
    scenario["mcmc_dualprocess_mean_residual"] = idata_2process
    # scenario["mcmc_samples"] = mcmc_2process.get_samples()

def generate_posterior_dualprocess_logvar(scenario,
                                        cindecies,
                                        rng_key,
                                        num_warmup,
                                        num_samples,
                                        num_chains):
    mcmc_2process = run_inference(
        tinygp_2process_model_logvar,
        rng_key,
        num_warmup,
        num_samples,
        num_chains,
        scenario,
        cindecies
    )
    idata_2process = az.from_numpyro(mcmc_2process)
    scenario["mcmc_dualprocess_logvar_residual"] = idata_2process
    # scenario["mcmc_samples"] = mcmc_2process.get_samples()