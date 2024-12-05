# %% Importing Packages
import numpyro
import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess
import jax
import jax.numpy as jnp
import arviz as az

from src.helper_functions import diagonal_noise
from src.helper_functions import run_inference

import numpy as np
from scipy.spatial import distance

from tinygp.kernels.distance import L2Distance

rng_key = jax.random.PRNGKey(1)
jax.config.update("jax_enable_x64", True)


##################### Site Level Model ########################################################################

def mean_model_obs(scenario):
    mean = numpyro.sample("mean_b0",scenario['meanfunc_b0_prior'])
    mean_b1 = numpyro.sample("mean_b1",scenario['meanfunc_b1_prior'])
    mean_b2 = numpyro.sample("mean_b2",scenario['meanfunc_b2_prior'])
    mean_noise = numpyro.sample("mean_noise",scenario['meanfunc_noise_prior'])

    mean_func = mean_b0 + mean_b1*scenario['oele_scaled'] + mean_b2*scenario['olat_scaled']
    mean = numpyro.sample("mean",dist.Normal(mean_func, mean_noise))
    
    logvar_b0 = numpyro.sample("logvar_b0",scenario['logvarfunc_b0_prior'])
    logvar_noise = numpyro.sample("logvar_noise",scenario['logvarfunc_noise_prior'])

    logvar_func = logvar_b0 * jnp.ones(scenario['ox'].shape[0])
    logvar = numpyro.sample("logvar",dist.Normal(logvar_func, logvar_noise))
    var = jnp.exp(logvar)

    obs_mask = (jnp.isnan(scenario['odata'])==False)
    numpyro.sample("Temperature", dist.Normal(mean, jnp.sqrt(var)).mask(obs_mask), obs=scenario["odata"])

def mean_model_climate(scenario):
    mean_b0 = numpyro.sample("mean_b0",scenario['meanfunc_b0_prior'])
    mean_b1 = numpyro.sample("mean_b1",scenario['meanfunc_b1_prior'])
    mean_b2 = numpyro.sample("mean_b2",scenario['meanfunc_b2_prior'])
    mean_noise = numpyro.sample("mean_noise",scenario['meanfunc_noise_prior'])

    mean_func = mean_b0 + mean_b1*scenario['cele_scaled'] + mean_b2*scenario['clat_scaled']
    mean = numpyro.sample("mean",dist.Normal(mean_func, mean_noise))

    logvar_b0 = numpyro.sample("logvar_b0",scenario['logvarfunc_b0_prior'])
    logvar_noise = numpyro.sample("logvar_noise",scenario['logvarfunc_noise_prior'])

    logvar_func = logvar_b0 * jnp.ones(scenario['cx'].shape[0])
    logvar = numpyro.sample("logvar",dist.Normal(logvar_func, logvar_noise))
    var = jnp.exp(logvar)

    numpyro.sample("Temperature", dist.Normal(mean, jnp.sqrt(var)), obs=scenario["cdata"])



##############################################################################################################

# %% Helper Functions

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



    # %%





def tinygp_2process_model_mean(scenario,cindecies):
    """
    Example model where the climate data is generated from 2 GPs,
    one of which also generates the observations and one of
    which generates bias in the climate model. We use mean = meanfunc + residual
    """
    kern_var = numpyro.sample("kern_var", scenario['mean_residual_obs_kvprior'])
    lengthscale = numpyro.sample("lengthscale", scenario['mean_residual_obs_klprior'])
    kernel = kern_var * kernels.Matern32(lengthscale,L2Distance())
    noise = numpyro.sample("noise", scenario['mean_residual_obs_nprior'])

    bkern_var = numpyro.sample("bkern_var", scenario['mean_residual_bias_kvprior'])
    blengthscale = numpyro.sample("blengthscale", scenario['mean_residual_bias_klprior'])
    bkernel = bkern_var * kernels.Matern32(blengthscale,L2Distance())
    bnoise = numpyro.sample("bnoise", scenario['mean_residual_bias_nprior'])

    ckernel = kernel + bkernel
    cnoise = noise + bnoise 
    cgp = GaussianProcess(ckernel, scenario["cx"][cindecies], diag=cnoise, mean=0)
    cres = numpyro.sample("climate_residual",
                                cgp.numpyro_dist(),
    )

    obs_conditional_climate_dist = generate_obs_conditional_climate_dist(
        scenario["ox"],
        scenario["cx"][cindecies],
        cres,
        ckernel,
        cnoise,
        kernel,
        noise,
    )

    ores = numpyro.sample(
        "aws_residual",
        obs_conditional_climate_dist
    )

    mean_b0 = numpyro.sample("mean_b0",scenario['meanfunc_b0_prior'])
    mean_b1 = numpyro.sample("mean_b1",scenario['meanfunc_b1_prior'])
    mean_b2 = numpyro.sample("mean_b2",scenario['meanfunc_b2_prior'])
    # bmean_b0 = numpyro.sample("bmean_b0",dist.Normal(0, 10.0))
    # bmean_b1 = numpyro.sample("bmean_b1",dist.Normal(0, 10.0))
    # bmean_b2 = numpyro.sample("bmean_b2",dist.Normal(0, 10.0))

    omeanfunc = mean_b0 + mean_b1*scenario['oele_scaled'] + mean_b2*scenario['olat_scaled']
    cmeanfunc = mean_b0 + mean_b1*scenario['cele_scaled'][cindecies] + mean_b2*scenario['clat_scaled'][cindecies]

    # cmeanfunc = ((mean_b0 + mean_b1*scenario['cele_scaled'][cindecies] + mean_b2*scenario['clat_scaled'][cindecies]) + 
    #              (bmean_b0 + bmean_b1*scenario['cele_scaled'][cindecies] + bmean_b2*scenario['clat_scaled'][cindecies])
    # )

    omean = omeanfunc + ores
    cmean = cmeanfunc + cres

    logvar_b0 = numpyro.sample("logvar_b0",scenario['logvarfunc_b0_prior'])
    logvar_noise = numpyro.sample("logvar_noise",scenario['logvarfunc_noise_prior'])
    logvar_func = logvar_b0 * jnp.ones(scenario['ox'].shape[0])
    logvar = numpyro.sample("logvar",dist.Normal(logvar_func, logvar_noise))
    var = jnp.exp(logvar)

    clogvar_b0 = numpyro.sample("clogvar_b0",scenario['logvarfunc_b0_prior'])
    clogvar_noise = numpyro.sample("clogvar_noise",scenario['logvarfunc_noise_prior'])
    clogvar_func = clogvar_b0 * jnp.ones(scenario['cx'][cindecies].shape[0])
    clogvar = numpyro.sample("clogvar",dist.Normal(clogvar_func, clogvar_noise))
    cvar = jnp.exp(clogvar)

    obs_mask = (jnp.isnan(scenario['odata'])==False)
    numpyro.sample("AWS Temperature", dist.Normal(omean, jnp.sqrt(var)).mask(obs_mask), obs=scenario["odata"])

    numpyro.sample("Climate Model Temperature", dist.Normal(cmean, jnp.sqrt(cvar)), obs=scenario["cdata"][:,cindecies])


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
    scenario["mcmc_dualprocess_hierarchical_test"] = idata_2process
    # scenario["mcmc_samples"] = mcmc_2process.get_samples()










# def mean_model_obs(scenario):
#     mean_b0 = numpyro.sample("mean_b0",scenario['meanfunc_b0_prior'])
#     mean_b1 = numpyro.sample("mean_b1",scenario['meanfunc_b1_prior'])
#     mean_b2 = numpyro.sample("mean_b2",scenario['meanfunc_b2_prior'])
#     mean_noise = numpyro.sample("mean_noise",scenario['meanfunc_noise_prior'])

#     mean_func = mean_b0 + mean_b1*scenario['oele_scaled'] + mean_b2*scenario['olat_scaled']
#     mean = numpyro.sample("mean",dist.Normal(mean_func, mean_noise))
    
#     logvar_b0 = numpyro.sample("logvar_b0",scenario['logvarfunc_b0_prior'])
#     logvar_noise = numpyro.sample("logvar_noise",scenario['logvarfunc_noise_prior'])

#     logvar_func = logvar_b0 * jnp.ones(scenario['ox'].shape[0])
#     logvar = numpyro.sample("logvar",dist.Normal(logvar_func, logvar_noise))
#     var = jnp.exp(logvar)

#     obs_mask = (jnp.isnan(scenario['odata'])==False)
#     numpyro.sample("Temperature", dist.Normal(mean, jnp.sqrt(var)).mask(obs_mask), obs=scenario["odata"])

# def mean_model_climate(scenario):
#     mean_b0 = numpyro.sample("mean_b0",scenario['meanfunc_b0_prior'])
#     mean_b1 = numpyro.sample("mean_b1",scenario['meanfunc_b1_prior'])
#     mean_b2 = numpyro.sample("mean_b2",scenario['meanfunc_b2_prior'])
#     mean_noise = numpyro.sample("mean_noise",scenario['meanfunc_noise_prior'])

#     mean_func = mean_b0 + mean_b1*scenario['cele_scaled'] + mean_b2*scenario['clat_scaled']
#     mean = numpyro.sample("mean",dist.Normal(mean_func, mean_noise))

#     logvar_b0 = numpyro.sample("logvar_b0",scenario['logvarfunc_b0_prior'])
#     logvar_noise = numpyro.sample("logvar_noise",scenario['logvarfunc_noise_prior'])

#     logvar_func = logvar_b0 * jnp.ones(scenario['cx'].shape[0])
#     logvar = numpyro.sample("logvar",dist.Normal(logvar_func, logvar_noise))
#     var = jnp.exp(logvar)

#     numpyro.sample("Temperature", dist.Normal(mean, jnp.sqrt(var)), obs=scenario["cdata"])

