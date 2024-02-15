import numpy as np
import numpyro.distributions as dist
from jax import random
import jax.numpy as jnp
from tinygp import kernels
from scipy.stats import multivariate_normal
from tinygp.kernels.distance import L2Distance
from tqdm import tqdm

from src.helper_functions import diagonal_noise

def generate_truth_predictive_dist_mean(nx, scenario, posterior_param_realisation):
    kern_var_realisation = posterior_param_realisation["kern_var_realisation"]
    lengthscale_realisation = posterior_param_realisation["lengthscale_realisation"]
    noise_realisation = posterior_param_realisation["noise_realisation"]

    bkern_var_realisation = posterior_param_realisation["bkern_var_realisation"]
    blengthscale_realisation = posterior_param_realisation["blengthscale_realisation"]
    bnoise_realisation = posterior_param_realisation["bnoise_realisation"]

    ox = scenario["ox"]
    cx = scenario["cx"]
    odata = scenario["exp_meanfunc_residual_obs"]
    odata_var = scenario['var_meanfunc_residual_obs']
    cdata = scenario["exp_meanfunc_residual_climate"]
    kernelo = kern_var_realisation * kernels.Matern32(lengthscale_realisation,L2Distance())
    kernelb = bkern_var_realisation * kernels.Matern32(blengthscale_realisation,L2Distance())

    noise = noise_realisation + odata_var
    bnoise = bnoise_realisation
    cnoise = noise_realisation + bnoise

    jitter = 1e-5

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], 0)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], 0), jnp.full(cx.shape[0], 0)]
    )
    k11 = kernelo(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([kernelo(nx, ox), kernelo(nx, cx)])
    k21 = jnp.vstack([kernelo(ox, nx), kernelo(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, noise), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, cnoise),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)

    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    k1g2 = k11 - jnp.matmul(jnp.matmul(k12, k22i), k21)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_bias_predictive_dist_mean(nx, scenario, posterior_param_realisation):
    kern_var_realisation = posterior_param_realisation["kern_var_realisation"]
    lengthscale_realisation = posterior_param_realisation["lengthscale_realisation"]
    noise_realisation = posterior_param_realisation["noise_realisation"]

    bkern_var_realisation = posterior_param_realisation["bkern_var_realisation"]
    blengthscale_realisation = posterior_param_realisation["blengthscale_realisation"]
    bnoise_realisation = posterior_param_realisation["bnoise_realisation"]

    ox = scenario["ox"]
    cx = scenario["cx"]
    odata = scenario["exp_meanfunc_residual_obs"]
    odata_var = scenario['var_meanfunc_residual_obs']
    cdata = scenario["exp_meanfunc_residual_climate"]
    kernelo = kern_var_realisation * kernels.Matern32(lengthscale_realisation,L2Distance())
    kernelb = bkern_var_realisation * kernels.Matern32(blengthscale_realisation,L2Distance())

    noise = noise_realisation + odata_var
    bnoise = bnoise_realisation
    cnoise = noise_realisation + bnoise

    jitter = 1e-5

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], 0)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], 0), jnp.full(cx.shape[0], 0)]
    )
    k11 = kernelb(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([jnp.full((len(nx), len(ox)), 0), kernelb(nx, cx)])
    k21 = jnp.vstack([jnp.full((len(ox), len(nx)), 0), kernelb(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, noise), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, cnoise),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn

def generate_posterior_predictive_realisations_dualprocess_mean(
    nx,
    scenario,
    num_parameter_realisations,
    num_posterior_pred_realisations,
    rng_key
):
    posterior = scenario["mcmc_dualprocess_mean_residual"].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    iteration = 0
    for i in tqdm(np.random.randint(posterior.draw.shape, size=num_parameter_realisations)):
        posterior_param_realisation = {
            "iteration": i,
            "kern_var_realisation": posterior["kern_var"].data[0, :][i],
            "lengthscale_realisation": posterior["lengthscale"].data[0, :][i],
            "noise_realisation": posterior["noise"].data[0, :][i],
            "bkern_var_realisation": posterior["bkern_var"].data[0, :][i],
            "blengthscale_realisation": posterior["blengthscale"].data[0, :][i],
            "bnoise_realisation": posterior["bnoise"].data[0, :][i],
        }

        truth_predictive_dist = generate_truth_predictive_dist_mean(
            nx, scenario, posterior_param_realisation
        )
        bias_predictive_dist = generate_bias_predictive_dist_mean(
            nx, scenario, posterior_param_realisation
        )
        iteration += 1

        # truth_predictive_realisations = truth_predictive_dist.rvs(num_posterior_pred_realisations)
        # bias_predictive_realisations = bias_predictive_dist.rvs(num_posterior_pred_realisations)
        # rng_key = random.PRNGKey(0)
        truth_predictive_realisations = truth_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)
        bias_predictive_realisations = bias_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)

        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(
        truth_posterior_predictive_realisations
    )
    bias_posterior_predictive_realisations = jnp.array(
        bias_posterior_predictive_realisations
    )
    truth_posterior_predictive_realisations = (
        truth_posterior_predictive_realisations.reshape(
            -1, truth_posterior_predictive_realisations.shape[-1]
        )
    )
    bias_posterior_predictive_realisations = (
        bias_posterior_predictive_realisations.reshape(
            -1, bias_posterior_predictive_realisations.shape[-1]
        )
    )
    # scenario[
    #     "truth_posterior_predictive_realisations_dualprocess_mean"
    # ] = truth_posterior_predictive_realisations
    # scenario[
    #     "bias_posterior_predictive_realisations_dualprocess_mean"
    # ] = bias_posterior_predictive_realisations

    return(truth_posterior_predictive_realisations,
           bias_posterior_predictive_realisations)


### Equivalent LOGVAR functions

def generate_truth_predictive_dist_logvar(nx, scenario, posterior_param_realisation):
    kern_var_realisation = posterior_param_realisation["kern_var_realisation"]
    lengthscale_realisation = posterior_param_realisation["lengthscale_realisation"]
    noise_realisation = posterior_param_realisation["noise_realisation"]

    bkern_var_realisation = posterior_param_realisation["bkern_var_realisation"]
    blengthscale_realisation = posterior_param_realisation["blengthscale_realisation"]
    bnoise_realisation = posterior_param_realisation["bnoise_realisation"]

    ox = scenario["ox"]
    cx = scenario["cx"]
    odata = scenario["exp_logvarfunc_residual_obs"]
    odata_var = scenario['var_logvarfunc_residual_obs']
    cdata = scenario["exp_logvarfunc_residual_climate"]
    kernelo = kern_var_realisation * kernels.Matern32(lengthscale_realisation,L2Distance())
    kernelb = bkern_var_realisation * kernels.Matern32(blengthscale_realisation,L2Distance())

    noise = noise_realisation + odata_var
    bnoise = bnoise_realisation
    cnoise = noise_realisation + bnoise

    jitter = 1e-5

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], 0)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], 0), jnp.full(cx.shape[0], 0)]
    )
    k11 = kernelo(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([kernelo(nx, ox), kernelo(nx, cx)])
    k21 = jnp.vstack([kernelo(ox, nx), kernelo(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, noise), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, cnoise),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)

    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    k1g2 = k11 - jnp.matmul(jnp.matmul(k12, k22i), k21)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_bias_predictive_dist_logvar(nx, scenario, posterior_param_realisation):
    kern_var_realisation = posterior_param_realisation["kern_var_realisation"]
    lengthscale_realisation = posterior_param_realisation["lengthscale_realisation"]
    noise_realisation = posterior_param_realisation["noise_realisation"]

    bkern_var_realisation = posterior_param_realisation["bkern_var_realisation"]
    blengthscale_realisation = posterior_param_realisation["blengthscale_realisation"]
    bnoise_realisation = posterior_param_realisation["bnoise_realisation"]

    ox = scenario["ox"]
    cx = scenario["cx"]
    odata = scenario["exp_logvarfunc_residual_obs"]
    odata_var = scenario['var_logvarfunc_residual_obs']
    cdata = scenario["exp_logvarfunc_residual_climate"]
    kernelo = kern_var_realisation * kernels.Matern32(lengthscale_realisation,L2Distance())
    kernelb = bkern_var_realisation * kernels.Matern32(blengthscale_realisation,L2Distance())

    noise = noise_realisation + odata_var
    bnoise = bnoise_realisation
    cnoise = noise_realisation + bnoise

    jitter = 1e-5

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], 0)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], 0), jnp.full(cx.shape[0], 0)]
    )
    k11 = kernelb(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([jnp.full((len(nx), len(ox)), 0), kernelb(nx, cx)])
    k21 = jnp.vstack([jnp.full((len(ox), len(nx)), 0), kernelb(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, noise), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, cnoise),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn

def generate_posterior_predictive_realisations_dualprocess_logvar(
    nx,
    scenario,
    num_parameter_realisations,
    num_posterior_pred_realisations,
    rng_key
):
    posterior = scenario["mcmc_dualprocess_logvar_residual"].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    iteration = 0
    for i in tqdm(np.random.randint(posterior.draw.shape, size=num_parameter_realisations)):
        posterior_param_realisation = {
            "iteration": i,
            "kern_var_realisation": posterior["kern_var"].data[0, :][i],
            "lengthscale_realisation": posterior["lengthscale"].data[0, :][i],
            "noise_realisation": posterior["noise"].data[0, :][i],
            "bkern_var_realisation": posterior["bkern_var"].data[0, :][i],
            "blengthscale_realisation": posterior["blengthscale"].data[0, :][i],
            "bnoise_realisation": posterior["bnoise"].data[0, :][i],
        }

        truth_predictive_dist = generate_truth_predictive_dist_logvar(
            nx, scenario, posterior_param_realisation
        )
        bias_predictive_dist = generate_bias_predictive_dist_logvar(
            nx, scenario, posterior_param_realisation
        )
        iteration += 1

        # truth_predictive_realisations = truth_predictive_dist.rvs(num_posterior_pred_realisations)
        # bias_predictive_realisations = bias_predictive_dist.rvs(num_posterior_pred_realisations)
        # rng_key = random.PRNGKey(0)
        truth_predictive_realisations = truth_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)
        bias_predictive_realisations = bias_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)

        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(
        truth_posterior_predictive_realisations
    )
    bias_posterior_predictive_realisations = jnp.array(
        bias_posterior_predictive_realisations
    )
    truth_posterior_predictive_realisations = (
        truth_posterior_predictive_realisations.reshape(
            -1, truth_posterior_predictive_realisations.shape[-1]
        )
    )
    bias_posterior_predictive_realisations = (
        bias_posterior_predictive_realisations.reshape(
            -1, bias_posterior_predictive_realisations.shape[-1]
        )
    )
    # scenario[
    #     "truth_posterior_predictive_realisations_dualprocess_logvar"
    # ] = truth_posterior_predictive_realisations
    # scenario[
    #     "bias_posterior_predictive_realisations_dualprocess_logvar"
    # ] = bias_posterior_predictive_realisations

    return(truth_posterior_predictive_realisations,
           bias_posterior_predictive_realisations)


##### Generating at Observation Locations
def generate_posterior_predictive_realisations_dualprocess_mean_station_locations(
    scenario,
    num_parameter_realisations,
    num_posterior_pred_realisations,
    rng_key,
    ):
    nx=scenario['ox']
    posterior = scenario["mcmc_dualprocess_mean_residual"].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    iteration = 0
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            "iteration": i,
            "kern_var_realisation": posterior["kern_var"].data[0, :][i],
            "lengthscale_realisation": posterior["lengthscale"].data[0, :][i],
            "noise_realisation": posterior["noise"].data[0, :][i],
            "bkern_var_realisation": posterior["bkern_var"].data[0, :][i],
            "blengthscale_realisation": posterior["blengthscale"].data[0, :][i],
            "bnoise_realisation": posterior["bnoise"].data[0, :][i],
        }

        truth_predictive_dist = generate_truth_predictive_dist_mean(
            nx, scenario, posterior_param_realisation
        )
        bias_predictive_dist = generate_bias_predictive_dist_mean(
            nx, scenario, posterior_param_realisation
        )
        iteration += 1

        truth_predictive_realisations = truth_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)
        bias_predictive_realisations = bias_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)

        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(
        truth_posterior_predictive_realisations
    )
    bias_posterior_predictive_realisations = jnp.array(
        bias_posterior_predictive_realisations
    )
    truth_posterior_predictive_realisations = (
        truth_posterior_predictive_realisations.reshape(
            -1, truth_posterior_predictive_realisations.shape[-1]
        )
    )
    bias_posterior_predictive_realisations = (
        bias_posterior_predictive_realisations.reshape(
            -1, bias_posterior_predictive_realisations.shape[-1]
        )
    )
    scenario[
        "truth_posterior_predictive_realisations_dualprocess_mean_station_locations"
    ] = truth_posterior_predictive_realisations
    scenario[
        "bias_posterior_predictive_realisations_dualprocess_mean_station_locations"
    ] = bias_posterior_predictive_realisations

def generate_posterior_predictive_realisations_dualprocess_logvar_station_locations(
    scenario,
    num_parameter_realisations,
    num_posterior_pred_realisations,
    rng_key
):
    posterior = scenario["mcmc_dualprocess_logvar_residual"].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    iteration = 0
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            "iteration": i,
            "kern_var_realisation": posterior["kern_var"].data[0, :][i],
            "lengthscale_realisation": posterior["lengthscale"].data[0, :][i],
            "noise_realisation": posterior["noise"].data[0, :][i],
            "bkern_var_realisation": posterior["bkern_var"].data[0, :][i],
            "blengthscale_realisation": posterior["blengthscale"].data[0, :][i],
            "bnoise_realisation": posterior["bnoise"].data[0, :][i],
        }

        truth_predictive_dist = generate_truth_predictive_dist_logvar(
            scenario['ox'], scenario, posterior_param_realisation
        )
        bias_predictive_dist = generate_bias_predictive_dist_logvar(
            scenario['ox'], scenario, posterior_param_realisation
        )
        iteration += 1

        truth_predictive_realisations = truth_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)
        bias_predictive_realisations = bias_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)

        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(
        truth_posterior_predictive_realisations
    )
    bias_posterior_predictive_realisations = jnp.array(
        bias_posterior_predictive_realisations
    )
    truth_posterior_predictive_realisations = (
        truth_posterior_predictive_realisations.reshape(
            -1, truth_posterior_predictive_realisations.shape[-1]
        )
    )
    bias_posterior_predictive_realisations = (
        bias_posterior_predictive_realisations.reshape(
            -1, bias_posterior_predictive_realisations.shape[-1]
        )
    )
    scenario[
        "truth_posterior_predictive_realisations_dualprocess_logvar_station_locations"
    ] = truth_posterior_predictive_realisations
    scenario[
        "bias_posterior_predictive_realisations_dualprocess_logvar_station_locations"
    ] = bias_posterior_predictive_realisations
