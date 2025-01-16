# %% Importing necessary libraries
import numpy as np
import numpyro.distributions as dist
import jax
from jax import random
import jax.numpy as jnp
from tinygp import kernels
from scipy.stats import multivariate_normal
from tinygp.kernels.distance import L2Distance
from tqdm import tqdm

from src.helper_functions import diagonal_noise
import pickle 

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)


# %% Loading the data
with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary.pkl', 'rb') as f:
    data_dictionary = pickle.load(f)

# %% Defining function for generating posterior predictive realisations of the residuals

def generate_truth_predictive_dist_mean(nx, data_dictionary, posterior_param_realisation):
    kern_var_realisation = posterior_param_realisation["kern_var_realisation"]
    lengthscale_realisation = posterior_param_realisation["lengthscale_realisation"]
    noise_realisation = posterior_param_realisation["noise_realisation"]

    bkern_var_realisation = posterior_param_realisation["bkern_var_realisation"]
    blengthscale_realisation = posterior_param_realisation["blengthscale_realisation"]
    bnoise_realisation = posterior_param_realisation["bnoise_realisation"]

    meanfunc_posterior = data_dictionary['meanfunc_posterior']
    omean_func_residual_exp = meanfunc_posterior['omean_func_residual'].mean(['draw','chain']).data
    omean_func_residual_var = meanfunc_posterior['omean_func_residual'].var(['draw','chain']).data
    cmean_func_residual_exp = meanfunc_posterior['cmean_func_residual'].mean(['draw','chain']).data

    ox = data_dictionary["ox"]
    cx = data_dictionary["cx"]
    odata = omean_func_residual_exp
    odata_var = omean_func_residual_var
    cdata = cmean_func_residual_exp
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


def generate_bias_predictive_dist_mean(nx, data_dictionary, posterior_param_realisation):
    kern_var_realisation = posterior_param_realisation["kern_var_realisation"]
    lengthscale_realisation = posterior_param_realisation["lengthscale_realisation"]
    noise_realisation = posterior_param_realisation["noise_realisation"]

    bkern_var_realisation = posterior_param_realisation["bkern_var_realisation"]
    blengthscale_realisation = posterior_param_realisation["blengthscale_realisation"]
    bnoise_realisation = posterior_param_realisation["bnoise_realisation"]

    meanfunc_posterior = data_dictionary['meanfunc_posterior']
    omean_func_residual_exp = meanfunc_posterior['omean_func_residual'].mean(['draw','chain']).data
    omean_func_residual_var = meanfunc_posterior['omean_func_residual'].var(['draw','chain']).data
    cmean_func_residual_exp = meanfunc_posterior['cmean_func_residual'].mean(['draw','chain']).data

    ox = data_dictionary["ox"]
    cx = data_dictionary["cx"]
    odata = omean_func_residual_exp
    odata_var = omean_func_residual_var
    cdata = cmean_func_residual_exp
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
    data_dictionary,
    num_parameter_realisations,
    num_posterior_pred_realisations,
    rng_key
):
    posterior = data_dictionary["idata_residual_model_mean"].posterior
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
            nx, data_dictionary, posterior_param_realisation
        )
        bias_predictive_dist = generate_bias_predictive_dist_mean(
            nx, data_dictionary, posterior_param_realisation
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
    data_dictionary[
        "truth_posterior_predictive_realisations_dualprocess_mean"
    ] = truth_posterior_predictive_realisations
    data_dictionary[
        "bias_posterior_predictive_realisations_dualprocess_mean"
    ] = bias_posterior_predictive_realisations

    return(truth_posterior_predictive_realisations,
           bias_posterior_predictive_realisations)

# %% Generating posterior predictive realisations
generate_posterior_predictive_realisations_dualprocess_mean(
    data_dictionary["cx"],
    data_dictionary,
    100,
    1,
    rng_key
)


# %% Saving the dictionary:
with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary.pkl', 'wb') as f:
    pickle.dump(data_dictionary, f)
