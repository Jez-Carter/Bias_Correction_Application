# %% Importing Packages
import numpy as np
import jax
from jax import random
import matplotlib.pyplot as plt

plt.rcParams["lines.markersize"] = 3
plt.rcParams["lines.linewidth"] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.spatial.dualprocess_prediction_functions import (
    generate_posterior_predictive_realisations_dualprocess_mean,
    generate_posterior_predictive_realisations_dualprocess_logvar,
    generate_posterior_predictive_realisations_dualprocess_mean_station_locations,
    generate_posterior_predictive_realisations_dualprocess_logvar_station_locations
)

inpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/"

base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

# %% Predictions at the station locations
generate_posterior_predictive_realisations_dualprocess_mean_station_locations(
    scenario,
    1,
    10,
    rng_key
)

# %%
from tinygp import kernels
from tinygp.kernels.distance import L2Distance
import jax.numpy as jnp
from src.helper_functions import diagonal_noise
import numpyro.distributions as dist

# %%
nx = scenario['ox']
num_parameter_realisations = 1
num_posterior_pred_realisations = 1
posterior = scenario["mcmc_dualprocess_mean_residual"].posterior
iteration = np.random.randint(posterior.draw.shape, size=num_parameter_realisations)

i = iteration[0]
posterior_param_realisation = {
        "iteration": i,
        "kern_var_realisation": posterior["kern_var"].data[0, :][i],
        "lengthscale_realisation": posterior["lengthscale"].data[0, :][i],
        "noise_realisation": posterior["noise"].data[0, :][i],
        "bkern_var_realisation": posterior["bkern_var"].data[0, :][i],
        "blengthscale_realisation": posterior["blengthscale"].data[0, :][i],
        "bnoise_realisation": posterior["bnoise"].data[0, :][i],
        }

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

k22i = jnp.linalg.inv(k22)

u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
k1g2 = k11 - jnp.matmul(jnp.matmul(k12, k22i), k21)
mvn = dist.MultivariateNormal(u1g2, k1g2)

# %%
mvn.sample(rng_key, sample_shape=(num_posterior_pred_realisations,))

# %%
print(f'''
kernelo: {kernelo}
kernelb: {kernelb}
noise_realisation: {noise_realisation}
odata_var mean: {odata_var.mean()}
noise shape: {noise.shape}
cnoise: {cnoise}
      ''')

# %%
print(f'''
y2 shape: {y2.shape}
u1 shape: {u1.shape}
u2 shape: {u2.shape}

      ''')
