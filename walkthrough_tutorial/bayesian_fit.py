# %% Importing necessary libraries
import jax
import jax.numpy as jnp
import jaxopt
import pickle 
from numpyro import distributions as dist
from tinygp import kernels, GaussianProcess
from tinygp.kernels.distance import L2Distance
from tinygp import kernels, GaussianProcess
from tinygp.kernels.distance import L2Distance
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import timeit
import arviz as az
import pickle 

rng_key = jax.random.PRNGKey(1)
jax.config.update("jax_enable_x64", True)

# %% Loading the data
with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary.pkl', 'rb') as f:
    data_dictionary = pickle.load(f)

# %% Setting priors
lengthscale_max = 20

data_dictionary['omean_func_residual_kvprior'] = dist.Uniform(0.1,100.0)
data_dictionary['omean_func_residual_klprior'] = dist.Uniform(1,lengthscale_max)
data_dictionary['omean_func_residual_nprior'] = dist.Uniform(0.1,20.0)

data_dictionary['bmean_func_residual_kvprior'] = dist.Uniform(0.1,100.0)
data_dictionary['bmean_func_residual_klprior'] = dist.Uniform(1,lengthscale_max)
data_dictionary['bmean_func_residual_nprior'] = dist.Uniform(0.1,20.0)

# %% Defining function for running inference
def run_inference(
    model, rng_key, num_warmup, num_samples, num_chains, *args, **kwargs
):
    """
    Helper function for doing MCMC inference
    Args:
        model (python function): function that follows numpyros syntax
        rng_key (np array): PRNGKey for reproducible results
        num_warmup (int): Number of MCMC steps for warmup
        num_samples (int): Number of MCMC samples to take of parameters after warmup
        data (jax device array): data in shape [#days,#months,#sites]
        distance_matrix_values(jax device array): matrix of distances between sites, shape [#sites,#sites]
    Returns:
        MCMC numpyro instance (class object): An MCMC class object with functions such as .get_samples() and .run()
    """
    starttime = timeit.default_timer()

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
    )

    mcmc.run(rng_key, *args, **kwargs)

    mcmc.print_summary()
    print("Time Taken:", timeit.default_timer() - starttime)
    return mcmc


# %% Defining the model

def diagonal_noise(coord, noise):
    return jnp.diag(jnp.full(coord.shape[0], noise))

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


def residual_model_mean(data_dictionary):
    """
    Example model where the climate data is generated from 2 GPs,
    one of which also generates the observations and one of
    which generates bias in the climate model.
    """
    meanfunc_posterior = data_dictionary['meanfunc_posterior']
    omean_func_residual_exp = meanfunc_posterior['omean_func_residual'].mean(['draw','chain']).data
    omean_func_residual_var = meanfunc_posterior['omean_func_residual'].var(['draw','chain']).data
    cmean_func_residual_exp = meanfunc_posterior['cmean_func_residual'].mean(['draw','chain']).data
    
    kern_var = numpyro.sample("kern_var", data_dictionary['omean_func_residual_kvprior'])
    lengthscale = numpyro.sample("lengthscale", data_dictionary['omean_func_residual_klprior'])
    kernel = kern_var * kernels.Matern32(lengthscale,L2Distance())
    noise = numpyro.sample("noise", data_dictionary['omean_func_residual_nprior'])
    var_obs = omean_func_residual_var
    
    bkern_var = numpyro.sample("bkern_var", data_dictionary['bmean_func_residual_kvprior'])
    blengthscale = numpyro.sample("blengthscale", data_dictionary['bmean_func_residual_klprior'])
    bkernel = bkern_var * kernels.Matern32(blengthscale,L2Distance())
    bnoise = numpyro.sample("bnoise", data_dictionary['bmean_func_residual_nprior'])

    ckernel = kernel + bkernel
    cnoise = noise + bnoise 
    cgp = GaussianProcess(ckernel, data_dictionary["cx"], diag=cnoise, mean=0)
    numpyro.sample("climate_temperature",
                   cgp.numpyro_dist(),
                   obs=cmean_func_residual_exp)

    obs_conditional_climate_dist = generate_obs_conditional_climate_dist(
        data_dictionary["ox"],
        data_dictionary["cx"],
        cmean_func_residual_exp,
        ckernel,
        cnoise,
        kernel,
        var_obs+noise
    )
    numpyro.sample(
        "obs_temperature",
        obs_conditional_climate_dist,
        obs=omean_func_residual_exp
    )

def generate_posterior_residual_model_mean(data_dictionary,
                                    rng_key,
                                    num_warmup,
                                    num_samples,
                                    num_chains):
    mcmc_residual_model_mean = run_inference(
        residual_model_mean,
        rng_key,
        num_warmup,
        num_samples,
        num_chains,
        data_dictionary,
    )
    idata_residual_model_mean = az.from_numpyro(mcmc_residual_model_mean)
    data_dictionary["idata_residual_model_mean"] = idata_residual_model_mean


# %% Running the inference
generate_posterior_residual_model_mean(data_dictionary,
                                rng_key,
                                1000,
                                1000,
                                4)

# %% Saving the dictionary:
with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary.pkl', 'wb') as f:
    pickle.dump(data_dictionary, f)
