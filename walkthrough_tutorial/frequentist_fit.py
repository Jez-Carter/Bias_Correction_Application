# %% Importing necessary libraries
import jax
import jax.numpy as jnp
import jaxopt
import pickle 
from numpyro import distributions as dist
from tinygp import kernels, GaussianProcess
from tinygp.kernels.distance import L2Distance

rng_key = jax.random.PRNGKey(1)
jax.config.update("jax_enable_x64", True)

# %% Loading the data
with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary.pkl', 'rb') as f:
    data_dictionary = pickle.load(f)

# %% Defining the frequentist model and helper functions

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

def residual_model_mean_negloglikelihood(theta_init_dict,x_dictionary,y_dictionary):
    """
    Example model where the climate data is generated from 2 GPs,
    one of which also generates the observations and one of
    which generates bias in the climate model.
    """
    omean_func_residual_exp = y_dictionary['omean_func_residual_exp']
    omean_func_residual_var = y_dictionary['omean_func_residual_var']
    cmean_func_residual_exp = y_dictionary['cmean_func_residual_exp']
    
    kern_var = jnp.exp(theta_init_dict['logomean_func_residual_kvinit'])
    lengthscale = jnp.exp(theta_init_dict['logomean_func_residual_klinit'])
    kernel = kern_var * kernels.Matern32(lengthscale,L2Distance())
    noise = jnp.exp(theta_init_dict['logomean_func_residual_ninit'])
    var_obs = omean_func_residual_var

    bkern_var = jnp.exp(theta_init_dict['logbmean_func_residual_kvinit'])
    blengthscale = jnp.exp(theta_init_dict['logbmean_func_residual_klinit'])
    bkernel = bkern_var * kernels.Matern32(blengthscale,L2Distance())
    bnoise = jnp.exp(theta_init_dict['logbmean_func_residual_ninit'])

    ckernel = kernel + bkernel
    cnoise = noise + bnoise
    cgp = GaussianProcess(ckernel, x_dictionary["cx"], diag=cnoise, mean=0)

    obs_conditional_climate_dist = generate_obs_conditional_climate_dist(
        x_dictionary["ox"],
        x_dictionary["cx"],
        cmean_func_residual_exp,
        ckernel,
        cnoise,
        kernel,
        var_obs+noise
    )

    negloglikelihood_cgp = -cgp.log_probability(cmean_func_residual_exp)
    negloglikelihood_obs_conditional_climate_dist = -obs_conditional_climate_dist.log_prob(omean_func_residual_exp)
    negloglikelihood = negloglikelihood_cgp + negloglikelihood_obs_conditional_climate_dist

    # jax.debug.print("negloglikelihood: {}", negloglikelihood,
    #                 "\n negloglikelihood_components: ", negloglikelihood_cgp, negloglikelihood_obs_conditional_climate_dist,
    #                 "\n okern_var: ", kern_var, "olengthscale: ", lengthscale, "onoise: ", noise,
    #                 "\n bkern_var: ", bkern_var, "blengthscale: ", blengthscale, "bnoise: ", bnoise)

    # jax.debug.print("intermediate value: {}", -cgp.log_probability(cmean_func_residual_exp) - obs_conditional_climate_dist.log_prob(omean_func_residual_exp))
    return negloglikelihood

# # %% Simplified version of the model

# def residual_model_mean_negloglikelihood_simplified(theta_init_dict,x_dictionary,y_dictionary):
#     """
#     Example model where the climate data is generated from 2 GPs,
#     one of which also generates the observations and one of
#     which generates bias in the climate model.
#     """
#     omean_func_residual_exp = y_dictionary['omean_func_residual_exp']
#     omean_func_residual_var = y_dictionary['omean_func_residual_var']
    
#     kern_var = jnp.exp(theta_init_dict['logomean_func_residual_kvinit'])
#     lengthscale = jnp.exp(theta_init_dict['logomean_func_residual_klinit'])
#     kernel = kern_var * kernels.Matern32(lengthscale,L2Distance())
#     noise = jnp.exp(theta_init_dict['logomean_func_residual_ninit'])
#     var_obs = omean_func_residual_var

#     gp = GaussianProcess(kernel, x_dictionary["ox"], diag=var_obs+noise, mean=0)

#     negloglikelihood = -gp.log_probability(omean_func_residual_exp)
#     # jax.debug.print("negloglikelihood: {}", negloglikelihood)#,
#     # jax.debug.print("kern_var: {}", kern_var)
#     # jax.debug.print("lengthscale: {}", lengthscale)
#     # jax.debug.print("noise: {}", noise)
#                     # "kern_var: {}", kern_var,
#                     # "lengthscale: {}", lengthscale,
#                     # "noise: {}", noise)
#     # jax.debug.print("negloglikelihood: {}", -gp.log_probability(omean_func_residual_exp))
#     return negloglikelihood


# %% Defining the observations and initial values for the parameters

data_dictionary['omean_func_residual_kvinit'] = 5.0
data_dictionary['omean_func_residual_klinit'] = 1.0
data_dictionary['omean_func_residual_ninit'] = 1.0
data_dictionary['bmean_func_residual_kvinit'] = 10.0
data_dictionary['bmean_func_residual_klinit'] = 5.0
data_dictionary['bmean_func_residual_ninit'] = 10.0

theta_init_keys = ['omean_func_residual_kvinit','omean_func_residual_klinit','omean_func_residual_ninit',
                    'bmean_func_residual_kvinit','bmean_func_residual_klinit','bmean_func_residual_ninit']
theta_init_dict = {key: data_dictionary[key] for key in theta_init_keys}
logtheta_init_dict = {f'log{key}':jnp.log(data_dictionary[key]) for key in theta_init_keys}

meanfunc_posterior = data_dictionary['meanfunc_posterior']
data_dictionary['omean_func_residual_exp'] = meanfunc_posterior['omean_func_residual'].mean(['draw','chain']).data
data_dictionary['omean_func_residual_var'] = meanfunc_posterior['omean_func_residual'].var(['draw','chain']).data
data_dictionary['cmean_func_residual_exp'] = meanfunc_posterior['cmean_func_residual'].mean(['draw','chain']).data

x_dictionary = {key:data_dictionary[key] for key in ['ox','cx']}
y_dictionary = {key:data_dictionary[key] for key in ['omean_func_residual_exp','omean_func_residual_var','cmean_func_residual_exp']}

# # %% Simplified parameters
# logtheta_init_dict_simplified = {f'log{key}':jnp.log(data_dictionary[key]) for key in theta_init_keys[:3]}
# y_dictionary_simplified = {key:data_dictionary[key] for key in ['omean_func_residual_exp','omean_func_residual_var']}

# %% Testing the model
obj = jax.jit(jax.value_and_grad(residual_model_mean_negloglikelihood))
print(f"Initial negative log likelihood: {obj(logtheta_init_dict, x_dictionary,y_dictionary)[0]}")
print(
    f"Gradient of the negative log likelihood, wrt the parameters:\n{obj(logtheta_init_dict, x_dictionary,y_dictionary)[1]}"
)

# # %% Testing the simplified model
# obj = jax.jit(jax.value_and_grad(residual_model_mean_negloglikelihood_simplified))
# print(f"Initial negative log likelihood: {obj(logtheta_init_dict_simplified, x_dictionary,y_dictionary_simplified)[0]}")
# print(
#     f"Gradient of the negative log likelihood, wrt the parameters:\n{obj(logtheta_init_dict_simplified, x_dictionary,y_dictionary_simplified)[1]}"
# )

# %% Optimizing the model

solver = jaxopt.ScipyMinimize(fun=residual_model_mean_negloglikelihood)
soln = solver.run(logtheta_init_dict,x_dictionary,y_dictionary)
print(f"Final negative log likelihood: {soln.state.fun_val}")
print(f"Optimized parameters: {soln.params}")

# %%
solver.params

# %% Transforming the optimized parameters
transformed_soln_dict = {key[3:]:jnp.exp(val) for key,val in zip(soln.params.keys(),soln.params.values())}
transformed_soln_dict


# %% Optimizing the simplified model
# from jaxopt import GradientDescent

solver = jaxopt.ScipyMinimize(fun=residual_model_mean_negloglikelihood_simplified,trace=True)
soln = solver.run(logtheta_init_dict_simplified,x_dictionary,y_dictionary_simplified)
print(f"Final negative log likelihood: {soln.state.fun_val}")
print(f"Optimized parameters: {soln.params}")


# %%
data_dictionary['omean_func_residual_var'].shape

# %%
data_dictionary['omean_func_residual_exp']


# %%
solver.fun