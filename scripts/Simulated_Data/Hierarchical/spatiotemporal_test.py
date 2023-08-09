# %%

#Importing Packages
import numpy as np
import numpyro.distributions as dist
from numpy.random import RandomState
from tinygp import kernels, GaussianProcess
import jax
import xarray as xr
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.simulated_data_functions_hierarchical import generate_underlying_data_hierarchical
from src.simulated_data_functions_hierarchical import plot_underlying_data_mean_1d
from src.simulated_data_functions_hierarchical import plot_underlying_data_std_1d
from src.simulated_data_functions_hierarchical import plot_pdfs_1d
from src.simulated_data_functions_hierarchical import plot_underlying_data_2d

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

outpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %%
min_x,max_x = 0,100
X = jnp.arange(min_x,max_x,0.1)

# Scenario: Similar Lengthscales, Sparse Observations
scenario = {
    'jitter': 1e-5,
    'MEAN_T_variance': 1.0,
    'MEAN_T_lengthscale': 3.0,
    'MEAN_T_mean': 1.0,
    'LOGVAR_T_variance': 1.0,
    'LOGVAR_T_lengthscale': 3.0,
    'LOGVAR_T_mean': 1.0,
    'MEAN_B_variance': 1.0,
    'MEAN_B_lengthscale': 10.0,
    'MEAN_B_mean': -1.0,
    'LOGVAR_B_variance': 1.0,
    'LOGVAR_B_lengthscale': 10.0,
    'LOGVAR_B_mean': -1.0,
    'osamples':20,
    'csamples':100,
    'ox': RandomState(0).uniform(low=min_x, high=max_x, size=(40,)),
    'cx': np.linspace(min_x,max_x,80) ,
    'X': X,
    'MEAN_T_variance_prior': dist.Gamma(1.0,1.5),
    'MEAN_T_lengthscale_prior': dist.Gamma(3.0,0.2),
    'MEAN_T_mean_prior': dist.Normal(0.0, 2.0),
    'LOGVAR_T_variance_prior': dist.Gamma(1.0,1.5),
    'LOGVAR_T_lengthscale_prior': dist.Gamma(3.0,0.2),
    'LOGVAR_T_mean_prior': dist.Normal(0.0, 2.0),
    'MEAN_B_variance_prior': dist.Gamma(1.0,1.5),
    'MEAN_B_lengthscale_prior': dist.Gamma(3.0,0.2),
    'MEAN_B_mean_prior': dist.Normal(0.0, 2.0),
    'LOGVAR_B_variance_prior': dist.Gamma(1.0,1.5),
    'LOGVAR_B_lengthscale_prior': dist.Gamma(3.0,0.2),
    'LOGVAR_B_mean_prior': dist.Normal(0.0, 2.0),
    'nx':X[::5]
}

# %%
############# DATA GENERATION #############
def generate_underlying_data_hierarchical(scenario,rng_key):
    rng_key, rng_key_ = random.split(rng_key)

    GP_MEAN_T = GaussianProcess(
        scenario['MEAN_T_variance'] * kernels.ExpSquared(scenario['MEAN_T_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['MEAN_T_mean'])
    GP_LOGVAR_T = GaussianProcess(
        scenario['LOGVAR_T_variance'] * kernels.ExpSquared(scenario['LOGVAR_T_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['LOGVAR_T_mean'])
    GP_MEAN_B = GaussianProcess(
        scenario['MEAN_B_variance'] * kernels.ExpSquared(scenario['MEAN_B_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['MEAN_B_mean'])
    GP_LOGVAR_B = GaussianProcess(
        scenario['LOGVAR_B_variance'] * kernels.ExpSquared(scenario['LOGVAR_B_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['LOGVAR_B_mean'])
    
    scenario['MEAN_T'] = GP_MEAN_T.sample(rng_key)
    scenario['LOGVAR_T'] = GP_LOGVAR_T.sample(rng_key_)
    rng_key, rng_key_ = random.split(rng_key)
    scenario['MEAN_B'] = GP_MEAN_B.sample(rng_key)
    scenario['LOGVAR_B'] = GP_LOGVAR_B.sample(rng_key_)
    scenario['MEAN_C'] = scenario['MEAN_T']+scenario['MEAN_B']
    scenario['LOGVAR_C'] = scenario['LOGVAR_T']+scenario['LOGVAR_B']

    rng_key, rng_key_ = random.split(rng_key)
    scenario['MEAN_T_obs'] = GP_MEAN_T.condition(scenario['MEAN_T'],scenario['ox']).gp.sample(rng_key)
    scenario['LOGVAR_T_obs'] = GP_MEAN_T.condition(scenario['LOGVAR_T'],scenario['ox']).gp.sample(rng_key_)
    rng_key, rng_key_ = random.split(rng_key)
    N_T_obs = dist.Normal(scenario['MEAN_T_obs'],jnp.sqrt(jnp.exp(scenario['LOGVAR_T_obs'])))
    scenario['odata'] = N_T_obs.sample(rng_key,(scenario['osamples'],))

    rng_key, rng_key_ = random.split(rng_key)
    scenario['MEAN_T_climate'] = GP_MEAN_T.condition(scenario['MEAN_T'],scenario['cx']).gp.sample(rng_key)
    scenario['LOGVAR_T_climate'] = GP_LOGVAR_T.condition(scenario['LOGVAR_T'],scenario['cx']).gp.sample(rng_key_)
    rng_key, rng_key_ = random.split(rng_key)
    scenario['MEAN_B_climate'] = GP_MEAN_B.condition(scenario['MEAN_B'],scenario['cx']).gp.sample(rng_key)
    scenario['LOGVAR_B_climate'] = GP_LOGVAR_B.condition(scenario['LOGVAR_B'],scenario['cx']).gp.sample(rng_key_)
    scenario['MEAN_C_climate'] = scenario['MEAN_T_climate']+scenario['MEAN_B_climate']
    scenario['LOGVAR_C_climate'] = scenario['LOGVAR_T_climate']+scenario['LOGVAR_B_climate']

    rng_key, rng_key_ = random.split(rng_key)
    N_C_climate = dist.Normal(scenario['MEAN_C_climate'],jnp.sqrt(jnp.exp(scenario['LOGVAR_C_climate'])))
    scenario['cdata'] = N_C_climate.sample(rng_key,(scenario['csamples'],))                 

# %%
generate_underlying_data_hierarchical(scenario,rng_key)

# %%
############# DATA GENERATION #############
def generate_underlying_data_hierarchical(scenario,rng_key):
    rng_key, rng_key_ = random.split(rng_key)

    GP_MEAN_T = GaussianProcess(
        scenario['MEAN_T_variance'] * kernels.ExpSquared(scenario['MEAN_T_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['MEAN_T_mean'])
    GP_LOGVAR_T = GaussianProcess(
        scenario['LOGVAR_T_variance'] * kernels.ExpSquared(scenario['LOGVAR_T_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['LOGVAR_T_mean'])
    GP_MEAN_B = GaussianProcess(
        scenario['MEAN_B_variance'] * kernels.ExpSquared(scenario['MEAN_B_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['MEAN_B_mean'])
    GP_LOGVAR_B = GaussianProcess(
        scenario['LOGVAR_B_variance'] * kernels.ExpSquared(scenario['LOGVAR_B_lengthscale']),
        scenario['X'],diag=scenario['jitter'],mean=scenario['LOGVAR_B_mean'])
    
    scenario['MEAN_T'] = GP_MEAN_T.sample(rng_key)
    scenario['LOGVAR_T'] = GP_LOGVAR_T.sample(rng_key_)
    rng_key, rng_key_ = random.split(rng_key)
    scenario['MEAN_B'] = GP_MEAN_B.sample(rng_key)
    scenario['LOGVAR_B'] = GP_LOGVAR_B.sample(rng_key_)
    scenario['MEAN_C'] = scenario['MEAN_T']+scenario['MEAN_B']
    scenario['LOGVAR_C'] = scenario['LOGVAR_T']+scenario['LOGVAR_B']

    def mean_function_t(x):
        return GP_MEAN_T.condition(scenario['MEAN_T'],x).gp.mean
    def kernel_function_t(x):
        return GP_MEAN_T.condition(scenario['MEAN_T'],x).gp.mean


    rng_key, rng_key_ = random.split(rng_key)



    # scenario['MEAN_T_obs'] = GP_MEAN_T.condition(scenario['MEAN_T'],scenario['ox']).gp.sample(rng_key)
    # scenario['LOGVAR_T_obs'] = GP_MEAN_T.condition(scenario['LOGVAR_T'],scenario['ox']).gp.sample(rng_key_)
    # rng_key, rng_key_ = random.split(rng_key)
    # N_T_obs = dist.Normal(scenario['MEAN_T_obs'],jnp.sqrt(jnp.exp(scenario['LOGVAR_T_obs'])))
    # scenario['odata'] = N_T_obs.sample(rng_key,(scenario['osamples'],))

    rng_key, rng_key_ = random.split(rng_key)
    scenario['MEAN_T_climate'] = GP_MEAN_T.condition(scenario['MEAN_T'],scenario['cx']).gp.sample(rng_key)
    scenario['LOGVAR_T_climate'] = GP_LOGVAR_T.condition(scenario['LOGVAR_T'],scenario['cx']).gp.sample(rng_key_)
    rng_key, rng_key_ = random.split(rng_key)
    scenario['MEAN_B_climate'] = GP_MEAN_B.condition(scenario['MEAN_B'],scenario['cx']).gp.sample(rng_key)
    scenario['LOGVAR_B_climate'] = GP_LOGVAR_B.condition(scenario['LOGVAR_B'],scenario['cx']).gp.sample(rng_key_)
    scenario['MEAN_C_climate'] = scenario['MEAN_T_climate']+scenario['MEAN_B_climate']
    scenario['LOGVAR_C_climate'] = scenario['LOGVAR_T_climate']+scenario['LOGVAR_B_climate']

    def mean_function()
    # rng_key, rng_key_ = random.split(rng_key)
    # N_C_climate = dist.Normal(scenario['MEAN_C_climate'],jnp.sqrt(jnp.exp(scenario['LOGVAR_C_climate'])))
    # scenario['cdata'] = N_C_climate.sample(rng_key,(scenario['csamples'],))  

# %%

X1 = np.arange(0,105,2)
X2 = np.arange(0,105,2)

ds = xr.Dataset(
    coords=dict(
        X1=("X1", X1),
        X2=("X2", X2),
    ),
)

ds_stacked = ds.stack(X=('X1', 'X2'))
X = np.array(list(map(np.array, ds_stacked.X.data)))

# %%

GP = GaussianProcess(1 * kernels.ExpSquared(3),X1,diag=1e-5)
sample = GP.sample(rng_key)

def mean_function(x):
    mean = GP.condition(sample,jnp.array([x[0]])).gp.mean[0]
    return mean

GPST = GaussianProcess(1 * kernels.ExpSquared(10),X,diag=1e-5,mean=mean_function)

# %%
def variance_function(x):
    return(0.05*jnp.abs(x[0])+0.1)

# %%
X.shape

# %%
variance_function(X[0])

# %%
kernels.ExpSquared(5)(X1,X1)

# %%
class KernelVariance(kernels.Kernel):
    def __init__(self,func):
        self.weight = func
    def evaluate(self, X1, X2):
        return jnp.atleast_1d(self.weight(X1,X2))

# %%
KernelVariance(variance_function)(X1,X1)

# %%
from tinygp.kernels.distance import Distance, L1Distance, L2Distance

class SpectralMixture(tinygp.kernels.Kernel):
    def __init__(self, weight, scale, freq):
        self.distance = L2Distance()
        self.weight = jnp.atleast_1d(weight)
        self.scale = jnp.atleast_1d(scale)

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]
        return jnp.sum(
            self.weight
            * jnp.prod(
                jnp.exp(-2 * jnp.pi**2 * tau**2 / self.scale**2)
                * jnp.cos(2 * jnp.pi * self.freq * tau),
                axis=0,
            )
        )
    
    def evaluate(self, X1, X2):
        r2 = self.distance.squared_distance(X1, X2) / jnp.square(self.scale)
        return jnp.exp(-0.5 * r2)
# %%
@dataclass
class ExpSquared(Stationary):
    r"""The exponential squared or radial basis function kernel

    .. math::

        k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp(-r^2 / 2)

    where, by default,

    .. math::

        r^2 = ||(\mathbf{x}_i - \mathbf{x}_j) / \ell||_2^2

    Args:
        scale: The parameter :math:`\ell`.
    """

    distance: Distance = L2Distance()

[docs]    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r2 = self.distance.squared_distance(X1, X2) / jnp.square(self.scale)
        return jnp.exp(-0.5 * r2)

# %%
GPST = GaussianProcess(variance_function * kernels.ExpSquared(10),X,diag=1e-5,mean=mean_function)



# %%
ds_stacked["Mean"]=(['X'],  GPST.sample(rng_key))
ds_stacked["Sample"]=(['X1'],  sample)
ds = ds_stacked.unstack()

# %%
ds['Mean'].isel(X2=10).plot()
ds['Sample'].plot()

# %%

GP = GaussianProcess(1 * kernels.ExpSquared(3),X1,diag=1e-5)
sample = GP.sample(rng_key)

def mean_function(x):
    mean = GP.condition(sample,jnp.array([x[0]])).gp.mean[0]
    return mean

def variance_function(x):
    variance = GP.condition(sample,jnp.array([x[0]])).gp.mean[0]
    return mean

GPST = GaussianProcess(1 * kernels.ExpSquared(10),X,diag=1e-5,mean=mean_function)


# %%
GPST.std
# %%
meanfunc([1,0])

# %%
GP.condition(sample,jnp.array([x])).gp.mean

# %%
jnp.array([X[0]])

# %%
X[0]


# %%
def meanfunc(x):
    return(0.05*x[0]+0.1)

mean_array = np.linspace(0,5,X.shape[0])
def meanarray(x):

    return(0.05*x[0]+0.1)


# %%

X.argwhere(X=[0,0])

# %%
meanfunc(X[0])

# %%
mean_array = np.linspace(0,5,X.shape[0])

# %%
mean_array.shape

# %%
GP = GaussianProcess(1 * kernels.ExpSquared(3),X,diag=1e-5,mean=meanfunc)

# %%
sample = GP.sample(rng_key)

# %%

output = GP.condition(sample,X).gp.mean

# %%
ds_stacked["Output"]=(['X'],  output)

ds = ds_stacked.unstack()

# %%
ds['Output'].sel(X2=0).plot()


# %%

GP = GaussianProcess(1 * kernels.ExpSquared(10),X,diag=1e-5,mean=mean_array)


# %%
Y = GP.sample(rng_key)

# %%
ds_stacked["Y"]=(['X'],  Y)

ds = ds_stacked.unstack()

# %%
ds['Y'].sel(X2=0).plot()

# %%
# #Creating Underlying Process Data 

# X1 = np.arange(0,105,2)
# X2 = np.arange(0,105,2)

# ds = xr.Dataset(
#     coords=dict(
#         X1=("X1", X1),
#         X2=("X2", X2),
#     ),
# )

# ds_stacked = ds.stack(X=('X1', 'X2'))
# X = np.array(list(map(np.array, ds_stacked.X.data)))

# #Truth
GP = GaussianProcess(1 * kernels.ExpSquared(10),X,diag=1e-5,mean=1.0)
Y = GP.sample(rng_key)
# %%
u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)
    mvn_dist = dist.MultivariateNormal(u1g2,k1g2)


# %%
mean_array = np.linspace(0,5,X.shape[0])
kernel_matrix = kernels.ExpSquared(3)(X,X)
mvn_dist = dist.MultivariateNormal(mean_array,kernel_matrix)

# %%
data = mvn_dist.sample(rng_key)

# %%
ds_stacked["Data"]=(['X'],  data)

ds = ds_stacked.unstack()
# %%

ds['Data'].sel(X2=0).plot()
