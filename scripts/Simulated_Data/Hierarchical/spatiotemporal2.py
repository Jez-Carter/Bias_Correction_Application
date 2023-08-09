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
%matplotlib widget


# %% Defining Coordinates X1=Space, X2=Time

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
GP = GaussianProcess(1 * kernels.ExpSquared(10),X,diag=1e-5)
sample = GP.sample(rng_key)
ds_stacked["Sample"]=(['X'],  sample)
ds = ds_stacked.unstack()

# %%
ds.Sample.shape

# %%
v1, v2 = np.meshgrid(np.linspace(0,5,len(X1)), np.ones(len(X2)), indexing='ij')
v = v1*v2
m1, m2 = np.meshgrid(np.linspace(5,0,len(X1)), np.ones(len(X2)), indexing='ij')
m = m1*m2

# %%
v1, v2 = np.meshgrid(np.linspace(5,0,len(X1)), np.ones(len(X2)), indexing='ij')
v = v1*v2 

# %%
plt.imshow(v)

# %%
ds['Sample_Scaled'] = ds['Sample']*v+m

# %%
# ds['Output'].sel(X2=0).plot()
ds['Sample'].plot.pcolormesh(alpha=0.5)

# ds['Sample'].plot.surface(alpha=0.5)
# %%
# ds['Sample_Scaled'].plot.pcolormesh(alpha=0.5)
cm = 1/2.54  # centimeters in inches

fig = plt.figure()
ax = plt.axes(projection='3d')
ds['Sample_Scaled'].plot.surface(ax=ax,alpha=1,cmap='viridis',rcount=100,ccount=100)
ax.set_zlim([-20,20])

ax.plot(np.zeros(len(X1)), X1, np.linspace(5,0,len(X1)), color='k')

# %%
ds['Sample_Scaled'].plot.pcolormesh(alpha=0.5)
ax.plot(x, np.zeros_like(x), 1, color='k')
