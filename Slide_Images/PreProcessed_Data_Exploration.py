# %% Importing Packages

import numpy as np
import xarray as xr

# %% Loading data
base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"

scenario = np.load(
    inpath, allow_pickle="TRUE",fix_imports=True,
).item()

ds_aws_june_filtered = scenario['ds_aws_june_filtered']

# %%












# %%
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import arviz as az
import pandas as pd

from src.helper_functions import run_inference

import jax
import jax.numpy as jnp
rng_key = jax.random.PRNGKey(1)
jax.config.update("jax_enable_x64", True)

plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1

legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm

# %% Loading data
base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"

scenario = np.load(
    inpath, allow_pickle="TRUE",fix_imports=True,
).item()

# Station = scenario['ds_aws_june_filtered']['Station']
# X = scenario['ds_climate_coarse_june_stacked_landonly']['X']

# %% 

ds_climate_coarse_june_stacked = scenario['ds_climate_coarse_june_stacked']
ds_climate_coarse_june_stacked_landonly = scenario['ds_climate_coarse_june_stacked_landonly']


# %%
ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked,ds_climate_coarse_june_stacked_landonly])

# %%
ds_climate_coarse_june_stacked.unstack()

# %%
ds_climate_coarse_june_stacked_landonly['Temperature'].unstack().pcolormesh()

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

ax.scatter(
    ds_climate_coarse_june_stacked_landonly['Elevation'],
    ds_climate_coarse_june_stacked_landonly['Temperature'].mean('Time'),
    # s=ds_aws_stacked['June Temperature Records']*2,
    # c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
    alpha=0.2,
)
