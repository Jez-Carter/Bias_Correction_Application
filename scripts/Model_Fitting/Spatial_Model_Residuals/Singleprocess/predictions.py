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

from src.spatial.singleprocess_model_functions import generate_posterior_predictive_realisations_singleprocess_mean
from src.spatial.singleprocess_model_functions import generate_posterior_predictive_realisations_singleprocess_logvar

# %% Loading data
base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

# %% Generating posterior predictive realisations
generate_posterior_predictive_realisations_singleprocess_mean(
    scenario["cx"], scenario, 1, 100,rng_key
)
generate_posterior_predictive_realisations_singleprocess_logvar(
    scenario["cx"], scenario, 1, 100,rng_key
)

# %% Saving Output
scenario_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'
np.save(scenario_outpath, scenario)

# %%
