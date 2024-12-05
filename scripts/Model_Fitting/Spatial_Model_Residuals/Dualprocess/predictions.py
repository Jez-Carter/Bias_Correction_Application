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

# %% Loading data
base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()


# %% Generating posterior predictive realisations
generate_posterior_predictive_realisations_dualprocess_mean(
    scenario["cx"],
    scenario,
    100,
    10,
    rng_key
)

# %%

generate_posterior_predictive_realisations_dualprocess_logvar(
    scenario["cx"],
    scenario,
    100,
    1,
    rng_key
)

# %% Predictions at the station locations
generate_posterior_predictive_realisations_dualprocess_mean_station_locations(
    scenario,
    100,
    1,
    rng_key
)

# %%

generate_posterior_predictive_realisations_dualprocess_logvar_station_locations(
    scenario,
    100,
    1,
    rng_key
)

# %%
scenario[
        "truth_posterior_predictive_realisations_dualprocess_mean"
    ]

# %%
scenario['truth_posterior_predictive_realisations_dualprocess_mean_station_locations']

# %%
scenario['truth_posterior_predictive_realisations_dualprocess_mean_station_locations'].shape

# %%
scenario['truth_posterior_predictive_realisations_dualprocess_logvar_station_locations']

# %%
scenario['truth_posterior_predictive_realisations_dualprocess_logvar_station_locations'].shape


# %% Saving Output
scenario_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'
np.save(scenario_outpath, scenario)
# %%
