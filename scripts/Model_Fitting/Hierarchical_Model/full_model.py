# %% Importing Packages
import numpy as np
import arviz as az
from src.model_fitting_Functions import generate_posterior_hierarchical

import jax
rng_key = jax.random.PRNGKey(1)
jax.config.update("jax_enable_x64", True)


# %% Loading data
inpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

# %% Sanity Check
print('Data Shapes: \n',
      f'ox.shape:{scenario["ox"].shape} \n',
      f'oele_scaled.shape:{scenario["oele"].shape} \n',
      f'olat_scaled.shape:{scenario["olat"].shape} \n',
      f'odata.shape:{scenario["odata"].shape} \n',
      f'cx.shape:{scenario["cx"].shape} \n',
      f'cele_scaled.shape:{scenario["cele"].shape} \n',
      f'clat_scaled.shape:{scenario["clat"].shape} \n',
      f'cdata.shape:{scenario["cdata"].shape} \n'
      )

print('Data Values: \n',
      f'ox min,max:{scenario["ox"].min(),scenario["ox"].max()} \n',
      f'oele_scaled min,max:{scenario["oele"].min(),scenario["oele"].max()} \n',
      f'olat_scaled min,max:{scenario["olat"].min(),scenario["olat"].max()} \n',
      f'odata min,max:{scenario["odata"].min(),scenario["odata"].max()} \n',
      f'cx min,max:{scenario["cx"].min(),scenario["cx"].max()} \n',
      f'cele_scaled min,max:{scenario["cele"].min(),scenario["cele"].max()} \n',
      f'clat_scaled min,max:{scenario["clat"].min(),scenario["clat"].max()} \n',
      f'cdata min,max:{scenario["cdata"].min(),scenario["cdata"].max()} \n'
      )

# %% Fitting the model
generate_posterior_hierarchical(scenario,rng_key,1000,2000,1)

# %% Examining posterior summary statistics
az.summary(scenario['mcmc'].posterior,hdi_prob=0.95)[100:160]

# %% Saving the output
outpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real_posterior.npy"
np.save(outpath, scenario)

# %%
