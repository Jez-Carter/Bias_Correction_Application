# %% Importing Packages
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import arviz as az

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
    inpath, allow_pickle="TRUE"
).item()

Station = scenario['ds_aws_june_filtered']['Station']
X = scenario['ds_climate_coarse_june_stacked_landonly']['X']

# %%
print('Useful Metrics for Priors: \n',
      f"""mean odata:
      min={np.nanmin(np.nanmean(scenario['odata'],axis=0)):.1f},
      mean={np.nanmean(np.nanmean(scenario['odata'],axis=0)):.1f},
      max={np.nanmax(np.nanmean(scenario['odata'],axis=0)):.1f},
      var={np.nanvar(np.nanmean(scenario['odata'],axis=0)):.1f},
      \n""",
      f"""logvar odata:
      min={np.nanmin(np.log(np.nanvar(scenario['odata'],axis=0))):.1f},
      mean={np.nanmean(np.log(np.nanvar(scenario['odata'],axis=0))):.1f},
      max={np.nanmax(np.log(np.nanvar(scenario['odata'],axis=0))):.1f},
      var={np.nanvar(np.log(np.nanvar(scenario['odata'],axis=0))):.1f},
      \n""",
      f"""mean cdata:
      min={np.nanmin(np.nanmean(scenario['cdata'],axis=0)):.1f},
      mean={np.nanmean(np.nanmean(scenario['cdata'],axis=0)):.1f},
      max={np.nanmax(np.nanmean(scenario['cdata'],axis=0)):.1f},
      var={np.nanvar(np.nanmean(scenario['cdata'],axis=0)):.1f},
      \n""",
      f"""logvar cdata:
      min={np.nanmin(np.log(np.nanvar(scenario['cdata'],axis=0))):.1f},
      mean={np.nanmean(np.log(np.nanvar(scenario['cdata'],axis=0))):.1f},
      max={np.nanmax(np.log(np.nanvar(scenario['cdata'],axis=0))):.1f},
      var={np.nanvar(np.log(np.nanvar(scenario['cdata'],axis=0))):.1f},
      \n""",
)

# %% Setting Priors
scenario.update({
    "meanfunc_b0_prior": dist.Normal(-33.7, 10.0),
    "meanfunc_b1_prior": dist.Normal(-10.0, 5.0),
    "meanfunc_b2_prior": dist.Normal(0.0, 5.0),
    "meanfunc_noise_prior": dist.Uniform(1e-2, 10.0),
    "logvarfunc_b0_prior": dist.Normal(2, 1.0),
    "logvarfunc_noise_prior": dist.Uniform(0, 2.0),

    # "MEAN_C_mean_b0_prior": dist.Normal(-18.0, 5.0),
    # "MEAN_C_mean_b1_prior": dist.Normal(0.0, 5.0),
    # "MEAN_C_mean_b2_prior": dist.Normal(0.0, 5.0),
    # "MEAN_C_noise_prior": dist.Uniform(1e-2, 10.0),
    # "LOGVAR_C_mean_b0_prior": dist.Normal(0, 1.0),
    # "LOGVAR_C_noise_prior": dist.Uniform(0, 1.0),
})

# %% Defining Model for Observation Data
def mean_model_obs(scenario):
    mean_b0 = numpyro.sample("mean_b0",scenario['meanfunc_b0_prior'])
    mean_b1 = numpyro.sample("mean_b1",scenario['meanfunc_b1_prior'])
    mean_b2 = numpyro.sample("mean_b2",scenario['meanfunc_b2_prior'])
    mean_noise = numpyro.sample("mean_noise",scenario['meanfunc_noise_prior'])

    mean_func = mean_b0 + mean_b1*scenario['oele_scaled'] + mean_b2*scenario['olat_scaled']
    mean = numpyro.sample("mean",dist.Normal(mean_func, mean_noise))
    
    logvar_b0 = numpyro.sample("logvar_b0",scenario['logvarfunc_b0_prior'])
    logvar_noise = numpyro.sample("logvar_noise",scenario['logvarfunc_noise_prior'])

    logvar_func = logvar_b0 * jnp.ones(scenario['ox'].shape[0])
    logvar = numpyro.sample("logvar",dist.Normal(logvar_func, logvar_noise))
    var = jnp.exp(logvar)

    obs_mask = (jnp.isnan(scenario['odata'])==False)
    numpyro.sample("Temperature", dist.Normal(mean, jnp.sqrt(var)).mask(obs_mask), obs=scenario["odata"])

def mean_model_climate(scenario):
    mean_b0 = numpyro.sample("mean_b0",scenario['meanfunc_b0_prior'])
    mean_b1 = numpyro.sample("mean_b1",scenario['meanfunc_b1_prior'])
    mean_b2 = numpyro.sample("mean_b2",scenario['meanfunc_b2_prior'])
    mean_noise = numpyro.sample("mean_noise",scenario['meanfunc_noise_prior'])

    mean_func = mean_b0 + mean_b1*scenario['cele_scaled'] + mean_b2*scenario['clat_scaled']
    mean = numpyro.sample("mean",dist.Normal(mean_func, mean_noise))

    logvar_b0 = numpyro.sample("logvar_b0",scenario['logvarfunc_b0_prior'])
    logvar_noise = numpyro.sample("logvar_noise",scenario['logvarfunc_noise_prior'])

    logvar_func = logvar_b0 * jnp.ones(scenario['cx'].shape[0])
    logvar = numpyro.sample("logvar",dist.Normal(logvar_func, logvar_noise))
    var = jnp.exp(logvar)

    numpyro.sample("Temperature", dist.Normal(mean, jnp.sqrt(var)), obs=scenario["cdata"])

# %% Running Inference
mcmc = run_inference(mean_model_obs, rng_key, 1000, 2000,1, scenario)
idata = az.from_numpyro(mcmc,
                coords={
                "Station": Station,
    },
                dims={"logvar": ["Station"],
                      "mean": ["Station"]})
idata.posterior['Station'] = Station
posterior = idata.posterior

mcmc_climate = run_inference(mean_model_climate, rng_key, 1000, 2000,1, scenario)
idata_climate = az.from_numpyro(mcmc_climate,
                coords={
                "X": X,
    },
                dims={"logvar": ["X"],
                      "mean": ["X"]})

posterior_climate = idata_climate.posterior
posterior_climate['X'] = X.reset_index('X')


# %% Assigning Some Coords and Computing Mean Function Prediction
# posterior = idata.posterior
# posterior = posterior.assign_coords(oele_scaled=("Station", scenario['oele_scaled']))
# posterior = posterior.assign_coords(olat_scaled=("Station", scenario['olat_scaled']))

posterior['meanfunc_prediction'] = (posterior['mean_b0']
                                + posterior['mean_b1']*posterior['Elevation_Scaled']
                                + posterior['mean_b2']*posterior['Latitude_Scaled'])
posterior['logvarfunc_prediction'] = (posterior['logvar_b0'])
                                # + posterior['logvar_b1']*posterior['oele_scaled']
                                # + posterior['logvar_b2']*posterior['olat_scaled'])

posterior['meanfunc_residual'] = posterior['mean'] - posterior['meanfunc_prediction']
posterior['logvarfunc_residual'] = posterior['logvar'] - posterior['logvarfunc_prediction']


# posterior_climate = idata_climate.posterior

# posterior_climate = posterior_climate.assign_coords(cele_scaled=("X", scenario['cele_scaled']))
# posterior_climate = posterior_climate.assign_coords(clat_scaled=("X", scenario['clat_scaled']))

posterior_climate['meanfunc_prediction'] = (posterior_climate['mean_b0']
                                + posterior_climate['mean_b1']*posterior_climate['Elevation_Scaled']
                                + posterior_climate['mean_b2']*posterior_climate['Latitude_Scaled'])
posterior_climate['logvarfunc_prediction'] = (posterior_climate['logvar_b0'])

posterior_climate['meanfunc_prediction_unbiased'] = (posterior['mean_b0']
                                + posterior['mean_b1']*posterior_climate['Elevation_Scaled']
                                + posterior['mean_b2']*posterior_climate['Latitude_Scaled'])
posterior_climate['logvarfunc_prediction_unbiased'] = (posterior['logvar_b0'])
                                # + posterior_climate['logvar_b1']*posterior_climate['cele_scaled']
                                # + posterior_climate['logvar_b2']*posterior_climate['clat_scaled'])

posterior_climate['meanfunc_residual'] = posterior_climate['mean'] - posterior_climate['meanfunc_prediction']
posterior_climate['logvarfunc_residual'] = posterior_climate['logvar'] - posterior_climate['logvarfunc_prediction']

# %% Saving Output
scenario['Mean_Function_Posterior'] = posterior
scenario['Mean_Function_Posterior_Climate'] = posterior_climate

scenario['exp_meanfunc_residual_obs'] = posterior['meanfunc_residual'].mean(['chain','draw']).data
scenario['var_meanfunc_residual_obs'] = posterior['meanfunc_residual'].var(['chain','draw']).data
scenario['exp_logvarfunc_residual_obs'] = posterior['logvarfunc_residual'].mean(['chain','draw']).data
scenario['var_logvarfunc_residual_obs'] = posterior['logvarfunc_residual'].var(['chain','draw']).data

scenario['exp_meanfunc_residual_climate'] = posterior_climate['meanfunc_residual'].mean(['chain','draw']).data
scenario['var_meanfunc_residual_climate'] = posterior_climate['meanfunc_residual'].var(['chain','draw']).data
scenario['exp_logvarfunc_residual_climate'] = posterior_climate['logvarfunc_residual'].mean(['chain','draw']).data
scenario['var_logvarfunc_residual_climate'] = posterior_climate['logvarfunc_residual'].var(['chain','draw']).data

scenario_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'
np.save(scenario_outpath, scenario)

# %%
list(scenario.keys()
)



















# %%
posterior['meanfunc_residual'] = posterior['mean'] - posterior['meanfunc_prediction']
posterior['logvarfunc_residual'] = posterior['logvar'] - posterior['logvarfunc_prediction']

posterior_climate['meanfunc_residual'] = posterior['mean'] - posterior['meanfunc_prediction']
posterior_climate['logvarfunc_residual'] = posterior['logvar'] - posterior['logvarfunc_prediction']



# %%

cmeans = np.nanmean(scenario['cdata_scaled'],axis=0)
cvars = np.nanvar(scenario['cdata_scaled'],axis=0)
clogvars = np.log(cvars)

# %% Computing Mean Function Prediction Expectation and Residual Compared with Empirical Value
posterior['mean_prediction_exp'] = posterior['mean_prediction'].mean(['chain','draw'])
posterior['logvar_prediction_exp'] = posterior['logvar_prediction'].mean(['chain','draw'])
posterior['mean_prediction_exp_residual'] = means-posterior['mean_prediction_exp']
posterior['logvar_prediction_exp_residual'] = logvars-posterior['logvar_prediction_exp']

posterior_climate['cmean_prediction_exp'] = posterior_climate['cmean_prediction'].mean(['chain','draw'])
posterior_climate['clogvar_prediction_exp'] = posterior_climate['clogvar_prediction'].mean(['chain','draw'])
posterior_climate['cmean_prediction_exp_residual'] = cmeans-posterior_climate['cmean_prediction_exp']
posterior_climate['clogvar_prediction_exp_residual'] = clogvars-posterior_climate['clogvar_prediction_exp']

# %% Including Empirical Values in Posterior Object
posterior['means'] = (('stations'), means)
posterior['vars'] = (('stations'), vars)
posterior['logvars'] = (('stations'), logvars)

posterior_climate['cmeans'] = (('CM Grid Cell'), cmeans)
posterior_climate['cvars'] = (('CM Grid Cell'), cvars)
posterior_climate['clogvars'] = (('CM Grid Cell'), clogvars)

# %% Saving Output
scenario['Mean_Function_Posterior'] = posterior
scenario['Mean_Function_Posterior_Climate'] = posterior_climate

scenario_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'
np.save(scenario_outpath, scenario)



# %%
idata

# %%
aws_stations.shape
# %%

#     mean = numpyro.sample("mean",dist.Normal(mean_func, mean_noise))

    logvar = 





    logvar_b0 = numpyro.sample("logvar_b0",dist.Uniform(-10.0, 1.0))
    logvar_b1 = numpyro.sample("logvar_b1",dist.Uniform(-10, 10.0))
    logvar_b2 = numpyro.sample("logvar_b2",dist.Uniform(-10, 10.0))
    logvar_noise = numpyro.sample("logvar_noise",dist.Uniform(-5.0, 5.0))

    logvar_func = logvar_b0 + logvar_b1*scenario['oele_scaled'] + logvar_b2*scenario['olat_scaled']
    logvar = numpyro.sample("logvar",dist.Normal(logvar_func, logvar_noise))
    var = jnp.exp(logvar)

    obs_mask = (jnp.isnan(scenario['odata_scaled'])==False)
    numpyro.sample("Temperature", dist.Normal(mean, jnp.sqrt(var)).mask(obs_mask), obs=scenario["odata_scaled"])

# %%


    "MEAN_B_mean_b0_prior": dist.Normal(0.0, 3.0),
    "MEAN_B_mean_b1_prior": dist.Normal(0.0, 3.0),
    "LOGVAR_T_mean_b0_prior": dist.Normal(0.0, 3.0),
    "LOGVAR_T_mean_b1_prior": dist.Normal(0.0, 3.0),
    "LOGVAR_T_mean_b2_prior": dist.Normal(0.0, 3.0),
    "LOGVAR_B_mean_b0_prior": dist.Normal(0.0, 3.0),
    "LOGVAR_B_mean_b1_prior": dist.Normal(0.0, 5.0),
})


# %%

def mean_model(scenario):
    mean_b0 = numpyro.sample("mean_b0",dist.Normal(0, 3.0))
    mean_b1 = numpyro.sample("mean_b1",dist.Normal(0, 3.0))
    mean_b2 = numpyro.sample("mean_b2",dist.Normal(0, 3.0))
    mean_noise = numpyro.sample("mean_noise",dist.Uniform(0.001, 1.0))

    mean_func = mean_b0 + mean_b1*scenario['oele_scaled'] + mean_b2*scenario['olat_scaled']
    mean = numpyro.sample("mean",dist.Normal(mean_func, mean_noise))

    logvar_b0 = numpyro.sample("logvar_b0",dist.Uniform(-10.0, 1.0))
    logvar_b1 = numpyro.sample("logvar_b1",dist.Uniform(-10, 10.0))
    logvar_b2 = numpyro.sample("logvar_b2",dist.Uniform(-10, 10.0))
    logvar_noise = numpyro.sample("logvar_noise",dist.Uniform(-5.0, 5.0))

    logvar_func = logvar_b0 + logvar_b1*scenario['oele_scaled'] + logvar_b2*scenario['olat_scaled']
    logvar = numpyro.sample("logvar",dist.Normal(logvar_func, logvar_noise))
    var = jnp.exp(logvar)

    obs_mask = (jnp.isnan(scenario['odata_scaled'])==False)
    numpyro.sample("Temperature", dist.Normal(mean, jnp.sqrt(var)).mask(obs_mask), obs=scenario["odata_scaled"])





# %%

ds_climate_stacked_jan = scenario['ds_climate_stacked_jan']
ds_climate_stacked_jan_filtered = scenario['ds_climate_stacked_jan_filtered']
ds_aws_stacked_jan_filtered = scenario['ds_aws_stacked_jan_filtered']
data_scaler = scenario['data_scaler'] 
ele_scaler = scenario['ele_scaler'] 
lat_scaler = scenario['lat_scaler'] 

# %% Sanity Check
print('Data Shapes: \n',
      f'oele_scaled.shape:{scenario["oele_scaled"].shape} \n',
      f'olat_scaled.shape:{scenario["olat_scaled"].shape} \n',
      f'odata_scaled.shape:{scenario["odata_scaled"].shape} \n',
      f'cele_scaled.shape:{scenario["cele_scaled"].shape} \n',
      f'clat_scaled.shape:{scenario["clat_scaled"].shape} \n',
      f'cdata_scaled.shape:{scenario["cdata_scaled"].shape} \n'
      )

print('Data Values: \n',
      f'oele_scaled min,max:{scenario["oele_scaled"].min(),scenario["oele_scaled"].max()} \n',
      f'olat_scaled min,max:{scenario["olat_scaled"].min(),scenario["olat_scaled"].max()} \n',
      f'odata_scaled min,max:{scenario["odata_scaled"].min(),scenario["odata_scaled"].max()} \n',
      f'cele_scaled min,max:{scenario["cele_scaled"].min(),scenario["cele_scaled"].max()} \n',
      f'clat_scaled min,max:{scenario["clat_scaled"].min(),scenario["clat_scaled"].max()} \n',
      f'cdata_scaled min,max:{scenario["cdata_scaled"].min(),scenario["cdata_scaled"].max()} \n'
      )

# %% Histogram of Scaled Elevation and Latitude

fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=300)#,frameon=False)

ax=axs[0]
ax.hist(scenario["oele_scaled"],bins=30,edgecolor='k',linewidth=0.2,density=1,alpha=0.5,label='AWS')
ax.hist(scenario["cele_scaled"],bins=30,edgecolor='k',linewidth=0.2,density=1,alpha=0.5,label='CM')
ax.set_xlabel('Scaled Elevation')
ax.set_ylabel('Probability Density')
ax.legend()

ax=axs[1]
ax.hist(scenario["olat_scaled"],bins=30,edgecolor='k',linewidth=0.2,density=1,alpha=0.5,label='AWS')
ax.hist(scenario["clat_scaled"],bins=30,edgecolor='k',linewidth=0.2,density=1,alpha=0.5,label='CM')
ax.set_xlabel('Scaled Latitude')
ax.set_ylabel('Probability Density')
ax.legend()


# %%
means = np.nanmean(scenario['odata_scaled'],axis=0)
vars = np.nanvar(scenario['odata_scaled'],axis=0)
logvars = np.log(vars)
cmeans = np.nanmean(scenario['cdata_scaled'],axis=0)
cvars = np.nanvar(scenario['cdata_scaled'],axis=0)
clogvars = np.log(cvars)
print('Prior Considerations: \n',
      f'Mean: min={means.min():.2f}, mean={means.mean():.2f}, max={means.max():.2f} \n',
      f'Var: min={vars.min():.2f}, mean={vars.mean():.2f}, max={vars.max():.2f} \n',
      f'Log(Var): min={logvars.min():.1f}, mean={logvars.mean():.1f}, max={logvars.max():.1f} \n',
      f'CMean: min={cmeans.min():.2f}, mean={cmeans.mean():.2f}, max={cmeans.max():.2f} \n',
      f'CVar: min={cvars.min():.2f}, mean={cvars.mean():.2f}, max={cvars.max():.2f} \n',
      f'CLog(Var): min={clogvars.min():.1f}, mean={clogvars.mean():.1f}, max={clogvars.max():.1f} \n',
)

# %%
def mean_model(scenario):

    mean_b0 = numpyro.sample("mean_b0",dist.Normal(0, 3.0))
    mean_b1 = numpyro.sample("mean_b1",dist.Normal(0, 3.0))
    mean_b2 = numpyro.sample("mean_b2",dist.Normal(0, 3.0))
    mean_noise = numpyro.sample("mean_noise",dist.Uniform(0.001, 1.0))

    mean_func = mean_b0 + mean_b1*scenario['oele_scaled'] + mean_b2*scenario['olat_scaled']
    mean = numpyro.sample("mean",dist.Normal(mean_func, mean_noise))

    logvar_b0 = numpyro.sample("logvar_b0",dist.Uniform(-10.0, 1.0))
    logvar_b1 = numpyro.sample("logvar_b1",dist.Uniform(-10, 10.0))
    logvar_b2 = numpyro.sample("logvar_b2",dist.Uniform(-10, 10.0))
    logvar_noise = numpyro.sample("logvar_noise",dist.Uniform(-5.0, 5.0))

    logvar_func = logvar_b0 + logvar_b1*scenario['oele_scaled'] + logvar_b2*scenario['olat_scaled']
    logvar = numpyro.sample("logvar",dist.Normal(logvar_func, logvar_noise))
    var = jnp.exp(logvar)

    obs_mask = (jnp.isnan(scenario['odata_scaled'])==False)
    numpyro.sample("Temperature", dist.Normal(mean, jnp.sqrt(var)).mask(obs_mask), obs=scenario["odata_scaled"])

def mean_model_climate(scenario):

    mean_b0 = numpyro.sample("mean_b0",dist.Normal(0, 3.0))
    mean_b1 = numpyro.sample("mean_b1",dist.Normal(0, 3.0))
    mean_b2 = numpyro.sample("mean_b2",dist.Normal(0, 3.0))
    mean_noise = numpyro.sample("mean_noise",dist.Uniform(0.001, 1.0))

    mean_func = mean_b0 + mean_b1*scenario['cele_scaled'] + mean_b2*scenario['clat_scaled']
    mean = numpyro.sample("mean",dist.Normal(mean_func, mean_noise))

    logvar_b0 = numpyro.sample("logvar_b0",dist.Uniform(-10.0, 1.0))
    logvar_b1 = numpyro.sample("logvar_b1",dist.Uniform(-10, 10.0))
    logvar_b2 = numpyro.sample("logvar_b2",dist.Uniform(-10, 10.0))
    logvar_noise = numpyro.sample("logvar_noise",dist.Uniform(-5.0, 5.0))

    logvar_func = logvar_b0 + logvar_b1*scenario['cele_scaled'] + logvar_b2*scenario['clat_scaled']
    logvar = numpyro.sample("logvar",dist.Normal(logvar_func, logvar_noise))
    var = jnp.exp(logvar)

    numpyro.sample("Temperature", dist.Normal(mean, jnp.sqrt(var)), obs=scenario["cdata_scaled"])

# %%
    

# %% Running Inference
mcmc = run_inference(mean_model, rng_key, 1000, 2000,1, scenario)
idata = az.from_numpyro(mcmc,
                coords={
                "stations": ds_aws_stacked_jan_filtered.Station.values,
    },
                dims={"logvar": ["stations"],
                      "mean": ["stations"]})

mcmc_climate = run_inference(mean_model_climate, rng_key, 1000, 2000,1, scenario)
idata_climate = az.from_numpyro(mcmc_climate,
                coords={
                "CM Grid Cell": ds_climate_stacked_jan_filtered.X.values,
    },
                dims={"logvar": ["CM Grid Cell"],
                      "mean": ["CM Grid Cell"]})

# %% Assigning Some Coords and Computing Mean Function Prediction
posterior = idata.posterior
posterior = posterior.assign_coords(oele_scaled=("stations", scenario['oele_scaled']))
posterior = posterior.assign_coords(olat_scaled=("stations", scenario['olat_scaled']))

posterior['mean_prediction'] = (posterior['mean_b0']
                                + posterior['mean_b1']*posterior['oele_scaled']
                                + posterior['mean_b2']*posterior['olat_scaled'])
posterior['logvar_prediction'] = (posterior['logvar_b0']
                                + posterior['logvar_b1']*posterior['oele_scaled']
                                + posterior['logvar_b2']*posterior['olat_scaled'])


# %% Computing Mean Function Predictions for the Climate Data 
posterior_climate = idata_climate.posterior

posterior_climate = posterior_climate.assign_coords(cele_scaled=("CM Grid Cell", scenario['cele_scaled']))
posterior_climate = posterior_climate.assign_coords(clat_scaled=("CM Grid Cell", scenario['clat_scaled']))

posterior_climate['cmean_prediction'] = (posterior_climate['mean_b0']
                                + posterior_climate['mean_b1']*posterior_climate['cele_scaled']
                                + posterior_climate['mean_b2']*posterior_climate['clat_scaled'])
posterior_climate['clogvar_prediction'] = (posterior_climate['logvar_b0']
                                + posterior_climate['logvar_b1']*posterior_climate['cele_scaled']
                                + posterior_climate['logvar_b2']*posterior_climate['clat_scaled'])

cmeans = np.nanmean(scenario['cdata_scaled'],axis=0)
cvars = np.nanvar(scenario['cdata_scaled'],axis=0)
clogvars = np.log(cvars)

# %% Computing Mean Function Prediction Expectation and Residual Compared with Empirical Value
posterior['mean_prediction_exp'] = posterior['mean_prediction'].mean(['chain','draw'])
posterior['logvar_prediction_exp'] = posterior['logvar_prediction'].mean(['chain','draw'])
posterior['mean_prediction_exp_residual'] = means-posterior['mean_prediction_exp']
posterior['logvar_prediction_exp_residual'] = logvars-posterior['logvar_prediction_exp']

posterior_climate['cmean_prediction_exp'] = posterior_climate['cmean_prediction'].mean(['chain','draw'])
posterior_climate['clogvar_prediction_exp'] = posterior_climate['clogvar_prediction'].mean(['chain','draw'])
posterior_climate['cmean_prediction_exp_residual'] = cmeans-posterior_climate['cmean_prediction_exp']
posterior_climate['clogvar_prediction_exp_residual'] = clogvars-posterior_climate['clogvar_prediction_exp']

# %% Including Empirical Values in Posterior Object
posterior['means'] = (('stations'), means)
posterior['vars'] = (('stations'), vars)
posterior['logvars'] = (('stations'), logvars)

posterior_climate['cmeans'] = (('CM Grid Cell'), cmeans)
posterior_climate['cvars'] = (('CM Grid Cell'), cvars)
posterior_climate['clogvars'] = (('CM Grid Cell'), clogvars)

# %% Saving Output
scenario['Mean_Function_Posterior'] = posterior
scenario['Mean_Function_Posterior_Climate'] = posterior_climate

scenario_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'
np.save(scenario_outpath, scenario)

