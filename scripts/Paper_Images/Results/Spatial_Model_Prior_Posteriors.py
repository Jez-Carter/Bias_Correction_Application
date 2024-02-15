# %% Importing Packages
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
from scipy.spatial import distance
import arviz as az
import pandas as pd

plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 1

legend_fontsize = 6
cm = 1 / 2.54
text_width = 17.68 * cm
page_width = 21.6 * cm

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

def plot_priors(scenario,prior_keys,axs,rng_key,vlinewidth):
    for key,ax in zip(prior_keys,axs):
        variable = key.split('_prior')[0]
        value = scenario[variable]
        prior_sample = scenario[key].sample(rng_key,(10000,))
        ax.hist(prior_sample,density=True,bins=100,alpha=0.6)

def plot_posteriors(posterior,posterior_keys,axs):
    for key,ax in zip(posterior_keys,axs):
        posterior_sample = posterior[key].data.reshape(-1)
        ax.hist(posterior_sample,density=True,bins=100,alpha=0.6)

results_path = '/home/jez/Bias_Correction_Application/results/Paper_Images/'

# %% Loading data
base_path = '/home/jez/'
inpath = f"{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()

# %% Tables: Prior and posterior distribution statistics
parameters = [
    "In-Situ Kernel Variance $v_{\phi_Y}$",
    "In-Situ Kernel Lengthscale $l_{\phi_Y}$",
    "In-Situ Observation Noise $\sigma_{\phi_Y}$",
    "Bias Kernel Variance $v_{\phi_B}$ ",
    "Bias Kernel Lengthscale $l_{\phi_B}$",
    "Bias Noise $\sigma_{\phi_B}$",
]

desired_index_order = [
    "kern_var",
    "lengthscale",
    "noise",
    "bkern_var",
    "blengthscale",
    "bnoise",
]

columns = ["Exp.", "Std. Dev.", "95\% C.I. L.B.", "95\% C.I. U.B."]

desired_columns = ["mean", "sd", "hdi_2.5%", "hdi_97.5%"]

dep_vars = ['mean','logvar']
df_concs = []

for var in dep_vars:
    df = az.summary(scenario[f"mcmc_dualprocess_{var}_residual"].posterior, hdi_prob=0.95)
    df = df.reindex(desired_index_order)
    df = df.set_index(np.array(parameters))
    df = df[desired_columns]
    df.columns = columns

    df_singleprocess = az.summary(scenario[f"mcmc_singleprocess_{var}_residual"].posterior, hdi_prob=0.95)
    df_singleprocess = df_singleprocess.reindex(desired_index_order)
    df_singleprocess = df_singleprocess.set_index(np.array(parameters))
    df_singleprocess = df_singleprocess[desired_columns]
    df_singleprocess.columns = columns

    expectations = []
    standard_deviations = []
    LB_CIs = []
    UB_CIs = []

    prior_keys = [
    f"{var}_residual_obs_kvprior",
    f"{var}_residual_obs_klprior",
    f"{var}_residual_obs_nprior",
    f"{var}_residual_bias_kvprior",
    f"{var}_residual_bias_klprior",
    f"{var}_residual_bias_nprior",
    ]

    for key in prior_keys:
        distribution = scenario[key]
        expectation = distribution.mean
        variance = distribution.variance
        standard_deviation = jnp.sqrt(variance)
        LB_CI = distribution.icdf(0.025)
        UB_CI = distribution.icdf(0.975)
        expectations.append(expectation)
        standard_deviations.append(standard_deviation)
        LB_CIs.append(LB_CI)
        UB_CIs.append(UB_CI)
    d = {
        columns[0]: expectations,
        columns[1]: standard_deviations,
        columns[2]: LB_CIs,
        columns[3]: UB_CIs,
    }
    df_prior = pd.DataFrame(data=d, index=parameters)

    df["Distribution"] = "Posterior Shared Process"
    df_singleprocess["Distribution"] = "Posterior Single Process"
    df_prior["Distribution"] = "Prior"

    # df_conc = pd.concat([df, df_singleprocess, df_prior])
    df_conc = pd.concat([df_prior, df, df_singleprocess])
    df_conc = df_conc.set_index([df_conc.index,'Distribution'])
    df_conc = df_conc.reindex(index=parameters,level=0)
    df_conc = df_conc.astype(float)
    df_conc = df_conc.round(2)
    df_concs.append(df_conc)

print(df_concs[0].to_latex(escape=False))
print(df_concs[1].to_latex(escape=False))

# %% Visualising prior and posterior distributions (Mean)

prior_keys = [
    "mean_residual_obs_kvprior",
    "mean_residual_obs_klprior",
    "mean_residual_obs_nprior",
    "mean_residual_bias_kvprior",
    "mean_residual_bias_klprior",
    "mean_residual_bias_nprior",
]

posterior_keys = [
    "kern_var",
    "lengthscale",
    "noise",
    "bkern_var",
    "blengthscale",
    "bnoise",
]

titles = [
    "a. $v_{\mu_Y}$",
    "b. $l_{\mu_Y}$",
    "c. $\sigma^2_{\mu_Y}$",
    "d. $v_{\mu_B}$",
    "e. $l_{\mu_B}$",
    "f. $\sigma^2_{\mu_B}$",
]

fig = plt.figure(figsize=(text_width, text_width*0.8), dpi=300)
gs = gridspec.GridSpec(2, 3)
gs.update(wspace=0.3)
gs.update(hspace=0.2)

axs = [
    plt.subplot(gs[0, 0]),
    plt.subplot(gs[0, 1]),
    plt.subplot(gs[0, 2]),
    plt.subplot(gs[1, 0]),
    plt.subplot(gs[1, 1]),
    plt.subplot(gs[1, 2]),
]

rng_key = random.PRNGKey(5)
plot_priors(scenario, prior_keys, axs, rng_key, 0.5)
plot_posteriors(scenario["mcmc_dualprocess_mean_residual"].posterior, posterior_keys, axs)
plot_posteriors(scenario["mcmc_singleprocess_mean_residual"].posterior, posterior_keys[:3], axs)

for ax, title in zip(axs, titles):
    ax.set_title(title, pad=3, loc="left", fontsize=8)

for ax in axs[::3]:
    ax.set_ylabel("Prob. Density")

for ax in axs[-3:]:
    ax.set_xlabel("Value")

labels = ["Prior", "Shared Process Posterior","Single Process Posterior"]
fig.legend(
    labels, fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.025), ncols=7, loc=10
)

# for ax in axs[2::3]:
# axs[-1].set_xlim([0, 3])

plt.tight_layout()
plt.show()
fig.savefig(f"{results_path}figa02.pdf", dpi=300, bbox_inches="tight")

# %% Visualising prior and posterior distributions (LogVar)

prior_keys = [
    "logvar_residual_obs_kvprior",
    "logvar_residual_obs_klprior",
    "logvar_residual_obs_nprior",
    "logvar_residual_bias_kvprior",
    "logvar_residual_bias_klprior",
    "logvar_residual_bias_nprior",
]

posterior_keys = [
    "kern_var",
    "lengthscale",
    "noise",
    "bkern_var",
    "blengthscale",
    "bnoise",
]

titles = [
    "a. $v_{log(\sigma_Y^2)}$",
    "b. $l_{log(\sigma_Y^2)}$",
    "c. $\sigma^2_{log(\sigma_Y^2)}$",
    "d. $v_{log(\sigma_B^2)}$",
    "e. $l_{log(\sigma_B^2)}$",
    "f. $\sigma^2_{log(\sigma_B^2)}$",
]

fig = plt.figure(figsize=(text_width, text_width*0.8), dpi=300)
gs = gridspec.GridSpec(2, 3)
gs.update(wspace=0.3)
gs.update(hspace=0.2)

axs = [
    plt.subplot(gs[0, 0]),
    plt.subplot(gs[0, 1]),
    plt.subplot(gs[0, 2]),
    plt.subplot(gs[1, 0]),
    plt.subplot(gs[1, 1]),
    plt.subplot(gs[1, 2]),
]

rng_key = random.PRNGKey(5)
plot_priors(scenario, prior_keys, axs, rng_key, 0.5)
plot_posteriors(scenario["mcmc_dualprocess_logvar_residual"].posterior, posterior_keys, axs)
plot_posteriors(scenario["mcmc_singleprocess_logvar_residual"].posterior, posterior_keys[:3], axs)

for ax, title in zip(axs, titles):
    ax.set_title(title, pad=3, loc="left", fontsize=8)

for ax in axs[::3]:
    ax.set_ylabel("Prob. Density")

for ax in axs[-3:]:
    ax.set_xlabel("Value")

labels = ["Prior", "Shared Process Posterior","Single Process Posterior"]
fig.legend(
    labels, fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.025), ncols=7, loc=10
)

# for ax in axs[::3]:
#     ax.set_xlim([0, 1])

# for ax in axs[2::3]:
#     ax.set_xlim([0, 0.1])

plt.tight_layout()
plt.show()
fig.savefig(f"{results_path}figa03.pdf", dpi=300, bbox_inches="tight")

# %%
