# %% Importing Packages
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import arviz as az
import numpyro
import pandas as pd

plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4
plt.rcParams.update({'font.size': 8})

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.simulated_data_functions import run_inference

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %% Specifications

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)

pd.options.display.max_colwidth = 100

plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1

legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm

out_path = '/home/jez/Bias_Correction/results/Paper_Images/'

jax.config.update("jax_enable_x64", True)

# %%
scenario_base = np.load(f'{inpath}scenario_base_hierarchical.npy',allow_pickle='TRUE').item()
scenario_2d = np.load(f'{inpath}scenario_2d_hierarchical.npy',allow_pickle='TRUE').item()

# %%
def site_level_model(scenario):
    mt = numpyro.sample("mt", scenario_base['MEAN_T_mean_prior'],
                        sample_shape=((scenario_base['odata'].shape[-1],)))
    lvt = numpyro.sample("lvt", scenario_base['LOGVAR_T_mean_prior'],
                        sample_shape=((scenario_base['odata'].shape[-1],)))
    vt = jnp.exp(lvt)
    st = jnp.sqrt(vt)

    mc_t = numpyro.sample("mc_t", scenario_base['MEAN_T_mean_prior'],
                        sample_shape=((scenario_base['cdata'].shape[-1],)))
    lvc_t = numpyro.sample("lvc_t", scenario_base['LOGVAR_T_mean_prior'],
                        sample_shape=((scenario_base['cdata'].shape[-1],)))
    mc_b = numpyro.sample("mc_b", scenario_base['MEAN_B_mean_prior'],
                        sample_shape=((scenario_base['cdata'].shape[-1],)))
    lvc_b = numpyro.sample("lvc_b", scenario_base['LOGVAR_B_mean_prior'],
                        sample_shape=((scenario_base['cdata'].shape[-1],)))
    mc = mc_t + mc_b
    lvc = lvc_t + lvc_b
    vc = jnp.exp(lvc)
    sc = jnp.sqrt(vc)

    numpyro.sample("t", dist.Normal(mt,st),obs=scenario_base['odata'])
    numpyro.sample("c", dist.Normal(mc,sc),obs=scenario_base['cdata'])

# %% 
rng_key = random.PRNGKey(0)
mcmc = run_inference(
        site_level_model, rng_key, 1000, 2000,1,scenario_base)
idata = az.from_numpyro(mcmc)
scenario_base['mcmc_step1'] = idata
scenario_base['mcmc_step1_samples']=mcmc.get_samples()

# %%
outpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
np.save(f'{outpath}scenario_base_hierarchical_step_by_step.npy', scenario_base) 

# %%
def plot_underlying_data_mean_1d(scenario,ax,ms):
    ax.plot(scenario['X'],scenario['MEAN_T'],label='Truth Mean',alpha=0.6)
    ax.plot(scenario['X'],scenario['MEAN_B'],label='Bias Mean',alpha=0.6)
    ax.plot(scenario['X'],scenario['MEAN_C'],label='Climate Model Mean',alpha=0.6)

    mt_samples = scenario['mcmc_step1_samples']['mt']
    mc_samples = scenario['mcmc_step1_samples']['mc_t']+scenario['mcmc_step1_samples']['mc_b']
    ax.errorbar(scenario_base['ox'],
                mt_samples.mean(axis=0),
                yerr=mt_samples.std(axis=0),
                ls='none',
                capsize=1,
                color='tab:blue',
                marker='.',
                ms=ms,
                label = 'In-Situ'
    )
    ax.errorbar(scenario_base['cx'],
                mc_samples.mean(axis=0),
                yerr=mc_samples.std(axis=0),
                ls='none',
                capsize=1,
                color='tab:green',
                marker='.',
                ms=ms,
                label = 'Climate'
    )
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.legend()

def plot_underlying_data_std_1d(scenario,ax,ms):
    ax.plot(scenario['X'],jnp.sqrt(jnp.exp(scenario['LOGVAR_T'])),label='Truth Std',alpha=0.6)
    ax.plot(scenario['X'],jnp.sqrt(jnp.exp(scenario['LOGVAR_B'])),label='Bias Std',alpha=0.6)
    ax.plot(scenario['X'],jnp.sqrt(jnp.exp(scenario['LOGVAR_C'])),label='Climate Model Std',alpha=0.6)

    lvt_samples = scenario['mcmc_step1_samples']['lvt']
    lvc_samples = scenario['mcmc_step1_samples']['lvc_t']+scenario['mcmc_step1_samples']['lvc_b']
    st_samples = jnp.sqrt(jnp.exp(lvt_samples))
    sc_samples = jnp.sqrt(jnp.exp(lvc_samples))
    ax.errorbar(scenario_base['ox'],
                st_samples.mean(axis=0),
                yerr=st_samples.std(axis=0),
                ls='none',
                capsize=1,
                color='tab:blue',
                marker='.',
                ms=ms,
                label = 'In-Situ'
    )
    ax.errorbar(scenario_base['cx'],
                sc_samples.mean(axis=0),
                yerr=sc_samples.std(axis=0),
                ls='none',
                capsize=1,
                color='tab:green',
                marker='.',
                ms=ms,
                label = 'Climate'
    )
    ax.set_xlabel('time')
    ax.set_ylabel('temperature std')
    ax.legend()

# %%
min_x,max_x = 0,100
X = jnp.arange(min_x,max_x,0.1)

scenario_base_hierarchical = scenario_base

fig, axs = plt.subplots(2,1,figsize=(17*cm, 10.0*cm),dpi= 300)
plot_underlying_data_mean_1d(scenario_base_hierarchical,axs[0],ms=2)
plot_underlying_data_std_1d(scenario_base_hierarchical,axs[1],ms=2)
axs[0].set_ylabel('Mean Parameter')
axs[1].set_ylabel('Std. Dev. Parameter')
for ax in axs:
    ax.get_legend().remove()

axs[1].set_xlabel('s')
axs[0].set_xticklabels([])

handles, labels = axs[0].get_legend_handles_labels()
labels = ['Parameter Value: In-Situ Data',
          'Parameter Bias',
          'Parameter Value: Climate Model Output',
          'In-Situ Observations Sample',
          'Climate Model Output Sample']

fig.legend(handles,labels,
           fontsize=legend_fontsize,
           bbox_to_anchor=(0.5, 0),
           ncols=3,
           loc=10)

axs[0].annotate('a.',xy=(0.01,1.01),xycoords='axes fraction')
axs[1].annotate('b.',xy=(0.01,1.01),xycoords='axes fraction')
plt.tight_layout()
plt.subplots_adjust(hspace=0.08)
plt.show()
