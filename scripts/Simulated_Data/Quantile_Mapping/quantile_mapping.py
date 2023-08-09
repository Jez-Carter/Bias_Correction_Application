# %% Importing packages
import numpyro.distributions as dist
import numpyro
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import jax
from jax import random
from tinygp import kernels, GaussianProcess
from src.simulated_data_functions import run_inference

rng_key = random.PRNGKey(3)
rng_key, rng_key_ = random.split(rng_key)
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1
legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm
out_path = '/home/jez/Bias_Correction/results/Paper_Images/'
jax.config.update("jax_enable_x64", True)

# %% Generating some simulated data
omean = 3
cmean = 2
ostd = 0.5
cstd = 1
t = np.arange(0,100,0.1)
lengthscale = 3

# GP = GaussianProcess(1 * kernels.ExpSquared(lengthscale),t)
GP = GaussianProcess(1 * kernels.ExpSineSquared(scale=20,gamma=2),t)

complete_realisation = GP.sample(rng_key)
ocomplete_realisation = complete_realisation * ostd+omean
ccomplete_realisation = complete_realisation * cstd+cmean

indecies = np.arange(0,len(t),1)
oindecies = np.random.choice(indecies,10,replace=False)
ot = t[oindecies]
ct = t[::10]
odata = ocomplete_realisation[oindecies]
cdata = ccomplete_realisation[::10]

# %% Defining a very basic Numpyro model for the single site 

def normal_model(data):
    mean = numpyro.sample("mean", dist.Normal(0.0, 2))
    standard_deviation = numpyro.sample("standard_deviation", dist.Normal(0.0, 2))
    numpyro.sample("Observations", dist.Normal(mean,standard_deviation),obs=data)

# %% Computing the posteriors
rng_key = random.PRNGKey(0)
    
mcmc = run_inference(normal_model, rng_key, 1000, 2000,1,odata)
oposterior_samples = mcmc.get_samples()
mcmc = run_inference(normal_model, rng_key, 1000, 2000,1,cdata)
cposterior_samples = mcmc.get_samples()

# %% Applying quantile mapping
cdf_c = norm.cdf(cdata,
                 cposterior_samples['mean'].reshape(-1,1),
                 cposterior_samples['standard_deviation'].reshape(-1,1)
                 )

c_corrected = norm.ppf(cdf_c,
                       oposterior_samples['mean'].reshape(-1,1),
                       oposterior_samples['standard_deviation'].reshape(-1,1)
                       )

# %%
quantile_mapping_scenario = {
    't': t,
    'ocomplete_realisation':ocomplete_realisation,
    'ccomplete_realisation':ccomplete_realisation,
    'ot':ot,
    'ct':ct,
    'odata':odata,
    'cdata':cdata,
    'c_corrected':c_corrected
}

outpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
np.save(f'{outpath}quantile_mapping_scenario.npy', quantile_mapping_scenario) 

# %% Plotting the output
fig, ax = plt.subplots(1,1,figsize=(12*cm, 5.0*cm),dpi= 300)

ax.plot(t,ocomplete_realisation,label='True Underlying Field',alpha=0.6)
ax.plot(t,ccomplete_realisation,label='Biased Underlying Field',alpha=0.6)
ax.scatter(ot,odata,label='In-Situ Observations',alpha=1.0,s=10,marker='x')
ax.scatter(ct,cdata,label='Climate Model Output',alpha=1.0,s=10,marker='x')

ax.plot(ct,c_corrected.mean(axis=0),
        label='Bias Corrected Output Expectation',
        color='k',alpha=0.6,linestyle='dotted')

ax.fill_between(ct,
                c_corrected.mean(axis=0)+3*c_corrected.std(axis=0),
                c_corrected.mean(axis=0)-3*c_corrected.std(axis=0),
                interpolate=True, color='k',alpha=0.2,
                label='Bias Corrected Output Uncertainty 3$\sigma$')

ax.set_xticklabels([])
ax.set_xlabel('Time')
ax.set_yticklabels([])
ax.set_ylabel('Value')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles,labels,
           fontsize=legend_fontsize,
           bbox_to_anchor=(0.5, -0.05),
           ncols=3,
           loc=10)
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig13.png',dpi=300,bbox_inches='tight')


# %%
