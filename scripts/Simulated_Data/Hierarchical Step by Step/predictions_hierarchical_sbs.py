
# %% Importing Packages
import numpy as np
import jax
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
import numpyro.distributions as dist
import pandas as pd
plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.simulated_data_functions import plot_underlying_data_1d
from src.simulated_data_functions import generate_posterior_predictive_realisations
from src.simulated_data_functions import plot_predictions_1d
from src.simulated_data_functions import plot_predictions_2d

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
scenario_base = np.load(f'{inpath}scenario_base_hierarchical_step_by_step.npy',allow_pickle='TRUE').item()

# %%
def diagonal_noise(coord,noise):
    return(jnp.diag(jnp.full(coord.shape[0],noise)))

def generate_truth_predictive_dist(scenario,
                                   posterior_param_realisation):
    mt_variance_realisation = posterior_param_realisation['mt_variance_realisation']
    mt_lengthscale_realisation = posterior_param_realisation['mt_lengthscale_realisation']
    mt_mean_realisation = posterior_param_realisation['mt_mean_realisation']
    mb_variance_realisation = posterior_param_realisation['mb_variance_realisation']
    mb_lengthscale_realisation = posterior_param_realisation['mb_lengthscale_realisation']
    mb_mean_realisation = posterior_param_realisation['mb_mean_realisation']
    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    jitter = scenario['jitter']

    mt_samples = scenario['mcmc_step1_samples']['mt']
    mc_samples = scenario['mcmc_step1_samples']['mc_t']+scenario['mcmc_step1_samples']['mc_b']
    mt_exp = mt_samples.mean(axis=0)
    mt_std = mt_samples.std(axis=0)
    mc_exp = mc_samples.mean(axis=0)
    mc_std = mc_samples.std(axis=0)

    cnoise_var = mc_std**2
    odata = mt_exp
    cdata = mc_exp
    omean = mt_mean_realisation
    bmean = mb_mean_realisation
    kernelo = mt_variance_realisation * kernels.ExpSquared(mt_lengthscale_realisation)
    kernelb = mb_variance_realisation * kernels.ExpSquared(mb_lengthscale_realisation)
    onoise_var = mt_std**2

    y2 = jnp.hstack([odata,cdata]) 
    u1 = jnp.full(nx.shape[0], omean)
    u2 = jnp.hstack([jnp.full(ox.shape[0], omean),jnp.full(cx.shape[0], omean+bmean)])
    k11 = kernelo(nx,nx) + diagonal_noise(nx,jitter)
    k12 = jnp.hstack([kernelo(nx,ox),kernelo(nx,cx)])
    k21 = jnp.vstack([kernelo(ox,nx),kernelo(cx,nx)])
    k22_upper = jnp.hstack([kernelo(ox,ox)+diagonal_noise(ox,onoise_var),kernelo(ox,cx)])
    k22_lower = jnp.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)+diagonal_noise(cx,cnoise_var)])
    k22 = jnp.vstack([k22_upper,k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    # k1g2 = k11 - jnp.matmul(p21.T,p21)
    k1g2 = k11 - jnp.matmul(jnp.matmul(k12,k22i),k21)
    k1g2 = k1g2
    # mvn = multivariate_normal(u1g2,k1g2)
    mvn = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn)

def generate_bias_predictive_dist(scenario,
                                   posterior_param_realisation):
    mt_variance_realisation = posterior_param_realisation['mt_variance_realisation']
    mt_lengthscale_realisation = posterior_param_realisation['mt_lengthscale_realisation']
    mt_mean_realisation = posterior_param_realisation['mt_mean_realisation']
    mb_variance_realisation = posterior_param_realisation['mb_variance_realisation']
    mb_lengthscale_realisation = posterior_param_realisation['mb_lengthscale_realisation']
    mb_mean_realisation = posterior_param_realisation['mb_mean_realisation']
    nx = scenario['nx']
    ox = scenario['ox']
    cx = scenario['cx']
    jitter = scenario['jitter']

    mt_samples = scenario['mcmc_step1_samples']['mt']
    mc_samples = scenario['mcmc_step1_samples']['mc_t']+scenario['mcmc_step1_samples']['mc_b']
    mt_exp = mt_samples.mean(axis=0)
    mt_std = mt_samples.std(axis=0)
    mc_exp = mc_samples.mean(axis=0)
    mc_std = mc_samples.std(axis=0)

    cnoise_var = mc_std**2
    odata = mt_exp
    cdata = mc_exp
    omean = mt_mean_realisation
    bmean = mb_mean_realisation
    kernelo = mt_variance_realisation * kernels.ExpSquared(mt_lengthscale_realisation)
    kernelb = mb_variance_realisation * kernels.ExpSquared(mb_lengthscale_realisation)
    onoise_var = mt_std**2

    y2 = jnp.hstack([odata,cdata]) 
    u1 = jnp.full(nx.shape[0], bmean)
    u2 = jnp.hstack([jnp.full(ox.shape[0], omean),jnp.full(cx.shape[0], omean+bmean)])
    k11 = kernelb(nx,nx) + diagonal_noise(nx,jitter)
    k12 = jnp.hstack([jnp.full((len(nx),len(ox)),0),kernelb(nx,cx)])
    k21 = jnp.vstack([jnp.full((len(ox),len(nx)),0),kernelb(cx,nx)])
    k22_upper = jnp.hstack([kernelo(ox,ox)+diagonal_noise(ox,onoise_var),kernelo(ox,cx)])
    k22_lower = jnp.hstack([kernelo(cx,ox),kernelo(cx,cx)+kernelb(cx,cx)+diagonal_noise(cx,cnoise_var)])
    k22 = jnp.vstack([k22_upper,k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12,k22i),y2-u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i,k21)
    k1g2 = k11 - jnp.matmul(p21.T,p21)
    # mvn = multivariate_normal(u1g2,k1g2)
    mvn = dist.MultivariateNormal(u1g2,k1g2)
    return(mvn)

def generate_posterior_predictive_realisations(
        scenario,
        num_parameter_realisations,num_posterior_pred_realisations):
    
    posterior_step2 = scenario['mcmc_step2'].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    iteration=0
    for i in np.random.randint(posterior_step2.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            'iteration':iteration,
            'mt_variance_realisation': posterior_step2['mt_kern_var'].data[0,:][i],
            'mt_lengthscale_realisation': posterior_step2['mt_lengthscale'].data[0,:][i],
            'mt_mean_realisation': posterior_step2['mt_mean'].data[0,:][i],
            'mb_variance_realisation': posterior_step2['mb_kern_var'].data[0,:][i],
            'mb_lengthscale_realisation': posterior_step2['mb_lengthscale'].data[0,:][i],
            'mb_mean_realisation': posterior_step2['mb_mean'].data[0,:][i],
        }
        
        truth_predictive_dist = generate_truth_predictive_dist(scenario,
                                   posterior_param_realisation)
        bias_predictive_dist = generate_bias_predictive_dist(scenario,
                                   posterior_param_realisation)
        iteration+=1

        # truth_predictive_realisations = truth_predictive_dist.rvs(num_posterior_pred_realisations)
        # bias_predictive_realisations = bias_predictive_dist.rvs(num_posterior_pred_realisations)
        rng_key = random.PRNGKey(0)
        truth_predictive_realisations = truth_predictive_dist.sample(rng_key,sample_shape=(num_posterior_pred_realisations,))
        bias_predictive_realisations = bias_predictive_dist.sample(rng_key,sample_shape=(num_posterior_pred_realisations,))
        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(truth_posterior_predictive_realisations)
    bias_posterior_predictive_realisations = jnp.array(bias_posterior_predictive_realisations)
    truth_posterior_predictive_realisations = truth_posterior_predictive_realisations.reshape(-1,truth_posterior_predictive_realisations.shape[-1])
    bias_posterior_predictive_realisations = bias_posterior_predictive_realisations.reshape(-1,bias_posterior_predictive_realisations.shape[-1])
    scenario['truth_posterior_predictive_realisations'] = truth_posterior_predictive_realisations
    scenario['bias_posterior_predictive_realisations'] = bias_posterior_predictive_realisations

# %%
generate_posterior_predictive_realisations(scenario_base,20,20)

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

axs[1].set_xlabel('s')
axs[0].set_xticklabels([])

handles, labels = axs[0].get_legend_handles_labels()
labels = ['Parameter Value: In-Situ Data',
          'Parameter Bias',
          'Parameter Value: Climate Model Output',
          'In-Situ Observations Sample',
          'Climate Model Output Sample']

plot_predictions_1d(scenario_base,'truth_posterior_predictive_realisations',axs[0],ms=20,color='tab:blue')
plot_predictions_1d(scenario_base,'bias_posterior_predictive_realisations',axs[0],ms=20,color='tab:orange')

for ax in axs:
    ax.get_legend().remove()

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

