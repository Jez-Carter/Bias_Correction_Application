# %% Importing packages
import numpyro.distributions as dist
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.patches as patches
from tinygp import kernels, GaussianProcess
from src.simulated_data_functions import plot_latent_data_1d
from src.quantile_mapping_functions import plotting_quantile_mapping

# %% Specifications
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

# %% Figure __
fig, ax = plt.subplots(figsize=(13*cm, 5.0*cm),dpi= 300)

xs = np.linspace(-1,5,100)
ys = norm.pdf(xs, 3, 0.5)
zs = norm.pdf(xs, 2, 1.0)
plot = ax.plot(xs, ys, lw=2,linestyle='dashed')
plt.hist(dist.Normal(3.0,0.5).sample(rng_key,(10000,)),density=True,bins=100,color=plot[0].get_color(),alpha=0.7,label='$Y\sim \mathcal{N}(3,0.5)$')
plot2 = ax.plot(xs, zs, lw=2,linestyle='dashed')
plt.hist(dist.Normal(2.0,1.0).sample(rng_key,(10000,)),density=True,bins=100,color=plot2[0].get_color(),alpha=0.7,label='$Z\sim \mathcal{N}(2,1)$')

y_percentile = np.percentile(dist.Normal(3.0,0.5).sample(rng_key,(10000,)), 20)
z_percentile = np.percentile(dist.Normal(2.0,1.0).sample(rng_key,(10000,)), 20)
plt.vlines(y_percentile,0,0.8,color='k',linestyle='dotted',label='20th Percentiles')
plt.vlines(z_percentile,0,0.8,color='k',linestyle='dotted')
style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="k")
arrow = patches.FancyArrowPatch((z_percentile, 0.8), (y_percentile, 0.8),**kw,
                             connectionstyle="arc3,rad=-.2")
plt.gca().add_patch(arrow)

ax.set_ylabel('Probability Density')
ax.set_xlabel('Value Measured')
plt.legend(fontsize=legend_fontsize)
plt.show()
fig.savefig(f'{out_path}fig00.png',dpi=300,bbox_inches='tight')

# %% Figure __
inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
qm_scenario = np.load(f'{inpath}quantile_mapping_scenario.npy',allow_pickle='TRUE').item()
fig, ax = plt.subplots(1,1,figsize=(12*cm, 7.0*cm),dpi= 300)

plotting_quantile_mapping(ax,qm_scenario)

ax.set_xticklabels([])
ax.set_xlabel('Time')
# ax.set_yticklabels([])
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

# %% Figure __

kern = 1 * kernels.ExpSquared(20)
x = np.arange(0,100,0.1)
x_samples = np.array([10,50,70])
y_samples = np.array([0.5,-0.3,0.8])
gp = GaussianProcess(kern,x_samples,diag=1e-10).condition(y_samples, x).gp
gp_samples = gp.sample(rng_key,(5,))

fig, axs = plt.subplots(1,2,figsize=(17*cm,7*cm))
axs[0].plot(x,kern(x,x)[0],label='kernel variance=1, lengthscale=20')
axs[0].set_xlabel(r"$d(s,s')$")
axs[0].set_ylabel(r"$RBF(s,s')$")

axs[1].plot(x,gp_samples[0],color='b',alpha=0.3,label='Realisations')
for sample in gp_samples[1:]:
    axs[1].plot(x,sample,color='b',alpha=0.3)
axs[1].fill_between(x,
                    gp_samples.mean(axis=0)-gp_samples.std(axis=0),
                    gp_samples.mean(axis=0)+gp_samples.std(axis=0),
                    color='k',
                    alpha=0.1,
                    label='$1\sigma$ Uncertainty')
axs[1].plot(x,gp_samples.mean(axis=0),color='k',alpha=0.5,linestyle='--',label='Expectation')
axs[1].scatter(x_samples,y_samples,color='r',s=30,label='Measurements')
axs[1].set_xlabel(r"$s$")
axs[1].set_ylabel(r"$Value$")

axs[0].annotate('a.',xy=(-0.08,-0.15),xycoords='axes fraction')
axs[1].annotate('b.',xy=(-0.08,-0.15),xycoords='axes fraction')

plt.legend(fontsize=legend_fontsize)
# plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig01.png',dpi=300,bbox_inches='tight')

# %% Figure __

min_x,max_x = 0,100
X = jnp.arange(min_x,max_x,0.1)

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'
scenario_base = np.load(f'{inpath}scenario_base.npy',allow_pickle='TRUE').item()

fig, ax = plt.subplots(figsize=(17*cm, 6.0*cm),dpi= 300)
plot_latent_data_1d(scenario_base,ax,ms=20)
ax.set_xlabel('Spatial Coordinate (s)')
ax.set_ylabel('Value')
ax.get_legend().remove()
labels= ["$\phi_Y \sim \mathcal{GP}(m=1,k_{RBF}(s,s'|v=1,l=3))$",
         "$\phi_B \sim \mathcal{GP}(m=-1,k_{RBF}(s,s'|v=1,l=10))$",
         '$\phi_Z = \phi_Y+\phi_B$']
fig.legend(labels,
           fontsize=legend_fontsize,
           bbox_to_anchor=(0.83, 0.85),
           ncols=1,
           loc=10)
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}fig02.png',dpi=300,bbox_inches='tight')