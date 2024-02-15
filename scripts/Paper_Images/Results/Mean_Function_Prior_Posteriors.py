# %% Importing Packages
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# %% 
az.summary(posterior_obs[desired_parameters])


# %% Table of Prior and Posterior Metrics
posterior_obs = scenario["Mean_Function_Posterior"]#.expand_dims(dim={"Source": ['Obs']})
posterior_climate = scenario["Mean_Function_Posterior_Climate"]#expand_dims(dim={"Source": ['Climate']})

desired_parameters = [
    "mean_b0",
    "mean_b1",
    "mean_b2",
    "mean_noise",
    "logvar_b0",
    "logvar_noise",
]
parameters = [
    r"Mean Function $\beta_{0,{\mu_Y}}$",
    r"Mean Function $\beta_{1,{\mu_Y}}$",
    r"Mean Function $\beta_{2,{\mu_Y}}$",
    r"Mean Function $n_{\mu_Y}$ ",
    r"Log Var. Function $\beta_{0,{{Logvar}_Y}}$",
    r"Log Var. Function $n_{{Logvar}_Y}$",
]

prior_keys = [
    "meanfunc_b0_prior",
    "meanfunc_b1_prior",
    "meanfunc_b2_prior",
    "meanfunc_noise_prior",
    "logvarfunc_b0_prior",
    "logvarfunc_noise_prior",
]

desired_columns = ["mean", "sd", "hdi_2.5%", "hdi_97.5%"]
columns = ["Exp.", "Std. Dev.", "95\% C.I. L.B.", "95\% C.I. U.B.","Distribution"]

df_obs = az.summary(posterior_obs[desired_parameters], hdi_prob=0.95)[desired_columns]
df_climate = az.summary(posterior_climate[desired_parameters], hdi_prob=0.95)[desired_columns]

df_obs['Distribution'] = 'Posterior AWS'
df_climate['Distribution'] = 'Posterior Climate Model'

df_merged = pd.concat([df_obs,df_climate])
# df_merged = df_merged.set_index([df_merged.index,'Source'])
# df_merged = df_merged.reindex(index=desired_parameters,level=0)
# df_merged = df_merged.rename_axis(['Parameter','Source'])
df_merged = df_merged.rename(index=dict(zip(desired_parameters, parameters)))
df_merged.columns = columns

expectations = []
standard_deviations = []
LB_CIs = []
UB_CIs = []

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
df_prior = df_prior.rename_axis(['Parameter'])
df_prior["Distribution"] = "Prior"

df_conc = pd.concat([df_merged,df_prior])

df_conc = df_conc.set_index([df_conc.index,'Distribution'])
df_conc = df_conc.reindex(index=parameters,level=0)
df_conc = df_conc.reindex(index=['Prior','Posterior AWS','Posterior Climate Model'],level=1)
df_conc = df_conc.rename_axis(['Parameter','Distribution'])

df_conc = df_conc.astype(float)
df_conc = df_conc.round(2)
print(df_conc.to_latex(escape=False))

# %%

df_conc


# %% Visualising prior and posterior distributions
prior_keys = [
    "meanfunc_b0_prior",
    "meanfunc_b1_prior",
    "meanfunc_b2_prior",
    "meanfunc_noise_prior",
    "logvarfunc_b0_prior",
    "logvarfunc_noise_prior",
]

posterior_keys = [
    "mean_b0",
    "mean_b1",
    "mean_b2",
    "mean_noise",
    "logvar_b0",
    "logvar_noise",
]

titles = [
    r"a. $\beta_{0,{\mu}}$",
    r"b. $\beta_{1,{\mu}}$",
    r"c. $\beta_{2,{\mu}}$",
    r"d. $n_{\mu}$",
    r"e. $\beta_{0,log(\sigma^2)}$",
    r"f. $n_{log(\sigma^2)}$",
]


fig = plt.figure(figsize=(text_width, text_width), dpi=300)
gs = gridspec.GridSpec(3, 2)
gs.update(wspace=0.2)
gs.update(hspace=0.2)

axs = [
    plt.subplot(gs[0, 0]),
    plt.subplot(gs[0, 1]),
    plt.subplot(gs[1, 0]),
    plt.subplot(gs[1, 1]),
    plt.subplot(gs[2, 0]),
    plt.subplot(gs[2, 1]),
]

rng_key = random.PRNGKey(5)
plot_priors(scenario, prior_keys, axs, rng_key, 0.5)
plot_posteriors(scenario["Mean_Function_Posterior"], posterior_keys, axs)
plot_posteriors(scenario["Mean_Function_Posterior_Climate"], posterior_keys, axs)

for ax, title in zip(axs, titles):
    # ax.set_title(title, pad=3, loc="left", fontsize=8)
    ax.annotate(title,xy=(0.02,0.93),xycoords='axes fraction')

for ax in axs[::2]:
    ax.set_ylabel("Prob. Density")

for ax in axs[-2:]:
    ax.set_xlabel("Value")

labels = ["Prior", "Posterior AWS","Posterior Climate Model"]
fig.legend(
    labels, fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.025), ncols=7, loc=10
)

# for ax in axs[2::3]:
# axs[-2].set_xlim([-3, 3])

plt.tight_layout()
plt.show()

fig.savefig(f"{results_path}figa01.pdf", dpi=300, bbox_inches="tight")
