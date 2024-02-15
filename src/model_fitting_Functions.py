import numpyro
import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess
from tinygp.kernels.distance import L2Distance
import jax
import jax.numpy as jnp
import arviz as az

from src.helper_functions import diagonal_noise
from src.helper_functions import run_inference

jax.config.update("jax_enable_x64", True)

def generate_mt_conditional_mc_dist(
    scenario, mc_kernel, mc_mean_cx, mt_kernel, mt_mean_ox, mc
):
    ox = scenario["ox"]
    cx = scenario["cx"]
    y2 = mc
    u1 = mt_mean_ox #jnp.full(ox.shape[0], mt_mean)
    u2 = mc_mean_cx #jnp.full(cx.shape[0], mc_mean) 
    k11 = mt_kernel(ox, ox) + jnp.eye(ox.shape[0])*scenario["jitter"] #+ diagonal_noise(ox, scenario["jitter"])
    k12 = mt_kernel(ox, cx)
    k21 = mt_kernel(cx, ox)
    k22 = mc_kernel(cx, cx) + jnp.eye(cx.shape[0])*scenario["jitter"] #+ diagonal_noise(cx, scenario["jitter"])
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn_dist = dist.MultivariateNormal(u1g2, k1g2)
    return mvn_dist


def generate_lvt_conditional_lvc_dist(
    scenario, lvc_kernel, lvc_mean_cx, lvt_kernel, lvt_mean_ox, lvc
):
    ox = scenario["ox"]
    cx = scenario["cx"]
    y2 = lvc
    u1 = lvt_mean_ox #jnp.full(ox.shape[0], lvt_mean)
    u2 = lvc_mean_cx #jnp.full(cx.shape[0], lvc_mean)
    k11 = lvt_kernel(ox, ox) + jnp.eye(ox.shape[0])*scenario["jitter"] #diagonal_noise(ox, scenario["jitter"])
    k12 = lvt_kernel(ox, cx)
    k21 = lvt_kernel(cx, ox)
    k22 = lvc_kernel(cx, cx) + jnp.eye(cx.shape[0])*scenario["jitter"] #+ diagonal_noise(cx, scenario["jitter"])
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn_dist = dist.MultivariateNormal(u1g2, k1g2)
    return mvn_dist


def hierarchical_model(scenario):
    mt_kern_var = numpyro.sample("mt_kern_var", scenario["MEAN_T_variance_prior"])
    mt_lengthscale = numpyro.sample(
        "mt_lengthscale", scenario["MEAN_T_lengthscale_prior"]
    )
    mt_kernel = mt_kern_var * kernels.Matern32(mt_lengthscale,L2Distance())
    mt_mean_bo = numpyro.sample("mt_mean_bo", scenario["MEAN_T_mean_b0_prior"])
    mt_mean_b1 = numpyro.sample("mt_mean_b1", scenario["MEAN_T_mean_b1_prior"])
    mt_mean_b2 = numpyro.sample("mt_mean_b2", scenario["MEAN_T_mean_b2_prior"])

    mb_kern_var = numpyro.sample("mb_kern_var", scenario["MEAN_B_variance_prior"])
    mb_lengthscale = numpyro.sample(
        "mb_lengthscale", scenario["MEAN_B_lengthscale_prior"]
    )
    mb_kernel = mb_kern_var * kernels.Matern32(mb_lengthscale,L2Distance())
    mb_mean_bo = numpyro.sample("mb_mean_bo", scenario["MEAN_B_mean_b0_prior"])
    mb_mean_b1 = numpyro.sample("mb_mean_b1", scenario["MEAN_B_mean_b1_prior"])

    mc_kernel = mt_kernel + mb_kernel
    mc_kernel_cx = mc_kernel(scenario['cx'],scenario['cx'])+jnp.eye(scenario['cx'].shape[0])*scenario["jitter"]#+diagonal_noise(scenario['cx'], scenario["jitter"])
    mt_mean_cx = mt_mean_bo + mt_mean_b1*scenario['cele'] + mt_mean_b2*scenario['clat']
    mb_mean_cx = mb_mean_bo + mb_mean_b1*scenario['cele']
    mc_mean_cx = mt_mean_cx + mb_mean_cx
    mc_mvn = dist.MultivariateNormal(mc_mean_cx, mc_kernel_cx)
    mc = numpyro.sample("mc", mc_mvn)
    mt_mean_ox = mt_mean_bo + mt_mean_b1*scenario['oele'] + mt_mean_b2*scenario['olat']
    mt_conditional_mc_dist = generate_mt_conditional_mc_dist(
        scenario, mc_kernel, mc_mean_cx, mt_kernel, mt_mean_ox, mc
    )
    mt = numpyro.sample("mt", mt_conditional_mc_dist)

    lvt_kern_var = numpyro.sample("lvt_kern_var", scenario["LOGVAR_T_variance_prior"])
    lvt_lengthscale = numpyro.sample(
        "lvt_lengthscale", scenario["LOGVAR_T_lengthscale_prior"]
    )
    lvt_kernel = lvt_kern_var * kernels.Matern32(lvt_lengthscale,L2Distance())
    lvt_mean_bo = numpyro.sample("lvt_mean_bo", scenario["LOGVAR_T_mean_b0_prior"])
    lvt_mean_b1 = numpyro.sample("lvt_mean_b1", scenario["LOGVAR_T_mean_b1_prior"])
    lvt_mean_b2 = numpyro.sample("lvt_mean_b2", scenario["LOGVAR_T_mean_b2_prior"])

    lvb_kern_var = numpyro.sample("lvb_kern_var", scenario["LOGVAR_B_variance_prior"])
    lvb_lengthscale = numpyro.sample(
        "lvb_lengthscale", scenario["LOGVAR_B_lengthscale_prior"]
    )
    lvb_kernel = lvb_kern_var * kernels.Matern32(lvb_lengthscale,L2Distance())
    lvb_mean_bo = numpyro.sample("lvb_mean_bo", scenario["LOGVAR_B_mean_b0_prior"])
    lvb_mean_b1 = numpyro.sample("lvb_mean_b1", scenario["LOGVAR_B_mean_b1_prior"])

    lvc_kernel = lvt_kernel + lvb_kernel
    lvc_kernel_cx = lvc_kernel(scenario['cx'],scenario['cx'])+jnp.eye(scenario['cx'].shape[0])*scenario["jitter"]#+diagonal_noise(scenario['cx'], scenario["jitter"])
    lvt_mean_cx = lvt_mean_bo + lvt_mean_b1*scenario['cele'] + lvt_mean_b2*scenario['clat']
    lvb_mean_cx = lvb_mean_bo + lvb_mean_b1*scenario['cele']
    lvc_mean_cx = lvt_mean_cx + lvb_mean_cx
    lvc_mvn = dist.MultivariateNormal(lvc_mean_cx, lvc_kernel_cx)
    lvc = numpyro.sample("lvc", lvc_mvn)
    lvt_mean_ox = lvt_mean_bo + lvt_mean_b1*scenario['oele'] + lvt_mean_b2*scenario['olat']
    lvt_conditional_lvc_dist = generate_lvt_conditional_lvc_dist(
        scenario, lvc_kernel, lvc_mean_cx, lvt_kernel, lvt_mean_ox, lvc
    )
    lvt = numpyro.sample("lvt", lvt_conditional_lvc_dist)

    vt = jnp.exp(lvt)
    vc = jnp.exp(lvc)

    obs_mask = (jnp.isnan(scenario['odata_scaled'])==False)
    numpyro.sample("t", dist.Normal(mt, jnp.sqrt(vt)).mask(obs_mask), obs=scenario["odata_scaled"])
    numpyro.sample("c", dist.Normal(mc, jnp.sqrt(vc)), obs=scenario["cdata_scaled"])


def generate_posterior_hierarchical(
    scenario, rng_key, num_warmup, num_samples, num_chains
):
    mcmc = run_inference(
        hierarchical_model, rng_key, num_warmup, num_samples, num_chains, scenario
    )
    idata = az.from_numpyro(mcmc)
    scenario["mcmc"] = idata
    scenario["mcmc_samples"] = mcmc.get_samples()
