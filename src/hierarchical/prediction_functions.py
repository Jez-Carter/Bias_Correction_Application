import numpy as np
import numpyro.distributions as dist
from tinygp import kernels
from tinygp.kernels.distance import L2Distance
import jax
from jax import random
import jax.numpy as jnp

from src.helper_functions import diagonal_noise

jax.config.update("jax_enable_x64", True)


def generate_mean_truth_predictive_dist(nx,nele,nlat, scenario, posterior_param_realisation):
    mt_variance_realisation = posterior_param_realisation["mt_variance_realisation"]
    mt_lengthscale_realisation = posterior_param_realisation[
        "mt_lengthscale_realisation"
    ]
    mt_mean_bo_realisation = posterior_param_realisation["mt_mean_bo_realisation"]
    mt_mean_b1_realisation = posterior_param_realisation["mt_mean_b1_realisation"]
    mt_mean_b2_realisation = posterior_param_realisation["mt_mean_b2_realisation"]
    mb_variance_realisation = posterior_param_realisation["mb_variance_realisation"]
    mb_lengthscale_realisation = posterior_param_realisation[
        "mb_lengthscale_realisation"
    ]
    mb_mean_bo_realisation = posterior_param_realisation["mb_mean_bo_realisation"]
    mb_mean_b1_realisation = posterior_param_realisation["mb_mean_b1_realisation"]

    mt_realisation = posterior_param_realisation["mt_realisation"]
    mc_realisation = posterior_param_realisation["mc_realisation"]

    ox = scenario["ox"]
    cx = scenario["cx"]
    jitter = scenario["jitter"]
    odata = mt_realisation
    cdata = mc_realisation
    oele = scenario["oele"]
    olat = scenario["olat"]
    cele = scenario["cele"]
    clat = scenario["clat"]
    omean_ox = mt_mean_bo_realisation + mt_mean_b1_realisation*oele + mt_mean_b2_realisation*olat
    omean_nx = mt_mean_bo_realisation + mt_mean_b1_realisation*nele + mt_mean_b2_realisation*nlat
    omean_cx = mt_mean_bo_realisation + mt_mean_b1_realisation*cele + mt_mean_b2_realisation*clat
    bmean_cx = mb_mean_bo_realisation + mb_mean_b1_realisation*cele 
    cmean_cx = omean_cx+bmean_cx
    kernelo = mt_variance_realisation * kernels.Matern32(mt_lengthscale_realisation,L2Distance())
    kernelb = mb_variance_realisation * kernels.Matern32(mb_lengthscale_realisation,L2Distance())

    y2 = jnp.hstack([odata, cdata])
    u1 = omean_nx 
    u2 = jnp.hstack([omean_ox, cmean_cx])

    k11 = kernelo(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([kernelo(nx, ox), kernelo(nx, cx)])
    k21 = jnp.vstack([kernelo(ox, nx), kernelo(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, jitter), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, jitter),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_logvar_truth_predictive_dist(nx,nele,nlat, scenario, posterior_param_realisation):
    lvt_variance_realisation = posterior_param_realisation["lvt_variance_realisation"]
    lvt_lengthscale_realisation = posterior_param_realisation[
        "lvt_lengthscale_realisation"
    ]
    # lvt_mean_realisation = posterior_param_realisation["lvt_mean_realisation"]
    lvt_mean_bo_realisation = posterior_param_realisation["lvt_mean_bo_realisation"]
    lvt_mean_b1_realisation = posterior_param_realisation["lvt_mean_b1_realisation"]
    lvt_mean_b2_realisation = posterior_param_realisation["lvt_mean_b2_realisation"]
    lvb_variance_realisation = posterior_param_realisation["lvb_variance_realisation"]
    lvb_lengthscale_realisation = posterior_param_realisation[
        "lvb_lengthscale_realisation"
    ]
    # lvb_mean_realisation = posterior_param_realisation["lvb_mean_realisation"]
    lvb_mean_bo_realisation = posterior_param_realisation["lvb_mean_bo_realisation"]
    lvb_mean_b1_realisation = posterior_param_realisation["lvb_mean_b1_realisation"]

    lvt_realisation = posterior_param_realisation["lvt_realisation"]
    lvc_realisation = posterior_param_realisation["lvc_realisation"]

    # nx = scenario['nx']
    ox = scenario["ox"]
    cx = scenario["cx"]
    jitter = scenario["jitter"]
    odata = lvt_realisation
    cdata = lvc_realisation
    oele = scenario["oele"]
    olat = scenario["olat"]
    cele = scenario["cele"]
    clat = scenario["clat"]
    omean_ox = lvt_mean_bo_realisation + lvt_mean_b1_realisation*oele + lvt_mean_b2_realisation*olat
    omean_nx = lvt_mean_bo_realisation + lvt_mean_b1_realisation*nele + lvt_mean_b2_realisation*nlat
    omean_cx = lvt_mean_bo_realisation + lvt_mean_b1_realisation*cele + lvt_mean_b2_realisation*clat
    bmean_cx = lvb_mean_bo_realisation + lvb_mean_b1_realisation*cele 
    cmean_cx = omean_cx+bmean_cx
    kernelo = lvt_variance_realisation * kernels.Matern32(lvt_lengthscale_realisation,L2Distance())
    kernelb = lvb_variance_realisation * kernels.Matern32(lvb_lengthscale_realisation,L2Distance())

    y2 = jnp.hstack([odata, cdata])
    u1 = omean_nx
    u2 = jnp.hstack([omean_ox, cmean_cx])
    k11 = kernelo(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([kernelo(nx, ox), kernelo(nx, cx)])
    k21 = jnp.vstack([kernelo(ox, nx), kernelo(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, jitter), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, jitter),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_mean_bias_predictive_dist(nx,nele,nlat, scenario, posterior_param_realisation):
    mt_variance_realisation = posterior_param_realisation["mt_variance_realisation"]
    mt_lengthscale_realisation = posterior_param_realisation[
        "mt_lengthscale_realisation"
    ]
    mt_mean_bo_realisation = posterior_param_realisation["mt_mean_bo_realisation"]
    mt_mean_b1_realisation = posterior_param_realisation["mt_mean_b1_realisation"]
    mt_mean_b2_realisation = posterior_param_realisation["mt_mean_b2_realisation"]
    mb_variance_realisation = posterior_param_realisation["mb_variance_realisation"]
    mb_lengthscale_realisation = posterior_param_realisation[
        "mb_lengthscale_realisation"
    ]
    mb_mean_bo_realisation = posterior_param_realisation["mb_mean_bo_realisation"]
    mb_mean_b1_realisation = posterior_param_realisation["mb_mean_b1_realisation"]

    mt_realisation = posterior_param_realisation["mt_realisation"]
    mc_realisation = posterior_param_realisation["mc_realisation"]

    # nx = scenario['nx']
    ox = scenario["ox"]
    cx = scenario["cx"]
    jitter = scenario["jitter"]
    odata = mt_realisation
    cdata = mc_realisation
    oele = scenario["oele"]
    olat = scenario["olat"]
    cele = scenario["cele"]
    clat = scenario["clat"]
    omean_ox = mt_mean_bo_realisation + mt_mean_b1_realisation*oele + mt_mean_b2_realisation*olat
    omean_cx = mt_mean_bo_realisation + mt_mean_b1_realisation*cele + mt_mean_b2_realisation*clat
    bmean_nx = mb_mean_bo_realisation + mb_mean_b1_realisation*nele 
    bmean_cx = mb_mean_bo_realisation + mb_mean_b1_realisation*cele 
    cmean_cx = omean_cx+bmean_cx
    kernelo = mt_variance_realisation * kernels.Matern32(mt_lengthscale_realisation,L2Distance())
    kernelb = mb_variance_realisation * kernels.Matern32(mb_lengthscale_realisation,L2Distance())

    y2 = jnp.hstack([odata, cdata])
    u1 = bmean_nx
    u2 = jnp.hstack(
        [omean_ox, cmean_cx]
    )
    k11 = kernelb(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([jnp.full((len(nx), len(ox)), 0), kernelb(nx, cx)])
    k21 = jnp.vstack([jnp.full((len(ox), len(nx)), 0), kernelb(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, jitter), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, jitter),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_logvar_bias_predictive_dist(nx,nele,nlat, scenario, posterior_param_realisation):
    lvt_variance_realisation = posterior_param_realisation["lvt_variance_realisation"]
    lvt_lengthscale_realisation = posterior_param_realisation[
        "lvt_lengthscale_realisation"
    ]
    lvt_mean_bo_realisation = posterior_param_realisation["lvt_mean_bo_realisation"]
    lvt_mean_b1_realisation = posterior_param_realisation["lvt_mean_b1_realisation"]
    lvt_mean_b2_realisation = posterior_param_realisation["lvt_mean_b2_realisation"]    
    lvb_variance_realisation = posterior_param_realisation["lvb_variance_realisation"]
    lvb_lengthscale_realisation = posterior_param_realisation[
        "lvb_lengthscale_realisation"
    ]
    lvb_mean_bo_realisation = posterior_param_realisation["lvb_mean_bo_realisation"]
    lvb_mean_b1_realisation = posterior_param_realisation["lvb_mean_b1_realisation"]

    lvt_realisation = posterior_param_realisation["lvt_realisation"]
    lvc_realisation = posterior_param_realisation["lvc_realisation"]

    # nx = scenario['nx']
    ox = scenario["ox"]
    cx = scenario["cx"]
    jitter = scenario["jitter"]
    odata = lvt_realisation
    cdata = lvc_realisation
    oele = scenario["oele"]
    olat = scenario["olat"]
    cele = scenario["cele"]
    clat = scenario["clat"]
    omean_ox = lvt_mean_bo_realisation + lvt_mean_b1_realisation*oele + lvt_mean_b2_realisation*olat
    omean_cx = lvt_mean_bo_realisation + lvt_mean_b1_realisation*cele + lvt_mean_b2_realisation*clat
    bmean_nx = lvb_mean_bo_realisation + lvb_mean_b1_realisation*nele 
    bmean_cx = lvb_mean_bo_realisation + lvb_mean_b1_realisation*cele 
    cmean_cx = omean_cx+bmean_cx
    kernelo = lvt_variance_realisation * kernels.Matern32(lvt_lengthscale_realisation,L2Distance())
    kernelb = lvb_variance_realisation * kernels.Matern32(lvb_lengthscale_realisation,L2Distance())

    y2 = jnp.hstack([odata, cdata])
    u1 = bmean_nx
    u2 = jnp.hstack(
        [omean_ox, cmean_cx]
    )
    k11 = kernelb(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([jnp.full((len(nx), len(ox)), 0), kernelb(nx, cx)])
    k21 = jnp.vstack([jnp.full((len(ox), len(nx)), 0), kernelb(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, jitter), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, jitter),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_posterior_predictive_realisations_hierarchical_mean(
    nx, nele, nlat, scenario, num_parameter_realisations, num_posterior_pred_realisations
):
    posterior = scenario["mcmc"].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            "mt_variance_realisation": posterior["mt_kern_var"].data[0, :][i],
            "mt_lengthscale_realisation": posterior["mt_lengthscale"].data[0, :][i],
            "mt_mean_bo_realisation": posterior["mt_mean_bo"].data[0, :][i],
            "mt_mean_b1_realisation": posterior["mt_mean_b1"].data[0, :][i],
            "mt_mean_b2_realisation": posterior["mt_mean_b2"].data[0, :][i],
            # "mt_mean_realisation": posterior["mt_mean"].data[0, :][i],
            "mb_variance_realisation": posterior["mb_kern_var"].data[0, :][i],
            "mb_lengthscale_realisation": posterior["mb_lengthscale"].data[0, :][i],
            "mb_mean_bo_realisation": posterior["mb_mean_bo"].data[0, :][i],
            "mb_mean_b1_realisation": posterior["mb_mean_b1"].data[0, :][i],
            # "mb_mean_realisation": posterior["mb_mean"].data[0, :][i],
            "mt_realisation": posterior["mt"].data[0, :][i],
            "mc_realisation": posterior["mc"].data[0, :][i],
        }

        truth_predictive_dist = generate_mean_truth_predictive_dist(
            nx, nele, nlat, scenario, posterior_param_realisation
        )
        bias_predictive_dist = generate_mean_bias_predictive_dist(
            nx, nele, nlat, scenario, posterior_param_realisation
        )

        rng_key = random.PRNGKey(0)
        truth_predictive_realisations = truth_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        bias_predictive_realisations = bias_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(
        truth_posterior_predictive_realisations
    )
    bias_posterior_predictive_realisations = jnp.array(
        bias_posterior_predictive_realisations
    )
    truth_posterior_predictive_realisations = (
        truth_posterior_predictive_realisations.reshape(
            -1, truth_posterior_predictive_realisations.shape[-1]
        )
    )
    bias_posterior_predictive_realisations = (
        bias_posterior_predictive_realisations.reshape(
            -1, bias_posterior_predictive_realisations.shape[-1]
        )
    )
    climate_posterior_predictive_realisations = (
        truth_posterior_predictive_realisations + bias_posterior_predictive_realisations
    )
    scenario[
        "mean_truth_posterior_predictive_realisations"
    ] = truth_posterior_predictive_realisations
    scenario[
        "mean_bias_posterior_predictive_realisations"
    ] = bias_posterior_predictive_realisations
    scenario[
        "mean_climate_posterior_predictive_realisations"
    ] = climate_posterior_predictive_realisations


def generate_posterior_predictive_realisations_hierarchical_std(
    nx, nele, nlat, scenario, num_parameter_realisations, num_posterior_pred_realisations
):
    posterior = scenario["mcmc"].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    for i in np.random.randint(posterior.draw.shape, size=num_parameter_realisations):
        posterior_param_realisation = {
            "lvt_variance_realisation": posterior["lvt_kern_var"].data[0, :][i],
            "lvt_lengthscale_realisation": posterior["lvt_lengthscale"].data[0, :][i],
            "lvt_mean_bo_realisation": posterior["lvt_mean_bo"].data[0, :][i],
            "lvt_mean_b1_realisation": posterior["lvt_mean_b1"].data[0, :][i],
            "lvt_mean_b2_realisation": posterior["lvt_mean_b2"].data[0, :][i],
            # "lvt_mean_realisation": posterior["lvt_mean"].data[0, :][i],
            "lvb_variance_realisation": posterior["lvb_kern_var"].data[0, :][i],
            "lvb_lengthscale_realisation": posterior["lvb_lengthscale"].data[0, :][i],
            "lvb_mean_bo_realisation": posterior["lvb_mean_bo"].data[0, :][i],
            "lvb_mean_b1_realisation": posterior["lvb_mean_b1"].data[0, :][i],
            # "lvb_mean_realisation": posterior["lvb_mean"].data[0, :][i],
            "lvt_realisation": posterior["lvt"].data[0, :][i],
            "lvc_realisation": posterior["lvc"].data[0, :][i],
        }

        truth_predictive_dist = generate_logvar_truth_predictive_dist(
            nx, nele, nlat, scenario, posterior_param_realisation
        )
        bias_predictive_dist = generate_logvar_bias_predictive_dist(
            nx, nele, nlat, scenario, posterior_param_realisation
        )

        rng_key = random.PRNGKey(0)
        truth_predictive_realisations = truth_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        bias_predictive_realisations = bias_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(
        truth_posterior_predictive_realisations
    )
    bias_posterior_predictive_realisations = jnp.array(
        bias_posterior_predictive_realisations
    )
    truth_posterior_predictive_realisations = (
        truth_posterior_predictive_realisations.reshape(
            -1, truth_posterior_predictive_realisations.shape[-1]
        )
    )
    bias_posterior_predictive_realisations = (
        bias_posterior_predictive_realisations.reshape(
            -1, bias_posterior_predictive_realisations.shape[-1]
        )
    )
    climate_posterior_predictive_realisations = (
        truth_posterior_predictive_realisations + bias_posterior_predictive_realisations
    )

    scenario["std_truth_posterior_predictive_realisations"] = jnp.sqrt(
        jnp.exp(truth_posterior_predictive_realisations)
    )
    scenario["std_bias_posterior_predictive_realisations"] = jnp.sqrt(
        jnp.exp(bias_posterior_predictive_realisations)
    )
    scenario["std_climate_posterior_predictive_realisations"] = jnp.sqrt(
        jnp.exp(climate_posterior_predictive_realisations)
    )

def generate_meanfunction_posterior_predictions(
    nele, nlat, scenario, num_realisations,
):
    posterior = scenario["mcmc"].posterior
    mt_predictions = []
    lvt_predictions = []
    mb_predictions = []
    lvb_predictions = []

    for i in np.random.randint(posterior.draw.shape, size=num_realisations):
        posterior_param_realisation = {
            "mt_mean_bo_realisation": posterior["mt_mean_bo"].data[0, :][i],
            "mt_mean_b1_realisation": posterior["mt_mean_b1"].data[0, :][i],
            "mt_mean_b2_realisation": posterior["mt_mean_b2"].data[0, :][i],
            "mb_mean_bo_realisation": posterior["mb_mean_bo"].data[0, :][i],
            "mb_mean_b1_realisation": posterior["mb_mean_b1"].data[0, :][i],
            "lvt_mean_bo_realisation": posterior["lvt_mean_bo"].data[0, :][i],
            "lvt_mean_b1_realisation": posterior["lvt_mean_b1"].data[0, :][i],
            "lvt_mean_b2_realisation": posterior["lvt_mean_b2"].data[0, :][i],
            "lvb_mean_bo_realisation": posterior["lvb_mean_bo"].data[0, :][i],
            "lvb_mean_b1_realisation": posterior["lvb_mean_b1"].data[0, :][i],
        }

        mt_prediction = (posterior_param_realisation['mt_mean_bo_realisation'] + 
                         posterior_param_realisation['mt_mean_b1_realisation']*nele +
                         posterior_param_realisation['mt_mean_b2_realisation']*nlat
        )
        lvt_prediction = (posterior_param_realisation['lvt_mean_bo_realisation'] + 
                         posterior_param_realisation['lvt_mean_b1_realisation']*nele +
                         posterior_param_realisation['lvt_mean_b2_realisation']*nlat
        )
        mb_prediction = (posterior_param_realisation['mb_mean_bo_realisation'] + 
                         posterior_param_realisation['mb_mean_b1_realisation']*nele
        )
        lvb_prediction = (posterior_param_realisation['lvb_mean_bo_realisation'] + 
                         posterior_param_realisation['lvb_mean_b1_realisation']*nele
        )

        mt_predictions.append(mt_prediction)
        lvt_predictions.append(lvt_prediction)
        mb_predictions.append(mb_prediction)
        lvb_predictions.append(lvb_prediction)

    mt_predictions = jnp.array(mt_predictions)
    lvt_predictions = jnp.array(lvt_predictions)
    mb_predictions = jnp.array(mb_predictions)
    lvb_predictions = jnp.array(lvb_predictions)

    scenario[
        "meanfunction_posterior_predictions_mt"
    ] = mt_predictions
    scenario[
        "meanfunction_posterior_predictions_lvt"
    ] = lvt_predictions
    scenario[
        "meanfunction_posterior_predictions_mb"
    ] = mb_predictions
    scenario[
        "meanfunction_posterior_predictions_lvb"
    ] = lvb_predictions
