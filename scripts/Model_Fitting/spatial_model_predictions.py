
# %% Importing Packages
import numpy as np
import jax
from jax import random
import arviz as az
import cartopy.crs as ccrs
import geopandas as gpd
import jax.numpy as jnp
import xarray as xr
from src.helper_functions import create_mask
import matplotlib.pyplot as plt
plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.linewidth'] = 0.4

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)

from src.simulated_data_functions import plot_underlying_data_1d
from src.simulated_data_functions import generate_posterior_predictive_realisations
from src.simulated_data_functions import plot_predictions_1d
from src.simulated_data_functions import plot_predictions_2d
from src.simulated_data_functions import create_levels

inpath = '/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Scenarios/'

# %% Loading scenario data
base_path = '/home/jez/'
scenariopath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real_residuals.npy'
scenario = np.load(scenariopath,allow_pickle='TRUE').item()

az.summary(scenario['mcmc'].posterior,hdi_prob=0.95)

# %% Updating prediction locations
cordex_inpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined_Residuals_Winter.nc'
ds_climate = xr.open_dataset(cordex_inpath)
ds_climate = ds_climate.coarsen(grid_latitude=7,grid_longitude=7).mean()

shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
icehsheet_shapefile = f'{shapefiles_path}/cst10_polygon.shp'
gdf_icesheet = gpd.read_file(icehsheet_shapefile)
gdf_icesheet = gdf_icesheet.explode().iloc[[61]]
gdf_icesheet = gdf_icesheet.reset_index().drop(columns=['level_0','level_1'])
rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)

map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)
ism_mask = create_mask(ds_climate.drop_dims('Model'),gdf_icesheet,map_proj)
ism_mask_fudgefix = ism_mask * ds_climate['LSM'].sel(Model='MAR(ERA5)').data
ds_climate['ISM'] = (('grid_latitude', 'grid_longitude'), ism_mask_fudgefix)
ds_climate_stacked = ds_climate.stack(X=('grid_latitude', 'grid_longitude'))
glon_min,glon_max = ds_climate_stacked['glon'].min().values,ds_climate_stacked['glon'].max().values
glat_min,glat_max = ds_climate_stacked['glat'].min().values,ds_climate_stacked['glat'].max().values
nglon,nglat = np.meshgrid(jnp.linspace(glon_min,glon_max,21),jnp.linspace(glat_min,glat_max,21))

nglon,nglat = ds_climate['glon'],ds_climate['glat']
nx = np.dstack([nglon.data.ravel(),nglat.data.ravel()])[0]

# nx1,nx2 = np.meshgrid(jnp.linspace(-25,25,21),jnp.linspace(-25,25,21))
# nx = np.dstack([nx1.ravel(),nx2.ravel()])[0]
scenario.update( 
    {'nx1': nglon,
     'nx2': nglat,
     'nx': nx}
)
# %%
nglon.data.ravel()

# %% Generating the posterior predictive distributions
generate_posterior_predictive_realisations(scenario,20,20)

# %% Function Creation

def plot_predictions_2d(scenario,axs):

    truth = scenario['truth_posterior_predictive_realisations']
    truth_mean = truth.mean(axis=0)
    truth_std = truth.std(axis=0)
    bias = scenario['bias_posterior_predictive_realisations']
    bias_mean = bias.mean(axis=0)
    bias_std = bias.std(axis=0)

    plots = []
    levels = create_levels(scenario,0.25,0,center=True)

    for ax,data in zip(axs[:,0],[truth_mean,bias_mean]):
        data = np.ma.array(data, mask = (ds_climate.ISM.data==False))
        plots.append(ax.contourf(scenario['nx1'],
                    scenario['nx2'],
                    data.reshape(scenario['nx1'].shape),
                    cmap='RdBu',
                    levels=levels
        ))

    for ax,data in zip(axs[:,1],[truth_std,bias_std]):
        data = np.ma.array(data, mask = (ds_climate.ISM.data==False))
        plots.append(ax.contourf(scenario['nx1'],
                    scenario['nx2'],
                    data.reshape(scenario['nx1'].shape),
                    cmap='viridis'
        ))

    for plot in plots:
        plt.colorbar(plot)

    for ax in axs.ravel():

        ax.scatter(scenario['ox'][:,0],
                    scenario['ox'][:,1],
                    s=30, marker='o', c="None",edgecolor='k',alpha=0.5)
        ax.scatter(scenario['cx'][:,0],
                    scenario['cx'][:,1],
                    s=30, marker='x', c="k",alpha=0.5)

# %%
truth = scenario['truth_posterior_predictive_realisations']
truth_mean = truth.mean(axis=0)
# %%
truth_mean.reshape(scenario['nx1'].shape).shape

# %%
np.ma.array(truth_mean, mask = ds_climate.ISM.data)

# %%
np.where(ds_climate.ISM.data)truth_mean,
# %%
nx.shape

# %% Importing ice sheet shapefile for plotting purposes
base_path = '/home/jez/'
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
gdf_icesheet = gpd.read_file(icehsheet_shapefile)

##### glon,glat #####
rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)

# %%
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
plot_predictions_2d(scenario,axs)

titles = ['a. Prediction Mean: Unbiased Process',
          'b. Prediction Standard Deviation: Unbiased Process',
          'c. Prediction Mean: Biased Process',
          'd. Prediction Standard Deviation: Biased Process']

for ax,title in zip(axs.ravel(),titles):
    ax.set_title(title,pad=3,loc='left',fontsize=10)
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)


# %%
axs[:,0]

for ax in axs[:,0]