# %% Importing Packages
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import mpl_scatter_density
import numpy as np
import skgstat as skg
from sklearn import preprocessing
import pandas as pd
from pyproj import Transformer
import numpyro.distributions as dist

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from src.helper_functions import grid_coords_to_2d_latlon_coords
from src.helper_functions import create_mask
from src.simulated_data_functions import generate_posterior

import jax.numpy as jnp
from jax import random
rng_key = random.PRNGKey(0)

# %% Importing data
base_path = '/home/jez/'
aws_inpath = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Processed/df_all_combined_75_mean_filtered_residuals_winter.csv'
cordex_inpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined_Residuals_Winter.nc'

df_aws = pd.read_csv(aws_inpath)
ds_climate = xr.open_dataset(cordex_inpath)
ds_climate = ds_climate.coarsen(grid_latitude=28,grid_longitude=28).mean()

# %% Importing ice sheet shapefile for plotting and mask purposes
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
icehsheet_shapefile = f'{shapefiles_path}/cst10_polygon.shp'

gdf_icesheet = gpd.read_file(icehsheet_shapefile)
gdf_icesheet = gdf_icesheet.explode().iloc[[61]]
gdf_icesheet = gdf_icesheet.reset_index().drop(columns=['level_0','level_1'])

##### glon,glat #####
rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)

# %% Creating ice sheet mask that excludes islands
map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)
ism_mask = create_mask(ds_climate.drop_dims('Model'),gdf_icesheet,map_proj)

ism_mask_fudgefix = ism_mask * ds_climate['LSM'].sel(Model='MAR(ERA5)').data

ds_climate['ISM'] = (('grid_latitude', 'grid_longitude'), ism_mask_fudgefix)

# %% Plotting Truth and Bias Fields from Observational Data Locations for MAR(ERA5)
fig, axs = plt.subplots(1, 3, figsize=(10, 5),dpi=600)

mean = 0
RCM = 'MAR(ERA5)'

ds_climate_masked = ds_climate.where(ds_climate.ISM)
ds_climate_masked['Mean Temperature Residual'].sel(Model=RCM).plot.pcolormesh(
    ax=axs[0],
    x='glon',
    y='glat',
    vmin=mean-4,
    vmax=mean+4,
    cmap = 'RdBu',
    add_colorbar=False,
)
axs[0].set_title('Climate Model')

plot = df_aws.plot.scatter(
    x='glon',
    y='glat',
    c=f'Mean Temperature Residual {RCM}',
    ax=axs[1],
    vmin=mean-4,
    vmax=mean+4,
    colorbar=False,
    cmap = 'RdBu',
    edgecolor='k'
    )
axs[1].set_title('Observations')

plot = df_aws.plot.scatter(
    x='glon',
    y='glat',
    c=f'Difference in Residuals {RCM}',
    ax=axs[2],
    vmin=mean-4,
    vmax=mean+4,
    colorbar=False,
    cmap = 'RdBu',
    edgecolor='k'
    )
axs[2].set_title('Difference (Nearest)')

for ax in axs.ravel():
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

plt.tight_layout()


# %% Reformatting data for model
ds_climate_stacked = ds_climate_masked.stack(X=('grid_latitude', 'grid_longitude'))
ds_climate_stacked = ds_climate_stacked.dropna('X')

cx = jnp.array(np.dstack([ds_climate_stacked['glon'],ds_climate_stacked['glat']])[0])
cdata = jnp.array(ds_climate_stacked['Mean Temperature Residual'].sel(Model='MAR(ERA5)').values)
ox = jnp.array(df_aws[['glon','glat']].values)
odata = jnp.array(df_aws['Mean Temperature Residual MAR(ERA5)'].values)

glon_min,glon_max = ds_climate_stacked['glon'].min().values,ds_climate_stacked['glon'].max().values
glat_min,glat_max = ds_climate_stacked['glat'].min().values,ds_climate_stacked['glat'].max().values
nglon,nglat = np.meshgrid(jnp.linspace(glon_min,glon_max,21),jnp.linspace(glat_min,glat_max,21))
nx = np.dstack([nglon.ravel(),nglat.ravel()])[0]

# %% Creating dictionary to work with
scenario_real_residuals = {
    'cx':cx,
    'cdata':cdata,
    'ox':ox,
    'odata':odata,
    'nx':nx,
    'nx1':nglon,
    'nx2':nglat,
    'cnoise': 1e-3,
    'jitter': 1e-10,
    't_variance_prior': dist.Gamma(1.0,1.5),
    't_lengthscale_prior': dist.Gamma(3.0,0.2),
    't_mean_prior': dist.Normal(0.0,2.0),
    'b_variance_prior': dist.Gamma(1.0,0.5),
    'b_lengthscale_prior': dist.Gamma(3.0,0.2),
    'b_mean_prior': dist.Normal(0.0, 2.0),
    'onoise_prior':dist.Uniform(0.0,0.5),
    'cnoise_prior':dist.Uniform(0.0,0.5),
}

# %% Fitting the model and producing the posterior
generate_posterior(scenario_real_residuals,rng_key,1000,2000,1)

# %% Saving the output
scenario_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real_residuals.npy'
np.save(scenario_outpath, scenario_real_residuals)


# %%
# stacked_climate_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined_Residuals_Winter_LandOnly_Stacked.nc'
stacked_climate_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined_Residuals_Summer_LandOnly_Stacked.nc'
ds_climate_stacked.to_netcdf(stacked_climate_outpath)
# %%
X1_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/X1.npy'
X2_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/X2.npy'
np.save(X1_outpath, nglon)
np.save(X2_outpath, nglat)

# %%
