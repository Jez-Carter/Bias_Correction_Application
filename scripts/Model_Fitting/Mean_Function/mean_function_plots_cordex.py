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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from src.helper_functions import grid_coords_to_2d_latlon_coords
from src.helper_functions import create_mask

# %%
def correlation(x,y):
    return(np.cov(x, y)[0,1] / (np.std(x) * np.std(y)))

# %% Loading the data
base_path = '/home/jez/'
aws_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Processed/df_all_combined_75_mean_filtered_residuals_january.csv'
cordex_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined_Residuals_Jan.nc'
ds = xr.open_dataset(cordex_path)
ds = ds.where(ds.LSM) 
ds_stacked = ds.stack(X=('grid_latitude', 'grid_longitude'))
df_aws_mean = pd.read_csv(aws_path)

##### glon,glat #####
rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
gdf_icesheet = gpd.read_file(icehsheet_shapefile)
gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)
# gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

# %% Scatter plots of mean prediction against the actual mean
fig, axs = plt.subplots(2, 4, figsize=(20, 10),dpi=600)

vmin,vmax = ds['Temperature'].min(),ds['Temperature'].max()

for ax,RCM in zip(axs.ravel(),ds.Model.data):
    ds_stacked_rcm = ds_stacked.sel(Model=RCM)
    ds_stacked_rcm.plot.scatter(
        'Temperature','Mean Temperature Pred',
        marker='x',
        alpha=0.05,
        ax=ax)
    df_aws_mean.plot.scatter(
        'Mean_Temperature',f'Mean Temperature Pred {RCM}',
        marker='x',
        alpha=1.0,
        color='tab:orange',
        ax=ax)

    ax.plot(np.arange(vmin,vmax,1),np.arange(vmin,vmax,1),linestyle='--',color='tab:orange')
    ax.set_ylim([vmin,vmax])
    ax.set_xlim([vmin,vmax])

# %% Scatter plots for residuals from climate model and observations
fig, axs = plt.subplots(2, 4, figsize=(20, 10),dpi=600)

residual_columns = [col for col in df_aws_mean.columns if 'Mean Temperature Residual' in col]
df_aws_mean[residual_columns].min().min()
vmin,vmax = df_aws_mean[residual_columns].min().min(),df_aws_mean[residual_columns].max().max()

for ax,RCM in zip(axs.ravel(),ds.Model.data):
    df_aws_mean.plot.scatter(x=f'Mean Temperature Residual {RCM}',
                             y=f'Mean Temperature Residual (NN Climate Model) {RCM}',
                             ax=ax)
    ax.plot(np.arange(vmin,vmax,1),np.arange(vmin,vmax,1),linestyle='--',color='tab:orange')
    cor = correlation(df_aws_mean[f'Mean Temperature Residual {RCM}'],df_aws_mean[f'Mean Temperature Residual (NN Climate Model) {RCM}'])
    ax.text(0.1,0.9,rf'$\rho=${round(cor,2)}',transform=ax.transAxes)
    ax.set_ylim([vmin,vmax])
    ax.set_xlim([vmin,vmax])
plt.tight_layout()

# %%
np.cov(df_aws_mean[f'Mean Temperature Residual {RCM}'],df_aws_mean[f'Mean Temperature Residual (NN Climate Model) {RCM}'])[0,1]


# %% Scatter plot for residual from observations and difference in residuals of nearest climate model output
fig, axs = plt.subplots(2, 4, figsize=(20, 10),dpi=600)

residual_columns = [col for col in df_aws_mean.columns if 'Mean Temperature Residual' in col]
residual_columns = [col for col in residual_columns if '(NN Climate Model)' not in col]
df_aws_mean[residual_columns].min().min()
vmin,vmax = df_aws_mean[residual_columns].min().min(),df_aws_mean[residual_columns].max().max()

for ax,RCM in zip(axs.ravel(),ds.Model.data):
    df_aws_mean.plot.scatter(x=f'Mean Temperature Residual {RCM}',
                             y=f'Difference in Residuals {RCM}',
                             ax=ax)
    # ax.plot(np.arange(vmin,vmax,1),np.arange(vmin,vmax,1),linestyle='--',color='tab:orange')
    cor = correlation(df_aws_mean[f'Mean Temperature Residual {RCM}'],df_aws_mean[f'Difference in Residuals {RCM}'])
    ax.text(0.1,0.9,rf'$\rho=${round(cor,2)}',transform=ax.transAxes)
    ax.set_ylim([vmin,vmax])
    ax.set_xlim([vmin,vmax])

plt.tight_layout()

# %% Plotting climate model residuals on map
fig, axs = plt.subplots(2, 4, figsize=(20, 7),dpi=600)

for ax,RCM in zip(axs.ravel(),ds.Model.data):
    ds_rcm = ds.sel(Model=RCM)
    ds_rcm['Mean Temperature Residual'].where(ds_rcm.LSM).plot.pcolormesh(
        x='glon',
        y='glat',
        ax=ax,
        alpha=0.8,
        vmin=-4,
        vmax=4,
        # edgecolors='k',
        linewidths=0.01,
        cmap = 'RdBu'
        )

    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

plt.tight_layout()

# %% Plotting Truth and Bias Fields from Observational Data Locations for MAR(ERA5)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5),dpi=600)

RCM = 'MAR(ERA5)'

mean = df_aws_mean[f'Mean Temperature Residual {RCM}'].mean()
plot = df_aws_mean.plot.scatter(
    x='glon',
    y='glat',
    c=f'Mean Temperature Residual {RCM}',
    ax=axs[0],
    vmin=mean-4,
    vmax=mean+4,
    colorbar=False,
    cmap = 'RdBu',
    edgecolor='k'
    )
axs[0].set_title('Truth Field')

mean = df_aws_mean[f'Difference in Residuals {RCM}'].mean()
plot = df_aws_mean.plot.scatter(
    x='glon',
    y='glat',
    c=f'Difference in Residuals {RCM}',
    ax=axs[1],
    vmin=mean-4,
    vmax=mean+4,
    colorbar=False,
    cmap = 'RdBu',
    edgecolor='k'
    )
axs[1].set_title('Bias Field')

for ax in axs:
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

# %% Plotting Truth and Bias Fields from Observational Data Locations for Ensemble of MAR(ERA5), RACMO(ERA5) and MetUM(011)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5),dpi=600)

ensemble = ['MAR(ERA5)','RACMO(ERA5)','MetUM(011)']

ensemble_residuals = []
ensemble_residuals_difference = []
for RCM in ensemble:
    ensemble_residuals.append(df_aws_mean[f'Mean Temperature Residual {RCM}'])
    ensemble_residuals_difference.append(df_aws_mean[f'Difference in Residuals {RCM}'])
df_aws_mean[f'Mean Temperature Residual Ensemble'] = np.array(ensemble_residuals).mean(axis=0)
df_aws_mean[f'Difference in Residuals Ensemble'] = np.array(ensemble_residuals_difference).mean(axis=0)

mean = df_aws_mean[f'Mean Temperature Residual Ensemble'].mean()
plot = df_aws_mean.plot.scatter(
    x='glon',
    y='glat',
    c=f'Mean Temperature Residual Ensemble',
    ax=axs[0],
    vmin=mean-4,
    vmax=mean+4,
    colorbar=False,
    cmap = 'RdBu',
    edgecolor='k'
    )
axs[0].set_title('Truth Field')

mean = df_aws_mean[f'Difference in Residuals Ensemble'].mean()
plot = df_aws_mean.plot.scatter(
    x='glon',
    y='glat',
    c=f'Difference in Residuals Ensemble',
    ax=axs[1],
    vmin=mean-4,
    vmax=mean+4,
    colorbar=False,
    cmap = 'RdBu',
    edgecolor='k'
    )
axs[1].set_title('Bias Field')

for ax in axs:
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

# %% Plotting bias (difference in residuals) on map
from matplotlib.cm import ScalarMappable

fig, axs = plt.subplots(2, 4, figsize=(20, 7),dpi=600)

for ax,RCM in zip(axs.ravel(),ds.Model.data):
    mean = df_aws_mean[f'Difference in Residuals {RCM}'].mean()
    plot = df_aws_mean.plot.scatter(
        x='glon',
        y='glat',
        c=f'Difference in Residuals {RCM}',
        ax=ax,
        vmin=mean-4,
        vmax=mean+4,
        colorbar=False,
        cmap = 'RdBu',
        edgecolor='k'
        )

    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

# scales = np.linspace(-4, 4, 8)
# norm = plt.Normalize(scales.min(), scales.max())
# sm =  ScalarMappable(norm=norm, cmap='RdBu')
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=axs[:,3])
# cbar.ax.set_title("scale")

plt.tight_layout()

# %% Plotting observation residual values on map
from matplotlib.cm import ScalarMappable

fig, axs = plt.subplots(2, 4, figsize=(20, 7),dpi=600)

for ax,RCM in zip(axs.ravel(),ds.Model.data):
    mean = df_aws_mean[f'Mean Temperature Residual {RCM}'].mean()
    plot = df_aws_mean.plot.scatter(
        x='glon',
        y='glat',
        c=f'Mean Temperature Residual {RCM}',
        ax=ax,
        vmin=mean-4,
        vmax=mean+4,
        colorbar=False,
        cmap = 'RdBu',
        edgecolor='k'
        )

    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

plt.tight_layout()

