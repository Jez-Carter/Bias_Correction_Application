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

# %% Loading the Climate Model Data and filtering to Jan
base_path = '/home/jez/'
climate_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/ProcessedData/MetUM_Reformatted.nc'
# climate_path = f'{base_path}Bias_Correction_Application/MAR3.11_TT_2010_daily_reformatted.nc' #MAR Data
ds_climate = xr.open_dataset(climate_path)
ds_climate = ds_climate.assign_coords({"month": (ds_climate.time.dt.month)})
ds_climate = ds_climate.assign_coords({"year": (ds_climate.time.dt.year)})
ds_climate = ds_climate.where(ds_climate.month==1,drop=True)
# ds_climate = ds_climate.where(ds_climate.year==2010,drop=True)
ds_climate_mean = ds_climate.mean('time')
ds_climate_stacked_mean = ds_climate_mean.stack(X=('grid_latitude', 'grid_longitude'))

# %% Fitting the Model to the Climate Model Data (Land only)
x1_train = ds_climate_stacked_mean.orog.where(ds_climate_stacked_mean.lsm,drop=True)
x2_train = ds_climate_stacked_mean.latitude.where(ds_climate_stacked_mean.lsm,drop=True)
Xtrain = np.dstack([x1_train,x2_train]).reshape(-1,2)
ytrain = ds_climate_stacked_mean['tas'].where(ds_climate_stacked_mean.lsm,drop=True)

model = LinearRegression()
poly_features = PolynomialFeatures(degree=2)#, include_bias=False)
Xtrain_poly = poly_features.fit_transform(Xtrain)
model.fit(Xtrain_poly, ytrain)

x1 = ds_climate_stacked_mean.orog
x2 = ds_climate_stacked_mean.latitude
X = np.dstack([x1,x2]).reshape(-1,2)
X_poly = poly_features.fit_transform(X)
y_pred = model.predict(X_poly)
ds_climate_stacked_mean['Mean Temperature Pred'] = (('X'), y_pred)
ds_climate_stacked_mean['Mean Temperature Residual'] = ds_climate_stacked_mean['tas']-ds_climate_stacked_mean['Mean Temperature Pred']
ds_climate_mean = ds_climate_stacked_mean.unstack()

# %%
# aws_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Processed/df_all_combined.csv'
# df_aws = pd.read_csv(aws_path)
# aws_path_75 = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Processed/df_all_combined_75.csv'
# df_aws_75 = pd.read_csv(aws_path_75)

# %% Applying the same mean function to the AWS data
aws_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Processed/df_all_combined_75.csv'
df_aws = pd.read_csv(aws_path)
df_aws = df_aws[df_aws['Month']==1]
# df_aws = df_aws[df_aws['Year']==2010]
df_aws = df_aws[df_aws['Elevation(m)'].isnull()==False]

index_group = ['Station','Lat(℃)','Lon(℃)','Elevation(m)','Institution']
df_aws_mean = df_aws.groupby(index_group).agg(
    Mean_Temperature=('Temperature', 'mean'),
    Temperature_Records=('Temperature', 'count')
    )
df_aws_mean = df_aws_mean.reset_index()
df_aws_mean = df_aws_mean[df_aws_mean['Temperature_Records']>100] # Filtering by # of Records

x1 = df_aws_mean['Elevation(m)']
x2 = df_aws_mean['Lat(℃)']
X = np.dstack([x1,x2]).reshape(-1,2)
X_poly = poly_features.fit_transform(X)
y_pred = model.predict(X_poly)
df_aws_mean['Mean Temperature Pred']=y_pred
df_aws_mean['Mean Temperature Residual']=df_aws_mean['Mean_Temperature']-df_aws_mean['Mean Temperature Pred']

# %% Including glon,glat,grid_longitude and grid_latitude coordinates in AWS data
lon=df_aws_mean['Lon(℃)']
lat=df_aws_mean['Lat(℃)']

##### glon,glat #####
rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
transformer = Transformer.from_crs("epsg:4326", rotated_coord_system)
#NOTE epsg:4326 takes (latitude,longitude) whereas rotated_coord_system takes (rotated long,rotated lat)
#The change in order of the axes is important and reflected in the below line
glon,glat = transformer.transform(lat,lon)
df_aws_mean['glon']=glon
df_aws_mean['glat']=glat

##### grid_longitude,grid_latitude #####
climate_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=0.0,
    globe=None,
)
def convert_180_to_360(long_value):
    if long_value<0:
        return(long_value+360)
    else:
        return(long_value)
#Including climate coordinate system useful for finding nearest points etc
transformer = Transformer.from_crs("epsg:4326", climate_coord_system)
#NOTE epsg:4326 takes (latitude,longitude) whereas rotated_coord_system takes (rotated long,rotated lat)
#The change in order of the axes is important and reflected in the below line
grid_longitude,grid_latitude = transformer.transform(lat,lon)
grid_longitude_360 = [convert_180_to_360(i) for i in grid_longitude]
df_aws_mean['grid_longitude']=grid_longitude_360
df_aws_mean['grid_latitude']=grid_latitude

# %% Computing the difference (bias) in the mean residual for each AWS
aws_grid_longitudes = df_aws_mean['grid_longitude'].values
aws_grid_latitudes = df_aws_mean['grid_latitude'].values
ds_climate_nearest = ds_climate_mean.sel(grid_longitude=aws_grid_longitudes,grid_latitude=aws_grid_latitudes,method='nearest')
ds_climate_nearest_residual_values = np.diag(ds_climate_nearest['Mean Temperature Residual'].data,k=0)
df_aws_mean['Mean Temperature Residual (NN Climate Model)'] = ds_climate_nearest_residual_values
df_aws_mean['Difference in Residuals'] = ds_climate_nearest_residual_values - df_aws_mean['Mean Temperature Residual']

# %% Importing ice sheet shapefile for plotting purposes
base_path = '/home/jez/'
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
gdf_icesheet = gpd.read_file(icehsheet_shapefile)

gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)
# gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

# %% Plotting the prediction against the actual mean
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=600)

ds_climate_stacked_mean.where(ds_climate_stacked_mean.lsm,drop=True).plot.scatter(
    'tas','Mean Temperature Pred',
    marker='x',
    alpha=0.2,
    ax=ax)
df_aws_mean.plot.scatter(
    'Mean_Temperature','Mean Temperature Pred',
    marker='x',
    alpha=0.5,
    color='tab:orange',
    ax=ax)

# %% Figure Residuals for the Climate Model Data and AWS Data
fig, ax = plt.subplots(1, 1, figsize=(10, 8),dpi=600)

ds_climate_mean_landonly = ds_climate_mean.where(ds_climate_mean.lsm)#,drop=True)
ds_climate_mean_landonly['Mean Temperature Residual'].plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.8,
    vmin=-4,
    vmax=4,
    edgecolors='k',
    linewidths=0.01,
    cmap = 'RdBu')

df_aws_mean.plot.scatter(
    x='glon',
    y='glat',
    c='Mean Temperature Residual',
    ax=ax,
    vmin=-4,
    vmax=4,
    colorbar=False,
    cmap = 'RdBu',
    edgecolor='k'
)

gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

# %% Figure Visualising Truth and Bias Fields
fig, axs = plt.subplots(1, 2, figsize=(10, 3.33),dpi=600)

df_aws_mean.plot.scatter(
    x='glon',
    y='glat',
    c='Mean Temperature Residual',
    ax=axs[0],
    vmin=-4,
    vmax=4,
    cmap = 'RdBu',
    edgecolor='k'
)

df_aws_mean.plot.scatter(
    x='glon',
    y='glat',
    c='Difference in Residuals',
    ax=axs[1],
    vmin=-4,
    vmax=4,
    cmap = 'RdBu',
    edgecolor='k',
)

for ax in axs:
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

plt.tight_layout()

# %% Figure Visualising Truth and Bias Fields
fig, ax = plt.subplots(1, 1, figsize=(10, 8),dpi=600)

df_aws_mean.plot.scatter(
    x='glon',
    y='glat',
    c='Difference in Residuals',
    ax=ax,
    vmin=-4,
    vmax=4,
    cmap = 'RdBu',
    edgecolor='k',
)

gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.3)

plt.tight_layout()

# %% Truth Against Bias correlation

df_aws_mean.plot.scatter(x='Mean Temperature Residual',
                         y='Difference in Residuals')
plt.plot(np.arange(-7,7,1),-np.arange(-7,7,1),linestyle='--',alpha=0.5)


# %%
df_aws_mean
Mean Temperature Residual (NN Climate Model)
# %%

df_aws_mean.plot.scatter(x='Mean Temperature Residual',
                         y='Mean Temperature Residual (NN Climate Model)')
# plt.plot(np.arange(-7,7,1),-np.arange(-7,7,1),linestyle='--',alpha=0.5)



# %%
ds_climate_nearest_elevation_values = np.diag(ds_climate_nearest['orog'].data,k=0)
df_aws_mean['Elevation (Climate Model)'] = ds_climate_nearest_elevation_values

df_aws_mean.plot.scatter(x='Elevation(m)',
                         y='Elevation (Climate Model)')
plt.plot(np.arange(0,4500,500),np.arange(0,4500,500),linestyle='--',alpha=0.5)

# %%
# %% Figure Residuals for the Climate Model Data and AWS Data
# fig, ax = plt.subplots(1, 1, figsize=(10, 8),dpi=600)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.33),dpi=600)

ds_climate_mean_landonly = ds_climate_mean.where(ds_climate_mean.lsm)#,drop=True)
ds_climate_mean_landonly['Mean Temperature Residual'].plot.pcolormesh(
    x='glon',
    y='glat',
    ax=axs[0],
    alpha=0.8,
    vmin=-6,
    vmax=6,
    edgecolors='k',
    linewidths=0.01,
    cmap = 'RdBu',
    add_colorbar=False)

df_aws_mean.plot.scatter(
    x='glon',
    y='glat',
    c='Mean Temperature Residual',
    ax=axs[1],
    vmin=-6,
    vmax=6,
    colorbar=False,
    cmap = 'RdBu',
    edgecolor='k'
)

gdf_icesheet_rotatedcoords.boundary.plot(ax=axs[0], color="k", linewidth=0.3)
gdf_icesheet_rotatedcoords.boundary.plot(ax=axs[1], color="k", linewidth=0.3)