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

# %% Specifying month and data range
month = 1 #January
year_min,year_max = 2010,2018

# %% Loading and filtering the Climate Model Data
base_path = '/home/jez/'
CORDEX_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'
ds_climate = xr.open_dataset(CORDEX_path)
ds_climate['Temperature']=ds_climate['Temperature'] - 273.15

month_condition = ds_climate.month==month
year_condition = (ds_climate.year>=year_min)&(ds_climate.year<=year_max)
ds_climate = ds_climate.where(month_condition&year_condition,drop=True)
ds_climate_mean = ds_climate.mean('Time')
ds_climate_stacked_mean = ds_climate_mean.stack(X=('grid_latitude', 'grid_longitude'))

# %% Loading and filtering the AWS data
aws_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Processed/df_all_combined_75.csv'
df_aws = pd.read_csv(aws_path)

month_condition = df_aws['Month']==month
year_condition = (year_min<=df_aws['Year'])&(df_aws['Year']<=year_max)
df_aws = df_aws[month_condition&year_condition]
df_aws = df_aws[df_aws['Temperature'].isnull()==False]
df_aws = df_aws[df_aws['Elevation(m)'].isnull()==False]

index_group = ['Station','Lat(℃)','Lon(℃)','Elevation(m)','Institution','glat','glon','grid_latitude', 'grid_longitude']
df_aws_mean = df_aws.groupby(index_group).agg(
    Mean_Temperature=('Temperature', 'mean'),
    Temperature_Records=('Temperature', 'count')
    )
df_aws_mean = df_aws_mean.reset_index()
df_aws_mean = df_aws_mean[df_aws_mean['Temperature_Records']>100] # Filtering by # of Records

# %% Defining mean function
def fit_mean_function(ds_stacked_mean):
    x1_train = ds_stacked_mean.Elevation.where(ds_stacked_mean.LSM,drop=True)
    x2_train = ds_stacked_mean.latitude.where(ds_stacked_mean.LSM,drop=True)
    Xtrain = np.dstack([x1_train,x2_train]).reshape(-1,2)
    ytrain = ds_stacked_mean['Temperature'].where(ds_stacked_mean.LSM,drop=True)

    model = LinearRegression()
    poly_features = PolynomialFeatures(degree=2)#, include_bias=False)
    Xtrain_poly = poly_features.fit_transform(Xtrain)
    model.fit(Xtrain_poly, ytrain)

    x1 = ds_stacked_mean.Elevation
    x2 = ds_stacked_mean.latitude
    X = np.dstack([x1,x2]).reshape(-1,2)
    X_poly = poly_features.fit_transform(X)
    y_pred = model.predict(X_poly)
    ds_stacked_mean['Mean Temperature Pred'] = (('X'), y_pred)
    ds_stacked_mean['Mean Temperature Residual'] = ds_stacked_mean['Temperature']-ds_stacked_mean['Mean Temperature Pred']
    
    return(ds_stacked_mean,model,poly_features)

# %% Fitting a mean function for each climate model
mean_functions = {}
ds_stacked_list = []
for RCM in ds_climate.Model.data:
    ds_stacked,model,poly_features = fit_mean_function(ds_climate_stacked_mean.sel(Model=RCM))
    mean_functions[f'{RCM}']={'mean_model':model,'poly_features':poly_features}
    ds_stacked_list.append(ds_stacked)
    print(RCM,model.intercept_, model.coef_)
ds_stacked = xr.concat(ds_stacked_list, 'Model')
ds = ds_stacked.unstack()

# %% Applying the same mean function to the AWS data
poly_features = mean_functions['ERAI']['poly_features']
x1 = df_aws_mean['Elevation(m)']
x2 = df_aws_mean['Lat(℃)']
X = np.dstack([x1,x2]).reshape(-1,2)
X_poly = poly_features.fit_transform(X)

for RCM in list(mean_functions.keys()):
    model = mean_functions[RCM]['mean_model']
    y_pred = model.predict(X_poly)
    df_aws_mean[f'Mean Temperature Pred {RCM}']=y_pred
    df_aws_mean[f'Mean Temperature Residual {RCM}']=df_aws_mean['Mean_Temperature']-df_aws_mean[f'Mean Temperature Pred {RCM}']

# %% Computing the difference (bias) in the mean residual for each AWS
aws_grid_longitudes = df_aws_mean['grid_longitude'].values
aws_grid_latitudes = df_aws_mean['grid_latitude'].values
for RCM in ds.Model.data:
    ds_rcm = ds.sel(Model=RCM)
    ds_nearest = ds_rcm.sel(grid_longitude=aws_grid_longitudes,grid_latitude=aws_grid_latitudes,method='nearest')
    ds_nearest_residual_values = np.diag(ds_nearest['Mean Temperature Residual'].data,k=0)
    df_aws_mean[f'Mean Temperature Residual (NN Climate Model) {RCM}'] = ds_nearest_residual_values
    df_aws_mean[f'Difference in Residuals {RCM}'] = ds_nearest_residual_values - df_aws_mean[f'Mean Temperature Residual {RCM}']

# %% Saving the output
aws_outpath = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Processed/df_all_combined_75_mean_filtered_residuals_january.csv'
df_aws_mean.to_csv(aws_outpath,index=False)
cordex_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined_Residuals_Jan.nc'
ds.to_netcdf(cordex_outpath)
# %%
