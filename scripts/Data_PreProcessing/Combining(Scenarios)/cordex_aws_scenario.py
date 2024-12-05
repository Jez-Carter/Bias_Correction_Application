# %% Importing Packages
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import pandas as pd
import numpyro.distributions as dist
from sklearn.preprocessing import StandardScaler 
from src.helper_functions import create_mask
from src.helper_functions import create_aws_mask
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import random
rng_key = random.PRNGKey(0)

# %% Importing data
base_path = '/home/jez/'
aws_inpath = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Processed/df_all_combined_75_monthly.csv'
cordex_inpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
icehsheet_main_shapefile = f'{shapefiles_path}/cst10_polygon.shp'

df_aws = pd.read_csv(aws_inpath)
group = ['Station','Year','Month']
df_aws = df_aws[~df_aws[group].duplicated(keep=False)] # oddity in the datasource, needs correcting (send email)
df_aws_group = df_aws.set_index(group)
coords = ['Lat(℃)','Lon(℃)','Elevation(m)','glat','glon','grid_latitude', 'grid_longitude']
df_aws_coords = df_aws.set_index(['Station'])[coords]
df_aws_coords = df_aws_coords[~df_aws_coords.duplicated()]

vars = ['Temperature']
ds_aws = xr.merge([df_aws_coords.to_xarray(),df_aws_group[vars].to_xarray()])
ds_aws = ds_aws.set_coords(coords)
ds_aws_june = ds_aws.sel(Month=6)

gdf_icesheet_main = gpd.read_file(icehsheet_main_shapefile)
gdf_icesheet_main = gdf_icesheet_main.explode().iloc[[61]]
gdf_icesheet_main = gdf_icesheet_main#.reset_index().drop(columns=['level_0','level_1'])
map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)
aws_mask = create_aws_mask(ds_aws,gdf_icesheet_main,map_proj)
mainland_stations = ds_aws['Station'][aws_mask].data
stations_recordsfilter_june = ds_aws_june.where(ds_aws_june.count(['Year'])>5,drop=True)['Station'].data
stations_filter_june = mainland_stations[np.isin(mainland_stations,stations_recordsfilter_june)]
ds_aws_june_filtered = ds_aws_june.sel(Station=stations_filter_june)

ds_climate = xr.open_dataset(cordex_inpath)
ds_climate = ds_climate.set_coords(("LSM", "Elevation"))
ds_climate = ds_climate.sel(Model='MAR(ERA5)')
ds_climate['Temperature'] = ds_climate['Temperature']-273.15
ds_climate_june = ds_climate.where(ds_climate.month==6,drop=True)
ds_climate_june_stacked = ds_climate_june.stack(X=('grid_latitude', 'grid_longitude'))
ds_climate_june_stacked_landonly = ds_climate_june_stacked.where(ds_climate_june_stacked.LSM>0.8).dropna('X')

ds_climate_coarse = ds_climate.coarsen(grid_latitude=14,grid_longitude=14).mean()
ds_climate_coarse_june = ds_climate_coarse.where(ds_climate_coarse.month==6,drop=True)
ds_climate_coarse_june_stacked = ds_climate_coarse_june.stack(X=('grid_latitude', 'grid_longitude'))
ds_climate_coarse_june_stacked_landonly = ds_climate_coarse_june_stacked.where(ds_climate_coarse_june_stacked.LSM>0.8).dropna('X')

# climate_mask = create_mask(ds_climate_coarse_june[['grid_longitude','grid_latitude']],gdf_icesheet_main,map_proj)

# %% Formatting Data & Standardising Elevation and Latitude Data

ox = jnp.array(np.dstack([ds_aws_june_filtered['glon'],ds_aws_june_filtered['glat']]))[0]
odata = jnp.array(ds_aws_june_filtered.transpose()['Temperature'].values)
olat = jnp.array(ds_aws_june_filtered['Lat(℃)'].values)
oele = jnp.array(ds_aws_june_filtered['Elevation(m)'].values)

ele_scaler = StandardScaler() 
oele_scaled = ele_scaler.fit_transform(oele.reshape(-1,1))[:,0]
lat_scaler = StandardScaler() 
olat_scaled = lat_scaler.fit_transform(olat.reshape(-1,1))[:,0]
data_scaler = StandardScaler()
odata_scaled = data_scaler.fit_transform(odata.reshape(-1,1)).reshape(odata.shape)

cx = jnp.array(np.dstack([ds_climate_coarse_june_stacked_landonly['glon'],ds_climate_coarse_june_stacked_landonly['glat']])[0])
cdata = jnp.array(ds_climate_coarse_june_stacked_landonly['Temperature'].values)
clat = jnp.array(ds_climate_coarse_june_stacked_landonly['latitude'].values)
cele = jnp.array(ds_climate_coarse_june_stacked_landonly['Elevation'].values)

cele_scaled = ele_scaler.transform(cele.reshape(-1,1))[:,0]
clat_scaled = lat_scaler.transform(clat.reshape(-1,1))[:,0]
cdata_scaled = data_scaler.transform(cdata.reshape(-1,1)).reshape(cdata.shape)

ds_aws_june_filtered = ds_aws_june_filtered.assign_coords(Elevation_Scaled=("Station", oele_scaled))
ds_aws_june_filtered = ds_aws_june_filtered.assign_coords(Latitude_Scaled=("Station", olat_scaled))
ds_climate_coarse_june_stacked_landonly = ds_climate_coarse_june_stacked_landonly.assign_coords(Elevation_Scaled=("X", cele_scaled))
ds_climate_coarse_june_stacked_landonly = ds_climate_coarse_june_stacked_landonly.assign_coords(Latitude_Scaled=("X", clat_scaled))

#Sanity Check
print('Data Shapes: \n',
      f'ox.shape:{ox.shape} \n',
      f'oele_scaled.shape:{oele_scaled.shape} \n',
      f'olat_scaled.shape:{olat_scaled.shape} \n',
      f'odata.shape:{odata.shape} \n',
      f'odata_scaled.shape:{odata.shape} \n',
      f'cx.shape:{cx.shape} \n',
      f'cele_scaled.shape:{cele_scaled.shape} \n',
      f'clat_scaled.shape:{clat_scaled.shape} \n',
      f'cdata.shape:{cdata.shape} \n',
      f'cdata_scaled.shape:{cdata.shape} \n',
      )
print('Data Values: \n',
      f'ox: min={ox.min():.1f}, mean={ox.mean():.1f}, max={ox.max():.1f} \n',
      f'odata: min={np.nanmin(odata):.1f}, mean={np.nanmean(odata):.1f}, max={np.nanmax(odata):.1f} \n',
      f'odata_scaled: min={np.nanmin(odata_scaled):.1f}, mean={np.nanmean(odata_scaled):.1f}, max={np.nanmax(odata_scaled):.1f} \n',
      f'oele: min={oele.min():.1f}, mean={oele.mean():.1f}, max={oele.max():.1f} \n',
      f'olat: min={olat.min():.1f}, mean={olat.mean():.1f}, max={olat.max():.1f} \n',
      f'oele_scaled: min={oele_scaled.min():.1f}, mean={oele_scaled.mean():.1f}, max={oele_scaled.max():.1f} \n',
      f'olat_scaled: min={olat_scaled.min():.1f}, mean={olat_scaled.mean():.1f}, max={olat_scaled.max():.1f} \n',
      f'cx: min={cx.min():.1f}, mean={cx.mean():.1f}, max={cx.max():.1f} \n',
      f'cdata: min={np.nanmin(cdata):.1f}, mean={np.nanmean(cdata):.1f}, max={np.nanmax(cdata):.1f} \n',
      f'cdata_scaled: min={np.nanmin(cdata_scaled):.1f}, mean={np.nanmean(cdata_scaled):.1f}, max={np.nanmax(cdata_scaled):.1f} \n',
      f'cele: min={cele.min():.1f}, mean={cele.mean():.1f}, max={cele.max():.1f} \n',
      f'clat: min={clat.min():.1f}, mean={clat.mean():.1f}, max={clat.max():.1f} \n',
      f'cele_scaled: min={cele_scaled.min():.1f}, mean={cele_scaled.mean():.1f}, max={cele_scaled.max():.1f} \n',
      f'clat_scaled: min={clat_scaled.min():.1f}, mean={clat_scaled.mean():.1f}, max={clat_scaled.max():.1f} \n',
)

# %%

print('Useful Metrics for Priors: \n',
      f"""mean odata:
      min={np.nanmin(np.nanmean(odata,axis=0)):.1f},
      mean={np.nanmean(np.nanmean(odata,axis=0)):.1f},
      max={np.nanmax(np.nanmean(odata,axis=0)):.1f},
      var={np.nanvar(np.nanmean(odata,axis=0)):.1f},
      \n""",
      f"""logvar odata:
      min={np.nanmin(np.log(np.nanvar(odata,axis=0))):.1f},
      mean={np.nanmean(np.log(np.nanvar(odata,axis=0))):.1f},
      max={np.nanmax(np.log(np.nanvar(odata,axis=0))):.1f},
      var={np.nanvar(np.log(np.nanvar(odata,axis=0))):.1f},
      \n""",
      f"""mean cdata:
      min={np.nanmin(np.nanmean(cdata,axis=0)):.1f},
      mean={np.nanmean(np.nanmean(cdata,axis=0)):.1f},
      max={np.nanmax(np.nanmean(cdata,axis=0)):.1f},
      var={np.nanvar(np.nanmean(cdata,axis=0)):.1f},
      \n""",
      f"""logvar cdata:
      min={np.nanmin(np.log(np.nanvar(cdata,axis=0))):.1f},
      mean={np.nanmean(np.log(np.nanvar(cdata,axis=0))):.1f},
      max={np.nanmax(np.log(np.nanvar(cdata,axis=0))):.1f},
      var={np.nanvar(np.log(np.nanvar(cdata,axis=0))):.1f},
      \n""",
)

print('Useful Metrics for Priors Scaled: \n',
      f"""mean odata_scaled:
      min={np.nanmin(np.nanmean(odata_scaled,axis=0)):.1f},
      mean={np.nanmean(np.nanmean(odata_scaled,axis=0)):.1f},
      max={np.nanmax(np.nanmean(odata_scaled,axis=0)):.1f},
      var={np.nanvar(np.nanmean(odata_scaled,axis=0)):.1f},
      \n""",
      f"""logvar odata_scaled:
      min={np.nanmin(np.log(np.nanvar(odata_scaled,axis=0))):.1f},
      mean={np.nanmean(np.log(np.nanvar(odata_scaled,axis=0))):.1f},
      max={np.nanmax(np.log(np.nanvar(odata_scaled,axis=0))):.1f},
      var={np.nanvar(np.log(np.nanvar(odata_scaled,axis=0))):.1f},
      \n""",
      f"""mean cdata_scaled:
      min={np.nanmin(np.nanmean(cdata_scaled,axis=0)):.1f},
      mean={np.nanmean(np.nanmean(cdata_scaled,axis=0)):.1f},
      max={np.nanmax(np.nanmean(cdata_scaled,axis=0)):.1f},
      var={np.nanvar(np.nanmean(cdata_scaled,axis=0)):.1f},
      \n""",
      f"""logvar cdata_scaled:
      min={np.nanmin(np.log(np.nanvar(cdata_scaled,axis=0))):.1f},
      mean={np.nanmean(np.log(np.nanvar(cdata_scaled,axis=0))):.1f},
      max={np.nanmax(np.log(np.nanvar(cdata_scaled,axis=0))):.1f},
      var={np.nanvar(np.log(np.nanvar(cdata_scaled,axis=0))):.1f},
      \n""",
)

# %% Creating Scenario

scenario_real = {
    'ds_aws_june_filtered':ds_aws_june_filtered,
    'ds_climate_coarse_june_stacked_landonly':ds_climate_coarse_june_stacked_landonly,
    'ds_climate_coarse_june_stacked':ds_climate_coarse_june_stacked,
    'ox':ox,
    'odata':odata,
    'olat':olat,
    'oele':oele,
    'data_scaler':data_scaler,
    'ele_scaler':ele_scaler,
    'lat_scaler':lat_scaler,
    'odata_scaled':jnp.array(odata_scaled),
    'olat_scaled':jnp.array(olat_scaled),
    'oele_scaled':jnp.array(oele_scaled),
    'cx':cx,
    'cdata':cdata,
    'clat':clat,
    'cele':cele,
    'cdata_scaled':jnp.array(cdata_scaled),
    'clat_scaled':jnp.array(clat_scaled),
    'cele_scaled':jnp.array(cele_scaled),
}



# %% Saving the scenario
scenario_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'
np.save(scenario_outpath, scenario_real)































# %% Creating Scenario
lengthscale_max = ((cx.max(axis=0)-cx.min(axis=0))/2).max()
lengthscale_prior = dist.Uniform(1,lengthscale_max/2)

scenario_real = {
    'ds_aws_june_filtered':ds_aws_june_filtered,
    'ds_climate_coarse_june_stacked_landonly':ds_climate_coarse_june_stacked_landonly,
    # 'ds_climate_stacked_jan_filtered':ds_climate_stacked_jan_filtered,
    # 'ds_aws_stacked_jan_filtered':ds_aws_stacked_jan_filtered,
    'ox':ox,
    'odata':odata,
    'olat':olat,
    'oele':oele,
    'data_scaler':data_scaler,
    'ele_scaler':ele_scaler,
    'lat_scaler':lat_scaler,
    'odata_scaled':jnp.array(odata_scaled),
    'olat_scaled':jnp.array(olat_scaled),
    'oele_scaled':jnp.array(oele_scaled),
    'cx':cx,
    'cdata':cdata,
    'clat':clat,
    'cele':cele,
    'cdata_scaled':jnp.array(cdata_scaled),
    'clat_scaled':jnp.array(clat_scaled),
    'cele_scaled':jnp.array(cele_scaled),

    'jitter': 1e-5,
    "MEAN_T_variance_prior": dist.Uniform(0.1,10.0),
    "MEAN_T_lengthscale_prior": lengthscale_prior,
    "MEAN_T_mean_prior": dist.Normal(0.0, 2.0),
    "LOGVAR_T_variance_prior": dist.Uniform(0.1,10.0),
    "LOGVAR_T_lengthscale_prior": lengthscale_prior,
    "LOGVAR_T_mean_prior": dist.Normal(-4.0, 5.0),
    "MEAN_B_variance_prior": dist.Uniform(0.1,10.0),
    "MEAN_B_lengthscale_prior": lengthscale_prior,
    "MEAN_B_mean_prior": dist.Normal(0.0, 1.0),
    "LOGVAR_B_variance_prior": dist.Uniform(0.1,10.0),
    "LOGVAR_B_lengthscale_prior": lengthscale_prior,
    "LOGVAR_B_mean_prior": dist.Normal(0.0, 5.0),
    "MEAN_T_mean_b0_prior": dist.Normal(0.0, 3.0),
    "MEAN_T_mean_b1_prior": dist.Normal(0.0, 3.0),
    "MEAN_T_mean_b2_prior": dist.Normal(0.0, 3.0),
    "MEAN_B_mean_b0_prior": dist.Normal(0.0, 3.0),
    "MEAN_B_mean_b1_prior": dist.Normal(0.0, 3.0),
    "LOGVAR_T_mean_b0_prior": dist.Normal(0.0, 3.0),
    "LOGVAR_T_mean_b1_prior": dist.Normal(0.0, 3.0),
    "LOGVAR_T_mean_b2_prior": dist.Normal(0.0, 3.0),
    "LOGVAR_B_mean_b0_prior": dist.Normal(0.0, 3.0),
    "LOGVAR_B_mean_b1_prior": dist.Normal(0.0, 5.0),

}




















# %% Formatting and Filtering Data
months = [12,1,2]
ds_climate_stacked = ds_climate_stacked.where(ds_climate_stacked.month.isin(months),drop=True)

ds_aws_stacked = ds_aws.stack(X=('Year','Month'))
ds_aws_stacked = ds_aws_stacked.where(ds_aws_stacked.Month.isin(months),drop=True)
ds_aws_stacked = ds_aws_stacked.where(ds_aws_stacked['Temperature'].count('X')>5,drop=True)

cx = jnp.array(np.dstack([ds_climate_stacked['glon'],ds_climate_stacked['glat']])[0])
cdata = jnp.array(ds_climate_stacked['Temperature'].values)
clat = jnp.array(ds_climate_stacked['latitude'].values)
cele = jnp.array(ds_climate_stacked['Elevation'].values)

ox = jnp.array(np.dstack([ds_aws_stacked['glon'].isel(X=0),ds_aws_stacked['glat'].isel(X=0)]))[0]
odata = jnp.array(ds_aws_stacked.transpose()['Temperature'].values)
olat = jnp.array(ds_aws_stacked['Lat(℃)'].isel(X=0).values)
oele = jnp.array(ds_aws_stacked['Elevation(m)'].isel(X=0).values)























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

ism_mask_fudgefix = ism_mask * ds_climate['LSM'].data
ds_climate['ISM'] = (('grid_longitude', 'grid_latitude'), ism_mask_fudgefix)

# %% Reformatting data for model
ds_climate_masked = ds_climate.where(ds_climate.ISM)
ds_climate_stacked = ds_climate_masked.stack(X=('grid_latitude', 'grid_longitude'))
ds_climate_stacked = ds_climate_stacked.dropna('X')
ds_climate_stacked = ds_climate_stacked.sel(Model='MAR(ERA5)')
ds_climate_stacked['Temperature'] = ds_climate_stacked['Temperature']-273.15

df_aws_group = df_aws.set_index(['Station','Year','Month'])
df_aws_group_coords = df_aws.set_index(['Station'])
da_temp = df_aws_group[~df_aws_group.index.duplicated()]['Temperature'].to_xarray()
ds_coords = df_aws_group_coords[~df_aws_group_coords.index.duplicated()][['Lat(℃)','Lon(℃)','Elevation(m)','glat','glon','grid_latitude', 'grid_longitude']].to_xarray()
ds_aws = xr.merge([ds_coords,da_temp])

# %% Formatting and Filtering Data
months = [12,1,2]

ds_climate_stacked = ds_climate_stacked.where(ds_climate_stacked.month.isin(months),drop=True)

ds_aws_stacked = ds_aws.stack(X=('Year','Month'))
ds_aws_stacked = ds_aws_stacked.where(ds_aws_stacked.Month.isin(months),drop=True)
ds_aws_stacked = ds_aws_stacked.where(ds_aws_stacked['Temperature'].count('X')>5,drop=True)

cx = jnp.array(np.dstack([ds_climate_stacked['glon'],ds_climate_stacked['glat']])[0])
cdata = jnp.array(ds_climate_stacked['Temperature'].values)
clat = jnp.array(ds_climate_stacked['latitude'].values)
cele = jnp.array(ds_climate_stacked['Elevation'].values)

ox = jnp.array(np.dstack([ds_aws_stacked['glon'].isel(X=0),ds_aws_stacked['glat'].isel(X=0)]))[0]
odata = jnp.array(ds_aws_stacked.transpose()['Temperature'].values)
olat = jnp.array(ds_aws_stacked['Lat(℃)'].isel(X=0).values)
oele = jnp.array(ds_aws_stacked['Elevation(m)'].isel(X=0).values)

# %% Examining Observation and Climate Data
text_width = 17.68/2.58
fig, axs = plt.subplots(1,2, figsize=(10, 5),dpi=300)#,frameon=False)
ds_climate_masked_filtered = ds_climate_masked['Temperature']
# ds_climate_masked_filtered = ds_climate_masked_filtered.where(ds_climate_masked_filtered.month.isin(months),drop=True)
# ds_climate_masked_filtered = ds_climate_masked_filtered-273.15

ax=axs[0]
ds_climate_masked_filtered.mean('Time').plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    cmap='jet',
    vmin=-40,
    vmax=-4.5,
    cbar_kwargs = {'fraction':0.030,
                   'pad':0.04,
                   'label':'Temperature Monthly Mean'}
)
ax.scatter(
    ox[:, 0],
    ox[:, 1],
    s=np.count_nonzero(~np.isnan(odata),axis=0)*1,
    marker="o",
    c=np.nanmean(odata,axis=0),
    cmap='jet',
    edgecolor="w",
    linewidth=0.4,
    vmin=-40,
    vmax=-4.5,
)

ax=axs[1]
ds_climate_masked_filtered.std('Time').plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    cmap='jet',
    vmin=1.5,
    vmax=5.5,
    cbar_kwargs = {'fraction':0.030,
                   'pad':0.04,
                   'label':'Temperature Monthly Std.Dev.'}
)
ax.scatter(
    ox[:, 0],
    ox[:, 1],
    s=np.count_nonzero(~np.isnan(odata),axis=0)*1,
    marker="o",
    c=np.nanstd(odata,axis=0),
    cmap='jet',
    edgecolor="w",
    linewidth=0.4,
    vmin=1.5,
    vmax=5.5,
)

for ax in axs:
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)

plt.tight_layout()

# %% Standardising Elevation and Latitude Data

ele_scaler = StandardScaler() 
oele_scaled = ele_scaler.fit_transform(oele.reshape(-1,1))[:,0]
lat_scaler = StandardScaler() 
olat_scaled = lat_scaler.fit_transform(olat.reshape(-1,1))[:,0]
cele_scaled = ele_scaler.transform(cele.reshape(-1,1))[:,0]
clat_scaled = lat_scaler.transform(clat.reshape(-1,1))[:,0]

# %% Sanity Check
print('Data Shapes: \n',
      f'ox.shape:{ox.shape} \n',
      f'oele_scaled.shape:{oele_scaled.shape} \n',
      f'olat_scaled.shape:{olat_scaled.shape} \n',
      f'odata.shape:{odata.shape} \n',
      f'cx.shape:{cx.shape} \n',
      f'cele_scaled.shape:{cele_scaled.shape} \n',
      f'clat_scaled.shape:{clat_scaled.shape} \n',
      f'cdata.shape:{cdata.shape} \n'
      )

# %% Creating Scenario

lengthscale_max = ((cx.max(axis=0)-cx.min(axis=0))/2).max()
lengthscale_prior = dist.Uniform(1,lengthscale_max)

scenario_real = {
    'ds_climate_stacked':ds_climate_stacked,
    'ds_aws_stacked':ds_aws_stacked,
    'jitter': 1e-5,
    'cx':cx,
    'cdata':cdata,
    'clat':jnp.array(clat_scaled),
    'cele':jnp.array(cele_scaled),
    'ox':ox,
    'odata':odata,
    'olat':jnp.array(olat_scaled),
    'oele':jnp.array(oele_scaled),
    'ele_scaler':ele_scaler,
    'lat_scaler':lat_scaler,
    "MEAN_T_variance_prior": dist.Uniform(100.0,300.0),
    "MEAN_T_lengthscale_prior": lengthscale_prior,
    "MEAN_T_mean_prior": dist.Normal(-25.0, 5.0),
    "LOGVAR_T_variance_prior": dist.Uniform(0.1,10.0),
    "LOGVAR_T_lengthscale_prior": lengthscale_prior,
    "LOGVAR_T_mean_prior": dist.Normal(3.6, 2.0),
    "MEAN_B_variance_prior": dist.Uniform(10.0,40.0),
    "MEAN_B_lengthscale_prior": lengthscale_prior,
    "MEAN_B_mean_prior": dist.Normal(2.8, 2.0),
    "LOGVAR_B_variance_prior": dist.Uniform(1.0,20.0),
    "LOGVAR_B_lengthscale_prior": lengthscale_prior,
    "LOGVAR_B_mean_prior": dist.Normal(-0.2, 1.0),
    "MEAN_T_mean_b0_prior": dist.Normal(-25.0, 10.0),
    "MEAN_T_mean_b1_prior": dist.Normal(-10.0, 10.0),
    "MEAN_T_mean_b2_prior": dist.Normal(6.0, 10.0),
    "MEAN_B_mean_b0_prior": dist.Normal(3.0, 5.0),
    "MEAN_B_mean_b1_prior": dist.Normal(2.0, 5.0),
    "LOGVAR_T_mean_b0_prior": dist.Normal(4.0, 5.0),
    "LOGVAR_T_mean_b1_prior": dist.Normal(0.0, 5.0),
    "LOGVAR_T_mean_b2_prior": dist.Normal(-0.4, 5.0),
    "LOGVAR_B_mean_b0_prior": dist.Normal(-0.2, 5.0),
    "LOGVAR_B_mean_b1_prior": dist.Normal(1.0, 5.0),
}

# %% Saving the scenario
scenario_outpath = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'
np.save(scenario_outpath, scenario_real)

# %%

# %% Legacy Code for Coarsening Climate Model Output

# ds_climate = ds_climate.coarsen(grid_latitude=28,grid_longitude=28).mean()

# # %% Importing ice sheet shapefile for plotting and mask purposes
# shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
# icehsheet_shapefile = f'{shapefiles_path}/cst10_polygon.shp'

# gdf_icesheet = gpd.read_file(icehsheet_shapefile)
# gdf_icesheet = gdf_icesheet.explode().iloc[[61]]
# gdf_icesheet = gdf_icesheet.reset_index().drop(columns=['level_0','level_1'])

# ##### glon,glat #####
# rotated_coord_system = ccrs.RotatedGeodetic(
#     13.079999923706055,
#     0.5199999809265137,
#     central_rotated_longitude=180.0,
#     globe=None,
# )
# gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)

# # %% Creating ice sheet mask that excludes islands
# map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)
# ism_mask = create_mask(ds_climate.drop_dims('Model'),gdf_icesheet,map_proj)

# ism_mask_fudgefix = ism_mask * ds_climate['LSM'].data
# ds_climate['ISM'] = (('grid_longitude', 'grid_latitude'), ism_mask_fudgefix)

# months = [12,1,2]
# ds_climate_stacked = ds_climate_stacked.where(ds_climate_stacked.month.isin(months),drop=True)