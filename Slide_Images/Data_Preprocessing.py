# %% Importing Packages 
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import cartopy.crs as ccrs

from src.helper_functions import create_aws_mask

###### Paths ######
base_path = '/home/jez/'
repo_path = f'{base_path}Bias_Correction_Application/'

aws_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/'
CORDEX_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'

shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
aws_shapefile = f'{shapefiles_path}/267AWS.shp'
icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
icehsheet_main_shapefile = f'{shapefiles_path}/cst10_polygon.shp'

internal_outpath = f'{repo_path}Slide_Images/Data/'
external_outpath = f'{base_path}DSNE_ice_sheets/Jez/Slides/'

# %% Automatic Weather Station Data 
df_all_combined = pd.read_csv(f'{aws_path}Processed/df_all_combined_75_monthly.csv')

####### Converting to Xarray #######
df_aws = df_all_combined
df_aws_group = df_aws.set_index(['Station','Year','Month'])
df_aws_group_coords = df_aws.set_index(['Station'])
da_temp = df_aws_group[~df_aws_group.index.duplicated()]['Temperature'].to_xarray()
ds_coords = df_aws_group_coords[~df_aws_group_coords.index.duplicated()][['Lat(℃)','Lon(℃)','Elevation(m)','glat','glon','grid_latitude', 'grid_longitude']].to_xarray()
ds_aws = xr.merge([ds_coords,da_temp])
ds_aws_stacked = ds_aws.stack(X=('Year','Month'))
ds_aws_stacked = ds_aws_stacked.set_coords(("glat", "glon"))
ds_aws_stacked['Mean Temperature'] = ds_aws_stacked.mean('X')['Temperature']
ds_aws_stacked['Temperature Records'] = ds_aws_stacked.count('X')['Temperature']
ds_aws_stacked['June Mean Temperature'] = ds_aws_stacked.where(ds_aws_stacked['Month']==6).mean('X')['Temperature']
ds_aws_stacked['June Temperature Records'] = ds_aws_stacked.where(ds_aws_stacked['Month']==6).count('X')['Temperature']

####### Creating Ice Sheet Mask #######
gdf_icesheet_main = gpd.read_file(icehsheet_main_shapefile)
gdf_icesheet_main = gdf_icesheet_main.explode().iloc[[61]]
map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)
aws_mask = create_aws_mask(ds_aws,gdf_icesheet_main,map_proj)
mainland_stations = ds_aws['Station'][aws_mask].data

####### Filtering AWS Data #######
ds_aws_stacked = ds_aws_stacked.sel(Station=mainland_stations)

# %% Climate Model Data
ds_climate = xr.open_dataset(CORDEX_path)
ds_climate = xr.open_dataset(CORDEX_path)
ds_climate = ds_climate.sel(Model='MAR(ERA5)')
ds_climate['Temperature']=ds_climate['Temperature'] - 273.15
ds_climate['Latitude']=ds_climate['latitude']

####### Summary Metrics #######
ds_climate['Mean Temperature'] = ds_climate.mean('Time')['Temperature']
ds_climate['Mean June Temperature'] = ds_climate.where(ds_climate.month==6).mean('Time')['Temperature']

####### Climate Model Nearest Grid Cells #######
aws_grid_longitudes = ds_aws_stacked.grid_longitude.data
aws_grid_latitudes = ds_aws_stacked.grid_latitude.data

ds_climate_nearest = ds_climate.sel(
    grid_longitude=aws_grid_longitudes,
    grid_latitude=aws_grid_latitudes,
    method='nearest')
ds_climate_nearest_stacked = ds_climate_nearest.stack(X=(('grid_longitude','grid_latitude')))
diagonal_indecies = np.diag(np.arange(0,len(aws_grid_longitudes)**2,1).reshape(len(aws_grid_longitudes),-1),k=0)
ds_climate_nearest_stacked = ds_climate_nearest_stacked.isel(X=diagonal_indecies)
ds_climate_nearest_stacked = ds_climate_nearest_stacked.assign_coords(Nearest_Station=("X", ds_aws_stacked.Station.data))
ds_climate_nearest_stacked = ds_climate_nearest_stacked.swap_dims({"X": "Nearest_Station"})

# %% Saving Data
ds_aws_stacked.reset_index('X').to_netcdf(f"{external_outpath}ds_aws_stacked.nc")

ds_climate.to_netcdf(f"{external_outpath}ds_climate.nc")
ds_climate_nearest_stacked.reset_coords(names="X", drop=True).to_netcdf(f"{external_outpath}ds_climate_nearest_stacked.nc")




# # %% Additional AWS Processing (Not Needed?)

# ####### Geopandas Geometry Data #######
# gdf_aws = gpd.read_file(aws_shapefile)
# gdf_aws['zhandian'] = gdf_aws['zhandian'].replace('Mt.Erebus','Mt. Erebus')
# gdf_aws['zhandian'] = gdf_aws['zhandian'].replace('Mt.Fleming','Mt. Fleming')
# gdf_all_combined = gdf_aws[['zhandian','geometry']].merge(
#     df_all_combined,
#     how='outer',
#     left_on='zhandian',
#     right_on='Station')
# gdf_all_combined = gdf_all_combined.drop('zhandian',axis=1)

# ####### Total Records Count Maps Data #######
# count_group = ['Station','Lat(℃)','Lon(℃)']#,'Elevation(m)','Institution']
# df_all_combined_count = gdf_all_combined.groupby(count_group).agg(
#     Temperature_Records=('Temperature', 'count'),
#     geometry=('geometry','first'),
#     glon = ('glon','first'),
#     glat = ('glat','first')
#     )
# gdf_all_combined_count = gpd.GeoDataFrame(
#     df_all_combined_count,
#     geometry=df_all_combined_count.geometry,
#     crs=gdf_all_combined.crs
# )
# gdf_all_combined_count_filtered = gdf_all_combined_count[gdf_all_combined_count.index.get_level_values('Lat(℃)')<-55]

# ####### Mean & Std Maps (all months) Data #######
# index_group = ['Station',
#                'Lat(℃)',
#                'Lon(℃)',
#                'Elevation(m)',
#                'Institution',
#                'glat',
#                'glon',
#                'grid_latitude',
#                'grid_longitude']
# df_all_combined_grouped = df_all_combined.groupby(index_group).agg(
#     Mean_Temperature=('Temperature', 'mean'),
#     Std_Temperature=('Temperature', 'std'),
#     Temperature_Records=('Temperature', 'count')
#     ).reset_index()

# ####### Mean & Std Maps by Month Data #######
# index_group.append('Month')
# df_all_combined_grouped_months = df_all_combined.groupby(index_group).agg(
#     Mean_Temperature=('Temperature', 'mean'),
#     Std_Temperature=('Temperature', 'std'),
#     Temperature_Records=('Temperature', 'count')
#     ).reset_index()

# gdf_all_combined_count_filtered.to_file(f"{internal_outpath}gdf_all_combined_count_filtered.gpkg", driver="GPKG")

# %%
