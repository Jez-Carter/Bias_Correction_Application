# %% Importing Packages
import os
import glob
import pandas as pd
import cartopy.crs as ccrs
from pyproj import Transformer

base_path = '/home/jez/'
data_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/The AntAWS dataset/'
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
outpath = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Processed/'

# %% Loading data
# csv_files = glob.glob(os.path.join(f'{data_path}Daily_25%/', "*"))
# summary_file = f'{data_path}Daily_25%/Main characteristics of automatic weather station.csv'
csv_files = glob.glob(os.path.join(f'{data_path}Daily_75%/', "*"))
summary_file = f'{data_path}Daily_75%/Main characteristics of automatic weather station.csv'
csv_station_files = [file for file in csv_files if summary_file not in file]

dfs = []
for file in csv_station_files:
    df = pd.read_csv(file,encoding_errors='ignore')
    station_name = file.split('/')[-1].split('_day.csv')[0]
    df['Station']=station_name
    dfs.append(df)
df_all = pd.concat(dfs)
df_all = df_all.reset_index(drop=True)
df_all = df_all[df_all['Year'].notna()]
df_all = df_all[['Station','Year','Month','Day','Temperature()']]
df_all = df_all.rename(columns={'Temperature()': 'Temperature'})

# %% Including the summary information for each site (mostly elevation)
df_summary = pd.read_csv(summary_file,encoding_errors='ignore')
df_summary = df_summary[['Station','Lat(℃)','Lon(℃)','Elevation(m)','Institution']]
df_summary['Station'] = df_summary['Station'].replace('aws11 ','aws11')
df_summary['Station'] = df_summary['Station'].replace('Rita ','Rita')
df_summary['Station'] = df_summary['Station'].replace('Mt.Erebus','Mt. Erebus')
df_summary['Station'] = df_summary['Station'].replace('Mt.Fleming','Mt. Fleming')

df_all_combined = df_summary.merge(df_all, how='outer', left_on='Station', right_on='Station')

# %% Including glon,glat,grid_longitude and grid_latitude coordinates in AWS data for comparison to climate model output
lon=df_all_combined['Lon(℃)']
lat=df_all_combined['Lat(℃)']

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
df_all_combined['glon']=glon
df_all_combined['glat']=glat

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
df_all_combined['grid_longitude']=grid_longitude_360
df_all_combined['grid_latitude']=grid_latitude

# %% Removing data points with null temperature or elevation
df_all_combined = df_all_combined[df_all_combined['Temperature'].isnull()==False]
df_all_combined = df_all_combined[df_all_combined['Elevation(m)'].isnull()==False]

# %% Saving output
df_all_combined.to_csv(f'{outpath}df_all_combined_75.csv',index=False)

# %%
