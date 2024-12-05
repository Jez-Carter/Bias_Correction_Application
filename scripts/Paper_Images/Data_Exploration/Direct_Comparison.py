# %% Importing Packages
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import xarray as xr
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import linregress
from src.helper_functions import create_aws_mask

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w"))

def regressfunc(x, y, ax=None, **kws):
    """Plot the linear regression coefficients in the top left hand corner of a plot."""
    ax = ax or plt.gca()
    slope, intercept, rvalue, pvalue, stderr = linregress(x=x, y=y)
    ax.annotate(f'y = {intercept:.2f}+x*{slope:.2f}', xy=(.1, .9), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w"))
    ax.annotate(f'slope_stderr = {stderr:.2f}', xy=(.1, .7), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w"))

def corrfunc_combined(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    if kws['label'] == 'AWS':
        ax.annotate(r'$\rho_{AWS}$'+f' = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w",alpha=0.8))
    else:
        ax.annotate(r'$\rho_{CM}$'+f' = {r:.2f}', xy=(.1, .75), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w",alpha=0.8))

pd.options.display.max_colwidth = 100

plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1

legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm

results_path = '/home/jez/Bias_Correction_Application/results/Paper_Images/'

# %% Loading Data
base_path = '/home/jez/'
repo_path = f'{base_path}Bias_Correction_Application/'

###### Ice Sheet Shapefile ######
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
icehsheet_main_shapefile = f'{shapefiles_path}/cst10_polygon.shp'

gdf_icesheet = gpd.read_file(icehsheet_shapefile)
gdf_icesheet_main = gpd.read_file(icehsheet_main_shapefile)
gdf_icesheet_main = gdf_icesheet_main.explode().iloc[[61]]
# gdf_icesheet_main = gdf_icesheet_main.reset_index().drop(columns=['level_0','level_1'])

rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)

###### Automatic Weather Station Data ######
aws_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/'
aws_shapefile = f'{shapefiles_path}/267AWS.shp'

df_all_combined = pd.read_csv(f'{aws_path}Processed/df_all_combined_75_monthly.csv')

gdf_aws = gpd.read_file(aws_shapefile)
gdf_aws['zhandian'] = gdf_aws['zhandian'].replace('Mt.Erebus','Mt. Erebus')
gdf_aws['zhandian'] = gdf_aws['zhandian'].replace('Mt.Fleming','Mt. Fleming')
gdf_all_combined = gdf_aws[['zhandian','geometry']].merge(
    df_all_combined,
    how='outer',
    left_on='zhandian',
    right_on='Station')
gdf_all_combined = gdf_all_combined.drop('zhandian',axis=1)

index_group = ['Station',
               'Lat(℃)',
               'Lon(℃)',
               'Elevation(m)',
               'Institution',
               'glat',
               'glon',
               'grid_latitude',
               'grid_longitude']
df_all_combined_grouped = df_all_combined.groupby(index_group).agg(
    Mean_Temperature=('Temperature', 'mean'),
    Std_Temperature=('Temperature', 'std'),
    Temperature_Records=('Temperature', 'count')
    ).reset_index()

df_aws = gdf_all_combined
df_aws_group = df_aws.set_index(['Station','Year','Month'])
df_aws_group_coords = df_aws.set_index(['Station'])
da_temp = df_aws_group[~df_aws_group.index.duplicated()]['Temperature'].to_xarray()
ds_coords = df_aws_group_coords[~df_aws_group_coords.index.duplicated()][['Lat(℃)','Lon(℃)','Elevation(m)','glat','glon','grid_latitude', 'grid_longitude']].to_xarray()
ds_aws = xr.merge([ds_coords,da_temp])
ds_aws_stacked = ds_aws.stack(X=('Year','Month'))

map_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=-90, globe=None)
aws_mask = create_aws_mask(df_all_combined_grouped,gdf_icesheet_main,map_proj)
mainland_stations = df_all_combined_grouped[aws_mask]['Station']
index_group.append('Month')
df_all_combined_grouped_months = df_all_combined.groupby(index_group).agg(
    Mean_Temperature=('Temperature', 'mean'),
    Std_Temperature=('Temperature', 'std'),
    Temperature_Records=('Temperature', 'count')
    ).reset_index()
df_all_combined_grouped_june = df_all_combined_grouped_months.where(
    df_all_combined_grouped_months.Month.isin([6])
    ).dropna()
stations_recordsfilter_june = df_all_combined_grouped_june[df_all_combined_grouped_june['Temperature_Records']>10]['Station']
stations_filter_june = mainland_stations[mainland_stations.isin(stations_recordsfilter_june)]
stations_filter_june = stations_filter_june.drop_duplicates() 

###### Climate Model Data ######
CORDEX_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'
ds_climate = xr.open_dataset(CORDEX_path)
ds_climate = xr.open_dataset(CORDEX_path)
ds_climate = ds_climate.sel(Model='MAR(ERA5)')
ds_climate['Temperature']=ds_climate['Temperature'] - 273.15
ds_climate['Latitude']=ds_climate['latitude']

# %% Climate Model Nearest Grid Cells
aws_grid_longitudes = df_all_combined_grouped['grid_longitude'].values
aws_grid_latitudes = df_all_combined_grouped['grid_latitude'].values
ds_climate_nearest = ds_climate.sel(
    grid_longitude=aws_grid_longitudes,
    grid_latitude=aws_grid_latitudes,
    method='nearest')
ds_climate_nearest_stacked = ds_climate_nearest.stack(X=(('grid_longitude','grid_latitude')))
diagonal_indecies = np.diag(np.arange(0,len(aws_grid_longitudes)**2,1).reshape(len(aws_grid_longitudes),-1),k=0)
ds_climate_nearest_stacked = ds_climate_nearest_stacked.isel(X=diagonal_indecies)
ds_climate_nearest_stacked = ds_climate_nearest_stacked.assign_coords(Nearest_Station=("X", df_all_combined_grouped.Station))
ds_climate_nearest_stacked = ds_climate_nearest_stacked.swap_dims({"X": "Nearest_Station"})

# %% Spatial Map Comparison
months = [6]
ds_climate_filtered = ds_climate.where(ds_climate.month.isin(months)).dropna('Time')
ds_climate_filtered = ds_climate_filtered.where(ds_climate_filtered.LSM)

df_all_combined_grouped_months
df_all_combined_grouped_june = df_all_combined_grouped_months.where(
    df_all_combined_grouped_months.Month.isin(months)
    ).dropna()

stations_filter_june_mask = df_all_combined_grouped_june['Station'].isin(stations_filter_june)
df_all_combined_grouped_june_filtered = df_all_combined_grouped_june[stations_filter_june_mask]

fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=300)#,frameon=False)

ax=axs[0]
ds_climate_filtered['Temperature'].mean('Time').plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    vmin=-55,
    vmax=-10,
    cmap='jet',
    cbar_kwargs = {'fraction':0.030,
                'pad':0.02,
                'label':'Mean January Temperature'}
)
ax.annotate('a.',xy=(0.05,0.95),xycoords='axes fraction')

ax.scatter(
    df_all_combined_grouped_june_filtered.glon,
    df_all_combined_grouped_june_filtered.glat,
    s=df_all_combined_grouped_june_filtered['Temperature_Records']/2,
    c=df_all_combined_grouped_june_filtered['Mean_Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
)

ax=axs[1]
ds_climate_filtered['Temperature'].std('Time').plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    vmin=1.0,
    vmax=5.0,
    cmap='jet',
    cbar_kwargs = {'fraction':0.030,
                   'pad':0.04,
                   'label':'Std.Dev. January Temperature'}
)
ax.annotate('b.',xy=(0.05,0.95),xycoords='axes fraction')
ax.scatter(
    df_all_combined_grouped_june_filtered.glon,
    df_all_combined_grouped_june_filtered.glat,
    s=df_all_combined_grouped_june_filtered['Temperature_Records']/2,
    c=df_all_combined_grouped_june_filtered['Std_Temperature'],
    cmap='jet',
    vmin=1.0,
    vmax=5.0,
    edgecolor='w',
    linewidths=0.5,
)

for ax in axs:

    bins = np.array([5,10,15,20,25,30])
    ax.add_artist(
        ax.legend(
            handles=[
                mlines.Line2D(
                    [],
                    [],
                    color="tab:blue",
                    markeredgecolor="k",
                    markeredgewidth=0.3,
                    lw=0,
                    marker="o",
                    markersize=np.sqrt(b/2),
                    label=str(int(b)),
                )
                for i, b in enumerate(bins)
            ],
            loc=3,
            fontsize = legend_fontsize,
            framealpha=0,
        )
    )

    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_title('')
    ax.set_xlabel('Grid Longitude')
    ax.set_ylabel('Grid Latitude')
    ax.set_axis_off()

plt.tight_layout()

# %% Correlation between AWS and Climate Model Nearest Neighbourgh

df_climate_nearest = ds_climate_nearest_stacked.swap_dims({"Nearest_Station": "X"}).to_dataframe().reset_index(drop=True)
df_aws_renamed = df_aws.rename(columns={'Elevation(m)':'Elevation',
                                        'Lat(℃)':'latitude',
                                        'Station':'Nearest_Station',
                                        'Month':'month'})
df_climate_nearest['Data_Source']='Climate Model'
df_aws_renamed['Data_Source']='AWS'

keys = ['Temperature','latitude','Elevation','Data_Source','Nearest_Station','month']
df_combined = pd.concat([df_aws_renamed[keys],df_climate_nearest[keys]])

stations_filter_mask = df_combined['Nearest_Station'].isin(stations_filter_june)
df_combined_filtered = df_combined[stations_filter_mask]
months = [6]
df_combined_filtered = df_combined_filtered[df_combined_filtered['month'].isin(months)]

index_group = ['Nearest_Station','latitude','Elevation','Data_Source']
df_combined_filtered_group = df_combined_filtered.groupby(index_group).agg(
    Mean_Temperature=('Temperature', 'mean'),
    Std_Temperature=('Temperature', 'std'),
    Temperature_Records=('Temperature', 'count')
    ).reset_index()

df_aws_filtered_group = df_combined_filtered_group[df_combined_filtered_group['Data_Source']=='AWS']
df_climate_filtered_group = df_combined_filtered_group[df_combined_filtered_group['Data_Source']=='Climate Model']

df_aws_filtered_group = df_aws_filtered_group.sort_values(['Nearest_Station'])
df_climate_filtered_group = df_climate_filtered_group.sort_values(['Nearest_Station'])

fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=300)#,frameon=False)

for ax,var in zip(axs,['Mean_Temperature','Std_Temperature']):
    ax.scatter(x=df_aws_filtered_group[var],
               y=df_climate_filtered_group[var],
            #    s=df_aws_filtered_group['Temperature_Records'],
               marker='+')
    corrfunc(x=df_aws_filtered_group[var],
             y=df_climate_filtered_group[var],
             ax=ax)
    ax.plot(np.linspace(min(df_aws_filtered_group[var]),
                          max(df_aws_filtered_group[var]),
                          10),
            np.linspace(min(df_aws_filtered_group[var]),
                          max(df_aws_filtered_group[var]),
                          10),
            linestyle='dotted')
    
axs[0].annotate('a.',xy=(0.03,0.95),xycoords='axes fraction')
axs[1].annotate('b.',xy=(0.03,0.95),xycoords='axes fraction')

axs[0].set_xlabel('AWS Mean Temperature / $^\circ$C')
axs[0].set_ylabel('NN Climate Model Mean Temperature / $^\circ$C')
axs[1].set_xlabel('AWS Std.Dev. Temperature / $^\circ$C')
axs[1].set_ylabel('NN Climate Model Std.Dev. Temperature / $^\circ$C')

plt.tight_layout()

fig.savefig(f"{results_path}fig08.pdf", dpi=300, bbox_inches="tight")

# %% Comparison Single Site
fig, axs = plt.subplots(1, 3, figsize=(text_width, text_width/3),dpi=300)#,frameon=False)

station = df_all_combined_grouped.sort_values('Temperature_Records').iloc[-2].Station
gdf_single_station = gdf_all_combined[gdf_all_combined['Station']==station]
print(station)
print(f'Min,Max Year = {gdf_single_station.Year.min(),gdf_single_station.Year.max()}')
print('Records')
print(gdf_single_station.groupby(['Year'])['Temperature'].count().sort_values().head(5))
gdf_single_station_june = gdf_single_station.where(gdf_single_station.Month.isin([6])).dropna()

ds_stacked_stations_filtered = ds_aws_stacked.sel(Station=station).reset_index('X')
ds_climate_nearest_stacked_filtered = ds_climate_nearest_stacked.sel(Nearest_Station=station)#.reset_index('X')
ds_climate_nearest_stacked_filtered = ds_climate_nearest_stacked_filtered.drop_vars('Time')
ds_climate_nearest_stacked_filtered_june = ds_climate_nearest_stacked_filtered.where(
    ds_climate_nearest_stacked_filtered['month']==6).dropna('Time')

ax=axs[0]
ds_climate_nearest_stacked_filtered['Temperature'].plot(ax=ax,hue='Station',alpha=0.7)
ds_stacked_stations_filtered['Temperature'].plot(ax=ax,hue='Station',alpha=0.7)

xticks = np.linspace(48,503,5).astype('int')
xticklabels = ds_stacked_stations_filtered['Year'].values[xticks].astype('int')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)#,rotation = 90)
ax.set_ylabel('Temperature')
ax.set_xlabel('Time')
ax.annotate('a.',xy=(0.03,0.95),xycoords='axes fraction')
ax.set_title('')

ax=axs[1]
gdf_single_station[['Temperature']].hist(bins=40,
                                         ax=ax,
                                         edgecolor='k',
                                         linewidth=0.2,
                                         grid=False,
                                         density=1,
                                         alpha=0.7
                                         )

ds_climate_nearest_stacked_filtered.to_dataframe()[['Temperature']].hist(bins=40,
                                         ax=ax,
                                         edgecolor='k',
                                         linewidth=0.2,
                                         grid=False,
                                         density=1,
                                         alpha=0.7
                                         )

ax.annotate('b.',xy=(0.03,0.95),xycoords='axes fraction')


ax=axs[2]
gdf_single_station_june[['Temperature']].hist(bins=10,
                                              ax=ax,
                                              edgecolor='k',
                                              linewidth=0.2,
                                              grid=False,
                                              density=1,
                                              alpha=0.7,
                                              label=f'{station} Station Records')
ds_climate_nearest_stacked_filtered_june.to_dataframe()[['Temperature']].hist(bins=10,
                                         ax=ax,
                                         edgecolor='k',
                                         linewidth=0.2,
                                         grid=False,
                                         density=1,
                                         alpha=0.7,
                                         label='Nearest Grid Cell Climate Model Output')

handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels, fontsize=legend_fontsize, bbox_to_anchor=(0.5, -0.05), ncols=2, loc=10
)

ax.annotate('c.',xy=(0.03,0.95),xycoords='axes fraction')

for ax in axs[1:]:
    ax.set_title('')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Density')

plt.tight_layout()

fig.savefig(f"{results_path}fig09.pdf", dpi=300, bbox_inches="tight")

# %% Pair Plot for Both Datasets

df_climate_nearest = ds_climate_nearest_stacked.swap_dims({"Nearest_Station": "X"}).to_dataframe().reset_index(drop=True)
df_aws_renamed = df_aws.rename(columns={'Elevation(m)':'Elevation',
                                        'Lat(℃)':'latitude',
                                        'Station':'Nearest_Station',
                                        'Month':'month'})
df_climate_nearest['Data_Source']='Climate Model'
df_aws_renamed['Data_Source']='AWS'

keys = ['Temperature','latitude','Elevation','Data_Source','Nearest_Station','month']
df_combined = pd.concat([df_aws_renamed[keys],df_climate_nearest[keys]])

stations_filter_mask = df_combined['Nearest_Station'].isin(stations_filter_june)
df_combined_filtered = df_combined[stations_filter_mask]
months = [6]
df_combined_filtered = df_combined_filtered[df_combined_filtered['month'].isin(months)]

index_group = ['Nearest_Station','latitude','Elevation','Data_Source']
df_combined_filtered_group = df_combined_filtered.groupby(index_group).agg(
    Mean_Temperature=('Temperature', 'mean'),
    Std_Temperature=('Temperature', 'std'),
    Temperature_Records=('Temperature', 'count')
    ).reset_index()

vars = ['Mean_Temperature','Std_Temperature','Elevation','latitude']

g = sns.pairplot(df_combined_filtered_group,
            vars=vars,
            plot_kws=dict(linewidth=0.3,alpha=0.6),
            # diag_kws=dict(bins=20),
            kind='scatter',
            hue='Data_Source',
            corner=True,
)

g.map_lower(corrfunc_combined)
g.axes[0,0].set_ylabel('Temperature Mean / $^\circ$C')
g.axes[1,0].set_ylabel('Temperature Monthly Std.Dev. / $^\circ$C')
g.axes[2,0].set_ylabel('Elevation / m')
g.axes[3,0].set_ylabel('Latitude / $^\circ$')
g.axes[3,0].set_xlabel('Temperature Mean / $^\circ$C')
g.axes[3,1].set_xlabel('Temperature Monthly Std.Dev. / $^\circ$C')
g.axes[3,2].set_xlabel('Elevation / m')
g.axes[3,3].set_xlabel('Latitude / $^\circ$')

g.fig.set_size_inches(text_width, text_width)

g._legend.set_bbox_to_anchor((0.9, 0.5))

g.fig.savefig(f"{results_path}fig10.pdf", dpi=300, bbox_inches="tight")

# %% Test Comparing Station Coordinates
df_combined_filtered_group[['Nearest_Station','Data_Source','latitude','Elevation']].sort_values(['Nearest_Station','Data_Source'])

# %% Pair Plot for Difference in Temperature Mean and Standard Deviation between Datasets

df_aws_filtered_group = df_combined_filtered_group[df_combined_filtered_group['Data_Source']=='AWS']
df_climate_filtered_group = df_combined_filtered_group[df_combined_filtered_group['Data_Source']=='Climate Model']

df_aws_filtered_group = df_aws_filtered_group.sort_values(['Nearest_Station'])
df_climate_filtered_group = df_climate_filtered_group.sort_values(['Nearest_Station'])

df_aws_filtered_group['Mean_Temperature_Diff'] = (df_climate_filtered_group['Mean_Temperature'].values
                                                  - df_aws_filtered_group['Mean_Temperature'].values)
df_aws_filtered_group['Std_Temperature_Ratio'] = (df_climate_filtered_group['Std_Temperature'].values
                                                   / df_aws_filtered_group['Std_Temperature'].values)

vars = ['Mean_Temperature_Diff','Std_Temperature_Ratio','Mean_Temperature','Std_Temperature','Elevation','latitude']

g = sns.pairplot(df_aws_filtered_group,
            vars=vars,
            plot_kws=dict(linewidth=0.3,alpha=0.6),
            diag_kws=dict(bins=20),
            kind='scatter',
            corner=True,
)

g.map_lower(corrfunc)
g.axes[0,0].set_ylabel('Diff. Temp. Mean / $^\circ$C')
g.axes[1,0].set_ylabel('Ratio Temp. Std.Dev.')
g.axes[2,0].set_ylabel('Temp. Mean / $^\circ$C')
g.axes[3,0].set_ylabel('Temp. Std.Dev. / $^\circ$C')
g.axes[4,0].set_ylabel('Elevation / m')
g.axes[5,0].set_ylabel('Latitude / $^\circ$')
g.axes[5,0].set_xlabel('Diff. Temp. Mean / $^\circ$C')
g.axes[5,1].set_xlabel('Ratio Temp. Std.Dev.')
g.axes[5,2].set_xlabel('Temp. Mean / $^\circ$C')
g.axes[5,3].set_xlabel('Temp. Std.Dev. / $^\circ$C')
g.axes[5,4].set_xlabel('Elevation / m')
g.axes[5,5].set_xlabel('Latitude / $^\circ$')

g.fig.set_size_inches(text_width, text_width)

# # %% Values for Informing Priors

# da_awslogvar = np.log(np.square(ds_climate_nearest_stacked['AWS_Std_Temperature']))
# da_biaslogvar = np.log(np.square(ds_climate_nearest_stacked['Bias_Std_Temperature']))

# ds_climate_nearest_stacked['AWS_LogVar_Temperature']= (('X'),da_awslogvar.data)
# ds_climate_nearest_stacked['Bias_LogVar_Temperature']= (('Model','X'),da_biaslogvar.data)

# vars = ['AWS_Mean_Temperature',
#         'AWS_LogVar_Temperature',
#         'Bias_Mean_Temperature',
#         'Bias_LogVar_Temperature']
# ds = ds_climate_nearest_stacked[vars].sel(Model='MAR(ERA5)')
# print(ds.mean())
# print(ds.var())


# # %%
# ds_climate_nearest_stacked
# %%
