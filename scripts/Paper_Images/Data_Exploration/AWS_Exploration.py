# %% Importing Packages
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
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
    ax.annotate(f'slope_stderr = {stderr:.2f}', xy=(.1, .8), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w"))

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
###### Automatic Weather Station Data ######
base_path = '/home/jez/'
repo_path = f'{base_path}Bias_Correction_Application/'
aws_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/'
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
aws_shapefile = f'{shapefiles_path}/267AWS.shp'
icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
icehsheet_main_shapefile = f'{shapefiles_path}/cst10_polygon.shp'

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

gdf_icesheet = gpd.read_file(icehsheet_shapefile)
gdf_icesheet_main = gpd.read_file(icehsheet_main_shapefile)
gdf_icesheet_main = gdf_icesheet_main.explode().iloc[[61]]
# gdf_icesheet_main = gdf_icesheet_main.reset_index().drop(columns=['level_0','level_1'])

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
stations_recordsfilter_june = df_all_combined_grouped_june[df_all_combined_grouped_june['Temperature_Records']>5]['Station']
stations_filter_june = mainland_stations[mainland_stations.isin(stations_recordsfilter_june)]
stations_filter_june = stations_filter_june.drop_duplicates() 

# stations_recordsfilter = df_all_combined_grouped[df_all_combined_grouped['Temperature_Records']>5]['Station']
# stations_filter = mainland_stations[mainland_stations.isin(stations_recordsfilter)]
# stations_filter = df_all_combined_grouped['Station'].isin(mainland_stations)

# gdf_all_combined

# %% Plots for a single site
station = df_all_combined_grouped.sort_values('Temperature_Records').iloc[-2].Station
gdf_single_station = gdf_all_combined[gdf_all_combined['Station']==station]
print(gdf_single_station.iloc[0])
gdf_single_station_jan = gdf_single_station.where(gdf_single_station.Month.isin([1])).dropna()
gdf_single_station_june = gdf_single_station.where(gdf_single_station.Month.isin([6])).dropna()
ds_stacked_stations_filtered = ds_aws_stacked.sel(Station=station).reset_index('X')

fig, axs = plt.subplots(1, 3, figsize=(text_width, text_width/3),dpi=300)#,frameon=False)

ax=axs[0]
ds_stacked_stations_filtered['Temperature'].plot(ax=ax,hue='Station')
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
                                         alpha=0.7)
ax.annotate('b.',xy=(0.03,0.95),xycoords='axes fraction')

ax=axs[2]
gdf_single_station_june[['Temperature']].hist(bins=10,
                                              ax=ax,
                                              edgecolor='k',
                                              linewidth=0.2,
                                              grid=False,
                                              density=1,
                                              alpha=0.7,
                                              label='June')
gdf_single_station_jan[['Temperature']].hist(bins=10,
                                             ax=ax,
                                             edgecolor='k',
                                             linewidth=0.2,
                                             grid=False,
                                             density=1,
                                             alpha=0.7,
                                             label='January')
ax.legend(loc=9)
ax.annotate('c.',xy=(0.03,0.95),xycoords='axes fraction')

for ax in axs[1:]:
    ax.set_title('')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Density')

plt.tight_layout()
fig.savefig(f"{results_path}fig01.pdf", dpi=300, bbox_inches="tight")

# %% Figure Histograms for Temporal Distribution 
fig, axs = plt.subplots(1, 2, figsize=(text_width*2/3, text_width/3),dpi=300)#,frameon=False)

ax=axs[0]
df = df_all_combined.groupby('Year').agg(Temperature_Records=('Temperature', 'count'))
df.index = df.index.astype('int')
df.plot.bar(ax=ax,
            edgecolor='k',
            linewidth=0.2,
            alpha=0.7)
for i, t in enumerate(ax.get_xticklabels()):
    if (i % 5) != 0:
        t.set_visible(False)
ax.get_legend().remove()
ax.set_xlabel('Year')
ax.set_ylabel('Number of Records')
ax.annotate('a.',xy=(-0.08,0.95),xycoords='axes fraction')
ax.grid(alpha=0.2,color='k',linestyle='--')

ax=axs[1]
df = df_all_combined.groupby('Month').agg(Temperature_Records=('Temperature', 'count'))
df.index = df.index.astype('int')
df.plot.bar(ax=ax,
            edgecolor='k',
            linewidth=0.2,
            alpha=0.7)
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.get_legend().remove()
ax.set_xlabel('Month')
ax.set_ylabel('Number of Records')
ax.annotate('b.',xy=(-0.08,0.95),xycoords='axes fraction')

ax.grid(alpha=0.2,color='k',linestyle='--')

for ax in axs:
    ax.set_title('')

plt.tight_layout()

fig.savefig(f"{results_path}fig02.pdf", dpi=300, bbox_inches="tight")


# %% Figure Histograms for Spatial Distribution 
fig, axs = plt.subplots(1, 3, figsize=(text_width, text_width/3),dpi=300)#,frameon=False)

ax=axs[0]
df = df_all_combined.groupby('Station').agg(Temperature_Records=('Temperature', 'count'))
df.hist(bins=40,
        ax=ax,
        edgecolor='k',
        linewidth=0.2,
        grid=False,
        alpha=0.7)#,cumulative=True)
ax.set_xlabel('Number of Records')
ax.set_ylabel('Number of Stations')
ax.annotate('a.',xy=(-0.08,0.95),xycoords='axes fraction')
ax.grid(alpha=0.2,color='k',linestyle='--')
print('Percent of Stations with <10 years of data: '+f'{(df[df<120].count()/df.count())[0]*100}')

ax=axs[1]
df = df_all_combined.groupby('Station').agg(Temperature_Records=('Temperature', 'count'))
df = df_all_combined.groupby('Station')[['Elevation(m)']].first()
df.hist(bins=40,
        ax=ax,
        edgecolor='k',
        linewidth=0.2,
        grid=False,
        alpha=0.7)#,cumulative=True)
ax.set_xlabel('Elevation')
ax.set_ylabel('Number of Stations')
ax.annotate('a.',xy=(-0.08,0.95),xycoords='axes fraction')
ax.grid(alpha=0.2,color='k',linestyle='--')

ax=axs[2]
df = df_all_combined.groupby('Station').agg(Temperature_Records=('Temperature', 'count'))
df = df_all_combined.groupby('Station')[['Lat(℃)']].first()
df.hist(bins=40,
        ax=ax,
        edgecolor='k',
        linewidth=0.2,
        grid=False,
        alpha=0.7)#,cumulative=True)
ax.set_xlabel('Latitude')
ax.set_ylabel('Number of Stations')
ax.annotate('b.',xy=(-0.08,0.95),xycoords='axes fraction')
ax.grid(alpha=0.2,color='k',linestyle='--')

for ax in axs:
    ax.set_title('')

plt.tight_layout()

fig.savefig(f"{results_path}fig03.pdf", dpi=300, bbox_inches="tight")

# %% Spatial Distribution
count_group = ['Station','Lat(℃)','Lon(℃)']#,'Elevation(m)','Institution']
df_all_combined_count = gdf_all_combined.groupby(count_group).agg(
    Temperature_Records=('Temperature', 'count'),
    geometry=('geometry','first')
    )
count_group.append('Month')
df_all_combined_count_months = gdf_all_combined.groupby(count_group).agg(
    Temperature_Records=('Temperature', 'count'),
    geometry=('geometry','first')
    )

df_all_combined_count = df_all_combined_count.reset_index()
df_all_combined_count_months = df_all_combined_count_months.reset_index()

df_all_combined_count_june = df_all_combined_count_months.where(
    df_all_combined_count_months.Month.isin([6])
    ).dropna()

gdf_all_combined_count = gpd.GeoDataFrame(
    df_all_combined_count,
    geometry=df_all_combined_count.geometry,
    crs=gdf_all_combined.crs
)
gdf_all_combined_count_june = gpd.GeoDataFrame(
    df_all_combined_count_june,
    geometry=df_all_combined_count_june.geometry,
    crs=gdf_all_combined.crs
)

gdf_all_combined_count_filtered = gdf_all_combined_count[
    gdf_all_combined_count['Lat(℃)']<-60]

stations_filter_june_mask = gdf_all_combined_count_june['Station'].isin(stations_filter_june)
gdf_all_combined_count_filtered_june = gdf_all_combined_count_june[stations_filter_june_mask]

map_proj = ccrs.SouthPolarStereo()
fig, axs = plt.subplots(1, 2, subplot_kw={"projection": map_proj}, figsize=(text_width, text_width/2),dpi=300)#,frameon=False)
# ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

ax=axs[0]
gdf_all_combined_count_filtered.to_crs(map_proj).plot(
    ax=ax,
    markersize=gdf_all_combined_count_filtered['Temperature_Records']/20,
    edgecolor='k',
    linewidths=0.3,
    legend=True
    )
ax.annotate('a.',xy=(0.05,0.95),xycoords='axes fraction')

ax=axs[1]
gdf_all_combined_count_filtered_june.to_crs(map_proj).plot(
    ax=ax,
    markersize=gdf_all_combined_count_filtered_june['Temperature_Records']/2,
    edgecolor='k',
    linewidths=0.3,
    legend=True
    )
ax.annotate('b.',xy=(0.05,0.95),xycoords='axes fraction')

for ax in axs:
    gdf_icesheet.to_crs(map_proj).boundary.plot(
        ax=ax,
        color='k',
        linewidth=0.3,
        alpha=0.4)
    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                    draw_labels=True,
                    lw=0.2,
                    ls=':',
                    rotate_labels=False,
                    color='k',
                    alpha=0.8)

    gl.ylocator = mticker.FixedLocator([-90, -85,-80,-75,-70,-65,-60])
    gl.xlabel_style = {'size': 5}#, 'color': 'gray'}
    gl.ylabel_style = {'size': 5}#, 'color': 'gray'}
    plt.draw()
    for ea in gl.label_artists:
        pos = ea.get_position()
        if pos[0] == 150:
            ea.set_position([120, pos[1]])

    ax.set_axis_off()

ax=axs[0]
bins = np.array([10,50,100,200,300,400])
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
                markersize=np.sqrt(b/20),
                label=str(int(b)),
            )
            for i, b in enumerate(bins)
        ],
        loc=3,
        fontsize = legend_fontsize,
        framealpha=0,
    )
)
ax=axs[1]
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

plt.tight_layout()
plt.show()

fig.savefig(f"{results_path}fig04.pdf", dpi=300, bbox_inches="tight")

# %% Figure: Pair Plot showing Correlation between Variables

vars = ['Mean_Temperature','Std_Temperature','Elevation(m)','Lat(℃)']
months = [6]
df = df_all_combined[df_all_combined['Month'].isin(months)]

stations_filter_june_mask = df['Station'].isin(stations_filter_june)
df = df[stations_filter_june_mask]

index_group = ['Station',
               'Lat(℃)',
               'Lon(℃)',
               'Elevation(m)',
               'Institution',
               'glat',
               'glon',
               'grid_latitude',
               'grid_longitude'
               ]
df = df.groupby(index_group).agg(
    Mean_Temperature=('Temperature', 'mean'),
    Std_Temperature=('Temperature', 'std'),
    Temperature_Records=('Temperature', 'count')
    ).reset_index()
df = df[vars].dropna()

g = sns.pairplot(df,
            vars=vars,
            plot_kws=dict(linewidth=0.3,alpha=0.6),
            diag_kws=dict(bins=20),
            kind='scatter',
            corner=True,
)
g.map_lower(corrfunc)
g.axes[0,0].set_ylabel('Temperature Mean / $^\circ$C')
g.axes[1,0].set_ylabel('Temperature Monthly Std.Dev. / $^\circ$C')
g.axes[2,0].set_ylabel('Elevation / m')
g.axes[3,0].set_ylabel('Latitude / $^\circ$')
g.axes[3,0].set_xlabel('Temperature Mean / $^\circ$C')
g.axes[3,1].set_xlabel('Temperature Monthly Std.Dev. / $^\circ$C')
g.axes[3,2].set_xlabel('Elevation / m')
g.axes[3,3].set_xlabel('Latitude / $^\circ$')

g.fig.set_size_inches(text_width, text_width)
g.fig.savefig(f"{results_path}fig05.pdf", dpi=300, bbox_inches="tight")


# %% Figure: Pair Plot showing Correlation between Variables (Jan & June)

vars = ['Month','Mean_Temperature','Std_Temperature','Elevation(m)','Lat(℃)']
months = [1,6]
df = df_all_combined[df_all_combined['Month'].isin(months)]
index_group = ['Station',
               'Lat(℃)',
               'Lon(℃)',
               'Elevation(m)',
               'Institution',
               'glat',
               'glon',
               'grid_latitude',
               'grid_longitude',
               'Month']
df = df.groupby(index_group).agg(
    Mean_Temperature=('Temperature', 'mean'),
    Std_Temperature=('Temperature', 'std'),
    Temperature_Records=('Temperature', 'count')
    ).reset_index()
stations_filter_mask = df['Station'].isin(stations_filter)
df = df[stations_filter_mask]
df = df[vars].dropna()

def month_label(row):
   if row['Month'] == 1:
      return 'January'
   if row['Month'] == 6:
      return 'June'
   
def corrfunc_combined(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    if kws['label'] == 'January':
        ax.annotate(r'$\rho_{January}$'+f' = {r:.2f}', xy=(.1, .9),
                    xycoords=ax.transAxes,
                    bbox=dict(boxstyle="round",fc="w",alpha=0.8)
                    )
    else:
        ax.annotate(r'$\rho_{June}$'+f' = {r:.2f}', xy=(.1, .75),
                    xycoords=ax.transAxes,
                    bbox=dict(boxstyle="round", fc="w",alpha=0.8)
                    )
   
df['Month_Name'] = df.apply(month_label, axis=1)
df = df.sort_values('Month')

g = sns.pairplot(df,
            vars=vars[1:],
            plot_kws=dict(linewidth=0.3,alpha=0.5),
            # diag_kws=dict(bins=20),
            kind='scatter',
            corner=True,
            hue='Month_Name'
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
# g.fig.savefig(f"{results_path}fig05.pdf", dpi=300, bbox_inches="tight")

# # %% Figure: Pair Plot showing Correlation between Variables

# vars = ['Elevation(m)','Lat(℃)','Mean_Temperature','Std_Temperature']
# df = df_all_combined_grouped[vars].dropna()

# g = sns.pairplot(df,
#             vars=vars,
#             corner=True,
#             plot_kws=dict(line_kws={'color':'red'}),
#             diag_kws=dict(bins=20),
#             kind='reg',
# )
# g.map_lower(regressfunc)
# g.axes[0,0].set_ylabel('Elevation / m')
# g.axes[1,0].set_ylabel('Latitude / $^\circ$')
# g.axes[2,0].set_ylabel('Temperature Mean / $^\circ$C')
# g.axes[3,0].set_ylabel('Temperature Monthly Std.Dev. / $^\circ$C')
# g.axes[3,0].set_xlabel('Elevation / m')
# g.axes[3,1].set_xlabel('Latitude / $^\circ$')
# g.axes[3,2].set_xlabel('Temperature Mean / $^\circ$C')
# g.axes[3,3].set_xlabel('Temperature Monthly Std.Dev. / $^\circ$C')

# g.fig.set_size_inches(text_width, text_width)

# %% Figure Spatial Plot of # of Temperature Records

# count_group = ['Station','Lat(℃)','Lon(℃)','Elevation(m)','Institution']
# df_all_combined_count = gdf_all_combined.groupby(count_group).agg(
#     Temperature_Records=('Temperature', 'count'),
#     geometry=('geometry','first')
#     )

# df_all_combined_count = df_all_combined_count.reset_index()

# gdf_all_combined_count = gpd.GeoDataFrame(
#     df_all_combined_count,
#     geometry=df_all_combined_count.geometry,
#     crs=gdf_all_combined.crs
# )

# gdf_all_combined_count_filtered = gdf_all_combined_count[
#     gdf_all_combined_count['Lat(℃)']<-60]

# map_proj = ccrs.SouthPolarStereo()
# fig, axs = plt.subplots(1, 2, subplot_kw={"projection": map_proj}, figsize=(text_width, text_width/2),dpi=300)#,frameon=False)

# ax=axs[0]
# gdf_all_combined_count_filtered.to_crs(map_proj).plot(
#     ax=ax,
#     markersize=gdf_all_combined_count['Temperature_Records']/20,
#     edgecolor='k',
#     linewidths=0.3,
#     legend=True
#     )

# gdf_icesheet.to_crs(map_proj).boundary.plot(
#     ax=ax,
#     color='k',
#     linewidth=0.3,
#     alpha=0.4)
# gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,lw=0.2,ls=':',rotate_labels=False,color='k',alpha=0.8)
# gl.xlabel_style = {'size': 5}#, 'color': 'gray'}
# gl.ylabel_style = {'size': 5}#, 'color': 'gray'}
# ax.annotate('a.',xy=(0.05,0.95),xycoords='axes fraction')

# bins = np.array([10,50,100,200,300,400])
# ax.add_artist(
#     ax.legend(
#         handles=[
#             mlines.Line2D(
#                 [],
#                 [],
#                 color="tab:blue",
#                 markeredgecolor="k",
#                 markeredgewidth=0.3,
#                 lw=0,
#                 marker="o",
#                 markersize=np.sqrt(b/20),
#                 label=str(int(b)),
#             )
#             for i, b in enumerate(bins)
#         ],
#         loc=3,
#         fontsize = legend_fontsize,
#         framealpha=0,
#     )
# )

# img = mpimg.imread('/home/jez/Bias_Correction_Application/Antarctica_Elevation.png')
# ax=axs[1]
# imgplot = ax.imshow(img)
# ax.annotate('b.',xy=(0.15,0.95),xycoords='axes fraction')

# for ax in axs:
#     ax.set_axis_off()

# img_leg = mpimg.imread('/home/jez/Bias_Correction_Application/Antarctica_Elevation_Legend.png')
# newax = fig.add_axes([0.99, 0., 0.08, 1.], anchor='NE', zorder=-1)#,frameon=False)
# newax.imshow(img_leg)
# newax.set_axis_off()

# plt.tight_layout()

# fig.savefig(f"{results_path}fig02.pdf", dpi=300, bbox_inches="tight")
