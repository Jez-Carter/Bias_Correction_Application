# %% Importing Packages
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import pearsonr

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w"))

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
outerboundary_shapefile = f'{shapefiles_path}/cst10_polygon.shp'

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
gdf_outerboundary = gpd.read_file(outerboundary_shapefile)

index_group = ['Station','Lat(℃)','Lon(℃)','Elevation(m)','Institution','glat','glon','grid_latitude', 'grid_longitude']
df_all_combined_grouped = df_all_combined.groupby(index_group).agg(
    Mean_Temperature=('Temperature', 'mean'),
    Std_Temperature=('Temperature', 'std'),
    Temperature_Records=('Temperature', 'count')
    ).reset_index()

# %% Figure Histogram of # of Stations against # of Records 
fig, axs = plt.subplots(1, 3, figsize=(text_width, 5*cm),dpi=300)#,frameon=False)

ax=axs[0]
df = df_all_combined.groupby('Station').agg(Temperature_Records=('Temperature', 'count'))
df.hist(bins=40,ax=ax,edgecolor='k',linewidth=0.2,grid=False)#,cumulative=True)
ax.set_xlabel('Number of Records')
ax.set_ylabel('Number of Stations')
ax.annotate('a.',xy=(-0.08,0.95),xycoords='axes fraction')
ax.grid(alpha=0.2,color='k',linestyle='--')

ax=axs[1]
df = df_all_combined.groupby('Year').agg(Temperature_Records=('Temperature', 'count'))
df.index = df.index.astype('int')
df.plot.bar(ax=ax,edgecolor='k',linewidth=0.2)
for i, t in enumerate(ax.get_xticklabels()):
    if (i % 5) != 0:
        t.set_visible(False)
ax.get_legend().remove()
ax.set_xlabel('Year')
ax.set_ylabel('Number of Records')
ax.annotate('b.',xy=(-0.08,0.95),xycoords='axes fraction')
ax.grid(alpha=0.2,color='k',linestyle='--')

ax=axs[2]
df = df_all_combined.groupby('Month').agg(Temperature_Records=('Temperature', 'count'))
df.index = df.index.astype('int')
df.plot.bar(ax=ax,edgecolor='k',linewidth=0.2)
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.get_legend().remove()
ax.set_xlabel('Month')
ax.set_ylabel('Number of Records')
ax.annotate('c.',xy=(-0.08,0.95),xycoords='axes fraction')
ax.grid(alpha=0.2,color='k',linestyle='--')

for ax in axs:
    ax.set_title('')

plt.tight_layout()

fig.savefig(f"{results_path}fig01.pdf", dpi=300, bbox_inches="tight")

# %% Figure Spatial Plot of # of Temperature Records

count_group = ['Station','Lat(℃)','Lon(℃)','Elevation(m)','Institution']
df_all_combined_count = gdf_all_combined.groupby(count_group).agg(
    Temperature_Records=('Temperature', 'count'),
    geometry=('geometry','first')
    )

df_all_combined_count = df_all_combined_count.reset_index()

gdf_all_combined_count = gpd.GeoDataFrame(
    df_all_combined_count,
    geometry=df_all_combined_count.geometry,
    crs=gdf_all_combined.crs
)

gdf_all_combined_count_filtered = gdf_all_combined_count[
    gdf_all_combined_count['Lat(℃)']<-60]

map_proj = ccrs.SouthPolarStereo()
fig, axs = plt.subplots(1, 2, subplot_kw={"projection": map_proj}, figsize=(text_width, text_width/2),dpi=300)#,frameon=False)

ax=axs[0]
gdf_all_combined_count_filtered.to_crs(map_proj).plot(
    ax=ax,
    markersize=gdf_all_combined_count['Temperature_Records']/20,
    edgecolor='k',
    linewidths=0.3,
    legend=True
    )

gdf_icesheet.to_crs(map_proj).boundary.plot(
    ax=ax,
    color='k',
    linewidth=0.3,
    alpha=0.4)
gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,lw=0.2,ls=':',rotate_labels=False,color='k',alpha=0.8)
gl.xlabel_style = {'size': 5}#, 'color': 'gray'}
gl.ylabel_style = {'size': 5}#, 'color': 'gray'}
ax.annotate('a.',xy=(0.05,0.95),xycoords='axes fraction')

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

img = mpimg.imread('/home/jez/Bias_Correction_Application/Antarctica_Elevation.png')
ax=axs[1]
imgplot = ax.imshow(img)
ax.annotate('b.',xy=(0.15,0.95),xycoords='axes fraction')

for ax in axs:
    ax.set_axis_off()

img_leg = mpimg.imread('/home/jez/Bias_Correction_Application/Antarctica_Elevation_Legend.png')
newax = fig.add_axes([0.99, 0., 0.08, 1.], anchor='NE', zorder=-1)#,frameon=False)
newax.imshow(img_leg)
newax.set_axis_off()

plt.tight_layout()

fig.savefig(f"{results_path}fig02.pdf", dpi=300, bbox_inches="tight")

# %% Figure: Pair Plot showing Correlation between Variables

vars = ['Mean_Temperature','Std_Temperature','Elevation(m)','Lat(℃)']
df = df_all_combined_grouped[vars].dropna()

g = sns.pairplot(df,
            vars=vars,
            plot_kws=dict(linewidth=0.3,alpha=0.6),
            diag_kws=dict(bins=20),
            kind='scatter',
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
# %%
