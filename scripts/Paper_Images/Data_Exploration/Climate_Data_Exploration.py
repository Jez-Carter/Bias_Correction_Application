# %% Importing Packages
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
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
base_path = '/home/jez/'
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
outerboundary_shapefile = f'{shapefiles_path}/cst10_polygon.shp'
gdf_icesheet = gpd.read_file(icehsheet_shapefile)
gdf_outerboundary = gpd.read_file(outerboundary_shapefile)

rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)

###### Native MAR Grid ######
mar_mask_path = f'{base_path}Bias_Correction_Application/mar_land_sea_mask.nc'
ds_mar_native = xr.open_dataset(mar_mask_path)
mar_coord_system = ccrs.epsg(3031)
gdf_icesheet_marcoords = gdf_icesheet.to_crs(mar_coord_system)

###### Climate Model Data ######
base_path = '/home/jez/'
CORDEX_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'
ds_climate = xr.open_dataset(CORDEX_path)
ds_climate = ds_climate.sel(Model='MAR(ERA5)')
ds_climate['Temperature']=ds_climate['Temperature'] - 273.15
ds_climate['Latitude']=ds_climate['latitude']

# %% Showing Grids of Interpolated vs Native MAR Data
fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=300)#,frameon=False)

ax=axs[0]
ax.grid(which='minor', alpha=0.2)
gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
ax.set_xticks(ds_climate['glon'].isel(grid_latitude=0), minor=True)
ax.set_yticks(ds_climate['glat'].isel(grid_longitude=0), minor=True)
ax.set_xlim(-24, -19.5) #16
ax.set_ylim(3, 7)
ax.annotate('a.',xy=(-0.05,0.95),xycoords='axes fraction')

ax=axs[1]
ax.grid(True, which='minor', axis='both', linestyle='-', color='k',linewidth=0.1)
gdf_icesheet_marcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
ax.set_xticks(ds_mar_native.x*1000, minor=True)
ax.set_yticks(ds_mar_native.y*1000, minor=True)
ax.set_xlim(-2.4e6, -1.9e6)
ax.set_ylim(0.95e6, 1.4e6)
ax.annotate('b.',xy=(-0.05,0.95),xycoords='axes fraction')
# m2km = lambda x, _: f'{x/1000:g}'
# ax.xaxis.set_major_formatter(m2km)
# ax.yaxis.set_major_formatter(m2km)

for ax in axs:
    ax.set_xticks([], major=True)
    ax.set_yticks([], major=True)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.remove_overlapping_locs = False
    ax.yaxis.remove_overlapping_locs = False

fig.savefig(f"{results_path}fig_a01.pdf", dpi=300, bbox_inches="tight")

# %% Probability Density of Temperature
ds_climate_southpole = ds_climate[['Temperature','Elevation','Latitude']].sel(
    grid_longitude=180,grid_latitude=0,method='nearest')
months = [1]
ds_climate_southpole_filtered = ds_climate_southpole.where(ds_climate_southpole.month.isin(months)).dropna('Time')

fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=300)#,frameon=False)

ax=axs[0]
df = ds_climate_southpole.to_dataframe()[['Temperature']]
df.hist(bins=40,ax=ax,edgecolor='k',linewidth=0.2,grid=False,density=1,alpha=0.7)
ax.annotate('a.',xy=(0.05,0.95),xycoords='axes fraction')

ax=axs[1]
df = ds_climate_southpole_filtered.to_dataframe()[['Temperature']]
df.hist(bins=10,ax=ax,edgecolor='k',linewidth=0.2,grid=False,density=1,alpha=0.7)
ax.annotate('b.',xy=(0.05,0.95),xycoords='axes fraction')

for ax in axs:
    ax.set_title('')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Density')

# fig.savefig(f"{results_path}fig06.pdf", dpi=300, bbox_inches="tight")

# %% Figure MAR Temperature Monthly Mean and Std.Dev 

months = [6]
ds_climate_filtered = ds_climate.where(ds_climate.month.isin(months)).dropna('Time')
ds_climate_filtered = ds_climate_filtered.where(ds_climate_filtered.LSM)

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
                'label':'Mean June Temperature'}
)
ax.annotate('a.',xy=(0.05,0.95),xycoords='axes fraction')

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
                   'label':'Std.Dev. June Temperature'}
)
ax.annotate('b.',xy=(0.05,0.95),xycoords='axes fraction')

for ax in axs:
    ax.set_title('')
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_xlabel('Grid Longitude')
    ax.set_ylabel('Grid Latitude')

    ax.set_axis_off()

plt.tight_layout()
fig.savefig(f"{results_path}fig06.pdf", dpi=300, bbox_inches="tight")


# %% Figure: Pair Plot showing Correlation between Variables
months = [6]
ds = ds_climate.where(ds_climate.LSM & ds_climate.month.isin(months))

ds['Mean_Temperature'] = ds['Temperature'].mean('Time')
ds['Std_Temperature'] = ds['Temperature'].std('Time')

ds = ds[['Mean_Temperature','Std_Temperature','Elevation','Latitude']]
# ds = ds.sel(grid_latitude=slice(None,None,10),grid_longitude=slice(None,None,10))
df = ds.to_dataframe().reset_index().dropna()

g = sns.pairplot(df,
            vars=ds.data_vars,
            kind='hist',
            corner=True,
)
g.map_lower(corrfunc)
# g.set_axis_labels('Total Bill Amount ($)', 'Tip Amount ($)')

g.axes[0,0].set_ylabel('Temperature Mean / $^\circ$C')
g.axes[1,0].set_ylabel('Temperature Monthly Std.Dev. / $^\circ$C')
g.axes[2,0].set_ylabel('Elevation / m')
g.axes[3,0].set_ylabel('Latitude / $^\circ$')
g.axes[3,0].set_xlabel('Temperature Mean / $^\circ$C')
g.axes[3,1].set_xlabel('Temperature Monthly Std.Dev. / $^\circ$C')
g.axes[3,2].set_xlabel('Elevation / m')
g.axes[3,3].set_xlabel('Latitude / $^\circ$')

g.fig.set_size_inches(text_width, text_width)

g.fig.savefig(f"{results_path}fig07.pdf", dpi=300, bbox_inches="tight")


