# %% Importing Packages
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

###### Climate Model Data ######
base_path = '/home/jez/'
CORDEX_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'
ds_climate = xr.open_dataset(CORDEX_path)
ds_climate['Temperature']=ds_climate['Temperature'] - 273.15
ds_climate['Mean_Temperature']=ds_climate['Temperature'].mean('Time')
ds_climate['Std_Temperature']=ds_climate['Temperature'].std('Time')
ds_climate['Latitude']=ds_climate['latitude']

# %% Figure MAR Temperature Monthly Mean and Std.Dev 

ds_climate_mar_mean = ds_climate['Mean_Temperature'].sel(Model='MAR(ERA5)').where(ds_climate.LSM)
ds_climate_mar_std = ds_climate['Std_Temperature'].sel(Model='MAR(ERA5)').where(ds_climate.LSM)

fig, axs = plt.subplots(1, 2, figsize=(text_width, text_width/2),dpi=300)#,frameon=False)

ax=axs[0]
ds_climate_mar_mean.plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    vmin=-55,
    vmax=0,
    cmap='jet',
    cbar_kwargs = {'fraction':0.030,
                'pad':0.02,
                'label':'Mean Temperature'}
)
ax.annotate('a.',xy=(0.05,0.95),xycoords='axes fraction')

ax=axs[1]
ds_climate_mar_std.plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    cmap='jet',
    cbar_kwargs = {'fraction':0.030,
                   'pad':0.04,
                   'label':'Temperature Monthly Std.Dev.'}
)
ax.annotate('b.',xy=(0.05,0.95),xycoords='axes fraction')

rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)

for ax in axs:
    ax.set_title('')
    gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)
    ax.set_xlabel('Grid Longitude')
    ax.set_ylabel('Grid Latitude')

    ax.set_axis_off()

plt.tight_layout()
fig.savefig(f"{results_path}fig03.pdf", dpi=300, bbox_inches="tight")

# %% Figure: Pair Plot showing Correlation between Variables

ds = ds_climate.where(ds_climate.LSM)
ds = ds[['Mean_Temperature','Std_Temperature','Elevation','Latitude']]
# ds = ds.sel(grid_latitude=slice(None,None,10),grid_longitude=slice(None,None,10))
df = ds.to_dataframe().reset_index().dropna()

g = sns.pairplot(df,
            vars=ds.data_vars,
            kind='hist',
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

g.fig.savefig(f"{results_path}fig04.pdf", dpi=300, bbox_inches="tight")
# %%
