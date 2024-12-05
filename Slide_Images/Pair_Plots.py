# %% Importing Packages
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from src.slide_functions import background_map, background_map_rotatedcoords, plot_hex_grid, markersize_legend, regressfunc, corrfunc
import xarray as xr
import seaborn as sns

base_path = '/home/jez/'
repo_path = f'{base_path}Bias_Correction_Application/'
internal_datapath = f'{repo_path}Slide_Images/Data/'
external_datapath = f'{base_path}DSNE_ice_sheets/Jez/Slides/'

# %% Loading Data
ds_aws_stacked = xr.open_dataset(f'{external_datapath}ds_aws_stacked.nc')
ds_climate = xr.open_dataset(f'{external_datapath}ds_climate.nc')

# months = 6 # June
# ds_climate_filtered = ds_climate.where(ds_climate.month.isin(months)).dropna('Time')
# ds_climate_filtered = ds_climate_filtered.where(ds_climate_filtered.LSM)


# %% Figure: Pair Plot showing Correlation between Variables
months = [6]
ds = ds_climate.where(ds_climate.LSM & ds_climate.month.isin(months))

ds['Mean_Temperature'] = ds['Temperature'].mean('Time')
ds['Std_Temperature'] = ds['Temperature'].std('Time')

# ds = ds[['Mean_Temperature','Std_Temperature','Elevation','Latitude']]
ds = ds[['Mean_Temperature','Elevation','Latitude']]

# %%

# ds = ds.sel(grid_latitude=slice(None,None,10),grid_longitude=slice(None,None,10))
df = ds.to_dataframe().reset_index().dropna()

# g = sns.pairplot(df,
#             vars=ds.data_vars,
#             kind='hist',
#             corner=True,
# )
g = sns.pairplot(df,
            vars=ds.data_vars,
            kind='reg',
            plot_kws={'line_kws':{'color':'red','scatter':False}},
            corner=True,
)
# g = sns.pairplot(iris, kind="reg", plot_kws={'line_kws':{'color':'red'}})

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

g.fig.set_size_inches(10, 10)

plt.show()


# %%

df

# g.fig.savefig(f"{results_path}fig07.pdf", dpi=300, bbox_inches="tight")


