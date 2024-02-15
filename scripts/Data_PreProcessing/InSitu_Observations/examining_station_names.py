# %% Importing Packages
import numpy as np
import plotly.express as px
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1
legend_fontsize=6
cm = 1/2.54  # centimeters in inches
text_width = 17.68*cm
page_width = 21.6*cm

import jax
rng_key = jax.random.PRNGKey(1)
jax.config.update("jax_enable_x64", True)


# %% Loading data
inpath = "/home/jez/DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy"
scenario = np.load(
    inpath, allow_pickle="TRUE"
).item()
ds_aws_june_filtered = scenario['ds_aws_june_filtered']

ds_aws_june_filtered['Temperature_Records'] = ds_aws_june_filtered.count('Year')['Temperature']

df = ds_aws_june_filtered['Temperature_Records'].to_dataframe()
df = df.reset_index()

# %%
fig = px.scatter(df.reset_index(),
                x="glon",
                y="glat",
                hover_name='Station',
                size='Temperature_Records',
                )#, color='species')
fig.show()

# %%
shapefiles_path = f'/home/jez/DSNE_ice_sheets/Jez/AWS_Observations/Shp/'
icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
gdf_icesheet = gpd.read_file(icehsheet_shapefile)
rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)

icehsheet_mask_shapefile = f'{shapefiles_path}/cst10_polygon.shp'
gdf_icesheet_mask = gpd.read_file(icehsheet_mask_shapefile)
gdf_icesheet_mask = gdf_icesheet_mask.explode().iloc[[61]]
gdf_icesheet_mask = gdf_icesheet_mask.reset_index().drop(columns=['level_0','level_1'])

# %%
fig, ax = plt.subplots(1, 1, figsize=(text_width, text_width/1.5),dpi=300)#,frameon=False)
# stations = ['Larsen Ice Shelf']
stations = ['Henry']

df_filtered = df[df['Station'].isin(stations)]
ax.scatter(
    df_filtered.glon,
    df_filtered.glat,
)
gdf_icesheet_rotatedcoords.boundary.plot(ax=ax, color="k", linewidth=0.1)

# %%
