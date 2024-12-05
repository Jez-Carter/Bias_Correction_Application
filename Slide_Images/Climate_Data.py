# %% Importing Packages
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

# %% Loading Data

###### Climate Model Data ######
base_path = '/home/jez/'
CORDEX_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/CORDEX_Combined.nc'
ds_climate = xr.open_dataset(CORDEX_path)
ds_climate = ds_climate.sel(Model='MAR(ERA5)')
ds_climate['Temperature'] = ds_climate['Temperature'] - 273.15
ds_climate['Latitude'] = ds_climate['latitude']

# Filter the climate data
ds_climate_filtered = ds_climate.where(ds_climate.LSM)

# Define the map projection and extent
map_proj = ccrs.SouthPolarStereo()
extent = [-180, 180, -90, -65]

# %% Mean Temperature Map

# Create the figure and axis with the specified projection
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": map_proj}, figsize=(10, 5), dpi=300)

# Plot the mean temperature
ds_climate_filtered['Temperature'].mean('Time').plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    vmin=-55,
    vmax=-10,
    cmap='jet',
    add_colorbar=False,
)

# Set the y-axis limits
ax.set_ylim(-25, 25)

# Remove the axis title and turn off the axis
ax.set_title('')
ax.set_axis_off()

plt.show()

