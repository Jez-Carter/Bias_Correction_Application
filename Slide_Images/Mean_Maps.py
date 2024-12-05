# %% Importing Packages
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from src.slide_functions import background_map, background_map_rotatedcoords, plot_hex_grid, markersize_legend, regressfunc
import xarray as xr

base_path = '/home/jez/'
repo_path = f'{base_path}Bias_Correction_Application/'
internal_datapath = f'{repo_path}Slide_Images/Data/'
external_datapath = f'{base_path}DSNE_ice_sheets/Jez/Slides/'

# %% Loading Data
ds_aws_stacked = xr.open_dataset(f'{external_datapath}ds_aws_stacked.nc')
ds_climate = xr.open_dataset(f'{external_datapath}ds_climate.nc')

months = 6 # June
ds_climate_filtered = ds_climate.where(ds_climate.month.isin(months)).dropna('Time')
ds_climate_filtered = ds_climate_filtered.where(ds_climate_filtered.LSM)

# %%
ds_climate_filtered

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

ax.scatter(
    ds_climate_filtered['Elevation'].mean('Time'),
    ds_climate_filtered['Temperature'].mean('Time'),
    # s=ds_aws_stacked['June Temperature Records']*2,
    # c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
    alpha=0.2,
)

ax.scatter(
    ds_aws_stacked['Elevation(m)'],
    ds_aws_stacked['June Mean Temperature'],
    # s=ds_aws_stacked['June Temperature Records']*2,
    # c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
)

regressfunc(x=ds_aws_stacked['Elevation(m)'],
             y=ds_aws_stacked['June Mean Temperature'],
             ax=ax)

# %%
sns.regplot(
    data=mpg, x="weight", y="horsepower",
    ci=99, marker="x", color=".3", line_kws=dict(color="r"),
)

# def regressfunc(x, y, ax=None, **kws):
#     """Plot the linear regression coefficients in the top left hand corner of a plot."""
#     ax = ax or plt.gca()
#     slope, intercept, rvalue, pvalue, stderr = linregress(x=x, y=y)
#     ax.annotate(f'y = {intercept:.2f}+x*{slope:.2f}', xy=(.1, .9), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w"))
#     ax.annotate(f'slope_stderr = {stderr:.2f}', xy=(.1, .7), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w"))


# %% Plotting Mean Temperature Map AWS Sites
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)

ax.scatter(
    ds_aws_stacked.glon,
    ds_aws_stacked.glat,
    s=ds_aws_stacked['June Temperature Records']*2,
    c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
)

ax.set_ylim([-26,26])
ax.set_axis_off()
markersize_legend(ax, [1,5,10,15,20,25,30,35,40], scale_multipler=2, legend_fontsize=10,loc=8,ncols=9,columnspacing=0.3,handletextpad=-0.4,bbox=(0.4,0.15))

plt.tight_layout()

                       

# %% Plotting Mean Temperature Map Climate Model & AWS Sites

fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)

ds_climate_filtered['Temperature'].mean('Time').plot.pcolormesh(
    x='glon',
    y='glat',
    ax=ax,
    alpha=0.9,
    vmin=-55,
    vmax=-10,
    cmap='jet',
    add_colorbar=False,
    # cbar_kwargs = {'fraction':0.030,
    #             'pad':0.02,
    #             'label':'Mean January Temperature'}
)

ax.scatter(
    ds_aws_stacked.glon,
    ds_aws_stacked.glat,
    s=ds_aws_stacked['June Temperature Records']*2,
    c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
)


# markersize_legend(ax, [1,5,10,15,20,25,30,35,40], scale_multipler=2, legend_fontsize=10)
ax.set_ylim([-26,26])
ax.set_axis_off()
ax.set_title('')
                       
plt.tight_layout()

# ds_aws_stacked.plot.scatter(
#     x='glon',
#     y='glat',
#     hue='Mean Temperature',
#     markersize='Temperature Records',
#     ax=ax,
#     alpha=0.9,
#     vmin=-55,
#     vmax=-10,
#     cmap='jet',
#     cbar_kwargs = {'fraction':0.030,
#                 'pad':0.02,
#                 'label':'Mean January Temperature'},
# )

# %%
ds_aws_stacked = ds_aws_stacked.where(ds_aws_stacked['June Temperature Records']>=5)#.mean('X')['June Mean Temperature']
# ds_aws_stacked['June Mean Temperature'] = ds_aws_stacked.where(ds_aws_stacked['Month']==6).mean('X')['Temperature']

# %%
ds_aws_stacked[['June Temperature Records','June Mean Temperature']].dropna('Station')#[ds_aws_stacked['June Temperature Records']>5]


# %%
ds.drop

# %%
ds_aws_stacked


# %% Plotting Mean Temperature Map AWS Sites

fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

background_map_rotatedcoords(ax)

# ds_aws_stacked[['June Temperature Records','June Mean Temperature']].dropna('Station').plot.scatter(
#     x='glon',
#     y='glat',
#     hue='June Mean Temperature',
#     markersize='June Temperature Records',
#     ax=ax,
#     alpha=0.9,
#     vmin=-55,
#     vmax=-10,
#     cmap='jet',
#     cbar_kwargs = {'fraction':0.030,
#                 'pad':0.02,
#                 'label':'Mean January Temperature'},
# )

# ds_aws_stacked.plot.scatter(
#     x='glon',
#     y='glat',
#     hue='June Mean Temperature',
#     markersize='June Temperature Records',
#     ax=ax,
#     alpha=0.9,
#     vmin=-55,
#     vmax=-10,
#     cmap='jet',
#     cbar_kwargs = {'fraction':0.030,
#                 'pad':0.02,
#                 'label':'Mean January Temperature'},
# )


ax.scatter(
    ds_aws_stacked.glon,
    ds_aws_stacked.glat,
    s=ds_aws_stacked['June Temperature Records']*2,
    c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
)

#NOTE ds_aws_stacked.plot.scatter( doesn't get the markersize right for some reason.

def markersize_legend(ax, bins, scale_multipler, legend_fontsize=10):
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
                    markersize=np.sqrt(b*scale_multipler),
                    label=str(int(b)),
                )
                for i, b in enumerate(bins)
            ],
            loc=3,
            fontsize = legend_fontsize,
            framealpha=0,
        )
    )

bins = np.array([1,5,10,15,20,25,30,35,40])
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
                markersize=np.sqrt(b*2),
                label=str(int(b)),
            )
            for i, b in enumerate(bins)
        ],
        loc=3,
        fontsize = 10,
        framealpha=0,
    )
)

# %%

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

ds_aws_stacked.plot.scatter(
    x='glon',
    y='glat',
    hue='June Mean Temperature',
    markersize='June Temperature Records',
    ax=ax,
    alpha=0.9,
    vmin=-55,
    vmax=-10,
    cmap='jet',
    cbar_kwargs = {'fraction':0.030,
                'pad':0.02,
                'label':'Mean January Temperature'},
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

ds_aws_stacked[['June Temperature Records','June Mean Temperature']].dropna('Station').drop_indexes('Station').plot.scatter(
    x='glon',
    y='glat',
    # hue='June Mean Temperature',
    markersize='June Temperature Records',
    ax=ax,
    alpha=0.9,
    vmin=-55,
    vmax=-10,
    cmap='jet',
    # cbar_kwargs = {'fraction':0.030,
    #             'pad':0.02,
    #             'label':'Mean January Temperature'},
)

# %%
ds_aws_stacked[['June Temperature Records','June Mean Temperature']].dropna('Station').drop_indexes('Station')




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