# %% Importing Packages

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

base_path = '/home/jez/'
repo_path = f'{base_path}Bias_Correction_Application/'
internal_datapath = f'{repo_path}Slide_Images/Data/'
external_datapath = f'{base_path}DSNE_ice_sheets/Jez/Slides/'

# %% Loading Data

ds_aws_stacked = xr.open_dataset(f'{external_datapath}ds_aws_stacked.nc')
ds_climate_nearest_stacked = xr.open_dataset(f'{external_datapath}ds_climate_nearest_stacked.nc')

# %% Consistent Coordinates

def consistent_index(years,months):
    index = (years-1980)*12 + months 
    return index

xticks = np.arange(0,45*12,12*5)
xticklabels = np.arange(1980,2025,5)

ds_aws_stacked = ds_aws_stacked.reindex({"X": consistent_index(ds_aws_stacked.Year,ds_aws_stacked.Month)})

ds_climate_nearest_stacked = ds_climate_nearest_stacked.reset_index('Time')
ds_climate_nearest_stacked = ds_climate_nearest_stacked.reindex({"Time": consistent_index(ds_climate_nearest_stacked.year,ds_climate_nearest_stacked.month)})

# %% Single Site Full Time Series

fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

station = 'Manuela'
# station = 'Byrd'

ds_climate_nearest_stacked.sel(Nearest_Station = station)['Temperature'].plot(ax=ax,
                                                                              hue='Station',
                                                                              alpha=0.7,
                                                                              label='Climate Model',
                                                                              marker='x',
                                                                              ms=1,
                                                                              color='tab:blue',
                                                                              linewidth=1.0)

ds_aws_stacked.sel(Station = station)['Temperature'].plot(ax=ax,
                                                          hue='Station',
                                                          alpha=0.7,
                                                          label='Weather Station',
                                                          marker='x',
                                                          ms=1,
                                                          color='tab:orange',
                                                          linewidth=1.5)

# ax.annotate('All Months',xy=(0.03,0.95),xycoords='axes fraction')

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)#,rotation = 90)
ax.set_ylabel('Temperature')
ax.set_xlabel('Time')
ax.legend()
ax.set_title('')

plt.tight_layout()
plt.show()


# %% Single Site June Time Series
month = 6 
fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

station = 'Manuela'
# station = 'Byrd'

ds_climate_nearest_stacked.sel(Nearest_Station = station).where(ds_climate_nearest_stacked['month']==month,drop=True)['Temperature'].plot(ax=ax,
                                                                                                                                hue='Station',
                                                                                                                                alpha=0.7,
                                                                                                                                label='Climate Model',
                                                                                                                                marker='x',
                                                                                                                                ms=1,
                                                                                                                                color='tab:orange',
                                                                                                                                linewidth=1.0)

ds_aws_stacked.sel(Station = station).where(ds_aws_stacked['Month']==month,drop=True)['Temperature'].plot(ax=ax,
                                                                                                hue='Station',
                                                                                                alpha=0.7,
                                                                                                label='Weather Station',
                                                                                                marker='x',
                                                                                                ms=1,
                                                                                                color='tab:blue',
                                                                                                linewidth=1.5)

ax.annotate('June',xy=(0.03,0.95),xycoords='axes fraction')

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)#,rotation = 90)
ax.set_ylabel('Temperature')
ax.set_xlabel('Time')
ax.legend()
ax.set_title('')

plt.tight_layout()
plt.show()


# %% Two Sites June Time Series
month = 6 
fig, axs = plt.subplots(2, 1, figsize=(10, 7), dpi=300)

for ax,station in zip(axs,['Manuela','Byrd']):
    ds_aws_stacked.sel(Station = station).where(ds_aws_stacked['Month']==month,drop=True)['Temperature'].plot(ax=ax,
                                                                                                    hue='Station',
                                                                                                    alpha=0.7,
                                                                                                    label='Weather Station',
                                                                                                    marker='x',
                                                                                                    ms=2)
    ds_climate_nearest_stacked.sel(Nearest_Station = station).where(ds_climate_nearest_stacked['month']==month,drop=True)['Temperature'].plot(ax=ax,
                                                                                                                                    hue='Station',
                                                                                                                                    alpha=0.7,
                                                                                                                                    label='Climate Model',
                                                                                                                                    marker='x',
                                                                                                                                    ms=2)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)#,rotation = 90)
    ax.set_ylabel('Temperature')
    ax.set_xlabel('Time')
    ax.legend()
    ax.set_title('')

plt.tight_layout()
plt.show()


# %% Histograms All Months

station = 'Manuela'

fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

ds_aws_stacked.sel(Station=station).to_dataframe()[['Temperature']].hist(bins=40,
                                         ax=ax,
                                         edgecolor='k',
                                         linewidth=0.2,
                                         grid=False,
                                         density=1,
                                         alpha=0.7,
                                         label = 'Weather Station',
                                         )
ds_climate_nearest_stacked.sel(Nearest_Station=station).to_dataframe()[['Temperature']].hist(bins=40,
                                         ax=ax,
                                         edgecolor='k',
                                         linewidth=0.2,
                                         grid=False,
                                         density=1,
                                         alpha=0.7,
                                        label = 'Climate Model',
                                         )
ax.annotate('All Months',xy=(0.03,0.95),xycoords='axes fraction')
ax.set_title('')
ax.set_xlabel('Temperature')
ax.set_ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# %% Histograms June
month = 6 
station = 'Manuela'
ds_climate_nearest_stacked_june = ds_climate_nearest_stacked.where(
    ds_climate_nearest_stacked['month']==month,drop=True)

ds_aws_stacked_june = ds_aws_stacked.where(
    ds_aws_stacked['Month']==month,drop=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

ds_aws_stacked_june.sel(Station=station).to_dataframe()[['Temperature']].hist(bins=12,
                                         ax=ax,
                                         edgecolor='k',
                                         linewidth=0.2,
                                         grid=False,
                                         density=1,
                                         alpha=0.7,
                                         label = 'Weather Station',
                                         )
ds_climate_nearest_stacked_june.sel(Nearest_Station=station).to_dataframe()[['Temperature']].hist(bins=12,
                                         ax=ax,
                                         edgecolor='k',
                                         linewidth=0.2,
                                         grid=False,
                                         density=1,
                                         alpha=0.7,
                                        label = 'Climate Model',
                                         )
ax.annotate('June',xy=(0.03,0.95),xycoords='axes fraction')
ax.set_title('')
ax.set_xlabel('Temperature')
ax.set_ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# %% Quantile Mapped Correction Single Site June Time Series
ds_climate_nearest_stacked_june
ds_aws_stacked_june
from scipy.stats import norm

cdf_c = norm.cdf(
    ds_climate_nearest_stacked_june.sel(Nearest_Station=station)['Temperature'],
    ds_climate_nearest_stacked_june.sel(Nearest_Station=station)['Temperature'].mean(),
    ds_climate_nearest_stacked_june.sel(Nearest_Station=station)['Temperature'].std(),
)

c_corrected = norm.ppf(
        cdf_c,
        ds_aws_stacked_june.sel(Station=station)['Temperature'].mean(),
        ds_aws_stacked_june.sel(Station=station)['Temperature'].std(),
    )


fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

station = 'Manuela'
# station = 'Byrd'

ax.plot(ds_climate_nearest_stacked_june['Time'],
        c_corrected,
        alpha=0.7,
        label='Corrected Climate Model',
        marker='x',
        ms=1,
        color='tab:orange',
        linewidth=1.0)

ds_aws_stacked.sel(Station = station).where(ds_aws_stacked['Month']==month,drop=True)['Temperature'].plot(ax=ax,
                                                                                                hue='Station',
                                                                                                alpha=0.7,
                                                                                                label='Weather Station',
                                                                                                marker='x',
                                                                                                ms=1,
                                                                                                color='tab:blue',
                                                                                                linewidth=1.5)

ax.annotate('June',xy=(0.03,0.95),xycoords='axes fraction')

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)#,rotation = 90)
ax.set_ylabel('Temperature')
ax.set_xlabel('Time')
ax.legend()
ax.set_title('')

plt.tight_layout()
plt.show()

# %%










# %%
# station = 'Byrd'

ds_climate_nearest_stacked_june = ds_climate_nearest_stacked.where(
    ds_climate_nearest_stacked['month']==month)

ds_aws_stacked_june = ds_aws_stacked.where(
    ds_aws_stacked['Month']==month)

fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

ax=axs[0]
ds_aws_stacked.sel(Station=station).to_dataframe()[['Temperature']].hist(bins=40,
                                         ax=ax,
                                         edgecolor='k',
                                         linewidth=0.2,
                                         grid=False,
                                         density=1,
                                         alpha=0.7,
                                         label = 'Weather Station',
                                         )

ds_climate_nearest_stacked.sel(Nearest_Station=station).to_dataframe()[['Temperature']].hist(bins=40,
                                         ax=ax,
                                         edgecolor='k',
                                         linewidth=0.2,
                                         grid=False,
                                         density=1,
                                         alpha=0.7,
                                        label = 'Climate Model',
                                         )
ax.annotate('All Months',xy=(0.03,0.95),xycoords='axes fraction')

ax=axs[1]
ds_aws_stacked_june.sel(Station=station).to_dataframe()[['Temperature']].hist(bins=10,
                                         ax=ax,
                                         edgecolor='k',
                                         linewidth=0.2,
                                         grid=False,
                                         density=1,
                                         alpha=0.7,
                                        label = 'Weather Station',
                                         )

ds_climate_nearest_stacked_june.sel(Nearest_Station=station).to_dataframe()[['Temperature']].hist(bins=10,
                                         ax=ax,
                                         edgecolor='k',
                                         linewidth=0.2,
                                         grid=False,
                                         density=1,
                                         alpha=0.7,
                                         label = 'Climate Model',
                                         )
ax.annotate('June',xy=(0.03,0.95),xycoords='axes fraction')

for ax in axs:
    ax.set_title('')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()







