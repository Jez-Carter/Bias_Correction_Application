# %% Importing Packages

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from src.slide_functions import background_map_rotatedcoords, plot_hex_grid, rotated_coord_system, markersize_legend, regressfunc
from scipy.spatial import distance

base_path = '/home/jez/'
repo_path = f'{base_path}Bias_Correction_Application/'
internal_datapath = f'{repo_path}Slide_Images/Data/'
external_datapath = f'{base_path}DSNE_ice_sheets/Jez/Slides/'
scenario_path = f'{base_path}DSNE_ice_sheets/Jez/Bias_Correction/Data/scenario_real.npy'

# %% Loading Data
ds_aws_stacked = xr.open_dataset(f'{external_datapath}ds_aws_stacked.nc')
ds_climate = xr.open_dataset(f'{external_datapath}ds_climate.nc')
ds_climate_nearest_stacked = xr.open_dataset(f'{external_datapath}ds_climate_nearest_stacked.nc')

######## Consistent Names ########
ds_aws_stacked = ds_aws_stacked.rename({'Elevation(m)':'Elevation',
                       'Lat(℃)':'Latitude',
                       'June Mean Temperature':'Mean June Temperature',
                       'Month':'month'})

# %% Climate Model High Resolution All Against AWS Sites

vars = ['Mean June Temperature','Elevation','Latitude']
predictors = ['Elevation','Latitude']
response = 'Mean June Temperature'
cond = (ds_climate.LSM)

fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

for ax, predictor in zip(axs, predictors):
    sns.regplot(data=ds_climate.where(cond)[vars].to_dataframe(),
                x=response,
                y=predictor,
                order=1,
                scatter=False,
                ax=ax)

    sns.histplot(data=ds_climate.where(cond)[vars].to_dataframe(),
                x=response,
                y=predictor,
                ax=ax)
    
    sns.regplot(data=ds_aws_stacked[vars].to_dataframe(),
            x=response,
            y=predictor,
            order=1,
            ax=ax)

# %% Climate Model High Resolution Nearest Grid Cells All Against AWS Sites
vars = ['Mean June Temperature','Elevation','Latitude']
predictors = ['Elevation','Latitude']
response = 'Mean June Temperature'

fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

for ax, predictor in zip(axs, predictors):
    sns.regplot(data=ds_climate_nearest_stacked[vars].to_dataframe(),
                x=response,
                y=predictor,
                order=1,
                ax=ax)

    sns.regplot(data=ds_aws_stacked[vars].to_dataframe(),
            x=response,
            y=predictor,
            order=1,
            ax=ax)
# %%

sns.regplot(data=ds_climate.where(cond)[vars].to_dataframe(),
            x="Mean June Temperature",
            y="Latitude",
            order=1,
            scatter=False,
            )

sns.histplot(data=ds_climate.where(cond)[vars].to_dataframe(),
            x="Mean June Temperature",
            # y="Elevation",
            y="Latitude",
            # bins=10,
            )

# %% Scatter Plots for NN Difference Comparisons






















# %% 



.rename({'Temperature':'LandOnly Temperature'})

f_aws_renamed = df_aws.rename(columns={'Elevation(m)':'Elevation',
                                        'Lat(℃)':'latitude',
                                        'Station':'Nearest_Station',
                                        'Month':'month'})


# %% Elevation against Temperature (High Resolution Climate Model)

vars = ['Mean June Temperature','Elevation','Latitude']
cond = (ds_climate.LSM) #& (ds_climate['Elevation']>100)

sns.regplot(data=ds_climate.where(cond)[vars].to_dataframe(),
            x="Mean June Temperature",
            # y="Elevation",
            y="Latitude",
            order=1,
            scatter=False,
            # scatter_kws={'alpha':0.5},
            )

sns.histplot(data=ds_climate.where(cond)[vars].to_dataframe(),
            x="Mean June Temperature",
            # y="Elevation",
            y="Latitude",
            # bins=10,
            )

vars = ['June Mean Temperature','Elevation(m)','Lat(℃)']
sns.regplot(data=ds_aws_stacked[vars].to_dataframe(),
            x="June Mean Temperature",
            # y="Elevation(m)",
            y="Lat(℃)",
            order=1,
            )

# %% Elevation against Temperature (High Resolution Climate Model)

vars = ['Mean June Temperature','Elevation','Latitude']
# cond = (ds_climate_nearest_stacked.LSM) #& (ds_climate['Elevation']>100)

sns.regplot(data=ds_climate_nearest_stacked[vars].to_dataframe(),
            x="Mean June Temperature",
            # y="Elevation",
            y="Latitude",
            order=1,
            # scatter=False,
            # scatter_kws={'alpha':0.5},
            )

# sns.histplot(data=ds_climate_nearest_stacked[vars].to_dataframe(),
#             x="Mean June Temperature",
#             # y="Elevation",
#             y="Latitude",
#             # bins=10,
#             )

vars = ['June Mean Temperature','Elevation(m)','Lat(℃)']
sns.regplot(data=ds_aws_stacked[vars].to_dataframe(),
            x="June Mean Temperature",
            # y="Elevation(m)",
            y="Lat(℃)",
            order=1,
            )



# %%


sns.regplot(data=ds_climate.where(ds_climate.LSM)[vars].to_dataframe(),
            x="Mean June Temperature",
            y="Elevation",
            order=1,
            scatter=False,
            # scatter_kws={'alpha':0.5},
            )

sns.histplot(data=ds_climate.where(ds_climate.LSM)[vars].to_dataframe(),
            x="Mean June Temperature",
            y="Elevation",
            # bins=10,
            )

vars = ['June Mean Temperature','Elevation(m)','Lat(℃)']
sns.regplot(data=ds_aws_stacked[vars].to_dataframe(),
            x="June Mean Temperature",
            y="Elevation(m)",
            order=1,
            )

# %% Elevation against Temperature (NN Climate Model)

vars = ['Mean June Temperature','Elevation','Latitude']

sns.regplot(data=ds_climate_nearest_stacked[vars].to_dataframe(),
            x="Mean June Temperature",
            y="Elevation",
            order=1,
            )

# sns.histplot(data=ds_climate_nearest_stacked[vars].to_dataframe(),
#             x="Mean June Temperature",
#             y="Elevation",
#             # bins=10,
#             )

vars = ['June Mean Temperature','Elevation(m)','Lat(℃)']
sns.regplot(data=ds_aws_stacked[vars].to_dataframe(),
            x="June Mean Temperature",
            y="Elevation(m)",
            order=1,
            )



# %%
# sns.regplot(data=ds_aws_stacked[['Elevation(m)','June Mean Temperature']].to_dataframe(),
#             x="June Mean Temperature",
#             y="Elevation(m)",
#             order=1,
#             scatter_kws={'alpha':0.5},
#             )


# %%  Pairplot Climate Model
vars = ['Mean June Temperature','Elevation','Latitude']

g = sns.pairplot(ds_climate.where(ds_climate.LSM)[vars].to_dataframe(),
            vars=vars,
            kind='reg',
            plot_kws={'line_kws':{'color':'red','scatter':False}},
            corner=True,
)

# %%


import matplotlib.pyplot as plt
import seaborn as sns

def plot_extra(x, y, **kwargs):
     if kwargs['label'] == first_label:
          sns.regplot(data=kwargs['data'], x=x.name, y=y.name, lowess=True, scatter=False, color=kwargs['color'])

def 



df = sns.load_dataset('iris')

first_label = df['species'][0]
pg = sns.pairplot(df, hue='species', plot_kws={'alpha': 0.5}, palette='turbo')

pg.map_offdiag(plot_extra, color='crimson', data=df)

legend_dict = {h.get_label(): h for h in pg.legend.legendHandles}  # the existing legend items
legend_dict['lowess regression'] = pg.axes[0, 1].lines[
     0]  # add the first line object of a non-diagonal ax to the legend

pg.legend.remove()  # remove existing legend
pg.add_legend(legend_dict, label_order=legend_dict.keys(), title='')  # create the new legend
plt.show()

# %%

vars = ['June Mean Temperature','Elevation(m)','Lat(℃)']
# fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

g = sns.pairplot(ds_aws_stacked[vars].to_dataframe(),
            vars=vars,
            kind='reg',
            # plot_kws={'line_kws':{'color':'red','scatter':False}},
            corner=True,
)

g.

g = sns.pairplot(ds_aws_stacked[vars].to_dataframe(),
            vars=vars,
            kind='reg',
            # plot_kws={'line_kws':{'color':'red','scatter':False}},
            corner=True,
)
# %%

# ds = ds.sel(grid_latitude=slice(None,None,10),grid_longitude=slice(None,None,10))
df = ds.to_dataframe().reset_index().dropna()

# g = sns.pairplot(df,
#             vars=ds.data_vars,
#             kind='hist',
#             corner=True,
# )



# %%

df = ds.to_dataframe().reset_index().dropna()

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















# %%






scenario = np.load(
    scenario_path, allow_pickle="TRUE",fix_imports=True,
).item()

ds_climate_coarse_june_stacked = scenario['ds_climate_coarse_june_stacked']
ds_climate_coarse_june_stacked_landonly = scenario['ds_climate_coarse_june_stacked_landonly'].rename({'Temperature':'LandOnly Temperature'})

ds_climate_coarse_june_stacked = xr.merge([ds_climate_coarse_june_stacked_landonly,ds_climate_coarse_june_stacked])
ds_climate_coarse_june = ds_climate_coarse_june_stacked.unstack()

ds_climate_coarse_june_stacked['Mean LandOnly Temperature'] = ds_climate_coarse_june_stacked['LandOnly Temperature'].mean('Time')

# Nearest Neighbors
nn_indecies = []
for point in scenario['ox']:
    nn_indecies.append(distance.cdist([point], scenario['cx']).argmin())

ds_climate_coarse_june_stacked_nn = ds_climate_coarse_june_stacked.isel({'X':nn_indecies})


# %%

ds_aws_stacked['June Mean Temperature'].dropna('Station')


# %%

sns.regplot(data=ds_climate_coarse_june_stacked_nn['Mean LandOnly Temperature'].to_dataframe(),
            x="Mean LandOnly Temperature",
            y="Elevation",
            order=1,
            scatter_kws={'alpha':0.5},
            )

# sns.regplot(data=ds_aws_stacked[['Elevation(m)','June Mean Temperature']].to_dataframe(),
#             x="June Mean Temperature",
#             y="Elevation(m)",
#             order=1,
#             scatter_kws={'alpha':0.5},
#             )



# %%

ds_climate_coarse_june_stacked.to_dataframe().index


# %%
ds_climate_coarse_june_stacked['Mean LandOnly Temperature'].to_dataframe()

# %%

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

ax.scatter(
    ds_climate_coarse_june_stacked['Elevation'],
    ds_climate_coarse_june_stacked['LandOnly Temperature'].mean('Time'),
    # s=ds_aws_stacked['June Temperature Records']*2,
    # c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
    alpha=0.2,
)

# ax.scatter(
#     ds_aws_stacked['Elevation(m)'],
#     ds_aws_stacked['June Mean Temperature'],
#     # s=ds_aws_stacked['June Temperature Records']*2,
#     # c=ds_aws_stacked['June Mean Temperature'],
#     cmap='jet',
#     vmin=-55,
#     vmax=-10,
#     edgecolor='w',
#     linewidths=0.5,
# )

regressfunc(x=ds_aws_stacked['Elevation(m)'],
             y=ds_aws_stacked['June Mean Temperature'],
             ax=ax)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10),dpi=300)#,frameon=False)

ax.scatter(
    ds_climate_coarse_june_stacked['Elevation'],
    ds_climate_coarse_june_stacked['LandOnly Temperature'].mean('Time'),
    # s=ds_aws_stacked['June Temperature Records']*2,
    # c=ds_aws_stacked['June Mean Temperature'],
    cmap='jet',
    vmin=-55,
    vmax=-10,
    edgecolor='w',
    linewidths=0.5,
    alpha=0.2,
)

# ax.scatter(
#     ds_aws_stacked['Elevation(m)'],
#     ds_aws_stacked['June Mean Temperature'],
#     # s=ds_aws_stacked['June Temperature Records']*2,
#     # c=ds_aws_stacked['June Mean Temperature'],
#     cmap='jet',
#     vmin=-55,
#     vmax=-10,
#     edgecolor='w',
#     linewidths=0.5,
# )

regressfunc(x=ds_aws_stacked['Elevation(m)'],
             y=ds_aws_stacked['June Mean Temperature'],
             ax=ax)


# %%
ds_climate_coarse_june_stacked_nn