import iris
from iris.analysis.cartography import unrotate_pole 
import numpy as np
from numpy import meshgrid
from iris.analysis.cartography import unrotate_pole
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point
import jax
jax.config.update("jax_enable_x64", True)

def grid_coords_to_2d_latlon_coords(ds,ref_file):
    ds_updated = ds.copy()
    rotated_grid_latitude = ds.grid_latitude.data
    rotated_grid_longitude = ds.grid_longitude.data
    rotated_grid_lons,rotated_grid_lats = meshgrid(rotated_grid_longitude, rotated_grid_latitude)
    cube = iris.load(ref_file)[0]
    cs = cube.coord_system()
    lons,lats = unrotate_pole(rotated_grid_lons,rotated_grid_lats, cs.grid_north_pole_longitude, cs.grid_north_pole_latitude)
    ds_updated = ds.assign_coords(
        latitude=(["grid_latitude","grid_longitude"], lats),
        longitude=(["grid_latitude","grid_longitude"], lons),
    )
    return (ds_updated)

def create_mask(da,gdf,projection=None):
    longitude = da.longitude
    latitude = da.latitude
    mask_shape = longitude.shape
    mask = np.empty(mask_shape,dtype=bool)

    if projection is not None:
        gdf = gdf.to_crs(projection)

    for i in range(mask_shape[0]):
        for j in range(mask.shape[1]):
            point = Point(longitude.data[i,j],latitude.data[i,j])
            point_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry=[point])
            if projection is not None:
                point_gdf = point_gdf.reset_index().to_crs(projection)
            if gdf.contains(point_gdf).values[0]:
                mask[i,j]=True
            else:
                mask[i,j]=False
    return(mask)

def regrid(cube,grid_cube,method):
    if isinstance(cube.data, np.ma.MaskedArray):
        #Replacing masked values with zero value
        cube.data = cube.data.filled(0)
    #rotating coordinates to equator so distances are approximately euclidean 
    lons,lats = cube.coord('longitude').points,cube.coord('latitude').points
    rot_lons,rot_lats = unrotate_pole(lons,lats, 180, 0)    
    #sample points:
    points = np.array(list(zip(rot_lats.ravel(),rot_lons.ravel())))
    values = cube.data.ravel()
    #new grid points:
    grid_lons, grid_lats = grid_cube.coord('longitude').points,grid_cube.coord('latitude').points
    rot_grid_lons,rot_grid_lats = unrotate_pole(grid_lons,grid_lats, 180, 0)  
    #interpolating:
    regridded_data = griddata(points, values, (rot_grid_lats, rot_grid_lons), method=method, fill_value = 0)
    cube_regridded = iris.cube.Cube(
        regridded_data,
        long_name='cube_regridded',
        aux_coords_and_dims=[(grid_cube.coord('latitude'),grid_cube.coord_dims('latitude')),(grid_cube.coord('longitude'),grid_cube.coord_dims('longitude'))]
        )
    return(cube_regridded[:])
