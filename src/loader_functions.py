import iris
import numpy as np
from iris.analysis.cartography import unrotate_pole
from scipy.interpolate import griddata

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