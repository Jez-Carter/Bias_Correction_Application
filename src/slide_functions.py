####### Importing Packages #######
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from scipy.stats import linregress,pearsonr

####### Paths #######
base_path = '/home/jez/'
shapefiles_path = f'{base_path}DSNE_ice_sheets/Jez/AWS_Observations/Shp/'

####### Ice Sheet Shapefile ######
icehsheet_shapefile = f'{shapefiles_path}/CAIS.shp'
gdf_icesheet = gpd.read_file(icehsheet_shapefile)
rotated_coord_system = ccrs.RotatedGeodetic(
    13.079999923706055,
    0.5199999809265137,
    central_rotated_longitude=180.0,
    globe=None,
)
gdf_icesheet_rotatedcoords = gdf_icesheet.to_crs(rotated_coord_system)

####### Background Map Functions #######
def background_map(ax,map_proj,extent):
    ax.set_extent(extent, ccrs.PlateCarree())
    gdf_icesheet.to_crs(map_proj).boundary.plot(
        ax=ax,
        color='k',
        linewidth=0.3,
        alpha=0.4)
    ax.set_axis_off()

def background_map_rotatedcoords(ax):
    gdf_icesheet_rotatedcoords.boundary.plot(
        ax=ax,
        color='k',
        linewidth=0.3,
        alpha=0.4)
    ax.set_axis_off()

####### Hexagonal Grid Functions #######

def create_grid(gdf=None, bounds=None, n_cells=10, overlap=False, crs="EPSG:29902"):
    """Create square grid that covers a geodataframe area
    or a fixed boundary with x-y coords
    returns: a GeoDataFrame of grid polygons
    see https://james-brennan.github.io/posts/fast_gridding_geopandas/
    """

    import geopandas as gpd
    import shapely

    if bounds != None:
        xmin, ymin, xmax, ymax= bounds
    else:
        xmin, ymin, xmax, ymax= gdf.total_bounds

    # get cell size
    cell_size = (xmax-xmin)/n_cells
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            x1 = x0-cell_size
            y1 = y0+cell_size
            poly = shapely.geometry.box(x0, y0, x1, y1)
            #print (gdf.overlay(poly, how='intersection'))
            grid_cells.append( poly )

    cells = gpd.GeoDataFrame(grid_cells, columns=['geometry'],
                                     crs=crs)
    if overlap == True:
        cols = ['grid_id','geometry','grid_area']
        cells = cells.sjoin(gdf, how='inner').drop_duplicates('geometry')
    return cells

def create_hex_grid(gdf=None, bounds=None, n_cells=10, overlap=False, crs="EPSG:29902"):

    """Hexagonal grid over geometry.
    See https://sabrinadchan.github.io/data-blog/building-a-hexagonal-cartogram.html
    """

    from shapely.geometry import Polygon
    import geopandas as gpd
    if bounds != None:
        xmin, ymin, xmax, ymax= bounds
    else:
        xmin, ymin, xmax, ymax= gdf.total_bounds

    unit = (xmax-xmin)/n_cells
    a = np.sin(np.pi / 3)
    cols = np.arange(np.floor(xmin), np.ceil(xmax), 3 * unit)
    rows = np.arange(np.floor(ymin) / a, np.ceil(ymax) / a, unit)

    #print (len(cols))
    hexagons = []
    for x in cols:
      for i, y in enumerate(rows):
        if (i % 2 == 0):
          x0 = x
        else:
          x0 = x + 1.5 * unit

        hexagons.append(Polygon([
          (x0, y * a),
          (x0 + unit, y * a),
          (x0 + (1.5 * unit), (y + unit) * a),
          (x0 + unit, (y + (2 * unit)) * a),
          (x0, (y + (2 * unit)) * a),
          (x0 - (0.5 * unit), (y + unit) * a),
        ]))

    grid = gpd.GeoDataFrame({'geometry': hexagons},crs=crs)
    grid["grid_area"] = grid.area
    grid = grid.reset_index().rename(columns={"index": "grid_id"})
    if overlap == True:
        cols = ['grid_id','geometry','grid_area']
        grid = grid.sjoin(gdf, how='inner').drop_duplicates('geometry')
    return grid

def plot_hex_grid(ax,map_proj,n_cells):
    hexgr = create_hex_grid(gdf_icesheet.to_crs(map_proj), n_cells=n_cells, overlap=True, crs=map_proj)
    hexgr.plot(fc="none", ec='black',linewidth=0.05,ax=ax)

####### Marker Size Scatter Plot Legend #######

def markersize_legend(ax, bins, scale_multipler, legend_fontsize=10,loc=3,ncols=1,columnspacing=0.8,handletextpad=0.1,bbox=(0.,0.)):
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
            loc=loc,
            fontsize = legend_fontsize,
            ncols=ncols,
            columnspacing=columnspacing,
            handletextpad=handletextpad,
            bbox_to_anchor=bbox,
            framealpha=0,
        )
    )

####### Correlation & Regression Function #######

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w"))

def regressfunc(x, y, ax=None, **kws):
    """Plot the linear regression coefficients in the top left hand corner of a plot."""
    ax = ax or plt.gca()
    slope, intercept, rvalue, pvalue, stderr = linregress(x=x, y=y)
    ax.annotate(f'y = {intercept:.2f}+x*{slope:.2f}', xy=(.1, .9), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w"))
    ax.annotate(f'slope_stderr = {stderr:.2f}', xy=(.1, .7), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w"))

def corrfunc_combined(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    if kws['label'] == 'AWS':
        ax.annotate(r'$\rho_{AWS}$'+f' = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w",alpha=0.8))
    else:
        ax.annotate(r'$\rho_{CM}$'+f' = {r:.2f}', xy=(.1, .75), xycoords=ax.transAxes,bbox=dict(boxstyle="round", fc="w",alpha=0.8))
