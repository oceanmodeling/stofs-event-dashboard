import matplotlib.pyplot as plt
import geopandas as gpd
from cartopy.feature import NaturalEarthFeature
import pathlib

def plot(storm_name, 
         storm_year, 
         shape_dict, 
         station_df):
    """
    """ 
    figure, axis = plt.subplots(1, 1)
    figure.set_size_inches(8, 6) 

    for k in shape_dict.keys():
        axis.plot(*shape_dict[k].exterior.xy, 
                  c='#fdcc8a', label=k)
        axis.text(shape_dict[k].exterior.coords.xy[0][0],
                  shape_dict[k].exterior.coords.xy[1][0],
                  k)

    axis.scatter(station_df.lon, station_df.lat, c='#525252')
    for ist, st in enumerate(station_df.index):
        axis.text(station_df.loc[st].lon + 0.2,
                  station_df.loc[st].lat,
                  st)

    xlim = axis.get_xlim()
    ylim = axis.get_ylim()

    gdf_countries = gpd.GeoSeries(
        NaturalEarthFeature(category='physical', scale='10m', name='land').geometries(), 
        crs=4326
    )
    gdf_countries.plot(color='lightgrey', ax=axis, zorder=-1)
    
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)

    axis.set_title(f'{storm_name} {storm_year}')
    figure.show()
    figure.savefig(pathlib.Path('../data') / f'{storm_name}_{storm_year}_static_map.png',
                   bbox_inches='tight')
    
    return
