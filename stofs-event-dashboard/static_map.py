"""Plot a static map showing station locations and storm wind swaths.

Functions
---------
plot(storm_name, storm_year, shape_dict, station_df)
    Plot station locations and region (e.g., wind swath) on a map; save file.

"""


import matplotlib.pyplot as plt
import pandas
import geopandas as gpd
import shapely
from cartopy.feature import NaturalEarthFeature
import pathlib
from typing import Dict


def plot(storm_name: str, 
         storm_year: int, 
         shape_dict: Dict[str, shapely.Polygon], 
         station_df: pandas.DataFrame) -> None:
    """Create and save a map showing station locations and region outline.

    A common usage of the shape_dict would be to plot the outlines of 
    wind swath(s) for a storm.

    Parameters
    ----------
    storm_name
        The name of the storm, usually lowercase, e.g., "milton". 
    storm_year
        The year in which the storm occured.
    shape_dict 
        A dictionary whose values are shapely Polygons that will be
        plotted on the map, and whose keys are used to label the Polygon.
    station_df
        A dataframe containing stations to be plotted on the map. 
        A point is plotted for each row. Locations require "lat" and
        "lon" columns. Stations are labeled by their index value. 

    Returns
    -------
    None

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

    # For now just saving to a hard-coded directory , the same as 
    # where the parquet files are saved.
    save_dir = pathlib.Path('../data')
    save_dir.mkdir(parents=True, exist_ok=True)

    figure.savefig(save_dir / f'{storm_name}_{storm_year}/static_map.png',
                   bbox_inches='tight')
    
    return
