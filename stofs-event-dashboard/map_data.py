import geopandas as gpd
import pandas as pd
from typing import Dict
import space_time_bounds
import write_output


def save_geopackage(
    stb: space_time_bounds.eventSpaceTimeBounds, 
    df_list: Dict[str, gpd.GeoDataFrame],
    output_config: dict
) -> None:
    """
    """
    # Get wind swaths.
    df_swaths = gpd.GeoDataFrame()
    for reg in ['box', '34kt', '50kt', '64kt']:
        shape = stb.get_region(reg)
        if shape:
            df_swaths = pd.concat(
                [df_swaths, 
                 gpd.GeoDataFrame(data={'region':[reg]},
                                  geometry=[shape])]
            )
            
    # Format stations.
    keep_cols = ['nos_id','nws_id', 'name', 'state', 
                 'longitude', 'latitude', 'geometry']
    df_stations = pd.concat([v for v in df_list.values()])[keep_cols]
    df_stations = df_stations.drop_duplicates()
    df_stations['station_type'] = ''
    for k, v in df_list.items():
        for st in v['nos_id']:
            st_row = df_stations['nos_id'] == st
            df_stations.loc[st_row, 'station_type'] = ', '.join(
                [k, df_stations.loc[st_row, 'station_type'].values[0]]
            )

    # Save dataframes to geopackage.
    data_dir = write_output.get_output_dir(output_config, stb)
    file_name = 'map_data.gpkg'
    df_swaths.to_file(
        data_dir / file_name, 
        layer='regions', 
        driver='GPKG'
    )
    df_stations.to_file(
        data_dir / file_name, 
        layer='stations', 
        driver='GPKG'
    )
    import pdb; pdb.set_trace()
    #
    return None