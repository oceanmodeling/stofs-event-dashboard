"""Process data for example dashboard using Hurricane Milton (2024).

Contains only the function "main".
Run from the command line without any additional arguments:
$ python milton_example.py

"""


import space_time_bounds
import shapely
import searvey
from station_obs import fetch_coops_multistation_df
import static_map
import pandas as pd
import pathlib
from write_parquet import df_to_sealens

import sys
sys.path.append('/home/jre/seanode')
sys.path.append('/home/jre/coastalmodeling-vdatum')
from seanode.api import get_surge_model_at_stations


def main():
    """Run STOFS dashboard data processing for hurricane Milton (2024) example event.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    storm_name = "milton"
    storm_year = 2024

    # Get event details.
    storm = space_time_bounds.get_nhc_windswath(storm_name, storm_year)
    
    # Get station observations.
    station_list = searvey.get_coops_stations(metadata_source='main')
    station_list = searvey.get_coops_stations(region=storm.region.buffer(1.0), 
                                              metadata_source='main')
    waterlevel_stations = station_list[(station_list['status'] == 'active') & 
                                       (station_list['station_type'] == 'waterlevels')]   
    waterlevel_obs = fetch_coops_multistation_df(
        waterlevel_stations.index,
        storm.start_time,
        storm.end_time,
        'water_level',
        datum='NAVD' 
    )  

    # Plot map.
    static_map.plot(storm_name, storm_year, 
                    {'50 kt':storm.region},
                    waterlevel_stations)
    
    # Get model data.
    stofs_2d_nowcast = get_surge_model_at_stations(
        'STOFS_2D_GLO',
        ['cwl_bias_corrected', 'cwl_raw'],
        waterlevel_stations.index,
        storm.start_time,
        storm.end_time,
        'nowcast',
        'points',
        'NAVD88',
        'AWS'
    )
    stofs_3d_nowcast = get_surge_model_at_stations(
        'STOFS_3D_ATL',
        ['cwl'],
        waterlevel_stations.index,
        storm.start_time,
        storm.end_time,
        'nowcast',
        'points',
        'NAVD88',
        'AWS'
    )
    
    # Save obs and model data to parquet files.
    df_to_sealens(
        waterlevel_obs,
        pathlib.Path('../data/obs'),
        column_name='value'
    )
    df_to_sealens(
        stofs_2d_nowcast,
        pathlib.Path('../data/models/stofs2d_nowcast_cwl_bias_corrected'),
        column_name='cwl_bias_corrected'
    )
    df_to_sealens(
        stofs_2d_nowcast,
        pathlib.Path('../data/models/stofs2d_nowcast_cwl_raw'),
        column_name='cwl_raw'
    )
    df_to_sealens(
        stofs_3d_nowcast,
        pathlib.Path('../data/models/stofs3d_nowcast_cwl_raw'),
        column_name='cwl'
    )
    #
    return


if __name__ == '__main__':
    main()
    print("Now run this command, and go to the specified URL:")
    print("    python -m panel serve dashboard*.py --dev --address=127.0.0.1 --port=5009 --allow-websocket-origin=localhost:5009 --allow-websocket-origin=127.0.0.1:5009  --log-level debug")
