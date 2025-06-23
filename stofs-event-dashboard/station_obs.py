""" Handle observational data from stations.

Functions
---------
fetch_coops_multistation_df(st_list, start_time, end_time, product, kwargs)
    Wrapper for "searvey" function that fetches a single station, to allow
    serial gathering of multiple stations and combining them into a single
    pandas data frame.

"""


import numpy as np
import pandas as pd
import datetime
import pathlib
import logging
import searvey
from searvey._coops_api import fetch_coops_station
from typing import Iterable, Union, Any
from space_time_bounds import eventSpaceTimeBounds
import write_output 


logger = logging.getLogger(__name__)


def fetch_coops_multistation_df(
        st_list: Iterable, 
        start_time: Union[pd.Timestamp, np.datetime64, datetime.datetime],
        end_time: Union[pd.Timestamp, np.datetime64, datetime.datetime], 
        product: Union[searvey._coops_api.COOPS_Product, str], 
        **kwargs: Any
    ) -> pd.DataFrame:
    """Fetch multiple stations' observations and output a single data frame.

    Keyword arguments are passed through to searvey's fetch_coops_station(...)
    function, which is called once per station in st_list.

    Parameters
    ----------
    st_list
        A list, pandas Index, pandas Series, or similar, of NOS station IDs.
    start_time
        The start time for which station observations are fetched.
    end_time
        The end time for which station observations are fetched.
    product
        The variable for which data is fetched. Will usually be a string
        corresponding to one of the COOPS_Product options.

    Returns
    -------
    pandas.DataFrame
        Table with (station, time) multi-index containing data for all 
        station locations for the specified product.

    """    
    result = pd.DataFrame()
    
    for nos_id in st_list:
        
        #print(f"Fetching observation data for NOS ID {nos_id}")
        
        # Wrangle the datetime formats to avoid searvey TimeZone warnings....
        if type(start_time) in [np.ndarray, np.datetime64]:
            start_time = pd.DatetimeIndex([pd.to_datetime(start_time)], tz='utc')[0]
        if type(end_time) in [np.ndarray, np.datetime64]:
            end_time = pd.DatetimeIndex([pd.to_datetime(end_time)], tz='utc')[0]
            
        # Download one station's worth of data.
        station_df = fetch_coops_station(
            station_id=nos_id,
            start_date=start_time,
            end_date=end_time,
            product=product,
            **kwargs
        )
        # Add "station" as a column and convert to multiindex.
        station_df.insert(0, 'station', nos_id)
        station_df = station_df.reset_index().set_index(['station', 'time'])
        
        # Concatenate with other stations.
        try:
            result = pd.concat(
                [result, station_df],
                axis=0,
                join="outer",
                ignore_index=False,
                sort=False
            )
        except:
            print(f"Warning: cannot concatenate station data frame for NOS ID {nos_id}: Skipping.")
            continue
            
    return result


def save_obs(
    stations: pd.DataFrame, 
    stb: eventSpaceTimeBounds, 
    var_name: str, 
    output_config: dict
) -> None:
    """Fetch and save observation data.

    Checks if/what period is needed, then fetches and saves to parquet files.

    Parameters
    ----------

    Returns
    -------
    
    """
    # Parse options.
    if var_name in ['cwl', 'water_level']:
        dir_var = 'cwl'
        query_var = 'water_level'
        cols_out = ['value']
    elif var_name in ['pressure', 'air_pressure']:
        dir_var = 'pressure'
        query_var = 'air_pressure'
        cols_out = ['value']
    elif var_name in ['wind']:
        dir_var = 'wind'
        query_var = 'wind'
        cols_out = ['wind_speed', 'degree', 'direction', 'gust', 
                    'u_wind', 'v_wind']
    else:
        raise ValueError(f'var_name {var_name} not recognized.')
    data_dir = write_output.get_output_dir(output_config, stb)
    out_dir = pathlib.Path(data_dir) / dir_var / 'obs'

    run_query = False
    append_data = False
    
    # Check if the obs files exist and need to be updated.
    existing_files = list(out_dir.glob('*.parquet'))
    if len(existing_files) == 0:
        # If no files exist, we want to run the query and save files. 
        run_query = True
        query_start_datetime = stb.start_datetime
    else:
        # If files exist, we only want to run a query if the end time
        # of the files is less than the end time of the event. Then we
        # want to append new data, instead of overwriting the file.
        # TODO: Check if there is a faster way to do this than reading all the files.
        latest_end_datetime = max([
            pd.read_parquet(fn).reset_index()['time'].max() 
            for fn in existing_files
        ]).to_pydatetime()
        if latest_end_datetime < stb.end_datetime:
            run_query = True
            append_data = True
            query_start_datetime = max(
                latest_end_datetime + datetime.timedelta(minutes=1),
                stb.start_datetime
            )

    # If needed, download and save the data.
    if run_query:
        logger.info(f'Fetching {len(stations.index)} stations for {var_name} data.')

        obs = fetch_coops_multistation_df(
            stations.nos_id,
            query_start_datetime,
            stb.end_datetime,
            query_var,
            datum=output_config['output_datum'] 
        )
        
        # Post-processing for wind data.
        if var_name in ['wind']:
            obs['u_wind'] = \
                obs['speed'] * np.cos(np.radians(270.0 - obs['degree']))
            obs['v_wind'] = \
                obs['speed'] * np.sin(np.radians(270.0 - obs['degree']))
            obs = obs.rename(columns={'speed':'wind_speed'})
            
        # Save the data in parquet files.
        write_output.df_sealens_parquet(obs, out_dir, 
                                        column_names=cols_out, 
                                        append=append_data)
    else:
        logger.info(f'Not fetching any {var_name} data, probably because it already exists for given date range ({stb.start_datetime} -- {stb.end_datetime})')
    #
    return
        
