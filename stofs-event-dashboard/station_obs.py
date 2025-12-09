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
import geopandas as gpd
import shapely
import datetime
import pathlib
import logging
import searvey
from searvey._coops_api import fetch_coops_station
from typing import Iterable, Union, Any
from space_time_bounds import eventSpaceTimeBounds
import write_output 


logger = logging.getLogger(__name__)


def fetch_metadata(
        config_stations: dict,
        stb: eventSpaceTimeBounds
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch station lists from multiple searvey sources.
    """
    waterlevel_stations = pd.DataFrame()
    met_stations = pd.DataFrame()
    for src in config_stations['source_list']:
        if src == 'coops':
            wl_st, met_st = fetch_coops_metadata(config_stations, stb)
        elif src == 'ioc':
            wl_st, met_st = fetch_ioc_metadata(config_stations, stb)
        elif src == 'ndbc':
            wl_st, met_st = fetch_ndbc_metadata(config_stations, stb)
        elif src == 'usgs':
            wl_st, met_st = fetch_usgs_metadata(config_stations, stb)
        else:
            logger.warning(f'Station data source {src} not recognized. Must be one of [coops, ioc, ndbc, usgs].')
            wl_st = pd.DataFrame()
            met_st = pd.DataFrame()
        waterlevel_stations = pd.concat(
            [waterlevel_stations, wl_st],
            axis='index',
            join='outer',
            ignore_index=True,
            sort=False
        )
        met_stations = pd.concat(
            [met_stations, met_st],
            axis='index',
            join='outer',
            ignore_index=True,
            sort=False
        )

    if ('synth_stations' in config_stations.keys() and 
        len(config_stations['synth_stations']) > 0):
        st_inds = []
        st_lat = []
        st_lon = []
        st_names = []
        geometry = []
        for ill, ll in enumerate(config_stations['synth_stations']):
            if len(ll) >2:
                st_names.append(ll[2])
                st_inds.append('synth_' + str(ill) + '_' + ll[2].replace(' ', '_'))
            else:
                st_names.append(f'User-defined synthetic station #' + str(ill))
                st_inds.append('synth_' + str(ill) + '_' + 
                               "{:.1f}".format(ll[0]) + "N_" + 
                               "{:.1f}".format(ll[1]) + "E")
            st_lat.append(ll[0])
            st_lon.append(ll[1])
            geometry.append(shapely.Point(ll[1], ll[0]))
        synth_stations = gpd.GeoDataFrame(
            data={'station':st_inds,
                  'latitude':st_lat,
                  'longitude':st_lon,
                  'station_name':st_names},
            geometry=geometry
        )
        synth_stations['station_id_type'] = 'synthetic'
        waterlevel_stations = pd.concat(
            [waterlevel_stations, synth_stations.copy()],
            axis='index',
            join='outer',
            ignore_index=True,
            sort=False
        )
        met_stations = pd.concat(
            [met_stations, synth_stations.copy()],
            axis='index',
            join='outer',
            ignore_index=True,
            sort=False
        )
    return (waterlevel_stations, met_stations)


def fetch_coops_metadata(
        config_stations: dict,
        stb: eventSpaceTimeBounds
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch CO-OPS station lists.
    """
    station_list = searvey.get_coops_stations(metadata_source='main')
    station_list = searvey.get_coops_stations(
        region=stb.get_region(config_stations['bounds'])\
                  .buffer(config_stations['buffer']),
        metadata_source='main'
    )
    station_list = station_list.reset_index()
    station_list = station_list.rename(
        columns={'lat':'latitude', 'lon':'longitude',
                 'name':'station_name'}
    )
    station_list['station'] = station_list['nos_id']
    station_list['station_id_type'] = 'NOS'
    waterlevel_stations = station_list[
        (station_list['status'] == 'active') &
        (station_list['station_type'] == 'waterlevels')
    ]
    met_stations = station_list[
        (station_list['status'] == 'active') &
        (station_list['station_type'] == 'met')
    ]
    return (waterlevel_stations, met_stations)


def fetch_ioc_metadata(
        config_stations: dict,
        stb: eventSpaceTimeBounds
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch IOC station lists.
    """
    ioc_stations = searvey.get_ioc_stations(
        region=stb.get_region(config_stations['bounds'])\
                  .buffer(config_stations['buffer'])
    )
    ioc_stations = ioc_stations.reset_index().drop(columns='index')
    ioc_stations = ioc_stations.rename(
        columns={'lat':'latitude', 'lon':'longitude',
                 'location':'station_name'}
    )
    ioc_stations['station'] = ioc_stations['ioc_code']
    ioc_stations['station_id_type'] = 'IOC'
    return (ioc_stations, ioc_stations.copy())


def fetch_ndbc_metadata(
        config_stations: dict,
        stb: eventSpaceTimeBounds
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch NDBC station lists.
    """
    ndbc_stations = searvey.get_ndbc_stations(
        region=stb.get_region(config_stations['bounds'])\
                  .buffer(config_stations['buffer'])
    )
    ndbc_stations = ndbc_stations.reset_index().drop(columns='index')
    ndbc_stations = ndbc_stations.rename(
        columns={
            'Station':'station',
            'lat':'latitude', 
            'lon':'longitude',
            'name':'station_name'
        }
    )
    ndbc_stations['station_id_type'] = 'NDBC'
    return (pd.DataFrame(), ndbc_stations)


def fetch_usgs_metadata(
        config_stations: dict,
        stb: eventSpaceTimeBounds
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch USGS station lists.
    """
    usgs_stations = searvey.usgs.get_usgs_stations(
        region=stb.get_region(config_stations['bounds'])\
                  .buffer(config_stations['buffer'])
    )
    usgs_stations = usgs_stations.reset_index().drop(columns='index')
    usgs_stations = usgs_stations.rename(
        columns={
            'site_no':'station',
            'dec_lat_va':'latitude', 
            'dec_long_va':'longitude',
            'station_nm':'station_name',
            'agency_cd':'station_id_type'
        }
    )
    return (usgs_stations, pd.DataFrame())


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
            stations.station[stations.station_id_type == 'NOS'],
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
        
