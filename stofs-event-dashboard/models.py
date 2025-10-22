"""Functions to handle model data requests.
"""


import datetime
from xml.parsers.expat import model
import pandas as pd
import numpy as np
from typing import List
import pathlib
import logging
import traceback
import seanode
from space_time_bounds import eventSpaceTimeBounds
import write_output


logger = logging.getLogger(__name__)


def get_forecast_init_times(
    model: str, 
    start_datetime: datetime.datetime, 
    end_datetime: datetime.datetime
) -> List[datetime.datetime]:
    """Get list of initialization times for a forecast.

    Parameters
    ----------
    model
        The name/abbreviation of the model.
    start_datetime
        The starting date time for the period in which we want
        model forecast initializations.
    end_datetime
        The ending date time for the period in which we want
        model forecast initializations.

    Returns
    -------
    List of model's initialization datetimes between start and end dates.
    
    """
    if model in ['stofs_2d_glo']:
        model_tasker = seanode.models.stofs_2d_glo.STOFS2DGloTaskCreator()
    elif model in ['stofs_3d_atl']:
        model_tasker = seanode.models.stofs_3d_atl.STOFS3DAtlTaskCreator()
    else:
        raise ValueError(f'Model {model} not available in stofs-event-dashboard.')
    # Use the seanode model task creator to get the forecast initialization times.
    # (Note that "nowcast" in the function name here is intentional -- it gets
    #  multiple initialization times over a window.)
    (result, windows) = model_tasker.get_init_times_nowcast(start_datetime, 
                                                            end_datetime)
    return result


def get_model_forcing(model: str) -> str:
    """Returns the model name used to download data."""
    if model in ['stofs_2d_glo']:
        forcing_name = 'GFS'
    elif model in ['stofs_3d_atl']:
        forcing_name = 'HRRR'
    else:
        raise ValueError(f'Model {model} not available in stofs-event-dashboard.')
    return forcing_name


def get_forcing_geometry(model: str) -> str:
    """Returns the file geometry for data from a given model/forcing."""
    if model in ['stofs_2d_glo']:
        geom = 'grid'
    elif model in ['stofs_3d_atl']:
        geom = 'mesh'
    else:
        raise ValueError(f'Model {model} not available in stofs-event-dashboard.')
    return geom


def get_dir_var_name(var_name: str) -> str:
    """Gets version of variable name used in data directory structure."""
    if var_name in ['cwl', 'water_level', 'waterlevel']:
        dir_var = 'cwl'
    elif var_name in ['pressure', 'air_pressure']:
        dir_var = 'pressure'
    elif var_name in ['wind']:
        dir_var = 'wind'
    else:
        raise ValueError(f'var_name {var_name} not recognized.')
    return dir_var


def check_run_query(
    forecast_type: str,
    out_dir: pathlib.Path,
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime | None
):
    """
    Determines whether a model data query should be executed.
     
    The determination is based on the presence and coverage 
    of existing data files.

    For 'nowcast', checks if any data files exist in the output directory. 
    If none exist, a query should be run.
    If files exist, checks if the latest data end time is earlier than the 
    requested end time, indicating that more data should be appended.
    For 'forecast', checks only for the existence of files; if none 
    exist, a query should be run.

    Parameters
    ----------
    forecast_type
        Type of forecast ('nowcast' or 'forecast').
    out_dir 
        Directory to check for existing data files.
    start_datetime
        Start time for the data query.
    end_datetime 
        End time for the data query (required for 'nowcast', ignored for 'forecast').

    Returns
    -------
    run_query : bool
        True if a query should be run, False otherwise.
    append_data : bool
        True if new data should be appended to existing files, False if files should be created.
    query_start_datetime : datetime.datetime or None
        The start time for the query, which may differ from the
        requested start datetime if some data is already present.
    
    """
    run_query = False
    append_data = False
    query_start_datetime = None

    if forecast_type == 'nowcast':
        # Check if the obs files exist and need to be updated.
        existing_files = list(out_dir.glob('*.parquet'))
        if len(existing_files) == 0:
            # If no files exist, we want to run the query and save files. 
            run_query = True
            query_start_datetime = start_datetime
        else:
            logger.debug(f"{len(existing_files)} existing files in {out_dir}, starting with {str(existing_files[0])}")
            # If files exist, we only want to run a query if the end time
            # of the files is less than the end time of the event. Then we
            # want to append new data, instead of overwriting the file.
            # TODO: Check if there is a faster way to do this than reading all the files.
            earliest_end_datetime = min([
                pd.read_parquet(fn).reset_index()['time'].max() 
                for fn in existing_files
            ]).to_pydatetime()
            if earliest_end_datetime < end_datetime:
                run_query = True
                append_data = True
                query_start_datetime = max(
                    earliest_end_datetime + datetime.timedelta(minutes=1),
                    start_datetime
                )
                
    elif forecast_type == 'forecast':
        # For forecast data, we just check if the output directory exists and 
        # if there are files in it.
        existing_files = list(out_dir.glob('*.parquet'))
        if len(existing_files) == 0:
            # If no files exist, we want to run the query and save files. 
            run_query = True
            query_start_datetime = start_datetime

    else:
        raise ValueError(f'forecast_type {forecast_type} not recognized in check_run_query.')
    #
    return run_query, append_data, query_start_datetime


def save_model(
    waterlevel_stations: pd.DataFrame, 
    met_stations: pd.DataFrame, 
    stb: eventSpaceTimeBounds, 
    model_name: str,
    model_config: dict,
    plot_types_config:dict,
    output_config: dict
) -> None:
    """
    Orchestrates the saving of model data for both nowcast and forecast.

    For nowcast, triggers data queries and saving for the specified model 
    and time bounds. For forecast, determines initialization times and 
    triggers data queries and saving for each forecast initialization.

    Parameters
    ----------
    waterlevel_stations
        DataFrame containing water level station information.
    met_stations
        DataFrame containing meteorological station information.
    stb
        Object specifying the event's spatial and temporal bounds.
    model_name
        Name of the model to query and save data for.
    model_config
        Configuration dictionary for the model.
    plot_types_config
        Dictionary specifying which data types (e.g., water level, wind, pressure) to process.
    output_config
        Dictionary specifying output directory and formatting options.

    Returns
    -------
    None

    """
    data_dir = write_output.get_output_dir(output_config, stb)
    # Handle nowcast data saving.
    if model_config['nowcast']:
        logger.info(f'Saving {model_name} nowcast data.')
        try:
            query_and_save_model(
                waterlevel_stations, 
                met_stations,
                model_name, 
                model_config,  
                plot_types_config, 
                data_dir,
                'nowcast',
                stb.start_datetime, 
                stb.end_datetime
            )
        except Exception as e:
            logger.warning(traceback.format_exc())

    # Handle forecast data saving.
    if model_config['all_forecasts']:
        forecast_inits = get_forecast_init_times(
            model_name, 
            stb.start_datetime, 
            stb.end_datetime
        )
    else: 
        if model_config['forecast_init_list']:
            forecast_inits = [
                datetime.datetime.fromisoformat(dt) for dt in 
                model_config['forecast_init_list']
            ]
        else:
            forecast_inits = []
    if forecast_inits:
        logger.info(f'{model_name} forecast initializations:')
        logger.info(f'{forecast_inits}')
        for fidt in forecast_inits:
            logger.info(f"Saving {model_name} forecast data for {fidt.strftime('%Y%m%dT%H%M')}.")
            try:
                query_and_save_model(
                    waterlevel_stations, 
                    met_stations,
                    model_name, 
                    model_config,  
                    plot_types_config, 
                    data_dir,
                    'forecast',
                    fidt, 
                    None
                )
            except Exception as e:
                logger.warning(traceback.format_exc())

    return None


def query_and_save_model(
    waterlevel_stations: pd.DataFrame, 
    met_stations: pd.DataFrame, 
    model_name: str,
    model_config: dict,
    plot_types_config:dict,
    data_dir: str | pathlib.Path,
    forecast_type: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime | None
) -> None:
    """
    Handles querying and saving of model data.

    Request is based on specified stations, variables, and time bounds.
    The function determines output directories, checks for existing data, 
    and performs data queries for water level, pressure, and wind 
    variables as requested. Saves results to disk, appending to or 
    creating files as needed.

    Parameters
    ----------
    waterlevel_stations
        DataFrame containing water level station information.
    met_stations
        DataFrame containing meteorological station information.
    model_name
        Name of the model to query and save data for.
    model_config
        Configuration dictionary for the model.
    plot_types_config
        Dictionary specifying which data types (e.g., water level, wind, pressure) to process.
    data_dir
        Output directory for saving data files.
    forecast_type
        Type of forecast ('nowcast' or 'forecast').
    start_time
        Start time for the data query.
    end_time
        End time for the data query (required for 'nowcast', ignored for 'forecast').

    Returns
    -------
    None

    """
    # Parse options.
    if forecast_type == 'nowcast':
        forecast_type_dir = 'nowcast'
    else:
        forecast_type_dir = f'forecast_{start_time.strftime('%Y%m%dT%H%M')}'
    out_dirs = {}
    for var in ['cwl', 'pressure', 'wind']:
        out_dirs[var] = (
            pathlib.Path(data_dir) 
            / var / forecast_type_dir / model_name
        )
    forcing_model = get_model_forcing(model_name)
    forcing_geom = get_forcing_geometry(model_name)
    
    # Download and save water level data.
    if plot_types_config['water_level']:
        (
            run_query, append_data, query_start_datetime
        ) = check_run_query(
            forecast_type, out_dirs['cwl'], start_time, end_time
        )
        if run_query:
            model_data = seanode.api.get_surge_model_at_stations(
                model_name.upper(),
                model_config['water_level_var_list'],
                waterlevel_stations.nos_id,
                query_start_datetime,
                end_time,
                forecast_type,
                'points',
                None,
                'AWS'
            )
            write_output.df_sealens_parquet(
                model_data,
                out_dirs['cwl'],
                column_names=model_config['water_level_var_list'],
                append=append_data
            )
            # Check for any missing stations and run fields file query if needed.
            if model_data.empty:
                missing_stations = set(waterlevel_stations.nos_id)
            else:
                missing_stations = (
                    set(waterlevel_stations.nos_id) -
                    set(model_data.index.unique(level='station'))
                )
            if missing_stations and model_config['fields_files']:
                logger.info(f'Missing stations for {model_name} CWL {forecast_type}: {missing_stations}')
                logger.info('Running fields file query for missing stations.')
                # Run fields file query for missing stations.
                model_fields_data = seanode.api.get_surge_model_at_stations(
                    model_name.upper(),
                    model_config['water_level_var_list'],
                    waterlevel_stations.set_index('station')\
                        .loc[list(missing_stations)].reset_index(),
                    query_start_datetime,
                    end_time,
                    forecast_type,
                    'mesh',
                    None,
                    'AWS'
                )
                write_output.df_sealens_parquet(
                    model_fields_data,
                    out_dirs['cwl'],
                    column_names=model_config['water_level_var_list'],
                    append=append_data
                )
                # TODO: Think about how to handle different datums 
                # between points and fields files.
        else:
            logger.info(f'Not fetching any {model_name.upper()} CWL {forecast_type} data, probably because it already exists for given date range ({start_time} -- {end_time})')
            
    # Download pressure and/or wind data.
    if plot_types_config['pressure'] & plot_types_config['wind']:
        (
            run_query_wind, append_data_wind, query_start_datetime_wind
        ) = check_run_query(
            forecast_type, out_dirs['wind'], start_time, end_time
        )
        (
            run_query_pressure, append_data_pressure, query_start_datetime_pressure
        ) = check_run_query(
            forecast_type, out_dirs['pressure'], start_time, end_time
        )
        run_query = run_query_wind | run_query_pressure
        append_data = append_data_wind | append_data_pressure
        if run_query:
            query_start_datetime = min(
                d for d in [
                    query_start_datetime_wind,
                    query_start_datetime_pressure
                ] if d is not None
            )
            model_data = seanode.api.get_surge_model_at_stations(
                forcing_model,
                model_config['pressure_var_list'] + 
                model_config['wind_var_list'],
                met_stations,
                query_start_datetime,
                end_time,
                forecast_type,
                forcing_geom,
                None,
                'AWS'
            )
        else:
            logger.info(f'Not fetching any {model_name.upper()} wind+pressure {forecast_type} data, probably because it already exists for given date range ({start_time} -- {end_time})')
    elif plot_types_config['pressure']:
        (
            run_query, append_data, query_start_datetime
        ) = check_run_query(
            forecast_type, out_dirs['pressure'], start_time, end_time
        )
        if run_query:
            model_data = seanode.api.get_surge_model_at_stations(
                forcing_model,
                model_config['pressure_var_list'],
                met_stations,
                query_start_datetime,
                end_time,
                forecast_type,
                forcing_geom,
                None,
                'AWS'
            )
        else:
            logger.info(f'Not fetching any {model_name.upper()} pressure {forecast_type} data, probably because it already exists for given date range ({start_time} -- {end_time})')
    elif plot_types_config['wind']:
        (
            run_query, append_data, query_start_datetime
        ) = check_run_query(
            forecast_type, out_dirs['wind'], start_time, end_time
        )
        if run_query:
            model_data = seanode.api.get_surge_model_at_stations(
                forcing_model,
                model_config['wind_var_list'],
                met_stations,
                query_start_datetime,
                end_time,
                forecast_type,
                forcing_geom,
                None,
                'AWS'
            )
        else:
            logger.info(f'Not fetching any {model_name.upper()} wind {forecast_type} data, probably because it already exists for given date range ({start_time} -- {end_time})')
    else:
        pass

    # Save wind data.
    if plot_types_config['wind']:
        if run_query:
            try:
                model_data['wind_speed'] = np.sqrt(model_data['u10']**2 + model_data['v10']**2)
                save_cols = model_config['wind_var_list'] + ['wind_speed']
            except:
                logger.warning('Unable to calculate wind speed from u10 and v10 variables.')
                save_cols = model_config['wind_var_list']
            write_output.df_sealens_parquet(
                model_data,
                out_dirs['wind'],
                column_names=save_cols,
                append=append_data
            )
    # Save pressure data.
    if plot_types_config['pressure']:
        if run_query:
            try:
                model_data['ps'] = model_data['ps'] / 100.0
            except:
                logger.warning('Unable to convert pressure to hPa.')
            write_output.df_sealens_parquet(
                model_data,
                out_dirs['pressure'],
                column_names=model_config['pressure_var_list'],
                append=append_data
            )
    return None
