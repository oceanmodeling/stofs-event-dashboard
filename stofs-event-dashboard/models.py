"""Functions to handle model data requests.
"""


import datetime
import pandas as pd
import numpy as np
from typing import List
import logging
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
    if var_name in ['cwl', 'water_level']:
        dir_var = 'cwl'
    elif var_name in ['pressure', 'air_pressure']:
        dir_var = 'pressure'
    elif var_name in ['wind']:
        dir_var = 'wind'
    else:
        raise ValueError(f'var_name {var_name} not recognized.')
    return dir_var


def check_run_query(
    forecast_type,
    out_dir,
    start_datetime,
    end_datetime
):
    """
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
    model_config_dict: dict,
    plot_types_config:dict,
    output_config: dict,
    start_time: datetime.datetime,
    end_time: datetime.datetime | None
) -> None:
    """
    """
    # Parse options.
    if len(model_config_dict.keys()) > 1:
        raise ValueError('Models should be passed to save_model(...) one at a time.')
    else:
        model_name = list(model_config_dict.keys())[0]
        model_config = model_config_dict[model_name]
    if end_time is not None:
        forecast_type = 'nowcast'
        forecast_type_dir = 'nowcast'
    else:
        forecast_type = 'forecast'
        forecast_type_dir = f'forecast_{start_time.strftime('%Y%m%dT%H%M')}'
    forcing_model = get_model_forcing(model_name)
    forcing_geom = get_forcing_geometry(model_name)
    if output_config['output_datum'] == 'NAVD':
        output_datum_model = 'NAVD88'
    else:
        output_datum_model = output_config['output_datum']
    data_dir = write_output.get_output_dir(output_config, stb)

    # Download and save water level data.
    if plot_types_config['water_level']:
        out_dir = data_dir / f'cwl/{forecast_type_dir}/{model_name}'
        run_query, append_data, query_start_datetime = check_run_query(
            forecast_type, out_dir, start_time, end_time
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
                output_datum_model,
                'AWS'
            )
            write_output.df_sealens_parquet(
                model_data,
                out_dir,
                column_names=model_config['water_level_var_list'],
                append=append_data
            )
        else:
            logger.info(f'Not fetching any {model_name.upper()} CWL {forecast_type} data, probably because it already exists for given date range ({start_time} -- {end_time})')
            
    # Download pressure and/or wind data.
    if plot_types_config['pressure'] & plot_types_config['wind']:
        # For simplicity we just test whether the wind data exists.
        out_dir = data_dir / f'wind/{forecast_type_dir}/{model_name}'
        run_query, append_data, query_start_datetime = check_run_query(
            forecast_type, out_dir, start_time, end_time
        )
        if run_query:
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
        out_dir = data_dir / f'pressure/{forecast_type_dir}/{model_name}'
        run_query, append_data, query_start_datetime = check_run_query(
            forecast_type, out_dir, start_time, end_time
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
        out_dir = data_dir / f'wind/{forecast_type_dir}/{model_name}'
        run_query, append_data, query_start_datetime = check_run_query(
            forecast_type, out_dir, start_time, end_time
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
                import pdb; pdb.set_trace()
                save_cols = model_config['wind_var_list']
            write_output.df_sealens_parquet(
                model_data,
                data_dir / f'wind/{forecast_type_dir}/{model_name}',
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
                data_dir / f'pressure/{forecast_type_dir}/{model_name}',
                column_names=model_config['pressure_var_list'],
                append=append_data
            )
        
