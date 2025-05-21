"""Process data for example dashboard using Hurricane Milton (2024).

Contains only the function "main".
Run from the command line without any additional arguments:
$ python milton_example.py

"""


import argparse
import json
import sys
import logging
#
import space_time_bounds
import shapely
import searvey
from station_obs import fetch_coops_multistation_df
from models import (
    get_forecast_init_times, 
    get_forcing_model, 
    get_forcing_geometry
)
import static_map
import pandas as pd
import numpy as np
import pathlib
from write_parquet import df_to_sealens
from seanode.api import get_surge_model_at_stations


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_event(config: dict) -> None:
    """Process data for an event defined in config dictionary.
    
    """
    # Get space and time bounds.
    stb = space_time_bounds.eventSpaceTimeBounds(config['event'])
    
    # Define output directory. 
    if config['output_dir']:
        data_dir = pathlib.Path(config['output_dir'])
    else:
        data_dir = pathlib.Path(f'../data/{'_'.join([stb.name, str(stb.year)])}')

    # Get station list for event region. 
    station_list = searvey.get_coops_stations(metadata_source='main')
    station_list = searvey.get_coops_stations(
        region=stb.get_region(config['stations']['bounds'])\
                  .buffer(config['stations']['buffer']),  
        metadata_source='main'
    )
    
    # ---------- Observations. ------------------------------
    if config['plot_types']['water_level']:
        waterlevel_stations = station_list[
            (station_list['status'] == 'active') & 
            (station_list['station_type'] == 'waterlevels')
        ]
        logger.info(f'Fetching {len(waterlevel_stations.index)} stations for water level data.')
        waterlevel_obs = fetch_coops_multistation_df(
            waterlevel_stations.index,
            stb.start_datetime,
            stb.end_datetime,
            'water_level',
            datum=config['output_datum'] 
        )
        df_to_sealens(
            waterlevel_obs,
            data_dir / 'cwl/obs',
            column_names=['value']
        )
    if config['plot_types']['pressure'] | config['plot_types']['wind']:
        met_stations = station_list[
            (station_list['status'] == 'active') & 
            (station_list['station_type'] == 'met')
        ]
    if config['plot_types']['pressure']:
        logger.info(f'Fetching {len(met_stations.index)} stations for pressure data.')
        pressure_obs = fetch_coops_multistation_df(
            met_stations.index,
            stb.start_datetime,
            stb.end_datetime,
            'air_pressure'
        ) 
        df_to_sealens(
            pressure_obs,
            data_dir / 'pressure/obs',
            column_names=['value']
        )
    if config['plot_types']['wind']:
        logger.info(f'Fetching {len(met_stations.index)} stations for wind data.')
        wind_obs = fetch_coops_multistation_df(
            met_stations.index,
            stb.start_datetime,
            stb.end_datetime,
            'wind' 
        ) 
        # Calculate u and v components.
        # u = speed * cos(270 - bearing_from)
        # v = speed * sin(270 - bearing_from)
        wind_obs['u_wind'] = \
            wind_obs['speed'] * np.cos(np.radians(270.0 - wind_obs['degree']))
        wind_obs['v_wind'] = \
            wind_obs['speed'] * np.sin(np.radians(270.0 - wind_obs['degree']))
        wind_obs = wind_obs.rename(columns={'speed':'wind_speed'})
        df_to_sealens(
            wind_obs,
            data_dir / 'wind/obs',
            column_names=['wind_speed', 'degree', 'direction', 'gust', 
                          'u_wind', 'v_wind']
        )

    # ---------- Model data. ------------------------------
    # Datum name.
    if config['output_datum'] == 'NAVD':
        output_datum_model = 'NAVD88'
    #
    for model in config['models'].keys():
        # Get the forecast times for this model.
        # This is reused across variables.
        if config['forecast_type']['all_forecasts']:
            forecast_inits = get_forecast_init_times(model, 
                                                     stb.start_datetime, 
                                                     stb.end_datetime)
        else:
            forecast_inits = config['models'][model]['forecast_init_list']
        logger.info(f'{model} forecast initializations: ')
        logger.info(f'{forecast_inits}')
        # 
        if config['plot_types']['water_level']:
            if config['forecast_type']['nowcast']:
                model_data = get_surge_model_at_stations(
                    model.upper(),
                    config['models'][model]['water_level_var_list'],
                    waterlevel_stations.index,
                    stb.start_datetime,
                    stb.end_datetime,
                    'nowcast',
                    'points',
                    output_datum_model,
                    'AWS'
                )
                df_to_sealens(
                    model_data,
                    data_dir / f'cwl/nowcast/{model}',
                    column_names=config['models'][model]['water_level_var_list']
                )
            if forecast_inits:
                for fidt in forecast_inits:
                    model_data = get_surge_model_at_stations(
                        model.upper(),
                        config['models'][model]['water_level_var_list'],
                        waterlevel_stations.index,
                        fidt,
                        None,
                        'forecast',
                        'points',
                        output_datum_model,
                        'AWS'
                    )
                    df_to_sealens(
                        model_data,
                        data_dir / f'cwl/forecast_{fidt.strftime('%Y%m%dT%H%M')}/{model}',
                        column_names=config['models'][model]['water_level_var_list']
                    )
        if config['forecast_type']['nowcast']:
            if config['plot_types']['pressure'] & config['plot_types']['wind']:
                # Combine wind and pressure data queries if both are needed.
                model_data = get_surge_model_at_stations(
                    get_forcing_model(model),
                    config['models'][model]['pressure_var_list'] + 
                    config['models'][model]['wind_var_list'],
                    met_stations.index,
                    stb.start_datetime,
                    stb.end_datetime,
                    'nowcast',
                    get_forcing_geometry(model),
                    None,
                    'AWS'
                )
            elif config['plot_types']['pressure']:
                model_data = get_surge_model_at_stations(
                    get_forcing_model(model),
                    config['models'][model]['pressure_var_list'],
                    met_stations.index,
                    stb.start_datetime,
                    stb.end_datetime,
                    'nowcast',
                    get_forcing_geometry(model),
                    None,
                    'AWS'
                )
            elif config['plot_types']['wind']:
                model_data = get_surge_model_at_stations(
                    get_forcing_model(model),
                    config['models'][model]['wind_var_list'],
                    met_stations.index,
                    stb.start_datetime,
                    stb.end_datetime,
                    'nowcast',
                    get_forcing_geometry(model),
                    None,
                    'AWS'
                )
            else:
                pass
            if config['plot_types']['wind']:
                try:
                    model_data['wind_speed'] = np.sqrt(model_data['u10']**2 + model_data['v10']**2)
                    save_cols = config['models'][model]['wind_var_list'] + ['wind_speed']
                except:
                    logger.warning('Unable to calculate wind speed from u10 and v10 variables.')
                    save_cols = config['models'][model]['wind_var_list']
                df_to_sealens(
                    model_data,
                    data_dir / f'wind/nowcast/{model}',
                    column_names=save_cols
                )
            if config['plot_types']['pressure']:
                df_to_sealens(
                    model_data,
                    data_dir / f'pressure/nowcast/{model}',
                    column_names=config['models'][model]['pressure_var_list']
                )
        #
        if forecast_inits:
            for fidt in forecast_inits:
                if config['plot_types']['pressure'] & config['plot_types']['wind']:
                    # Combine wind and pressure data queries if both are needed.
                    model_data = get_surge_model_at_stations(
                        get_forcing_model(model),
                        config['models'][model]['pressure_var_list'] + 
                        config['models'][model]['wind_var_list'],
                        met_stations.index,
                        fidt,
                        None,
                        'forecast',
                        get_forcing_geometry(model),
                        None,
                        'AWS'
                    )
                elif config['plot_types']['pressure']:
                    model_data = get_surge_model_at_stations(
                        get_forcing_model(model),
                        config['models'][model]['pressure_var_list'],
                        met_stations.index,
                        fidt,
                        None,
                        'forecast',
                        get_forcing_geometry(model),
                        None,
                        'AWS'
                    )
                elif config['plot_types']['wind']:
                    model_data = get_surge_model_at_stations(
                        get_forcing_model(model),
                        config['models'][model]['wind_var_list'],
                        met_stations.index,
                        fidt,
                        None,
                        'forecast',
                        get_forcing_geometry(model),
                        None,
                        'AWS'
                    )
                else:
                    pass
                if config['plot_types']['pressure']:
                    df_to_sealens(
                        model_data,
                        data_dir / f'pressure/forecast_{fidt.strftime('%Y%m%dT%H%M')}/{model}',
                        column_names=config['models'][model]['pressure_var_list']
                    )
                if config['plot_types']['wind']:
                    try:
                        model_data['wind_speed'] = np.sqrt(model_data['u10']**2 + model_data['v10']**2)
                        save_cols = config['models'][model]['wind_var_list'] + ['wind_speed']
                    except:
                        logger.warning('Unable to calculate wind speed from u10 and v10 variables.')
                        save_cols = config['models'][model]['wind_var_list']
                    df_to_sealens(
                        model_data,
                        data_dir / f'wind/forecast_{fidt.strftime('%Y%m%dT%H%M')}/{model}',
                        column_names=save_cols
                    )
    #
    return stb
        

if __name__ == '__main__':
    # Set up command line argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath", type=str,
                        help="path of event definition configuration file")
    args = parser.parse_args()
    
    # Check if a config file exists at specified path.
    config_path = pathlib.Path(args.config_filepath).resolve()
    if config_path.is_file():
        logger.info(f'Using event config file {config_path}')
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
                stb = process_event(config)
        except json.decoder.JSONDecodeError as e:
            logger.exception(f'{config_path} does not appear to be a JSON file:')
    else:
        raise FileNotFoundError(f'No file found at {config_path}')
