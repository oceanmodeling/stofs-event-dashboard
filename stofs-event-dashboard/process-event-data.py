"""Process data for dashboard.

Run from the command line:
    $ python process-event-data.py <config_path>

E.g., 
    $ python process-event-data.py ../test_2025.conf

"""


import argparse
import json
import sys
import logging
import shapely
import searvey
import pandas as pd
import geopandas as gpd
import numpy as np
import pathlib
from seanode.api import get_surge_model_at_stations
import space_time_bounds
from station_obs import save_obs
from models import (
    get_forecast_init_times, 
    save_model
)
import static_map
import map_data
from write_output import df_sealens_parquet


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_event(config: dict) -> None:
    """Process data for an event defined in config dictionary.
    
    """
    # Get space and time bounds.
    stb = space_time_bounds.eventSpaceTimeBounds(config['event'])

    # Get station lists for event region, and format as needed. 
    station_list = searvey.get_coops_stations(metadata_source='main')
    station_list = searvey.get_coops_stations(
        region=stb.get_region(config['stations']['bounds'])\
                  .buffer(config['stations']['buffer']),  
        metadata_source='main'
    )
    station_list= station_list.reset_index().rename(
        columns={'lat':'latitude', 'lon':'longitude'}
    )
    station_list['station'] = station_list['nos_id']
    waterlevel_stations = station_list[
        (station_list['status'] == 'active') & 
        (station_list['station_type'] == 'waterlevels')
    ]
    met_stations = station_list[
        (station_list['status'] == 'active') & 
        (station_list['station_type'] == 'met')
    ]
    
    # Save map data in geopackage.
    map_data.save_geopackage(stb, 
                             {'waterlevel':waterlevel_stations, 
                              'met':met_stations},
                             config['output'])
    
    # ---------- Observations. ------------------------------
    if config['plot_types']['water_level']:
        save_obs(waterlevel_stations, stb, 'water_level', config['output'])
    if config['plot_types']['pressure']:
        save_obs(met_stations, stb, 'pressure', config['output'])
    if config['plot_types']['wind']:
        save_obs(met_stations, stb, 'wind', config['output'])

    # ---------- Model data. ------------------------------
    # Loop over models.
    for model in config['models'].keys():
        
        if config['forecast_type']['nowcast']:
            logger.info(f'Saving {model} nowcast data.')
            save_model(waterlevel_stations, met_stations,
                       stb, {model:config['models'][model]},
                       config['plot_types'], config['output'],
                       stb.start_datetime, stb.end_datetime)
            
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
        if forecast_inits:
            for fidt in forecast_inits:
                logger.info(f'Saving {model} forecast data for {fidt.strftime('%Y%m%dT%H%M')}.')
                save_model(waterlevel_stations, met_stations,
                           stb, {model:config['models'][model]},
                           config['plot_types'], config['output'],
                           fidt, None)
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
