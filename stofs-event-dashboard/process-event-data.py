"""Process data for dashboard.

Run from the command line:
    $ python process-event-data.py <config_path>

E.g., 
    $ python process-event-data.py ../test_2025.conf

"""


import argparse
import json
import sys
import os
import logging
import traceback
import shapely
import searvey
import pandas as pd
import geopandas as gpd
import numpy as np
import pathlib
import datetime
import time
import asyncio
from seanode.api import get_surge_model_at_stations
import space_time_bounds
from station_obs import fetch_metadata, save_obs
from models import (
    get_forecast_init_times, 
    save_model
)
import map_data
import write_output


log_dir = pathlib.Path(__file__).parents[1] / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = f"process_event_data.log.{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}"
fh = logging.FileHandler(log_dir / log_file)
ch = logging.StreamHandler()
formatter = '%(asctime)s %(name)-15s %(levelname)-8s %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=formatter,
    handlers=[
        ch,
        fh
    ]
)
logger = logging.getLogger(__name__)


def process_event(config: dict) -> None:
    """Process data for an event defined in config dictionary.
    
    """
    # Get space and time bounds.
    stb = space_time_bounds.eventSpaceTimeBounds(config['event'])

    # Get station lists for event region, and format as needed. 
    """
    station_list = searvey.get_coops_stations(metadata_source='main')
    station_list = searvey.get_coops_stations(
        region=stb.get_region(config['stations']['bounds'])\
                  .buffer(config['stations']['buffer']),  
        metadata_source='main'
    )
    ioc_stations = searvey.get_ioc_stations(
        region=stb.get_region(config['stations']['bounds'])\
                  .buffer(config['stations']['buffer'])
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
    """
    waterlevel_stations, met_stations = fetch_metadata(config['stations'], stb)
    
    # Save map data in geopackage.
    map_data.save_geopackage(stb, 
                             {'waterlevel':waterlevel_stations, 
                              'met':met_stations},
                             config['output'])
    
    # ---------- Observations. ------------------------------
    # TODO: Move the obs network checks to the station_obs module.
    if "coops" in config['stations']['source_list']:
        if config['plot_types']['water_level']:
            save_obs(waterlevel_stations, stb, 'water_level', config['output'])
        if config['plot_types']['pressure']:
            save_obs(met_stations, stb, 'pressure', config['output'])
        if config['plot_types']['wind']:
            save_obs(met_stations, stb, 'wind', config['output'])

    # ---------- Model data. ------------------------------
    n_attempts = 2
    # Loop over models.
    for model in config['models'].keys():
        logger.info(f'Saving {model} data.')
        for attempt in range(n_attempts):
            logger.info(f'Attempt #{attempt + 1} of {n_attempts}')
            try:
                save_model(waterlevel_stations, met_stations,
                           stb, model, config['models'][model],
                           config['plot_types'], config['output'])
            except Exception as e:
                logger.warning(traceback.format_exc())
                
    # ---------- Summarize data. ------------------------------
    logger.info('---------- Event data summary ----------')
    output_dir = write_output.get_output_dir(
        config['output'], stb, allow_mkdir=False
    )
    for root, dirs, files in os.walk(output_dir):
        parquet_files = [p for p in files if p[-7:] == 'parquet']
        if len(parquet_files) > 0:
            logger.info(f"{root}: {len(parquet_files)} parquet files")
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
