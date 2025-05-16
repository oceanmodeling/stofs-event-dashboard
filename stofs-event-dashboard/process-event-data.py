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
import static_map
import pandas as pd
import pathlib
from write_parquet import df_to_sealens
from seanode.api import get_surge_model_at_stations


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def process_event(config: dict) -> None:
    """Process data for an event defined in config dictionary.
    
    """
    # Get space and time bounds.
    stb = space_time_bounds.eventSpaceTimeBounds(config['event'])
    """
    if config['event']['nhc']:
        try:
            st_nm = config['event']['name']
            if config['event']['nhc_name']:
                st_nm = config['event']['nhc_name']
            event_stb = space_time_bounds.get_nhc_windswath(st_nm, 
                                                            config['event']['year'])
        except:
            raise ValueError(f'No NHC named storm for name {st_nm} in year {config['event']['year'].}')
    else:
        space_time_bounds.get_custom_bounds(
            config['event']['start_date'],
            config['event']['end_date'],
            *config['event']['bounding_box']
        )
    """
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
