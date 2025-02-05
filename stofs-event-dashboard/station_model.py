"""Handle model data at station locations.

This module contains utilities for processing and transforming model
data that has already been extracted at station locations.

Functions
---------
extract_model_station_df(ds, station_id_list, data_var)
    Extracts model data for variable data_var at station locations
    defined in station_id_list, from a dataset that already contains
    model data at station locations.

"""


import numpy as np
import pandas as pd
import xarray as xr
from typing import Iterable


def extract_model_station_df(
        ds: xr.Dataset, 
        station_id_list: Iterable,
        data_var: str = 'zeta'
    ) -> pd.DataFrame:
    """Extract dataframe of model data at listed station locations.

    Starting from an xarray dataset that contains model data at station 
    locations, this function extracts data from variable data_var at
    station locations listed in station_id_list, and returns a pandas
    data frame in the same format as used for observational data. 

    Parameters
    ----------
    ds 
        Contains variable data_var, with dimensions [time, station], 
        and variable "station_name", also with dimensions [time, station],
        that is searched for matches with stations listed in station_id_list.
    station_id_list
        An iterable (e.g., list, pandas Index, or pandas Series) containing
        station IDs that are matched with "station_name" in ds.
    data_var 
        The name of the variable for which model data is extracted.

    Returns
    -------
    pandas DataFrame
        Table with (station, time) multi-index containing data for all
        station locations for variable data_var.

    """
    result = pd.DataFrame()

    for nos_id in station_id_list:

        # Find the model station names that match this NOS ID.
        obs_name_in_model = [nos_id in nm.decode('utf-8') 
                             for nm in ds.station_name[0,:].data]

        # Check that only one model station matchis this ID.
        if np.sum(obs_name_in_model) > 1:
            print(f"Warning: more than one model station matches NOS ID {nos_id}")
        elif np.sum(obs_name_in_model) == 0:
            print(f"Warning: no model station matches for NOS ID {nos_id}")
        else:
            # Check if the name for this station is constant in time.
            if len(pd.unique(ds.station_name[:,obs_name_in_model].squeeze().data)) > 1:
                print(f"Warning: model station name is not constant for NOS ID {nos_id}:")
                print(pd.unique(ds.station_name[:,obs_name_in_model].squeeze().data))
                
            # If available, concatenate the data with other stations.
            station_df = ds[data_var][:,obs_name_in_model]\
                .assign_coords(station=np.array([nos_id]))\
                .to_dataframe(dim_order=('station', 'time'))
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
