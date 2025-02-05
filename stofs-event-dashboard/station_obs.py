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
import searvey
from searvey._coops_api import fetch_coops_station
from typing import Iterable, Union, Any


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
