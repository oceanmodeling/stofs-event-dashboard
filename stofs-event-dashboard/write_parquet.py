"""Functions to write data to parquet files.

Functions
---------
df_to_sealens(df, dir, column_name)

"""


import pandas as pd
import pathlib
from typing import Union, List


def df_to_sealens(df: pd.DataFrame, 
                  dir: Union[str, pathlib.Path], 
                  column_names: List[str]) -> None:
    """
    Saves a multi-station dataframe as one parquet file per station.

    The file layout, and the convention to have one per station, is
    for compatibility with the sealens visualization package. 

    Parameters
    ----------
    df
        pandas dataframe with a (station, time) multi-index.
    dir
        The location in which to save all parquet files. Will be
        created if it doesn't already exist.
    column_name
        The name of the only column saved in the files.

    Returns
    -------
    None

    """
    # Get rid of timezone.
    df_no_tz = df.copy()
    df_no_tz.index = df_no_tz.index.set_levels(
        df_no_tz.index.levels[1].tz_localize(None), level=1
    )

    # Make sure target directory exists.
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    # Loop over stations and save files.
    for st in df_no_tz.index.unique(level='station'): 
        filename = 'nos_' + st + '.parquet'
        df_no_tz.loc[st][column_names].to_parquet(
            path = pathlib.Path(dir) / filename,
            index = True
        )

    return


