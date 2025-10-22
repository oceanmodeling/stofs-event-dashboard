"""Functions to write data to parquet files.

Functions
---------
df_to_sealens(df, dir, column_name)

"""


import pandas as pd
import pathlib
from typing import Union, List
import space_time_bounds


def df_sealens_parquet(df: pd.DataFrame, 
                       dir: Union[str, pathlib.Path], 
                       column_names: List[str],
                       append: bool = False) -> None:
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
    append
        If True, read any existing files, add the new data to 
        them, and save the file back out. 

    Returns
    -------
    None

    """
    # Get rid of timezone.
    df_no_tz = df.copy()
    ind_orig = df.index.names
    df_no_tz = df_no_tz.reset_index()
    df_no_tz.time = df_no_tz.time.dt.tz_localize(None)
    df_no_tz = df_no_tz.set_index(ind_orig)

    # Work out which columns to save.
    cols_to_save = [col for col in column_names 
                    if col in df_no_tz.columns]

    # Make sure target directory exists.
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    # Loop over stations and save files.
    for st in df_no_tz.index.unique(level='station'): 
        filepath = pathlib.Path(dir) / ('nos_' + st + '.parquet')
        if append & filepath.exists():
            try:
                existing_data = pd.read_parquet(filepath)
                new_data = df_no_tz.loc[st][cols_to_save]
                pd.concat([existing_data, new_data]).to_parquet(
                    path = filepath,
                    index = True
                )
            except:
                raise OSError(f'Error saving new data and existing data from file {filepath}.')
        else:
            try:
                df_no_tz.loc[st][cols_to_save].to_parquet(
                    path = filepath,
                    index = True
                )
            except:
                raise OSError(f'Error saving new data to file {filepath}.')

    return


def get_output_dir(
    output_config: dict,
    stb: space_time_bounds.eventSpaceTimeBounds,
    allow_mkdir: bool = True
) -> pathlib.Path:
    """Get output directory from "output" section of config or event bounds."""
    if output_config['output_dir']:
        data_dir = pathlib.Path(output_config['output_dir'])
    else:
        data_dir = pathlib.Path(__file__).parents[1] / 'data' /\
            '_'.join([stb.name, str(stb.year)])
    if allow_mkdir:
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


