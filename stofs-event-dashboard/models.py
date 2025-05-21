"""Functions to handle model data requests.
"""


import datetime
from typing import List
import logging
import seanode


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


def get_forcing_model(model: str) -> str:
    """Returns the atmosphere model used to force a given surge model."""
    if model in ['stofs_2d_glo']:
        forcing_model = 'GFS'
    elif model in ['stofs_3d_atl']:
        forcing_model = 'HRRR'
    else:
        raise ValueError(f'Model {model} not available in stofs-event-dashboard.')
    return forcing_model


def get_forcing_geometry(model: str) -> str:
    """Returns the file geometry for data from a given model's forcing."""
    if model in ['stofs_2d_glo']:
        geom = 'grid'
    elif model in ['stofs_3d_atl']:
        geom = 'mesh'
    else:
        raise ValueError(f'Model {model} not available in stofs-event-dashboard.')
    return geom
        