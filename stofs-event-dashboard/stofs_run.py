"""Define class and methods to handle STOFS raw model output streams.

Classes
-------
StofsRun

Functions
---------
StofsRun.get_station_nowcast(daterange, output_type=None)
StofsRun.get_station_forecast(first_forecast_date, output_type=None)

Instances
---------
stofs_2d: StofsRun
stofs_3d_atl: StofsRun

"""


from _STOFS import get_station_nowcast_data
from _STOFS import get_station_data
from typing import Union, Dict, List
import xarray
import datetime


class StofsRun():
    """Model run details and methods to access data.
    
    Attributes
    ----------
    bucket
    subdirectory
    model
    output_types
    forecast_cycles
    nowcast_steps

    Methods
    -------
    get_station_nowcast
    get_station_forecast

    Notes
    -----
    Instances defined in this module include:
    stofs_2d
    stofs_3d_atl

    """

    def __init__(
            self, 
            bucket: str, 
            subdirectory: str, 
            model: str, 
            output_types: Dict[str, str], 
            forecast_cycles: List[str], 
            nowcast_steps: int
        ) -> None:
        """Initialize model run locations and other details.

        Parameters
        ----------
        bucket
            AWS bucket containing data (e.g., 'noaa-gestofs-pds')
        subdirectory
            Possible subdirectory for this run within bucket.
            Use '' if not applicable.
        model
            Code/abbreviation for the model (e.g., 'stofs_2d_glo')
        output_types
            A dictionary of possible output types of the form
            {name: filename_preposition}.
            Must contain at least a "default" key, like
            {"default" : default_file_preposition}
        forecast_cycles
            A list of forecast cycle initialization times used to
            differentiate different runs in the same bucket.
        nowcast_steps
            The number of hours at the beginning of each model run
            that correspond to nowcast data.

        Returns
        -------
        None.

        """
        self.bucket = bucket
        self.subdirectory = subdirectory
        self.model = model
        self.output_types = output_types
        self.forecast_cycles = forecast_cycles
        self.nowcast_steps = nowcast_steps
        
    def get_station_nowcast(
            self, 
            daterange: List[str], 
            output_type: Union[str, None] = None
        ) -> xarray.Dataset:
        """Return model nowcast data covering a given date range.

        This method stitches together the nowcast periods from consecutive
        forecasts, so can be used for arbitrarily long date ranges. 
        (However, that might be prohibitively slow.)

        Parameters
        ----------
        daterange
            List of two strings, each of the form YYYYMMDD ('%Y%m%d'),
            that defines the period for which data is retrieved.
        output_type
            The type of data to retrieve. Default value is None, which
            results in fetching the "default" output type defined in 
            the StofsRun class.

        Returns
        -------
        xarray.Dataset
            Dataset containing model nowcast data at station locations 
            for the given output_type and date range. 

        """
        # Select output type (variable, bias correction, etc.)
        if not output_type:
            filename = self.output_types["default"]
        else:
            filename = self.output_types[output_type]
            
        ds = get_station_nowcast_data(filename, 
                                      self.model, 
                                      self.subdirectory, 
                                      self.bucket, 
                                      daterange, 
                                      self.nowcast_steps, 
                                      self.forecast_cycles)
        
        # TODO (Jack): add method to extract NOS ID and assign it as a coord?
        # TODO (Jack): maybe convert to pandas data frame here?
        
        return ds
        
    def get_station_forecast(
            self, 
            first_forecast_date: datetime.datetime, 
            output_type: Union[str, None] = None):
        """Return model forecast data covering a given date range.

        This method retrieves a single forecast run starting on 
        a given date, for the given output_type.

        Parameters
        ----------
        first_forecast_date
            The first date/time needed in the forecast. The
            returned data may start before this if it doesn't 
            correspond with a model initialization time.
        output_type
            The type of data to retrieve. Default value is None, which 
            results in fetching the "default" output type defined in 
            the StofsRun class.
        
        Returns
        -------
        xarray.Dataset
            Dataset containing model forecast data at station locations 
            for the given output_type and date range.

        """
        # Select output type (variable, bias correction, etc.)
        if not output_type:
            filename = self.output_types["default"]
        else:
            filename = self.output_types[output_type]
            
        # Method for choosing forecast cycle.
        first_hour = first_forecast_date.hour
        cycle_int = max([int(ci) if int(ci) <= first_hour else 0 
                         for ci in self.forecast_cycles])
        # TODO (Jack): This needs some thought for more general cases.
        # E.g., if forecast_cycles are [3,9,15,21], and we have
        # a start time of 0213 UTC, then we would want to select the 
        # 21UTC forecast from the previous day. The current version
        # would fail in such a case.
        
        ds = get_station_data(filename, 
                              self.model, 
                              self.subdirectory, 
                              self.bucket, 
                              first_forecast_date.strftime('%Y%m%d'), 
                              "{:02n}".format(cycle_int))
        
        # TODO (Jack): add method to extract NOS ID and assign it as a coord?
        # TODO (Jack): maybe convert to pandas data frame here?
        
        return ds

# Define some commonly-used instances.
stofs_2d = StofsRun(
    'noaa-gestofs-pds', 
    '', 
    'stofs_2d_glo', 
    {"default":"points.cwl", "noanomaly":"points.cwl.noanomaly", 
     "velocity":"points.cwl.vel", "autoval":"points.autoval.cwl"}, 
    ['00','06','12','18'], 
    60
)

stofs_3d_atl = StofsRun(
    'noaa-nos-stofs3d-pds', 
    'STOFS-3D-Atl', 
    'stofs_3d_atl', 
    {"default":"points.cwl",  
     "tsv":"points.cwl.temp.salt.vel", "autoval":"points.autoval.cwl"}, 
    ['12'], 
    240
)
