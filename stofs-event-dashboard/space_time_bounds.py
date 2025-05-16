"""Define class and functions used when specifying an event. 

Classes
-------
spaceTimeBounds: pydantic BaseModel to contain event time and space bounds.

Functions
---------
get_nhc_windswath: return spaceTimeBounds instance for an NHC-named storm.
get_custom_bounds: return spaceTimeBounds instance for user-defined event.

"""


import stormevents
import shapely
import pandas
import numpy
import datetime
import sys
from pydantic import BaseModel
from typing import Union
import logging


logger = logging.getLogger(__name__)


# TODO: consider enum class for wind speed


class eventSpaceTimeBounds:
    """Define start & end time and spatial bounds of an event.

    Attributes
    ----------
    name: str
        The user-defined name of the storm/event.
    year: int
        The year of the storm/event.
    nhc_name: str | None
        The name of the storm as used by the National Hurricane Center
        If None, defaults to the name attribute, in lower case.
    nhc_storm_event: StormEvent | None
        StormEvent instance from the stormevents package
        
    {TBC}
    
    """

    def __init__(self, conf: dict) -> None:
        """Initialize eventSpaceTimeBounds instance.

        Parameters
        ----------
        conf: dict
            The "event" section of an event's config file.
            Do not pass the entire config file dictionary!
            
        """
        # Name and year (required).
        self.name = conf['name']
        self.year = conf['year']
        # Most values default to None.
        self.nhc_name = None
        self.nhc_storm_event = None
        self.nhc_track_type = None
        self.nhc_track_datetime = None
        self.user_start_datetime = None
        self.user_end_datetime = None
        self.user_bounding_box = None
        # NHC details (optional).
        if conf['nhc_named_storm']:
            if conf['nhc_name']:
                self.nhc_name = conf['nhc_name']
            else:
                self.nhc_name = conf['name'].lower()
            self.nhc_storm_event = stormevents.StormEvent(self.nhc_name, 
                                                          self.year)
            self.nhc_track_type = conf['nhc_track_type']
            if conf['nhc_track_datetime']:
                self.nhc_track_datetime = datetime.datetime.fromisoformat(
                    conf['nhc_track_datetime']
                )
        # User-defined details (required if not an NHC named storm).
        if conf['user_start_datetime']:
            self.user_start_datetime = datetime.datetime.fromisoformat(
                conf['user_start_datetime']
            )
        if conf['user_end_datetime']:
            self.user_end_datetime = datetime.datetime.fromisoformat(
                conf['user_end_datetime']
            )
        if conf['user_bounding_box']:
            self.user_bounding_box = shapely.box(*conf['user_bounding_box'])
        return None
        
    @property
    def start_datetime(self):
        if self.user_start_datetime:
            return self.user_start_datetime
        else:
            return self.nhc_storm_event.start_date.to_pydatetime()
                
    @property
    def end_datetime(self):
        if self.user_end_datetime:
            return self.user_end_datetime
        else:
            return self.nhc_storm_event.end_date.to_pydatetime()
                
    def get_swath_shape(self, wind_speed: int) -> shapely.Polygon:
        """
        """
        swaths = self.nhc_storm_event.track().wind_swaths(wind_speed=wind_speed)
        # Get/check track type ('BEST', 'OFCL', ...).
        if self.nhc_track_type in swaths.keys():
            track_type = self.nhc_track_type
        else:
            raise ValueError(f'Track type {self.nhc_track_type} not available in storm event.')
        # Get/check track date.
        track_dates = list(swaths[track_type].keys())
        if self.nhc_track_datetime:
            if self.nhc_track_datetime in track_dates:
                track_date = self.nhc_track_datetime
            else:
                raise ValueError(f'Track date {self.nhc_track_datetime} not available in storm event')
        else:
            if len(track_dates) > 1:
                logger.info("More than one track date given in wind swaths: using the last.")
            track_date = track_dates[-1]
        # Extract the swath.
        return swaths[track_type][track_date]
            
    def get_region(self, region_type: str):
        """
        """
        if region_type == 'box':
            if self.user_bounding_box:
                return self.user_bounding_box
            else:
                swath = self.get_swath_shape(34)
                return shapely.box(*shapely.bounds(swath))
        elif region_type in ['34kt', 34]:
            return self.get_swath_shape(34)
        elif region_type in ['50kt', 50]:
            return self.get_swath_shape(50)
        elif region_type in ['64kt', 64]:
            return self.get_swath_shape(64)
        else:
            raise ValueError(f'region_type {region_type} not recognized.')           



