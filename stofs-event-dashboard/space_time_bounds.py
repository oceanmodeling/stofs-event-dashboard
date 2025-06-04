"""Define class used when specifying an event. 

Classes
-------
eventSpaceTimeBounds: contains event time and space bounds.

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
        StormEvent instance from the stormevents package. This attribute 
        is initialized only if the config has "nhc_named_storm":true.
        If None, then user_start_datetime, user_end_datetime, and
        user_bounding_box must be provided in the config.
    nhc_track_type: str 
        The track type descriptor used by NHC.
        Usually "BEST", or maybe "OFCL".
    nhc_track_datetime: datetime | None
        The datetime label associated with a storm track, as used
        by NHC. If None, defaults to the latest available. 
    user_start_datetime: datetime | None
        User defined event start datetime.
    user_end_datetime: datetime | None
        User defined event end datetime.
    user_bounding_box: 
        User defined event bounds, with the format:
        list[lon_min, lat_min, lon_max, lat_max].
    start_datetime
        Returns user_start_datetime if defined, otherwise NHC storm start time.
        Actually a @property method.
    end_datetime
        Returns user_end_datetime if defined, otherwise NHC storm end time.
        Actually a @property method.
        
    Methods
    -------
    get_swath_shape(wind_speed)
        Returns a polygon defining a wind swath for the given wind_speed.
    get_region(region_type)
        Returns a polygon corresponding to either a wind swath or 
        a simple lat-lon box. For box type, the polygon is either
        user_bounding_box, if defined, or a box containing the 
        34 knot wind swath.
    
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
        """user_start_datetime if defined, otherwise NHC storm start time."""
        if self.user_start_datetime:
            return self.user_start_datetime
        else:
            return self.nhc_storm_event.start_date.to_pydatetime()
                
    @property
    def end_datetime(self):
        """user_end_datetime if defined, otherwise NHC storm end time."""
        if self.user_end_datetime:
            return self.user_end_datetime
        else:
            return self.nhc_storm_event.end_date.to_pydatetime()
                
    def get_swath_shape(self, wind_speed: int) -> shapely.Polygon:
        """Return wind swath for one of 3 available wind speeds.

        A wind swath delineates all the locations that experienced winds
        at or greater than the given wind_speed, over the history of a storm.

        Parameters
        ----------
        wind_speed
            Wind speed lower limit in knots. One of {34, 50, 64}.

        Returns
        -------
        Shapely polygon defining the wind swath.
            
        """
        if self.nhc_storm_event:
            swaths = self.nhc_storm_event.track().wind_swaths(wind_speed=wind_speed)
            if swaths:
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
            else:
                logger.info(f'{wind_speed} kt wind swath not available for this event.')
                return None
        else:
            logger.info('Wind swaths not available for this event.')
            return None
            
    def get_region(self, region_type: str):
        """Return a wind swath or box polygon for an event.

        The requested region can be one of the pre-defined wind speeds
         as used by NHC (34, 50, or 64 knots), or "box". If a box is
         requested, the polygon is either the user-defined user_bounding_box
         or the lat-lon box that contains the 34 knot wind swath.

        Parameters
        ----------
        region_type
            The type of region polygon to return. One of
            {'box', '34kt', '50kt', '64kt'}.

        Returns
        -------
        Shapely polygon corresponding to the requested region.
        
        """
        if region_type == 'box':
            if self.user_bounding_box:
                return self.user_bounding_box
            else:
                swath = self.get_swath_shape(34)
                return shapely.box(*shapely.bounds(swath))
        elif region_type in ['34kt', '34', 34]:
            return self.get_swath_shape(34)
        elif region_type in ['50kt', '50', 50]:
            return self.get_swath_shape(50)
        elif region_type in ['64kt', '64', 64]:
            return self.get_swath_shape(64)
        else:
            raise ValueError(f'region_type {region_type} not recognized.')           



