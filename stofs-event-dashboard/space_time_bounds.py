import stormevents
import shapely
import pandas
import numpy
import datetime
import sys
from pydantic import BaseModel
from typing import Union

# TODO: add proper logging
# TODO: consider enum class for wind speed

class spaceTimeBounds(BaseModel):
    start_time: Union[pandas.Timestamp, numpy.datetime64, datetime.datetime] 
    end_time: Union[pandas.Timestamp, numpy.datetime64, datetime.datetime]
    region: shapely.Polygon
    # TODO: add region_type string or similar?
    #
    class Config:
        arbitrary_types_allowed = True
        strict = True


def get_nhc_windswath(storm_name: str, 
                      storm_year: int, 
                      wind_speed:int = 50) -> spaceTimeBounds:
    """Get a wind swath for a given named storm.

    Arguments:
        storm_name: the name of the storm, usually lowercase, e.g., "milton"
        storm_year: the year in which the storm occured
        wind_speed: the wind speed of the isotach defining the wind swath polygon.
            Must be one of {34, 50, 68}.

    Returns:
        A spaceTimeBounds object containing the storm start and end dates
        and a shapely polygon defining the region which experienced wind speeds 
        greater than or equal to the specified wind_speed.
    """

    try:
        storm = stormevents.StormEvent(storm_name, storm_year)
        swath = storm.track().wind_swaths(wind_speed=wind_speed)
    except:
        # TODO add some interactivity code here?
        sys.exit(f"Cannot find data for {storm_name} in {storm_year}")

    if 'BEST' in swath.keys():
        track_type = 'BEST'
    elif 'OFCL' in swath.keys():
        track_type = 'OFCL'
    else:
        sys.exit("wind_swaths does not have BEST or OFCL track type.")

    track_dates = list(swath[track_type].keys())
    if len(track_dates) > 1:
        print("Warning: more than one track date given in wind_swaths: using the last.")

    return spaceTimeBounds(start_time=storm.start_date,
                           end_time=storm.end_date,
                           region=swath[track_type][track_dates[-1]])


def get_custom_bounds(
        start_time: Union[pandas.Timestamp, numpy.datetime64, datetime.datetime],
        end_time: Union[pandas.Timestamp, numpy.datetime64, datetime.datetime],
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float
    ) -> spaceTimeBounds:
    """Get a custom spaceTimeBounds object with a simple box polygon.
    """
    return spaceTimeBounds(start_time=start_time,
                           end_time=end_time,
                           region=shapely.box(lon_min, lat_min, lon_max, lat_max))

