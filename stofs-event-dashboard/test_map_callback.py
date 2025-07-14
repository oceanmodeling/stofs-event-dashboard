from __future__ import annotations

import datetime
import json
import logging
import typing as T

import colorcet as cc
import holoviews as hv
import hvplot.pandas  # noqa: F401
import folium
import numpy as np
import natsort
import pandas as pd
import geopandas as gpd
import panel as pn
import pyarrow.parquet as pq
import pyextremes
import seastats
from upath import UPath
import thalassa
import geoviews as gv
import holoviews as hv
from holoviews import opts as hvopts


pn.extension(
    "tabulator",
    throttled=True,
    inline=True,
    ready_notification="Ready",
    sizing_mode="stretch_width",
)

DATA_DIR = UPath("../data")

def apply_callback(event):
    print(event)
    pn.state.location._update_synced(event)
    print(111)


class UI_storm:
    storm = pn.widgets.Select(
        name="Event",
        options=['alberto_2024']
    )
    station_options = list(gpd.read_file(DATA_DIR / f"alberto_2024/map_data.gpkg", layer='stations').loc[:,'nos_id'].values)
    apply_storm = pn.widgets.Button(name="Load event", button_type="primary")
    apply_storm.on_click(apply_callback)

    
class UI:
    on_storm_apply = pn.depends(UI_storm.apply_storm)
    station = pn.widgets.Select(
        name="Station",
        options=UI_storm.station_options
    )

    
def get_thalassa_map(event):
    # Set some defaults for the visualization of the graphs
    hv.extension("bokeh")
    hvopts.defaults(
        hvopts.Image(
            width=800,
            height=1500,
            show_title=True,
            tools=["hover"],
            active_tools=["pan", "box_zoom"],
        ),
    )
    try:
        # Load data.
        storm = UI_storm.storm.value #'alberto_2024'
        map_regions = gpd.read_file(DATA_DIR / f"{storm}/map_data.gpkg", 
                                    layer='regions')
        map_stations = gpd.read_file(DATA_DIR / f"{storm}/map_data.gpkg", 
                                     layer='stations')
        # Define center point of map
        try:
            centr = map_regions.geometry[0].centroid
        except:
            centr = map_stations.geometry[0]
        # Create map.
        tm = thalassa.api.get_tiles().opts(height=500, width=1000)
        # Add regions.
        reg_colors = {'box':'#377eb8', '34kt':'#ff7f00', '50kt':'#4daf4a', '64kt':'#f781bf'}
        for reg in map_regions.index:
            print(map_regions.loc[[reg]])
            tm = tm * gv.Path(map_regions.loc[[reg]], vdims=['region']).opts(tools=['hover'], color=reg_colors[map_regions.loc[reg, 'region']])
        # Add station markers.
        st_plot = gv.Points(map_stations, vdims=['name', 'nos_id', 'nws_id', 'station_type']).opts(tools=['hover'], size=8)
        # Add station marker interactivity.
        stream = hv.streams.Tap(source=st_plot, x=np.nan, y=np.nan)
        @pn.depends(stream.param.x, stream.param.y)
        def update_station_from_map(x, y):
            print(x)
            print(y)
            matching_station_id = map_stations.nos_id[
                (np.abs(map_stations.latitude - y) <= 0.1) &
                (np.abs(map_stations.longitude - x) <= 0.1)
            ].values
            if len(matching_station_id) > 0:
                print(f"Updating station to {matching_station_id[0]} based on map click ")
                UI.station.value = matching_station_id[0]
                print(f"UI.station.value is now {UI.station.value}")
                print(matching_station_id[0])
            return pn.pane.Str(matching_station_id)
        @pn.depends(UI.station, watch=True)
        def get_UI_station(stn):
            return pn.pane.Str(stn)
        tm = pn.Column(tm * st_plot, update_station_from_map, get_UI_station)
    except Exception as e:
        print(e)
        tm = thalassa.api.get_tiles()
    return tm


def get_page():
    on_storm_apply = pn.depends(UI_storm.apply_storm)
    page = pn.template.MaterialTemplate(
        title="STOFS event analysis",
        sidebar=[
            pn.pane.Str('\nEvent'),
            UI_storm.storm,
            UI_storm.apply_storm,
            UI.station
        ],
        sidebar_width=350,
        main=pn.Column(
            pn.Tabs(
                ('Map', on_storm_apply(get_thalassa_map)),
                tabs_location='left',
                dynamic=True
            )
        ),
    )
    return page


page = get_page()
page.servable()