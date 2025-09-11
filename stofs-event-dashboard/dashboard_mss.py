"""STOFS event dashboard front end for multi-station stats.

Throughout this module, "storm" refers to the weather event,
while "event" refers to web page interactive click events.

Data is stored in parquet files with a directory structure like:
    data/<storm>/<variable group>/<forecast_type>/<model>/<station>.parquet
and
    data/<storm>/<variable group>/obs/<station>.parquet
    
E.g., data/milton_2024/cwl/nowcast/stofs2d/nos_8720219.parquet
      data/milton_2024/cwl/obs/nos_8720219.parquet

Each file can have multiple columns, each of which can be 
displayed on the same plot (e.g., cwl_raw and cwl_bias_corrected)

Options are then selected
1. Storm (drop down)
2. Plot type (drop down) [group of variables, e.g., CWL, wind]
3. Forecast type (drop down) [nowcast or specific forecast init]
4. Station (drop down)
5. Model + variable (toggle buttons)

"""


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
from dashboard import (
    get_event_paths,
    get_event_names,
    get_plot_type_paths,
    get_plot_type_names,
    get_fc_type_paths,
    get_fc_type_names,
    get_all_station_paths,
    get_all_station_names,
    get_model_paths,
    get_model_names,
    get_model_vars,
    load_data,
    apply_callback,
    UI_storm,
    load_obs_sims
)


pn.extension(
    "tabulator",
    throttled=True,
    inline=True,
    ready_notification="Ready",
    sizing_mode="stretch_width",
)


FORMAT = "{levelname:8s}; {asctime:s}; {name:<25s} {funcName:<15s} {lineno:4d}; {message:s}"

@pn.cache
def reconfig_basic_config(format_=FORMAT, level=logging.DEBUG):
    """(Re-)configure logging"""
    logging.basicConfig(format=format_, style="{", level=level, force=True)

reconfig_basic_config()
logger = logging.getLogger(name="sealens")


DATA_DIR = UPath("../data")

# ----------------------------------------------------------------------------
# Functions for user interface (data selection and updates).
# ----------------------------------------------------------------------------

class UI:
    on_storm_apply = pn.depends(UI_storm.apply_storm)
    plot_type = pn.widgets.Select(
        name="Plot type",
        options=on_storm_apply(get_plot_type_names),
    )
    forecast_type = pn.widgets.Select(
        name="Forecast type",
        options=on_storm_apply(get_fc_type_names),
    )
    station = pn.widgets.Select(
        name="Station",
        options=on_storm_apply(get_all_station_names),
    )
    metric = pn.widgets.Select(
        name="Metric",
        options=seastats.SUGGESTED_METRICS,
    )
    quantile = pn.widgets.FloatInput(
        name="POT Quantile (%)",
        value=99.0,
        step=0.1,
        start=0.0,
        end=99.9,
        page_step_multiplier=10,
    )
    window = pn.widgets.IntInput(
        name="POT Window (hours)",
        value=24,
        start=1,
        step=6,
    )
    plot_apply = pn.widgets.Button(name="Update plots", button_type="primary")
    plot_apply.on_click(apply_callback)


class UI_toggle:
    on_plot_apply = pn.depends(UI.plot_apply)
    models_vars = pn.widgets.CheckButtonGroup(
        name="Models/variables",
        button_type="primary",
        description="which models and variables to include",
        orientation="vertical",
        button_style="outline",
        options=on_plot_apply(get_model_vars),
    )
    

# ----------------------------------------------------------------------------
# Functions for data plots.
# ----------------------------------------------------------------------------

def get_plot_label(plot_type: str) -> str:
    """Get plot axis label with units."""
    if plot_type in ['cwl']:
        return 'water elevation (m)'
    elif plot_type in ['pressure']:
        return 'surface air pressure (hPa)'
    elif plot_type in ['wind']:
        return 'wind speed (m/s)'
    else:
        logger.info(f'No plot label for plot_type {plot_type}.')
        return ''


def load_all_obs_sims

def standalone_stats

def comparison_stats


def plot_table_comparison(event):
    quantile: float = T.cast(float,  UI.quantile.value) / 100
    title = f"{UI_storm.storm.value}: {UI.forecast_type.value} at {UI.station.value}"
    try:
        # Load obs and model data.
        obs, sims = load_obs_sims(
            UI_storm.storm.value, 
            UI.plot_type.value, 
            UI.forecast_type.value, 
            UI_toggle.models_vars.value, 
            UI.station.value
        )
        if (not obs.empty) & (len(sims) > 0):
            # Do separate storm and general stats calculations because
            # they should use different quantiles.
            stats = {}
            for model, sim in sims.items():
                try:
                    stats_general = seastats.get_stats(
                        sim, obs,
                        metrics=seastats.GENERAL_METRICS,
                        quantile=0, round=3
                    )
                except Exception as e:
                    logger.info(f'{title}: Error while calculating {model} {UI.plot_type.value} general statistics: {e}')
                    stats_general = {}
                try:
                    stats_storm = seastats.get_stats(
                        sim, obs,
                        metrics=seastats.STORM_METRICS,
                        quantile=quantile,
                        cluster=UI.window.value, round=3
                    )
                except Exception as e:
                    logger.info(f'{title}: Error while calculating {model} {UI.plot_type.value} storm statistics: {e}')
                    stats_storm = {}
                stats[model] = {**stats_general, **stats_storm}
            logger.info("stats:\n%r", stats)
            return pn.widgets.DataFrame(pd.DataFrame(stats).T, 
                                        sortable=True)
        else:
            return pn.pane.Str(f'{title}: No {UI.plot_type.value} statistics data.')
    except Exception as e:
        logger.info(f'{title}: Error while calculating {UI.plot_type.value} statistics: {e}')
        return pn.pane.Str(f'{title}: Error while calculating {UI.plot_type.value} statistics.')


def get_stats_tabs(event):
    """
    """
    # Get stats for obs.
    result = pn.Tabs(
        ('obs.', on_plot_apply(plot_ts)), 
        tabs_location='above',
        dynamic=True
    )
    # Loop over models and add 
    # standalone stats
    # and
    # obs bias stats
    for model in models:
        ...
    return result

# ----------------------------------------------------------------------------
# Put together dashboard.
# ----------------------------------------------------------------------------

def get_page():
    on_storm_apply = pn.depends(UI_storm.apply_storm)
    on_plot_apply = pn.depends(UI.plot_apply)
    tabs = on_plot_apply(get_stats_tabs)
    page = pn.template.MaterialTemplate(
        title="STOFS event analysis",
        sidebar=[
            pn.pane.Str('\nEvent'),
            UI_storm.storm,
            UI_storm.apply_storm,
            pn.pane.Str('\n\nData'),
            UI.plot_type,
            UI.forecast_type,
            pn.pane.Str('\n\nStatistics'),
            #UI.metric,
            UI.quantile,
            UI.window,
            UI.plot_apply,
        ],
        sidebar_width=350,
        main=pn.Column(tabs)
    return page


page = get_page()
page.servable()
