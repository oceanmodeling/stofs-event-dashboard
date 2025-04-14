from __future__ import annotations

import datetime
import json
import logging
import typing as T

import colorcet as cc
import holoviews as hv
import hvplot.pandas  # noqa: F401
import numpy as np
import natsort
import pandas as pd
import panel as pn
import pyarrow.parquet as pq
import pyextremes
import seastats
from upath import UPath

from sealens._common import get_template_class

pn.extension(
    "tabulator",
    throttled=True,
    inline=True,
    ready_notification="Ready",
    sizing_mode="stretch_width",
)


# FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
FORMAT = "{levelname:8s}; {asctime:s}; {name:<25s} {funcName:<15s} {lineno:4d}; {message:s}"


@pn.cache
def reconfig_basic_config(format_=FORMAT, level=logging.DEBUG):
    """(Re-)configure logging"""
    logging.basicConfig(format=format_, style="{", level=level, force=True)


reconfig_basic_config()
logger = logging.getLogger(name="sealens")

DATA_DIR = UPath("../data")
OBS_DIR = DATA_DIR / "obs"


def get_event_paths() -> list[UPath]:
    paths = natsort.humansorted(DATA_DIR.glob("[!.]*/"))
    return paths

    
def get_event_names() -> list[str]:
    names = [p.name for p in get_event_paths()]
    return names

    
def get_parquet_files(mn: str) -> list[UPath]:
    paths = natsort.natsorted(DATA_DIR.joinpath(mn).glob("models/**/*.parquet"))
    return paths


def get_model_paths(mn: str) -> list[UPath]:
    paths = natsort.natsorted(set(path.parent for path in get_parquet_files(mn)))
    return paths


def get_model_names(mn: str) -> list[str]:
    names = [path.name for path in get_model_paths(mn)]
    return names


def get_obs_station_paths(mn: str) -> list[UPath]:
    paths = natsort.natsorted(DATA_DIR.joinpath(mn).glob("obs/*.parquet"))
    return paths


def get_obs_station_names(mn: str) -> list[str]:
    names = [path.stem for path in get_obs_station_paths(mn)]
    return names


def get_station_names(mn: str) -> list[str]:
    stations = set(get_obs_station_names(mn))
    for model in get_model_paths(mn):
        stations.update(path.stem for path in model.glob("*.parquet"))
    return natsort.natsorted(stations)


def get_parquet_attrs(path):
    pq_metadata = pq.read_metadata(path)
    return json.loads(pq_metadata.metadata[b"PANDAS_ATTRS"])


def get_observation_metadata() -> pd.DataFrame:
    df = pd.DataFrame(get_parquet_attrs(path) for path in get_obs_station_paths())
    return df


@pn.cache
def load_data(path: UPath) -> pd.Series[float]:
    df = pd.read_parquet(path)
    column = df.columns[0]
    return df[column]


class UI:
    weather_events = pn.widgets.Select(
        name="Events",
        options=get_event_names()
    )
    models = pn.widgets.CheckButtonGroup(
        name="Models",
        button_type="primary",
        description="which models to include",
        orientation="vertical",
        button_style="outline",
        options=pn.bind(get_model_names, weather_events.param.value),
    )
    station = pn.widgets.Select(
        name="Station",
        options=pn.bind(get_station_names, weather_events.param.value),
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


@pn.depends(UI.weather_events)
def get_static_map(weather_event):
    static_map = pn.pane.image.PNG(
        object=DATA_DIR / f"{weather_event}/static_map.png", 
        height=400
    )
    return static_map


@pn.depends(UI.weather_events,
            UI.models,
            UI.station, 
            UI.quantile, 
            UI.window)
def plot_ts(weather_event, models, station, percentile, window):
    quantile = percentile / 100
    # obs
    obs = load_data(DATA_DIR / f"{weather_event}/obs/{station}.parquet")
    start, end = obs.index.min(), obs.index.max()
    obs_threshold = obs.quantile(quantile)
    logger.info("obs len: %r", len(obs))
    logger.info("obs quantile: %r", obs_threshold)
    logger.info("obs describe:\n%r", obs.describe())
    # obs = obs.resample("4min").mean().shift(freq="2min")
    obs_ext = pyextremes.get_extremes(obs, "POT", threshold=obs_threshold, r=f"{window}h")
    # obs_ext = pyextremes.get_extremes(
    #     obs[obs >= obs_threshold], "BM", block_size=f"{window}h", errors="ignore"
    # ).sort_values(ascending=False)
    logger.info("obs ext:\n%r", obs_ext)
    # sims
    sims = {model: load_data(DATA_DIR / f"{weather_event}/models/{model}/{station}.parquet")[start:end] for model in models}
    # plots
    timeseries = [obs.hvplot(label="obs", color="lightgrey")]
    for i, (model, ts) in enumerate(sims.items()):
        timeseries += [ts.hvplot(label=model).opts(color=cc.glasbey_dark[i])]
    timeseries += [hv.HLine(obs_threshold).opts(color="grey", line_dash="dashed", line_width=1)]
    timeseries += [obs_ext.hvplot.scatter(label="obs extreme")]
    return hv.Overlay(timeseries).opts(show_grid=True, active_tools=["box_zoom"], min_height=300, ylabel="sea elevation")
    

page = pn.template.MaterialTemplate(
    title="STOFS event analysis",
    sidebar=[
        pn.pane.Str('\nData'),
        UI.weather_events,
        UI.station,
        UI.models,
        pn.pane.Str('\n\n\n\nStatistics'),
        #UI.metric,
        UI.quantile,
        UI.window,
    ],
    sidebar_width=430,
    main=pn.Column(
        get_static_map,
        plot_ts,
    ),
)
page.servable()