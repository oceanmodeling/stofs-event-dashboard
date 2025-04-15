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
        height=300
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

@pn.depends(UI.weather_events,
            UI.models,
            UI.station, 
            UI.quantile, 
            UI.window)
def plot_table_comparison(weather_event, models, station, percentile, window):
    quantile: float = T.cast(float, percentile) / 100
    obs = load_data(DATA_DIR / f"{weather_event}/obs/{station}.parquet")
    start, end = obs.index.min(), obs.index.max()
    sims = {model: load_data(DATA_DIR / f"{weather_event}/models/{model}/{station}.parquet")[start:end] for model in models}
    stats = {
        model: seastats.get_stats(sim, obs, quantile=quantile, cluster=window, round=3)
        for model, sim in sims.items()
    }
    logger.info("stats:\n%r", stats)
    return pd.DataFrame(stats).T


@pn.depends(UI.weather_events,
            UI.models,
            UI.station)
def plot_scatter(weather_event, models, station):
    # data
    obs = load_data(DATA_DIR / f"{weather_event}/obs/{station}.parquet")
    start, end = obs.index.min(), obs.index.max()
    sims = {model: load_data(DATA_DIR / f"{weather_event}/models/{model}/{station}.parquet")[start:end] for model in models}
    # plots
    plots = [hv.Slope(1, 0, label="45°").opts(color="grey", show_grid=True)]
    pp_plots = []
    for i, (model, sim) in enumerate(sims.items()):
        color = cc.glasbey_dark[i]
        sim_, obs_ = seastats.align_ts(sim, obs)
        # slope, intercept = seastats.get_slope_intercept(sim_, obs_)
        scatter = hv.Scatter((sim_, obs_), label=model)
        plots.append(scatter.opts(color=color, muted=True))
        plots.append(
            hv.Slope.from_scatter(scatter).opts(
                color=color,
                line_dash="dashed",
            )
        )
        pc1, pc2 = seastats.get_percentiles(sim_, obs_, higher_tail=True)
        pp_scatter = hv.Scatter((pc1, pc2), label=f"Percentiles {model}")
        pp_plots += [
            pp_scatter.opts(color=color, line_color="k", line_width=2, size=8, tools=["hover"]),
            hv.Curve((pc1, pc2)).opts(color=cc.glasbey_dark[-i]),
        ]
    overlay = hv.Overlay(plots + pp_plots)
    overlay = overlay.opts(
        min_height=400,
        max_height=500,
        show_grid=True,
        legend_position="right",
        title="Title",
        xlabel="observed",
        ylabel="model",
    )
    return overlay


@pn.depends(UI.weather_events,
            UI.models,
            UI.station, 
            UI.quantile, 
            UI.window)
def plot_extremes_table(weather_event, models, station, percentile, window):
    quantile = percentile / 100
    # data
    obs = load_data(DATA_DIR / f"{weather_event}/obs/{station}.parquet")
    start, end = obs.index.min(), obs.index.max()
    sims = {model: load_data(DATA_DIR / f"{weather_event}/models/{model}/{station}.parquet")[start:end] for model in models}
    model_dfs = []
    for i, (model, sim) in enumerate(sims.items()):
        # color = cc.glasbey_dark[i]
        ext_df = match_extremes(sim=sim, obs=obs, quantile=quantile, cluster=window)
        logger.info("\n%r", ext_df)
        model_dfs += [ext_df[["model"]].rename(columns={"model": model})]
    if len(sims) > 0:
        model_dfs.insert(0, ext_df[["observed"]].rename(columns={"observed": "obs"}))
        df = pd.concat(model_dfs, axis=1)
        table = pn.widgets.Tabulator(df.round(3), max_height=400)
    else:
        table = pn.widgets.Tabulator(pd.DataFrame())
    return table


def match_extremes(
    sim: pd.Series[float],
    obs: pd.Series[float],
    quantile: float,
    cluster: int
) -> pd.DataFrame:
    # get observed extremes
    ext = pyextremes.get_extremes(obs, "POT", threshold=obs.quantile(quantile), r=f"{cluster}h")
    ext_values_dict: dict[str, T.Any] = {}
    ext_values_dict["observed"] = ext.values
    ext_values_dict["time observed"] = ext.index.values
    #
    max_in_window = []
    tmax_in_window = []
    # match simulated values with observed events
    for it, itime in enumerate(ext.index):
        snippet = sim[itime - pd.Timedelta(hours=cluster / 2) : itime + pd.Timedelta(hours=cluster / 2)]
        try:
            tmax_in_window.append(snippet.index[int(snippet.argmax())])
            max_in_window.append(snippet.max())
        except Exception:
            tmax_in_window.append(itime)
            max_in_window.append(np.nan)
    ext_values_dict["model"] = max_in_window
    ext_values_dict["time model"] = tmax_in_window
    #
    df = pd.DataFrame(ext_values_dict)
    df = df.dropna(subset="model")
    df = df.sort_values("observed", ascending=False)
    df["diff"] = df["model"] - df["observed"]
    df["error"] = abs(df["diff"])
    df["error_norm"] = abs(df["diff"] / df["observed"])
    df["tdiff"] = df["time model"] - df["time observed"]
    df["tdiff"] = df["tdiff"].apply(lambda x: x.total_seconds() / 3600)
    df = df.set_index("time observed")
    return df
    

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
    sidebar_width=350,
    main=pn.Column(
        get_static_map,
        pn.Tabs(
            ('Time series', plot_ts),
            ('Statistics', plot_table_comparison),
            ('Scatter', plot_scatter), 
            ('Extremes', plot_extremes_table),
            tabs_location='left',
            dynamic=True
        )
    ),
)
page.servable()