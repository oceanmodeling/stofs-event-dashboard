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


"""
Data storage is like:
    data/<event>/<variable group>/<forecast_type>/<model>/<station>.parquet
and
    data/<event>/<variable group>/obs/<station>.parquet
E.g., data/milton_2024/cwl/nowcast/stofs2d/nos_8720219.parquet
      data/milton_2024/cwl/obs/nos_8720219.parquet

Each file can have multiple columns, each of which can be 
displayed on the same plot (e.g., cwl_raw and cwl_bias_corrected)

Options are then selected
1. Event (drop down)
2. Plot type (drop down) [group of variables, e.g., CWL, wind]
3. Forecast type (drop down) [nowcast or specific forecast init]
4. Station (drop down)
5. Model + variable (toggle buttons)
"""


def get_event_paths() -> list[UPath]:
    paths = natsort.humansorted(DATA_DIR.glob("[!.]*/"))
    return paths

    
def get_event_names() -> list[str]:
    names = [p.name for p in get_event_paths()]
    return names


def get_plot_type_paths(ev: str) -> list[UPath]:
    paths = natsort.humansorted(DATA_DIR
                                .joinpath(ev)
                                .glob("[!.]*/"))
    return paths

    
def get_plot_type_names(ev: str) -> list[str]:
    names = [p.name for p in get_plot_type_paths(ev)]
    return names


def get_fc_type_paths(ev: str, plot: str) -> list[UPath]:
    paths = natsort.humansorted(DATA_DIR
                                .joinpath(ev)
                                .joinpath(plot)
                                .glob("[!.]*/"))
    remove_types = ['obs']
    return [p for p in paths if p.name not in remove_types]

    
def get_fc_type_names(ev: str, plot: str) -> list[str]:
    names = [p.name for p in get_fc_type_paths(ev, plot)]
    return names


def get_model_paths(ev: str, plot: str, fc: str) -> list[UPath]:
    paths = natsort.humansorted(DATA_DIR
                                .joinpath(ev)
                                .joinpath(plot)
                                .joinpath(fc)
                                .glob("[!.]*/"))
    return paths


def get_model_names(ev: str, plot: str, fc: str) -> list[str]:
    names = [path.name for path in get_model_paths(ev, plot, fc)]
    return names

    
def get_all_station_paths(ev: str, plot: str) -> list[UPath]:
    paths = natsort.natsorted(DATA_DIR
                              .joinpath(ev)
                              .joinpath(plot)
                              .glob("**/*.parquet"))
    return paths


def get_all_station_names(ev: str, plot: str) -> list[str]:
    names = [path.stem for path in get_all_station_paths(ev, plot)]
    return names


def get_obs_station_paths(ev: str, plot: str) -> list[UPath]:
    paths = natsort.natsorted(DATA_DIR
                              .joinpath(ev)
                              .joinpath(plot)
                              .glob("obs/*.parquet"))
    return paths


def get_obs_station_names(ev: str, plot: str) -> list[str]:
    names = [path.stem for path in get_obs_station_paths(ev, plot)]
    return names


def get_station_names(ev: str, plot: str) -> list[str]:
    stations = set(get_all_station_names(ev, plot))
    return natsort.natsorted(stations)


def get_model_vars(ev: str, plot: str, fc: str, stn:str) -> list[str]:
    model_names = get_model_names(ev, plot, fc)
    result = []
    for mn in model_names:
        df = load_data(DATA_DIR
                       .joinpath(ev)
                       .joinpath(plot)
                       .joinpath(fc)
                       .joinpath(mn)
                       .joinpath(stn + '.parquet'))
        for var in df.columns:
            result.append((mn, var))
    return result


def get_parquet_attrs(path):
    pq_metadata = pq.read_metadata(path)
    return json.loads(pq_metadata.metadata[b"PANDAS_ATTRS"])


def get_observation_metadata() -> pd.DataFrame:
    df = pd.DataFrame(get_parquet_attrs(path) for path in get_obs_station_paths())
    return df


@pn.cache
def load_data(path: UPath) -> pd.Series[float]:
    df = pd.read_parquet(path)
    return df


class UI:
    weather_event = pn.widgets.Select(
        name="Event",
        options=get_event_names()
    )
    plot_type = pn.widgets.Select(
        name="Plot type",
        options=pn.bind(get_plot_type_names, 
                        weather_event.param.value),
    )
    forecast_type = pn.widgets.Select(
        name="Forecast type",
        options=pn.bind(get_fc_type_names, 
                        weather_event.param.value, 
                        plot_type.param.value),
    )
    station = pn.widgets.Select(
        name="Station",
        options=pn.bind(get_station_names, 
                        weather_event.param.value, 
                        plot_type.param.value),
    )
    models_vars = pn.widgets.CheckButtonGroup(
        name="Models/variables",
        button_type="primary",
        description="which models and variables to include",
        orientation="vertical",
        button_style="outline",
        options=pn.bind(get_model_vars, 
                        weather_event.param.value, 
                        plot_type.param.value, 
                        forecast_type.param.value, 
                        station.param.value),
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


@pn.depends(UI.weather_event)
def get_static_map(wx_ev):
    static_map = pn.pane.image.PNG(
        object=DATA_DIR / f"{wx_ev}/static_map.png", 
        height=300
    )
    return static_map


@pn.depends(UI.weather_event,
            UI.station)
def get_folium_map(wx_ev, station):
    # Load data.
    map_regions = gpd.read_file(DATA_DIR / f"{wx_ev}/map_data.gpkg", 
                                layer='regions')
    map_stations = gpd.read_file(DATA_DIR / f"{wx_ev}/map_data.gpkg", 
                                 layer='stations')
    # Define center point of map
    try:
        centr = map_regions.geometry[0].centroid
    except:
        centr = map_stations.geometry[0]
    # Create map.
    fm = folium.Map(
        location=(centr.y, centr.x), 
        zoom_start=6,
    )
    # Add regions.
    reg_colors = {'box':'#377eb8', '34kt':'#ff7f00', '50kt':'#4daf4a', '64kt':'#f781bf'}
    for reg in map_regions.index:
        folium.Polygon(
            locations=[[y,x] for x,y in 
                       map_regions.loc[reg, 'geometry'].exterior.coords],
            color=reg_colors[map_regions.loc[reg, 'region']],
            weight=2,
            fill=False,
            tooltip=map_regions.loc[reg, 'region'],
        ).add_to(fm)
    # Add station markers.
    for st in map_stations.index:
        iframe = folium.IFrame(f"<b>{map_stations.loc[st, 'nos_id']}</b><br><i>Name:</i> {map_stations.loc[st, 'name']}<br><i>Station type(s):</i> {map_stations.loc[st, 'station_type']}<br><i>NOS ID:</i> {map_stations.loc[st, 'nos_id']}<br><i>NWS ID:</i> {map_stations.loc[st, 'nws_id']}")
        popup = folium.Popup(iframe, min_width=300, max_width=300, max_height=120, min_height=120)
        folium.Marker(
            [map_stations.geometry[st].y,map_stations.geometry[st].x],
            popup=popup 
        ).add_to(fm)
    return fm


def get_plot_label(plot_type: str) -> str:
    if plot_type in ['cwl']:
        return 'water elevation (m)'
    else:
        logger.info(f'No plot label for plot_type {plot_type}.')
        return ''
        

@pn.depends(UI.weather_event,
            UI.plot_type,
            UI.forecast_type,
            UI.models_vars,
            UI.station, 
            UI.quantile, 
            UI.window)
def plot_ts(wx_ev, plot_type, fc_type, modvars, station, percentile, window):
    quantile = percentile / 100
    # obs
    obs = load_data(DATA_DIR.joinpath(wx_ev, plot_type, 'obs', station +'.parquet')).iloc[:,0]
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
    sims = {model + '_' + var: load_data(DATA_DIR.joinpath(
        wx_ev, 
        plot_type, 
        fc_type, 
        model, 
        station + '.parquet'
    )).loc[start:end, var] for (model,var) in modvars}
    print(sims)
    # plots
    timeseries = [obs.hvplot(label="obs", color="lightgrey")]
    for i, (model, ts) in enumerate(sims.items()):
        timeseries += [ts.hvplot(label=model).opts(color=cc.glasbey_dark[i])]
    timeseries += [hv.HLine(obs_threshold).opts(color="grey", line_dash="dashed", line_width=1)]
    timeseries += [obs_ext.hvplot.scatter(label="obs extreme")]
    ylabel = get_plot_label(plot_type)
    return hv.Overlay(timeseries).opts(show_grid=True, active_tools=["box_zoom"], min_height=300, ylabel=ylabel)


@pn.depends(UI.weather_event,
            UI.plot_type,
            UI.forecast_type,
            UI.models_vars,
            UI.station, 
            UI.quantile, 
            UI.window)
def plot_table_comparison(wx_ev, plot_type, fc_type, modvars, station, percentile, window):
    quantile: float = T.cast(float, percentile) / 100
    obs = load_data(DATA_DIR.joinpath(wx_ev, plot_type, 'obs', station +'.parquet')).iloc[:,0]
    start, end = obs.index.min(), obs.index.max()
    sims = {model + '_' + var: load_data(DATA_DIR.joinpath(wx_ev, 
                                               plot_type, 
                                               fc_type, 
                                               model, 
                                               station + '.parquet'))
                   .loc[start:end, var] 
            for (model,var) in modvars}
    stats = {
        model: seastats.get_stats(sim, obs, quantile=quantile, cluster=window, round=3)
        for model, sim in sims.items()
    }
    logger.info("stats:\n%r", stats)
    return pd.DataFrame(stats).T


@pn.depends(UI.weather_event,
            UI.plot_type,
            UI.forecast_type,
            UI.models_vars,
            UI.station)
def plot_scatter(wx_ev, plot_type, fc_type, modvars, station):
    # data
    obs = load_data(DATA_DIR.joinpath(wx_ev, plot_type, 'obs', station +'.parquet')).iloc[:,0]
    start, end = obs.index.min(), obs.index.max()
    sims = {model + '_' + var: load_data(DATA_DIR.joinpath(wx_ev, 
                                               plot_type, 
                                               fc_type, 
                                               model, 
                                               station + '.parquet'))
                   .loc[start:end, var] 
            for (model,var) in modvars}
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
    plot_type_desc = get_plot_label(plot_type)
    overlay = overlay.opts(
        min_height=400,
        max_height=500,
        show_grid=True,
        legend_position="right",
        title="Title",
        xlabel=f"observed {plot_type_desc}",
        ylabel=f"model {plot_type_desc}",
    )
    return overlay


@pn.depends(UI.weather_event,
            UI.plot_type,
            UI.forecast_type,
            UI.models_vars,
            UI.station, 
            UI.quantile, 
            UI.window)
def plot_extremes_table(wx_ev, plot_type, fc_type, modvars, station, percentile, window):
    quantile = percentile / 100
    # data
    obs = load_data(DATA_DIR.joinpath(wx_ev, plot_type, 'obs', station +'.parquet')).iloc[:,0]
    start, end = obs.index.min(), obs.index.max()
    sims = {model + '_' + var: load_data(DATA_DIR.joinpath(wx_ev, 
                                               plot_type, 
                                               fc_type, 
                                               model, 
                                               station + '.parquet'))
                   .loc[start:end, var] 
            for (model,var) in modvars}
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
        UI.weather_event,
        UI.plot_type,
        UI.forecast_type,
        UI.station,
        UI.models_vars,
        pn.pane.Str('\n\n\n\nStatistics'),
        #UI.metric,
        UI.quantile,
        UI.window,
    ],
    sidebar_width=350,
    main=pn.Column(
        #pn.pane.plot.Folium(get_folium_map, height=300),
        pn.Tabs(
            ('Map', pn.pane.plot.Folium(get_folium_map, height=500)),
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