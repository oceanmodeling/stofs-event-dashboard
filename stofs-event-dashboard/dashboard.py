"""STOFS event dashboard front end.

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
# Functions for enumerating available storm events.
# ----------------------------------------------------------------------------

def get_event_paths() -> list[UPath]:
    """Return paths of storm event data directories."""
    paths = natsort.humansorted(DATA_DIR.glob("[!.]*/"))
    return paths

    
def get_event_names() -> list[str]:
    """Return names of storm events with available data."""
    names = [p.name for p in get_event_paths()]
    if "tests" in names:
        names.remove("tests")
    return names


def get_plot_type_paths(event) -> list[UPath]:
    paths = natsort.humansorted(DATA_DIR
                                .joinpath(UI_storm.storm.value)
                                .glob("[!.]*/"))
    return paths

    
def get_plot_type_names(event) -> list[str]:
    names = [p.name for p in get_plot_type_paths(event)]
    return names


def get_fc_type_paths(event) -> list[UPath]:
    paths = natsort.humansorted(DATA_DIR
                                .joinpath(UI_storm.storm.value)
                                .glob("[!.]*/[!.]*/"))
    remove_types = ['obs']
    return [p for p in paths if p.name not in remove_types]

    
def get_fc_type_names(event) -> list[str]:
    names = set([p.name for p in get_fc_type_paths(event)])
    return natsort.natsorted(names)

    
def get_all_station_paths(event) -> list[UPath]:
    paths = natsort.natsorted(DATA_DIR
                              .joinpath(UI_storm.storm.value)
                              .glob("**/*.parquet"))
    return paths


def get_all_station_names(event) -> list[str]:
    names = set([path.stem for path in get_all_station_paths(event)])
    return natsort.natsorted(names)


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


def get_model_vars(event) -> list[str]:
    plot = UI.plot_type.value 
    fc = UI.forecast_type.value 
    stn = UI.station.value
    model_names = get_model_names(UI_storm.storm.value, plot, fc)
    result = []
    for mn in model_names:
        df = load_data(DATA_DIR
                       .joinpath(UI_storm.storm.value)
                       .joinpath(plot)
                       .joinpath(fc)
                       .joinpath(mn)
                       .joinpath(stn + '.parquet'))
        try:
            for var in df.columns:
                result.append((mn, var))
        except Exception as e:
            logger.info('Error getting data columns from model data frame.')
    return result


#  @pn.cache
def load_data(path: UPath) -> pd.Series[float]:
    """Load data frame from a parquet file."""
    try:
        df = pd.read_parquet(path)
    except:
        df = pd.DataFrame()
    return df


#  @pn.cache
def load_station_metadata() -> pd.Series[float]:
    """Load station metadata from a json file."""
    feet_to_meters = 0.3048
    metadata_path = DATA_DIR.joinpath(
        UI_storm.storm.value,
        'station_metadata',
        f'{UI.station.value}.json'
    )
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        # Change everything to meters (even if we end up
        # changing it back later in the dashboard).
        for levels in ['datums', 'floodlevels']:
            if levels in metadata:
                units = metadata[levels].get('units', 'unknown')
                if units in ['feet', 'foot', 'ft']:
                    for k, v in metadata[levels].get('elevations', {}).items():
                        metadata[levels]['elevations'][k] = v * feet_to_meters
                    metadata[levels]['units'] = 'meters'
                elif units == 'unknown':
                    logger.info(f'Unknown units for {levels} in station metadata at {metadata_path}. Cannot use these {levels}.')
                    metadata[levels] = {}
                elif units in ['meters', 'meter', 'm']:
                    pass
    except:
        metadata = pd.Series()
    return metadata


#  @pn.cache
def get_storm_default_datum(storm: str) -> str:
    """Get default datum for storm event."""
    # Try different possible locations for config file.
    config_paths = [
        UPath(DATA_DIR, storm, f'{storm}.conf').resolve(),
        UPath(DATA_DIR, '..', f'{storm}.conf').resolve(),
        UPath(DATA_DIR, '..', 'conf', f'{storm}.conf').resolve()
    ]
    for config_path in config_paths:
        print('Looking for config file at', config_path)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                output_config = config.get('output', {})
            storm_datum = output_config.get('output_datum', 'No datum found in config file')
            if storm_datum == 'NAVD':
                storm_datum = 'NAVD88'
            return storm_datum
    return 'No config file found'


def get_flood_levels() -> dict[str, float]:
    flood_levels = {}
    st_metadata = load_station_metadata()
    st_datums = st_metadata.get('datums', {}).get('elevations', {})
    st_floodlevels = st_metadata.get('floodlevels', {}).get('elevations', {})
    if UI.datum.value not in st_datums:
        print(f'Cannot find datum {UI.datum.value} in station metadata; returning empty flood levels.')
        return flood_levels
    if ('nos_minor' in st_floodlevels) and (~np.isnan(np.float64(st_floodlevels['nos_minor']))):
        flood_levels['minor'] = float(st_floodlevels['nos_minor']) - st_datums[UI.datum.value]
    elif ('nws_minor' in st_floodlevels) and (~np.isnan(np.float64(st_floodlevels['nws_minor']))):
        flood_levels['minor'] = float(st_floodlevels['nws_minor']) - st_datums[UI.datum.value]
    if ('nos_moderate' in st_floodlevels) and (~np.isnan(np.float64(st_floodlevels['nos_moderate']))):
        flood_levels['moderate'] = float(st_floodlevels['nos_moderate']) - st_datums[UI.datum.value]
    elif ('nws_moderate' in st_floodlevels) and (~np.isnan(np.float64(st_floodlevels['nws_moderate']))):
        flood_levels['moderate'] = float(st_floodlevels['nws_moderate']) - st_datums[UI.datum.value]
    if ('nos_major' in st_floodlevels) and (~np.isnan(np.float64(st_floodlevels['nos_major']))):
        flood_levels['major'] = float(st_floodlevels['nos_major']) - st_datums[UI.datum.value]
    elif ('nws_major' in st_floodlevels) and (~np.isnan(np.float64(st_floodlevels['nws_major']))):
        flood_levels['major'] = float(st_floodlevels['nws_major']) - st_datums[UI.datum.value]
    return flood_levels



# ----------------------------------------------------------------------------
# Functions for user interface (data selection and updates).
# ----------------------------------------------------------------------------

def apply_callback(event):
    print(event)
    pn.state.location._update_synced(event)
    print(111)


def update_storm_datum_display(event):
    """Callback to update the storm datum display."""
    datum = get_storm_default_datum(UI_storm.storm.value)
    UI_storm.storm_default_datum = datum
    UI_storm.storm_default_datum_display.object = f'Default datum: {datum}'


class UI_storm:
    storm = pn.widgets.Select(
        name="Event",
        options=get_event_names()
    )
    storm_default_datum = get_storm_default_datum(storm.value)
    storm_default_datum_display = pn.pane.Str(f'Default datum: {storm_default_datum}')
    apply_storm = pn.widgets.Button(name="Load event", 
                                    button_type="primary")
    apply_storm.on_click(apply_callback)
    apply_storm.on_click(update_storm_datum_display)


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
    datum = pn.widgets.Select(
        name="Datum",
        options=["MSL", "MLLW", "MHHW", "NAVD88"],
        value="MSL"
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
# Functions for data handling.
# ----------------------------------------------------------------------------

def load_obs_sims(
    storm,
    plot_type, 
    forecast_type,
    models_vars,
    station,
    datum
):  
    """Load obs and model data.

    Returns
    -------
    obs
        Pandas data frame of observations; empty if none are available.
    sims
        Dictionary of form {model_var: DataFrame}
        Empty if no model data is available for given inputs.
    """
    try:
        # Load obs.
        obs = load_data(
            DATA_DIR.joinpath(
                storm, 
                plot_type, 
                'obs', 
                station +'.parquet'
            )
        )
        if plot_type in ['cwl']:
            obs = obs.rename(columns={'value':'cwl'})
        obs = convert_datum(obs, datum)
        # Load model data.
        sims = {}
        for (model,var) in models_vars:
            df_sim = load_data(
                DATA_DIR.joinpath(
                    storm, 
                    plot_type, 
                    forecast_type, 
                    model, 
                    station + '.parquet'
                )
            )
            df_sim = convert_datum(df_sim, datum)
            if not df_sim.empty:
                sims[model + '_' + var] = df_sim.loc[:, var]
        # Subset if possible.
        if not obs.empty:
            obs = obs.iloc[:,0]
            start, end = obs.index.min(), obs.index.max()
            sims = {model:ts.loc[start:end] for (model,ts) in sims.items()}
        return obs, sims
    except Exception as e:
        logger.info(f'{storm}: {forecast_type} at {station}: Error while extracting {plot_type} data: {e}')
        return pd.DataFrame(), {}


def convert_datum(df: pd.DataFrame, output_datum: str) -> pd.DataFrame:
    """Convert datum for data frame columns."""
    # Get datum for input data frame.
    if not hasattr(df, 'attrs'):
        df.attrs = {}
    if 'ColumnMetaData' not in df.attrs:
        df.attrs['ColumnMetaData'] = {}
    for col in df.columns:
        if (col not in df.attrs['ColumnMetaData']) and \
            (col in ['cwl', 'cwl_raw', 'cwl_bias_corrected',
                    'water_level', 'waterlevel', 'htp']):
            df.attrs['ColumnMetaData'][col] = {}
        if 'datum' not in df.attrs['ColumnMetaData'][col]:
            storm_datum = get_storm_default_datum(UI_storm.storm.value)
            # Add datum to data frame metadata.
            df.attrs['ColumnMetaData'][col]['datum'] = storm_datum 
    # Get datum conversion values.
    st_metadata = load_station_metadata().get('datums', {}).get('elevations', {})
    # Apply conversions.
    for col in df.columns:
        # By this point, any columns that require conversion should have
        # datum info in their metadata.
        if 'datum' in df.attrs['ColumnMetaData'][col]:
            input_datum = df.attrs['ColumnMetaData'][col]['datum']
            if input_datum != output_datum:
                try:
                    # Conversion uses formula:
                    # height_above_input_datum + input_datum = height_above_output_datum + output_datum
                    input_datum_value = st_metadata.get(input_datum.upper(), np.nan)
                    output_datum_value = st_metadata.get(output_datum.upper(), np.nan)
                    # Note that the NaN values mean that no data will be available
                    # if either of the input datums is not found in the station metadata.
                    datum_conversion = input_datum_value - output_datum_value
                    df[col] = df[col] + datum_conversion
                    # Update datum in data frame metadata.
                    df.attrs['ColumnMetaData'][col]['datum'] = output_datum
                except Exception as e:
                    logger.info(f'Error converting {col} from {input_datum} to {output_datum}: {e}')
    return df
    

# ----------------------------------------------------------------------------
# Functions for data plots.
# ----------------------------------------------------------------------------

def get_map(event):
    try:
        map = get_thalassa_map()
    except:
        map = get_folium_map()
    return map
    

def get_thalassa_map():
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
        storm = UI_storm.storm.value
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
            tm = tm * gv.Polygons(map_regions.loc[[reg]], vdims=['region']).opts(fill_alpha=0.0, line_color=reg_colors[map_regions.loc[reg, 'region']], color=reg_colors[map_regions.loc[reg, 'region']])
            tm = tm * gv.Path(map_regions.loc[[reg]], vdims=['region']).opts(tools=['hover'], color=reg_colors[map_regions.loc[reg, 'region']])
        # Add station markers.
        try:
            vdims = ['station_name', 'station', 'station_id_type']
            for var in ['nos_id', 'nws_id', 'station_type']:
                if var in map_stations:
                    vdims.append(var)
            st_plot = gv.Points(
                map_stations, 
                vdims=vdims
            ).opts(tools=['hover'], size=12)
        except:
            st_plot = gv.Points(
                map_stations,
                vdims=['name', 'nos_id', 'nws_id', 'station_type']
            ).opts(tools=['hover'], size=12)
        # Add station marker interactivity.
        stream = hv.streams.Tap(source=st_plot, x=np.nan, y=np.nan)
        @pn.depends(stream.param.x, stream.param.y)
        def update_station_from_map(x, y):
            matching_stations = map_stations[
                (np.abs(map_stations.latitude - y) <= 0.1) &
                (np.abs(map_stations.longitude - x) <= 0.1)
            ]
            if 'station' in map_stations.columns:
                matching_station_id = matching_stations.station.values
            else:
                matching_station_id = matching_stations.nos_id.values
            if len(matching_station_id) > 0:
                if len(matching_station_id) > 1:
                    logger.debug(f'Finding closest station out of {len(matching_station_id)} ({matching_station_id})')
                    i_min = np.argmin((matching_stations.latitude - y)**2 + 
                                      (matching_stations.longitude - x)**2)
                    logger.debug(f'Index of closest station: {i_min}')
                else:
                    i_min = 0
                formatted_id = 'nos_' + matching_station_id[i_min]
                if formatted_id in UI.station.options:
                    logger.debug(f"Updating station to {formatted_id} based on map click ")
                    UI.station.value = formatted_id
                    logger.debug(f"UI.station.value is now {UI.station.value}")
                    return pn.pane.Str(f"Selected station: {formatted_id}")
                else:
                    logger.debug(f"{formatted_id} is not available.")
                    logger.debug(f"UI.station.value is now {UI.station.value}")
                    return pn.pane.Str(f"{formatted_id} is not available: select another.")
        @pn.depends(UI.station, watch=True)
        def get_UI_station(stn):
            return pn.pane.Str(f"Currently selected: {stn}")
        tm = pn.Column(tm * st_plot, update_station_from_map, get_UI_station)
    except Exception as e:
        print(e)
        tm = thalassa.api.get_tiles().opts(height=500)
    return tm


def get_folium_map():
    try:
        storm = UI_storm.storm.value
        # Load data.
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
        fm = folium.Map(
            location=(centr.y, centr.x), 
            zoom_start=5,
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
    except:
        # Create map.
        fm = folium.Map(
            location=(30.0, -75.0), 
            zoom_start=5,
        )
    return pn.pane.plot.Folium(fm, height=500)


def get_plot_label(plot_type: str) -> str:
    """Get plot axis label with units."""
    if plot_type in ['cwl']:
        return f'water elevation (m, {UI.datum.value})'
    elif plot_type in ['pressure']:
        return 'surface air pressure (hPa)'
    elif plot_type in ['wind']:
        return 'wind speed (m/s)'
    else:
        logger.info(f'No plot label for plot_type {plot_type}.')
        return ''
    
    
def plot_ts(event):
    quantile = UI.quantile.value / 100
    title = f"{UI_storm.storm.value}: {UI.forecast_type.value} at {UI.station.value}"
    ylabel = get_plot_label(UI.plot_type.value)
    try:
        timeseries = []
        # Load obs and model data.
        obs, sims = load_obs_sims(
            UI_storm.storm.value, 
            UI.plot_type.value, 
            UI.forecast_type.value, 
            UI_toggle.models_vars.value, 
            UI.station.value,
            UI.datum.value
        )
        # Plot obs (if available).
        if not obs.empty:
            logger.info("obs len: %r", len(obs))
            logger.info("obs describe:\n%r", obs.describe())
            timeseries += [obs.hvplot(label="obs", color="lightgrey")]
            try:
                obs_threshold = obs.quantile(quantile)
                logger.info("obs quantile: %r", obs_threshold)
                obs_ext = pyextremes.get_extremes(obs, "POT", threshold=obs_threshold, r=f"{UI.window.value}h")
                logger.info("obs ext:\n%r", obs_ext)
                timeseries += [hv.HLine(obs_threshold).opts(color="grey", line_dash="dashed", line_width=1)]
                timeseries += [obs_ext.hvplot.scatter(label="obs extreme")]
            except Exception as e:
                logger.info(f'Error while calculating obs. extremes: {e}')
        # Plot models (if available).
        if sims:
            for i, (model, ts) in enumerate(sims.items()):
                timeseries += [ts.hvplot(label=model).opts(color=cc.glasbey_dark[i])]
        # If obs and/or model available, plot them;
        # otherwise, return a "No Data" message.
        if timeseries:
            # Get flood levels too, if available.
            try:
                flood_levels = get_flood_levels()
                flood_colors = {'minor':'#fd8d3c', 'moderate':'#f03b20', 'major':'#800080'}
                for level, value in flood_levels.items():
                    timeseries += [hv.HLine(value).opts(
                        color=flood_colors.get(level, "red"), 
                        line_dash="dotted", 
                        line_width=1
                    )]
            except Exception as e:
                logger.info(f'Error while getting flood levels: {e}')
            return hv.Overlay(timeseries).opts(show_grid=True, active_tools=["box_zoom"], min_height=300, ylabel=ylabel, title=title)
        else:
            return pn.pane.Str(f'{title}: No {UI.plot_type.value} time series data.')
    except Exception as e:
        logger.info(f'{title}: Error while plotting {UI.plot_type.value} time series: {e}')
        return pn.pane.Str(f'{title}: Error while plotting {UI.plot_type.value} time series.')


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
            UI.station.value,
            UI.datum.value
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


def plot_scatter(event):
    title = f"{UI_storm.storm.value}: {UI.forecast_type.value} at { UI.station.value}"
    try:
        # Load obs and model data.
        obs, sims = load_obs_sims(
            UI_storm.storm.value, 
            UI.plot_type.value, 
            UI.forecast_type.value, 
            UI_toggle.models_vars.value, 
            UI.station.value,
            UI.datum.value
        )
        # Need both obs and model for this plot.
        if (not obs.empty) & (len(sims) > 0):
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
            plot_type_desc = get_plot_label(UI.plot_type.value)
            overlay = overlay.opts(
                min_height=400,
                max_height=500,
                show_grid=True,
                legend_position="right",
                title=title,
                ylabel=f"observed {plot_type_desc}",
                xlabel=f"model {plot_type_desc}",
            )
            return overlay
        else:
            return pn.pane.Str(f'{title}: No {UI.plot_type.value} scatter data.')
    except Exception as e:
        logger.info(f'{title}: Error while plotting {UI.plot_type.value} scatter: {e}')
        return pn.pane.Str(f'{title}: Error while plotting {UI.plot_type.value} scatter.')


def plot_extremes_table(event):
    quantile = UI.quantile.value / 100
    title = f"{UI_storm.storm.value}: {UI.forecast_type.value} at { UI.station.value}"
    try:
        # Load obs and model data.
        obs, sims = load_obs_sims(
            UI_storm.storm.value, 
            UI.plot_type.value, 
            UI.forecast_type.value, 
            UI_toggle.models_vars.value, 
            UI.station.value,
            UI.datum.value
        )
        model_dfs = []
        for i, (model, sim) in enumerate(sims.items()):
            ext_df = match_extremes(sim=sim, obs=obs, 
                                    quantile=quantile, 
                                    cluster=UI.window.value)
            logger.info("\n%r", ext_df)
            model_dfs += [ext_df[["model"]].rename(columns={"model": model})]
        if len(sims) > 0:
            model_dfs.insert(0, ext_df[["observed"]].rename(columns={"observed": "obs"}))
            df = pd.concat(model_dfs, axis=1)
            table = pn.widgets.Tabulator(df.round(3), max_height=400)
            return table
        elif not obs.empty:
            ext = pyextremes.get_extremes(obs, "POT", 
                                          threshold=obs.quantile(quantile), 
                                          r=f"{UI.window.value}h")
            ext_values_dict: dict[str, T.Any] = {}
            ext_values_dict["obs"] = ext.values
            ext_values_dict["time observed"] = ext.index.values
            ext_df = pd.DataFrame(ext_values_dict)
            df = df.set_index("time observed")
            table = pn.widgets.Tabulator(df.round(3), max_height=400)
            return table
        else:
            return pn.pane.Str(f'{title}: No {UI.plot_type.value} extremes data.')
    except Exception as e:
        logger.info(f'{title}: Error while calculating {UI.plot_type.value} extremes: {e}')
        return pn.pane.Str(f'{title}: Error while calculating {UI.plot_type.value} extremes.')
        


# TODO: should we be using the version of this function from seastats.storms ?
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


# ----------------------------------------------------------------------------
# Put together dashboard.
# ----------------------------------------------------------------------------

def get_page():
    on_storm_apply = pn.depends(UI_storm.apply_storm)
    on_plot_apply = pn.depends(UI.plot_apply)
    page = pn.template.MaterialTemplate(
        title="STOFS event analysis",
        sidebar=[
            pn.pane.Str('\nEvent'),
            UI_storm.storm,
            UI_storm.storm_default_datum_display,
            UI_storm.apply_storm,
            pn.pane.Str('\n\nData'),
            UI.plot_type,
            UI.forecast_type,
            UI.station,
            UI_toggle.models_vars,
            UI.datum,
            pn.pane.Str('\n\nStatistics'),
            #UI.metric,
            UI.quantile,
            UI.window,
            UI.plot_apply,
        ],
        sidebar_width=350,
        main=pn.Column(
            pn.Tabs(
                ('Map', on_storm_apply(get_map)),
                ('Time series', on_plot_apply(plot_ts)),
                ('Statistics', on_plot_apply(plot_table_comparison)),
                ('Scatter', on_plot_apply(plot_scatter)),
                ('Extremes', on_plot_apply(plot_extremes_table)),
                tabs_location='left',
                dynamic=True
            )
        ),
    )
    return page


page = get_page()
page.servable()
