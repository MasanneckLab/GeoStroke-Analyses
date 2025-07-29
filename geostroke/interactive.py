"""Interactive Plotly maps for GeoStroke.

Quick wrapper that reproduces the dropdown map (All, Supra-regional,
Thrombectomy, Regional) from the original notebook in a single call.
"""

from __future__ import annotations

from typing import Dict

import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from . import config

__all__ = ["make_condition_dropdown_map"]


TIME_ORDER = ["15 min", "30 min", "45 min", "60 min"]
ORDER_DICT = {"60 min": 0, "45 min": 1, "30 min": 2, "15 min": 3}

_cmap = plt.get_cmap("RdBu_r")
TIME_COLOURS = {
    t: mcolors.to_hex(_cmap(i / (len(TIME_ORDER) - 1))) for i, t in enumerate(TIME_ORDER)
}


def _prep_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return copy of *gdf* with draw_order column for correct stacking."""
    gdf = gdf.copy()
    gdf["draw_order"] = gdf["Time"].map(ORDER_DICT)
    return gdf.sort_values("draw_order")


def make_condition_dropdown_map(
    gdf_by_cond: Dict[str, gpd.GeoDataFrame],
    df_points_by_cond: Dict[str, pd.DataFrame],
    *,
    width: int = 750,
    height: int = 1000,
    mapbox_style: str = "open-street-map",
):
    """Build a Plotly Mapbox figure with dropdown for each condition.

    Parameters
    ----------
    gdf_by_cond : dict[str, GeoDataFrame]
        Keys are condition labels (e.g. "All", "Supraregional")
        Each GeoDataFrame must contain a "Time" categorical column.
    df_points_by_cond : dict[str, DataFrame]
        Facility DataFrames (lon/lat + name) keyed the same as
        *gdf_by_cond*.
    """

    traces = []
    buttons = []
    n_traces_per_cond = None
    cond_order = list(gdf_by_cond.keys())

    for idx, cond in enumerate(cond_order):
        gdf = _prep_gdf(gdf_by_cond[cond])
        geojson = gdf.__geo_interface__

        fig_px = px.choropleth_mapbox(
            gdf,
            geojson=geojson,
            locations=gdf.index,
            color="Time",
            category_orders={"Time": TIME_ORDER},
            color_discrete_map=TIME_COLOURS,
            center={"lat": 51.4, "lon": 10.45},
            mapbox_style=mapbox_style,
            opacity=0.4,
            zoom=5,
            width=width,
            height=height,
        )
        condition_traces = list(fig_px.data)

        # Add point layer
        df_pts = df_points_by_cond[cond]
        condition_traces.append(  # type: ignore[arg-type]
            go.Scattermapbox(
                lon=df_pts["longitude"],
                lat=df_pts["latitude"],
                mode="markers",
                marker=dict(color="rgba(0,0,0,0.8)", size=10),
                text=df_pts.get("name", None),
                hoverinfo="text",
                name=f"{cond} centers",
            )
        )

        if n_traces_per_cond is None:
            n_traces_per_cond = len(condition_traces)

        # Initial visibility – only first condition shown
        for tr in condition_traces:
            tr.visible = idx == 0  # type: ignore[attr-defined]
        traces.extend(condition_traces)

    total_traces = len(traces)
    for i, cond in enumerate(cond_order):
        vis = [False] * total_traces
        if n_traces_per_cond is None:
            continue
        start = i * n_traces_per_cond  # type: ignore[operator]
        for j in range(n_traces_per_cond):
            vis[start + j] = True  # type: ignore[index]
        buttons.append(
            dict(
                label=cond,
                method="update",
                args=[{"visible": vis}, {"title": f"Interactive Map: {cond}"}],
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        mapbox_style=mapbox_style,
        mapbox=dict(center=dict(lat=51.4, lon=10.45), zoom=5),
        updatemenus=[dict(active=0, buttons=buttons, direction="down", x=0.05, y=0.95)],
        margin=dict(r=0, t=40, l=0, b=0),
        title="Interactive Map: All Stroke Units",
    )
    return fig 