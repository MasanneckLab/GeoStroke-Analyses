"""Population coverage tables – national, state, county.

This is a thin wrapper around `geostroke.population.population_within`.
It expects *unioned* polygons for each travel-time band and produces
pandas DataFrames ready to save as Excel or render as a table in a
notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from . import population, data, config

Geometry = Polygon | MultiPolygon

__all__ = ["national_table", "state_table", "county_table"]


def _band_rows(polys: Dict[int, Geometry], region_label: str, time_bins: list[int] | None = None) -> List[dict]:
    """Generate coverage rows for specified time bins.
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS for backward compatibility.
    """
    rows: list[dict] = []
    # Use provided time_bins or default to backward-compatible bins
    coverage_time_bins = time_bins or config.DEFAULT_TIME_BINS
    
    for t in coverage_time_bins:
        geom = polys.get(t)
        pop = population.population_within(geom) if geom is not None else 0
        rows.append(
            {
                "region": region_label,
                "time_min": t,
                "covered_pop": pop,
                "percentage": round(pop / config.POP_TOTAL * 100, 2),
            }
        )
    return rows


def national_table(polys: Dict[int, Geometry], time_bins: list[int] | None = None) -> pd.DataFrame:
    """Return national coverage DataFrame (one row per time band).
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    return pd.DataFrame(_band_rows(polys, "Germany", time_bins))


def state_table(states_gdf: gpd.GeoDataFrame, polys: Dict[int, Geometry], time_bins: list[int] | None = None) -> pd.DataFrame:
    """Coverage per federal state (NAME_1 column expected).
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    # Use provided time_bins or default to backward-compatible bins
    coverage_time_bins = time_bins or config.DEFAULT_TIME_BINS
    
    rows: list[dict] = []
    for _, st in states_gdf.iterrows():
        name = str(st.NAME_1 if "NAME_1" in st else st["name"])
        mask = st.geometry
        sub_polys: Dict[int, Geometry] = {t: (polys[t].intersection(mask) if polys[t] is not None else None) for t in coverage_time_bins}  # type: ignore[assignment]
        rows.extend(_band_rows(sub_polys, name, coverage_time_bins))
    return pd.DataFrame(rows)


def county_table(counties_gdf: gpd.GeoDataFrame, polys: Dict[int, Geometry], time_bins: list[int] | None = None) -> pd.DataFrame:
    """Coverage per county (expects column `county_name`).
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    # Use provided time_bins or default to backward-compatible bins
    coverage_time_bins = time_bins or config.DEFAULT_TIME_BINS
    
    rows: list[dict] = []
    for _, ct in counties_gdf.iterrows():
        name = str(ct.county_name if "county_name" in ct else ct.NAME_2)
        mask = ct.geometry
        sub_polys: Dict[int, Geometry] = {t: (polys[t].intersection(mask) if polys[t] is not None else None) for t in coverage_time_bins}  # type: ignore[assignment]
        rows.extend(_band_rows(sub_polys, name, coverage_time_bins))
    return pd.DataFrame(rows) 