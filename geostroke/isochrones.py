"""Isochrone helpers.

A thin abstraction over *openrouteservice* so that other modules can
obtain polygons for multiple travel-time bands with a single call.  For
reproducibility the module prefers cached pickle files; live API calls
are triggered only if requested and if an ORS server is reachable.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence, Any

import openrouteservice as ors
from shapely.geometry import Polygon

from . import config

__all__ = [
    "get_isochrones",
    "load_cached",
    "cache_polygons",
]

# ----------------------------------------------------------------------------
# Cache utilities
# ----------------------------------------------------------------------------

_PICKLE_TEMPLATE = "poly{minutes}{}.pkl"  # minutes will be formatted


def _cache_path(minutes: int, suffix: str = "") -> Path:
    return config.DATA_DIR / _PICKLE_TEMPLATE.format(minutes=minutes, suffix=suffix)


def cache_polygons(minutes: int, polygons: Sequence[Polygon], *, suffix: str = "") -> Path:
    """Persist *polygons* to pickle under the standard naming convention."""

    p = _cache_path(minutes, suffix)
    with open(p, "wb") as fh:
        pickle.dump(list(polygons), fh, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_cached(minutes: int, *, suffix: str = "") -> list[Polygon]:
    """Load previously cached polygons for *minutes* band.  Raises *FileNotFoundError* if absent."""

    p = _cache_path(minutes, suffix)
    with open(p, "rb") as fh:
        return pickle.load(fh)


# ----------------------------------------------------------------------------
# Live ORS requests
# ----------------------------------------------------------------------------

def _ors_client() -> ors.Client:
    return ors.Client(base_url=config.ORS_BASE_URL, timeout=config.ORS_TIMEOUT)


def _make_request(lon: float, lat: float) -> Any:  # raw JSON response
    cl = _ors_client()
    params = {
        "profile": config.ORS_PROFILE,
        "range": config.ISO_RANGE,
        "interval": config.ISO_INTERVAL,
        "locations": [[lon, lat]],
    }
    return cl.isochrones(**params)  # type: ignore[attr-defined]


def _json_to_polys(resp: Any) -> list[Polygon]:
    """Convert ORS GeoJSON *resp* to a list of Shapely polygons (ascending time)."""

    feats = resp["features"]
    polygons: list[Polygon] = []
    for feat in feats:
        coords = feat["geometry"]["coordinates"][0]
        polygons.append(Polygon([tuple(pt) for pt in coords]))
    return polygons


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def get_isochrones(lon: float, lat: float, *, use_cache: bool = True) -> list[Polygon]:
    """Return list of polygons (15, 30, 45, 60 min) around a point.

    If *use_cache* is True, the function will attempt to read a pickle
    file first.  The file naming format is ``poly15.pkl`` etc.  If the
    cache is missing or loading fails, a live ORS request is issued and
    the result cached immediately.
    """

    # 1. Attempt cache – this assumes polygons are stored individually per band.
    if use_cache:
        try:
            return [_ for _ in (load_cached(m) for m in config.TIME_BINS)]  # type: ignore[misc]
        except FileNotFoundError:
            pass  # fall back to live query

    # 2. Live query
    resp = _make_request(lon, lat)
    polys = _json_to_polys(resp)

    # 3. Persist split bands
    for minutes, poly in zip(config.TIME_BINS, polys, strict=False):
        cache_polygons(minutes, [poly])

    return polys 