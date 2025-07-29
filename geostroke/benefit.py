"""Benefit analysis - comparing CT hospitals vs stroke units.

This module provides functionality to analyze where CT hospitals provide
significant time advantages over stroke units, accounting for optional
penalties and generating maps and population coverage statistics.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import hashlib
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import fiona
import datashader as ds
import datashader.transfer_functions as tf
from matplotlib.colors import ListedColormap
import math
from collections import Counter

from . import config, coverage, data

__all__ = [
    # NOTE: no-stipple patch: public API unchanged, we just stop using stippling
    "scale_markersize",
    "calculate_time_benefits",
    "calculate_time_benefits_parallel", 
    "calculate_time_benefits_chunked",
    "calculate_comprehensive_access_analysis",
    "create_benefit_map",
    "calculate_benefit_coverage", 
    "benefit_analysis_full",
    "generate_benefit_maps_states",
    "generate_benefit_maps_counties",
    "create_four_scenario_comparison",
    "create_four_scenario_comparison_proper",
    "three_speed_scenarios",
    "three_penalty_scenarios",
    "calculate_comprehensive_population_coverage",
    "calculate_scenario_benefit_coverage",
    "datashade_categories",
    "imshow_datashaded",
    "build_fishnet",
    "build_hex_grid",
    "aggregate_points_to_grid",
    "plot_grid_categories",
    "diagnose_emergency_scenario_issue"
]

# ---------------------------------------------------------------------------
# Patch settings (no stipple/overlay wanted)
# ---------------------------------------------------------------------------

# Set a single, solid grey for areas where neither is reachable
SOLID_GREY = "#bdbdbd"
# If you ever want stipples back, toggle this flag
USE_STIPPLE = False

# ---------------------------------------------------------------------------
# Core benefit calculation functions
# ---------------------------------------------------------------------------

def scale_markersize(figsize_current, figsize_ref=(12, 10), markersize_ref=0.5):
    """Scale marker size based on figure size to maintain consistent visual density.
    
    Parameters
    ----------
    figsize_current : tuple
        Current figure size (width, height).
    figsize_ref : tuple, optional
        Reference figure size. Default (12, 10).
    markersize_ref : float, optional
        Reference marker size. Default 0.5.
        
    Returns
    -------
    float
        Scaled marker size.
    """
    # Scale based on area (width*height)
    scale = (figsize_current[0] * figsize_current[1]) / (figsize_ref[0] * figsize_ref[1])
    # We want the *visual* density constant, so if the figure gets smaller, increase markersize
    return markersize_ref / scale

def _load_isochrones(suffix: str, time_bins: Optional[List[int]] = None) -> Dict[int, List[Polygon]]:
    """Load isochrone polygons for a given suffix."""
    if time_bins is None:
        time_bins = config.TIME_BINS
    
    polygons = {}
    for t in time_bins:
        cache_path = config.DATA_DIR / f"poly{t}{suffix}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                polygons[t] = pickle.load(f)
            
            # Diagnostic information for emergency scenarios
            if "_emergency" in suffix or "_bad_traffic" in suffix:
                valid_polys = [p for p in polygons[t] if p is not None and not p.is_empty]
                print(f"   📊 {cache_path.name}: {len(valid_polys)}/{len(polygons[t])} valid polygons")
                if len(valid_polys) == 0:
                    print(f"   ⚠️  WARNING: No valid polygons in {cache_path.name}")
        else:
            print(f"⚠️  Missing isochrone file: {cache_path}")
            polygons[t] = []
    
    return polygons


def _calculate_travel_time_at_point(point: Point, polygons: List[List[Polygon]], time_bins: List[int]) -> float:
    """Calculate minimum travel time to reach a point from any facility.
    
    Returns the minimum time in minutes, or infinity if unreachable.
    """
    for time, poly_list in zip(time_bins, polygons):
        if isinstance(poly_list, list):
            for poly in poly_list:
                if poly.contains(point) or poly.intersects(point):
                    return time
    return float('inf')  # Point not reachable within max time


def calculate_time_benefits(
    ct_suffix: str = "_all_CTs",
    stroke_suffix: str = "",
    time_bins: Optional[List[int]] = None,
    ct_penalty: float = 0.0,
    benefit_threshold: float = 10.0,
    grid_resolution: float = 0.01,  # degrees, roughly 1km at German latitudes
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> gpd.GeoDataFrame:
    """Calculate areas where CT hospitals provide significant time benefits over stroke units.
    
    Parameters
    ----------
    ct_suffix : str, optional
        Suffix for CT hospital isochrone files. Default "_all_CTs".
    stroke_suffix : str, optional  
        Suffix for stroke unit isochrone files. Default "" (no suffix).
    time_bins : List[int], optional
        Time bins to analyze. If None, uses config.TIME_BINS.
    ct_penalty : float, optional
        Additional time penalty for CT hospitals in minutes. Default 0.0.
    benefit_threshold : float, optional
        Minimum time difference (stroke - CT) to consider significant benefit in minutes. Default 10.0.
    grid_resolution : float, optional
        Grid resolution in degrees for analysis. Default 0.01 (~1km).
    bounds : tuple, optional
        Analysis bounds (minx, miny, maxx, maxy). If None, uses Germany bounds.
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with benefit areas and time differences.
    """
    
    if time_bins is None:
        time_bins = config.TIME_BINS
    
    print(f"🔍 Calculating CT vs Stroke benefits (threshold: {benefit_threshold} min, CT penalty: {ct_penalty} min)")
    
    # Load isochrones
    print("📦 Loading isochrone data...")
    ct_polygons = _load_isochrones(ct_suffix, time_bins)
    stroke_polygons = _load_isochrones(stroke_suffix, time_bins)
    
    # Check that we have data
    if not ct_polygons or not stroke_polygons:
        raise ValueError("Missing isochrone data - ensure both CT and stroke isochrones exist")
    
    # Load Germany boundary for bounds if not provided
    if bounds is None:
        germany = data.load_germany_outline()
        bounds_tuple = tuple(germany.total_bounds)  # Convert to tuple
    else:
        bounds_tuple = bounds
    
    minx, miny, maxx, maxy = bounds_tuple
    print(f"📐 Analysis bounds: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")
    
    # Create analysis grid
    print(f"🔬 Creating analysis grid (resolution: {grid_resolution:.4f}°)...")
    x_coords = np.arange(minx, maxx + grid_resolution, grid_resolution)
    y_coords = np.arange(miny, maxy + grid_resolution, grid_resolution)
    
    print(f"   Grid size: {len(x_coords)} x {len(y_coords)} = {len(x_coords) * len(y_coords):,} points")
    
    # Pre-organize polygons by time for efficiency
    ct_polys_by_time = []
    stroke_polys_by_time = []
    
    for t in sorted(time_bins):
        # Apply penalty to CT times by shifting the time bins
        ct_time_with_penalty = t + ct_penalty
        
        # Find the appropriate CT time bin considering penalty
        ct_effective_time = min([tb for tb in time_bins if tb >= ct_time_with_penalty], default=max(time_bins))
        
        ct_polys_by_time.append((ct_effective_time, ct_polygons.get(ct_effective_time, [])))
        stroke_polys_by_time.append((t, stroke_polygons.get(t, [])))
    
    # Calculate benefits
    print("⚡ Analyzing time benefits...")
    benefit_points = []
    total_points = len(x_coords) * len(y_coords)
    processed = 0
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            point = Point(float(x), float(y))
            
            # Calculate minimum travel times
            ct_time = float('inf')
            stroke_time = float('inf')
            
            # Check CT hospitals (with penalty applied)
            for ct_t, ct_polys in ct_polys_by_time:
                for poly in ct_polys:
                    if poly.contains(point) or poly.intersects(point):
                        ct_time = min(ct_time, ct_t)
                        break
                if ct_time < float('inf'):
                    break
            
            # Check stroke units
            for stroke_t, stroke_polys in stroke_polys_by_time:
                for poly in stroke_polys:
                    if poly.contains(point) or poly.intersects(point):
                        stroke_time = min(stroke_time, stroke_t)
                        break
                if stroke_time < float('inf'):
                    break
            
            # Calculate benefit accounting for all coverage scenarios
            if ct_time < float('inf'):
                # CT service reaches this point
                
                if ct_time < float('inf') and stroke_time < float('inf'):
                    # Both services reach - standard comparison
                    time_benefit = stroke_time - ct_time
                elif ct_time < float('inf') and stroke_time == float('inf'):
                    # Only CT reaches - major CT advantage
                    time_benefit = 60 - ct_time  # Use 60 min as reference for unreachable stroke
                
                # Store all scenarios where CT provides coverage
                if abs(time_benefit) >= benefit_threshold or stroke_time == float('inf'):
                    benefit_points.append({
                        'geometry': point,
                        'ct_time': ct_time,
                        'stroke_time': stroke_time if stroke_time < float('inf') else 99,  # 99 = not reachable
                        'time_benefit': time_benefit,
                        'benefit_category': _categorize_benefit_extended(time_benefit, ct_time, stroke_time)
                    })
            
            elif ct_time == float('inf') and stroke_time == float('inf'):
                # Neither service reaches - healthcare desert
                benefit_points.append({
                    'geometry': point,
                    'ct_time': 99,  # 99 = not reachable
                    'stroke_time': 99,  # 99 = not reachable
                    'time_benefit': 0,  # No meaningful comparison
                    'benefit_category': "Neither reachable within 60 min"
                })
            
            processed += 1
            if processed % 50000 == 0:
                progress = processed / total_points * 100
                print(f"   Progress: {progress:.1f}% ({processed:,}/{total_points:,})")
    
    print(f"✅ Found {len(benefit_points):,} points with coverage differences")
    
    if not benefit_points:
        print("⚠️  No significant benefits found with current parameters")
        # Return empty GeoDataFrame with correct structure
        return gpd.GeoDataFrame({
            'geometry': [],
            'ct_time': [],
            'stroke_time': [],
            'time_benefit': [],
            'benefit_category': []
        }, crs="EPSG:4326")
    
    # Create GeoDataFrame
    benefit_gdf = gpd.GeoDataFrame(benefit_points, crs="EPSG:4326")
    
    print(f"📊 Benefit statistics:")
    print(f"   Mean benefit: {benefit_gdf['time_benefit'].mean():.1f} minutes")
    print(f"   Max benefit: {benefit_gdf['time_benefit'].max():.1f} minutes")
    print(f"   Benefit categories: {benefit_gdf['benefit_category'].value_counts().to_dict()}")
    
    return benefit_gdf


def _process_grid_chunk(args: Tuple) -> List[Dict]:
    """Process a chunk of grid points for parallel benefit calculation.
    
    This function processes a subset of grid coordinates and returns benefit points.
    It's designed to be used with multiprocessing.
    """
    (x_coords, y_coords, ct_polys_by_time, stroke_polys_by_time, 
     benefit_threshold, start_x_idx, start_y_idx) = args
    
    benefit_points = []
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            point = Point(float(x), float(y))
            
            # Calculate minimum travel times
            ct_time = float('inf')
            stroke_time = float('inf')
            
            # Check CT hospitals (with penalty already applied in time bins)
            for ct_t, ct_polys in ct_polys_by_time:
                for poly in ct_polys:
                    if poly.contains(point) or poly.intersects(point):
                        ct_time = min(ct_time, ct_t)
                        break
                if ct_time < float('inf'):
                    break
            
            # Check stroke units
            for stroke_t, stroke_polys in stroke_polys_by_time:
                for poly in stroke_polys:
                    if poly.contains(point) or poly.intersects(point):
                        stroke_time = min(stroke_time, stroke_t)
                        break
                if stroke_time < float('inf'):
                    break
            
            # Calculate benefit accounting for all coverage scenarios
            if ct_time < float('inf'):
                # CT service reaches this point
                
                if ct_time < float('inf') and stroke_time < float('inf'):
                    # Both services reach - standard comparison
                    time_benefit = stroke_time - ct_time
                elif ct_time < float('inf') and stroke_time == float('inf'):
                    # Only CT reaches - major CT advantage
                    time_benefit = 60 - ct_time  # Use 60 min as reference for unreachable stroke
                
                # Store all scenarios where CT provides coverage
                if abs(time_benefit) >= benefit_threshold or stroke_time == float('inf'):
                    benefit_points.append({
                        'geometry': point,
                        'ct_time': ct_time,
                        'stroke_time': stroke_time if stroke_time < float('inf') else 99,  # 99 = not reachable
                        'time_benefit': time_benefit,
                        'benefit_category': _categorize_benefit_extended(time_benefit, ct_time, stroke_time)
                    })
            
            elif ct_time == float('inf') and stroke_time == float('inf'):
                # Neither service reaches - healthcare desert
                benefit_points.append({
                    'geometry': point,
                    'ct_time': 99,  # 99 = not reachable
                    'stroke_time': 99,  # 99 = not reachable
                    'time_benefit': 0,  # No meaningful comparison
                    'benefit_category': "Neither reachable within 60 min"
                })
    
    return benefit_points


def calculate_time_benefits_parallel(
    ct_suffix: str = "_all_CTs",
    stroke_suffix: str = "",
    time_bins: Optional[List[int]] = None,
    ct_penalty: float = 0.0,
    benefit_threshold: float = 10.0,
    grid_resolution: float = 0.01,  # degrees, roughly 1km at German latitudes
    bounds: Optional[Tuple[float, float, float, float]] = None,
    max_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    force_recalc: bool = False
) -> gpd.GeoDataFrame:
    """Parallel version of calculate_time_benefits for significant speedup.
    
    This version uses multiprocessing to parallelize the grid point analysis,
    which can provide 4-8x speedup on multi-core systems.
    
    Parameters
    ----------
    ct_suffix : str, optional
        Suffix for CT hospital isochrone files. Default "_all_CTs".
    stroke_suffix : str, optional  
        Suffix for stroke unit isochrone files. Default "" (no suffix).
    time_bins : List[int], optional
        Time bins to analyze. If None, uses config.TIME_BINS.
    ct_penalty : float, optional
        Additional time penalty for CT hospitals in minutes. Default 0.0.
    benefit_threshold : float, optional
        Minimum time difference (stroke - CT) to consider significant benefit in minutes. Default 10.0.
    grid_resolution : float, optional
        Grid resolution in degrees for analysis. Default 0.01 (~1km).
    bounds : tuple, optional
        Analysis bounds (minx, miny, maxx, maxy). If None, uses Germany bounds.
    max_workers : int, optional
        Maximum number of parallel processes. If None, uses CPU count.
    chunk_size : int, optional
        Number of grid rows per chunk. If None, auto-calculated.
    force_recalc : bool, optional
        If True, forces recalculation of isochrones even if cached. Default False.
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with benefit areas and time differences.
    """
    
    if time_bins is None:
        time_bins = config.TIME_BINS
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 10)  # Don't use too many processes
    
    # Create cache directory
    cache_dir = config.DATA_DIR / "benefit_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache key based on parameters
    cache_params = {
        "ct_suffix": ct_suffix,
        "stroke_suffix": stroke_suffix,
        "time_bins": time_bins if time_bins is not None else config.TIME_BINS,
        "ct_penalty": ct_penalty,
        "benefit_threshold": benefit_threshold,
        "grid_resolution": grid_resolution,
        "bounds": bounds
    }
    cache_key = hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()
    cache_file = cache_dir / f"benefit_analysis_{cache_key}.pkl"
    
    # Check for cached results
    if not force_recalc and cache_file.exists():
        try:
            print(f"🔍 Loading cached benefit analysis from {cache_file.name}")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            benefit_gdf = gpd.GeoDataFrame(cached_data["data"], crs="EPSG:4326")
            print(f"✅ Loaded {len(benefit_gdf):,} cached benefit points")
            return benefit_gdf
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}, recalculating...")
    
    print(f"🚀 Calculating CT vs Stroke benefits with {max_workers} processes")
    print(f"   Threshold: {benefit_threshold} min, CT penalty: {ct_penalty} min")
    
    # Load isochrones
    print("📦 Loading isochrone data...")
    ct_polygons = _load_isochrones(ct_suffix, time_bins)
    stroke_polygons = _load_isochrones(stroke_suffix, time_bins)
    
    # Check that we have data
    if not ct_polygons or not stroke_polygons:
        raise ValueError("Missing isochrone data - ensure both CT and stroke isochrones exist")
    
    # Load Germany boundary for bounds if not provided
    if bounds is None:
        germany = data.load_germany_outline()
        bounds_tuple = tuple(germany.total_bounds)
    else:
        bounds_tuple = bounds
    
    minx, miny, maxx, maxy = bounds_tuple
    print(f"📐 Analysis bounds: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")
    
    # Create analysis grid
    print(f"🔬 Creating analysis grid (resolution: {grid_resolution:.4f}°)...")
    x_coords = np.arange(minx, maxx + grid_resolution, grid_resolution)
    y_coords = np.arange(miny, maxy + grid_resolution, grid_resolution)
    
    total_points = len(x_coords) * len(y_coords)
    print(f"   Grid size: {len(x_coords)} x {len(y_coords)} = {total_points:,} points")
    
    # Pre-organize polygons by time for efficiency
    ct_polys_by_time = []
    stroke_polys_by_time = []
    
    for t in sorted(time_bins):
        # Apply penalty to CT times by shifting the time bins
        ct_time_with_penalty = t + ct_penalty
        
        # Find the appropriate CT time bin considering penalty
        ct_effective_time = min([tb for tb in time_bins if tb >= ct_time_with_penalty], default=max(time_bins))
        
        ct_polys_by_time.append((ct_effective_time, ct_polygons.get(ct_effective_time, [])))
        stroke_polys_by_time.append((t, stroke_polygons.get(t, [])))
    
    # Determine chunk size
    if chunk_size is None:
        # Aim for reasonable chunks that balance overhead vs parallelism
        chunk_size = max(10, len(y_coords) // (max_workers * 4))
    
    print(f"⚡ Processing with {max_workers} workers, chunk size: {chunk_size} rows")
    
    # Create chunks
    chunks = []
    for i in range(0, len(y_coords), chunk_size):
        y_chunk = y_coords[i:i + chunk_size]
        chunks.append((
            x_coords, y_chunk, ct_polys_by_time, stroke_polys_by_time,
            benefit_threshold, 0, i
        ))
    
    print(f"   Split into {len(chunks)} chunks")
    
    # Process chunks in parallel
    all_benefit_points = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(_process_grid_chunk, chunk): i 
            for i, chunk in enumerate(chunks)
        }
        
        completed = 0
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_results = future.result()
                all_benefit_points.extend(chunk_results)
                completed += 1
                
                progress = completed / len(chunks) * 100
                print(f"   Chunk {completed}/{len(chunks)} complete ({progress:.1f}%)")
                
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_idx}: {e}")
    
    print(f"✅ Found {len(all_benefit_points):,} points with coverage differences")
    
    if not all_benefit_points:
        print("⚠️  No significant benefits found with current parameters")
        return gpd.GeoDataFrame({
            'geometry': [],
            'ct_time': [],
            'stroke_time': [],
            'time_benefit': [],
            'benefit_category': []
        }, crs="EPSG:4326")
    
    # Create GeoDataFrame
    benefit_gdf = gpd.GeoDataFrame(all_benefit_points, crs="EPSG:4326")
    
    # Cache the results
    try:
        cache_data = {
            "data": benefit_gdf.to_dict('records'),
            "params": cache_params,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"💾 Results cached to {cache_file.name}")
    except Exception as e:
        print(f"⚠️  Error caching results: {e}")
    
    print(f"📊 Benefit statistics:")
    print(f"   Mean benefit: {benefit_gdf['time_benefit'].mean():.1f} minutes")
    print(f"   Max benefit: {benefit_gdf['time_benefit'].max():.1f} minutes")
    print(f"   Benefit categories: {benefit_gdf['benefit_category'].value_counts().to_dict()}")
    
    return benefit_gdf


def calculate_time_benefits_chunked(
    ct_suffix: str = "_all_CTs",
    stroke_suffix: str = "",
    time_bins: Optional[List[int]] = None,
    ct_penalty: float = 0.0,
    benefit_threshold: float = 10.0,
    grid_resolution: float = 0.01,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    chunk_size: int = 1000000  # Process 1M points at a time
) -> gpd.GeoDataFrame:
    """Memory-efficient chunked version for very large grids.
    
    This version processes the grid in smaller chunks to reduce memory usage
    while still being faster than the original sequential version.
    
    Parameters
    ----------
    ct_suffix : str, optional
        Suffix for CT hospital isochrone files. Default "_all_CTs".
    stroke_suffix : str, optional  
        Suffix for stroke unit isochrone files. Default "" (no suffix).
    time_bins : List[int], optional
        Time bins to analyze. If None, uses config.TIME_BINS.
    ct_penalty : float, optional
        Additional time penalty for CT hospitals in minutes. Default 0.0.
    benefit_threshold : float, optional
        Minimum time difference (stroke - CT) to consider significant benefit in minutes. Default 10.0.
    grid_resolution : float, optional
        Grid resolution in degrees for analysis. Default 0.01 (~1km).
    bounds : tuple, optional
        Analysis bounds (minx, miny, maxx, maxy). If None, uses Germany bounds.
    chunk_size : int, optional
        Number of points to process per chunk. Default 1M.
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with benefit areas and time differences.
    """
    
    if time_bins is None:
        time_bins = config.TIME_BINS
    
    print(f"🔄 Calculating CT vs Stroke benefits in chunks (size: {chunk_size:,})")
    print(f"   Threshold: {benefit_threshold} min, CT penalty: {ct_penalty} min")
    
    # Load isochrones
    print("📦 Loading isochrone data...")
    ct_polygons = _load_isochrones(ct_suffix, time_bins)
    stroke_polygons = _load_isochrones(stroke_suffix, time_bins)
    
    if not ct_polygons or not stroke_polygons:
        raise ValueError("Missing isochrone data - ensure both CT and stroke isochrones exist")
    
    # Load Germany boundary for bounds if not provided
    if bounds is None:
        germany = data.load_germany_outline()
        bounds_tuple = tuple(germany.total_bounds)
    else:
        bounds_tuple = bounds
    
    minx, miny, maxx, maxy = bounds_tuple
    
    # Create analysis grid
    x_coords = np.arange(minx, maxx + grid_resolution, grid_resolution)
    y_coords = np.arange(miny, maxy + grid_resolution, grid_resolution)
    
    total_points = len(x_coords) * len(y_coords)
    print(f"   Grid size: {len(x_coords)} x {len(y_coords)} = {total_points:,} points")
    
    # Pre-organize polygons by time
    ct_polys_by_time = []
    stroke_polys_by_time = []
    
    for t in sorted(time_bins):
        ct_time_with_penalty = t + ct_penalty
        ct_effective_time = min([tb for tb in time_bins if tb >= ct_time_with_penalty], default=max(time_bins))
        
        ct_polys_by_time.append((ct_effective_time, ct_polygons.get(ct_effective_time, [])))
        stroke_polys_by_time.append((t, stroke_polygons.get(t, [])))
    
    # Process in chunks
    all_benefit_points = []
    processed = 0
    
    num_chunks = (total_points + chunk_size - 1) // chunk_size
    print(f"⚡ Processing {num_chunks} chunks...")
    
    chunk_num = 0
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            point = Point(float(x), float(y))
            
            # Calculate minimum travel times
            ct_time = float('inf')
            stroke_time = float('inf')
            
            # Check CT hospitals
            for ct_t, ct_polys in ct_polys_by_time:
                for poly in ct_polys:
                    if poly.contains(point) or poly.intersects(point):
                        ct_time = min(ct_time, ct_t)
                        break
                if ct_time < float('inf'):
                    break
            
            # Check stroke units
            for stroke_t, stroke_polys in stroke_polys_by_time:
                for poly in stroke_polys:
                    if poly.contains(point) or poly.intersects(point):
                        stroke_time = min(stroke_time, stroke_t)
                        break
                if stroke_time < float('inf'):
                    break
            
            # Calculate benefit
            if ct_time < float('inf'):
                if stroke_time < float('inf'):
                    # Both services reach
                    time_benefit = stroke_time - ct_time
                else:
                    # Only CT reaches
                    time_benefit = 60 - ct_time
                
                if abs(time_benefit) >= benefit_threshold or stroke_time == float('inf'):
                    all_benefit_points.append({
                        'geometry': point,
                        'ct_time': ct_time,
                        'stroke_time': stroke_time if stroke_time < float('inf') else 99,
                        'time_benefit': time_benefit,
                        'benefit_category': _categorize_benefit_extended(time_benefit, ct_time, stroke_time)
                    })
            elif ct_time == float('inf') and stroke_time == float('inf'):
                 # Neither service reaches - healthcare desert
                 all_benefit_points.append({
                     'geometry': point,
                     'ct_time': 99,
                     'stroke_time': 99,
                     'time_benefit': 0,
                     'benefit_category': "Neither reachable within 60 min"
                 })
            
            processed += 1
            
            # Progress update per chunk
            if processed % chunk_size == 0:
                chunk_num += 1
                progress = processed / total_points * 100
                print(f"   Chunk {chunk_num}/{num_chunks} complete ({progress:.1f}%)")
    
    print(f"✅ Found {len(all_benefit_points):,} points with coverage differences")
    
    if not all_benefit_points:
        return gpd.GeoDataFrame({
            'geometry': [],
            'ct_time': [],
            'stroke_time': [],
            'time_benefit': [],
            'benefit_category': []
        }, crs="EPSG:4326")
    
    # Create GeoDataFrame
    benefit_gdf = gpd.GeoDataFrame(all_benefit_points, crs="EPSG:4326")
    
    print(f"📊 Benefit statistics:")
    print(f"   Mean benefit: {benefit_gdf['time_benefit'].mean():.1f} minutes")
    print(f"   Max benefit: {benefit_gdf['time_benefit'].max():.1f} minutes")
    
    return benefit_gdf


def _categorize_benefit(time_benefit: float) -> str:
    """Categorize time benefit into descriptive categories."""
    if time_benefit >= 30:
        return "High (30+ min)"
    elif time_benefit >= 20:
        return "Medium (20-30 min)"
    elif time_benefit >= 10:
        return "Low (10-20 min)"
    else:
        return "Likely irrelevant (<10 min)"


def _categorize_benefit_extended(time_benefit: float, ct_time: float, stroke_time: float) -> str:
    """Extended categorization that handles coverage gaps and exclusive coverage scenarios."""
    
    # Neither service reaches - healthcare desert
    if ct_time == float('inf') and stroke_time == float('inf'):
        return "Neither reachable within 60 min"
    
    # CT-only coverage (stroke unreachable)
    elif ct_time < float('inf') and stroke_time == float('inf'):
        return "Only CT reachable within 60 min"
    
    # Both services reach - use standard categorization
    else:
        return _categorize_benefit(time_benefit)


def create_benefit_map(
    benefit_gdf: gpd.GeoDataFrame,
    output_path: Optional[Path] = None,
    title: str = "CT Hospital Time Benefits",
    ct_penalty: float = 0.0,
    benefit_threshold: float = 10.0,
    figsize: Tuple[float, float] = (12, 10),
    show_healthcare_deserts: bool = True,
    border_buffer_km: float = 0.0
) -> Path:
    """Create a map showing areas with CT time benefits.
    
    Parameters
    ----------
    benefit_gdf : gpd.GeoDataFrame
        Benefit analysis results from calculate_time_benefits.
    output_path : Path, optional
        Output file path. If None, saves to Graphs directory.
    title : str, optional
        Map title.
    ct_penalty : float, optional
        CT penalty used in analysis (for display in legend).
    benefit_threshold : float, optional
        Benefit threshold used (for display in legend).
    figsize : tuple, optional
        Figure size (width, height).
    show_healthcare_deserts : bool, optional
        Whether to show red "Neither reachable within 60 min" areas. Default True.
    border_buffer_km : float, optional
        Buffer distance in kilometers to apply inward from Germany's borders 
        before filtering. Helps avoid border artifacts. Default 0.0 (no buffer).
        
    Returns
    -------
    Path
        Path to saved figure.
    """
    
    if output_path is None:
        output_path = config.GRAPH_DIR / f"ct_benefit_map_threshold_{benefit_threshold}min_penalty_{ct_penalty}min.png"
    
    print(f"🎨 Creating benefit map: {output_path.name}")
    if border_buffer_km > 0:
        print(f"   Using {border_buffer_km}km inward buffer from borders")
    if not show_healthcare_deserts:
        print("   Healthcare deserts (red areas) will be hidden")
    
    # Load geographic context
    germany = data.load_germany_outline()
    
    # Apply inward buffer if specified
    germany_filtered = germany.copy()
    if border_buffer_km > 0:
        # Convert km to degrees (approximate, works well for Germany's latitude)
        buffer_degrees = border_buffer_km / 111.0  # ~111km per degree
        print(f"   Applying {buffer_degrees:.6f}° inward buffer...")
        # Apply buffer and ensure we maintain GeoDataFrame structure
        buffered_geom = germany_filtered.geometry.buffer(-buffer_degrees)
        germany_filtered = gpd.GeoDataFrame(
            geometry=buffered_geom, 
            crs=germany_filtered.crs,
            index=germany_filtered.index
        )
    
    # Filter benefit data to only include points within (buffered) Germany
    if not benefit_gdf.empty:
        print("🗺️  Filtering benefit data to Germany boundaries...")
        # Ensure same CRS for spatial filtering
        if benefit_gdf.crs != germany_filtered.crs and germany_filtered.crs is not None:
            benefit_gdf_filtered = benefit_gdf.to_crs(germany_filtered.crs)
        else:
            benefit_gdf_filtered = benefit_gdf.copy()
        
        # Use spatial intersection to keep only points within Germany
        # Create a proper GeoDataFrame from germany for sjoin
        germany_geom = gpd.GeoDataFrame(geometry=germany_filtered.geometry, crs=germany_filtered.crs)
        benefit_gdf_filtered = gpd.sjoin(
            benefit_gdf_filtered, 
            germany_geom, 
            how='inner', 
            predicate='within'
        )
        
        # Remove the extra index column added by sjoin
        if 'index_right' in benefit_gdf_filtered.columns:
            benefit_gdf_filtered = benefit_gdf_filtered.drop(columns=['index_right'])
            
        # Optionally remove healthcare desert areas
        if not show_healthcare_deserts:
            before_count = len(benefit_gdf_filtered)
            mask = benefit_gdf_filtered['benefit_category'] != "Neither reachable within 60 min"
            benefit_gdf_filtered = benefit_gdf_filtered[mask].copy()
            removed_count = before_count - len(benefit_gdf_filtered)
            if removed_count > 0:
                print(f"   Removed {removed_count:,} healthcare desert points")
            
        print(f"   Filtered from {len(benefit_gdf):,} to {len(benefit_gdf_filtered):,} points within Germany")
    else:
        benefit_gdf_filtered = benefit_gdf.copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Setup axis without frame (like other maps in the project)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Plot Germany outline (original, not buffered)
    germany.boundary.plot(ax=ax, color='black', linewidth=0.6, zorder=3)
    
    # Define benefit colors (updated for new categorization including coverage scenarios)
    benefit_colors = {
        "High (30+ min)": "#006837",      # Dark green
        "Medium (20-30 min)": "#31a354",  # Medium green  
        "Low (10-20 min)": "#78c679",     # Light green
        "Likely irrelevant (<10 min)": "white",        # No/low benefit
        "Only CT reachable within 60 min": "#41b6c4",  # Teal
        "Neither reachable within 60 min": SOLID_GREY  # Solid grey
    }
    
    # Remove healthcare desert color from legend if not showing them
    if not show_healthcare_deserts:
        benefit_colors = {k: v for k, v in benefit_colors.items() 
                         if k != "Neither reachable within 60 min"}
    
    # Plot benefit areas by category (simple solid fills, no stipple/hatch)
    if not benefit_gdf_filtered.empty:
        for category in benefit_colors.keys():
            category_data = benefit_gdf_filtered[benefit_gdf_filtered['benefit_category'] == category]
            if category_data.empty:
                continue

            category_data.plot(
                ax=ax,
                color=benefit_colors[category],
                markersize=0.5,
                alpha=0.8 if category == "Neither reachable within 60 min" else 0.7,
                edgecolor='none',
                linewidth=0,
                zorder=2,
                rasterized=True
            )
    
    # Set bounds with margins
    ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
    ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend with frames around colors for visibility (only showing categories present in filtered data)
    legend_elements = []
    for cat in benefit_colors.keys():
        if cat in benefit_gdf_filtered['benefit_category'].values or cat == "Likely irrelevant (<10 min)":
            legend_elements.append(
                mpatches.Patch(
                    facecolor=benefit_colors[cat],
                    label=cat,
                    edgecolor='black',
                    linewidth=1.0
                )
            )
    
    # Position legend in Eastern Germany indent area
    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.93, 0.35),
              fontsize=8, frameon=True, fancybox=True, shadow=True, 
              title='Categories', title_fontsize=9, facecolor='white', edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Map saved: {output_path}")
    return output_path


def calculate_benefit_coverage(
    benefit_gdf: gpd.GeoDataFrame,
    region_name: str = "Germany"
) -> pd.DataFrame:
    """Calculate population coverage for areas with CT time benefits.
    
    Parameters
    ----------
    benefit_gdf : gpd.GeoDataFrame
        Benefit analysis results from calculate_time_benefits.
    region_name : str, optional
        Name of the region for the results table.
        
    Returns
    -------
    pd.DataFrame
        Population coverage statistics by benefit category.
    """
    
    if not config.POP_RASTER.exists():
        print(f"⚠️  Population raster not found: {config.POP_RASTER}")
        return pd.DataFrame()
    
    print(f"📊 Calculating population coverage for benefit areas in {region_name}")
    
    if benefit_gdf.empty:
        print("⚠️  No benefit areas to analyze")
        return pd.DataFrame({
            'region': [region_name],
            'benefit_category': ['No benefits found'],
            'covered_pop': [0],
            'percentage': [0.0]
        })
    
    # Load population raster
    with rasterio.open(config.POP_RASTER) as src:
        # Define nodata value for consistent use throughout function
        nodata = src.nodata if src.nodata is not None else -200
        
        # Get total population for percentage calculations
        if region_name == "Germany":
            germany = data.load_germany_outline()
            total_img, _ = mask(src, germany.geometry, crop=True)
            total_arr = total_img[0]
            total_arr[total_arr == nodata] = 0
            total_pop = total_arr.sum()
        else:
            # For scenario analysis, use total German population as baseline
            germany = data.load_germany_outline()
            total_img, _ = mask(src, germany.geometry, crop=True)
            total_arr = total_img[0]
            total_arr[total_arr == nodata] = 0
            total_pop = total_arr.sum()
        
        # Calculate coverage by benefit category
        results = []
        
        for category in benefit_gdf['benefit_category'].unique():
            category_data = benefit_gdf[benefit_gdf['benefit_category'] == category]
            
            if not category_data.empty:
                # Create union of all benefit areas in this category
                geometries = category_data.geometry.values
                if len(geometries) > 0:
                    # Convert to list of geometries, filtering out any invalid ones
                    valid_geometries = [geom for geom in geometries if geom is not None and hasattr(geom, 'is_valid') and geom.is_valid]
                    if valid_geometries:
                        benefit_union = unary_union(valid_geometries)
                    else:
                        benefit_union = None
                else:
                    benefit_union = None
                
                if benefit_union is None or benefit_union.is_empty:
                    covered_pop = 0
                else:
                    # Calculate population in benefit areas
                    try:
                        benefit_img, _ = mask(src, [benefit_union], crop=True)
                        benefit_arr = benefit_img[0]
                        benefit_arr[benefit_arr == nodata] = 0
                        covered_pop = int(benefit_arr.sum())
                    except Exception as e:
                        print(f"⚠️  Error calculating coverage for {category}: {e}")
                        covered_pop = 0
                
                percentage = (covered_pop / total_pop * 100) if total_pop > 0 else 0
                
                results.append({
                    'region': region_name,
                    'benefit_category': category,
                    'covered_pop': covered_pop,
                    'percentage': round(percentage, 2)
                })
    
    coverage_df = pd.DataFrame(results)
    
    if not coverage_df.empty:
        print("📋 Coverage by benefit category:")
        for _, row in coverage_df.iterrows():
            print(f"   {row['benefit_category']}: {row['covered_pop']:,} people ({row['percentage']:.1f}%)")
    
    return coverage_df


def benefit_analysis_full(
    ct_penalty: float = 0.0,
    benefit_threshold: float = 10.0,
    time_bins: Optional[List[int]] = None,
    grid_resolution: float = 0.01,
    create_map: bool = True,
    output_dir: Optional[Path] = None,
    use_parallel: bool = True,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """Perform complete benefit analysis with mapping and population coverage.
    
    Parameters
    ----------
    ct_penalty : float, optional
        Additional time penalty for CT hospitals in minutes.
    benefit_threshold : float, optional
        Minimum time difference to consider significant benefit in minutes.
    time_bins : List[int], optional
        Time bins to analyze. If None, uses config.TIME_BINS.
    grid_resolution : float, optional
        Grid resolution for analysis in degrees.
    create_map : bool, optional
        Whether to create benefit map.
    output_dir : Path, optional
        Output directory for results.
    use_parallel : bool, optional
        Whether to use parallel processing for benefit calculation. Default True.
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses CPU count.
        
    Returns
    -------
    dict
        Results containing benefit_gdf, coverage_df, and map_path (if created).
    """
    
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    
    print(f"🔬 Starting full benefit analysis")
    print(f"   CT penalty: +{ct_penalty} min")
    print(f"   Benefit threshold: ≥{benefit_threshold} min") 
    print(f"   Grid resolution: {grid_resolution}°")
    print(f"   Parallel processing: {'enabled' if use_parallel else 'disabled'}")
    
    # Calculate benefits
    if use_parallel:
        benefit_gdf = calculate_time_benefits_parallel(
            ct_penalty=ct_penalty,
            benefit_threshold=benefit_threshold,
            time_bins=time_bins,
            grid_resolution=grid_resolution,
            max_workers=max_workers
        )
    else:
        benefit_gdf = calculate_time_benefits(
            ct_penalty=ct_penalty,
            benefit_threshold=benefit_threshold,
            time_bins=time_bins,
            grid_resolution=grid_resolution
        )
    
    # Calculate population coverage
    coverage_df = calculate_benefit_coverage(benefit_gdf)
    
    # Save coverage results
    coverage_path = output_dir / f"ct_benefit_coverage_threshold_{benefit_threshold}min_penalty_{ct_penalty}min.xlsx"
    coverage_df.to_excel(coverage_path, index=False)
    print(f"💾 Coverage results saved: {coverage_path}")
    
    results = {
        'benefit_gdf': benefit_gdf,
        'coverage_df': coverage_df,
        'coverage_path': coverage_path
    }
    
    # Create map if requested
    if create_map:
        map_path = create_benefit_map(
            benefit_gdf,
            ct_penalty=ct_penalty,
            benefit_threshold=benefit_threshold
        )
        results['map_path'] = map_path
    
    return results


def generate_benefit_maps_states(
    benefit_gdf: gpd.GeoDataFrame,
    ct_penalty: float = 0.0,
    benefit_threshold: float = 10.0,
    output_dir: Optional[Path] = None
) -> Path:
    """Generate state-level benefit maps in a multi-page PDF.
    
    Parameters
    ----------
    benefit_gdf : gpd.GeoDataFrame
        Benefit analysis results.
    ct_penalty : float, optional
        CT penalty used in analysis.
    benefit_threshold : float, optional
        Benefit threshold used.
    output_dir : Path, optional
        Output directory.
        
    Returns
    -------
    Path
        Path to the generated PDF.
    """
    
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "States"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / f"State_CT_Benefits_threshold_{benefit_threshold}min_penalty_{ct_penalty}min.pdf"
    
    print(f"📚 Generating state-level benefit maps: {pdf_path.name}")
    
    # Load states
    states = data.load_states()
    
    # Benefit colors (including coverage scenarios)
    benefit_colors = {
        "High (30+ min)": "#006837",
        "Medium (20-30 min)": "#31a354", 
        "Low (10-20 min)": "#78c679",
        "Likely irrelevant (<10 min)": "white",  # White - includes both <10min and no significant benefit
        "Only CT reachable within 60 min": "#41b6c4",
        "Neither reachable within 60 min": "#d73027"
    }
    
    with PdfPages(pdf_path) as pdf:
        for _, state in states.iterrows():
            state_name = state.get('NAME_1', 'Unknown State')
            
            # Clip benefit data to state
            state_gdf = gpd.GeoDataFrame([state], crs=states.crs)
            if not benefit_gdf.empty and states.crs is not None:
                benefit_in_state = gpd.clip(benefit_gdf.to_crs(states.crs), state_gdf)
            else:
                benefit_in_state = gpd.GeoDataFrame()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
            
            # Plot state boundary
            state_gdf.boundary.plot(ax=ax, color='black', linewidth=1.0, zorder=3)
            
            # Plot benefit areas
            if not benefit_in_state.empty:
                for category in benefit_colors.keys():
                    category_data = benefit_in_state[benefit_in_state['benefit_category'] == category]
                    if not category_data.empty:
                        category_data.plot(
                            ax=ax,
                            color=benefit_colors[category],
                            markersize=1.0,
                            alpha=0.8,
                            zorder=2
                        )
            
            # Set bounds
            bounds = state_gdf.total_bounds
            margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.05
            ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
            ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
            
            # Styling
            ax.set_facecolor('#f0f8ff')
            ax.set_title(f"CT Time Benefits - {state_name}", fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Legend
            legend_elements = [
                mpatches.Patch(facecolor=benefit_colors[cat], label=cat, edgecolor='black', linewidth=1.0)
                for cat in benefit_colors.keys()
            ]
            ax.legend(handles=legend_elements, loc='best', fontsize=9)
            
            # Statistics
            if not benefit_in_state.empty:
                stats_text = f"Benefit points: {len(benefit_in_state):,}"
                ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"✅ State maps saved: {pdf_path}")
    return pdf_path


def generate_benefit_maps_counties(
    benefit_gdf: gpd.GeoDataFrame,
    ct_penalty: float = 0.0,
    benefit_threshold: float = 10.0,
    output_dir: Optional[Path] = None,
    max_counties: Optional[int] = None
) -> Path:
    """Generate county-level benefit maps in a multi-page PDF.
    
    Parameters
    ----------
    benefit_gdf : gpd.GeoDataFrame
        Benefit analysis results.
    ct_penalty : float, optional
        CT penalty used in analysis.
    benefit_threshold : float, optional
        Benefit threshold used.
    output_dir : Path, optional
        Output directory.
    max_counties : int, optional
        Maximum number of counties to process (for testing).
        
    Returns
    -------
    Path
        Path to the generated PDF.
    """
    
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "Counties"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / f"County_CT_Benefits_threshold_{benefit_threshold}min_penalty_{ct_penalty}min.pdf"
    
    print(f"📚 Generating county-level benefit maps: {pdf_path.name}")
    
    # Load counties
    counties = data.load_counties()
    
    if max_counties:
        counties = counties.head(max_counties)
        print(f"   Limited to first {max_counties} counties for testing")
    
    # Benefit colors (including coverage scenarios)
    benefit_colors = {
        "High (30+ min)": "#006837",
        "Medium (20-30 min)": "#31a354",
        "Low (10-20 min)": "#78c679", 
        "Likely irrelevant (<10 min)": "white",  # White - includes both <10min and no significant benefit
        "Only CT reachable within 60 min": "#41b6c4",
        "Neither reachable within 60 min": "#d73027"
    }
    
    with PdfPages(pdf_path) as pdf:
        for idx, (_, county) in enumerate(counties.iterrows()):
            county_name = county.get('NAME_2', county.get('krs_name', 'Unknown County'))
            state_name = county.get('NAME_1', county.get('lan_name', 'Unknown State'))
            
            # Clip benefit data to county
            county_gdf = gpd.GeoDataFrame([county], crs=counties.crs)
            if not benefit_gdf.empty and counties.crs is not None:
                benefit_in_county = gpd.clip(benefit_gdf.to_crs(counties.crs), county_gdf)
            else:
                benefit_in_county = gpd.GeoDataFrame()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            
            # Plot county boundary
            county_gdf.boundary.plot(ax=ax, color='black', linewidth=1.0, zorder=3)
            
            # Plot benefit areas
            if not benefit_in_county.empty:
                for category in benefit_colors.keys():
                    category_data = benefit_in_county[benefit_in_county['benefit_category'] == category]
                    if not category_data.empty:
                        category_data.plot(
                            ax=ax,
                            color=benefit_colors[category],
                            markersize=0.8,
                            alpha=0.8,
                            zorder=2
                        )
            
            # Set bounds
            bounds = county_gdf.total_bounds
            margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
            ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
            ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
            
            # Styling
            ax.set_facecolor('#f0f8ff')
            ax.set_title(f"CT Time Benefits - {county_name}\n({state_name})", 
                        fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Compact legend
            if idx == 0:  # Only show legend on first page
                legend_elements = [
                    mpatches.Patch(facecolor=benefit_colors[cat], label=cat, edgecolor='black', linewidth=1.0)
                    for cat in benefit_colors.keys()
                ]
                legend_elements.append(mpatches.Patch(facecolor='#f0f8ff', label='No benefit', 
                                                    edgecolor='black', linewidth=1.0))
                ax.legend(handles=legend_elements, loc='best', fontsize=8)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"   Processed {idx + 1}/{len(counties)} counties...")
    
    print(f"✅ County maps saved: {pdf_path}")
    return pdf_path 


def create_four_scenario_comparison(
    benefit_threshold: float = 10.0,
    ct_penalty: float = 5.0,
    grid_resolution: float = 0.015,
    time_bins: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (16, 12)
) -> Path:
    """Create four-panel comparison map for key scenarios.
    
    Compares CT benefits across four scenarios:
    1. Normal driving conditions
    2. Emergency conditions (+20% speed)
    3. Traffic conditions (-20% speed)  
    4. Normal + penalty (additional CT processing time)
    
    Parameters
    ----------
    benefit_threshold : float, optional
        Minimum time difference to consider significant benefit. Default 10.0.
    ct_penalty : float, optional
        Additional time penalty for CT hospitals in Normal+penalty scenario. Default 5.0.
    grid_resolution : float, optional
        Grid resolution for analysis in degrees. Default 0.015.
    time_bins : List[int], optional
        Time bins to analyze. If None, uses [15, 30, 45, 60].
    output_path : Path, optional
        Output file path. If None, saves to Graphs directory.
    figsize : tuple, optional
        Figure size (width, height). Default (16, 12).
        
    Returns
    -------
    Path
        Path to saved comparison figure.
    """
    
    if time_bins is None:
        time_bins = [15, 30, 45, 60]  # Display time bins for faster processing
    
    if output_path is None:
        output_path = config.GRAPH_DIR / f"ct_benefit_four_scenario_comparison_threshold_{benefit_threshold}min.png"
    
    print(f"🎨 Creating four-scenario comparison (threshold: {benefit_threshold} min)")
    
    # Define scenarios
    scenarios = {
        'Normal': {'suffix': '', 'penalty': 0, 'description': 'Standard driving'},
        'Emergency': {'suffix': '_emergency', 'penalty': 0, 'description': 'Emergency (+20% speed)'},
        'Traffic': {'suffix': '_bad_traffic', 'penalty': 0, 'description': 'Bad traffic (-20% speed)'},
        f'Normal +{ct_penalty}min': {'suffix': '', 'penalty': ct_penalty, 'description': f'Standard + {ct_penalty}min penalty'}
    }
    
    # Calculate benefits for each scenario
    scenario_gdfs = {}
    for scenario_name, scenario_config in scenarios.items():
        try:
            print(f"   🔍 Calculating {scenario_name}...")
            benefit_gdf = calculate_time_benefits(
                ct_suffix=f"_all_CTs{scenario_config['suffix']}",
                stroke_suffix=scenario_config['suffix'],
                ct_penalty=scenario_config['penalty'],
                benefit_threshold=benefit_threshold,
                grid_resolution=grid_resolution,
                time_bins=time_bins
            )
            scenario_gdfs[scenario_name] = benefit_gdf
            print(f"     Found {len(benefit_gdf):,} benefit areas")
            
        except Exception as e:
            print(f"     ⚠️ Error: {e}")
            scenario_gdfs[scenario_name] = gpd.GeoDataFrame()
    
    if not scenario_gdfs or all(gdf.empty for gdf in scenario_gdfs.values()):
        print("⚠️  No benefit data to visualize")
        return output_path
    
    # Load Germany outline
    germany = data.load_germany_outline()
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=200)
    axes = axes.flatten()
    
    # Benefit colors (including coverage scenarios)
    benefit_colors = {
        "High (30+ min)": "#006837",     # Dark green
        "Medium (20-30 min)": "#31a354", # Medium green  
        "Low (10-20 min)": "#78c679",    # Light green
        "Likely irrelevant (<10 min)": "white",  # White - includes both <10min and no significant benefit
        "Only CT reachable within 60 min": "#41b6c4",  # Teal - harmonizes with greens
        "Neither reachable within 60 min": "#d73027"  # Red - healthcare desert
    }
    
    # Plot each scenario
    for i, (scenario_name, benefit_gdf) in enumerate(list(scenario_gdfs.items())[:4]):
        ax = axes[i]
        
        # Plot Germany outline
        germany.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=3)
        
        # Plot benefit areas
        if not benefit_gdf.empty:
            for category in benefit_colors.keys():
                category_data = benefit_gdf[benefit_gdf['benefit_category'] == category]
                if not category_data.empty:
                    category_data.plot(
                        ax=ax,
                        color=benefit_colors[category],
                        markersize=0.3,
                        alpha=0.7,
                        zorder=2
                    )
        
        # Styling
        ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
        ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)
        ax.set_facecolor('#f0f8ff')  # Light blue background
        ax.set_title(f"{scenario_name}\n({len(benefit_gdf):,} benefit areas)", 
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide empty subplots if less than 4 scenarios
    for i in range(len(scenario_gdfs), 4):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'CT Hospital Time Benefits: Scenario Comparison\n(Areas with ≥{benefit_threshold} minute advantage over stroke units)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Add legend with frames around colors for visibility
    legend_elements = [
        mpatches.Patch(facecolor=benefit_colors[cat], label=cat, edgecolor='black', linewidth=1.0)
        for cat in benefit_colors.keys()
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=10, 
              frameon=True, fancybox=True, shadow=True, facecolor='white', edgecolor='black')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.9)
    
    # Save the comparison figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Four-scenario comparison saved: {output_path}")
    return output_path 


def _process_comprehensive_grid_chunk(args: Tuple) -> List[Dict]:
    """Process a chunk of grid points for comprehensive access analysis.
    
    Unlike the benefit-focused version, this includes ALL coverage scenarios.
    """
    (x_coords, y_coords, ct_polys_by_time, stroke_polys_by_time, start_x_idx, start_y_idx) = args
    
    all_points = []
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            point = Point(float(x), float(y))
            
            # Calculate minimum travel times
            ct_time = float('inf')
            stroke_time = float('inf')
            
            # Check CT hospitals
            for ct_t, ct_polys in ct_polys_by_time:
                for poly in ct_polys:
                    if poly.contains(point) or poly.intersects(point):
                        ct_time = min(ct_time, ct_t)
                        break
                if ct_time < float('inf'):
                    break
            
            # Check stroke units
            for stroke_t, stroke_polys in stroke_polys_by_time:
                for poly in stroke_polys:
                    if poly.contains(point) or poly.intersects(point):
                        stroke_time = min(stroke_time, stroke_t)
                        break
                if stroke_time < float('inf'):
                    break
            
            # Include ALL scenarios where at least one service is reachable
            if ct_time < float('inf') or stroke_time < float('inf'):
                
                # Calculate time benefit
                if ct_time < float('inf') and stroke_time < float('inf'):
                    # Both services reach
                    time_benefit = stroke_time - ct_time
                elif ct_time < float('inf') and stroke_time == float('inf'):
                    # Only CT reaches
                    time_benefit = 60 - ct_time
                elif ct_time == float('inf') and stroke_time < float('inf'):
                    # Only stroke reaches  
                    time_benefit = -(stroke_time - 60)  # Negative indicates stroke advantage
                else:
                    time_benefit = 0  # Should not happen in this branch
                
                all_points.append({
                    'geometry': point,
                    'ct_time': ct_time if ct_time < float('inf') else 99,
                    'stroke_time': stroke_time if stroke_time < float('inf') else 99,
                    'time_benefit': time_benefit,
                    'access_category': _categorize_access_comprehensive(time_benefit, ct_time, stroke_time)
                })
            
            # Also include healthcare deserts for completeness
            elif ct_time == float('inf') and stroke_time == float('inf'):
                all_points.append({
                    'geometry': point,
                    'ct_time': 99,
                    'stroke_time': 99,
                    'time_benefit': 0,
                    'access_category': "Neither reachable within 60 min"
                })
    
    return all_points


def calculate_comprehensive_access_analysis(
    ct_suffix: str = "_all_CTs",
    stroke_suffix: str = "",
    time_bins: Optional[List[int]] = None,
    ct_penalty: float = 0.0,
    grid_resolution: float = 0.01,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    max_workers: Optional[int] = None,
    force_recalc: bool = False
) -> gpd.GeoDataFrame:
    """Calculate comprehensive healthcare access including ALL coverage scenarios.
    
    Unlike calculate_time_benefits(), this function includes ALL areas where at least one
    service is reachable, providing a complete picture for urban/rural analysis.
    
    Includes:
    - Areas with significant CT benefits
    - Areas with minimal differences between services  
    - Areas where only CT hospitals are reachable
    - Areas where only stroke units are reachable
    - Healthcare deserts (neither reachable)
    
    Parameters
    ----------
    ct_suffix : str, optional
        Suffix for CT hospital isochrone files. Default "_all_CTs".
    stroke_suffix : str, optional  
        Suffix for stroke unit isochrone files. Default "" (no suffix).
    time_bins : List[int], optional
        Time bins to analyze. If None, uses config.TIME_BINS.
    ct_penalty : float, optional
        Additional time penalty for CT hospitals in minutes. Default 0.0.
    grid_resolution : float, optional
        Grid resolution in degrees for analysis. Default 0.01 (~1km).
    bounds : tuple, optional
        Analysis bounds (minx, miny, maxx, maxy). If None, uses Germany bounds.
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses CPU count.
    force_recalc : bool, optional
        Force recalculation even if cached results exist.
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with ALL coverage scenarios including minimal differences.
    """
    
    if time_bins is None:
        time_bins = config.TIME_BINS
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 10)
    
    # Create cache directory
    cache_dir = config.DATA_DIR / "comprehensive_access_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache key
    cache_params = {
        "ct_suffix": ct_suffix,
        "stroke_suffix": stroke_suffix,
        "time_bins": time_bins,
        "ct_penalty": ct_penalty,
        "grid_resolution": grid_resolution,
        "bounds": bounds
    }
    cache_key = hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()
    cache_file = cache_dir / f"comprehensive_access_{cache_key}.pkl"
    
    # Check for cached results
    if not force_recalc and cache_file.exists():
        try:
            print(f"🔍 Loading cached comprehensive access analysis from {cache_file.name}")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            access_gdf = gpd.GeoDataFrame(cached_data["data"], crs="EPSG:4326")
            print(f"✅ Loaded {len(access_gdf):,} cached access points")
            return access_gdf
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}, recalculating...")
    
    print(f"🌍 Calculating COMPREHENSIVE healthcare access analysis")
    print(f"   Includes ALL areas with any service coverage")
    print(f"   CT penalty: +{ct_penalty} min")
    print(f"   Using {max_workers} parallel workers")
    
    # Load isochrones
    print("📦 Loading isochrone data...")
    ct_polygons = _load_isochrones(ct_suffix, time_bins)
    stroke_polygons = _load_isochrones(stroke_suffix, time_bins)
    
    if not ct_polygons or not stroke_polygons:
        raise ValueError("Missing isochrone data - ensure both CT and stroke isochrones exist")
    
    # Load Germany boundary for bounds if not provided
    if bounds is None:
        germany = data.load_germany_outline()
        bounds_tuple = tuple(germany.total_bounds)
    else:
        bounds_tuple = bounds
    
    minx, miny, maxx, maxy = bounds_tuple
    print(f"📐 Analysis bounds: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")
    
    # Create analysis grid
    print(f"🔬 Creating analysis grid (resolution: {grid_resolution:.4f}°)...")
    x_coords = np.arange(minx, maxx + grid_resolution, grid_resolution)
    y_coords = np.arange(miny, maxy + grid_resolution, grid_resolution)
    
    total_points = len(x_coords) * len(y_coords)
    print(f"   Grid size: {len(x_coords)} x {len(y_coords)} = {total_points:,} points")
    
    # Pre-organize polygons by time
    ct_polys_by_time = []
    stroke_polys_by_time = []
    
    for t in sorted(time_bins):
        ct_time_with_penalty = t + ct_penalty
        ct_effective_time = min([tb for tb in time_bins if tb >= ct_time_with_penalty], default=max(time_bins))
        
        ct_polys_by_time.append((ct_effective_time, ct_polygons.get(ct_effective_time, [])))
        stroke_polys_by_time.append((t, stroke_polygons.get(t, [])))
    
    # Determine chunk size
    chunk_size = max(10, len(y_coords) // (max_workers * 4))
    print(f"⚡ Processing with {max_workers} workers, chunk size: {chunk_size} rows")
    
    # Create chunks
    chunks = []
    for i in range(0, len(y_coords), chunk_size):
        y_chunk = y_coords[i:i + chunk_size]
        chunks.append((
            x_coords, y_chunk, ct_polys_by_time, stroke_polys_by_time, 0, i
        ))
    
    print(f"   Split into {len(chunks)} chunks")
    
    # Process chunks in parallel
    all_access_points = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(_process_comprehensive_grid_chunk, chunk): i 
            for i, chunk in enumerate(chunks)
        }
        
        completed = 0
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_results = future.result()
                all_access_points.extend(chunk_results)
                completed += 1
                
                progress = completed / len(chunks) * 100
                print(f"   Chunk {completed}/{len(chunks)} complete ({progress:.1f}%)")
                
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_idx}: {e}")
    
    print(f"✅ Found {len(all_access_points):,} points with coverage")
    
    if not all_access_points:
        return gpd.GeoDataFrame({
            'geometry': [],
            'ct_time': [],
            'stroke_time': [],
            'time_benefit': [],
            'access_category': []
        }, crs="EPSG:4326")
    
    # Create GeoDataFrame
    access_gdf = gpd.GeoDataFrame(all_access_points, crs="EPSG:4326")
    
    # Cache the results
    try:
        cache_data = {
            "data": access_gdf.to_dict('records'),
            "params": cache_params,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"💾 Results cached to {cache_file.name}")
    except Exception as e:
        print(f"⚠️  Error caching results: {e}")
    
    print(f"📊 Comprehensive access statistics:")
    print(f"   Access categories: {access_gdf['access_category'].value_counts().to_dict()}")
    
    return access_gdf


def _categorize_access_comprehensive(time_benefit: float, ct_time: float, stroke_time: float) -> str:
    """Comprehensive categorization focusing on CT benefits and equivalent access."""
    
    # Neither service reaches - healthcare desert
    if ct_time == float('inf') and stroke_time == float('inf'):
        return "Neither reachable within 60 min"
    
    # CT-only coverage (stroke unreachable)
    elif ct_time < float('inf') and stroke_time == float('inf'):
        return "Only CT reachable within 60 min"
    
    # Stroke-only coverage (CT unreachable) 
    elif ct_time == float('inf') and stroke_time < float('inf'):
        return "Only stroke reachable within 60 min"
    
    # Both services reach - focus on CT benefits and equivalent access
    else:
        if time_benefit >= 30:
            return "High CT benefit (30+ min)"
        elif time_benefit >= 20:
            return "Medium CT benefit (20-30 min)"  
        elif time_benefit >= 10:
            return "Low CT benefit (10-20 min)"
        elif time_benefit >= 5:
            return "Minimal CT benefit (5-10 min)"
        else:
            # Group all small differences as equivalent access
            # This includes small stroke advantages which aren't clinically meaningful
            return "Equivalent access (<5 min difference)" 


def create_four_scenario_comparison_proper(
    benefit_threshold: float = 10.0,
    ct_penalty: float = 5.0,
    grid_resolution: float = 0.015,
    time_bins: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (16, 12),
    use_parallel: bool = True,
    max_workers: Optional[int] = None
) -> Path:
    """Create publication-ready four-panel comparison using proper plotting infrastructure.
    
    This function builds on the existing create_benefit_map infrastructure to ensure
    consistent styling and proper geographic filtering.
    
    Compares CT benefits across four scenarios:
    1. Normal driving conditions
    2. Emergency conditions (+20% speed)
    3. Traffic conditions (-20% speed)  
    4. Normal + penalty (additional CT processing time)
    
    Parameters
    ----------
    benefit_threshold : float, optional
        Minimum time difference to consider significant benefit. Default 10.0.
    ct_penalty : float, optional
        Additional time penalty for CT hospitals in Normal+penalty scenario. Default 5.0.
    grid_resolution : float, optional
        Grid resolution for analysis in degrees. Default 0.015.
    time_bins : List[int], optional
        Time bins to analyze. If None, uses [15, 30, 45, 60].
    output_path : Path, optional
        Output file path. If None, saves to Graphs directory.
    figsize : tuple, optional
        Figure size (width, height). Default (16, 12).
    use_parallel : bool, optional
        Whether to use parallel processing. Default True.
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses CPU count.
        
    Returns
    -------
    Path
        Path to saved comparison figure.
    """
    
    if time_bins is None:
        time_bins = config.TIME_BINS  # Use full time bins to support penalty calculations
    
    if output_path is None:
        output_path = config.GRAPH_DIR / f"ct_benefit_four_scenario_comparison_threshold_{benefit_threshold}min.png"
    
    print(f"🎨 Creating publication-ready four-scenario comparison")
    print(f"   Threshold: {benefit_threshold} min, CT penalty: {ct_penalty} min")
    
    # First, check for data quality issues
    print(f"\n🔍 Checking scenario data quality...")
    data_ok = diagnose_emergency_scenario_issue()
    if not data_ok:
        print(f"\n⚠️  WARNING: Data quality issues detected!")
        print(f"   The generated maps may show artifacts (large teal areas)")
        print(f"   Consider fixing the issues before proceeding")
        print(f"   Continuing anyway...\n")
    else:
        print(f"✅ All scenario data looks good!\n")
    
    # Define scenarios
    scenarios = {
        'Normal': {'suffix': '', 'penalty': 0, 'description': 'Standard driving'},
        '+20% Driving Speed (for Emergency)': {'suffix': '_emergency', 'penalty': 0, 'description': 'Emergency (+20% speed)'},
        '-20% Driving Speed for Traffic': {'suffix': '_bad_traffic', 'penalty': 0, 'description': 'Bad traffic (-20% speed)'},
        '5 min tele-setup penalty': {'suffix': '', 'penalty': ct_penalty, 'description': f'Standard + {ct_penalty}min penalty'}
    }
    
    # Calculate benefits for each scenario using the proper function
    scenario_gdfs = {}
    for scenario_name, scenario_config in scenarios.items():
        try:
            print(f"   🔍 Calculating {scenario_name}...")
            
            if use_parallel:
                benefit_gdf = calculate_time_benefits_parallel(
                    ct_suffix=f"_all_CTs{scenario_config['suffix']}",
                    stroke_suffix=scenario_config['suffix'],
                    ct_penalty=scenario_config['penalty'],
                    benefit_threshold=benefit_threshold,
                    grid_resolution=grid_resolution,
                    time_bins=time_bins,
                    max_workers=max_workers
                )
            else:
                benefit_gdf = calculate_time_benefits(
                    ct_suffix=f"_all_CTs{scenario_config['suffix']}",
                    stroke_suffix=scenario_config['suffix'],
                    ct_penalty=scenario_config['penalty'],
                    benefit_threshold=benefit_threshold,
                    grid_resolution=grid_resolution,
                    time_bins=time_bins
                )
            
            scenario_gdfs[scenario_name] = benefit_gdf
            print(f"     Found {len(benefit_gdf):,} benefit areas")
            
        except Exception as e:
            print(f"     ⚠️ Error: {e}")
            scenario_gdfs[scenario_name] = gpd.GeoDataFrame()
    
    if not scenario_gdfs or all(gdf.empty for gdf in scenario_gdfs.values()):
        print("⚠️  No benefit data to visualize")
        return output_path
    
    # Load Germany outline
    germany = data.load_germany_outline()
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=200)
    axes = axes.flatten()
    
    # Calculate scaled marker size for consistent visual density
    ms = scale_markersize(figsize_current=figsize, figsize_ref=(12, 10), markersize_ref=0.5)
    
    # Use the same colors and styling as create_benefit_map
    benefit_colors = {
        "High (30+ min)": "#006837",     # Dark green
        "Medium (20-30 min)": "#31a354", # Medium green  
        "Low (10-20 min)": "#78c679",    # Light green
        "Likely irrelevant (<10 min)": "white",  # White - includes both <10min and no significant benefit
        "Only CT reachable within 60 min": "#41b6c4",  # Teal - harmonizes with greens
        "Neither reachable within 60 min": SOLID_GREY  # Solid grey
    }
    
    # Plot each scenario using the same logic as create_benefit_map
    panel_letters = ['a)', 'b)', 'c)', 'd)']  # Lowercase letters with parentheses for panels
    for i, (scenario_name, benefit_gdf) in enumerate(list(scenario_gdfs.items())[:4]):
        ax = axes[i]
        
        # Apply the same geographic filtering as create_benefit_map
        if not benefit_gdf.empty:
            print(f"   🗺️  Filtering {scenario_name} data to Germany boundaries...")
            # Filter benefit data to only include points within Germany (same as create_benefit_map)
            if benefit_gdf.crs != germany.crs and germany.crs is not None:
                benefit_gdf_filtered = benefit_gdf.to_crs(germany.crs)
            else:
                benefit_gdf_filtered = benefit_gdf.copy()
            
            # Use spatial intersection to keep only points within Germany
            germany_geom = gpd.GeoDataFrame(geometry=germany.geometry, crs=germany.crs)
            benefit_gdf_filtered = gpd.sjoin(
                benefit_gdf_filtered, 
                germany_geom, 
                how='inner', 
                predicate='within'
            )
            
            # Remove the extra index column added by sjoin
            if 'index_right' in benefit_gdf_filtered.columns:
                benefit_gdf_filtered = benefit_gdf_filtered.drop(columns=['index_right'])
        else:
            benefit_gdf_filtered = benefit_gdf.copy()
        
        # Setup axis without frame (same as create_benefit_map)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Plot Germany outline
        germany.boundary.plot(ax=ax, color='black', linewidth=0.6, zorder=3)
        
        # Plot benefit areas using the same logic as create_benefit_map
        if not benefit_gdf_filtered.empty:
            for category in benefit_colors.keys():
                category_data = benefit_gdf_filtered[benefit_gdf_filtered['benefit_category'] == category]
                if not category_data.empty:
                    category_data.plot(
                        ax=ax,
                        color=benefit_colors[category],
                        markersize=ms,
                        alpha=0.7 if category != "Neither reachable within 60 min" else 0.8,
                        edgecolor='none',
                        linewidth=0,
                        zorder=2
                    )
        
        # Set bounds with margins (same as create_benefit_map)
        ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
        ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)
        
        # Clean title
        ax.set_title(scenario_name, fontsize=10, fontweight='bold', pad=10)
        
        # Add panel letter (lowercase, upper left)
        if i < len(panel_letters):
            # Position letters above the title for better visibility
            title_y = 1.1  # Positioned above the title
            ax.text(
                0.01,
                title_y,
                panel_letters[i],
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=12,
                family="Times New Roman",
                va="top",
                ha="left",
            )
    
    # Hide empty subplots if less than 4 scenarios
    for i in range(len(scenario_gdfs), 4):
        axes[i].set_visible(False)
    
    # No overall title for cleaner publication-ready appearance
    
    # Add legend with frames around colors for visibility (same style as create_benefit_map)
    legend_elements = []
    for cat in benefit_colors.keys():
        legend_elements.append(
            mpatches.Patch(facecolor=benefit_colors[cat], label=cat, 
                           edgecolor='black', linewidth=1.0)
        )
    
    # Position legend in Eastern Germany indent area (same as create_benefit_map)
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=10, 
              frameon=True, fancybox=True, shadow=True, facecolor='white', edgecolor='black')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.9)
    
    # Save the comparison figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Publication-ready four-scenario comparison saved: {output_path}")
    return output_path


def three_speed_scenarios(
    benefit_threshold: float = 10.0,
    grid_resolution: float = 0.015,
    time_bins: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (4.21, 11),
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    force_recalc: bool = False,
    render_mode: str = "points",  # "points" | "raster" | "polygons"
    raster_width: int = 900,
    raster_height: int = 1200,
    grid_cell_deg: float = 0.02,    # for polygon mode (~2 km at DE latitudes)
    grid_shape: str = "square"      # "square" or "hex"
) -> Path:
    """Create three-panel speed scenario comparison.
    
    Compares CT benefits across three speed scenarios:
    1. Normal driving conditions
    2. +20% Driving Speed (Emergency conditions)
    3. -20% Driving Speed (Traffic conditions)
    
    Parameters
    ----------
    benefit_threshold : float, optional
        Minimum time difference to consider significant benefit. Default 10.0.
    grid_resolution : float, optional
        Grid resolution for analysis in degrees. Default 0.015.
    time_bins : List[int], optional
        Time bins to analyze. If None, uses config.TIME_BINS.
    output_path : Path, optional
        Output file path. If None, saves to Graphs directory.
    figsize : tuple, optional
        Figure size (width, height). Default (4.21, 11).
    use_parallel : bool, optional
        Whether to use parallel processing. Default True.
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses CPU count.
    force_recalc : bool, optional
        Whether to force recalculation of benefits even if cached. Default False.
    render_mode : str, optional
        Rendering mode: "points" | "raster" | "polygons". Default "points".
    raster_width : int, optional
        Width in pixels for raster mode. Default 900.
    raster_height : int, optional
        Height in pixels for raster mode. Default 1200.
    grid_cell_deg : float, optional
        Cell size in degrees for polygon mode (~2 km at DE latitudes). Default 0.02.
    grid_shape : str, optional
        Grid shape for polygon mode: "square" or "hex". Default "square".
        
    Returns
    -------
    Path
        Path to saved comparison figure.
    """
    
    if time_bins is None:
        time_bins = config.TIME_BINS
    
    if output_path is None:
        base_name = f"three_speed_scenarios_normal_threshold_{benefit_threshold}min"
        output_path = config.GRAPH_DIR / f"{base_name}.png"
    
    print(f"🎨 Creating three-speed scenarios comparison")
    print(f"   Threshold: {benefit_threshold} min")
    
    # Define speed scenarios
    scenarios = {
        'Normal': {'suffix': '', 'penalty': 0, 'description': 'Standard driving'},
        '+20% Driving Speed': {'suffix': '_emergency', 'penalty': 0, 'description': 'Emergency (+20% speed)'},
        '-20% Driving Speed': {'suffix': '_bad_traffic', 'penalty': 0, 'description': 'Bad traffic (-20% speed)'}
    }
    
    # Calculate benefits for each scenario
    scenario_gdfs = {}
    for scenario_name, scenario_config in scenarios.items():
        try:
            print(f"   🔍 Calculating {scenario_name}...")
            
            if use_parallel:
                benefit_gdf = calculate_time_benefits_parallel(
                    ct_suffix=f"_all_CTs{scenario_config['suffix']}",
                    stroke_suffix=scenario_config['suffix'],
                    ct_penalty=scenario_config['penalty'],
                    benefit_threshold=benefit_threshold,
                    grid_resolution=grid_resolution,
                    time_bins=time_bins,
                    max_workers=max_workers,
                    force_recalc=force_recalc
                )
            else:
                benefit_gdf = calculate_time_benefits(
                    ct_suffix=f"_all_CTs{scenario_config['suffix']}",
                    stroke_suffix=scenario_config['suffix'],
                    ct_penalty=scenario_config['penalty'],
                    benefit_threshold=benefit_threshold,
                    grid_resolution=grid_resolution,
                    time_bins=time_bins
                )
            
            scenario_gdfs[scenario_name] = benefit_gdf
            print(f"     Found {len(benefit_gdf):,} benefit areas")
            
        except Exception as e:
            print(f"     ⚠️ Error: {e}")
            scenario_gdfs[scenario_name] = gpd.GeoDataFrame()
    
    if not scenario_gdfs or all(gdf.empty for gdf in scenario_gdfs.values()):
        print("⚠️  No benefit data to visualize")
        return output_path
    
    # Load Germany outline
    germany = data.load_germany_outline()
    
    # Create 3x1 subplot figure (vertical layout)
    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=600)
    
    # Calculate scaled marker size for consistent visual density
    ms = scale_markersize(figsize_current=figsize, figsize_ref=(12, 10), markersize_ref=0.5)
    
    # Use the same colors and styling as the four-panel function
    benefit_colors = {
        "High (30+ min)": "#006837",     # Dark green
        "Medium (20-30 min)": "#31a354", # Medium green  
        "Low (10-20 min)": "#78c679",    # Light green
        "Likely irrelevant (<10 min)": "white",  # White - includes both <10min and no significant benefit
        "Only CT reachable within 60 min": "#41b6c4",  # Teal - harmonizes with greens
        "Neither reachable within 60 min": SOLID_GREY  # Solid grey
    }
    
    # Plot each scenario
    panel_letters = ['a)', 'b)', 'c)']  # Lowercase letters with parentheses for panels
    for i, (scenario_name, benefit_gdf) in enumerate(list(scenario_gdfs.items())[:3]):
        ax = axes[i]
        
        # Apply the same geographic filtering
        if not benefit_gdf.empty:
            print(f"   🗺️  Filtering {scenario_name} data to Germany boundaries...")
            if benefit_gdf.crs != germany.crs and germany.crs is not None:
                benefit_gdf_filtered = benefit_gdf.to_crs(germany.crs)
            else:
                benefit_gdf_filtered = benefit_gdf.copy()
            
            # Use spatial intersection to keep only points within Germany
            germany_geom = gpd.GeoDataFrame(geometry=germany.geometry, crs=germany.crs)
            benefit_gdf_filtered = gpd.sjoin(
                benefit_gdf_filtered, 
                germany_geom, 
                how='inner', 
                predicate='within'
            )
            
            # Remove the extra index column added by sjoin
            if 'index_right' in benefit_gdf_filtered.columns:
                benefit_gdf_filtered = benefit_gdf_filtered.drop(columns=['index_right'])
        else:
            benefit_gdf_filtered = benefit_gdf.copy()
        
        # Setup axis without frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Plot Germany outline
        germany.boundary.plot(ax=ax, color='black', linewidth=0.6, zorder=3)
        
        if render_mode == "raster" and not benefit_gdf_filtered.empty:
            img, extent, _ = datashade_categories(
                benefit_gdf_filtered, germany,
                category_col="benefit_category",
                cmap_dict=benefit_colors,
                width=raster_width,
                height=raster_height
            )
            imshow_datashaded(ax, img, extent)
        elif render_mode == "polygons" and not benefit_gdf_filtered.empty:
            grid = aggregate_points_to_grid(
                benefit_gdf_filtered, germany,
                category_col="benefit_category",
                cell_size_deg=grid_cell_deg,
                grid_shape=grid_shape,
                agg="mode"
            )
            plot_grid_categories(
                grid_gdf=grid,
                germany=germany,
                ax=ax,
                benefit_colors=benefit_colors,
                show_healthcare_deserts=True
            )
        else:
            if not benefit_gdf_filtered.empty:
                for category in benefit_colors.keys():
                    category_data = benefit_gdf_filtered[benefit_gdf_filtered['benefit_category'] == category]
                    if category_data.empty:
                        continue
                    category_data.plot(
                        ax=ax,
                        color=benefit_colors[category],
                        markersize=ms,
                        alpha=0.7,
                        zorder=2,
                        rasterized=True)
        
        # Set bounds with margins
        ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
        ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)
        
        # Title and panel letter
        ax.set_title(scenario_name, fontsize=10, fontweight='bold', pad=10)
        
        # Add panel letter (lowercase, upper left)
        # Position letters above the title for better visibility
        title_y = 1.1  # Positioned above the title
        ax.text(
            0.01,
            title_y,
            panel_letters[i],
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            family="Times New Roman",
            va="top",
            ha="left",
        )
    
    # Add legend on the right side with more rows
    legend_elements = []
    for cat in benefit_colors.keys():
        legend_elements.append(
            mpatches.Patch(facecolor=benefit_colors[cat], label=cat, 
                           edgecolor='black', linewidth=1.0)
        )
    
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=10, 
              frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the comparison figure in multiple formats
    base_name = output_path.stem
    output_dir = output_path.parent
    
    # Save as PNG
    png_path = output_dir / f"{base_name}.png"
    plt.savefig(png_path, dpi=500, bbox_inches='tight')
    
    # Save as EPS
    eps_path = output_dir / f"{base_name}.eps"
    plt.savefig(eps_path, dpi=500, bbox_inches='tight')
    
    # Save as TIFF
    tiff_path = output_dir / f"{base_name}.tiff"
    plt.savefig(tiff_path, dpi=500, bbox_inches='tight')
    
    plt.close()
    
    print(f"✅ Three-speed scenarios comparison saved:")
    print(f"   PNG: {png_path}")
    print(f"   EPS: {eps_path}")
    print(f"   TIFF: {tiff_path}")
    return output_path


def three_penalty_scenarios(
    benefit_threshold: float = 10.0,
    grid_resolution: float = 0.015,
    time_bins: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (4.21, 11),
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    force_recalc: bool = False,
    render_mode: str = "points",  # "points" | "raster" | "polygons"
    raster_width: int = 900,
    raster_height: int = 1200,
    grid_cell_deg: float = 0.02,    # for polygon mode (~2 km at DE latitudes)
    grid_shape: str = "square"      # "square" or "hex"
) -> Path:
    """Create three-panel penalty scenario comparison.
    
    Compares CT benefits across three penalty scenarios:
    1. Normal + 10 min penalty
    2. Normal + 20 min penalty
    3. Normal + 30 min penalty
    
    Parameters
    ----------
    benefit_threshold : float, optional
        Minimum time difference to consider significant benefit. Default 10.0.
    grid_resolution : float, optional
        Grid resolution for analysis in degrees. Default 0.015.
    time_bins : List[int], optional
        Time bins to analyze. If None, uses config.TIME_BINS.
    output_path : Path, optional
        Output file path. If None, saves to Graphs directory.
    figsize : tuple, optional
        Figure size (width, height). Default (4.21, 11).
    use_parallel : bool, optional
        Whether to use parallel processing. Default True.
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses CPU count.
    force_recalc : bool, optional
        Whether to force recalculation of benefits even if cached. Default False.
    render_mode : str, optional
        Rendering mode: "points" | "raster" | "polygons". Default "points".
    raster_width : int, optional
        Width in pixels for raster mode. Default 900.
    raster_height : int, optional
        Height in pixels for raster mode. Default 1200.
    grid_cell_deg : float, optional
        Cell size in degrees for polygon mode (~2 km at DE latitudes). Default 0.02.
    grid_shape : str, optional
        Grid shape for polygon mode: "square" or "hex". Default "square".
        
    Returns
    -------
    Path
        Path to saved comparison figure.
    """
    
    if time_bins is None:
        time_bins = config.TIME_BINS
    
    if output_path is None:
        base_name = f"three_penalty_scenarios_normal_threshold_{benefit_threshold}min"
        output_path = config.GRAPH_DIR / f"{base_name}.png"
    
    print(f"🎨 Creating three-penalty scenarios comparison")
    print(f"   Threshold: {benefit_threshold} min")
    
    # Define penalty scenarios (all use normal driving conditions)
    scenarios = {
        '+10 min Penalty': {'suffix': '', 'penalty': 10, 'description': 'Normal + 10min penalty'},
        '+20 min Penalty': {'suffix': '', 'penalty': 20, 'description': 'Normal + 20min penalty'},
        '+30 min Penalty': {'suffix': '', 'penalty': 30, 'description': 'Normal + 30min penalty'}
    }
    
    # Calculate benefits for each scenario
    scenario_gdfs = {}
    for scenario_name, scenario_config in scenarios.items():
        try:
            print(f"   🔍 Calculating {scenario_name}...")
            
            if use_parallel:
                benefit_gdf = calculate_time_benefits_parallel(
                    ct_suffix=f"_all_CTs{scenario_config['suffix']}",
                    stroke_suffix=scenario_config['suffix'],
                    ct_penalty=scenario_config['penalty'],
                    benefit_threshold=benefit_threshold,
                    grid_resolution=grid_resolution,
                    time_bins=time_bins,
                    max_workers=max_workers,
                    force_recalc=force_recalc
                )
            else:
                benefit_gdf = calculate_time_benefits(
                    ct_suffix=f"_all_CTs{scenario_config['suffix']}",
                    stroke_suffix=scenario_config['suffix'],
                    ct_penalty=scenario_config['penalty'],
                    benefit_threshold=benefit_threshold,
                    grid_resolution=grid_resolution,
                    time_bins=time_bins
                )
            
            scenario_gdfs[scenario_name] = benefit_gdf
            print(f"     Found {len(benefit_gdf):,} benefit areas")
            
        except Exception as e:
            print(f"     ⚠️ Error: {e}")
            scenario_gdfs[scenario_name] = gpd.GeoDataFrame()
    
    if not scenario_gdfs or all(gdf.empty for gdf in scenario_gdfs.values()):
        print("⚠️  No benefit data to visualize")
        return output_path
    
    # Load Germany outline
    germany = data.load_germany_outline()
    
    # Create 3x1 subplot figure (vertical layout)
    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=500)
    
    # Calculate scaled marker size for consistent visual density
    ms = scale_markersize(figsize_current=figsize, figsize_ref=(12, 10), markersize_ref=0.5)
    
    # Use the same colors and styling as the four-panel function
    benefit_colors = {
        "High (30+ min)": "#006837",     # Dark green
        "Medium (20-30 min)": "#31a354", # Medium green  
        "Low (10-20 min)": "#78c679",    # Light green
        "Likely irrelevant (<10 min)": "white",  # White - includes both <10min and no significant benefit
        "Only CT reachable within 60 min": "#41b6c4",  # Teal - harmonizes with greens
        "Neither reachable within 60 min": SOLID_GREY  # Solid grey
    }
    
    # Plot each scenario
    panel_letters = ['a)', 'b)', 'c)']  # Lowercase letters with parentheses for panels
    for i, (scenario_name, benefit_gdf) in enumerate(list(scenario_gdfs.items())[:3]):
        ax = axes[i]
        
        # Apply the same geographic filtering
        if not benefit_gdf.empty:
            print(f"   🗺️  Filtering {scenario_name} data to Germany boundaries...")
            if benefit_gdf.crs != germany.crs and germany.crs is not None:
                benefit_gdf_filtered = benefit_gdf.to_crs(germany.crs)
            else:
                benefit_gdf_filtered = benefit_gdf.copy()
            
            # Use spatial intersection to keep only points within Germany
            germany_geom = gpd.GeoDataFrame(geometry=germany.geometry, crs=germany.crs)
            benefit_gdf_filtered = gpd.sjoin(
                benefit_gdf_filtered, 
                germany_geom, 
                how='inner', 
                predicate='within'
            )
            
            # Remove the extra index column added by sjoin
            if 'index_right' in benefit_gdf_filtered.columns:
                benefit_gdf_filtered = benefit_gdf_filtered.drop(columns=['index_right'])
        else:
            benefit_gdf_filtered = benefit_gdf.copy()
        
        # Setup axis without frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Plot Germany outline
        germany.boundary.plot(ax=ax, color='black', linewidth=0.6, zorder=3)
        
        if render_mode == "raster" and not benefit_gdf_filtered.empty:
            img, extent, _ = datashade_categories(
                benefit_gdf_filtered, germany,
                category_col="benefit_category",
                cmap_dict=benefit_colors,
                width=raster_width,
                height=raster_height
            )
            imshow_datashaded(ax, img, extent)
        elif render_mode == "polygons" and not benefit_gdf_filtered.empty:
            grid = aggregate_points_to_grid(
                benefit_gdf_filtered, germany,
                category_col="benefit_category",
                cell_size_deg=grid_cell_deg,
                grid_shape=grid_shape,
                agg="mode"
            )
            plot_grid_categories(
                grid_gdf=grid,
                germany=germany,
                ax=ax,
                benefit_colors=benefit_colors,
                show_healthcare_deserts=True
            )
        else:
            if not benefit_gdf_filtered.empty:
                for category in benefit_colors.keys():
                    category_data = benefit_gdf_filtered[benefit_gdf_filtered['benefit_category'] == category]
                    if category_data.empty:
                        continue
                    if category == "Neither reachable within 60 min":
                        category_data.plot(
                            ax=ax, color=benefit_colors[category], markersize=ms,
                            alpha=0.8, edgecolor='black', linewidth=0.1,
                            zorder=2, rasterized=True)
                    else:
                        category_data.plot(
                            ax=ax, color=benefit_colors[category], markersize=ms,
                            alpha=0.7, zorder=2, rasterized=True)
        
        # Set bounds with margins
        ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
        ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)
        
        # Title and panel letter
        ax.set_title(scenario_name, fontsize=10, fontweight='bold', pad=10)
        
        # Add panel letter (lowercase, upper left)
        # Position letters above the title for better visibility
        title_y = 1.1  # Positioned above the title
        ax.text(
            0.01,
            title_y,
            panel_letters[i],
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            family="Times New Roman",
            va="top",
            ha="left",
        )
    
    # Add legend on the right side with more rows
    legend_elements = []
    for cat in benefit_colors.keys():
        legend_elements.append(
            mpatches.Patch(facecolor=benefit_colors[cat], label=cat, 
                           edgecolor='black', linewidth=1.0)
        )
    
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=10, 
              frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the comparison figure in multiple formats
    base_name = output_path.stem
    output_dir = output_path.parent
    
    # Save as PNG
    png_path = output_dir / f"{base_name}.png"
    plt.savefig(png_path, dpi=500, bbox_inches='tight')
    
    # Save as EPS
    eps_path = output_dir / f"{base_name}.eps"
    plt.savefig(eps_path, dpi=500, bbox_inches='tight')
    
    # Save as TIFF
    tiff_path = output_dir / f"{base_name}.tiff"
    plt.savefig(tiff_path, dpi=500, bbox_inches='tight')
    
    plt.close()
    
    print(f"✅ Three-penalty scenarios comparison saved:")
    print(f"   PNG: {png_path}")
    print(f"   EPS: {eps_path}")
    print(f"   TIFF: {tiff_path}")
    return output_path


def calculate_comprehensive_population_coverage(
    benefit_threshold: float = 10.0,
    ct_penalty: float = 5.0,
    grid_resolution: float = 0.015,
    time_bins: Optional[List[int]] = None,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """Calculate comprehensive population coverage for all scenarios with complete statistics.
    
    This function provides complete population coverage analysis including:
    - All four scenarios (Normal, Emergency, Traffic, Normal+penalty)
    - All benefit categories including "no significant benefit" areas
    - Total German population breakdown by scenario
    - Export to Excel with multiple sheets
    
    Parameters
    ----------
    benefit_threshold : float, optional
        Minimum time difference to consider significant benefit. Default 10.0.
    ct_penalty : float, optional
        Additional time penalty for CT hospitals in Normal+penalty scenario. Default 5.0.
    grid_resolution : float, optional
        Grid resolution for analysis in degrees. Default 0.015.
    time_bins : List[int], optional
        Time bins to analyze. If None, uses [15, 30, 45, 60].
    use_parallel : bool, optional
        Whether to use parallel processing. Default True.
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses CPU count.
    output_dir : Path, optional
        Output directory for results. If None, uses config.RESULTS_DIR.
        
    Returns
    -------
    dict
        Dictionary containing coverage DataFrames for each scenario and summary.
    """
    
    if time_bins is None:
        time_bins = [15, 30, 45, 60]
    
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    
    print(f"📊 Comprehensive Population Coverage Analysis")
    print(f"   Threshold: {benefit_threshold} min, CT penalty: {ct_penalty} min")
    print(f"   Grid resolution: {grid_resolution}°")
    
    # First, check for data quality issues
    print(f"\n🔍 Checking scenario data quality...")
    data_ok = diagnose_emergency_scenario_issue()
    if not data_ok:
        print(f"\n⚠️  WARNING: Data quality issues detected!")
        print(f"   Population statistics may be inaccurate for affected scenarios")
        print(f"   Consider fixing the issues before proceeding")
        print(f"   Continuing anyway...\n")
    else:
        print(f"✅ All scenario data looks good!\n")
    
    # Define scenarios
    scenarios = {
        'Normal': {'suffix': '', 'penalty': 0, 'description': 'Standard driving'},
        'Emergency': {'suffix': '_emergency', 'penalty': 0, 'description': 'Emergency (+20% speed)'},
        'Traffic': {'suffix': '_bad_traffic', 'penalty': 0, 'description': 'Bad traffic (-20% speed)'},
        f'Normal +{ct_penalty}min': {'suffix': '', 'penalty': ct_penalty, 'description': f'Standard + {ct_penalty}min penalty'}
    }
    
    # Calculate benefits for each scenario
    scenario_results = {}
    coverage_results = {}
    
    # Load total German population for context
    if config.POP_RASTER.exists():
        with rasterio.open(config.POP_RASTER) as src:
            germany = data.load_germany_outline()
            total_img, _ = mask(src, germany.geometry, crop=True)
            nodata = src.nodata if src.nodata is not None else -200
            total_arr = total_img[0]
            total_arr[total_arr == nodata] = 0
            total_german_pop = int(total_arr.sum())
        print(f"📍 Total German population: {total_german_pop:,}")
    else:
        total_german_pop = config.POP_TOTAL
        print(f"📍 Using default population estimate: {total_german_pop:,}")
    
    for scenario_name, scenario_config in scenarios.items():
        try:
            print(f"\n🔍 Analyzing {scenario_name}...")
            
            # Calculate scenario benefits
            if use_parallel:
                benefit_gdf = calculate_time_benefits_parallel(
                    ct_suffix=f"_all_CTs{scenario_config['suffix']}",
                    stroke_suffix=scenario_config['suffix'],
                    ct_penalty=scenario_config['penalty'],
                    benefit_threshold=benefit_threshold,
                    grid_resolution=grid_resolution,
                    time_bins=time_bins,
                    max_workers=max_workers
                )
            else:
                benefit_gdf = calculate_time_benefits(
                    ct_suffix=f"_all_CTs{scenario_config['suffix']}",
                    stroke_suffix=scenario_config['suffix'],
                    ct_penalty=scenario_config['penalty'],
                    benefit_threshold=benefit_threshold,
                    grid_resolution=grid_resolution,
                    time_bins=time_bins
                )
            
            scenario_results[scenario_name] = benefit_gdf
            
            # Calculate population coverage for this scenario
            if not benefit_gdf.empty:
                coverage_df = calculate_benefit_coverage(benefit_gdf, scenario_name)
                
                # Calculate population NOT in any benefit area
                covered_pop = coverage_df['covered_pop'].sum()
                uncovered_pop = total_german_pop - covered_pop
                uncovered_pct = (uncovered_pop / total_german_pop * 100) if total_german_pop > 0 else 0
                
                # Add row for uncovered population
                uncovered_row = pd.DataFrame({
                    'region': [scenario_name],
                    'benefit_category': ['No significant benefit'],
                    'covered_pop': [uncovered_pop],
                    'percentage': [uncovered_pct]
                })
                
                coverage_df = pd.concat([coverage_df, uncovered_row], ignore_index=True)
                coverage_results[scenario_name] = coverage_df
                
                print(f"   Areas with benefits: {len(benefit_gdf):,}")
                print(f"   Population with benefits: {covered_pop:,} ({100-uncovered_pct:.1f}%)")
                print(f"   Population without benefits: {uncovered_pop:,} ({uncovered_pct:.1f}%)")
            else:
                # No benefits found - entire population has no significant benefit
                coverage_df = pd.DataFrame({
                    'region': [scenario_name],
                    'benefit_category': ['No significant benefit'],
                    'covered_pop': [total_german_pop],
                    'percentage': [100.0]
                })
                coverage_results[scenario_name] = coverage_df
                print(f"   No benefit areas found - entire population without significant benefits")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            # Create error placeholder
            coverage_results[scenario_name] = pd.DataFrame({
                'region': [scenario_name],
                'benefit_category': ['Analysis failed'],
                'covered_pop': [0],
                'percentage': [0.0]
            })
    
    # Create comprehensive summary
    print(f"\n📊 Creating comprehensive summary...")
    
    # Combine all scenario results
    all_coverage = pd.concat(coverage_results.values(), ignore_index=True)
    
    # Create scenario comparison table
    scenario_summary = []
    for scenario_name, coverage_df in coverage_results.items():
        if not coverage_df.empty:
            # Calculate summary statistics for this scenario
            benefit_pop = coverage_df[coverage_df['benefit_category'] != 'No significant benefit']['covered_pop'].sum()
            no_benefit_pop = coverage_df[coverage_df['benefit_category'] == 'No significant benefit']['covered_pop'].sum()
            
            scenario_summary.append({
                'scenario': scenario_name,
                'population_with_benefits': benefit_pop,
                'population_without_benefits': no_benefit_pop,
                'pct_with_benefits': (benefit_pop / total_german_pop * 100) if total_german_pop > 0 else 0,
                'pct_without_benefits': (no_benefit_pop / total_german_pop * 100) if total_german_pop > 0 else 0,
                'total_population': total_german_pop
            })
    
    scenario_summary_df = pd.DataFrame(scenario_summary)
    
    # Create category breakdown across scenarios
    category_summary = all_coverage.pivot_table(
        index='benefit_category',
        columns='region', 
        values='covered_pop',
        fill_value=0
    ).reset_index()
    
    # Add percentage columns
    for scenario in scenarios.keys():
        if scenario in category_summary.columns:
            pct_col = f'{scenario}_pct'
            category_summary[pct_col] = (
                category_summary[scenario] / total_german_pop * 100
            ).round(2)
    
    # Save comprehensive results to Excel
    output_file = output_dir / f"comprehensive_population_coverage_threshold_{benefit_threshold}min.xlsx"
    print(f"💾 Saving comprehensive results to: {output_file.name}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Summary sheet
        scenario_summary_df.to_excel(writer, sheet_name='Scenario Summary', index=False)
        
        # Category breakdown sheet
        category_summary.to_excel(writer, sheet_name='Category Breakdown', index=False)
        
        # All coverage details
        all_coverage.to_excel(writer, sheet_name='All Coverage Details', index=False)
        
        # Individual scenario sheets
        for scenario_name, coverage_df in coverage_results.items():
            sheet_name = scenario_name.replace('/', '_').replace('+', 'plus')[:31]  # Excel sheet name limit
            coverage_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"✅ Comprehensive population coverage analysis complete!")
    print(f"\n📊 Summary across all scenarios:")
    print(scenario_summary_df.to_string(index=False))
    
    return {
        'scenario_summary': scenario_summary_df,
        'category_breakdown': category_summary,
        'all_coverage': all_coverage,
        'individual_scenarios': coverage_results,
        'output_file': output_file
    }


# -----------------------------------------------------------------------------
# Quick Datashader path (for ultra dense point rendering)
# -----------------------------------------------------------------------------

def datashade_categories(
    gdf: gpd.GeoDataFrame,
    germany: gpd.GeoDataFrame,
    category_col: str,
    cmap_dict: Dict[str, str],
    width: int = 800,
    height: int = 1000,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
):
    """Rasterize categorical point data using datashader.

    Parameters
    ----------
    gdf : GeoDataFrame with point geometries
    germany : GeoDataFrame (for bounds)
    category_col : name of the categorical column
    cmap_dict : mapping from category -> hex color
    width, height : size of the raster in pixels
    x_range, y_range : manual ranges; if None, use Germany bounds

    Returns
    -------
    img : numpy.ndarray (RGBA) suitable for imshow
    extent : tuple (xmin, xmax, ymin, ymax) to match imshow
    categories : list of categories rendered (ordering matches cmap)
    """
    if gdf.empty:
        # return empty transparent image
        xmin, ymin, xmax, ymax = germany.total_bounds
        img = np.zeros((height, width, 4), dtype=np.uint8)
        return img, (xmin, xmax, ymin, ymax), []

    # Ensure CRS
    if gdf.crs != germany.crs and germany.crs is not None:
        gdf = gdf.to_crs(germany.crs)

    xmin, ymin, xmax, ymax = germany.total_bounds
    if x_range is None:
        x_range = (xmin, xmax)
    if y_range is None:
        y_range = (ymin, ymax)

    # Map categories to integer codes to make life easier
    cats = list(cmap_dict.keys())
    cat_to_int = {c: i for i, c in enumerate(cats)}
    gdf = gdf.copy()
    gdf["_cat_int"] = gdf[category_col].map(cat_to_int).fillna(-1).astype(int)
    
    # Extract x and y coordinates from Point geometries
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y

    # Prepare datashader canvas
    cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=x_range, y_range=y_range)
    # Aggregate: maximum category index per pixel wins (prefer higher benefits?)
    agg = cvs.points(gdf, x="x", y="y", agg=ds.max("_cat_int"))

    # Convert aggregation to RGBA using our palette
    # Create a ListedColormap including transparent for -1
    palette = ["#00000000"] + [cmap_dict[c] for c in cats]  # prepend transparent
    cmap = ListedColormap(palette)

    # Convert to numpy array of ints
    arr = agg.to_pandas().values
    arr = arr.filled(-1).astype(int)
    # Shift by +1 so -1 -> 0 index (transparent), 0->1 etc.
    arr_shift = arr + 1

    rgba = cmap(arr_shift, bytes=True)  # returns uint8 RGBA

    # Datashader returns top-left origin; matplotlib expects lower-left
    rgba = np.flipud(rgba)

    extent = (x_range[0], x_range[1], y_range[0], y_range[1])
    return rgba, extent, cats


def imshow_datashaded(ax: plt.Axes, img: np.ndarray, extent: Tuple[float, float, float, float]) -> None:
    """Convenience wrapper to draw the datashaded RGBA on an axis."""
    ax.imshow(img, extent=extent, origin='lower', interpolation='nearest')


# -----------------------------------------------------------------------------
# Polygon-based aggregation for ultra-crisp small figures
# -----------------------------------------------------------------------------

def build_fishnet(bounds: Tuple[float, float, float, float], cell_size_deg: float, crs: str) -> gpd.GeoDataFrame:
    """Create a square fishnet grid covering the bounds."""
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, cell_size_deg)
    ys = np.arange(miny, maxy, cell_size_deg)
    polys = []
    for x in xs:
        for y in ys:
            polys.append(Polygon([(x, y), (x + cell_size_deg, y), (x + cell_size_deg, y + cell_size_deg), (x, y + cell_size_deg)]))
    return gpd.GeoDataFrame({"geometry": polys}, crs=crs)


def build_hex_grid(bounds: Tuple[float, float, float, float], cell_size_deg: float, crs: str) -> gpd.GeoDataFrame:
    """Create a hexagonal grid covering bounds with roughly cell_size_deg spacing.
    The vertical spacing is sqrt(3)/2 * width for hex packing.
    """
    minx, miny, maxx, maxy = bounds
    w = cell_size_deg
    h = math.sqrt(3) * w / 2  # vertical spacing for hex rows

    cols = int(math.ceil((maxx - minx) / w)) + 1
    rows = int(math.ceil((maxy - miny) / h)) + 1

    polys = []
    for row in range(rows):
        y = miny + row * h
        # offset every other row by half a width
        x_offset = (w / 2) if (row % 2) else 0
        for col in range(cols):
            x = minx + col * w + x_offset
            # build hex centered at (x,y)
            poly = Polygon([
                (x - w/2,     y),
                (x - w/4,     y + h/2),
                (x + w/4,     y + h/2),
                (x + w/2,     y),
                (x + w/4,     y - h/2),
                (x - w/4,     y - h/2)
            ])
            polys.append(poly)
    return gpd.GeoDataFrame({"geometry": polys}, crs=crs)


def aggregate_points_to_grid(
    pts_gdf: gpd.GeoDataFrame,
    germany: gpd.GeoDataFrame,
    category_col: str = "benefit_category",
    cell_size_deg: float = 0.02,
    grid_shape: str = "square",
    agg: str = "mode",
) -> gpd.GeoDataFrame:
    """Aggregate point categories to a square or hex grid.

    Parameters
    ----------
    pts_gdf : GeoDataFrame
        Point data with a categorical column.
    germany : GeoDataFrame
        Boundary for clipping and extent.
    category_col : str
        Name of the categorical column to aggregate.
    cell_size_deg : float
        Cell size in degrees (approx 1 deg ~ 111 km at DE latitude).
    grid_shape : str
        'square' or 'hex'.
    agg : str
        Aggregation: 'mode' (most frequent) or 'max_benefit' (uses time_benefit field).
    """
    if pts_gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry", category_col], crs=germany.crs)

    # Ensure CRS consistency
    if pts_gdf.crs != germany.crs and germany.crs is not None:
        pts = pts_gdf.to_crs(germany.crs)
    else:
        pts = pts_gdf.copy()

    bounds = tuple(germany.total_bounds)
    if grid_shape == "hex":
        grid = build_hex_grid(bounds, cell_size_deg, germany.crs)
    else:
        grid = build_fishnet(bounds, cell_size_deg, germany.crs)

    # Clip to Germany
    grid = gpd.overlay(grid, germany[["geometry"]], how="intersection")

    # Spatial join to assign points to cells
    joined = gpd.sjoin(pts[["geometry", category_col, "time_benefit"]], grid, how="inner", predicate="within")

    if agg == "mode":
        def _mode(vals):
            return Counter(vals).most_common(1)[0][0] if len(vals) else None
        cat_series = joined.groupby("index_right")[category_col].apply(_mode)
        grid[category_col] = cat_series

    elif agg == "max_benefit":
        def _best(group):
            row = group.loc[group["time_benefit"].idxmax()]
            return row[category_col]
        best_series = joined.groupby("index_right").apply(_best)
        grid[category_col] = best_series
    else:
        raise ValueError("Unsupported agg method")

    grid = grid.dropna(subset=[category_col])
    return grid


def plot_grid_categories(
    grid_gdf: gpd.GeoDataFrame,
    germany: gpd.GeoDataFrame,
    ax: plt.Axes,
    benefit_colors: Dict[str, str],
    show_healthcare_deserts: bool = True,
) -> None:
    """Plot the aggregated polygon grid as categories.
    Mirrors the styling of the point-based plot.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cats = list(benefit_colors.keys())
    if not show_healthcare_deserts:
        cats = [c for c in cats if c != "Neither reachable within 60 min"]

    # Simple polygon fill for all categories, no stippling
    for cat in cats:
        sub = grid_gdf[grid_gdf['benefit_category'] == cat]
        if sub.empty:
            continue
        
        # Optional: dissolve deserts so you draw ONE big polygon, not 10 000 cells
        if cat == "Neither reachable within 60 min":
            sub = sub.dissolve()
        
        sub.plot(
            ax=ax,
            facecolor=benefit_colors[cat],
            edgecolor='none',
            linewidth=0,
            zorder=2,
            rasterized=True
        )

    germany.boundary.plot(ax=ax, color='black', linewidth=0.6, zorder=3)
    ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
    ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)


def _stipple_hollow(ax,
                    deserts_gdf,
                    target_spacing_px=9,
                    edgecolor='black',
                    facecolor='white',
                    size=4.0,          # bigger size (points^2) because we have hollow centers
                    alpha=0.7,
                    max_points=70_000):
    """
    (Patched) No-op when USE_STIPPLE is False.
    Original implementation drew hollow dots; we skip that for solid grey areas.
    """
    if not USE_STIPPLE:
        return

    # --- original code retained below for optional future use ---
    # Draw hollow (white-filled) dots over the polygon to avoid 'grey mush'.
    if deserts_gdf.empty: return

    union_poly = unary_union(deserts_gdf.geometry.values)
    if union_poly.is_empty: return

    minx, miny, maxx, maxy = union_poly.bounds
    trans = ax.transData
    x0, y0 = trans.transform((minx, miny))
    x1, y1 = trans.transform((maxx, maxy))
    bbox_px_area = abs((x1 - x0) * (y1 - y0))

    est_points = int(bbox_px_area / (target_spacing_px ** 2))
    n_points = min(est_points, max_points)
    if n_points <= 0: return

    xs = np.random.uniform(minx, maxx, n_points * 2)
    ys = np.random.uniform(miny, maxy, n_points * 2)
    mask = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=deserts_gdf.crs).within(union_poly)
    xs = xs[mask.values][:n_points]
    ys = ys[mask.values][:n_points]

    # Draw twice: white dot first, thin black edge on top to keep sharpness
    ax.scatter(xs, ys, s=size, c=facecolor, alpha=1.0, linewidths=0,
               rasterized=True, zorder=3)
    ax.scatter(xs, ys, s=size, facecolors='none', edgecolors=edgecolor,
               alpha=alpha, linewidths=0.15, rasterized=True, zorder=4)


def diagnose_emergency_scenario_issue():
    """Diagnose potential issues with emergency scenario isochrone data."""
    print("🔍 Diagnosing Emergency Scenario Issue")
    print("=" * 40)
    
    scenarios_to_check = [
        ("Normal", ""),
        ("Emergency", "_emergency"), 
        ("Traffic", "_bad_traffic")
    ]
    
    time_bins = [15, 30, 45, 60]
    issues_found = []
    
    for scenario_name, suffix in scenarios_to_check:
        print(f"\n📋 {scenario_name} Scenario (suffix: '{suffix}'):")
        
        # Check stroke unit files
        print(f"   Stroke Units:")
        for t in time_bins:
            stroke_path = config.DATA_DIR / f"poly{t}{suffix}.pkl"
            if stroke_path.exists():
                with open(stroke_path, "rb") as f:
                    polys = pickle.load(f)
                valid_count = len([p for p in polys if p is not None and not p.is_empty])
                total_count = len(polys)
                
                if valid_count < total_count:
                    missing = total_count - valid_count
                    print(f"     {t}min: ⚠️  {valid_count}/{total_count} valid ({missing} MISSING)")
                    issues_found.append(f"{scenario_name} stroke units {t}min: {missing} missing polygons")
                    
                    # Find which specific facilities failed
                    failed_facilities = []
                    for i, poly in enumerate(polys):
                        if poly is None or poly.is_empty:
                            failed_facilities.append(i)
                    
                    if failed_facilities:
                        print(f"         Failed facilities: {failed_facilities}")
                        issues_found.append(f"  -> Facilities {failed_facilities} failed in {scenario_name}")
                else:
                    print(f"     {t}min: ✅ {valid_count}/{total_count} valid polygons")
            else:
                print(f"     {t}min: ❌ Missing file")
                issues_found.append(f"{scenario_name} stroke units {t}min: File missing")
        
        # Check CT hospital files  
        print(f"   CT Hospitals:")
        for t in time_bins:
            ct_path = config.DATA_DIR / f"poly{t}_all_CTs{suffix}.pkl"
            if ct_path.exists():
                with open(ct_path, "rb") as f:
                    polys = pickle.load(f)
                valid_count = len([p for p in polys if p is not None and not p.is_empty])
                total_count = len(polys)
                
                if valid_count < total_count:
                    missing = total_count - valid_count
                    print(f"     {t}min: ⚠️  {valid_count}/{total_count} valid ({missing} MISSING)")
                    issues_found.append(f"{scenario_name} CT hospitals {t}min: {missing} missing polygons")
                else:
                    print(f"     {t}min: ✅ {valid_count}/{total_count} valid polygons")
            else:
                print(f"     {t}min: ❌ Missing file")
                issues_found.append(f"{scenario_name} CT hospitals {t}min: File missing")
    
    # Summary of issues
    if issues_found:
        print(f"\n❌ ISSUES FOUND:")
        print("=" * 20)
        for issue in issues_found:
            print(f"   • {issue}")
        
        print(f"\n🔧 SOLUTION:")
        print(f"   1. Go to notebooks/01_Isochrone_Generation.ipynb")
        print(f"   2. Find the stroke unit generation section")  
        print(f"   3. Set force_recalc=True")
        print(f"   4. Re-run emergency scenario generation")
        print(f"   5. The missing polygons are causing large teal areas in your maps!")
        
        return False  # Issues found
    else:
        print(f"\n✅ NO ISSUES FOUND - All scenarios have complete polygon data")
        return True  # No issues


def calculate_scenario_benefit_coverage(
    benefit_threshold: float = 10.0,
    grid_resolution: float = 0.015,
    time_bins: Optional[List[int]] = None,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    force_recalc: bool = False,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Calculate comprehensive coverage analysis for three-speed and three-penalty scenarios.
    
    This function calculates both population and area coverage for each benefit category
    across six scenarios: three speed scenarios and three penalty scenarios.
    
    Parameters
    ----------
    benefit_threshold : float, optional
        Minimum time difference to consider significant benefit. Default 10.0.
    grid_resolution : float, optional
        Grid resolution for analysis in degrees. Default 0.015.
    time_bins : List[int], optional
        Time bins to analyze. If None, uses config.TIME_BINS.
    use_parallel : bool, optional
        Whether to use parallel processing. Default True.
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses CPU count.
    force_recalc : bool, optional
        Whether to force recalculation even if cached. Default False.
    output_dir : Path, optional
        Output directory for results. If None, uses config.RESULTS_DIR.
        
    Returns
    -------
    pd.DataFrame
        Coverage statistics with columns: scenario, benefit_category, covered_pop, 
        population_pct, covered_area_km2, area_pct
    """
    
    if time_bins is None:
        time_bins = config.TIME_BINS
    
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    
    print(f"📊 Calculating scenario benefit coverage analysis")
    print(f"   Threshold: {benefit_threshold} min")
    print(f"   Grid resolution: {grid_resolution}°")
    
    # Define all six scenarios
    speed_scenarios = {
        'Normal Speed': {'suffix': '', 'penalty': 0},
        '+20% Speed (Emergency)': {'suffix': '_emergency', 'penalty': 0},
        '-20% Speed (Traffic)': {'suffix': '_bad_traffic', 'penalty': 0}
    }
    
    penalty_scenarios = {
        '+10min Penalty': {'suffix': '', 'penalty': 10},
        '+20min Penalty': {'suffix': '', 'penalty': 20},
        '+30min Penalty': {'suffix': '', 'penalty': 30}
    }
    
    all_scenarios = {**speed_scenarios, **penalty_scenarios}
    
    # Load Germany for area calculations
    germany = data.load_germany_outline()
    germany_area_km2 = germany.to_crs(3035).area.sum() / 1_000_000  # Convert to km²
    
    # Load total population
    if config.POP_RASTER.exists():
        with rasterio.open(config.POP_RASTER) as src:
            total_img, _ = mask(src, germany.geometry, crop=True)
            nodata = src.nodata if src.nodata is not None else -200
            total_arr = total_img[0]
            total_arr[total_arr == nodata] = 0
            total_german_pop = int(total_arr.sum())
    else:
        total_german_pop = config.POP_TOTAL
    
    print(f"📍 Total German population: {total_german_pop:,}")
    print(f"📍 Total German area: {germany_area_km2:,.0f} km²")
    
    # Calculate benefits for each scenario
    all_results = []
    
    for scenario_name, scenario_config in all_scenarios.items():
        try:
            print(f"\n🔍 Analyzing {scenario_name}...")
            
            # Calculate scenario benefits
            if use_parallel:
                benefit_gdf = calculate_time_benefits_parallel(
                    ct_suffix=f"_all_CTs{scenario_config['suffix']}",
                    stroke_suffix=scenario_config['suffix'],
                    ct_penalty=scenario_config['penalty'],
                    benefit_threshold=benefit_threshold,
                    grid_resolution=grid_resolution,
                    time_bins=time_bins,
                    max_workers=max_workers,
                    force_recalc=force_recalc
                )
            else:
                benefit_gdf = calculate_time_benefits(
                    ct_suffix=f"_all_CTs{scenario_config['suffix']}",
                    stroke_suffix=scenario_config['suffix'],
                    ct_penalty=scenario_config['penalty'],
                    benefit_threshold=benefit_threshold,
                    grid_resolution=grid_resolution,
                    time_bins=time_bins
                )
            
            # Calculate coverage for each benefit category
            if not benefit_gdf.empty:
                # Ensure benefit_gdf is in the same CRS as Germany
                if benefit_gdf.crs != germany.crs and germany.crs is not None:
                    benefit_gdf_proj = benefit_gdf.to_crs(germany.crs)
                else:
                    benefit_gdf_proj = benefit_gdf.copy()
                
                # Filter to Germany boundaries
                germany_geom = gpd.GeoDataFrame(geometry=germany.geometry, crs=germany.crs)
                benefit_gdf_filtered = gpd.sjoin(
                    benefit_gdf_proj, 
                    germany_geom, 
                    how='inner', 
                    predicate='within'
                )
                if 'index_right' in benefit_gdf_filtered.columns:
                    benefit_gdf_filtered = benefit_gdf_filtered.drop(columns=['index_right'])
                
                print(f"   Filtered to {len(benefit_gdf_filtered):,} points within Germany")
                
                # Calculate coverage for each category
                categories = benefit_gdf_filtered['benefit_category'].unique()
                
                for category in categories:
                    category_data = benefit_gdf_filtered[benefit_gdf_filtered['benefit_category'] == category]
                    
                    if not category_data.empty:
                        # Population coverage calculation
                        if config.POP_RASTER.exists():
                            try:
                                # Create buffer around points to approximate coverage area
                                # Use a buffer based on grid resolution
                                buffer_size = grid_resolution / 2  # Half the grid resolution
                                category_buffered = category_data.buffer(buffer_size)
                                category_union = unary_union(category_buffered.values)
                                
                                # Calculate population in buffered areas
                                with rasterio.open(config.POP_RASTER) as src:
                                    if hasattr(category_union, 'geoms'):
                                        # MultiPolygon
                                        geometries = list(category_union.geoms)
                                    else:
                                        # Single Polygon
                                        geometries = [category_union]
                                    
                                    benefit_img, _ = mask(src, geometries, crop=True)
                                    benefit_arr = benefit_img[0]
                                    benefit_arr[benefit_arr == nodata] = 0
                                    covered_pop = int(benefit_arr.sum())
                                
                                # Calculate area coverage
                                category_area_km2 = category_union.area / 1_000_000  # Convert to km²
                                
                            except Exception as e:
                                print(f"     ⚠️ Error calculating coverage for {category}: {e}")
                                covered_pop = 0
                                category_area_km2 = 0
                        else:
                            covered_pop = 0
                            category_area_km2 = 0
                        
                        # Calculate percentages
                        population_pct = (covered_pop / total_german_pop * 100) if total_german_pop > 0 else 0
                        area_pct = (category_area_km2 / germany_area_km2 * 100) if germany_area_km2 > 0 else 0
                        
                        all_results.append({
                            'scenario': scenario_name,
                            'benefit_category': category,
                            'covered_pop': covered_pop,
                            'population_pct': round(population_pct, 2),
                            'covered_area_km2': round(category_area_km2, 1),
                            'area_pct': round(area_pct, 2)
                        })
                        
                        print(f"     {category}: {covered_pop:,} people ({population_pct:.1f}%), {category_area_km2:.1f} km² ({area_pct:.1f}%)")
            
            else:
                print(f"     No benefit areas found")
                # Add a row indicating no benefits
                all_results.append({
                    'scenario': scenario_name,
                    'benefit_category': 'No significant benefit',
                    'covered_pop': total_german_pop,
                    'population_pct': 100.0,
                    'covered_area_km2': germany_area_km2,
                    'area_pct': 100.0
                })
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            # Add error placeholder
            all_results.append({
                'scenario': scenario_name,
                'benefit_category': 'Analysis failed',
                'covered_pop': 0,
                'population_pct': 0.0,
                'covered_area_km2': 0.0,
                'area_pct': 0.0
            })
    
    # Create comprehensive DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_file = output_dir / f"scenario_benefit_coverage_analysis_threshold_{benefit_threshold}min.xlsx"
    print(f"\n💾 Saving results to: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # All results
        results_df.to_excel(writer, sheet_name='All Results', index=False)
        
        # Pivot tables for easier analysis
        pop_pivot = results_df.pivot_table(
            index='scenario',
            columns='benefit_category', 
            values='population_pct',
            fill_value=0
        )
        pop_pivot.to_excel(writer, sheet_name='Population Coverage %')
        
        area_pivot = results_df.pivot_table(
            index='scenario',
            columns='benefit_category', 
            values='area_pct',
            fill_value=0
        )
        area_pivot.to_excel(writer, sheet_name='Area Coverage %')
    
    print(f"✅ Scenario benefit coverage analysis complete!")
    return results_df