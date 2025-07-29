"""Reporting helpers – generate publication-style composite figures and maps."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Union
from shapely.geometry import Polygon, MultiPolygon

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from . import plotting, config, coverage, population

# Type alias for geometry
Geometry = Polygon | MultiPolygon

__all__ = ["national_four_panel", "generate_county_reports"]


def national_four_panel(
    germany: gpd.GeoDataFrame,
    gdf_all: gpd.GeoDataFrame,
    gdf_uber: gpd.GeoDataFrame,
    gdf_thromb: gpd.GeoDataFrame,
    gdf_reg: gpd.GeoDataFrame,
    df_all: pd.DataFrame,
    df_uber: pd.DataFrame,
    df_thromb: pd.DataFrame,
    df_reg: pd.DataFrame,
    *,
    out_dir: Path | str | None = None,
    time_bins: list[int] | None = None,
) -> Path:
    """Create the 4-panel national map and save PNG + SVG.
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to plot. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """

    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plotting.four_panel_isochrones(
        germany,
        [gdf_all, gdf_uber, gdf_thromb, gdf_reg],
        [df_all, df_uber, df_thromb, df_reg],
        titles=[
            "All Stroke Units",
            "Supra-Regional Stroke Units",
            "Thrombectomy-Certified Stroke Units",
            "Regional / Telemed Stroke Units",
        ],
        letters=["a", "b", "c", "d"],  # Use lowercase letters for journal compliance
        figsize=(18, 14),
        time_bins=time_bins,
    )

    # Use appropriate legend based on time_bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        plotting.add_legend(fig, time_bins=time_bins)
    else:
        plotting.add_legend(fig)  # Default backward-compatible legend
        
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # type: ignore[arg-type]

    # Adjust filename if using custom time bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        suffix = f"_custom_{'_'.join(map(str, time_bins))}"
        eps_path = out_dir / f"combined_stroke_units_4panels_short_driving{suffix}.eps"
        tiff_path = out_dir / f"combined_stroke_units_4panels_short_driving{suffix}.tiff"
        png_path = out_dir / f"combined_stroke_units_4panels_short_driving{suffix}.png"
    else:
        eps_path = out_dir / "combined_stroke_units_4panels_short_driving.eps"
        tiff_path = out_dir / "combined_stroke_units_4panels_short_driving.tiff"
        png_path = out_dir / "combined_stroke_units_4panels_short_driving.png"

    import matplotlib.pyplot as plt

    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', dpi=500, bbox_inches="tight")
    fig.savefig(tiff_path, format='tiff', dpi=500, bbox_inches="tight")
    fig.savefig(png_path, format='png', dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Four-panel figure saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path


def generate_county_reports(
    counties_gdf: gpd.GeoDataFrame,
    stroke_unions: Dict[int, Geometry],
    ct_unions: Dict[int, Geometry] | None = None,
    *,
    out_dir: Path | str | None = None,
    time_bins: list[int] | None = None,
) -> Path:
    """Generate comprehensive county-level reports with Excel files and PDF.
    
    This function creates:
    1. Master Excel file with all county coverage data
    2. Per-county Excel files organized by state 
    3. Multi-page PDF with county maps
    4. Meta JSON file for web interface
    
    Parameters
    ----------
    counties_gdf : gpd.GeoDataFrame
        German counties with proper administrative boundaries
    stroke_unions : Dict[int, any]
        Union polygons for stroke units by time band
    ct_unions : Dict[int, any], optional  
        Union polygons for CT hospitals by time band
    out_dir : Path | str | None
        Output directory (defaults to Results/Counties)
    time_bins : list[int], optional
        Specific time bins to include in reports. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    
    Returns
    -------
    Path
        Path to the master Excel file
    """
    
    out_dir = Path(out_dir or config.RESULTS_DIR / "Counties")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Generating comprehensive county reports...")
    print(f"   Output directory: {out_dir}")
    if time_bins:
        print(f"   Time bins: {time_bins}")
    else:
        print(f"   Time bins: {config.DEFAULT_TIME_BINS} (default)")
    
    # Calculate county-level coverage for stroke units
    stroke_coverage = coverage.county_table(counties_gdf, stroke_unions, time_bins)
    
    # Also calculate CT coverage if provided
    if ct_unions:
        ct_coverage = coverage.county_table(counties_gdf, ct_unions, time_bins)
        ct_coverage['facility_type'] = 'CT Hospitals'
    
    # Combine coverage data
    stroke_coverage['facility_type'] = 'Stroke Units'
    
    if ct_unions:
        all_coverage = pd.concat([stroke_coverage, ct_coverage], ignore_index=True)
    else:
        all_coverage = stroke_coverage
    
    # Adjust filename if using custom time bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        suffix = f"_custom_{'_'.join(map(str, time_bins))}"
        master_excel = out_dir / f"master_county_coverage{suffix}.xlsx"
    else:
        master_excel = out_dir / "master_county_coverage.xlsx"
        
    # Save master Excel file
    all_coverage.to_excel(master_excel, index=False)
    print(f"✅ Master county coverage: {master_excel}")
    
    # Generate per-county Excel files organized by state
    county_meta = {}
    
    # Ensure we have state information
    if 'state_name' not in counties_gdf.columns:
        # Add state info if missing (simplified approach)
        counties_gdf['state_name'] = 'Unknown'
    
    # Determine which time bin to use for meta coverage (default to 60 min, fallback to last available)
    coverage_time_bins = time_bins or config.DEFAULT_TIME_BINS
    meta_time_bin = 60 if 60 in coverage_time_bins else coverage_time_bins[-1]
    
    for _, county in counties_gdf.iterrows():
        county_name = str(county.get('county_name', 'Unknown'))
        state_name = str(county.get('state_name', 'Unknown'))
        
        # Create state directory
        state_dir = out_dir / state_name.replace(' ', '_')
        state_dir.mkdir(exist_ok=True)
        
        # Filter data for this county
        county_data = all_coverage[all_coverage['region'] == county_name]
        
        if not county_data.empty:
            # Adjust filename if using custom time bins
            if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
                suffix = f"_custom_{'_'.join(map(str, time_bins))}"
                county_excel = state_dir / f"coverage_{county_name.replace(' ', '_')}{suffix}.xlsx"
                excel_filename = f"coverage_{county_name.replace(' ', '_')}{suffix}.xlsx"
            else:
                county_excel = state_dir / f"coverage_{county_name.replace(' ', '_')}.xlsx"
                excel_filename = f"coverage_{county_name.replace(' ', '_')}.xlsx"
                
            # Save county Excel file
            county_data.to_excel(county_excel, index=False)
            
            # Add to meta for web interface
            if state_name not in county_meta:
                county_meta[state_name] = {}
            
            # Get coverage percentage for meta time bin safely
            county_meta_data = county_data[county_data['time_min'] == meta_time_bin]
            coverage_pct = 0.0
            if len(county_meta_data) > 0:
                coverage_values = county_meta_data['percentage'].values
                if len(coverage_values) > 0:
                    coverage_pct = float(coverage_values[0])
            
            county_meta[state_name][county_name] = {
                "excel": excel_filename,
                f"coverage_{meta_time_bin}min": coverage_pct
            }
    
    # Save meta JSON
    import json
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        suffix = f"_custom_{'_'.join(map(str, time_bins))}"
        meta_json = out_dir / f"meta{suffix}.json"
    else:
        meta_json = out_dir / "meta.json"
        
    with open(meta_json, 'w', encoding='utf-8') as f:
        json.dump(county_meta, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Per-county Excel files generated")
    print(f"✅ Meta JSON saved: {meta_json}")
    
    return master_excel 