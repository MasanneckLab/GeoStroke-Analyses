"""Helper utilities to tag a benefit GeoDataFrame with rural/urban labels
   using the GHS-SMOD raster and to compute percentage summaries.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

# --- Constants -------------------------------------------------------------
# UN Degree‑of‑Urbanisation level‑2 classes (Stage I)
SMOD_URBAN_CODES = [30, 23, 22, 21]  # 21 = suburban/peri‑urban (often treated as urban)
SMOD_RURAL_CODES = [13, 12, 11]


# --- Core functions --------------------------------------------------------

def add_smod_labels(
    benefit_gdf: gpd.GeoDataFrame,
    smod_raster_path: str,
) -> gpd.GeoDataFrame:
    """Attach `smod_code` + coarse `urban_rural` label to every point.

    The function keeps the original CRS of *benefit_gdf*; your raster must be
    EPSG:4326 (the native GHSL grid). If *benefit_gdf* is in a different CRS
    it will be re-projected on the fly.
    
    Parameters
    ----------
    benefit_gdf : gpd.GeoDataFrame
        GeoDataFrame containing benefit analysis results with point geometries
    smod_raster_path : str
        Path to the GHS-SMOD raster file
        
    Returns
    -------
    gpd.GeoDataFrame
        Copy of input GeoDataFrame with added 'smod_code' and 'urban_rural' columns
    """

    # Open the SMOD raster once
    with rasterio.open(smod_raster_path) as src:
        # Store original CRS to convert back later
        original_crs = benefit_gdf.crs
        
        # Reproject to raster CRS if needed
        if benefit_gdf.crs != src.crs and src.crs is not None:
            gdf_reproj = benefit_gdf.to_crs(src.crs)  # type: ignore
        else:
            gdf_reproj = benefit_gdf

        # Extract the raster value for every point (x, y order!)
        coords = [(geom.x, geom.y) for geom in gdf_reproj.geometry]
        smod_vals = np.array([v[0] for v in src.sample(coords)], dtype="int16")  # type: ignore

    # Copy to avoid mutating caller's frame
    gdf = benefit_gdf.copy()
    gdf["smod_code"] = smod_vals

    # Map codes to two‑class label
    gdf["urban_rural"] = np.where(
        gdf["smod_code"].isin(SMOD_URBAN_CODES),
        "urban",
        np.where(gdf["smod_code"].isin(SMOD_RURAL_CODES), "rural", "other"),
    )
    
    # Ensure we maintain the original CRS
    if gdf.crs != original_crs:
        gdf = gdf.to_crs(original_crs)
    
    return gdf


def summary_by_category(gdf: gpd.GeoDataFrame, category_column: str = None) -> pd.DataFrame:
    """Return a table with counts and percentages urban/rural by category.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with category and 'urban_rural' columns
    category_column : str, optional
        Name of the category column. If None, auto-detects 'benefit_category' or 'access_category'
        
    Returns
    -------
    pd.DataFrame
        Summary table with counts and percentages by category and urban/rural classification
    """
    
    # Auto-detect category column if not specified
    if category_column is None:
        if 'benefit_category' in gdf.columns:
            category_column = 'benefit_category'
        elif 'access_category' in gdf.columns:
            category_column = 'access_category'
        else:
            raise ValueError("Could not find 'benefit_category' or 'access_category' column. "
                           "Please specify category_column parameter.")
    
    table = (
        gdf.groupby([category_column, "urban_rural"]).size().unstack(fill_value=0)
    )
    table["total"] = table.sum(axis=1)
    
    # Calculate percentages, handling cases where columns might not exist
    for col in ["urban", "rural", "other"]:
        if col in table.columns:
            pct_col = f"pct_{col}"
            table[pct_col] = (table[col] / table["total"] * 100).round(1)
        else:
            table[f"pct_{col}"] = 0.0
    
    return table


def detailed_summary_by_category(gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a more detailed table including all SMOD codes by benefit category.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with 'benefit_category', 'urban_rural', and 'smod_code' columns
        
    Returns
    -------
    pd.DataFrame
        Detailed summary table with individual SMOD codes
    """
    # First get the basic urban/rural summary
    basic_summary = summary_by_category(gdf)
    
    # Then get detailed SMOD code breakdown
    smod_summary = (
        gdf.groupby(["benefit_category", "smod_code"])
        .size()
        .unstack(fill_value=0)
    )
    
    # Add total and percentage columns for each SMOD code
    smod_summary["total"] = smod_summary.sum(axis=1)
    for col in smod_summary.columns[:-1]:  # Exclude 'total' column
        pct_col = f"pct_smod_{col}"
        smod_summary[pct_col] = (smod_summary[col] / smod_summary["total"] * 100).round(1)
    
    return basic_summary, smod_summary


# --- Example CLI -----------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Annotate benefit GDF with urban/rural from SMOD")
    parser.add_argument("benefit_file", help="Pickled or GeoPackage file containing benefit GeoDataFrame")
    parser.add_argument("smod_raster", help="Path to GHS-SMOD raster (e.g. ghs_smod_2025_v2_1km.tif)")
    args = parser.parse_args()

    print("Loading benefit dataset …")
    if args.benefit_file.endswith(".pkl"):
        import pickle
        with open(args.benefit_file, "rb") as f:
            benefit_gdf = gpd.GeoDataFrame(pickle.load(f), crs="EPSG:4326")
    else:
        benefit_gdf = gpd.read_file(args.benefit_file)

    annotated = add_smod_labels(benefit_gdf, args.smod_raster)
    summary = summary_by_category(annotated)
    print(summary.to_markdown())

    # Persist enriched data if desired
    annotated.to_file("annotated_benefits.gpkg", driver="GPKG") 