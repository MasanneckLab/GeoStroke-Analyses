"""High-level figure generators.

The functions in this module are thin wrappers that orchestrate data
loading, minimal geoprocessing (unions) and then delegate the actual
drawing to :pymod:`geostroke.plotting`.

Two entry points correspond to the publication-ready static figures used
in the original notebook:

* :func:`create_figure_1` - CT hospitals vs All Stroke Units vs
  Thrombectomy certified
* :func:`create_figure_2` - Stroke units hierarchical split
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
import pickle

from . import config, data, plotting, iso_manager

__all__ = ["create_figure_1", "create_figure_1_extended", "create_figure_2", "create_supplemental_figure", "create_scenario_figure_1", "create_scenario_figure_2", "generate_all_scenario_figures", "create_journal_figure_1", "create_journal_figure_1_extended", "create_journal_figure_2", "run_journal_publication_figures"]


def _union(polys_by_time: dict, time_bins: list[int] | None = None) -> gpd.GeoDataFrame:
    """Create union polygons for each time bin and combine into a single GeoDataFrame.
    
    This function takes a dictionary of polygons organized by time bin and creates
    separate union geometries for each time bin, then combines them into a single 
    GeoDataFrame with a 'Time' column for proper plotting.
    
    Parameters
    ----------
    polys_by_time : dict
        Dictionary with time bins as keys and lists of polygons as values.
        Expected format: {15: [poly1, poly2, ...], 30: [poly3, poly4, ...], ...}
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS for backward compatibility.
    
    Returns
    -------
    gpd.GeoDataFrame
        Combined GeoDataFrame with separate rows for each time bin.
    """
    gdfs = []
    
    # Use provided time_bins or default to backward-compatible bins
    union_time_bins = time_bins or config.DEFAULT_TIME_BINS
    
    for t in union_time_bins:
        if t in polys_by_time and polys_by_time[t]:
            # Get polygons for this time bin
            time_polys = [p for p in polys_by_time[t] if p is not None and not p.is_empty]
            if time_polys:
                # Create union for this time bin
                union_geom = unary_union(time_polys)
                if union_geom and not union_geom.is_empty:
                    gdfs.append(
                        gpd.GeoDataFrame({"geometry": [union_geom], "Time": [f"{t} min"]}, crs="EPSG:4326")
                    )
    
    # Combine all time bins into a single GeoDataFrame
    if gdfs:
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    else:
        return gpd.GeoDataFrame({"geometry": [], "Time": []}, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Figure 1 – CT vs Stroke vs Thrombectomy
# ---------------------------------------------------------------------------

def create_figure_1(out_dir: Path | str | None = None, time_bins: list[int] | None = None) -> Path:
    """Create Figure 1: CT hospitals vs All Stroke Units vs Thrombectomy certified.
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data ------------------------------------------------------------------
    df_stroke = data.load_stroke_units()
    df_ct = data.load_hospitals_ct()

    uber_mask, reg_mask, thromb_mask = data.stroke_unit_masks(df_stroke)

    # Pre-computed polygons – reuse pickles (fast) --------------------------
    def _load_pkl(p: Path):  # local helper
        with open(p, "rb") as fh:
            return pickle.load(fh)

    # Determine which time bins to load polygons for
    load_time_bins = time_bins or config.DEFAULT_TIME_BINS
    
    # Load polygons for the specified time bins only
    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}
    ct_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_all_CTs.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_all_CTs.pkl").exists()}

    # Union bands (uses DEFAULT_TIME_BINS unless specified) ---------------
    stroke_union = _union(stroke_polys, time_bins)
    thromb_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if thromb_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)
    ct_union = _union(ct_polys, time_bins)

    germany = data.load_germany_outline()

    # Plot ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5), dpi=500)

    plotting.plot_isochrones_panel(
        axes[0], germany, ct_union, df_ct, title="Hospitals with CT", panel_letter="a", time_bins=time_bins
    )
    plotting.plot_isochrones_panel(
        axes[1], germany, stroke_union, df_stroke, title="All Stroke Units", panel_letter="b", time_bins=time_bins
    )
    df_thromb = df_stroke.loc[thromb_mask].copy()
    plotting.plot_isochrones_panel(
        axes[2], germany, thromb_union, df_thromb,
        title="Thrombectomy-Certified", panel_letter="c", time_bins=time_bins
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
        eps_path = out_dir / f"figure_1_CT_vs_stroke_thromb{suffix}.eps"
        tiff_path = out_dir / f"figure_1_CT_vs_stroke_thromb{suffix}.tiff"
        png_path = out_dir / f"figure_1_CT_vs_stroke_thromb{suffix}.png"
    else:
        eps_path = out_dir / "figure_1_CT_vs_stroke_thromb.eps"
        tiff_path = out_dir / "figure_1_CT_vs_stroke_thromb.tiff"
        png_path = out_dir / "figure_1_CT_vs_stroke_thromb.png"
        
    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', bbox_inches="tight", dpi=500)
    fig.savefig(tiff_path, format='tiff', bbox_inches="tight", dpi=500)
    fig.savefig(png_path, format='png', bbox_inches="tight", dpi=500)
    plt.close(fig)

    print(f"✅ Figure 1 saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path


def create_figure_1_extended(out_dir: Path | str | None = None, time_bins: list[int] | None = None) -> Path:
    """Create Figure 1 Extended: CT vs Extended Stroke vs All Stroke vs Thrombectomy.
    
    This is an alternate version of Figure 1 with a 2x2 layout:
    - A: Hospitals with CT (top left)
    - B: Frequent Stroke-Care Hospitals (extended stroke units, top right)  
    - C: All Stroke Units (bottom left)
    - D: Thrombectomy-Certified (bottom right)
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data ------------------------------------------------------------------
    df_stroke = data.load_stroke_units()
    df_ct = data.load_hospitals_ct()
    
    # Load extended stroke units
    try:
        df_extended_stroke = data.load_extended_stroke_units()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Extended stroke units not available: {e}\n"
            f"Please run additional_stroke_centers.ipynb to generate the extended dataset."
        )

    uber_mask, reg_mask, thromb_mask = data.stroke_unit_masks(df_stroke)

    # Pre-computed polygons – reuse pickles (fast) --------------------------
    def _load_pkl(p: Path):  # local helper
        with open(p, "rb") as fh:
            return pickle.load(fh)

    # Determine which time bins to load polygons for
    load_time_bins = time_bins or config.DEFAULT_TIME_BINS

    # Load polygons for the specified time bins only
    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}
    ct_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_all_CTs.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_all_CTs.pkl").exists()}
    extended_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_extended_stroke.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_extended_stroke.pkl").exists()}
    
    # Check if we have extended stroke polygons
    if not extended_polys:
        raise FileNotFoundError(
            "Extended stroke isochrones not found. Please generate isochrones for extended stroke units first.\n"
            "Expected files: poly{t}_extended_stroke.pkl in the DATA_DIR"
        )

    # Union bands (uses DEFAULT_TIME_BINS unless specified) ---------------
    stroke_union = _union(stroke_polys, time_bins)
    thromb_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if thromb_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)
    ct_union = _union(ct_polys, time_bins)
    extended_union = _union(extended_polys, time_bins)

    germany = data.load_germany_outline()

    # Plot ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=500)
    axes = axes.flatten()  # Make indexing easier

    # Panel A: Hospitals with CT (top left)
    plotting.plot_isochrones_panel(
        axes[0], germany, ct_union, df_ct, title="Hospitals with CT", panel_letter="a", time_bins=time_bins
    )
    
    # Panel B: Stroke-Ready Hospitals (extended stroke, top right)
    plotting.plot_isochrones_panel(
        axes[1], germany, extended_union, df_extended_stroke, title="Stroke-Ready Hospitals", panel_letter="b", time_bins=time_bins
    )
    
    # Panel C: All Stroke Units (bottom left)
    plotting.plot_isochrones_panel(
        axes[2], germany, stroke_union, df_stroke, title="All Stroke Units", panel_letter="c", time_bins=time_bins
    )
    
    # Panel D: Thrombectomy-Certified (bottom right)
    df_thromb = df_stroke.loc[thromb_mask].copy()
    plotting.plot_isochrones_panel(
        axes[3], germany, thromb_union, df_thromb,
        title="Thrombectomy-Certified", panel_letter="d", time_bins=time_bins
    )

    # Use appropriate legend based on time_bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        plotting.add_legend(fig, time_bins=time_bins)
    else:
        plotting.add_legend(fig)  # Default backward-compatible legend
        
    fig.tight_layout(rect=(0, 0.08, 1, 1))  # type: ignore[arg-type] Leave more space for legend

    # Adjust filename if using custom time bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        suffix = f"_custom_{'_'.join(map(str, time_bins))}"
        eps_path = out_dir / f"figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb{suffix}.eps"
        tiff_path = out_dir / f"figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb{suffix}.tiff"
        png_path = out_dir / f"figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb{suffix}.png"
    else:
        eps_path = out_dir / f"figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb.eps"
        tiff_path = out_dir / f"figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb.tiff"
        png_path = out_dir / f"figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb.png"
        
    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', bbox_inches="tight", dpi=500)
    fig.savefig(tiff_path, format='tiff', bbox_inches="tight", dpi=500)
    fig.savefig(png_path, format='png', bbox_inches="tight", dpi=500)
    plt.close(fig)

    print(f"✅ Figure 1 Extended saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Figure 2 – Stroke hierarchy split
# ---------------------------------------------------------------------------

def create_figure_2(out_dir: Path | str | None = None, time_bins: list[int] | None = None) -> Path:
    """Create Figure 2: Stroke units hierarchical split.
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_stroke = data.load_stroke_units()

    uber_mask, reg_mask, _ = data.stroke_unit_masks(df_stroke)

    def _load_pkl(p: Path):  # local helper
        with open(p, "rb") as fh:
            return pickle.load(fh)

    # Determine which time bins to load polygons for
    load_time_bins = time_bins or config.DEFAULT_TIME_BINS

    # Load polygons for the specified time bins only
    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}

    stroke_union = _union(stroke_polys, time_bins)
    uber_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if uber_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)
    reg_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if reg_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)

    germany = data.load_germany_outline()

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5), dpi=500)

    plotting.plot_isochrones_panel(
        axes[0], germany, stroke_union, df_stroke, title="All Stroke Units", panel_letter="a", time_bins=time_bins
    )
    df_uber = df_stroke.loc[uber_mask].copy()
    df_reg  = df_stroke.loc[reg_mask].copy()
    plotting.plot_isochrones_panel(
        axes[1], germany, uber_union, df_uber, title="Supra-Regional", panel_letter="b", time_bins=time_bins
    )
    plotting.plot_isochrones_panel(
        axes[2], germany, reg_union, df_reg, title="Regional / Telemed", panel_letter="c", time_bins=time_bins
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
        eps_path = out_dir / f"figure_2_stroke_hierarchy{suffix}.eps"
        tiff_path = out_dir / f"figure_2_stroke_hierarchy{suffix}.tiff"
        png_path = out_dir / f"figure_2_stroke_hierarchy{suffix}.png"
    else:
        eps_path = out_dir / "figure_2_stroke_hierarchy.eps"
        tiff_path = out_dir / "figure_2_stroke_hierarchy.tiff"
        png_path = out_dir / "figure_2_stroke_hierarchy.png"
        
    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', bbox_inches="tight", dpi=500)
    fig.savefig(tiff_path, format='tiff', bbox_inches="tight", dpi=500)
    fig.savefig(png_path, format='png', bbox_inches="tight", dpi=500)
    plt.close(fig)

    print(f"✅ Figure 2 saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Supplemental Figure – Custom time bins with easy interface
# ---------------------------------------------------------------------------

def create_supplemental_figure(
    time_bins: list[int],
    out_dir: Path | str | None = None,
    include_ct: bool = True,
    figure_type: str = "comparison"
) -> Path:
    """Create supplemental figures with custom time bins.
    
    Parameters
    ----------
    time_bins : list[int]
        Specific time bins to include (e.g., [10, 20, 25] for gap analysis).
    include_ct : bool, optional
        Whether to include CT hospitals comparison (default: True).
    figure_type : str, optional
        Type of figure: "comparison" (CT vs Stroke) or "hierarchy" (stroke hierarchy).
    
    Returns
    -------
    Path
        Path to the generated figure.
        
    Examples
    --------
    # Create gap analysis figure
    create_supplemental_figure([10, 15, 25, 30], figure_type="comparison")
    
    # Create detailed stroke hierarchy figure
    create_supplemental_figure([5, 10, 15, 20, 30, 45, 60], figure_type="hierarchy")
    """
    
    if figure_type == "comparison":
        return create_figure_1(out_dir=out_dir, time_bins=time_bins)
    elif figure_type == "hierarchy":
        return create_figure_2(out_dir=out_dir, time_bins=time_bins)
    else:
        raise ValueError(f"Unknown figure_type: {figure_type}. Use 'comparison' or 'hierarchy'.")


# ---------------------------------------------------------------------------
# Scenario-specific figures for emergency and bad traffic conditions
# ---------------------------------------------------------------------------

def create_scenario_figure_1(
    scenario: str,
    out_dir: Path | str | None = None, 
    time_bins: list[int] | None = None,
    force_recalc: bool = False
) -> Path:
    """Create Figure 1 for a specific scenario (emergency or bad traffic).
    
    Parameters
    ----------
    scenario : str
        Scenario to use. Options: 'emergency', 'bad_traffic', 'normal'.
    out_dir : Path | str | None, optional
        Output directory for figures.
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    force_recalc : bool, optional
        If True, regenerate isochrones even if cached.
        
    Returns
    -------
    Path
        Path to the generated figure.
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate scenario
    if scenario not in config.SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {list(config.SCENARIOS.keys())}")
    
    scenario_config = config.SCENARIOS[scenario]
    print(f"🎨 Generating Figure 1 for scenario: {scenario_config['description']}")

    # Data ------------------------------------------------------------------
    df_stroke = data.load_stroke_units()
    df_ct = data.load_hospitals_ct()

    uber_mask, reg_mask, thromb_mask = data.stroke_unit_masks(df_stroke)

    # Generate isochrones for the specific scenario ----------------------- 
    print("📊 Generating stroke unit isochrones...")
    stroke_polygons = iso_manager.ensure_polygons(
        df_stroke,
        force_recalc=force_recalc,
        scenario=scenario
    )
    
    print("🏥 Generating CT hospital isochrones...")
    ct_polygons = iso_manager.ensure_polygons(
        df_ct,
        force_recalc=force_recalc,
        suffix="_all_CTs",
        scenario=scenario
    )

    # Union bands (uses DEFAULT_TIME_BINS unless specified) ---------------
    stroke_union = _union(stroke_polygons, time_bins)
    thromb_union = _union({t: [p for i, p in enumerate(stroke_polygons[t]) if thromb_mask.iloc[i]] for t in stroke_polygons.keys()}, time_bins)
    ct_union = _union(ct_polygons, time_bins)

    germany = data.load_germany_outline()

    # Plot ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5), dpi=500)

    plotting.plot_isochrones_panel(
        axes[0], germany, ct_union, df_ct, title="Hospitals with CT", panel_letter="a", time_bins=time_bins
    )
    plotting.plot_isochrones_panel(
        axes[1], germany, stroke_union, df_stroke, title="All Stroke Units", panel_letter="b", time_bins=time_bins
    )
    df_thromb = df_stroke.loc[thromb_mask].copy()
    plotting.plot_isochrones_panel(
        axes[2], germany, thromb_union, df_thromb,
        title="Thrombectomy-Certified", panel_letter="c", time_bins=time_bins
    )

    # Use appropriate legend based on time_bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        plotting.add_legend(fig, time_bins=time_bins)
    else:
        plotting.add_legend(fig)  # Default backward-compatible legend
        
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # type: ignore[arg-type]

    # Create filename with scenario identifier
    scenario_suffix = scenario_config["suffix"] if scenario != "normal" else ""
    time_suffix = f"_custom_{'_'.join(map(str, time_bins))}" if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS) else ""
    
    eps_path = out_dir / f"figure_1_CT_vs_stroke_thromb{scenario_suffix}{time_suffix}.eps"
    tiff_path = out_dir / f"figure_1_CT_vs_stroke_thromb{scenario_suffix}{time_suffix}.tiff"
    png_path = out_dir / f"figure_1_CT_vs_stroke_thromb{scenario_suffix}{time_suffix}.png"
        
    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', bbox_inches="tight", dpi=500)
    fig.savefig(tiff_path, format='tiff', bbox_inches="tight", dpi=500)
    fig.savefig(png_path, format='png', bbox_inches="tight", dpi=500)
    plt.close(fig)

    print(f"✅ Figure 1 for {scenario} scenario saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path


def create_scenario_figure_2(
    scenario: str,
    out_dir: Path | str | None = None, 
    time_bins: list[int] | None = None,
    force_recalc: bool = False
) -> Path:
    """Create Figure 2 for a specific scenario (emergency or bad traffic).
    
    Parameters
    ----------
    scenario : str
        Scenario to use. Options: 'emergency', 'bad_traffic', 'normal'.
    out_dir : Path | str | None, optional
        Output directory for figures.
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    force_recalc : bool, optional
        If True, regenerate isochrones even if cached.
        
    Returns
    -------
    Path
        Path to the generated figure.
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate scenario
    if scenario not in config.SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {list(config.SCENARIOS.keys())}")
    
    scenario_config = config.SCENARIOS[scenario]
    print(f"🎨 Generating Figure 2 for scenario: {scenario_config['description']}")

    # Data ------------------------------------------------------------------
    df_stroke = data.load_stroke_units()
    uber_mask, reg_mask, _ = data.stroke_unit_masks(df_stroke)

    # Generate isochrones for the specific scenario -----------------------
    print("📊 Generating stroke unit isochrones...")
    stroke_polygons = iso_manager.ensure_polygons(
        df_stroke,
        force_recalc=force_recalc,
        scenario=scenario
    )

    # Union bands (uses DEFAULT_TIME_BINS unless specified) ---------------
    stroke_union = _union(stroke_polygons, time_bins)
    uber_union = _union({t: [p for i, p in enumerate(stroke_polygons[t]) if uber_mask.iloc[i]] for t in stroke_polygons.keys()}, time_bins)
    reg_union = _union({t: [p for i, p in enumerate(stroke_polygons[t]) if reg_mask.iloc[i]] for t in stroke_polygons.keys()}, time_bins)

    germany = data.load_germany_outline()

    # Plot ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5), dpi=500)

    plotting.plot_isochrones_panel(
        axes[0], germany, stroke_union, df_stroke, title="All Stroke Units", panel_letter="a", time_bins=time_bins
    )
    df_uber = df_stroke.loc[uber_mask].copy()
    df_reg  = df_stroke.loc[reg_mask].copy()
    plotting.plot_isochrones_panel(
        axes[1], germany, uber_union, df_uber, title="Supra-Regional", panel_letter="b", time_bins=time_bins
    )
    plotting.plot_isochrones_panel(
        axes[2], germany, reg_union, df_reg, title="Regional / Telemed", panel_letter="c", time_bins=time_bins
    )

    # Use appropriate legend based on time_bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        plotting.add_legend(fig, time_bins=time_bins)
    else:
        plotting.add_legend(fig)  # Default backward-compatible legend
        
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # type: ignore[arg-type]

    # Create filename with scenario identifier
    scenario_suffix = scenario_config["suffix"] if scenario != "normal" else ""
    time_suffix = f"_custom_{'_'.join(map(str, time_bins))}" if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS) else ""
    
    eps_path = out_dir / f"figure_2_stroke_hierarchy{scenario_suffix}{time_suffix}.eps"
    tiff_path = out_dir / f"figure_2_stroke_hierarchy{scenario_suffix}{time_suffix}.tiff"
    png_path = out_dir / f"figure_2_stroke_hierarchy{scenario_suffix}{time_suffix}.png"
        
    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', bbox_inches="tight", dpi=500)
    fig.savefig(tiff_path, format='tiff', bbox_inches="tight", dpi=500)
    fig.savefig(png_path, format='png', bbox_inches="tight", dpi=500)
    plt.close(fig)

    print(f"✅ Figure 2 for {scenario} scenario saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path


def generate_all_scenario_figures(
    scenarios: list[str] | None = None,
    out_dir: Path | str | None = None,
    time_bins: list[int] | None = None,
    force_recalc: bool = False,
    include_normal: bool = False
) -> dict[str, dict[str, Path]]:
    """Generate both Figure 1 and Figure 2 for all specified scenarios.
    
    Parameters
    ----------
    scenarios : list[str], optional
        List of scenarios to generate. If None, generates 'emergency' and 'bad_traffic' only.
    out_dir : Path | str | None, optional
        Output directory for figures.
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    force_recalc : bool, optional
        If True, regenerate isochrones even if cached.
    include_normal : bool, optional
        If True, also generate figures for the normal scenario.
        
    Returns
    -------
    dict
        Nested dictionary with structure: {scenario: {'figure_1': path, 'figure_2': path}}
        
    Examples
    --------
    # Generate supplemental figures for manuscript
    paths = generate_all_scenario_figures()
    
    # Generate all scenarios including normal
    paths = generate_all_scenario_figures(include_normal=True)
    
    # Generate specific scenarios
    paths = generate_all_scenario_figures(['emergency'])
    """
    
    if scenarios is None:
        scenarios = ['emergency', 'bad_traffic']
        if include_normal:
            scenarios.append('normal')
    
    print("🎨 Generating scenario-specific figures for supplemental material...")
    print(f"   Scenarios: {scenarios}")
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n🔄 Processing scenario: {scenario}")
        
        try:
            # Generate Figure 1 for this scenario
            fig1_path = create_scenario_figure_1(
                scenario=scenario,
                out_dir=out_dir,
                time_bins=time_bins,
                force_recalc=force_recalc
            )
            
            # Generate Figure 2 for this scenario  
            fig2_path = create_scenario_figure_2(
                scenario=scenario,
                out_dir=out_dir,
                time_bins=time_bins,
                force_recalc=force_recalc
            )
            
            results[scenario] = {
                'figure_1': fig1_path,
                'figure_2': fig2_path
            }
            
            print(f"✅ Completed figures for scenario: {scenario}")
            
        except Exception as e:
            print(f"❌ Failed to generate figures for scenario {scenario}: {e}")
            results[scenario] = {'figure_1': None, 'figure_2': None}
    
    print(f"\n🎉 Scenario figure generation complete!")
    print(f"   Generated figures for {len([s for s in results if results[s]['figure_1']])} scenarios")
    
    return results 

# ---------------------------------------------------------------------------
# Journal-compliant figures (190mm width) - Keep originals as fallback
# ---------------------------------------------------------------------------

def create_journal_figure_1(out_dir: Path | str | None = None, time_bins: list[int] | None = None) -> Path:
    """Create journal-compliant Figure 1: CT hospitals vs All Stroke Units vs Thrombectomy certified.
    
    Uses 190mm width (7.48 inches) with optimized layout for journal submission.
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data ------------------------------------------------------------------
    df_stroke = data.load_stroke_units()
    df_ct = data.load_hospitals_ct()

    uber_mask, reg_mask, thromb_mask = data.stroke_unit_masks(df_stroke)

    # Pre-computed polygons – reuse pickles (fast) --------------------------
    def _load_pkl(p: Path):  # local helper
        with open(p, "rb") as fh:
            return pickle.load(fh)

    # Determine which time bins to load polygons for
    load_time_bins = time_bins or config.DEFAULT_TIME_BINS
    
    # Load polygons for the specified time bins only
    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}
    ct_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_all_CTs.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_all_CTs.pkl").exists()}

    # Union bands (uses DEFAULT_TIME_BINS unless specified) ---------------
    stroke_union = _union(stroke_polys, time_bins)
    thromb_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if thromb_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)
    ct_union = _union(ct_polys, time_bins)

    germany = data.load_germany_outline()

    # Plot ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    # Journal-compliant figure size: 190mm width (7.48 inches), proportional height
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 3.48), dpi=500)

    plotting.plot_journal_isochrones_panel(
        axes[0], germany, ct_union, df_ct, title="Hospitals with CT", panel_letter="a", time_bins=time_bins
    )
    plotting.plot_journal_isochrones_panel(
        axes[1], germany, stroke_union, df_stroke, title="All Stroke Units", panel_letter="b", time_bins=time_bins
    )
    df_thromb = df_stroke.loc[thromb_mask].copy()
    plotting.plot_journal_isochrones_panel(
        axes[2], germany, thromb_union, df_thromb,
        title="Thrombectomy-Certified", panel_letter="c", time_bins=time_bins
    )

    # Use appropriate legend based on time_bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        plotting.add_journal_legend(fig, time_bins=time_bins)
    else:
        plotting.add_journal_legend(fig)  # Default backward-compatible legend
        
    fig.tight_layout(rect=(0, 0.08, 1, 1))  # Optimized for journal layout

    # Adjust filename if using custom time bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        suffix = f"_custom_{'_'.join(map(str, time_bins))}"
        eps_path = out_dir / f"journal_figure_1_CT_vs_stroke_thromb{suffix}.eps"
        tiff_path = out_dir / f"journal_figure_1_CT_vs_stroke_thromb{suffix}.tiff"
        png_path = out_dir / f"journal_figure_1_CT_vs_stroke_thromb{suffix}.png"
    else:
        eps_path = out_dir / "journal_figure_1_CT_vs_stroke_thromb.eps"
        tiff_path = out_dir / "journal_figure_1_CT_vs_stroke_thromb.tiff"
        png_path = out_dir / "journal_figure_1_CT_vs_stroke_thromb.png"
        
    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', bbox_inches="tight", dpi=500)
    fig.savefig(tiff_path, format='tiff', bbox_inches="tight", dpi=500)
    fig.savefig(png_path, format='png', bbox_inches="tight", dpi=500)
    plt.close(fig)

    print(f"✅ Journal Figure 1 saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path


def create_journal_figure_1_extended(out_dir: Path | str | None = None, time_bins: list[int] | None = None) -> Path:
    """Create journal-compliant Figure 1 Extended: CT vs Extended Stroke vs All Stroke vs Thrombectomy.
    
    Uses 190mm width (7.48 inches) with optimized 2x2 layout for journal submission.
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data ------------------------------------------------------------------
    df_stroke = data.load_stroke_units()
    df_ct = data.load_hospitals_ct()
    
    # Load extended stroke units
    try:
        df_extended_stroke = data.load_extended_stroke_units()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Extended stroke units not available: {e}\n"
            f"Please run additional_stroke_centers.ipynb to generate the extended dataset."
        )

    uber_mask, reg_mask, thromb_mask = data.stroke_unit_masks(df_stroke)

    # Pre-computed polygons – reuse pickles (fast) --------------------------
    def _load_pkl(p: Path):  # local helper
        with open(p, "rb") as fh:
            return pickle.load(fh)

    # Determine which time bins to load polygons for
    load_time_bins = time_bins or config.DEFAULT_TIME_BINS

    # Load polygons for the specified time bins only
    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}
    ct_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_all_CTs.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_all_CTs.pkl").exists()}
    extended_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_extended_stroke.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_extended_stroke.pkl").exists()}
    
    # Check if we have extended stroke polygons
    if not extended_polys:
        raise FileNotFoundError(
            "Extended stroke isochrones not found. Please generate isochrones for extended stroke units first.\n"
            "Expected files: poly{t}_extended_stroke.pkl in the DATA_DIR"
        )

    # Union bands (uses DEFAULT_TIME_BINS unless specified) ---------------
    stroke_union = _union(stroke_polys, time_bins)
    thromb_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if thromb_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)
    ct_union = _union(ct_polys, time_bins)
    extended_union = _union(extended_polys, time_bins)

    germany = data.load_germany_outline()

    # Plot ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    # Journal-compliant figure size: 190mm width (7.48 inches), 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.62), dpi=500)
    axes = axes.flatten()  # Make indexing easier

    # Panel A: Hospitals with CT (top left)
    plotting.plot_journal_isochrones_panel(
        axes[0], germany, ct_union, df_ct, title="Hospitals with CT", panel_letter="a", time_bins=time_bins
    )
    
    # Panel B: Stroke-Ready Hospitals (extended stroke, top right)
    plotting.plot_journal_isochrones_panel(
        axes[1], germany, extended_union, df_extended_stroke, title="Stroke-Ready Hospitals", panel_letter="b", time_bins=time_bins
    )
    
    # Panel C: All Stroke Units (bottom left)
    plotting.plot_journal_isochrones_panel(
        axes[2], germany, stroke_union, df_stroke, title="All Stroke Units", panel_letter="c", time_bins=time_bins
    )
    
    # Panel D: Thrombectomy-Certified (bottom right)
    df_thromb = df_stroke.loc[thromb_mask].copy()
    plotting.plot_journal_isochrones_panel(
        axes[3], germany, thromb_union, df_thromb,
        title="Thrombectomy-Certified", panel_letter="d", time_bins=time_bins
    )

    # Use appropriate legend based on time_bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        plotting.add_journal_legend(fig, time_bins=time_bins)
    else:
        plotting.add_journal_legend(fig)  # Default backward-compatible legend
        
    fig.tight_layout(rect=(0, 0.1, 1, 1))  # More space for legend in 2x2 layout

    # Adjust filename if using custom time bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        suffix = f"_custom_{'_'.join(map(str, time_bins))}"
        eps_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb{suffix}.eps"
        tiff_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb{suffix}.tiff"
        png_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb{suffix}.png"
    else:
        eps_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb.eps"
        tiff_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb.tiff"
        png_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb.png"
        
    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', bbox_inches="tight", dpi=500)
    fig.savefig(tiff_path, format='tiff', bbox_inches="tight", dpi=500)
    fig.savefig(png_path, format='png', bbox_inches="tight", dpi=500)
    plt.close(fig)

    print(f"✅ Journal Figure 1 Extended saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path


def create_journal_figure_2(out_dir: Path | str | None = None, time_bins: list[int] | None = None) -> Path:
    """Create journal-compliant Figure 2: Stroke units hierarchical split.
    
    Uses 190mm width (7.48 inches) with optimized layout for journal submission.
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_stroke = data.load_stroke_units()

    uber_mask, reg_mask, _ = data.stroke_unit_masks(df_stroke)

    def _load_pkl(p: Path):  # local helper
        with open(p, "rb") as fh:
            return pickle.load(fh)

    # Determine which time bins to load polygons for
    load_time_bins = time_bins or config.DEFAULT_TIME_BINS

    # Load polygons for the specified time bins only
    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}

    stroke_union = _union(stroke_polys, time_bins)
    uber_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if uber_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)
    reg_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if reg_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)

    germany = data.load_germany_outline()

    import matplotlib.pyplot as plt

    # Journal-compliant figure size: 190mm width (7.48 inches), proportional height
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 3.48), dpi=500)

    plotting.plot_journal_isochrones_panel(
        axes[0], germany, stroke_union, df_stroke, title="All Stroke Units", panel_letter="a", time_bins=time_bins
    )
    df_uber = df_stroke.loc[uber_mask].copy()
    df_reg  = df_stroke.loc[reg_mask].copy()
    plotting.plot_journal_isochrones_panel(
        axes[1], germany, uber_union, df_uber, title="Supra-Regional", panel_letter="b", time_bins=time_bins
    )
    plotting.plot_journal_isochrones_panel(
        axes[2], germany, reg_union, df_reg, title="Regional / Telemed", panel_letter="c", time_bins=time_bins
    )

    # Use appropriate legend based on time_bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        plotting.add_journal_legend(fig, time_bins=time_bins)
    else:
        plotting.add_journal_legend(fig)  # Default backward-compatible legend
        
    fig.tight_layout(rect=(0, 0.08, 1, 1))  # Optimized for journal layout

    # Adjust filename if using custom time bins
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        suffix = f"_custom_{'_'.join(map(str, time_bins))}"
        eps_path = out_dir / f"journal_figure_2_stroke_hierarchy{suffix}.eps"
        tiff_path = out_dir / f"journal_figure_2_stroke_hierarchy{suffix}.tiff"
        png_path = out_dir / f"journal_figure_2_stroke_hierarchy{suffix}.png"
    else:
        eps_path = out_dir / "journal_figure_2_stroke_hierarchy.eps"
        tiff_path = out_dir / "journal_figure_2_stroke_hierarchy.tiff"
        png_path = out_dir / "journal_figure_2_stroke_hierarchy.png"
        
    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', bbox_inches="tight", dpi=500)
    fig.savefig(tiff_path, format='tiff', bbox_inches="tight", dpi=500)
    fig.savefig(png_path, format='png', bbox_inches="tight", dpi=500)
    plt.close(fig)

    print(f"✅ Journal Figure 2 saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path 

def run_journal_publication_figures(
    out_dir: Path | str | None = None,
    time_bins: list[int] | None = None,
    force_recalc: bool = False
) -> dict[str, dict[str, Path]]:
    """Generate all journal-compliant figures for a publication.
    
    This function orchestrates the generation of all journal-compliant
    figures (Figure 1, Figure 1 Extended, Figure 2) for a given scenario.
    
    Parameters
    ----------
    out_dir : Path | str | None, optional
        Output directory for figures.
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    force_recalc : bool, optional
        If True, regenerate isochrones even if cached.
        
    Returns
    -------
    dict
        Nested dictionary with structure: {scenario: {'figure_1': path, 'figure_1_extended': path, 'figure_2': path}}
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🎨 Generating all journal-compliant figures for publication...")
    
    results = {}
    
    # Generate Figure 1 (Journal)
    print("\n🔄 Generating Journal Figure 1...")
    try:
        journal_fig1_path = create_journal_figure_1(out_dir=out_dir, time_bins=time_bins)
        results["Journal Figure 1"] = {"figure_1": journal_fig1_path}
        print(f"✅ Journal Figure 1 saved: {journal_fig1_path}")
    except Exception as e:
        print(f"❌ Failed to generate Journal Figure 1: {e}")
        results["Journal Figure 1"] = {"figure_1": None}

    # Generate Figure 1 Extended (Journal)
    print("\n🔄 Generating Journal Figure 1 Extended...")
    try:
        journal_fig1_extended_path = create_journal_figure_1_extended(out_dir=out_dir, time_bins=time_bins)
        results["Journal Figure 1 Extended"] = {"figure_1_extended": journal_fig1_extended_path}
        print(f"✅ Journal Figure 1 Extended saved: {journal_fig1_extended_path}")
    except Exception as e:
        print(f"❌ Failed to generate Journal Figure 1 Extended: {e}")
        results["Journal Figure 1 Extended"] = {"figure_1_extended": None}

    # Generate Figure 2 (Journal)
    print("\n🔄 Generating Journal Figure 2...")
    try:
        journal_fig2_path = create_journal_figure_2(out_dir=out_dir, time_bins=time_bins)
        results["Journal Figure 2"] = {"figure_2": journal_fig2_path}
        print(f"✅ Journal Figure 2 saved: {journal_fig2_path}")
    except Exception as e:
        print(f"❌ Failed to generate Journal Figure 2: {e}")
        results["Journal Figure 2"] = {"figure_2": None}

    print(f"\n🎉 Journal publication figure generation complete!")
    print(f"   Generated figures for {len([s for s in results if results[s]['figure_1']])} journal figures")
    
    return results 

def create_journal_figure_1_standardized(out_dir: Path | str | None = None, time_bins: list[int] | None = None) -> Path:
    """Create standardized journal-compliant Figure 1 with three_penalty_scenarios layout.
    
    Uses the same layout specifications as three_penalty_scenarios:
    - figsize: (4.21, 11)
    - Export DPI: 600
    - Headings: Times New Roman, 10 pt, bold
    - Panel letters: lowercase with parentheses
    - Vertical 3x1 layout
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data ------------------------------------------------------------------
    df_stroke = data.load_stroke_units()
    df_ct = data.load_hospitals_ct()

    uber_mask, reg_mask, thromb_mask = data.stroke_unit_masks(df_stroke)

    # Pre-computed polygons – reuse pickles (fast) --------------------------
    def _load_pkl(p: Path):  # local helper
        with open(p, "rb") as fh:
            return pickle.load(fh)

    # Determine which time bins to load polygons for
    load_time_bins = time_bins or config.DEFAULT_TIME_BINS
    
    # Load polygons for the specified time bins only
    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}
    ct_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_all_CTs.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_all_CTs.pkl").exists()}

    # Union bands (uses DEFAULT_TIME_BINS unless specified) ---------------
    stroke_union = _union(stroke_polys, time_bins)
    thromb_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if thromb_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)
    ct_union = _union(ct_polys, time_bins)

    germany = data.load_germany_outline()

    # Plot ------------------------------------------------------------------
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    # Standardized figure size matching three_penalty_scenarios: vertical 3x1 layout
    fig, axes = plt.subplots(3, 1, figsize=(4.21, 11), dpi=600)

    # Define panel info
    panels = [
        ("Hospitals with CT", "a", ct_union, df_ct),
        ("All Stroke Units", "b", stroke_union, df_stroke),
        ("Thrombectomy-\nCertified", "c", thromb_union, df_stroke.loc[thromb_mask].copy())
    ]

    for i, (title, panel_letter, union_data, df_points) in enumerate(panels):
        ax = axes[i]
        
        # Setup axis without frame (matching three_penalty_scenarios)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Plot Germany outline
        germany.boundary.plot(ax=ax, color='black', linewidth=0.6, zorder=3)
        
        # Plot isochrone bands – biggest first so smallest stays on top
        plot_time_bins = time_bins or config.DEFAULT_TIME_BINS
        for t in sorted(plot_time_bins, reverse=True):
            subset = union_data[union_data["Time"] == f"{t} min"]
            if not subset.empty:
                subset.plot(
                    ax=ax,
                    color=config.TIME_COLOURS[t],
                    alpha=0.9,
                    edgecolor="none",  # remove polygon outlines for clean journal style
                    linewidth=0.0,
                    zorder=2,
                )

        # Facility scatter (extra small)
        ax.scatter(
            df_points["longitude"],
            df_points["latitude"],
            color="black",
            s=0.2,
            alpha=0.8,
            marker="o",
            zorder=4,
        )
        
        # Set bounds with margins (matching three_penalty_scenarios)
        ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
        ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)
        
        # Title (Times New Roman 10pt bold)
        ax.set_title(title, fontsize=10, fontweight='bold', family='Times New Roman', pad=10)
        
        # Panel letter (Times New Roman 10pt bold) - no parentheses, left of title
        ax.text(0.01, 1.06, panel_letter, transform=ax.transAxes,
                fontweight='bold', fontsize=10, family='Times New Roman', va='top', ha='left')

    # Legend (matching three_penalty_scenarios style)
    legend_time_bins = time_bins or config.DEFAULT_TIME_BINS
    handles = [
        mpatches.Patch(color=config.TIME_COLOURS[t], label=f"{t} min")
        for t in legend_time_bins if t in config.TIME_COLOURS
    ]
    handles.append(
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=2.5,
            color="black",
            label="Facility",
        )
    )
    
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=1,
        fontsize=10,
        frameon=False,
        prop={'family': 'Times New Roman', 'size': 10}
    )
    
    # Layout adjustments (matching three_penalty_scenarios)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Adjust filename for standardized version
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        suffix = f"_custom_{'_'.join(map(str, time_bins))}"
        eps_path = out_dir / f"journal_figure_1_CT_vs_stroke_thromb_standardized{suffix}.eps"
        tiff_path = out_dir / f"journal_figure_1_CT_vs_stroke_thromb_standardized{suffix}.tiff"
        png_path = out_dir / f"journal_figure_1_CT_vs_stroke_thromb_standardized{suffix}.png"
    else:
        eps_path = out_dir / f"journal_figure_1_CT_vs_stroke_thromb_standardized.eps"
        tiff_path = out_dir / f"journal_figure_1_CT_vs_stroke_thromb_standardized.tiff"
        png_path = out_dir / f"journal_figure_1_CT_vs_stroke_thromb_standardized.png"
        
    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', bbox_inches="tight", dpi=600)
    fig.savefig(tiff_path, format='tiff', bbox_inches="tight", dpi=600)
    fig.savefig(png_path, format='png', bbox_inches="tight", dpi=600)
    plt.close(fig)

    print(f"✅ Standardized Journal Figure 1 saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path


def create_journal_figure_2_standardized(out_dir: Path | str | None = None, time_bins: list[int] | None = None) -> Path:
    """Create standardized journal-compliant Figure 2 with three_penalty_scenarios layout.
    
    Uses the same layout specifications as three_penalty_scenarios:
    - figsize: (4.21, 11)
    - Export DPI: 600
    - Headings: Times New Roman, 10 pt, bold
    - Panel letters: lowercase with parentheses
    - Vertical 3x1 layout
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_stroke = data.load_stroke_units()

    uber_mask, reg_mask, _ = data.stroke_unit_masks(df_stroke)

    def _load_pkl(p: Path):  # local helper
        with open(p, "rb") as fh:
            return pickle.load(fh)

    # Determine which time bins to load polygons for
    load_time_bins = time_bins or config.DEFAULT_TIME_BINS

    # Load polygons for the specified time bins only
    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}

    stroke_union = _union(stroke_polys, time_bins)
    uber_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if uber_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)
    reg_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if reg_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)

    germany = data.load_germany_outline()

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    # Standardized figure size matching three_penalty_scenarios: vertical 3x1 layout
    fig, axes = plt.subplots(3, 1, figsize=(4.21, 11), dpi=600)

    # Define panel info
    panels = [
        ("All Stroke Units", "a", stroke_union, df_stroke),
        ("Supra-Regional", "b", uber_union, df_stroke.loc[uber_mask].copy()),
        ("Regional / Telemed", "c", reg_union, df_stroke.loc[reg_mask].copy())
    ]

    for i, (title, panel_letter, union_data, df_points) in enumerate(panels):
        ax = axes[i]
        
        # Setup axis without frame (matching three_penalty_scenarios)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Plot Germany outline
        germany.boundary.plot(ax=ax, color='black', linewidth=0.6, zorder=3)
        
        # Plot isochrone bands – biggest first so smallest stays on top
        plot_time_bins = time_bins or config.DEFAULT_TIME_BINS
        for t in sorted(plot_time_bins, reverse=True):
            subset = union_data[union_data["Time"] == f"{t} min"]
            if not subset.empty:
                subset.plot(
                    ax=ax,
                    color=config.TIME_COLOURS[t],
                    alpha=0.9,
                    edgecolor="none",  # remove polygon outlines for clean journal style
                    linewidth=0.0,
                    zorder=2,
                )

        # Facility scatter (extra small)
        ax.scatter(
            df_points["longitude"],
            df_points["latitude"],
            color="black",
            s=0.2,
            alpha=0.8,
            marker="o",
            zorder=4,
        )
        
        # Set bounds with margins (matching three_penalty_scenarios)
        ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
        ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)
        
        # Title (Times New Roman 10pt bold)
        ax.set_title(title, fontsize=10, fontweight='bold', family='Times New Roman', pad=10)
        
        # Panel letter (Times New Roman 10pt bold) - no parentheses, left of title
        ax.text(0.01, 1.06, panel_letter, transform=ax.transAxes,
                fontweight='bold', fontsize=10, family='Times New Roman', va='top', ha='left')

    # Legend (matching three_penalty_scenarios style)
    legend_time_bins = time_bins or config.DEFAULT_TIME_BINS
    handles = [
        mpatches.Patch(color=config.TIME_COLOURS[t], label=f"{t} min")
        for t in legend_time_bins if t in config.TIME_COLOURS
    ]
    handles.append(
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=2.5,
            color="black",
            label="Facility",
        )
    )
    
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=1,
        fontsize=10,
        frameon=False,
        prop={'family': 'Times New Roman', 'size': 10}
    )
    
    # Layout adjustments (matching three_penalty_scenarios)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Adjust filename for standardized version
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        suffix = f"_custom_{'_'.join(map(str, time_bins))}"
        eps_path = out_dir / f"journal_figure_2_stroke_hierarchy_standardized{suffix}.eps"
        tiff_path = out_dir / f"journal_figure_2_stroke_hierarchy_standardized{suffix}.tiff"
        png_path = out_dir / f"journal_figure_2_stroke_hierarchy_standardized{suffix}.png"
    else:
        eps_path = out_dir / f"journal_figure_2_stroke_hierarchy_standardized.eps"
        tiff_path = out_dir / f"journal_figure_2_stroke_hierarchy_standardized.tiff"
        png_path = out_dir / f"journal_figure_2_stroke_hierarchy_standardized.png"
        
    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', bbox_inches="tight", dpi=600)
    fig.savefig(tiff_path, format='tiff', bbox_inches="tight", dpi=600)
    fig.savefig(png_path, format='png', bbox_inches="tight", dpi=600)
    plt.close(fig)

    print(f"✅ Standardized Journal Figure 2 saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path


def create_journal_figure_1_extended_standardized(out_dir: Path | str | None = None, time_bins: list[int] | None = None) -> Path:
    """Create standardized journal-compliant Figure 1 Extended with consistent styling.
    
    Updates the 4-panel figure to use consistent heading and font styling:
    - Maintains 2x2 layout but updates styling to match standardized format
    - Panel letters: lowercase with parentheses  
    - Consistent font styling with other standardized figures
    - Max width: 107 mm (4.21 inches)
    - Export DPI: 600
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS [15, 30, 45, 60].
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data ------------------------------------------------------------------
    df_stroke = data.load_stroke_units()
    df_ct = data.load_hospitals_ct()
    
    # Load extended stroke units
    try:
        df_extended_stroke = data.load_extended_stroke_units()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Extended stroke units not available: {e}\n"
            f"Please run additional_stroke_centers.ipynb to generate the extended dataset."
        )

    uber_mask, reg_mask, thromb_mask = data.stroke_unit_masks(df_stroke)

    # Pre-computed polygons – reuse pickles (fast) --------------------------
    def _load_pkl(p: Path):  # local helper
        with open(p, "rb") as fh:
            return pickle.load(fh)

    # Determine which time bins to load polygons for
    load_time_bins = time_bins or config.DEFAULT_TIME_BINS

    # Load polygons for the specified time bins only
    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}
    ct_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_all_CTs.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_all_CTs.pkl").exists()}
    extended_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_extended_stroke.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_extended_stroke.pkl").exists()}
    
    # Check if we have extended stroke polygons
    if not extended_polys:
        raise FileNotFoundError(
            "Extended stroke isochrones not found. Please generate isochrones for extended stroke units first.\n"
            "Expected files: poly{t}_extended_stroke.pkl in the DATA_DIR"
        )

    # Union bands (uses DEFAULT_TIME_BINS unless specified) ---------------
    stroke_union = _union(stroke_polys, time_bins)
    thromb_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if thromb_mask.iloc[i]] for t in stroke_polys.keys()}, time_bins)
    ct_union = _union(ct_polys, time_bins)
    extended_union = _union(extended_polys, time_bins)

    germany = data.load_germany_outline()

    # Plot ------------------------------------------------------------------
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    # Maintain 2x2 layout; scale to 107mm width (4.21 inches). Height scaled proportionally (~3.16 inches)
    # Tight 2x2 layout: reduce inter-axes spacing to enlarge maps
    fig, axes = plt.subplots(2, 2, figsize=(4.21, 5), dpi=600)
    plt.subplots_adjust(wspace=0.0, hspace=0.06)
    axes = axes.flatten()  # Make indexing easier

    # Define panel info
    panels = [
        ("Hospitals \n with CT", "a", ct_union, df_ct),
        ("Stroke-Ready \n Hospitals", "b", extended_union, df_extended_stroke),
        ("All \n Stroke Units", "c", stroke_union, df_stroke),
        ("Thrombectomy-\nCertified", "d", thromb_union, df_stroke.loc[thromb_mask].copy())
    ]

    for i, (title, panel_letter, union_data, df_points) in enumerate(panels):
        ax = axes[i]
        
        # Setup axis without frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Plot Germany outline
        germany.boundary.plot(ax=ax, color='black', linewidth=0.4, zorder=3)  # Thinner for smaller panels
        
        # Plot isochrone bands – biggest first so smallest stays on top
        plot_time_bins = time_bins or config.DEFAULT_TIME_BINS
        for t in sorted(plot_time_bins, reverse=True):
            subset = union_data[union_data["Time"] == f"{t} min"]
            if not subset.empty:
                subset.plot(
                    ax=ax,
                    color=config.TIME_COLOURS[t],
                    alpha=0.9,
                    edgecolor="none",  # remove polygon outlines for clean journal style
                    linewidth=0.0,
                    zorder=2,
                )

        # Facility scatter (slightly smaller)
        ax.scatter(
            df_points["longitude"],
            df_points["latitude"],
            color="black",
            s=0.15,
            alpha=1,
            marker="o",
            zorder=4,
        )
        
        # Set bounds with margins
        ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
        ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)
        
        # Title (Times New Roman 10pt bold)
        ax.set_title(title, fontsize=10, fontweight='bold', family='Times New Roman', pad=5)  # Reduced pad for four-panel layout
        
        # Panel letter (Times New Roman 10pt bold) - no parentheses, left of title
        ax.text(-0.3, 1.3, panel_letter, transform=ax.transAxes,
                fontweight='bold', fontsize=10, family='Times New Roman', va='top', ha='left')

    # Legend (standardized style) – two rows to maximize map area
    legend_time_bins = time_bins or config.DEFAULT_TIME_BINS
    handles = [
        mpatches.Patch(color=config.TIME_COLOURS[t], label=f"{t} min")
        for t in legend_time_bins if t in config.TIME_COLOURS
    ]
    handles.append(
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=2.5,
            color="black",
            label="Facility",
        )
    )
    
    # Arrange legend in two rows; split into roughly half per row + facility
    ncols = max(3, (len(legend_time_bins) + 1 + 1) // 2)
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.04),
        ncol=ncols,
        fontsize=10,
        frameon=False,
        prop={'family': 'Times New Roman', 'size': 10}
    )

    # Tighten layout to free vertical space for maps
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, wspace=0.0, hspace=0.08)

    # Adjust filename for standardized version
    if time_bins and set(time_bins) != set(config.DEFAULT_TIME_BINS):
        suffix = f"_custom_{'_'.join(map(str, time_bins))}"
        eps_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb_standardized{suffix}.eps"
        tiff_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb_standardized{suffix}.tiff"
        png_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb_standardized{suffix}.png"
    else:
        eps_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb_standardized.eps"
        tiff_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb_standardized.tiff"
        png_path = out_dir / f"journal_figure_1_extended_CT_vs_extended_stroke_vs_stroke_thromb_standardized.png"
        
    # Save in journal formats: EPS, TIFF, PNG
    fig.savefig(eps_path, format='eps', bbox_inches="tight", dpi=600)
    fig.savefig(tiff_path, format='tiff', bbox_inches="tight", dpi=600)
    fig.savefig(png_path, format='png', bbox_inches="tight", dpi=600)
    plt.close(fig)

    print(f"✅ Standardized Journal Figure 1 Extended saved: {eps_path}, {tiff_path}, and {png_path}")
    return png_path 


def create_journal_figure_1_extended_standardized_allbins_continuous(
    out_dir: Path | str | None = None,
    time_bins: list[int] | None = None,
) -> Path:
    """Create an additional 2x2 extended figure using ALL time bins (5–60 min)
    and a continuous colorbar legend suitable for journal submission.

    - Layout: 2x2, width 107 mm (4.21 in), height ~3.16 in
    - Headings: Times New Roman 10 pt bold
    - Facility markers: extra small
    - Isochrone polygons: no edges, alpha=0.9
    - Legend: horizontal continuous colorbar with small height
    - Export DPI: 600
    """

    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data ------------------------------------------------------------------
    df_stroke = data.load_stroke_units()
    df_ct = data.load_hospitals_ct()

    try:
        df_extended_stroke = data.load_extended_stroke_units()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Extended stroke units not available: {e}\n"
            f"Please run additional_stroke_centers.ipynb to generate the extended dataset."
        )

    uber_mask, reg_mask, thromb_mask = data.stroke_unit_masks(df_stroke)

    def _load_pkl(p: Path):
        with open(p, "rb") as fh:
            return pickle.load(fh)

    # Use ALL time bins for this variant
    load_time_bins = config.TIME_BINS

    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}
    ct_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_all_CTs.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_all_CTs.pkl").exists()}
    extended_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_extended_stroke.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_extended_stroke.pkl").exists()}

    if not extended_polys:
        raise FileNotFoundError(
            "Extended stroke isochrones not found. Please generate isochrones for extended stroke units first.\n"
            "Expected files: poly{t}_extended_stroke.pkl in the DATA_DIR"
        )

    stroke_union = _union(stroke_polys, load_time_bins)
    thromb_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if thromb_mask.iloc[i]] for t in stroke_polys.keys()}, load_time_bins)
    ct_union = _union(ct_polys, load_time_bins)
    extended_union = _union(extended_polys, load_time_bins)

    germany = data.load_germany_outline()

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.lines as mlines

    fig, axes = plt.subplots(2, 2, figsize=(4.21, 6), dpi=600)
    axes = axes.flatten()

    panels = [
        ("Hospitals \n with CT", "a", ct_union, df_ct),
        ("Stroke-Ready \n Hospitals", "b", extended_union, df_extended_stroke),
        ("All \n Stroke Units", "c", stroke_union, df_stroke),
        ("Thrombectomy-\nCertified", "d", thromb_union, df_stroke.loc[thromb_mask].copy()),
    ]

    for i, (title, panel_letter, union_data, df_points) in enumerate(panels):
        ax = axes[i]
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)

        germany.boundary.plot(ax=ax, color='black', linewidth=0.35, zorder=3)

        for t in sorted(load_time_bins, reverse=True):
            subset = union_data[union_data["Time"] == f"{t} min"]
            if not subset.empty:
                subset.plot(
                    ax=ax,
                    color=config.TIME_COLOURS[t],
                    alpha=0.9,
                    edgecolor='none',
                    linewidth=0.0,
                    zorder=2,
                )

        ax.scatter(
            df_points["longitude"],
            df_points["latitude"],
            color="black",
            s=0.25,
            alpha=1,
            marker="o",
            zorder=4,
        )

        ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
        ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)

        ax.set_title(title, fontsize=10, fontweight='bold', family='Times New Roman', pad=4)
        title_y = 1.2
        ax.text(-0.05, title_y, panel_letter, transform=ax.transAxes,
                fontweight='bold', fontsize=10, family='Times New Roman', va='top', ha='left')

    # Continuous colorbar (tiny) placed near facility legend
    norm = Normalize(vmin=min(config.TIME_BINS), vmax=max(config.TIME_BINS))
    sm = ScalarMappable(norm=norm, cmap=config.CMAP)
    sm.set_array([])
    # Ensure colorbar doesn't exceed figure width; keep small
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.015, pad=0.015, aspect=17)
    cbar.set_label('Time (min)', family='Times New Roman', fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    for lbl in cbar.ax.get_xticklabels():
        lbl.set_family('Times New Roman')
    cbar.set_ticks([5, 15, 30, 45, 60])

    # Tiny facility legend on bottom-right
    facility_handle = mlines.Line2D([], [], marker='o', linestyle='', markersize=1.8, color='black', label='Facility')
    # Place legend and colorbar close together at bottom-right
    fig.legend(handles=[facility_handle], loc='lower right', bbox_to_anchor=(0.9, 0.08), frameon=False,
               prop={'family': 'Times New Roman', 'size': 10})

    fig.tight_layout(rect=(0, 0.1, 1, 1))

    eps_path = out_dir / f"journal_figure_1_extended_allbins_continuous_standardized.eps"
    tiff_path = out_dir / f"journal_figure_1_extended_allbins_continuous_standardized.tiff"
    png_path = out_dir / f"journal_figure_1_extended_allbins_continuous_standardized.png"

    fig.savefig(eps_path, format='eps', bbox_inches='tight', dpi=600)
    fig.savefig(tiff_path, format='tiff', bbox_inches='tight', dpi=600)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=600)
    plt.close(fig)

    print(f"✅ Standardized Journal Figure 1 Extended (all bins + continuous) saved: {png_path}")
    return png_path


def create_journal_figure_1_extended_standardized_vertical_allbins_continuous(
    out_dir: Path | str | None = None,
) -> Path:
    """Create an additional VERTICAL extended figure using ALL time bins (5–60)
    with a continuous colorbar. Constrained to width 107 mm and height ≤ 190 mm.

    - Layout: 4×1 stacked panels
    - Size: 4.21 × 7.48 inches
    - Headings: Times New Roman 10 pt bold; panel letters 10 pt
    - Facility markers: extra small
    - Export DPI: 600
    """

    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data ------------------------------------------------------------------
    df_stroke = data.load_stroke_units()
    df_ct = data.load_hospitals_ct()
    try:
        df_extended_stroke = data.load_extended_stroke_units()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Extended stroke units not available: {e}\n"
            f"Please run additional_stroke_centers.ipynb to generate the extended dataset."
        )

    uber_mask, reg_mask, thromb_mask = data.stroke_unit_masks(df_stroke)

    def _load_pkl(p: Path):
        with open(p, "rb") as fh:
            return pickle.load(fh)

    load_time_bins = config.TIME_BINS

    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}
    ct_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_all_CTs.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_all_CTs.pkl").exists()}
    extended_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}_extended_stroke.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}_extended_stroke.pkl").exists()}

    if not extended_polys:
        raise FileNotFoundError(
            "Extended stroke isochrones not found. Please generate isochrones for extended stroke units first.\n"
            "Expected files: poly{t}_extended_stroke.pkl in the DATA_DIR"
        )

    stroke_union = _union(stroke_polys, load_time_bins)
    thromb_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if thromb_mask.iloc[i]] for t in stroke_polys.keys()}, load_time_bins)
    ct_union = _union(ct_polys, load_time_bins)
    extended_union = _union(extended_polys, load_time_bins)

    germany = data.load_germany_outline()

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.lines as mlines

    fig, axes = plt.subplots(4, 1, figsize=(4.21, 7.1), dpi=600)
    plt.subplots_adjust(hspace=0.04)

    panels = [
        ("Hospitals with CT", "a", ct_union, df_ct),
        ("Stroke-Ready Hospitals", "b", extended_union, df_extended_stroke),
        ("All Stroke Units", "c", stroke_union, df_stroke),
        ("Thrombectomy-Certified", "d", thromb_union, df_stroke.loc[thromb_mask].copy()),
    ]

    for i, (title, panel_letter, union_data, df_points) in enumerate(panels):
        ax = axes[i]
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_visible(False)

        germany.boundary.plot(ax=ax, color='black', linewidth=0.35, zorder=3)

        for t in sorted(load_time_bins, reverse=True):
            subset = union_data[union_data['Time'] == f"{t} min"]
            if not subset.empty:
                subset.plot(ax=ax, color=config.TIME_COLOURS[t], alpha=0.9, edgecolor='none', linewidth=0.0, zorder=2)

        ax.scatter(
            df_points['longitude'], df_points['latitude'],
            color='black', s=0.3, alpha=0.8, marker='o', zorder=4
        )

        ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
        ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)

        ax.set_title(title, fontsize=10, fontweight='bold', family='Times New Roman', pad=6)
        ax.text(0.01, 1.08, panel_letter, transform=ax.transAxes, fontweight='bold', fontsize=10,
                family='Times New Roman', va='top', ha='left')

    # Continuous colorbar (tiny) next to facility legend
    norm = Normalize(vmin=min(config.TIME_BINS), vmax=max(config.TIME_BINS))
    sm = ScalarMappable(norm=norm, cmap=config.CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.014, pad=0.016, aspect=30)
    cbar.set_label('Time (min)', family='Times New Roman', fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    for lbl in cbar.ax.get_xticklabels():
        lbl.set_family('Times New Roman')
    cbar.set_ticks([5, 15, 30, 45, 60])

    facility_handle = mlines.Line2D([], [], marker='o', linestyle='', markersize=1.6, color='black', label='Facility')
    fig.legend(handles=[facility_handle], loc='lower right', bbox_to_anchor=(0.99, 0.06), frameon=False,
               prop={'family': 'Times New Roman', 'size': 10})

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    eps_path = out_dir / f"journal_figure_1_extended_vertical_allbins_continuous_standardized.eps"
    tiff_path = out_dir / f"journal_figure_1_extended_vertical_allbins_continuous_standardized.tiff"
    png_path = out_dir / f"journal_figure_1_extended_vertical_allbins_continuous_standardized.png"

    fig.savefig(eps_path, format='eps', bbox_inches='tight', dpi=600)
    fig.savefig(tiff_path, format='tiff', bbox_inches='tight', dpi=600)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=600)
    plt.close(fig)

    print(f"✅ Standardized Journal Figure 1 Extended (vertical, all bins + continuous) saved: {png_path}")
    return png_path

def create_journal_figure_2_standardized_allbins_continuous(out_dir: Path | str | None = None) -> Path:
    """Create an additional standardized Journal Figure 2 (3x1) using all bins
    with a continuous colorbar legend.
    """
    out_dir = Path(out_dir or config.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_stroke = data.load_stroke_units()
    uber_mask, reg_mask, _ = data.stroke_unit_masks(df_stroke)

    def _load_pkl(p: Path):
        with open(p, "rb") as fh:
            return pickle.load(fh)

    load_time_bins = config.TIME_BINS
    stroke_polys = {t: _load_pkl(config.DATA_DIR / f"poly{t}.pkl") for t in load_time_bins if (config.DATA_DIR / f"poly{t}.pkl").exists()}

    stroke_union = _union(stroke_polys, load_time_bins)
    uber_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if uber_mask.iloc[i]] for t in stroke_polys.keys()}, load_time_bins)
    reg_union = _union({t: [p for i, p in enumerate(stroke_polys[t]) if reg_mask.iloc[i]] for t in stroke_polys.keys()}, load_time_bins)

    germany = data.load_germany_outline()

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, axes = plt.subplots(3, 1, figsize=(4.21, 11), dpi=600)

    panels = [
        ("All Stroke Units", "a", stroke_union, df_stroke),
        ("Supra-Regional", "b", uber_union, df_stroke.loc[uber_mask].copy()),
        ("Regional / Telemed", "c", reg_union, df_stroke.loc[reg_mask].copy()),
    ]

    for i, (title, panel_letter, union_data, df_points) in enumerate(panels):
        ax = axes[i]
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_visible(False)

        germany.boundary.plot(ax=ax, color='black', linewidth=0.6, zorder=3)

        for t in sorted(load_time_bins, reverse=True):
            subset = union_data[union_data['Time'] == f"{t} min"]
            if not subset.empty:
                subset.plot(ax=ax, color=config.TIME_COLOURS[t], alpha=0.9, edgecolor='none', linewidth=0.0, zorder=2)

        ax.scatter(df_points['longitude'], df_points['latitude'], color='black', s=0.15, alpha=0.8, marker='o', zorder=4)
        ax.set_xlim(germany.total_bounds[0] - 0.5, germany.total_bounds[2] + 0.5)
        ax.set_ylim(germany.total_bounds[1] - 0.5, germany.total_bounds[3] + 0.5)

        ax.set_title(title, fontsize=10, fontweight='bold', family='Times New Roman', pad=10)
        ax.text(0.01, 1.06, panel_letter, transform=ax.transAxes, fontweight='bold', fontsize=10,
                family='Times New Roman', va='top', ha='left')
    # --- continuous colorbar and facility legend (compact, under last subplot) ---
    norm = Normalize(vmin=min(config.TIME_BINS), vmax=max(config.TIME_BINS))
    sm = ScalarMappable(norm=norm, cmap=config.CMAP); sm.set_array([])

    plt.tight_layout()

    last_pos   = axes[2].get_position()
    bar_width  = last_pos.width * 0.33                  # narrow bar
    bar_left   = last_pos.x0 + (last_pos.width - bar_width) / 2 - 0.05  # slight left shift
    bar_bottom = last_pos.y0 - 0.038                    # closer to the subplot

    cb_ax = fig.add_axes([bar_left, bar_bottom, bar_width, 0.013])      # lower height
    cbar  = fig.colorbar(sm, cax=cb_ax, orientation='horizontal')
    cbar.set_label('Time (min)', family='Times New Roman', fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    for lbl in cbar.ax.get_xticklabels():
        lbl.set_family('Times New Roman')
    cbar.set_ticks([5, 15, 30, 45, 60])

    # facility legend just to the right of the bar
    import matplotlib.lines as mlines
    facility_handle = mlines.Line2D([], [], color='black', marker='o',
                                    linestyle='None', markersize=4, label='Facility')
    leg_ax = fig.add_axes([bar_left + bar_width + 0.006, bar_bottom - 0.001, 0.1, 0.028])
    leg_ax.axis('off')
    leg_ax.legend(handles=[facility_handle], loc='center left',
                  frameon=False, fontsize=10, prop={'family': 'Times New Roman'})

    eps_path = out_dir / f"journal_figure_2_stroke_hierarchy_standardized_allbins_continuous.eps"
    tiff_path = out_dir / f"journal_figure_2_stroke_hierarchy_standardized_allbins_continuous.tiff"
    png_path = out_dir / f"journal_figure_2_stroke_hierarchy_standardized_allbins_continuous.png"

    fig.savefig(eps_path, format='eps', bbox_inches='tight', dpi=600)
    fig.savefig(tiff_path, format='tiff', bbox_inches='tight', dpi=600)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=600)
    plt.close(fig)

    print(f"✅ Standardized Journal Figure 2 (all bins + continuous) saved: {png_path}")
    return png_path

