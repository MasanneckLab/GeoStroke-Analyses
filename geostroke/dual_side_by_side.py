"""
Module: dual_side_by_side_parallel_module
Purpose: Run dual (Regular vs Extended) scenario rendering + stats in parallel (states & counties)
Platform friendly: works on macOS (spawn) because workers live in this module
Parallel backend: concurrent.futures with mp_context="fork" when available, falling back to default.

Usage from Jupyter (minimal cell at bottom of this file's docstring).

Assumptions: you already have these objects in your session or can import them:
- gs, benefit  (your own packages)
- GERMANY (GeoDataFrame), GERMANY_GRID (GeoSeries/GeoDataFrame of points), states, counties
- pop_arr_full (numpy array), pop_transform (affine), POP_RASTER_CRS (pyproj CRS or str)
- TIME_BINS (list/np.array), GRID_RESOLUTION (float)

If names differ, pass them explicitly to Context.
"""

from __future__ import annotations

import os, math, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Iterable

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from rasterio.features import rasterize, geometry_mask
import matplotlib
matplotlib.use("Agg")  # safe headless render in workers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

import multiprocessing as mp
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------------- Dataclasses / Config ----------------------------------------

@dataclass
class Context:
    # geo and raster context
    GERMANY: gpd.GeoDataFrame
    GERMANY_GRID: Iterable
    states: gpd.GeoDataFrame
    counties: gpd.GeoDataFrame
    pop_arr_full: np.ndarray
    pop_transform: object  # affine.Affine
    POP_RASTER_CRS: object  # pyproj CRS or str
    # params
    TIME_BINS: Iterable
    GRID_RESOLUTION: float

@dataclass
class RunConfig:
    RENDER_MODE: str = "polygons"          # "points" | "polygons" | "raster"
    GRID_SHAPE: str = "square"
    CELL_SIZE_DEG: float = 0.01
    DISSOLVE_POLYS: bool = True
    PDF_STATES: bool = True
    PDF_COUNTIES: bool = True
    RUN_COUNTIES: bool = True
    CPU_STATES: int = max(1, mp.cpu_count() - 1)
    CPU_COUNTIES: int = max(1, mp.cpu_count() - 1)
    OUT_ROOT: Path | None = None
    # Colors and bins
    CATEGORY_ORDER: Tuple[str, ...] = (
        "High (30+ min)",
        "Medium (20-30 min)",
        "Low (10-20 min)",
        "Only CT reachable within 60 min",
        "Neither reachable within 60 min",
        "Likely irrelevant (<10 min)",
    )
    BENEFIT_COLORS: Dict[str, str] = None

    def __post_init__(self):
        if self.BENEFIT_COLORS is None:
            self.BENEFIT_COLORS = {
                "High (30+ min)": "#006837",
                "Medium (20-30 min)": "#31a354",
                "Low (10-20 min)": "#78c679",
                "Only CT reachable within 60 min": "#41b6c4",
                "Neither reachable within 60 min": "#bdbdbd",
                "Likely irrelevant (<10 min)": "white",
            }

# Default scenarios
DEFAULT_SCENARIOS = {
    "Normal Speed": {"suffix": "", "penalty": 0.0},
    "+20% Speed (Emergency)": {"suffix": "_emergency", "penalty": 0.0},
    "-20% Speed (Traffic)": {"suffix": "_bad_traffic", "penalty": 0.0},
    "+10min Penalty": {"suffix": "", "penalty": 10},
    "+20min Penalty": {"suffix": "", "penalty": 20},
    "+30min Penalty": {"suffix": "", "penalty": 30},
}

# ----------------------------- Helper functions (pure) ----------------------------------------

def square_cell(pt: Point, half: float) -> Polygon:
    x, y = pt.x, pt.y
    return Polygon([(x-half, y-half), (x+half, y-half), (x+half, y+half), (x-half, y+half)])


def classify_grid_points(ctx: Context, benefit_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid_df = gpd.GeoDataFrame(geometry=ctx.GERMANY_GRID, crs=ctx.GERMANY.crs)
    joined = gpd.sjoin(grid_df, benefit_gdf[["benefit_category", "geometry"]], how="left", predicate="within")
    joined["benefit_category"] = joined["benefit_category"].fillna("Likely irrelevant (<10 min)")
    return joined.drop(columns=["index_right"], errors="ignore")


def build_cat_geoms_from_grid(ctx: Context, grid_with_cat: gpd.GeoDataFrame, category_order: Tuple[str, ...]) -> Dict[str, Polygon]:
    g = grid_with_cat.copy()
    half = ctx.GRID_RESOLUTION / 2.0
    g["cell_poly"] = g.geometry.apply(lambda p: square_cell(p, half))
    cat_geoms: Dict[str, Polygon] = {}
    remaining = ctx.GERMANY.geometry.iloc[0]
    for cat in category_order[:-1]:
        sub = g.loc[g["benefit_category"] == cat, "cell_poly"]
        if sub.empty:
            cat_geoms[cat] = None
            continue
        u = unary_union(sub.values)
        if not u.is_empty:
            u = u.intersection(remaining)
        cat_geoms[cat] = u
        remaining = remaining.difference(u)
    cat_geoms[category_order[-1]] = remaining
    return cat_geoms


def pop_for_geom(ctx: Context, geom) -> int:
    gdf = gpd.GeoDataFrame(geometry=[geom], crs=ctx.GERMANY.crs)
    if gdf.crs != ctx.POP_RASTER_CRS:
        gdf = gdf.to_crs(ctx.POP_RASTER_CRS)
    mask_bool = geometry_mask(gdf.geometry, transform=ctx.pop_transform,
                              invert=True, out_shape=ctx.pop_arr_full.shape)
    return int(ctx.pop_arr_full[mask_bool].sum())


def pop_by_categories(ctx: Context, cat_geoms: Dict[str, Polygon], category_order: Tuple[str, ...], region_geom=None) -> Dict[str, int]:
    reg_mask = None
    if region_geom is not None:
        gdf = gpd.GeoDataFrame(geometry=[region_geom], crs=ctx.GERMANY.crs)
        if gdf.crs != ctx.POP_RASTER_CRS:
            gdf = gdf.to_crs(ctx.POP_RASTER_CRS)
        reg_mask = geometry_mask(gdf.geometry, transform=ctx.pop_transform,
                                 invert=True, out_shape=ctx.pop_arr_full.shape)
    out = {}
    for i, cat in enumerate(category_order, start=1):
        geom = cat_geoms.get(cat)
        if geom is None or geom.is_empty:
            out[cat] = 0
            continue
        geoms_iter = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        shapes = []
        for g in geoms_iter:
            gg = gpd.GeoDataFrame(geometry=[g], crs=ctx.GERMANY.crs)
            if gg.crs != ctx.POP_RASTER_CRS:
                gg = gg.to_crs(ctx.POP_RASTER_CRS)
            shapes.append((gg.geometry.iloc[0], i))
        rast = rasterize(shapes, out_shape=ctx.pop_arr_full.shape, transform=ctx.pop_transform,
                         fill=0, dtype="uint16")
        mask_cat = (rast == i)
        if reg_mask is not None:
            mask_cat &= reg_mask
        out[cat] = int(ctx.pop_arr_full[mask_cat].sum())
    return out


def stats_for_region(ctx: Context, region_geom, region_name, grid_with_cat, cat_geoms, category_order) -> pd.DataFrame:
    reg_gdf = gpd.GeoDataFrame(geometry=[region_geom], crs=grid_with_cat.crs)
    pts_in = gpd.sjoin(grid_with_cat[["geometry", "benefit_category"]], reg_gdf,
                       how="inner", predicate="within").drop(columns="index_right", errors="ignore")
    total_pts = len(pts_in)
    area_counts = pts_in["benefit_category"].value_counts()

    reg_pop = pop_for_geom(ctx, region_geom)
    pop_counts = pop_by_categories(ctx, cat_geoms, category_order, region_geom)
    diff = reg_pop - sum(pop_counts.values())
    if diff != 0:
        pop_counts[category_order[-1]] += diff

    rows = []
    for cat in category_order:
        a = int(area_counts.get(cat, 0))
        rows.append({
            "region": region_name,
            "category": cat,
            "area_pts": a,
            "area_pct": round(a / total_pts * 100, 1) if total_pts else 0,
            "population": pop_counts.get(cat, 0),
            "population_pct": round(pop_counts.get(cat, 0) / reg_pop * 100, 1) if reg_pop else 0,
        })
    return pd.DataFrame(rows)


def make_benefit_handles(color_dict):
    return [mpatches.Patch(facecolor=col, edgecolor="black", linewidth=0.6, label=cat)
            for cat, col in color_dict.items()]


def _ensure_benefit_column(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "benefit_category" not in df.columns and df.index.name == "benefit_category":
        df = df.reset_index()
    return df


def _build_grid(bounds: Tuple[float, float, float, float], shape: str, size_deg: float, crs) -> gpd.GeoDataFrame:
    if shape == "hex":
        w = size_deg; h = math.sqrt(3) * w / 2
        minx, miny, maxx, maxy = bounds
        cols = int(np.ceil((maxx - minx) / w)) + 1
        rows = int(np.ceil((maxy - miny) / h)) + 1
        polys = []
        for r in range(rows):
            y = miny + r * h
            x_off = (w / 2) if (r % 2) else 0
            for c in range(cols):
                x = minx + c * w + x_off
                polys.append(Polygon([(x - w/2, y), (x - w/4, y + h/2), (x + w/4, y + h/2),
                                      (x + w/2, y), (x + w/4, y - h/2), (x - w/4, y - h/2)]))
        return gpd.GeoDataFrame({"geometry": polys}, crs=crs)
    else:
        minx, miny, maxx, maxy = bounds
        xs = np.arange(minx, maxx, size_deg)
        ys = np.arange(miny, maxy, size_deg)
        polys = [Polygon([(x, y), (x + size_deg, y), (x + size_deg, y + size_deg), (x, y + size_deg)])
                 for x in xs for y in ys]
        return gpd.GeoDataFrame({"geometry": polys}, crs=crs)


def region_plot(ax, region_geom, benefit_gdf, grid_with_cat, render_mode,
                benefit_colors, germany, markersize=0.5,
                raster_w=900, raster_h=1200,
                grid_shape="square", cell_size_deg=0.02, dissolve=False):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    gpd.GeoDataFrame(geometry=[region_geom], crs=germany.crs).boundary.plot(
        ax=ax, color="black", linewidth=0.6, zorder=3)

    if render_mode == "polygons":
        pts_in = gpd.sjoin(grid_with_cat[["geometry", "benefit_category"]],
                           gpd.GeoDataFrame(geometry=[region_geom], crs=grid_with_cat.crs),
                           how="inner", predicate="within").drop(columns="index_right", errors="ignore")
        if pts_in.empty:
            return

        bounds = region_geom.bounds
        grid = _build_grid(bounds, grid_shape, cell_size_deg, germany.crs)
        grid = gpd.overlay(grid, gpd.GeoDataFrame(geometry=[region_geom], crs=germany.crs), how="intersection")

        joined = gpd.sjoin(pts_in, grid, how="inner", predicate="within")
        cats = joined.groupby("index_right")["benefit_category"].agg(lambda v: v.value_counts().idxmax())
        grid["benefit_category"] = cats
        grid = grid.dropna(subset=["benefit_category"])
        if dissolve:
            grid = grid.dissolve(by="benefit_category")
            grid = _ensure_benefit_column(grid)
        for cat, col in benefit_colors.items():
            sub_poly = grid[grid["benefit_category"] == cat]
            if sub_poly.empty:
                continue
            sub_poly.plot(ax=ax, facecolor=col, edgecolor='none', linewidth=0, zorder=2, rasterized=True)
    else:
        pts_in = gpd.sjoin(grid_with_cat[["geometry", "benefit_category"]],
                           gpd.GeoDataFrame(geometry=[region_geom], crs=grid_with_cat.crs),
                           how="inner", predicate="within").drop(columns="index_right", errors="ignore")
        for cat, col in benefit_colors.items():
            sub_pts = pts_in[pts_in["benefit_category"] == cat]
            if sub_pts.empty:
                continue
            sub_pts.plot(ax=ax, color=col, markersize=markersize, linewidth=0,
                         alpha=0.8, zorder=2, rasterized=True)

# ----------------------------- Worker wrappers ------------------------------------------------

def process_state_worker(st_name, geom, scen_name, scen_slug,
                         ctx: Context,
                         grid_reg, cats_reg,
                         grid_ext, cats_ext,
                         benefit_gdf_reg, benefit_gdf_ext,
                         render_mode, grid_shape, cell_size_deg, dissolve,
                         benefit_colors, handles, germany,
                         s_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    df_r = stats_for_region(ctx, geom, f"{st_name} (Regular)", grid_reg, cats_reg, ctx_cfg.CATEGORY_ORDER)
    df_r['analysis_type'] = 'Regular'; df_r['scenario'] = scen_name

    df_e = stats_for_region(ctx, geom, f"{st_name} (Extended)", grid_ext, cats_ext, ctx_cfg.CATEGORY_ORDER)
    df_e['analysis_type'] = 'Extended'; df_e['scenario'] = scen_name

    fig = plt.figure(figsize=(16, 6), dpi=300)
    ax_left, ax_right = fig.subplots(1, 2)

    region_plot(ax_left, geom, benefit_gdf_reg, grid_reg, render_mode,
                benefit_colors, germany, markersize=0.3,
                grid_shape=grid_shape, cell_size_deg=cell_size_deg,
                dissolve=dissolve)
    ax_left.set_title(f"Certified Stroke Units {st_name}", fontsize=10, fontweight='bold')

    region_plot(ax_right, geom, benefit_gdf_ext, grid_ext, render_mode,
                benefit_colors, germany, markersize=0.3,
                grid_shape=grid_shape, cell_size_deg=cell_size_deg,
                dissolve=dissolve)
    ax_right.set_title(f"Extended Hospitals {st_name}", fontsize=10, fontweight='bold')

    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=0.14)

    png_path = s_dir / f"Dual_Benefit_{st_name.replace(' ', '_')}_{scen_slug}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return df_r, df_e, str(png_path)


def process_county_worker(c_name, s_name, geom, scen_name, scen_slug,
                          ctx: Context,
                          grid_reg, cats_reg,
                          grid_ext, cats_ext,
                          benefit_gdf_reg, benefit_gdf_ext,
                          render_mode, grid_shape, cell_size_deg, dissolve,
                          benefit_colors, handles, germany,
                          c_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    left_stats = stats_for_region(ctx, geom, f"{c_name} (Regular)", grid_reg, cats_reg, ctx_cfg.CATEGORY_ORDER)
    left_stats['state'] = s_name; left_stats['analysis_type'] = 'Regular'; left_stats['scenario'] = scen_name

    right_stats = stats_for_region(ctx, geom, f"{c_name} (Extended)", grid_ext, cats_ext, ctx_cfg.CATEGORY_ORDER)
    right_stats['state'] = s_name; right_stats['analysis_type'] = 'Extended'; right_stats['scenario'] = scen_name

    fig = plt.figure(figsize=(14, 5), dpi=300)
    ax_left, ax_right = fig.subplots(1, 2)

    region_plot(ax_left, geom, benefit_gdf_reg, grid_reg, render_mode,
                benefit_colors, germany, markersize=0.3,
                grid_shape=grid_shape, cell_size_deg=cell_size_deg,
                dissolve=dissolve)
    ax_left.set_title(f"Certified Stroke Units {c_name} ({s_name})", fontsize=9, fontweight='bold')

    region_plot(ax_right, geom, benefit_gdf_ext, grid_ext, render_mode,
                benefit_colors, germany, markersize=0.3,
                grid_shape=grid_shape, cell_size_deg=cell_size_deg,
                dissolve=dissolve)
    ax_right.set_title(f"Extended Hospitals {c_name} ({s_name})", fontsize=9, fontweight='bold')

    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=7)
    fig.subplots_adjust(bottom=0.16)

    s_folder = c_dir / s_name.replace(' ', '_')
    s_folder.mkdir(parents=True, exist_ok=True)
    png_path = s_folder / f"Dual_Benefit_{c_name.replace(' ', '_')}_{scen_slug}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return left_stats, right_stats, str(png_path)

# ctx_cfg is a module-level alias used inside workers (pickled once)
ctx_cfg: RunConfig

# ----------------------------- Main runner ----------------------------------------------------

def run_all_scenarios(ctx: Context,
                      scenarios: Dict[str, Dict[str, float]] = None,
                      run_cfg: RunConfig = None,
                      benefit_module=None,
                      gs_module=None) -> None:
    """Master function to execute everything.
    Parameters:
        ctx: Context object with geo/raster data
        scenarios: dict like DEFAULT_SCENARIOS
        run_cfg: RunConfig with flags and output path
        benefit_module: module that provides calculate_time_benefits_parallel
        gs_module: module providing config (only used for default OUT_ROOT if None)
    """
    global ctx_cfg
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if run_cfg is None:
        run_cfg = RunConfig()
    ctx_cfg = run_cfg  # expose to workers

    if run_cfg.OUT_ROOT is None:
        if gs_module is None:
            raise ValueError("OUT_ROOT not set in RunConfig and gs_module is None")
        run_cfg.OUT_ROOT = Path(gs_module.config.RESULTS_DIR) / "Dual_All_Scenarios"
    run_cfg.OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # BLAS env (avoid thread explosion)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    handles = make_benefit_handles(run_cfg.BENEFIT_COLORS)

    state_records: List[pd.DataFrame] = []
    county_records: List[pd.DataFrame] = []

    # choose mp context
    try:
        CTX = get_context("fork")  # macOS: enable fork to preserve globals if desired
    except ValueError:
        CTX = mp.get_context()

    for scen_name, cfg in scenarios.items():
        print(f"=== {scen_name} ===")
        scen_slug = scen_name.replace(" ", "_").replace("%", "pct").replace("(", "").replace(")", "")

        # Output dirs per scenario
        S_DIR = run_cfg.OUT_ROOT / scen_slug / "States"
        C_DIR = run_cfg.OUT_ROOT / scen_slug / "Counties"
        S_DIR.mkdir(parents=True, exist_ok=True)
        C_DIR.mkdir(parents=True, exist_ok=True)

        # Load both scenarios from cache
        benefit_gdf_reg = benefit_module.calculate_time_benefits_parallel(
            ct_suffix=f"_all_CTs{cfg['suffix']}",
            stroke_suffix=cfg['suffix'],
            ct_penalty=cfg['penalty'],
            benefit_threshold=10.0,
            grid_resolution=ctx.GRID_RESOLUTION,
            time_bins=ctx.TIME_BINS,
            max_workers=10,
            force_recalc=False,
        )
        if benefit_gdf_reg.crs != ctx.GERMANY.crs:
            benefit_gdf_reg = benefit_gdf_reg.to_crs(ctx.GERMANY.crs)
        benefit_gdf_reg = gpd.sjoin(benefit_gdf_reg, ctx.GERMANY[["geometry"]], how="inner", predicate="within")
        benefit_gdf_reg.drop(columns=["index_right"], inplace=True, errors="ignore")

        benefit_gdf_ext = benefit_module.calculate_time_benefits_parallel(
            ct_suffix=f"_all_CTs{cfg['suffix']}",
            stroke_suffix=f"_extended_stroke{cfg['suffix']}",
            ct_penalty=cfg['penalty'],
            benefit_threshold=10.0,
            grid_resolution=ctx.GRID_RESOLUTION,
            time_bins=ctx.TIME_BINS,
            max_workers=10,
            force_recalc=False,
        )
        if benefit_gdf_ext.crs != ctx.GERMANY.crs:
            benefit_gdf_ext = benefit_gdf_ext.to_crs(ctx.GERMANY.crs)
        benefit_gdf_ext = gpd.sjoin(benefit_gdf_ext, ctx.GERMANY[["geometry"]], how="inner", predicate="within")
        benefit_gdf_ext.drop(columns=["index_right"], inplace=True, errors="ignore")

        # classify + cat polys
        grid_reg = classify_grid_points(ctx, benefit_gdf_reg)
        cats_reg = build_cat_geoms_from_grid(ctx, grid_reg, run_cfg.CATEGORY_ORDER)
        grid_ext = classify_grid_points(ctx, benefit_gdf_ext)
        cats_ext = build_cat_geoms_from_grid(ctx, grid_ext, run_cfg.CATEGORY_ORDER)

        # -------- STATES --------
        state_results = []
        with ProcessPoolExecutor(max_workers=run_cfg.CPU_STATES, mp_context=CTX) as ex:
            futs = []
            for _, st in ctx.states.iterrows():
                st_name = st.get('NAME_1', 'Unknown')
                fut = ex.submit(process_state_worker, st_name, st.geometry, scen_name, scen_slug,
                                 ctx, grid_reg, cats_reg, grid_ext, cats_ext,
                                 benefit_gdf_reg, benefit_gdf_ext,
                                 run_cfg.RENDER_MODE, run_cfg.GRID_SHAPE, run_cfg.CELL_SIZE_DEG,
                                 run_cfg.DISSOLVE_POLYS,
                                 run_cfg.BENEFIT_COLORS, handles, ctx.GERMANY,
                                 S_DIR)
                futs.append(fut)
            for fut in as_completed(futs):
                df_r, df_e, png = fut.result()
                state_results.append((df_r, df_e, png))

        for df_r, df_e, _ in state_results:
            state_records.append(df_r)
            state_records.append(df_e)

        if run_cfg.PDF_STATES:
            pdf_state_path = S_DIR / f"Dual_Benefit_states_{scen_slug}.pdf"
            with PdfPages(pdf_state_path) as pdf_state:
                for _, _, png in sorted(state_results, key=lambda t: t[2]):
                    fig = plt.figure(figsize=(16, 6), dpi=300)
                    img = plt.imread(png)
                    plt.imshow(img); plt.axis('off')
                    pdf_state.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            print(f"   -> State PDF saved: {pdf_state_path.name}")

        # -------- COUNTIES --------
        if run_cfg.RUN_COUNTIES:
            county_results = []
            with ProcessPoolExecutor(max_workers=run_cfg.CPU_COUNTIES, mp_context=CTX) as ex:
                futs = []
                for i, ct in ctx.counties.iterrows():
                    c_name, s_name = ct.county_name, ct.state_name
                    fut = ex.submit(process_county_worker, c_name, s_name, ct.geometry, scen_name, scen_slug,
                                     ctx, grid_reg, cats_reg, grid_ext, cats_ext,
                                     benefit_gdf_reg, benefit_gdf_ext,
                                     run_cfg.RENDER_MODE, run_cfg.GRID_SHAPE, run_cfg.CELL_SIZE_DEG,
                                     run_cfg.DISSOLVE_POLYS,
                                     run_cfg.BENEFIT_COLORS, handles, ctx.GERMANY,
                                     C_DIR)
                    futs.append(fut)
                for idx, fut in enumerate(as_completed(futs), start=1):
                    ldf, rdf, png = fut.result()
                    county_results.append((ldf, rdf, png))
                    if idx % 50 == 0:
                        print(f"   {idx}/{len(ctx.counties)} counties …")

            for ldf, rdf, _ in county_results:
                county_records.append(ldf)
                county_records.append(rdf)

            if run_cfg.PDF_COUNTIES:
                pdf_cnt_path = C_DIR / f"Dual_Benefit_counties_{scen_slug}.pdf"
                with PdfPages(pdf_cnt_path) as pdf_cnt:
                    for _, _, png in sorted(county_results, key=lambda t: t[2]):
                        fig = plt.figure(figsize=(14, 5), dpi=300)
                        img = plt.imread(png)
                        plt.imshow(img); plt.axis('off')
                        pdf_cnt.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                print(f"   -> County PDF saved: {pdf_cnt_path.name}")

    # ======================= SAVE STATS (combined) ==========================================
    print("Saving combined statistics …")

    if state_records:
        state_df_all = pd.concat(state_records, ignore_index=True)
        xlsx_state_all = run_cfg.OUT_ROOT / "dual_benefit_state_stats_ALL.xlsx"
        with pd.ExcelWriter(xlsx_state_all, engine='openpyxl') as w:
            state_df_all.to_excel(w, sheet_name="long", index=False)
            for (scen, atype), df_sub in state_df_all.groupby(['scenario', 'analysis_type']):
                sheet = f"{scen[:20]}_{atype.lower()}_pop_pct".replace(' ', '_')
                (df_sub.pivot_table(index="region", columns="category", values="population_pct", fill_value=0)
                 .reindex(columns=run_cfg.CATEGORY_ORDER)).to_excel(w, sheet_name=sheet[:31])
        print(f" -> {xlsx_state_all.name}")

    if county_records:
        county_df_all = pd.concat(county_records, ignore_index=True)
        xlsx_county_all = run_cfg.OUT_ROOT / "dual_benefit_county_stats_ALL.xlsx"
        with pd.ExcelWriter(xlsx_county_all, engine='openpyxl') as w:
            county_df_all.to_excel(w, sheet_name="long", index=False)
            for (scen, atype), df_sub in county_df_all.groupby(['scenario', 'analysis_type']):
                sheet = f"{scen[:20]}_{atype.lower()}_pop_pct".replace(' ', '_')
                (df_sub.pivot_table(index="region", columns="category", values="population_pct", fill_value=0)
                 .reindex(columns=run_cfg.CATEGORY_ORDER)).to_excel(w, sheet_name=sheet[:31])
        print(f" -> {xlsx_county_all.name}")

    print("✅ Dual side-by-side analysis for all scenarios complete (parallel states & counties).")


# --------------------------------- CLI hook ---------------------------------------------------

def _cli():
    import importlib
    # naive CLI: expects your env vars to locate gs/benefit
    gs = importlib.import_module('gs')
    benefit = importlib.import_module('benefit')
    from your_data_module import (GERMANY, GERMANY_GRID, states, counties,
                                  pop_arr_full, pop_transform, POP_RASTER_CRS,
                                  TIME_BINS, GRID_RESOLUTION)

    ctx = Context(GERMANY, GERMANY_GRID, states, counties,
                  pop_arr_full, pop_transform, POP_RASTER_CRS,
                  TIME_BINS, GRID_RESOLUTION)
    run_all_scenarios(ctx, DEFAULT_SCENARIOS, RunConfig(OUT_ROOT=Path(gs.config.RESULTS_DIR)/"Dual_All_Scenarios"),
                      benefit_module=benefit, gs_module=gs)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    _cli()

"""
Minimal configurable Jupyter cell:
---------------------------------

```python
from pathlib import Path
import dual_side_by_side_parallel_module as dsp

# 1) Build context
ctx = dsp.Context(
    GERMANY=GERMANY,
    GERMANY_GRID=GERMANY_GRID,
    states=states,
    counties=counties,
    pop_arr_full=pop_arr_full,
    pop_transform=pop_transform,
    POP_RASTER_CRS=POP_RASTER_CRS,
    TIME_BINS=TIME_BINS,
    GRID_RESOLUTION=GRID_RESOLUTION,
)

# 2) Configure run
run_cfg = dsp.RunConfig(
    OUT_ROOT=Path(gs.config.RESULTS_DIR)/"Dual_All_Scenarios",
    CPU_STATES=8,
    CPU_COUNTIES=8,
    RUN_COUNTIES=True,
    PDF_STATES=True,
    PDF_COUNTIES=True,
    RENDER_MODE="polygons",
)

# 3) Run
import benefit  # your module
import gs       # for OUT_ROOT default if you didn't set it

dsp.run_all_scenarios(ctx, scenarios=dsp.DEFAULT_SCENARIOS, run_cfg=run_cfg,
                      benefit_module=benefit, gs_module=gs)
```
"""