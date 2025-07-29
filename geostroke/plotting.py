"""Plotting utilities for GeoStroke.

This sub-module centralises the Matplotlib logic for drawing the
isochrone maps so that both the command-line pipeline and the showcase
notebook can import the same functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.artist import Artist

from . import config

__all__ = [
    "plot_isochrones_panel",
    "add_legend",
    "add_legend_all_time_bins",
    "plot_journal_isochrones_panel",
    "add_journal_legend",
]


def _setup_ax(ax: Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_isochrones_panel(
    ax: Axes,
    germany_outline: gpd.GeoDataFrame,
    gdf_union: gpd.GeoDataFrame,
    df_points: pd.DataFrame,
    *,
    title: str,
    panel_letter: str | None = None,
    time_bins: list[int] | None = None,
) -> None:
    """Draw a single panel (isochrones + facility points).
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to plot. If None, uses DEFAULT_TIME_BINS for backward compatibility.
    """

    _setup_ax(ax)

    # Use provided time_bins or default to backward-compatible bins
    plot_time_bins = time_bins or config.DEFAULT_TIME_BINS

    # background
    germany_outline.boundary.plot(ax=ax, color="black", linewidth=0.6, zorder=3)

    # isochrone bands – biggest first so smallest stays on top
    for t in sorted(plot_time_bins, reverse=True):
        subset = gdf_union[gdf_union["Time"] == f"{t} min"]
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=config.TIME_COLOURS[t],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.1,
                zorder=2,
            )

    # facility scatter
    ax.scatter(
        df_points["longitude"],
        df_points["latitude"],
        color="black",
        s=1,  # Reduced from 8 for better visibility in extended figures
        alpha=0.8,
        marker="o",
        zorder=4,
    )

    # Panel title - use default font
    title_obj = ax.set_title(title, fontweight="bold", fontsize=10)
    
    if panel_letter:
        # Use letters with parentheses in Times New Roman font, positioned above title
        title_y = 1.1  # Positioned above the title for better visibility
        ax.text(
            0.01,
            title_y,
            panel_letter,  # Use as-is (already includes parentheses)
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,  # Same size as title for perfect alignment
            family="Times New Roman",
            va="top",  # Align from top like title
            ha="left",
        )


def add_legend(fig: Figure, *, loc: str = "lower center", ncol: int | None = None, time_bins: list[int] | None = None) -> None:
    """Append a shared legend to *fig* covering time bins + facility dot.
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include in legend. If None, uses DEFAULT_TIME_BINS for backward compatibility.
    """

    # Default to original time bins for backward compatibility
    legend_time_bins = time_bins or config.DEFAULT_TIME_BINS
    
    if ncol is None:
        ncol = len(legend_time_bins) + 1

    handles: list[Artist] = [
        mpatches.Patch(color=config.TIME_COLOURS[t], label=f"{t} min")
        for t in legend_time_bins if t in config.TIME_COLOURS
    ]
    
    handles.append(
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            color="black",
            label="Facility",
        )
    )
    # Legend with default font
    fig.legend(handles=handles, loc=loc, ncol=ncol, frameon=False, 
              prop={'size': 10})


def add_legend_all_time_bins(fig: Figure, *, loc: str = "lower center", ncol: int | None = None) -> None:
    """Append a shared legend with ALL available time bins (for supplemental figures)."""
    
    if ncol is None:
        ncol = min(len(config.TIME_BINS) + 1, 8)  # Limit columns for readability

    handles: list[Artist] = [
        mpatches.Patch(color=config.TIME_COLOURS[t], label=f"{t} min")
        for t in config.TIME_BINS if t in config.TIME_COLOURS
    ]
    handles.append(
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            color="black",
            label="Facility",
        )
    )
    # Legend with default font
    fig.legend(handles=handles, loc=loc, ncol=ncol, frameon=False, 
              prop={'size': 10})

def plot_journal_isochrones_panel(
    ax: Axes,
    germany_outline: gpd.GeoDataFrame,
    gdf_union: gpd.GeoDataFrame,
    df_points: pd.DataFrame,
    *,
    title: str,
    panel_letter: str | None = None,
    time_bins: list[int] | None = None,
) -> None:
    """Draw a single panel for journal submission (optimized for 190mm width).
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to plot. If None, uses DEFAULT_TIME_BINS for backward compatibility.
    """

    _setup_ax(ax)

    # Use provided time_bins or default to backward-compatible bins
    plot_time_bins = time_bins or config.DEFAULT_TIME_BINS

    # background
    germany_outline.boundary.plot(ax=ax, color="black", linewidth=0.4, zorder=3)  # Slightly thinner for smaller figures

    # isochrone bands – biggest first so smallest stays on top
    for t in sorted(plot_time_bins, reverse=True):
        subset = gdf_union[gdf_union["Time"] == f"{t} min"]
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=config.TIME_COLOURS[t],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.08,  # Thinner lines for journal figures
                zorder=2,
            )

    # facility scatter - smaller markers for journal figures
    ax.scatter(
        df_points["longitude"],
        df_points["latitude"],
        color="black",
        s=1,  # Very small markers for journal figures
        alpha=0.8,
        marker="o",
        zorder=4,
    )

    # Panel title - Times New Roman, bold  
    title_obj = ax.set_title(title, fontweight="bold", fontsize=10, family="Times New Roman")
    
    if panel_letter:
        # Use letters with parentheses in Times New Roman font, positioned above title
        title_y = 1.1  # Positioned above the title for better visibility
        ax.text(
            0.01,
            title_y,
            panel_letter,  # Use as-is (already includes parentheses)
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,  # Same size as title for perfect alignment
            family="Times New Roman",
            va="top",  # Align from top like title
            ha="left",
        )


def add_journal_legend(fig: Figure, *, loc: str = "lower center", ncol: int | None = None, time_bins: list[int] | None = None) -> None:
    """Append a shared legend optimized for journal figures (190mm width).
    
    Parameters
    ----------
    time_bins : list[int], optional
        Specific time bins to include in legend. If None, uses DEFAULT_TIME_BINS for backward compatibility.
    """

    # Default to original time bins for backward compatibility
    legend_time_bins = time_bins or config.DEFAULT_TIME_BINS
    
    if ncol is None:
        ncol = len(legend_time_bins) + 1

    handles: list[Artist] = [
        mpatches.Patch(color=config.TIME_COLOURS[t], label=f"{t} min")
        for t in legend_time_bins if t in config.TIME_COLOURS
    ]
    
    handles.append(
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=2.5,  # Very small marker in legend to match plot
            color="black",
            label="Facility",
        )
    )
    # Legend with Times New Roman font, optimized size for journal
    fig.legend(handles=handles, loc=loc, ncol=ncol, frameon=False, 
              prop={'family': 'Times New Roman', 'size': 9})

# ---------------------------------------------------------------------------
# Convenience wrapper – four-panel figure
# ---------------------------------------------------------------------------


def four_panel_isochrones(
    germany_outline: gpd.GeoDataFrame,
    gdf_panels: list[gpd.GeoDataFrame],
    df_points: list[pd.DataFrame],
    *,
    titles: list[str],
    letters: list[str] | None = None,
    figsize: tuple[int, int] = (18, 14),
    time_bins: list[int] | None = None,
) -> Figure:
    """Create a 2×2 figure and return the Matplotlib Figure.

    Parameters
    ----------
    germany_outline : GeoDataFrame
        High-resolution outline (EPSG:4326).
    gdf_panels : list[GeoDataFrame]
        Four GeoDataFrames (unioned polygons) in panel order.
    df_points : list[DataFrame]
        Facility DataFrames (must match order of ``gdf_panels``).
    titles : list[str]
        Panel titles A, B, C, D.
    letters : list[str] | None
        Panel letters to paint top-left (defaults to A,B,C,D).
    figsize : tuple[int, int]
        Figure size in inches.
    time_bins : list[int], optional
        Specific time bins to plot. If None, uses DEFAULT_TIME_BINS.
    """

    if len(gdf_panels) != 4 or len(df_points) != 4:
        raise ValueError("four_panel_isochrones expects four panels")

    if letters is None:
        letters = ["A", "B", "C", "D"]

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=500)
    axs = axs.flatten()

    for ax, gdf_u, df_p, title, letter in zip(
        axs, gdf_panels, df_points, titles, letters, strict=False
    ):
        plot_isochrones_panel(
            ax,
            germany_outline,
            gdf_u,
            df_p,
            title=title,
            panel_letter=letter,
            time_bins=time_bins,
        )

    return fig 