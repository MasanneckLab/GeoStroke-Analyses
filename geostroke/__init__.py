"""GeoStroke: Geographic analysis of stroke unit accessibility in Germany.

This package provides tools for analyzing stroke unit coverage and accessibility
across Germany using isochrone analysis and population data.
"""

from . import config
from . import data
from . import isochrones
from . import iso_manager
from . import coverage
from . import population
from . import figures
from . import reports
from . import interactive
from . import benefit
from . import urban_rural_annotation

__version__ = "1.0.0"

__all__ = [
    "config",
    "data", 
    "isochrones",
    "iso_manager",
    "coverage",
    "population",
    "figures",
    "reports",
    "interactive",
    "benefit",
    "urban_rural_annotation"
]

# Top-level helpers ----------------------------------------------------------

def run_publication_figures(output_dir: str | None = None, time_bins: list[int] | None = None) -> None:
    """Recreate the two publication figures and save them to *output_dir*.

    Parameters
    ----------
    output_dir : str | Path | None, optional
        Target directory; defaults to ``Graphs/`` at project root.
    time_bins : list[int] | None, optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS.
        Use config.DISPLAY_TIME_BINS for [15, 30, 45, 60] subset.
    """
    from pathlib import Path
    from .figures import create_figure_1, create_figure_2  # type: ignore

    out = Path(output_dir or 'Graphs')
    out.mkdir(exist_ok=True, parents=True)

    create_figure_1(out, time_bins=time_bins)
    create_figure_2(out, time_bins=time_bins)

    print('✓ Publication figures written to', out.resolve())


def run_publication_figures_extended(output_dir: str | None = None, time_bins: list[int] | None = None) -> None:
    """Recreate the extended publication figures and save them to *output_dir*.

    This generates the alternate Figure 1 with extended stroke units in a 2x2 layout:
    - A: Hospitals with CT
    - B: Frequent Stroke-Care Hospitals (extended stroke units)  
    - C: All Stroke Units
    - D: Thrombectomy-Certified

    Parameters
    ----------
    output_dir : str | Path | None, optional
        Target directory; defaults to ``Graphs/`` at project root.
    time_bins : list[int] | None, optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS.
        Use config.DISPLAY_TIME_BINS for [15, 30, 45, 60] subset.
    """
    from pathlib import Path
    from .figures import create_figure_1_extended, create_figure_2  # type: ignore

    out = Path(output_dir or 'Graphs')
    out.mkdir(exist_ok=True, parents=True)

    create_figure_1_extended(out, time_bins=time_bins)
    create_figure_2(out, time_bins=time_bins)

    print('✓ Extended publication figures written to', out.resolve())


def run_journal_publication_figures(output_dir: str | None = None, time_bins: list[int] | None = None) -> None:
    """Generate journal-compliant publication figures and save them to *output_dir*.

    Uses 190mm width (7.48 inches) with optimized layout for journal submission.

    Parameters
    ----------
    output_dir : str | Path | None, optional
        Target directory; defaults to ``Graphs/`` at project root.
    time_bins : list[int] | None, optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS.
        Use config.DISPLAY_TIME_BINS for [15, 30, 45, 60] subset.
    """
    from pathlib import Path
    from .figures import create_journal_figure_1, create_journal_figure_2  # type: ignore

    out = Path(output_dir or 'Graphs')
    out.mkdir(exist_ok=True, parents=True)

    create_journal_figure_1(out, time_bins=time_bins)
    create_journal_figure_2(out, time_bins=time_bins)

    print('✓ Journal publication figures written to', out.resolve())


def run_journal_publication_figures_extended(output_dir: str | None = None, time_bins: list[int] | None = None) -> None:
    """Generate journal-compliant extended publication figures and save them to *output_dir*.

    Uses 190mm width (7.48 inches) with optimized layout for journal submission.

    This generates the journal-compliant alternate Figure 1 with extended stroke units in a 2x2 layout:
    - a: Hospitals with CT
    - b: Frequent Stroke-Care Hospitals (extended stroke units)  
    - c: All Stroke Units
    - d: Thrombectomy-Certified

    Parameters
    ----------
    output_dir : str | Path | None, optional
        Target directory; defaults to ``Graphs/`` at project root.
    time_bins : list[int] | None, optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS.
        Use config.DISPLAY_TIME_BINS for [15, 30, 45, 60] subset.
    """
    from pathlib import Path
    from .figures import create_journal_figure_1_extended, create_journal_figure_2  # type: ignore

    out = Path(output_dir or 'Graphs')
    out.mkdir(exist_ok=True, parents=True)

    create_journal_figure_1_extended(out, time_bins=time_bins)
    create_journal_figure_2(out, time_bins=time_bins)

    print('✓ Journal extended publication figures written to', out.resolve())


def run_journal_publication_figures_standardized(output_dir: str | None = None, time_bins: list[int] | None = None) -> None:
    """Generate standardized journal-compliant publication figures with three_penalty_scenarios layout.

    Uses the same layout specifications as three_penalty_scenarios:
    - 3-panel figures use vertical layout (4.21, 11) 
    - 4-panel figure uses standardized styling
    - Consistent font styling and panel letter positioning
    - DPI=500 for all figures

    Parameters
    ----------
    output_dir : str | Path | None, optional
        Target directory; defaults to ``Graphs/`` at project root.
    time_bins : list[int] | None, optional
        Specific time bins to include. If None, uses DEFAULT_TIME_BINS.
        Use config.DISPLAY_TIME_BINS for [15, 30, 45, 60] subset.
    """
    from pathlib import Path
    from .figures import (
        create_journal_figure_1_standardized, 
        create_journal_figure_2_standardized,
        create_journal_figure_1_extended_standardized
    )

    out = Path(output_dir or 'Graphs')
    out.mkdir(exist_ok=True, parents=True)

    create_journal_figure_1_standardized(out, time_bins=time_bins)
    create_journal_figure_2_standardized(out, time_bins=time_bins)
    create_journal_figure_1_extended_standardized(out, time_bins=time_bins)

    print('✓ Standardized journal publication figures written to', out.resolve())


def get_all_time_bins() -> list[int]:
    """Return all available time bins for comprehensive analysis.
    
    Returns
    -------
    list[int]
        All time bins: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        
    Examples
    --------
    # Use all time bins for detailed analysis
    gs.run_publication_figures(time_bins=gs.get_all_time_bins())
    gs.reports.national_four_panel(..., time_bins=gs.get_all_time_bins())
    """
    return config.get_all_time_bins()


def get_display_time_bins() -> list[int]:
    """Return display time bins for publication figures.
    
    Returns
    -------
    list[int]
        Display time bins: [15, 30, 45, 60]
        
    Examples
    --------
    # Use display time bins for cleaner publication figures
    gs.run_publication_figures(time_bins=gs.get_display_time_bins())
    gs.reports.national_four_panel(..., time_bins=gs.get_display_time_bins())
    """
    return config.get_display_time_bins() 