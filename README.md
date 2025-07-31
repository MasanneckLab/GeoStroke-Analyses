# GeoStroke-Analyses: Optimizing Prehospital Stroke Care Access in Germany

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](#requirements)
[![Status](https://img.shields.io/badge/status-in%20review-yellow.svg)](#publication)
[![Platform](https://img.shields.io/badge/platform-macOS%20|%20Linux%20|%20Windows-lightgrey.svg)](#system-requirements)
[![OpenRouteService](https://img.shields.io/badge/OpenRouteService-required-orange.svg)](#openrouteservice-configuration)

## 📄 Publication

**Masanneck et al. (2025)**. "Optimizing Prehospital Stroke Care: Mapping Stroke Unit Access and Leveraging CT Availability Across Germany – a Cross-sectional Analysis"

**Status**: Under review at *Lancet Regional Health Europe*

**Interactive Results**: [GeoStroke Visualizer](https://masannecklab.github.io/GeoStroke-Visualizer/)

### Citation

```bibtex
@article{masanneck2025geostroke,
  title={Optimizing Prehospital Stroke Care: Mapping Stroke Unit Access and Leveraging CT Availability Across Germany – a Cross-sectional Analysis},
  author={Masanneck, Lars and [Authors]},
  journal={Lancet Regional Health Europe},
  year={2025},
  status={Under Review}
}
```

### Software Citation

```bibtex
@software{geostroke2025,
  title={GeoStroke: Geographic Analysis of Stroke Unit Accessibility},
  author={Masanneck, Lars},
  year={2025},
  url={https://github.com/masannecklab/GeoStroke-Analyses},
  version={1.0.0}
}
```

---

## Overview

This repository provides a **production-ready, reproducible analysis pipeline** for optimizing prehospital stroke care accessibility in Germany. The framework implements advanced geospatial analysis methodologies to compare direct transport to stroke units versus CT-equipped hospital strategies with telestroke consultation.

### Research Objectives

- **Primary Hypothesis**: Determine if CT + telestroke strategy reduces time-to-treatment compared to direct stroke unit transport
- **Geographic Analysis**: Quantify accessibility patterns across urban/rural gradients using high-resolution population data
- **Policy Impact Assessment**: Evaluate population-level coverage under multiple transport scenarios

### Technical Approach

The analysis employs **isochrone-based accessibility modeling** using the OpenRouteService API to generate travel-time polygons for:

- **1,566 CT-equipped hospitals** across Germany
- **349 certified stroke units** (comprehensive stroke centers)
- **465 frequent stroke-care hospitals** (≥100 cases/year)

Population coverage is calculated using the **Global Human Settlement Layer (GHS-POP 2025)** at 3-arcsecond resolution (~100m), providing precise demographic weighting for accessibility metrics.

#### Global Human Settlement Layer (GHS) Datasets

| Dataset | Purpose | Resolution | Size | Download Link |
|---------|---------|------------|------|---------------|
| **GHS-POP 2025** | Population distribution and density | 3-arcsecond (~100m) | ~10GB | [Download GHS-POP](https://ghsl.jrc.ec.europa.eu/download.php?ds=ghs_pop_2025) |
| **GHS-SMOD 2025** | Settlement typology (urban/rural classification) | 1km | ~2GB | [Download GHS-SMOD](https://ghsl.jrc.ec.europa.eu/download.php?ds=ghs_smod_2025) |

#### Hospital Data
- **CT Hospitals**: Aggregated from [Deutsches Krankenhaus Verzeichnis](https://www.deutsches-krankenhaus-verzeichnis.de/app/suche)
- **Quality Reports**: Federal quality database ([G-BA Reference Database](https://qb-referenzdatenbank.g-ba.de/#/login))
  
  ⚠️ **Note**: Quality report access requires separate application. Processed datasets are included in repository.
---

## 🚀 Quick Start

### Installation

```bash
# Clone repository and setup environment
git clone https://github.com/masannecklab/GeoStroke-Analyses.git
cd GeoStroke-Analyses
python3.12 -m venv geostroke-env
source geostroke-env/bin/activate  # Windows: geostroke-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Essential Data Setup

```bash
# Download GHS Population data (required for analysis)
# Place GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif in raw_data/
wget "https://ghsl.jrc.ec.europa.eu/download.php?ds=ghs_pop_2025" \
     -O raw_data/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif

# Download GHS Settlement Model data (for urban/rural classification)
# Place GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0.tif in raw_data/
```

### OpenRouteService Configuration

```bash
# Option 1: Local Docker deployment (recommended)
docker run -d --name ors-app -p 8080:8080 \
  -v $(pwd)/graphs:/home/ors/ors-core/data/graphs \
  -e "REBUILD_GRAPHS=True" \
  openrouteservice/openrouteservice:latest

# Option 2: Use hosted API (configure API key in environment)
export ORS_API_KEY="your_api_key_here"
export ORS_BASE_URL="https://api.openrouteservice.org"
```

### Basic Analysis Workflow

```python
import geostroke as gs

# 1. Generate publication figures with standard time bins
gs.run_publication_figures(time_bins=gs.get_display_time_bins())  # [15, 30, 45, 60]

# 2. Generate comprehensive analysis with all time bins
gs.run_publication_figures(time_bins=gs.get_all_time_bins())      # [5, 10, ..., 60]

# 3. Create journal-compliant figures
gs.run_journal_publication_figures_standardized()

# 4. Advanced programmatic usage
df_stroke = gs.data.load_stroke_units()
df_ct = gs.data.load_hospitals_ct()
germany = gs.data.load_germany_outline()

# Generate isochrones with caching
polygons_stroke = gs.iso_manager.ensure_polygons(df_stroke, time_bins=[15, 30, 45, 60])
polygons_ct = gs.iso_manager.ensure_polygons(df_ct, suffix="_all_CTs", time_bins=[15, 30, 45, 60])

# Calculate population coverage
coverage = gs.coverage.national_table(polygons_stroke, time_bins=[15, 30, 45, 60])
```

### Automated Batch Processing

```bash
# Test OpenRouteService connectivity
python batch_isochrone_generation.py --test-only

# Generate all isochrones (uses intelligent caching)
python batch_isochrone_generation.py

# Force complete regeneration
python batch_isochrone_generation.py --force-recalc
```

### Interactive Analysis

```bash
# Launch Jupyter Lab for interactive exploration
jupyter lab notebooks/01_Isochrone_Generation.ipynb
jupyter lab notebooks/02_end_to_end.ipynb
jupyter lab notebooks/03_benefit_analysis.ipynb
jupyter lab notebooks/04_urban_rural_benefit_analysis.ipynb
```

---

## 📊 Repository Structure

### Core Analysis Module

| File | Description | Lines | Size |
|------|-------------|-------|------|
| `geostroke/__init__.py` | Public API and main functions | 210 | 7.3KB |
| `geostroke/config.py` | Configuration management | 175 | 6.2KB |
| `geostroke/data.py` | Data loading and preprocessing | 169 | 5.7KB |
| `geostroke/isochrones.py` | Core isochrone generation | 112 | 3.6KB |
| `geostroke/iso_manager.py` | Robust isochrone management with caching | 788 | 32KB |
| `geostroke/coverage.py` | Population coverage calculations | 100 | 3.7KB |
| `geostroke/population.py` | Raster-based population analysis | 68 | 1.9KB |
| `geostroke/figures.py` | Publication-ready visualization | 1537 | 64KB |
| `geostroke/plotting.py` | Core plotting utilities | 334 | 10KB |
| `geostroke/interactive.py` | Interactive Plotly visualizations | 130 | 3.9KB |
| `geostroke/benefit.py` | Dual strategy analysis | 3455 | 137KB |
| `geostroke/reports.py` | Automated report generation | 234 | 8.7KB |
| `geostroke/urban_rural_annotation.py` | Settlement classification | 171 | 5.9KB |

### Analysis Notebooks

| Notebook | Purpose | Size | Description |
|----------|---------|------|-------------|
| `01_Isochrone_Generation.ipynb` | Isochrone generation pipeline | 995KB | Batch processing of 7,660+ polygons with retry logic |
| `02_end_to_end.ipynb` | Complete workflow demonstration | 75MB | Population coverage analysis and visualization |
| `03_benefit_analysis.ipynb` | Dual strategy comparison | 16MB | CT + telestroke vs. direct transport analysis |
| `04_urban_rural_benefit_analysis.ipynb` | Urban/rural stratification | 68KB | GHS-SMOD classification and accessibility patterns |
| `additional_stroke_centers.ipynb` | Extended stroke unit analysis | 22KB | Supplementary facility analysis |

### Source Data Files

| File | Purpose | Size | Source |
|------|---------|------|--------|
| `raw_data/stroke_units_geocoded.csv` | Certified stroke units (349 facilities) | 98KB | Processed from quality reports |
| `stroke_units_extended_geocoded.csv` | Frequent stroke-care hospitals (465 facilities) | 138KB | Quality reports + hospital directory |
| `raw_data/Hospitals_with_CT.xlsx` | CT-equipped hospitals (1,566 facilities) | Included | German hospital directory |
| `shp/germany-states.geojson` | Federal state boundaries | 99KB | Administrative boundaries |
| `shp/georef-germany-kreis@public.geojson` | County boundaries (401 counties) | 16MB | Administrative boundaries |
| `shp/germany-detailed-boundary_917.geojson` | National boundary | 65KB | Administrative boundaries |

### Cached Isochrone Files

| Pattern | Description | Count | Total Size |
|---------|-------------|-------|------------|
| `poly{time}_{facility_type}_{scenario}.pkl` | Cached isochrone polygons | 150+ files | >2GB |
| Time bins: | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 minutes | 12 bins | |
| Facility types: | stroke units, extended_stroke, all_CTs | 3 types | |
| Scenarios: | normal, emergency (+20% speed), bad_traffic (-20% speed) | 3 scenarios | |

### Results and Outputs

| Directory | Content | Count | Description |
|-----------|---------|-------|-------------|
| `Results/Counties/` | County-level analysis by state | 800+ files | Population tables and maps |
| `Results/States/` | State-level aggregations | 33 files | Summary statistics and visualizations |
| `Results/Dual_All_Scenarios/` | Multi-scenario benefit analysis | 5000+ files | Comprehensive scenario comparisons |
| `Graphs/` | Publication figures | 20 files | High-resolution EPS and PNG outputs |

---

## 🛠️ System Requirements

### Hardware Specifications
- **RAM**: 16GB minimum, 32GB+ recommended for large-scale analysis
- **Storage**: 50GB free space for cache files and results
- **CPU**: Multi-core processor recommended for parallel processing

### Software Dependencies

#### Core Scientific Stack
```
python>=3.12
geopandas>=1.0.0,<2.0
rasterio>=1.3,<2.0
pandas>=2.0,<3.0
numpy>=1.24,<2.0
matplotlib>=3.7,<4.0
```

#### Geospatial Analysis
```
openrouteservice>=2.3,<3.0
shapely>=2.0,<3.0
pyproj>=3.7
fiona>=1.10
```

#### Visualization and Reporting
```
plotly>=5.20,<6.0
datashader>=0.18
seaborn>=0.13,<0.14
bokeh>=3.4,<4.0
```

### External Services

#### OpenRouteService Requirements
- **Local deployment**: Docker with 8GB+ allocated memory
- **API access**: Valid API key with sufficient quota
- **Network**: Stable internet connection for routing requests

---

## 🔧 Advanced Configuration

### Environment Variables

```bash
# Data paths
export GEOSTROKE_DATA="/path/to/data"
export GEOSTROKE_RESULTS="/path/to/results"
export GEOSTROKE_POP_RASTER="/path/to/GHS_POP_2025.tif"

# OpenRouteService configuration
export ORS_BASE_URL="http://localhost:8080/ors"
export ORS_API_KEY="your_api_key"
export ORS_TIMEOUT=5000

# Analysis parameters
export GEOSTROKE_DEFAULT_TIME_BINS="15,30,45,60"
export GEOSTROKE_CACHE_DIR="/path/to/cache"
```

### Time Bin Configuration

The framework supports flexible time bin selection across all analysis functions:

```python
# Predefined sets
gs.get_all_time_bins()     # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
gs.get_display_time_bins() # [15, 30, 45, 60] - publication standard

# Custom configurations
custom_bins = [10, 20, 30]  # Application-specific analysis
gs.run_publication_figures(time_bins=custom_bins)
```

### Performance Optimization

#### Caching Strategy
- **Intelligent polygon caching**: Avoids redundant API calls
- **Benefit analysis caching**: Stores computationally expensive calculations
- **Population analysis caching**: Optimized raster processing

#### Parallel Processing
- **Concurrent isochrone generation**: Configurable worker pools
- **Vectorized population calculations**: NumPy/Rasterio optimization
- **Batch polygon operations**: GeoPandas spatial indexing

---

## 🧪 Validation and Quality Assurance

### Reproducibility Features
- **Deterministic processing**: Consistent results across platforms
- **Version-pinned dependencies**: Exact environment reproduction
- **Comprehensive logging**: Detailed operation tracking
- **Statistical validation**: Population coverage verification

### Quality Control Metrics
- **Polygon validity checks**: Geometry validation and repair
- **Population coverage validation**: Cross-reference with official statistics
- **Distance calculation verification**: Ground truth validation
- **Multi-scenario consistency**: Results coherence across scenarios

### Performance Benchmarks
- **Isochrone generation**: ~65 minutes for complete dataset (7,660 polygons)
- **Population analysis**: ~15 minutes for national coverage calculation
- **Figure generation**: ~5 minutes for publication-ready outputs
- **Memory efficiency**: Peak usage <16GB for full analysis

---

## 📈 Analysis Capabilities

### Accessibility Metrics
- **Population coverage**: Inhabitants within time thresholds
- **Geographic coverage**: Area-based accessibility analysis
- **Urban/rural stratification**: Settlement-type specific metrics
- **Multi-modal comparison**: Transport strategy effectiveness

### Scenario Analysis
- **Emergency vehicle privileges**: +20% speed increase
- **Adverse conditions**: -20% speed reduction (traffic/weather)
- **Telestroke delays**: +10/20/30 minute consultation penalties
- **Sensitivity analysis**: Parameter variation assessment

### Statistical Outputs
- **National summaries**: Population-weighted accessibility
- **State-level aggregations**: Federal comparison metrics
- **County-level analysis**: Local accessibility patterns
- **Facility-specific metrics**: Individual hospital coverage

---

## 🤝 Contributing

### Development Setup
```bash
# Development installation with linting/testing tools
pip install -e ".[dev]"
pre-commit install

# Run quality checks
black geostroke/
flake8 geostroke/
mypy geostroke/
```

### Extension Guidelines
- **Geographic adaptation**: Modify `config.py` for other countries
- **Healthcare specialties**: Extend facility classification system
- **Transport modes**: Add public transit/helicopter analysis
- **Temporal analysis**: Integrate historical accessibility trends

---

## 📚 Documentation and Support

### API Documentation
Comprehensive docstrings throughout the codebase provide detailed function documentation:

```python
help(gs.run_publication_figures)
help(gs.iso_manager.ensure_polygons)
help(gs.coverage.national_table)
```

### Troubleshooting

#### Common Issues
1. **Missing GHS data**: Download population raster separately
2. **OpenRouteService connectivity**: Verify Docker deployment/API key
3. **Memory constraints**: Reduce time bins or use chunked processing
4. **Large file handling**: Configure Git LFS for cache files

#### Performance Optimization
- Use local OpenRouteService for faster processing
- Configure SSD storage for cache files
- Allocate sufficient RAM for large raster operations
- Monitor disk space during extensive analysis

### Community Resources
- **GitHub Issues**: Bug reports and feature requests
- **Interactive Visualizer**: Results exploration platform
- **Publication**: Detailed methodology and validation

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🎯 Professional Applications

### Research and Academia
- **Publication-ready analysis**: Automated figure generation with journal specifications
- **Methodology validation**: Reproducible workflows with comprehensive documentation
- **Cross-country adaptation**: Configurable framework for international studies

### Healthcare Policy
- **Access equity assessment**: Urban/rural disparities quantification
- **Resource optimization**: Facility placement and capacity planning
- **Emergency response planning**: Transport strategy optimization

### Technical Integration
- **API-first design**: Programmatic access to all functionality
- **Modular architecture**: Component-level extensibility
- **Production deployment**: Robust error handling and logging

**🌐 [Explore Interactive Results](https://masannecklab.github.io/GeoStroke-Visualizer/) | 📊 [View Source Code](https://github.com/masannecklab/GeoStroke-Analyses) | 📧 [Contact](mailto:lars.masanneck@med.uni-duesseldorf.de)** 
