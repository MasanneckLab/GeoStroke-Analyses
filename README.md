# GeoStroke-Analyses

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![Status](https://img.shields.io/badge/status-in%20review-yellow.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20|%20Linux%20|%20Windows-lightgrey.svg)
![OpenRouteService](https://img.shields.io/badge/OpenRouteService-required-orange.svg)

**Optimizing Prehospital Stroke Care: Mapping Stroke Unit Access and Leveraging CT Availability Across Germany – a Cross-sectional Analysis**

*Repository for the paper currently under review at Lancet Regional Health Europe*

Link to the paper: XXX Placeholder (under review)

## 🌐 Interactive Visualization

**📊 Explore the results interactively: [GeoStroke Visualizer](https://masannecklab.github.io/GeoStroke-Visualizer/)**

The interactive web application allows you to explore driving-time-based accessibility to stroke care across Germany, comparing direct transport to stroke units versus CT + telestroke strategies.


---

## Table of Contents

- [Overview](#overview)
- [Publication](#publication)
- [Key Features](#key-features)
- [The GeoStroke Module](#the-geostroke-module)
- [Installation](#installation)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Notebooks Overview](#notebooks-overview)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview

This repository contains the complete analysis pipeline for optimizing prehospital stroke care in Germany. Our research investigates whether it may be clinically faster to transport stroke patients to the nearest CT-equipped hospital for immediate imaging and telestroke consultation, rather than direct transport to certified stroke units. The analysis focuses on detecting where exactly this might be a useful approach in Germany. 

### Research Questions

- **Primary**: Can a "CT + telestroke" strategy reduce time to treatment compared to direct transport to stroke units?
- **Secondary**: How does this strategy perform across different geographic regions (urban vs. rural)?
- **Tertiary**: What are the population-level impacts under various scenario conditions?

### Methodology

Using **isochrone analysis** with OpenRouteService, we calculated driving times from every location in Germany to:
- 🏥 **CT-equipped hospitals** (1,400+ facilities)
- 🧠 **Certified stroke units** (340+ facilities)  
- 📊 **Frequent stroke hospitals** (100+ stroke cases/year - 400+ facities)

The analysis incorporates multiple scenarios including emergency vehicle speeds, traffic conditions, and telestroke consultation delays.

## Publication

📄 **Paper**: Masanneck et al. (2025), "Optimizing Prehospital Stroke Care: Mapping Stroke Unit Access and Leveraging CT Availability Across Germany – a Cross-sectional Analysis"

🔬 **Status**: Under review at *Lancet Regional Health Europe*

🌐 **Interactive Results**: [https://masannecklab.github.io/GeoStroke-Visualizer/](https://masannecklab.github.io/GeoStroke-Visualizer/)

## Key Features

### 🗺️ Comprehensive Geographic Coverage
- **State-level analysis**: All 16 German federal states
- **County-level analysis**: 400+ German counties (Landkreise/Kreisfreie Städte)
- **Population-weighted metrics**: Using GHS-POP 2025 data

### 🏥 Hospital Classification System
1. **CT-equipped hospitals**: Facilities with CT resources available
2. **Frequent stroke hospitals**: ≥100 stroke patients/year or certified stroke units - as analyzted by German official quality reports of hospitals
3. **Certified stroke units**: Specialized stroke treatment centers

### 📊 Advanced Analytics
- **Benefit analysis**: Comparing transport strategies to either frequent stroke-care hospitals or stroe units
- **Urban/rural stratification**: Using GHS-SMOD classification
- **Population impact assessment**: Accessibility metrics
- **Interactive visualizations**: Web-based exploration tools

### 🚗 Multiple Scenario Analysis for Benedit calculation
- **Normal Speed**: Baseline driving conditions
- **+20% Emergency Speed**: Emergency vehicle privileges  
- **-20% Traffic Speed**: Congested/adverse weather conditions
- **+10/20/30min Telestroke Penalty**: Remote consultation delays

## The GeoStroke Module

The `geostroke` package is designed as a partially **reusable module** for geographic healthcare accessibility analyses. It can be easily adapted for other countries, healthcare systems, or medical specialties with little modifications. The most important settings can be adjusted in **`config.py`**. 

### Core Components

- 🗺️ **`isochrones`**: Travel-time polygon generation
- 📊 **`coverage`**: Population coverage analysis  
- 🏥 **`data`**: Healthcare facility management
- 📈 **`figures`**: Publication-ready visualizations
- 🔍 **`benefit`**: Comparative strategy analysis
- 🏙️ **`urban_rural_annotation`**: Settlement classification
- 📋 **`reports`**: Automated result generation

### Reusability Features

- Configurable via environment variables
- Modular design for easy extension
- Comprehensive caching system to keep compute as low as possible
- Multi-scenario analysis framework
- Standardized data interfaces

## Installation

### Prerequisites

- **Python 3.13+** (tested on Python 3.13)
- **OpenRouteService** (local Docker instance recommended)
- **Sufficient disk space** (30+ GB for cache files and datasets)

### Environment Setup

1. **Create virtual environment**:
```bash
python3.13 -m venv geostroke-env
source geostroke-env/bin/activate  # On Windows: geostroke-env\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install geostroke module in development mode**:
```bash
pip install -e .
```

## Requirements

### Core Dependencies
- `geopandas` (≥1.0.0): Geospatial data manipulation
- `rasterio` (≥1.3): Raster data processing
- `openrouteservice` (≥2.3): Routing and isochrone generation
- `pandas` (≥2.0): Data analysis and manipulation
- `matplotlib` (≥3.7): Visualization and plotting
- `datashader` (≥0.18): Large-scale data visualization

### Development System
- **Platform**: MacBook Pro (Apple Silicon M4) with 48GB RAM -
- **Memory**: tested on 48GB RAM 
- **Storage**: 50+ GB free space for cache files and results

### External Services

#### OpenRouteService Setup
We recommend using the official Docker container for local deployment:

```bash
# Pull and run OpenRouteService with German OSM data
docker pull openrouteservice/openrouteservice:latest
docker run -dt --name ors-app -p 8080:8080 -v $(pwd)/graphs:/home/ors/ors-core/data/graphs \
  -v $(pwd)/elevation_cache:/home/ors/ors-core/data/elevation_cache \
  -v $(pwd)/logs:/var/log/ors \
  -v $(pwd)/conf:/home/ors/ors-conf \
  -e "REBUILD_GRAPHS=True" \
  -e "CONTAINER_LOG_LEVEL=INFO" \
  openrouteservice/openrouteservice:latest
```

**🔗 Docker Hub**: [https://hub.docker.com/r/openrouteservice/openrouteservice](https://hub.docker.com/r/openrouteservice/openrouteservice)

## Datasets

### Required Downloads

#### 1. GHS Population Data (2025 estimate)
- **File**: `GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif`
- **Source**: [Global Human Settlement Layer](https://human-settlement.emergency.copernicus.eu/download.php)
- **Location**: Place in `raw_data/` directory

#### 2. GHS Settlement Model Data  (2025 estimate)
- **File**: `GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0.tif`
- **Source**: [GHS Settlement Model](https://human-settlement.emergency.copernicus.eu/download.php)
- **Location**: Place in `raw_data/` directory

### Data Sources

#### Hospital Data
- **CT Hospitals**: Aggregated from [Deutsches Krankenhaus Verzeichnis](https://www.deutsches-krankenhaus-verzeichnis.de/app/suche)
- **Quality Reports**: Federal quality database ([G-BA Reference Database](https://qb-referenzdatenbank.g-ba.de/#/login))
  
  ⚠️ **Note**: Quality report access requires separate application. Processed datasets are included in repository.

#### Geographic Data
- **German boundaries**: Included in `shp/` directory
- **Administrative divisions**: States, counties, and municipalities

## Repository Structure

```
GeoStroke-Analyses/
├── 📁 geostroke/                    # Core analysis module
│   ├── __init__.py                  # Package initialization
│   ├── benefit.py                   # Dual strategy analysis
│   ├── config.py                    # Configuration management
│   ├── coverage.py                  # Population coverage metrics
│   ├── data.py                      # Data loading utilities
│   ├── figures.py                   # Publication figures
│   ├── isochrones.py               # Travel-time calculations
│   ├── iso_manager.py              # Isochrone cache management
│   ├── population.py               # Population analysis
│   ├── reports.py                  # Automated reporting
│   └── urban_rural_annotation.py   # Settlement classification
│
├── 📁 notebooks/                    # Analysis workflows
│   ├── 01_Isochrone_Generation.ipynb
│   ├── 02_end_to_end.ipynb
│   ├── 03_benefit_analysis.ipynb
│   └── 04_urban_rural_benefit_analysis.ipynb
│
├── 📁 raw_data/                     # Source datasets
│   ├── benefit_cache/               # Cached analysis results
│   ├── *.csv                        # Hospital and facility data
│   └── *.tif                        # Population and settlement rasters
│
├── 📁 Results/                      # Analysis outputs
│   ├── Counties/                    # County-level results
│   ├── States/                      # State-level results
│   └── Dual_All_Scenarios/         # Multi-scenario analysis
│
├── 📁 Graphs/                       # Publication figures
├── 📁 shp/                         # Geographic boundaries
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Notebooks Overview

### 📓 `01_Isochrone_Generation.ipynb`
**Purpose**: Generate travel-time polygons for all healthcare facilities

**Key Features**:
- 🔧 OpenRouteService connectivity testing
- 🗺️ Batch isochrone generation for 2,000+ facilities
- 💾 Intelligent caching system
- 📊 Progress tracking and error handling
- ✅ Validation and quality control

### 📓 `02_end_to_end.ipynb`  
**Purpose**: Complete workflow from data loading to publication figures for basic isochrone analysis

**Key Features**:
- 📊 Population coverage analysis
- 🎨 Publication-ready visualizations of unionized isochrones
- 📈 State and county-level aggregations of coverage
- 🗺️ Interactive map generation
- 📋 Statistical summaries

**Dependencies**: Requires completed isochrone generation

### 📓 `03_benefit_analysis.ipynb`
**Purpose**: Dual strategy comparison (CT + telestroke vs. direct transport)

**Key Features**:
- ⚖️ Time benefit calculations across scenarios
- 🗺️ Benefit mapping and visualization  
- 📊 Population impact assessment
- 🏥 Facility-specific analysis
- 📈 Multi-scenario comparisons

**Output**: Comprehensive benefit maps and population statistics
**Dependencies**: Requires completed isochrone generation


### 📓 `04_urban_rural_benefit_analysis.ipynb`
**Purpose**: Urban/rural stratification of accessibility patterns

**Key Features**:
- 🏙️ GHS-SMOD urban/rural classification
- 📊 Settlement-stratified analysis
- 🌾 Rural healthcare access patterns
- 🏘️ Urban coverage density analysis
- 📈 Population-weighted accessibility metrics

**Dependencies**: Requires previous notebooks

## Usage

### Quick Start

1. **Setup environment** (see [Installation](#installation))

2. **Download required datasets** (see [Datasets](#datasets))

3. **Start OpenRouteService**:
```bash
docker run -p 8080:8080 openrouteservice/openrouteservice:latest
```

4. **Run Notebooks**
```

### Notebook Execution Order

1. **Start with isochrones**: `01_Isochrone_Generation.ipynb`
2. **Basic analysis**: `02_end_to_end.ipynb`  
3. **Strategy comparison**: `03_benefit_analysis.ipynb`
4. **Urban/rural analysis**: `04_urban_rural_benefit_analysis.ipynb`

### Configuration

Key settings can be modified via environment variables:

```bash
export GEOSTROKE_DATA="/path/to/data"
export GEOSTROKE_RESULTS="/path/to/results"
export ORS_BASE_URL="http://localhost:8080/ors"
```

## Results

### Generated Outputs

- 🗺️ **Interactive maps**: County and state-level visualizations
- 📊 **Excel reports**: Detailed population and accessibility metrics
- 📈 **Publication figures**: High-resolution publication-ready graphics
- 🎨 **PDF summaries**: Comprehensive result compilations

### Key Findings

Results are available through the [interactive visualizer](https://masannecklab.github.io/GeoStroke-Visualizer/). Key patterns include:

- 🌾 **Rural areas** benefit significantly from CT + telestroke strategy
- 🏙️ **Urban areas** show more equivalent access patterns
- ⏱️ **Time savings** of 10-30 minutes possible in rural regions
- 📊 **Population impact** varies significantly by scenario and region

## Contributing

We welcome contributions to improve the analysis methodology, extend geographic coverage, or adapt the framework for other healthcare systems.
In this case just open another branch and get started! 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or methodology in your research, please cite (to be updated after potential acceptance):

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

## Acknowledgments

- **OpenRouteService** for routing and isochrone services
- **Global Human Settlement Layer** for population and settlement data
- **German Hospital Directory** for facility location data
- **Federal Quality Assurance** for stroke unit certification data

## Contact

For questions about the methodology, data access, or collaboration opportunities:

📧 **Lars Masanneck**: [Email](mailto:lars.masanneck@med.uni-duesseldorf.de)  


---

**🌐 [Explore Results Interactively](https://masannecklab.github.io/GeoStroke-Visualizer/) | 📊 [View Publication](link-when-published) | 💻 [Source Code](https://github.com/masannecklab/GeoStroke-Analyses)** 