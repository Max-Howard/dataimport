# Data Import For ADS-B Informed OpenAVEM

This project provides tools and scripts for importing and processing ADS-B and MERRA-2 wind data needed for the ADS-B informed fork of openAVEM.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Flight Data Processing](#flight-data-processing)
  - [Import Process](#import-process)
  - [Processing and Cleaning](#processing-and-cleaning)
  - [Output structure](#output-structure)
- [Wind Data](#wind-data)
  - [Wind Data Download](#wind-data-download)
  - [Processed for ADS-B GS to TAS Conversion (Time-Retained)](#processed-for-ads-b-gs-to-tas-conversion-time-retained)
  - [For OpenAVEM (Time-Averaged)](#for-openavem-time-averaged)

## Features

- **ADS-B Data Import**: Download flight trajectories from OpenSky Network
- **MERRA-2 Wind Data Processing**: Process wind data for use in this repo and in openAVEM
- **True Airspeed Calculation**: Convert ground speed to true airspeed using wind data
- **Flight Trajectory Processing**: Remove anomalies and simplify trajectories

## Prerequisites

- Conda package manager
- OpenSky Network account (for flight data access)
- Access to the BADA performance files

## Installation

1. **Clone the repository**

2. **Create and activate the conda environment**:

   ```bash
   conda env create -f environment.yml
   conda activate opensky-dataimport
   ```

3. **Set up OpenSky credentials**:

   To find the path to the file where you must add your opensky credentials, run the following:

   ```python
   from pyopensky.config import opensky_config_dir
   print(opensky_config_dir)
   ```

4. **Add BADA files**:

   Place BADA performance files in the `BADA/` directory (required for flight import and/or futher analysis).

## Flight Data Processing

### Import Process

Flight path data can be imported using the `python import_test_flights.py` script.

**Configuration parameters** (edit in script):

- `scanning_start`/`scanning_stop`: Time range for flight search (landing time of flights)
- `region`: Geographic filter ("EU", "US", "AS", "AF", "OC", "SA", "AN", or None)
- `limit`: Maximum number of flights to download

### Processing and Cleaning

The `process_adsb_data.py` script performs data cleaning and simplification

**Configuration parameters** (edit in script):

- `CALC_TAS`: Weather or not to calculate the TAS from wind data and GS
- `TIME_SLICE`: Index slice of MERRA-2 files to load into memory, used to keep RAM usage reasonable. Ensure that all time in the ADS-B data is is covered when slicing

### Output structure

```text
FlightData/
├── Raw/                    # Downloaded ADS-B trajectories
├── Processed/              # Cleaned and processed flights
├── flights.csv             # Flight list (used to resume downloads)
└── run_parameters.json     # Import configuration (to ensure resuming with same config)
```

## Wind Data

### Wind Data Download

The `download_wind_files.py` script is used to download the unprocessed MERRA-2 files.

### Processed for ADS-B GS to TAS Conversion (Time-Retained)

To prepare the files for GS to TAS conversion for the ADS-B data `process_wind_data_adsb.py` is used. This script concatenates the files into monthly files, removes unneeded data fields, converts U and V magnitudes into wind speed and direction, removes altitudes outside aircraft operating ranges, and adds altitudes corresponding to pressure levels.

### For OpenAVEM (Time-Averaged)

For the wind file required by openAVEM a similar process is employed using `process_wind_data_openavem.py`, however in this case the results are time averaged.

## Other Data

- **Aircraft Data** sourced from opensky is included in the repo. To get an up to date copy, download the database from opensky, and process it with ``clean_aircraft_db.py``.

- **Airport Data** is sourced from the openAVEM repo and used as is in the data filtering step to ensure that flight data starts close to origin airport and ends near destination.
