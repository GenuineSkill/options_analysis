# Documentation Directory

This directory contains technical documentation and reference materials for the options analysis project.

## Files Overview

### sample_data_format.csv
- Sample data file showing the structure of input data
- Contains truncated time series for:
  - Index levels (SPX, SX5E, UKX)
  - Risk-free rates (*_RF)
  - Dividend yields (*_DY)
  - Implied volatilities (*_1M, *_2M, etc.)
- Used as reference for data ingestion and validation

### setup_project.py
- Python script for creating project directory structure
- Creates all necessary directories and files
- Initializes Python packages with __init__.py files
- Sets up testing framework
- Creates base configuration files
- Use this script when setting up a new development environment

### methodology.md
- Detailed technical implementation guide
- Documents key equations and algorithms
- Explains GARCH model ensemble methodology
- Details holiday detection and handling
- Provides database schema and optimization strategies
- Reference for key statistical measures and calculations

## Using These Documents

### For Developers
- Start with main README.md in project root for overview
- Use setup_project.py to create development environment
- Reference methodology.md for implementation details
- Use sample_data_format.csv to validate data ingestion

### For Data Scientists
- Review methodology.md for statistical methodology
- Reference sample_data_format.csv for data requirements
- Understand holiday handling and GARCH estimation procedures

### For Database Engineers
- Check methodology.md for database schema details
- Review computational efficiency considerations
- Understand expanding window storage requirements

## Maintaining Documentation
When making significant changes to the project:
1. Update relevant documentation files
2. Keep sample data format current
3. Ensure setup script reflects any new requirements
4. Update methodology document with new procedures

## Additional Resources
- Original paper: Durham (2022) "Value for Equity Index Options: Expected—Not Realized—Volatility and the Distribution of Forecasts"
- Project GitHub repository
- DuckDB documentation for database optimization

## Data Files

### Required Data File
- File: "Global Equity Vol Durham 12.13.2024.xlsx"
- Location: Place in `data_manager/data/` directory
- Not tracked in Git repository due to size/update frequency
- Required columns match format shown in sample_data_format.csv
- Contains daily data for:
  - Equity indices (SPX, SX5E, UKX) from 1987
  - Risk-free rates (*_RF)
  - Dividend yields (*_DY)
  - Implied volatilities (*_1M through *_12M) from 2004/2005

### Data Setup Instructions
1. Create `data_manager/data/` directory if it doesn't exist
2. Copy "Global Equity Vol Durham 12.13.2024.xlsx" to this directory
3. This location is git-ignored (.gitignore contains `data_manager/data/*`)
4. Each developer needs to manually add this file to their local repository
5. Data updates should be managed outside of git version control

### Data Validation
- Use `sample_data_format.csv` in docs/ to verify data structure
- Required date range: 1987-present for index data
- Required columns must match sample format exactly
- Data cleaning scripts will validate format on load