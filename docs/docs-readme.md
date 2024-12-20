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
