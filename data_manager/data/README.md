# Data Directory

This directory should contain the following data file(s):
- "Global Equity Vol Durham 12.13.2024.xlsx"

## Important Notes
- Data files in this directory are NOT tracked in git
- Each developer must manually add required data files
- Contact project maintainers to obtain the latest data files
- See main documentation in docs/README.md for detailed data requirements

## Data File Requirements
- Daily frequency data from 1987-present
- Index levels: SPX, SX5E, UKX
- Risk-free rates: SPX_RF, SX5E_RF, UKX_RF
- Dividend yields: SPX_DY, SX5E_DY, UKX_DY
- Implied volatilities: Series like SPX_1M, SPX_2M, etc.

## Validation
Run data validation scripts before processing:
```python
python data_manager/data_loader.py --validate
```