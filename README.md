# Global Equity Options Analysis
## Project Overview
This repository implements the analysis methodology from Durham (2022) "Value for Equity Index Options: Expected—Not Realized—Volatility and the Distribution of Forecasts" with focus on SPX, SX5E, and UKX indices.

### Key Features
- GARCH model ensemble estimation and forecasting
- Expanding window analysis with efficient storage/retrieval
- Level and error-correction regression analysis  
- Options strategy backtesting
- Incremental update capability for new data

## Data Structure

### Input Data (daily frequency)
- Equity indices (1987-present): SPX, SX5E, UKX
- Risk-free rates: SPX_RF, SX5E_RF, UKX_RF 
- Dividend yields: SPX_DY, SX5E_DY, UKX_DY
- Implied volatilities (2004/2005-present):
  - Tenors: 1M, 2M, 3M, 6M, 12M
  - Format: {INDEX}_{TENOR} (e.g., SPX_1M)

### Holiday Detection and Trading Day Analysis
- NTRADE dummy variable creation:
  - Default: Mondays = 1
  - Mid-week holidays = 1 (for next trading day)
  - Holiday detection algorithm:
    ```python
    # Holiday detection (per index)
    is_holiday = (index_level_t == index_level_t_minus_1)
    
    # NTRADE dummy (per index)
    ntrade = (
        (day_of_week == 'Monday') | 
        (is_holiday.shift(-1)) | 
        (is_holiday.shift(-1) & is_holiday.shift(-2))  # Multi-day holidays
    ).astype(int)
    ```
- Holiday handling in GARCH estimation:
  - Exclude holidays when calculating log returns
  - Use only actual trading days for parameter estimation
  - Example:
    ```python
    # Correct log return calculation
    log_returns = np.log(
        prices[~is_holiday] / prices[~is_holiday].shift(1)
    )
    ```

### Database Design
Using DuckDB for efficient storage and querying of:
- Raw time series data
- GARCH model parameters
- Expanding window results
- Strategy backtesting results

Key advantages of DuckDB for this project:
- Column-oriented storage optimal for time series
- Efficient parallel processing
- SQL interface for complex queries
- Excellent integration with pandas
- Zero-dependency embedded database

## Project Components

### 1. Data Management (`data_manager/`)
- Data ingestion and cleaning
- DuckDB database initialization
- Incremental update detection
- Data validation and consistency checks

### 2. GARCH Models (`garch/`)
- Implementation of 6 GARCH variants:
  - GARCH(1,1) with Gaussian/t-distribution
  - E-GARCH(1,1) with Gaussian/t-distribution
  - GJR-GARCH(1,1) with Gaussian/t-distribution
- Parallel model estimation
- Rolling parameter estimation
- Forecast generation

### 3. Ensemble Statistics (`ensemble_stats/`)
- GEV (Mean GARCH Expected Volatility)
- EVOEV (Expected Volatility of Expected Volatility)
- DEV (Dispersion of Expected Volatility)
- KEV (Skewness of Expected Volatility)
- SEVTS (Slope of Expected Volatility Term Structure)

### 4. Regression Analysis (`regression/`)
- Expanding window level regressions
- First-difference regressions
- Error correction estimation
- Statistical inference

### 5. Strategy Backtesting (`strategy/`)
- BSM option pricing
- Delta-neutral portfolio construction
- Time series strategy implementation
- Cross-sectional strategy implementation
- Performance analytics

### 6. Results Storage (`results/`)
- Compressed storage of expanding window results
- Efficient retrieval mechanisms
- Incremental update functionality

## Key Equations for Implementation

### 1. GARCH Model Expected Volatility (GEV)
```python
# For option expiry τ and J GARCH models
GEV_tau = (1/τ) * (1/J) * sum(
    sum(sigma_GARCH[j, t+i] for i in range(1, τ+1))
    for j in range(1, J+1)
)
```

### 2. Expected Volatility of Expected Volatility (EVOEV)
```python
# Standard deviation of mean forecasts through expiry
EVOEV_tau = sqrt(
    (1/τ) * sum(
        (mean_GARCH_forecast[t+i] - GEV_tau)**2 
        for i in range(1, τ+1)
    )
)

# where mean_GARCH_forecast[t] = (1/J) * sum(sigma_GARCH[j,t] for j in range(J))
```

### 3. Dispersion of Expected Volatility (DEV)
```python
# Time-series average of cross-sectional standard deviation
DEV_tau = (1/τ) * sum(
    std([sigma_GARCH[j,t+i] for j in range(J)])
    for i in range(1, τ+1)
)
```

### 4. Skewness of Expected Volatility (KEV)
```python
# Time-series average of cross-sectional skewness
KEV_tau = (1/τ) * sum(
    skew([sigma_GARCH[j,t+i] for j in range(J)])
    for i in range(1, τ+1)
)
```

### 5. Slope of Expected Volatility Term Structure (SEVTS)
```python
# Mean slope of EV term structure within option horizon
SEVTS_tau = (1/J) * sum(
    sigma_GARCH[j,τ] - sigma_GARCH[j,t+1]
    for j in range(J)
)
```

### 6. Level Regression Model
```python
# For each index i and expiry τ
IV[i,τ] = β₀ + β₁*GEV + β₂*EVOEV + β₃*DEV + β₄*KEV + 
          β₅*SEVTS + β₆*NTRADE + ε
```

### 7. Error Correction Model
```python
# First difference regression
ΔIV[t] = α₀ + α₁*ΔX[t] + α_ε*ε[t-1] + μ[t]

# where ε[t-1] is the lagged residual from level regression
# and ΔX contains changes in all factors
```

### 8. Strategy Returns
```python
# Time series strategy returns
r_TS = {
    r_straddle  if ε[t-1] <= -1
    -r_straddle if ε[t-1] >= 1
    0           if -1 < ε[t-1] < 1
}

# Cross-sectional strategy returns
r_CS = {
    r_straddle[τ_min] - r_straddle[τ_max]  if ε[τ_max] - ε[τ_min] >= 1
    0                                       otherwise
}

# where τ_min and τ_max are tenors with min/max standardized valuations
```

## Optimization Approaches

### Computational Efficiency
1. Development Mode:
   - Use 5-year subset for testing (minimum needed for GARCH convergence)
   - Cache intermediate results
   - Parallel processing of GARCH models

2. Production Mode:
   - Full historical calculation
   - Store expanding window results
   - Incremental updates only

### Database Schema
```sql
-- Example key tables
CREATE TABLE daily_data (
    date DATE,
    index_id VARCHAR,
    price DECIMAL,
    rf_rate DECIMAL,
    div_yield DECIMAL
);

CREATE TABLE implied_vols (
    date DATE, 
    index_id VARCHAR,
    tenor VARCHAR,
    iv DECIMAL
);

CREATE TABLE garch_params (
    date DATE,
    index_id VARCHAR,
    model_type VARCHAR,
    parameters JSONB
);

CREATE TABLE expanding_window_results (
    date DATE,
    index_id VARCHAR,
    window_start DATE,
    statistics JSONB
);
```

## Usage Pipeline

1. Initial Setup:
```python
# Initialize database and load historical data
python setup_database.py

# Perform full historical calculation
python calculate_historical.py
```

2. Regular Updates:
```python
# Check for new data and update if needed
python update_analysis.py
```

3. Strategy Execution:
```python
# Run backtesting analysis
python run_strategy.py
```

## Implementation Notes

### Holiday Handling Requirements
1. Database schema must track holidays per index
2. GARCH estimation pipeline must:
   - Filter out holidays before parameter estimation
   - Handle gaps in time series appropriately
   - Adjust degrees of freedom calculations
3. Strategy implementation must:
   - Account for holiday effects on position holding periods
   - Adjust NTRADE dummy variable after holidays
   - Handle multi-day holiday periods correctly

## Dependencies
- Python 3.8+
- DuckDB
- NumPy
- Pandas
- Scipy
- arch (for GARCH models)
- statsmodels
- pytest (for testing)

## Testing
- Unit tests for each component
- Integration tests for full pipeline
- Validation against paper results

## Performance Considerations
- Full historical calculation: ~4-6 hours
- Daily updates: ~5-10 minutes
- Strategy backtesting: ~30-45 minutes

## Future Enhancements
- GPU acceleration for GARCH estimation
- Additional GARCH model variants
- Real-time strategy execution capabilities
- Web dashboard for results visualization
