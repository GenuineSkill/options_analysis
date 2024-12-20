# Technical Methodology

## Paper Implementation Details

### GARCH Model Ensemble
- 6 GARCH variants implemented:
  - GARCH(1,1) with Gaussian/t-distribution
  - E-GARCH(1,1) with Gaussian/t-distribution
  - GJR-GARCH(1,1) with Gaussian/t-distribution
- Each model estimated with minimum 5-year rolling window
- Holiday-adjusted log returns to prevent bias

### Holiday Detection and Trading Day Analysis
```python
# Holiday detection algorithm
is_holiday = (index_level_t == index_level_t_minus_1)

# NTRADE dummy creation
ntrade = (
    (day_of_week == 'Monday') | 
    (is_holiday.shift(-1)) | 
    (is_holiday.shift(-1) & is_holiday.shift(-2))  # Multi-day holidays
).astype(int)
```

### Key Statistical Measures
1. GEV (GARCH Expected Volatility):
   - Mean forecast across all models through option expiry
   ```python
   GEV_tau = (1/τ) * (1/J) * sum(
       sum(sigma_GARCH[j, t+i] for i in range(1, τ+1))
       for j in range(1, J+1)
   )
   ```

2. EVOEV (Expected Volatility of Expected Volatility):
   - Standard deviation of mean forecasts
   ```python
   EVOEV_tau = sqrt(
       (1/τ) * sum(
           (mean_GARCH_forecast[t+i] - GEV_tau)**2 
           for i in range(1, τ+1)
       )
   )
   ```

3. DEV (Dispersion of Expected Volatility):
   - Cross-sectional standard deviation
   ```python
   DEV_tau = (1/τ) * sum(
       std([sigma_GARCH[j,t+i] for j in range(J)])
       for i in range(1, τ+1)
   )
   ```

4. KEV (Skewness of Expected Volatility):
   - Cross-sectional skewness
   ```python
   KEV_tau = (1/τ) * sum(
       skew([sigma_GARCH[j,t+i] for j in range(J)])
       for i in range(1, τ+1)
   )
   ```

5. SEVTS (Slope of Expected Volatility Term Structure):
   ```python
   SEVTS_tau = (1/J) * sum(
       sigma_GARCH[j,τ] - sigma_GARCH[j,t+1]
       for j in range(J)
   )
   ```

### Expanding Window Analysis
- Initial window: 5 years minimum
- Incremental updates stored in DuckDB
- Database schema optimized for time series

### Strategy Implementation
1. Time Series Strategy:
   ```python
   r_TS = {
       r_straddle  if ε[t-1] <= -1
       -r_straddle if ε[t-1] >= 1
       0           if -1 < ε[t-1] < 1
   }
   ```

2. Cross-sectional Strategy:
   ```python
   r_CS = {
       r_straddle[τ_min] - r_straddle[τ_max]  if ε[τ_max] - ε[τ_min] >= 1
       0                                       otherwise
   }
   ```

## Database Schema Details

### Main Tables
```sql
-- Time series data
CREATE TABLE daily_data (
    date DATE,
    index_id VARCHAR,
    price DECIMAL,
    rf_rate DECIMAL,
    div_yield DECIMAL,
    is_holiday BOOLEAN
);

-- Implied volatilities
CREATE TABLE implied_vols (
    date DATE, 
    index_id VARCHAR,
    tenor VARCHAR,
    iv DECIMAL
);

-- GARCH parameters
CREATE TABLE garch_params (
    date DATE,
    index_id VARCHAR,
    model_type VARCHAR,
    parameters JSONB
);

-- Expanding window results
CREATE TABLE expanding_window_results (
    date DATE,
    index_id VARCHAR,
    window_start DATE,
    statistics JSONB
);
```

## Computational Efficiency Considerations
1. Development Mode:
   - Use 5-year subset
   - Cache intermediate results
   - Parallel GARCH estimation

2. Production Mode:
   - Full historical calculation
   - Incremental updates only
   - Optimized storage/retrieval
