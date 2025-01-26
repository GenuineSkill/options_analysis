-- This is a script to migrate all tables and data into a new duckdb file
-- The old historical_results.db file was corrupted in some weird way
-- To create the new db file run: duckdb -init db_sql.sql new_historical_results.db

CREATE SEQUENCE IF NOT EXISTS window_id_seq;
CREATE TABLE IF NOT EXISTS forecast_windows (
    window_id INTEGER DEFAULT(nextval('window_id_seq')) PRIMARY KEY,
    index_id VARCHAR NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    "returns" BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT(CURRENT_TIMESTAMP)
);

CREATE TABLE IF NOT EXISTS ensemble_stats (
    window_id INTEGER,
    horizon VARCHAR NOT NULL,
    horizon_days INTEGER NOT NULL,
    gev DOUBLE,
    evoev DOUBLE,
    dev DOUBLE,
    kev DOUBLE,
    sevts DOUBLE,
    n_models INTEGER,
    FOREIGN KEY (window_id) REFERENCES forecast_windows(window_id)
);

CREATE TABLE IF NOT EXISTS error_correction_coefficients (
    date DATE,
    index_id VARCHAR,
    tenor VARCHAR,
    "variable" VARCHAR,
    coefficient DECIMAL(18,3),
    t_stat DECIMAL(18,3),
    p_value DECIMAL(18,3),
    PRIMARY KEY(date, index_id, tenor, "variable")
);

CREATE TABLE IF NOT EXISTS error_correction_results (
    date DATE,
    index_id VARCHAR,
    tenor VARCHAR,
    delta_iv DECIMAL(18,3),
    delta_gev DECIMAL(18,3),
    delta_evoev DECIMAL(18,3),
    delta_dev DECIMAL(18,3),
    delta_kev DECIMAL(18,3),
    delta_sevts DECIMAL(18,3),
    error_correction DECIMAL(18,3),
    fitted_change DECIMAL(18,3),
    residual DECIMAL(18,3),
    r_squared DECIMAL(18,3),
    PRIMARY KEY(date, index_id, tenor)
);

CREATE TABLE IF NOT EXISTS expanding_window_coefficients (
    window_end_date DATE,
    index_id VARCHAR,
    tenor VARCHAR,
    "variable" VARCHAR,
    coefficient DECIMAL(18,3),
    t_stat DECIMAL(18,3),
    p_value DECIMAL(18,3),
    PRIMARY KEY(window_end_date, index_id, tenor, "variable")
);

CREATE TABLE IF NOT EXISTS expanding_window_regressions
    (window_end_date DATE,
    index_id VARCHAR,
    tenor VARCHAR,
    window_start_date DATE,
    n_observations INTEGER,
    r_squared DECIMAL(18,3),
    adj_r_squared DECIMAL(18,3),
    PRIMARY KEY(window_end_date, index_id, tenor)
);

CREATE TABLE IF NOT EXISTS expanding_window_residuals
    (date DATE,
    index_id VARCHAR,
    tenor VARCHAR,
    window_end_date DATE,
    residual DECIMAL(18,3),
    standardized_residual DECIMAL(18,3),
    iv_actual DECIMAL(18,3),
    iv_fitted DECIMAL(18,3),
    PRIMARY KEY(date, index_id, tenor, window_end_date)
);

CREATE TABLE IF NOT EXISTS garch_results (
    window_id INTEGER,
    model_type VARCHAR NOT NULL,
    distribution VARCHAR NOT NULL,
    parameters VARCHAR NOT NULL,
    forecast_path BLOB NOT NULL,
    volatility_path BLOB NOT NULL,
    CHECK((model_type IN ('garch', 'egarch', 'gjrgarch'))),
    CHECK((distribution IN ('normal', 'studentst'))),
    FOREIGN KEY (window_id) REFERENCES forecast_windows(window_id)
);

CREATE TABLE IF NOT EXISTS implied_volatilities (
    date TIMESTAMP,
    index_id VARCHAR,
    tenor VARCHAR,
    implied_vol DOUBLE,
    is_trading_day BOOLEAN
);

CREATE TABLE IF NOT EXISTS implied_vol_aligned (
    date TIMESTAMP,
    index_id VARCHAR,
    tenor VARCHAR,
    implied_vol DOUBLE,
    is_trading_day BOOLEAN
);

CREATE TABLE IF NOT EXISTS regression_residuals (
    date DATE,
    index_id VARCHAR,
    tenor VARCHAR,
    residual DECIMAL(18,3),
    standardized_residual DECIMAL(18,3),
    window_id INTEGER,
    iv_actual DECIMAL(18,3),
    iv_fitted DECIMAL(18,3),
    FOREIGN KEY (window_id) REFERENCES forecast_windows(window_id),
    PRIMARY KEY(date, index_id, tenor)
);

ATTACH 'results/historical/historical_results.db' AS old_db;

INSERT INTO forecast_windows SELECT * FROM old_db.forecast_windows;
INSERT INTO ensemble_stats SELECT * FROM old_db.ensemble_stats;
INSERT INTO error_correction_coefficients SELECT * FROM old_db.error_correction_coefficients;
INSERT INTO error_correction_results SELECT * FROM old_db.error_correction_results;
INSERT INTO expanding_window_coefficients SELECT * FROM old_db.expanding_window_coefficients;
INSERT INTO expanding_window_regressions SELECT * FROM old_db.expanding_window_regressions;
INSERT INTO expanding_window_residuals SELECT * FROM old_db.expanding_window_residuals;
INSERT INTO garch_results SELECT * FROM old_db.garch_results;
INSERT INTO implied_volatilities SELECT * FROM old_db.implied_volatilities;
INSERT INTO implied_vol_aligned SELECT * FROM old_db.implied_vol_aligned;
INSERT INTO regression_residuals SELECT * FROM old_db.regression_residuals;

DETACH old_db;