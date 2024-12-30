import duckdb
from pathlib import Path

# Window parameters
TRAIN_YEARS = 12  # Update from 5 to 12
TEST_YEARS = 2
TRAIN_DAYS = int(TRAIN_YEARS * 252)  # ~3024 days

def show_consecutive_windows():
    """Show details for consecutive window_ids"""
    db_path = Path("results/historical/historical_results.db")
    conn = duckdb.connect(str(db_path), read_only=True)
    
    # Query for 5 consecutive windows
    query = """
    SELECT 
        window_id,
        index_id,
        start_date::DATE as start_date,
        end_date::DATE as end_date
    FROM forecast_windows
    ORDER BY window_id
    LIMIT 10
    """
    
    results = conn.execute(query).df()
    print("\nConsecutive Window Examples:")
    print(results)
    
    # Show associated GARCH results for first window
    first_window = results.window_id.iloc[0]
    query = f"""
    SELECT 
        window_id,
        model_type,
        distribution
    FROM garch_results
    WHERE window_id = {first_window}
    """
    
    print(f"\nGARCH models for window_id {first_window}:")
    print(conn.execute(query).df())
    
    # Show ensemble stats for first window
    query = f"""
    SELECT 
        window_id,
        horizon,
        gev,
        evoev
    FROM ensemble_stats
    WHERE window_id = {first_window}
    ORDER BY horizon_days
    """
    
    print(f"\nEnsemble stats for window_id {first_window}:")
    print(conn.execute(query).df())
    
    conn.close()

if __name__ == "__main__":
    show_consecutive_windows()