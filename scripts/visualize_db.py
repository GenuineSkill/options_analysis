"""
Interactive visualization of GARCH ensemble results using Streamlit
"""

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from datetime import datetime

def load_data(db_path: str):
    """Load data from DuckDB"""
    conn = duckdb.connect(db_path)
    
    # Define available statistics and their descriptions
    ensemble_stats = {
        'gev': 'Global Ensemble Volatility',
        'evoev': 'Ensemble Volatility of Volatility',
        'dev': 'Dispersion of Ensemble Volatility',
        'kev': 'Kurtosis of Ensemble Volatility',
        'sevts': 'Skewness of Ensemble Volatility Term Structure'
    }
    
    # Get date range
    date_range = conn.execute("""
        SELECT 
            MIN(end_date)::DATE as earliest,
            MAX(end_date)::DATE as latest
        FROM forecast_windows
    """).fetchone()
    
    conn.close()
    return ensemble_stats, date_range

def create_visualization():
    st.title("GARCH Ensemble Analysis Dashboard")
    
    # Set path to database
    db_path = Path("results/historical/historical_results.db")
    
    if not db_path.exists():
        st.error(f"Database not found at: {db_path}")
        return
    
    # Connect to database
    conn = duckdb.connect(str(db_path))
    
    # Sidebar controls
    st.sidebar.header("Analysis Options")
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Ensemble Statistics Time Series",
         "Model Performance Comparison",
         "Term Structure Analysis",
         "Data Quality Check"]
    )
    
    if analysis_type == "Ensemble Statistics Time Series":
        # Load available statistics and date range
        ensemble_stats, date_range = load_data(str(db_path))
        
        # Statistic selection
        stat_name = st.sidebar.selectbox(
            "Select Statistic",
            options=list(ensemble_stats.keys()),
            format_func=lambda x: f"{x.upper()} - {ensemble_stats[x]}"
        )
        
        # Horizon selection
        horizon = st.sidebar.selectbox(
            "Select Horizon",
            options=['1M', '2M', '3M', '6M', '12M']
        )
        
        # Date range selection
        min_date = pd.to_datetime(date_range[0])
        max_date = pd.to_datetime(date_range[1])
        
        start_date = st.sidebar.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        # Query data
        query = f"""
            SELECT 
                fw.end_date::DATE as date,
                es.{stat_name} as value,
                fw.index_id
            FROM forecast_windows fw
            JOIN ensemble_stats es ON fw.window_id = es.window_id
            WHERE fw.end_date::DATE BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                AND es.horizon = '{horizon}'
            ORDER BY fw.end_date
        """
        
        df = conn.execute(query).df()
        
        # Create visualization
        fig = px.line(
            df,
            x='date',
            y='value',
            color='index_id',
            title=f"{stat_name.upper()} - {ensemble_stats[stat_name]} ({horizon} Horizon)"
        )
        
        st.plotly_chart(fig)
        
    elif analysis_type == "Model Performance Comparison":
        # Query model performance
        query = """
            SELECT 
                CASE 
                    WHEN model_type IN ('GARCH', 'garch') AND parameters::json->>'o' = '0' THEN 'GARCH'
                    WHEN model_type IN ('EGARCH', 'egarch') THEN 'EGARCH'
                    WHEN model_type IN ('GARCH', 'garch') AND parameters::json->>'o' = '1' THEN 'GJR-GARCH'
                    ELSE upper(model_type)
                END as model_type,
                distribution,
                COUNT(*) as n_estimates,
                AVG(forecast_accuracy) as avg_accuracy
            FROM garch_results
            GROUP BY 
                CASE 
                    WHEN model_type IN ('GARCH', 'garch') AND parameters::json->>'o' = '0' THEN 'GARCH'
                    WHEN model_type IN ('EGARCH', 'egarch') THEN 'EGARCH'
                    WHEN model_type IN ('GARCH', 'garch') AND parameters::json->>'o' = '1' THEN 'GJR-GARCH'
                    ELSE upper(model_type)
                END,
                distribution
        """
        
        df = conn.execute(query).df()
        
        fig = px.bar(
            df,
            x='model_type',
            y='n_estimates',
            color='distribution',
            facet_col='index_id',
            title="GARCH Model Usage by Type and Distribution"
        )
        
        st.plotly_chart(fig)
        
    elif analysis_type == "Term Structure Analysis":
        # Query term structure data
        query = """
            SELECT 
                fw.end_date::DATE as date,
                es.horizon,
                es.gev as volatility,
                fw.index_id
            FROM forecast_windows fw
            JOIN ensemble_stats es ON fw.window_id = es.window_id
            WHERE fw.end_date IN (
                SELECT DISTINCT end_date 
                FROM forecast_windows 
                ORDER BY end_date DESC 
                LIMIT 5
            )
            ORDER BY fw.end_date, es.horizon_days
        """
        
        df = conn.execute(query).df()
        
        fig = px.line(
            df,
            x='horizon',
            y='volatility',
            color='date',
            facet_col='index_id',
            title="Recent Volatility Term Structures"
        )
        
        st.plotly_chart(fig)
        
    elif analysis_type == "Data Quality Check":
        # Query data quality metrics
        metrics = conn.execute("""
            SELECT 
                'Forecast Windows' as table_name,
                COUNT(*) as record_count,
                COUNT(DISTINCT index_id) as unique_indices,
                MIN(end_date)::DATE as earliest_date,
                MAX(end_date)::DATE as latest_date
            FROM forecast_windows
            
            UNION ALL
            
            SELECT 
                'GARCH Results' as table_name,
                COUNT(*) as record_count,
                COUNT(DISTINCT model_type || distribution) as unique_models,
                NULL as earliest_date,
                NULL as latest_date
            FROM garch_results
            
            UNION ALL
            
            SELECT 
                'Ensemble Stats' as table_name,
                COUNT(*) as record_count,
                COUNT(DISTINCT horizon) as unique_horizons,
                NULL as earliest_date,
                NULL as latest_date
            FROM ensemble_stats
        """).df()
        
        st.write("Database Overview")
        st.dataframe(metrics)
        
        # Show sample of recent data
        st.write("Sample of Recent Data")
        recent = conn.execute("""
            SELECT 
                fw.end_date::DATE as date,
                fw.index_id,
                es.horizon,
                es.gev,
                es.evoev,
                es.dev,
                es.kev,
                es.sevts
            FROM forecast_windows fw
            JOIN ensemble_stats es ON fw.window_id = es.window_id
            WHERE fw.end_date >= (SELECT MAX(end_date) - INTERVAL '7 days' FROM forecast_windows)
            ORDER BY fw.end_date DESC, fw.index_id, es.horizon
            LIMIT 10
        """).df()
        
        st.dataframe(recent)
    
    conn.close()

if __name__ == "__main__":
    create_visualization()