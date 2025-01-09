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
    
    # Get date range and available horizons
    date_range = conn.execute("""
        SELECT 
            MIN(end_date)::DATE as earliest,
            MAX(end_date)::DATE as latest
        FROM forecast_windows
    """).fetchone()
    
    # Get available horizons
    horizons = conn.execute("""
        SELECT DISTINCT horizon, horizon_days
        FROM ensemble_stats
        ORDER BY horizon_days
    """).df()
    
    conn.close()
    return ensemble_stats, date_range, horizons

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
        ensemble_stats, date_range, horizons = load_data(str(db_path))
        
        # Statistic selection
        stat_name = st.sidebar.selectbox(
            "Select Statistic",
            options=list(ensemble_stats.keys()),
            format_func=lambda x: f"{x.upper()} - {ensemble_stats[x]}"
        )
        
        # Horizon selection
        horizon = st.sidebar.selectbox(
            "Select Horizon",
            options=horizons['horizon'].tolist(),
            format_func=lambda x: f"{x} ({horizons[horizons['horizon'] == x]['horizon_days'].iloc[0]} days)"
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
            WITH parsed_models AS (
                SELECT 
                    model_type,
                    distribution,
                    TRY_CAST(parameters::json->>'o' AS INTEGER) as o_param
                FROM garch_results
            )
            SELECT 
                CASE 
                    WHEN model_type IN ('GARCH', 'garch') AND o_param = 0 THEN 'GARCH'
                    WHEN model_type IN ('EGARCH', 'egarch') THEN 'EGARCH'
                    WHEN model_type IN ('GARCH', 'garch') AND o_param = 1 THEN 'GJR-GARCH'
                    ELSE upper(model_type)
                END as model_type,
                distribution,
                COUNT(*) as n_estimates
            FROM parsed_models
            GROUP BY 
                CASE 
                    WHEN model_type IN ('GARCH', 'garch') AND o_param = 0 THEN 'GARCH'
                    WHEN model_type IN ('EGARCH', 'egarch') THEN 'EGARCH'
                    WHEN model_type IN ('GARCH', 'garch') AND o_param = 1 THEN 'GJR-GARCH'
                    ELSE upper(model_type)
                END,
                distribution
            ORDER BY model_type, distribution
        """
        
        df = conn.execute(query).df()
        
        fig = px.bar(
            df,
            x='model_type',
            y='n_estimates',
            color='distribution',
            title="GARCH Model Usage by Type and Distribution",
            labels={
                'model_type': 'Model Type',
                'n_estimates': 'Number of Estimates',
                'distribution': 'Error Distribution'
            }
        )
        
        # Improve layout
        fig.update_layout(
            xaxis_title="Model Type",
            yaxis_title="Number of Estimates",
            legend_title="Error Distribution",
            barmode='group'
        )
        
        st.plotly_chart(fig)
        
    elif analysis_type == "Term Structure Analysis":
        # Date selection for term structure
        _, date_range, _ = load_data(str(db_path))
        max_date = pd.to_datetime(date_range[1])
        
        selected_date = st.sidebar.date_input(
            "Select Date",
            value=max_date,
            min_value=pd.to_datetime(date_range[0]),
            max_value=max_date
        )
        
        # Query term structure data
        query = f"""
            SELECT 
                fw.end_date::DATE as date,
                es.horizon,
                es.horizon_days,
                es.gev as volatility,
                fw.index_id
            FROM forecast_windows fw
            JOIN ensemble_stats es ON fw.window_id = es.window_id
            WHERE fw.end_date::DATE = DATE '{selected_date.strftime('%Y-%m-%d')}'
            ORDER BY fw.end_date, es.horizon_days
        """
        
        df = conn.execute(query).df()
        
        fig = px.line(
            df,
            x='horizon_days',
            y='volatility',
            color='index_id',
            title=f"Volatility Term Structure ({selected_date})",
            labels={'horizon_days': 'Days', 'volatility': 'Volatility (%)'}
        )
        
        # Add hover text with horizon labels
        fig.update_traces(
            hovertemplate="Horizon: %{customdata}<br>Days: %{x}<br>Volatility: %{y:.1f}%",
            customdata=df['horizon']
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
                MIN(fw.end_date)::DATE as earliest_date,
                MAX(fw.end_date)::DATE as latest_date
            FROM garch_results gr
            JOIN forecast_windows fw ON gr.window_id = fw.window_id
            
            UNION ALL
            
            SELECT 
                'Ensemble Stats' as table_name,
                COUNT(*) as record_count,
                COUNT(DISTINCT horizon) as unique_horizons,
                MIN(fw.end_date)::DATE as earliest_date,
                MAX(fw.end_date)::DATE as latest_date
            FROM ensemble_stats es
            JOIN forecast_windows fw ON es.window_id = fw.window_id
        """).df()
        
        st.write("Database Overview")
        st.dataframe(metrics)
        
        # Show sample of recent data
        st.write("Sample of Recent Data")
        recent = conn.execute("""
            WITH recent_data AS (
                SELECT 
                    fw.end_date::DATE as date,
                    fw.index_id,
                    es.horizon,
                    es.gev,
                    es.evoev,
                    COALESCE(es.dev, NULL) as dev,
                    COALESCE(es.kev, NULL) as kev,
                    es.sevts
                FROM forecast_windows fw
                JOIN ensemble_stats es ON fw.window_id = es.window_id
                WHERE fw.end_date >= (SELECT MAX(end_date) - INTERVAL '7 days' FROM forecast_windows)
            )
            SELECT *
            FROM recent_data
            ORDER BY date DESC, index_id, horizon
            LIMIT 10
        """).df()
        
        st.dataframe(recent)
    
    conn.close()

if __name__ == "__main__":
    create_visualization()