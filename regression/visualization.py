"""Visualization module for regression analysis"""

import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class RegressionVisualizer:
    def __init__(self, level_regression):
        """Initialize with LevelRegression instance"""
        self.reg = level_regression
        
    def plot_residuals(self, 
                      index_id: str, 
                      tenor: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> go.Figure:
        """
        Plot regression residuals with trading signals
        
        Parameters:
        - index_id: Market identifier (e.g., 'SPX')
        - tenor: Option expiry tenor (e.g., '1M')
        - start_date: Optional start date filter (YYYY-MM-DD)
        - end_date: Optional end date filter (YYYY-MM-DD)
        """
        try:
            # Get residuals and signals
            residuals = self.reg.get_residuals(index_id, tenor, start_date, end_date)
            signals = self.reg.get_trading_signals(index_id, tenor)
            
            # Merge data
            data = residuals.merge(
                signals[['date', 'signal']],
                on='date',
                how='left'
            )
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    f'IV vs Fitted Values - {index_id} {tenor}',
                    'Standardized Residuals with Trading Signals',
                    'Residual Distribution'
                ),
                vertical_spacing=0.1,
                row_heights=[0.4, 0.4, 0.2]
            )
            
            # Plot 1: IV vs Fitted
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['iv_actual'],
                    name='Actual IV',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['iv_fitted'],
                    name='Fitted IV',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
            
            # Plot 2: Standardized Residuals
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['standardized_residual'],
                    name='Std Residual',
                    line=dict(color='gray')
                ),
                row=2, col=1
            )
            
            # Add trading signals
            buy_signals = data[data['signal'] == 1]
            sell_signals = data[data['signal'] == -1]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['date'],
                    y=buy_signals['standardized_residual'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['date'],
                    y=sell_signals['standardized_residual'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=2, col=1
            )
            
            # Add threshold lines
            for z in [-2, 2]:
                fig.add_shape(
                    type='line',
                    x0=data['date'].min(),
                    x1=data['date'].max(),
                    y0=z,
                    y1=z,
                    line=dict(color='gray', dash='dash'),
                    row=2, col=1
                )
            
            # Plot 3: Residual Distribution
            fig.add_trace(
                go.Histogram(
                    x=data['standardized_residual'],
                    name='Distribution',
                    nbinsx=30,
                    marker_color='lightblue'
                ),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=1000,
                title_text=f'Regression Analysis - {index_id} {tenor}',
                showlegend=True,
                template='plotly_white'
            )
            
            # Add statistics annotations
            stats_text = f"""
            Mean Residual: {data['residual'].mean():.4f}
            Std Dev: {data['residual'].std():.4f}
            Buy Signals: {len(buy_signals)}
            Sell Signals: {len(sell_signals)}
            """
            
            fig.add_annotation(
                text=stats_text,
                xref="paper", yref="paper",
                x=1.0, y=1.0,
                showarrow=False,
                align='left'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            raise
            
    def save_plots(self, 
                   index_id: str, 
                   output_dir: str = 'results/plots'):
        """Save plots for all tenors"""
        import os
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for tenor in ['1M', '2M', '3M', '6M', '12M']:
                fig = self.plot_residuals(index_id, tenor)
                output_file = os.path.join(output_dir, f'{index_id}_{tenor}_analysis.html')
                fig.write_html(output_file)
                logger.info(f"Saved plot to {output_file}")
                
        except Exception as e:
            logger.error(f"Error saving plots: {str(e)}")
            raise