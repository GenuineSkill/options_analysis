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
            
    def plot_error_correction(self,
                             index_id: str,
                             tenor: str,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> go.Figure:
        """
        Plot error correction model analysis
        
        Creates a multi-panel plot showing:
        1. IV Changes vs Fitted Values
        2. Error Correction Term and Signals
        3. Coefficient Evolution
        4. Signal Success Rates
        """
        try:
            # Get error correction results
            query = """
                SELECT 
                    date,
                    delta_iv,
                    fitted_change,
                    error_correction,
                    residual
                FROM error_correction_results
                WHERE index_id = ?
                    AND tenor = ?
                ORDER BY date
            """
            params = [index_id, tenor]
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            data = self.reg.conn.execute(query, params).df()
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'IV Changes vs Fitted',
                    'Error Correction Term',
                    'Residual Distribution',
                    'Signal Success Rate',
                    'Coefficient Significance',
                    'Model Performance'
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.1,
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "histogram"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ]
            )
            
            # 1. IV Changes vs Fitted
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['delta_iv'],
                    name='Actual Changes',
                    mode='markers',
                    marker=dict(size=4, color='blue', opacity=0.6)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['fitted_change'],
                    name='Fitted Changes',
                    mode='lines',
                    line=dict(color='red', width=1)
                ),
                row=1, col=1
            )
            
            # 2. Error Correction Term
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['error_correction'],
                    name='Error Correction',
                    mode='lines+markers',
                    marker=dict(size=4),
                    line=dict(color='green', width=1)
                ),
                row=1, col=2
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
                    row=1, col=2
                )
            
            # 3. Residual Distribution
            fig.add_trace(
                go.Histogram(
                    x=data['residual'],
                    name='Residuals',
                    nbinsx=30,
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            # 4. Signal Success Rate
            signals = self.reg.get_trading_signals(index_id, tenor)
            success_rates = []
            for signal in [-1, 0, 1]:
                signal_data = signals[signals['signal'] == signal]
                if len(signal_data) > 0:
                    if signal == 1:
                        success = (signal_data['delta_iv'] > 0).mean()
                        label = 'Buy'
                    elif signal == -1:
                        success = (signal_data['delta_iv'] < 0).mean()
                        label = 'Sell'
                    else:
                        success = 0.5
                        label = 'Hold'
                    
                    success_rates.append({
                        'signal': label,
                        'success_rate': success,
                        'count': len(signal_data)
                    })
            
            success_df = pd.DataFrame(success_rates)
            fig.add_trace(
                go.Bar(
                    x=success_df['signal'],
                    y=success_df['success_rate'],
                    name='Success Rate',
                    text=[f"{v:.1%}" for v in success_df['success_rate']],
                    textposition='auto'
                ),
                row=2, col=2
            )
            
            # 5. Coefficient Significance
            coef_data = self.reg.conn.execute("""
                SELECT 
                    variable,
                    coefficient,
                    t_stat,
                    ABS(t_stat) as abs_t_stat
                FROM error_correction_coefficients
                WHERE index_id = ?
                    AND tenor = ?
                    AND date = (SELECT MAX(date) 
                               FROM error_correction_coefficients
                               WHERE index_id = ?
                                    AND tenor = ?)
                ORDER BY abs_t_stat DESC
            """, [index_id, tenor, index_id, tenor]).df()
            
            fig.add_trace(
                go.Bar(
                    x=coef_data['variable'],
                    y=coef_data['t_stat'],
                    name='t-statistics',
                    text=[f"{v:.2f}" for v in coef_data['t_stat']],
                    textposition='auto'
                ),
                row=3, col=1
            )
            
            # Add significance lines
            for t in [-1.96, 1.96]:
                fig.add_shape(
                    type='line',
                    x0=-0.5,
                    x1=len(coef_data)-0.5,
                    y0=t,
                    y1=t,
                    line=dict(color='red', dash='dash'),
                    row=3, col=1
                )
            
            # 6. Model Performance Over Time
            rolling_r2 = signals.rolling(window=63)['delta_iv'].corr(
                signals['fitted_change']
            )**2
            
            fig.add_trace(
                go.Scatter(
                    x=signals['date'],
                    y=rolling_r2,
                    name='Rolling R²',
                    line=dict(color='purple')
                ),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=1200,
                width=1600,
                title_text=f'Error Correction Analysis - {index_id} {tenor}',
                showlegend=True,
                template='plotly_white'
            )
            
            # Add annotations for model statistics
            stats_text = f"""
            Observations: {len(data)}
            R-squared: {data['delta_iv'].corr(data['fitted_change'])**2:.3f}
            Mean EC: {data['error_correction'].mean():.3f}
            EC Std: {data['error_correction'].std():.3f}
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
            logger.error(f"Error creating error correction visualization: {str(e)}")
            raise