from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class GARCHVisualizer:
    """Visualization utilities for GARCH analysis"""
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        style : str
            Matplotlib style to use. Default is 'seaborn'.
            Available styles can be listed with `plt.style.available`
        """
        # Set style safely with fallback options
        try:
            plt.style.use(style)
        except OSError:
            try:
                # Try seaborn's default style
                plt.style.use('seaborn')
            except OSError:
                # Fallback to matplotlib's default
                plt.style.use('default')
                logger.warning(f"Style '{style}' not found, using default style")
        
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
    def plot_garch_forecasts(self,
                          dates: np.ndarray,
                          forecasts: np.ndarray,
                          model_types: List[str],
                          title: Optional[str] = None,
                          save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot GARCH forecasts from multiple models
        
        Parameters:
        -----------
        dates : array-like
            Dates for x-axis
        forecasts : array-like
            Forecasts from different models (shape: n_models x n_dates)
        model_types : list
            Names of models corresponding to forecasts
        title : str, optional
            Plot title
        save_path : Path, optional
            Path to save figure
        """
        if len(dates) == 0 or len(forecasts) == 0 or len(model_types) == 0:
            raise ValueError("Empty input data")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (forecast, model) in enumerate(zip(forecasts, model_types)):
            ax.plot(dates, forecast, label=model, color=self.colors[i])
            
        ax.set_xlabel('Date')
        ax.set_ylabel('Annualized Volatility (%)')
        if title:
            ax.set_title(title)
        ax.legend()
        
        if save_path:
            fig.savefig(save_path)
            
        return fig
        
    def plot_ensemble_stats(self,
                         stats_df: pd.DataFrame,
                         stats: List[str] = ['GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS'],
                         title: Optional[str] = None,
                         save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot ensemble statistics over time
        
        Parameters:
        -----------
        stats_df : DataFrame
            DataFrame containing ensemble statistics
        stats : list
            Statistics to plot
        title : str, optional
            Plot title
        save_path : Path, optional
            Path to save figure
        """
        n_stats = len(stats)
        fig, axes = plt.subplots(n_stats, 1, figsize=(12, 4*n_stats))
        
        if n_stats == 1:
            axes = [axes]
            
        for ax, stat in zip(axes, stats):
            ax.plot(stats_df.index, stats_df[stat], color=self.colors[0])
            ax.set_xlabel('Date')
            ax.set_ylabel(stat)
            ax.grid(True)
            
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            
        return fig
        
    def plot_error_correction(self,
                           error_df: pd.DataFrame,
                           tenor: str,
                           title: Optional[str] = None,
                           save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot error correction analysis
        
        Parameters:
        -----------
        error_df : DataFrame
            DataFrame containing error terms
        tenor : str
            Tenor to analyze
        title : str, optional
            Plot title
        save_path : Path, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot raw errors
        ax1.plot(error_df.index, error_df['error'], color=self.colors[0])
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Error (Raw)')
        ax1.grid(True)
        
        # Plot standardized errors
        ax2.plot(error_df.index, error_df['std_error'], color=self.colors[1])
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Error (Standardized)')
        ax2.grid(True)
        
        if title:
            fig.suptitle(f"{title} - {tenor}")
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            
        return fig
        
    def plot_term_structure(self,
                         term_df: pd.DataFrame,
                         dates: List[datetime],
                         title: Optional[str] = None,
                         save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot volatility term structure
        
        Parameters:
        -----------
        term_df : DataFrame
            DataFrame with tenors as columns
        dates : list
            Dates to plot
        title : str, optional
            Plot title
        save_path : Path, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, date in enumerate(dates):
            data = term_df.loc[date]
            ax.plot(range(len(data)), data, 
                   marker='o',
                   label=date.strftime('%Y-%m-%d'),
                   color=self.colors[i])
            
        ax.set_xticks(range(len(term_df.columns)))
        ax.set_xticklabels(term_df.columns)
        ax.set_xlabel('Tenor')
        ax.set_ylabel('Implied Volatility (%)')
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        if save_path:
            fig.savefig(save_path)
            
        return fig
        
    def plot_diagnostic_grid(self,
                          df: pd.DataFrame,
                          tenor: str,
                          title: Optional[str] = None,
                          save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create diagnostic grid of plots
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing all metrics
        tenor : str
            Tenor to analyze
        title : str, optional
            Plot title
        save_path : Path, optional
            Path to save figure
        """
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # Implied vs GEV
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df.index, df['iv'], label='IV', color=self.colors[0])
        ax1.plot(df.index, df['gev'], label='GEV', color=self.colors[1])
        ax1.set_title('IV vs GEV')
        ax1.legend()
        
        # Error distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(df['std_error'], ax=ax2, bins=30)
        ax2.set_title('Standardized Error Distribution')
        
        # EVOEV and DEV
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df.index, df['evoev'], label='EVOEV', color=self.colors[2])
        ax3.plot(df.index, df['dev'], label='DEV', color=self.colors[3])
        ax3.set_title('Volatility Uncertainty')
        ax3.legend()
        
        # KEV and SEVTS
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df.index, df['kev'], label='KEV', color=self.colors[4])
        ax4.plot(df.index, df['sevts'], label='SEVTS', color=self.colors[5])
        ax4.set_title('Higher Moments')
        ax4.legend()
        
        # Error autocorrelation
        ax5 = fig.add_subplot(gs[2, :])
        pd.plotting.autocorrelation_plot(df['std_error'], ax=ax5)
        ax5.set_title('Error Autocorrelation')
        
        if title:
            fig.suptitle(f"{title} - {tenor}", y=1.02)
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            
        return fig
        
    def close_all(self):
        """Close all open figures"""
        plt.close('all')
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
        
    def plot_results(self, results: Dict, output_path: Path, show_plots: bool = False):
        """Plot analysis results and save to output directory"""
        try:
            # Create output directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Extract data
            dates = results['dates']
            window_results = results['results']
            
            if not window_results:  # Check if results are empty
                raise ValueError("No results to plot")
            
            # Extract metrics from results
            means = [r['window_mean'] for r in window_results]
            stds = [r['window_std'] for r in window_results]
            iv_means = [r['implied_vols_mean'] for r in window_results]
            
            # Verify we have data to plot
            if not means or not stds or not iv_means:
                raise ValueError("Empty metrics in results")
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot window statistics
            plt.subplot(2, 1, 1)
            plt.plot(dates[-len(means):], means, label='Rolling Mean')
            plt.plot(dates[-len(stds):], stds, label='Rolling Std')
            plt.title(f"Rolling Statistics - {results['index_id']}")
            plt.legend()
            plt.grid(True)
            
            # Plot implied volatilities
            plt.subplot(2, 1, 2)
            plt.plot(dates[-len(iv_means):], iv_means, label='Implied Vol Mean')
            plt.title(f"Implied Volatilities - {results['index_id']}")
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_path / f"{results['index_id']}_analysis.png")
            
            if show_plots:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logging.getLogger('utils.visualization').error(f"Error plotting results: {str(e)}")
            raise

# Example usage
if __name__ == '__main__':
    # Create sample data
    dates = pd.date_range('2015-01-01', periods=100)
    forecasts = np.random.normal(15, 2, (3, 100))
    model_types = ['GARCH', 'EGARCH', 'GJR-GARCH']
    
    # Initialize visualizer
    viz = GARCHVisualizer()
    
    # Create example plot
    fig = viz.plot_garch_forecasts(
        dates=dates,
        forecasts=forecasts,
        model_types=model_types,
        title='Sample GARCH Forecasts'
    )
    plt.show()