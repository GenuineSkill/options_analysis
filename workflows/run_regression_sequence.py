"""Complete regression workflow with proper sequencing"""

import logging
from pathlib import Path
import sys
import duckdb
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from regression.expanding_window import ExpandingWindowRegression
from regression.error_correction import ErrorCorrectionModel
from regression.validation import verify_residual_storage, verify_temporal_consistency

def run_regression_sequence(
    conn,
    index_id: str,
    tenor: str,
    min_window_size: int = 126,
    step_size: int = 21
) -> dict:
    """
    Run complete regression sequence with proper temporal ordering
    
    Steps:
    1. Run expanding window level regressions
    2. Store residuals with window dates
    3. Run error correction model using only past residuals
    """
    logger = logging.getLogger('regression_sequence')
    
    try:
        # 1. Initialize models
        exp_reg = ExpandingWindowRegression(
            conn=conn,
            min_window_size=min_window_size
        )
        
        ec_model = ErrorCorrectionModel(
            conn=conn,
            dev_mode=(step_size == 21)
        )
        
        # 2. Run expanding window analysis first
        logger.info(f"Running expanding window analysis for {index_id} {tenor}")
        exp_results = exp_reg.run_expanding_windows(
            index_id=index_id,
            tenor=tenor,
            step_size=step_size
        )
        
        # 3. Verify residuals are stored properly
        verify_residual_storage(conn, index_id, tenor)
        
        # 4. Run error correction model
        logger.info(f"Running error correction model for {index_id} {tenor}")
        ec_results = ec_model.estimate(
            index_id=index_id,
            tenor=tenor
        )
        
        # 5. Verify temporal consistency
        verify_temporal_consistency(
            conn=conn,
            index_id=index_id,
            tenor=tenor
        )
        
        return {
            'expanding_window': exp_results,
            'error_correction': ec_results
        }
        
    except Exception as e:
        logger.error(f"Error in regression sequence: {str(e)}")
        raise 