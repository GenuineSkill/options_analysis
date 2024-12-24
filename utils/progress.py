from typing import Optional
import logging
from tqdm import tqdm
import time

class ProgressMonitor:
    def __init__(self, total: int, desc: str = "Processing", 
                 logger: Optional[logging.Logger] = None):
        """Initialize progress monitor with total steps and description"""
        self.logger = logger or logging.getLogger('progress')
        self.pbar = tqdm(total=total, desc=desc)
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self.description = desc
        
    def update(self, n: int = 1, status: str = ""):
        """Update progress by n steps with optional status message"""
        self.current += n
        self.pbar.update(n)
        
        # Log progress message if provided
        if status:
            self.logger.info(f"{self.description}: {status}")
        
        # Log progress every 100 steps
        if self.logger and self.current % 100 == 0:
            elapsed = time.time() - self.start_time
            progress = self.current / self.total
            eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            
            self.logger.info(
                f"Progress: {self.current}/{self.total} "
                f"({progress*100:.1f}%) - "
                f"Elapsed: {elapsed/3600:.1f}h - "
                f"ETA: {eta/3600:.1f}h"
            )
            
    def close(self):
        """Close progress bar and log final statistics"""
        self.pbar.close()
        total_time = time.time() - self.start_time
        self.logger.info(
            f"Completed {self.description} in {total_time/3600:.1f} hours"
        )