from typing import Optional
import logging
from tqdm import tqdm

class ProgressMonitor:
    def __init__(self, total: int, desc: str = "Processing", logger: Optional[logging.Logger] = None):
        """Initialize progress monitor with total steps and description"""
        self.logger = logger or logging.getLogger('progress')
        self.pbar = tqdm(total=total, desc=desc)
        self.total = total
        self.current = 0
        
    def update(self, n: int = 1):
        """Update progress by n steps"""
        self.current += n
        self.pbar.update(n)
        if self.logger and self.current % 100 == 0:  # Log every 100 steps
            self.logger.info(f"Progress: {self.current}/{self.total} ({self.current/self.total*100:.1f}%)")
            
    def close(self):
        """Close progress bar"""
        self.pbar.close()