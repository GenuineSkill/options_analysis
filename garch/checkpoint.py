from pathlib import Path
import pickle
import logging
from typing import Optional, Dict
import pandas as pd

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('checkpoint_manager')
        
    def save_checkpoint(self, date: pd.Timestamp, data: Dict):
        """Save estimation checkpoint"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{date.strftime('%Y%m%d')}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
            
    def load_checkpoint(self, date: pd.Timestamp) -> Optional[Dict]:
        """Load checkpoint if it exists"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{date.strftime('%Y%m%d')}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None