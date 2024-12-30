import pytest
import numpy as np

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    n_obs = 3024  # Update from 1260
    returns = np.random.normal(0, 0.01, n_obs)
    return returns
