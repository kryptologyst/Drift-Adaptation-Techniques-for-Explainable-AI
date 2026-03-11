"""Utility functions and classes for drift adaptation."""

import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        Available PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class Config:
    """Configuration management for drift adaptation experiments."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
        """
        self._config = config_dict or {}
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key.
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key.
            value: Configuration value.
        """
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            config_dict: Dictionary of new configuration values.
        """
        self._config.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary.
        """
        return self._config.copy()


def validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate input arrays.
    
    Args:
        X: Feature matrix.
        y: Optional target vector.
        
    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    
    if y is not None:
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy array")
        
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")


def compute_statistical_distance(
    X1: np.ndarray, 
    X2: np.ndarray, 
    method: str = "euclidean"
) -> float:
    """Compute statistical distance between two datasets.
    
    Args:
        X1: First dataset.
        X2: Second dataset.
        method: Distance method ('euclidean', 'manhattan', 'cosine').
        
    Returns:
        Statistical distance value.
    """
    if method == "euclidean":
        return np.linalg.norm(np.mean(X1, axis=0) - np.mean(X2, axis=0))
    elif method == "manhattan":
        return np.sum(np.abs(np.mean(X1, axis=0) - np.mean(X2, axis=0)))
    elif method == "cosine":
        mean1 = np.mean(X1, axis=0)
        mean2 = np.mean(X2, axis=0)
        return 1 - np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2))
    else:
        raise ValueError(f"Unknown distance method: {method}")


def create_sliding_window(
    data: np.ndarray, 
    window_size: int, 
    step_size: int = 1
) -> np.ndarray:
    """Create sliding windows from time series data.
    
    Args:
        data: Input time series data.
        window_size: Size of each window.
        step_size: Step size between windows.
        
    Returns:
        Array of sliding windows.
    """
    if len(data) < window_size:
        raise ValueError("Data length must be >= window_size")
    
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    
    return np.array(windows)
