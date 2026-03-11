"""Drift detection algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_kernels

from ..utils import validate_inputs, set_seed


class DriftDetector(ABC):
    """Abstract base class for drift detectors."""
    
    def __init__(self, threshold: float = 0.05, random_state: int = 42):
        """Initialize drift detector.
        
        Args:
            threshold: Detection threshold.
            random_state: Random seed.
        """
        self.threshold = threshold
        self.random_state = random_state
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> "DriftDetector":
        """Fit the drift detector on reference data.
        
        Args:
            X: Reference data.
            
        Returns:
            Self.
        """
        pass
    
    @abstractmethod
    def detect_drift(self, X: np.ndarray) -> Tuple[bool, float]:
        """Detect drift in new data.
        
        Args:
            X: New data to test for drift.
            
        Returns:
            Tuple of (drift_detected, drift_score).
        """
        pass
    
    def fit_detect(self, X_ref: np.ndarray, X_test: np.ndarray) -> Tuple[bool, float]:
        """Fit detector and detect drift in one call.
        
        Args:
            X_ref: Reference data.
            X_test: Test data.
            
        Returns:
            Tuple of (drift_detected, drift_score).
        """
        self.fit(X_ref)
        return self.detect_drift(X_test)


class PSIDetector(DriftDetector):
    """Population Stability Index (PSI) drift detector."""
    
    def __init__(self, threshold: float = 0.1, n_bins: int = 10, random_state: int = 42):
        """Initialize PSI detector.
        
        Args:
            threshold: PSI threshold for drift detection.
            n_bins: Number of bins for histogram.
            random_state: Random seed.
        """
        super().__init__(threshold, random_state)
        self.n_bins = n_bins
        self.bins = None
        
    def fit(self, X: np.ndarray) -> "PSIDetector":
        """Fit PSI detector on reference data.
        
        Args:
            X: Reference data.
            
        Returns:
            Self.
        """
        validate_inputs(X)
        set_seed(self.random_state)
        
        # Create bins based on reference data
        self.bins = []
        for i in range(X.shape[1]):
            feature_min = X[:, i].min()
            feature_max = X[:, i].max()
            bins_i = np.linspace(feature_min, feature_max, self.n_bins + 1)
            self.bins.append(bins_i)
        
        self.is_fitted = True
        return self
    
    def detect_drift(self, X: np.ndarray) -> Tuple[bool, float]:
        """Detect drift using PSI.
        
        Args:
            X: Test data.
            
        Returns:
            Tuple of (drift_detected, psi_score).
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting drift")
        
        validate_inputs(X)
        
        psi_scores = []
        for i in range(X.shape[1]):
            psi_score = self._compute_psi(X[:, i], i)
            psi_scores.append(psi_score)
        
        max_psi = max(psi_scores)
        drift_detected = max_psi > self.threshold
        
        return drift_detected, max_psi
    
    def _compute_psi(self, feature_values: np.ndarray, feature_idx: int) -> float:
        """Compute PSI for a single feature.
        
        Args:
            feature_values: Feature values.
            feature_idx: Index of the feature.
            
        Returns:
            PSI score.
        """
        # This is a simplified PSI computation
        # In practice, you'd need reference data to compute actual PSI
        bins = self.bins[feature_idx]
        
        # Compute histogram
        hist, _ = np.histogram(feature_values, bins=bins)
        
        # Normalize to get probabilities
        hist = hist.astype(float)
        hist[hist == 0] = 1e-6  # Avoid division by zero
        hist = hist / hist.sum()
        
        # Compute PSI (simplified version)
        expected = np.ones_like(hist) / len(hist)
        psi = np.sum((hist - expected) * np.log(hist / expected))
        
        return psi


class KSDetector(DriftDetector):
    """Kolmogorov-Smirnov test drift detector."""
    
    def __init__(self, threshold: float = 0.05, random_state: int = 42):
        """Initialize KS detector.
        
        Args:
            threshold: P-value threshold for drift detection.
            random_state: Random seed.
        """
        super().__init__(threshold, random_state)
        self.reference_data = None
        
    def fit(self, X: np.ndarray) -> "KSDetector":
        """Fit KS detector on reference data.
        
        Args:
            X: Reference data.
            
        Returns:
            Self.
        """
        validate_inputs(X)
        set_seed(self.random_state)
        
        self.reference_data = X.copy()
        self.is_fitted = True
        return self
    
    def detect_drift(self, X: np.ndarray) -> Tuple[bool, float]:
        """Detect drift using KS test.
        
        Args:
            X: Test data.
            
        Returns:
            Tuple of (drift_detected, min_p_value).
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting drift")
        
        validate_inputs(X)
        
        p_values = []
        for i in range(X.shape[1]):
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[:, i], 
                X[:, i]
            )
            p_values.append(p_value)
        
        min_p_value = min(p_values)
        drift_detected = min_p_value < self.threshold
        
        return drift_detected, min_p_value


class MMDDetector(DriftDetector):
    """Maximum Mean Discrepancy (MMD) drift detector."""
    
    def __init__(
        self, 
        threshold: float = 0.05, 
        kernel: str = "rbf", 
        gamma: Optional[float] = None,
        random_state: int = 42
    ):
        """Initialize MMD detector.
        
        Args:
            threshold: MMD threshold for drift detection.
            kernel: Kernel function ('rbf', 'linear', 'poly').
            gamma: Kernel parameter for RBF kernel.
            random_state: Random seed.
        """
        super().__init__(threshold, random_state)
        self.kernel = kernel
        self.gamma = gamma
        self.reference_data = None
        
    def fit(self, X: np.ndarray) -> "MMDDetector":
        """Fit MMD detector on reference data.
        
        Args:
            X: Reference data.
            
        Returns:
            Self.
        """
        validate_inputs(X)
        set_seed(self.random_state)
        
        self.reference_data = X.copy()
        self.is_fitted = True
        return self
    
    def detect_drift(self, X: np.ndarray) -> Tuple[bool, float]:
        """Detect drift using MMD.
        
        Args:
            X: Test data.
            
        Returns:
            Tuple of (drift_detected, mmd_score).
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting drift")
        
        validate_inputs(X)
        
        # Compute MMD
        mmd_score = self._compute_mmd(self.reference_data, X)
        drift_detected = mmd_score > self.threshold
        
        return drift_detected, mmd_score
    
    def _compute_mmd(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """Compute MMD between two datasets.
        
        Args:
            X1: First dataset.
            X2: Second dataset.
            
        Returns:
            MMD score.
        """
        # Compute kernel matrices
        K11 = pairwise_kernels(X1, X1, metric=self.kernel, gamma=self.gamma)
        K22 = pairwise_kernels(X2, X2, metric=self.kernel, gamma=self.gamma)
        K12 = pairwise_kernels(X1, X2, metric=self.kernel, gamma=self.gamma)
        
        # Compute MMD
        n1, n2 = len(X1), len(X2)
        mmd = (K11.sum() / (n1 * n1) + 
               K22.sum() / (n2 * n2) - 
               2 * K12.sum() / (n1 * n2))
        
        return mmd


class ADWINDetector(DriftDetector):
    """ADWIN (Adaptive Windowing) drift detector for online detection."""
    
    def __init__(
        self, 
        threshold: float = 0.05, 
        min_window_size: int = 10,
        random_state: int = 42
    ):
        """Initialize ADWIN detector.
        
        Args:
            threshold: Detection threshold.
            min_window_size: Minimum window size.
            random_state: Random seed.
        """
        super().__init__(threshold, random_state)
        self.min_window_size = min_window_size
        self.window = []
        self.drift_points = []
        
    def fit(self, X: np.ndarray) -> "ADWINDetector":
        """Initialize ADWIN with reference data.
        
        Args:
            X: Reference data.
            
        Returns:
            Self.
        """
        validate_inputs(X)
        set_seed(self.random_state)
        
        # Initialize window with reference data
        self.window = X.tolist()
        self.is_fitted = True
        return self
    
    def detect_drift(self, X: np.ndarray) -> Tuple[bool, float]:
        """Detect drift using ADWIN (simplified version).
        
        Args:
            X: New data points.
            
        Returns:
            Tuple of (drift_detected, drift_score).
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting drift")
        
        validate_inputs(X)
        
        drift_detected = False
        drift_score = 0.0
        
        for x in X:
            self.window.append(x)
            
            # Simplified ADWIN: check for significant change in mean
            if len(self.window) >= self.min_window_size * 2:
                mid_point = len(self.window) // 2
                window1 = np.array(self.window[:mid_point])
                window2 = np.array(self.window[mid_point:])
                
                # Compute mean difference
                mean_diff = abs(np.mean(window1) - np.mean(window2))
                
                if mean_diff > self.threshold:
                    drift_detected = True
                    drift_score = mean_diff
                    self.drift_points.append(len(self.window))
                    
                    # Reset window
                    self.window = self.window[mid_point:]
        
        return drift_detected, drift_score


def create_drift_detector(
    method: str, 
    **kwargs
) -> DriftDetector:
    """Factory function to create drift detectors.
    
    Args:
        method: Detection method ('psi', 'ks', 'mmd', 'adwin').
        **kwargs: Additional arguments for the detector.
        
    Returns:
        Drift detector instance.
        
    Raises:
        ValueError: If method is unknown.
    """
    detectors = {
        "psi": PSIDetector,
        "ks": KSDetector,
        "mmd": MMDDetector,
        "adwin": ADWINDetector,
    }
    
    if method not in detectors:
        raise ValueError(f"Unknown drift detection method: {method}")
    
    return detectors[method](**kwargs)
