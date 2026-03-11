"""Data generation and preprocessing utilities."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ..utils import set_seed, validate_inputs


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    drift_type: str = "concept",
    drift_strength: float = 0.3,
    drift_point: Optional[int] = None,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data with concept drift.
    
    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features.
        n_classes: Number of classes.
        drift_type: Type of drift ('concept', 'covariate', 'label').
        drift_strength: Strength of the drift (0-1).
        drift_point: Point where drift occurs (None for gradual).
        noise_level: Level of noise in the data.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
        
    Raises:
        ValueError: If parameters are invalid.
    """
    set_seed(random_state)
    
    if drift_strength < 0 or drift_strength > 1:
        raise ValueError("drift_strength must be between 0 and 1")
    
    if drift_point is None:
        drift_point = n_samples // 2
    
    # Generate base dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_redundant=0,
        n_informative=n_features,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, X.shape)
        X = X + noise
    
    # Apply drift
    if drift_type == "concept":
        X, y = _apply_concept_drift(X, y, drift_strength, drift_point)
    elif drift_type == "covariate":
        X, y = _apply_covariate_drift(X, y, drift_strength, drift_point)
    elif drift_type == "label":
        X, y = _apply_label_drift(X, y, drift_strength, drift_point)
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    return X_train, y_train, X_test, y_test


def _apply_concept_drift(
    X: np.ndarray, 
    y: np.ndarray, 
    drift_strength: float, 
    drift_point: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply concept drift by changing decision boundaries.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        drift_strength: Strength of drift.
        drift_point: Point where drift occurs.
        
    Returns:
        Tuple of drifted (X, y).
    """
    X_drifted = X.copy()
    y_drifted = y.copy()
    
    # Gradually change feature weights
    drift_samples = X[drift_point:]
    drift_factor = np.linspace(0, drift_strength, len(drift_samples))
    
    for i, (sample, factor) in enumerate(zip(drift_samples, drift_factor)):
        # Modify features to simulate concept drift
        X_drifted[drift_point + i] = sample * (1 + factor * np.random.normal(0, 0.1, sample.shape))
    
    return X_drifted, y_drifted


def _apply_covariate_drift(
    X: np.ndarray, 
    y: np.ndarray, 
    drift_strength: float, 
    drift_point: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply covariate drift by changing feature distributions.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        drift_strength: Strength of drift.
        drift_point: Point where drift occurs.
        
    Returns:
        Tuple of drifted (X, y).
    """
    X_drifted = X.copy()
    y_drifted = y.copy()
    
    # Shift feature distributions
    drift_samples = X[drift_point:]
    shift = np.random.normal(0, drift_strength, X.shape[1])
    
    for i, sample in enumerate(drift_samples):
        X_drifted[drift_point + i] = sample + shift
    
    return X_drifted, y_drifted


def _apply_label_drift(
    X: np.ndarray, 
    y: np.ndarray, 
    drift_strength: float, 
    drift_point: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply label drift by changing class distributions.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        drift_strength: Strength of drift.
        drift_point: Point where drift occurs.
        
    Returns:
        Tuple of drifted (X, y).
    """
    X_drifted = X.copy()
    y_drifted = y.copy()
    
    # Change labels for a fraction of samples
    drift_samples = y[drift_point:]
    n_drift = int(len(drift_samples) * drift_strength)
    
    drift_indices = np.random.choice(
        len(drift_samples), 
        size=n_drift, 
        replace=False
    )
    
    for idx in drift_indices:
        # Change label to a different class
        current_label = y_drifted[drift_point + idx]
        available_labels = [l for l in np.unique(y) if l != current_label]
        if available_labels:
            y_drifted[drift_point + idx] = np.random.choice(available_labels)
    
    return X_drifted, y_drifted


def load_dataset(
    dataset_name: str = "iris",
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, any]]:
    """Load a standard dataset.
    
    Args:
        dataset_name: Name of dataset ('iris', 'synthetic').
        test_size: Fraction of data for testing.
        random_state: Random seed.
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, metadata).
        
    Raises:
        ValueError: If dataset name is unknown.
    """
    set_seed(random_state)
    
    if dataset_name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        
        metadata = {
            "feature_names": data.feature_names,
            "target_names": data.target_names,
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "n_samples": len(X),
        }
        
    elif dataset_name == "synthetic":
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            random_state=random_state,
        )
        
        metadata = {
            "feature_names": [f"feature_{i}" for i in range(X.shape[1])],
            "target_names": ["class_0", "class_1"],
            "n_features": X.shape[1],
            "n_classes": 2,
            "n_samples": len(X),
        }
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, y_train, X_test, y_test, metadata


def preprocess_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: str = "standard",
    fit_on_train: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess data using specified method.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        method: Preprocessing method ('standard', 'minmax', 'robust').
        fit_on_train: Whether to fit scaler only on training data.
        
    Returns:
        Tuple of preprocessed (X_train, X_test).
        
    Raises:
        ValueError: If preprocessing method is unknown.
    """
    validate_inputs(X_train)
    validate_inputs(X_test)
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
    
    if fit_on_train:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_combined = np.vstack([X_train, X_test])
        X_combined_scaled = scaler.fit_transform(X_combined)
        X_train_scaled = X_combined_scaled[:len(X_train)]
        X_test_scaled = X_combined_scaled[len(X_train):]
    
    return X_train_scaled, X_test_scaled


def create_metadata(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None,
    sensitive_attributes: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Create metadata dictionary for dataset.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        feature_names: Names of features.
        target_names: Names of target classes.
        sensitive_attributes: Names of sensitive attributes.
        
    Returns:
        Metadata dictionary.
    """
    metadata = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
        "feature_names": feature_names or [f"feature_{i}" for i in range(X.shape[1])],
        "target_names": target_names or [f"class_{i}" for i in np.unique(y)],
        "sensitive_attributes": sensitive_attributes or [],
        "feature_types": ["continuous"] * X.shape[1],
        "feature_ranges": {
            f"feature_{i}": [float(X[:, i].min()), float(X[:, i].max())]
            for i in range(X.shape[1])
        },
    }
    
    return metadata
