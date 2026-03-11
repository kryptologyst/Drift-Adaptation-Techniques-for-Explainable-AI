"""Model adaptation strategies for handling drift."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from ..utils import validate_inputs, set_seed


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    def __init__(self, random_state: int = 42):
        """Initialize model adapter.
        
        Args:
            random_state: Random seed.
        """
        self.random_state = random_state
        self.is_fitted = False
        
    @abstractmethod
    def adapt_model(
        self, 
        model: BaseEstimator, 
        X_new: np.ndarray, 
        y_new: np.ndarray,
        **kwargs
    ) -> BaseEstimator:
        """Adapt model to new data.
        
        Args:
            model: Model to adapt.
            X_new: New feature data.
            y_new: New target data.
            **kwargs: Additional adaptation parameters.
            
        Returns:
            Adapted model.
        """
        pass


class OnlineLearningAdapter(ModelAdapter):
    """Adapter using online learning techniques."""
    
    def __init__(
        self, 
        learning_rate: float = 0.01,
        random_state: int = 42
    ):
        """Initialize online learning adapter.
        
        Args:
            learning_rate: Learning rate for online learning.
            random_state: Random seed.
        """
        super().__init__(random_state)
        self.learning_rate = learning_rate
        self.online_model = None
        
    def adapt_model(
        self, 
        model: BaseEstimator, 
        X_new: np.ndarray, 
        y_new: np.ndarray,
        **kwargs
    ) -> BaseEstimator:
        """Adapt model using online learning.
        
        Args:
            model: Model to adapt.
            X_new: New feature data.
            y_new: New target data.
            **kwargs: Additional parameters.
            
        Returns:
            Adapted model.
        """
        validate_inputs(X_new, y_new)
        set_seed(self.random_state)
        
        # Create online learning model if not exists
        if self.online_model is None:
            self.online_model = SGDClassifier(
                learning_rate='adaptive',
                eta0=self.learning_rate,
                random_state=self.random_state
            )
            
            # Initialize with original model predictions
            X_train = kwargs.get('X_train', X_new[:10])  # Use small sample for initialization
            y_train = kwargs.get('y_train', y_new[:10])
            self.online_model.fit(X_train, y_train)
        
        # Update model with new data
        self.online_model.partial_fit(X_new, y_new)
        
        return self.online_model


class EnsembleAdapter(ModelAdapter):
    """Adapter using ensemble methods for adaptation."""
    
    def __init__(
        self, 
        n_estimators: int = 5,
        adaptation_weight: float = 0.3,
        random_state: int = 42
    ):
        """Initialize ensemble adapter.
        
        Args:
            n_estimators: Number of ensemble members.
            adaptation_weight: Weight for new data in adaptation.
            random_state: Random seed.
        """
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.adaptation_weight = adaptation_weight
        self.ensemble_models = []
        
    def adapt_model(
        self, 
        model: BaseEstimator, 
        X_new: np.ndarray, 
        y_new: np.ndarray,
        **kwargs
    ) -> BaseEstimator:
        """Adapt model using ensemble methods.
        
        Args:
            model: Model to adapt.
            X_new: New feature data.
            y_new: New target data.
            **kwargs: Additional parameters.
            
        Returns:
            Adapted ensemble model.
        """
        validate_inputs(X_new, y_new)
        set_seed(self.random_state)
        
        # Create ensemble of adapted models
        adapted_models = []
        
        for i in range(self.n_estimators):
            # Create a copy of the original model
            adapted_model = type(model)(**model.get_params())
            adapted_model.random_state = self.random_state + i
            
            # Train on combination of old and new data
            X_train = kwargs.get('X_train', X_new)
            y_train = kwargs.get('y_train', y_new)
            
            # Weighted combination of old and new data
            if 'X_old' in kwargs and 'y_old' in kwargs:
                X_old, y_old = kwargs['X_old'], kwargs['y_old']
                
                # Sample from old data with reduced weight
                n_old_samples = int(len(X_old) * (1 - self.adaptation_weight))
                if n_old_samples > 0:
                    old_indices = np.random.choice(
                        len(X_old), 
                        size=n_old_samples, 
                        replace=False
                    )
                    X_combined = np.vstack([X_old[old_indices], X_new])
                    y_combined = np.hstack([y_old[old_indices], y_new])
                else:
                    X_combined, y_combined = X_new, y_new
            else:
                X_combined, y_combined = X_new, y_new
            
            # Train adapted model
            adapted_model.fit(X_combined, y_combined)
            adapted_models.append(adapted_model)
        
        # Create voting ensemble
        ensemble_model = VotingClassifier(
            estimators=[(f'model_{i}', model) for i, model in enumerate(adapted_models)],
            voting='soft'
        )
        
        self.ensemble_models = adapted_models
        return ensemble_model


class UncertaintyBasedAdapter(ModelAdapter):
    """Adapter using uncertainty-based adaptation."""
    
    def __init__(
        self, 
        uncertainty_threshold: float = 0.5,
        adaptation_rate: float = 0.1,
        random_state: int = 42
    ):
        """Initialize uncertainty-based adapter.
        
        Args:
            uncertainty_threshold: Threshold for high uncertainty.
            adaptation_rate: Rate of adaptation.
            random_state: Random seed.
        """
        super().__init__(random_state)
        self.uncertainty_threshold = uncertainty_threshold
        self.adaptation_rate = adaptation_rate
        self.uncertainty_history = []
        
    def adapt_model(
        self, 
        model: BaseEstimator, 
        X_new: np.ndarray, 
        y_new: np.ndarray,
        **kwargs
    ) -> BaseEstimator:
        """Adapt model based on prediction uncertainty.
        
        Args:
            model: Model to adapt.
            X_new: New feature data.
            y_new: New target data.
            **kwargs: Additional parameters.
            
        Returns:
            Adapted model.
        """
        validate_inputs(X_new, y_new)
        set_seed(self.random_state)
        
        # Compute prediction uncertainty
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X_new)
            uncertainty = 1 - np.max(predictions, axis=1)
        else:
            # Fallback: use prediction confidence
            predictions = model.predict(X_new)
            uncertainty = np.abs(predictions - y_new)
        
        self.uncertainty_history.extend(uncertainty)
        
        # Identify high-uncertainty samples
        high_uncertainty_mask = uncertainty > self.uncertainty_threshold
        X_uncertain = X_new[high_uncertainty_mask]
        y_uncertain = y_new[high_uncertainty_mask]
        
        if len(X_uncertain) > 0:
            # Create adapted model
            adapted_model = type(model)(**model.get_params())
            adapted_model.random_state = self.random_state
            
            # Train on uncertain samples with higher weight
            X_train = kwargs.get('X_train', X_new)
            y_train = kwargs.get('y_train', y_new)
            
            # Combine with uncertain samples
            X_combined = np.vstack([X_train, X_uncertain])
            y_combined = np.hstack([y_train, y_uncertain])
            
            # Duplicate uncertain samples to increase their weight
            n_duplicates = int(len(X_uncertain) * self.adaptation_rate)
            if n_duplicates > 0:
                duplicate_indices = np.random.choice(
                    len(X_uncertain), 
                    size=n_duplicates, 
                    replace=True
                )
                X_combined = np.vstack([X_combined, X_uncertain[duplicate_indices]])
                y_combined = np.hstack([y_combined, y_uncertain[duplicate_indices]])
            
            adapted_model.fit(X_combined, y_combined)
            return adapted_model
        
        return model


class RetrainingAdapter(ModelAdapter):
    """Adapter using full model retraining."""
    
    def __init__(
        self, 
        retraining_frequency: int = 100,
        random_state: int = 42
    ):
        """Initialize retraining adapter.
        
        Args:
            retraining_frequency: Frequency of retraining (in samples).
            random_state: Random seed.
        """
        super().__init__(random_state)
        self.retraining_frequency = retraining_frequency
        self.sample_count = 0
        
    def adapt_model(
        self, 
        model: BaseEstimator, 
        X_new: np.ndarray, 
        y_new: np.ndarray,
        **kwargs
    ) -> BaseEstimator:
        """Adapt model using full retraining.
        
        Args:
            model: Model to adapt.
            X_new: New feature data.
            y_new: New target data.
            **kwargs: Additional parameters.
            
        Returns:
            Retrained model.
        """
        validate_inputs(X_new, y_new)
        set_seed(self.random_state)
        
        self.sample_count += len(X_new)
        
        # Check if retraining is needed
        if self.sample_count >= self.retraining_frequency:
            # Create new model
            retrained_model = type(model)(**model.get_params())
            retrained_model.random_state = self.random_state
            
            # Combine all available data
            X_train = kwargs.get('X_train', X_new)
            y_train = kwargs.get('y_train', y_new)
            
            if 'X_old' in kwargs and 'y_old' in kwargs:
                X_old, y_old = kwargs['X_old'], kwargs['y_old']
                X_combined = np.vstack([X_old, X_train])
                y_combined = np.hstack([y_old, y_train])
            else:
                X_combined, y_combined = X_train, y_train
            
            # Retrain model
            retrained_model.fit(X_combined, y_combined)
            self.sample_count = 0
            
            return retrained_model
        
        return model


def create_model_adapter(
    strategy: str, 
    **kwargs
) -> ModelAdapter:
    """Factory function to create model adapters.
    
    Args:
        strategy: Adaptation strategy ('online', 'ensemble', 'uncertainty', 'retraining').
        **kwargs: Additional arguments for the adapter.
        
    Returns:
        Model adapter instance.
        
    Raises:
        ValueError: If strategy is unknown.
    """
    adapters = {
        "online": OnlineLearningAdapter,
        "ensemble": EnsembleAdapter,
        "uncertainty": UncertaintyBasedAdapter,
        "retraining": RetrainingAdapter,
    }
    
    if strategy not in adapters:
        raise ValueError(f"Unknown adaptation strategy: {strategy}")
    
    return adapters[strategy](**kwargs)
