"""Test suite for drift detection and adaptation package."""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from drift_adaptation import (
    PSIDetector,
    KSDetector,
    MMDDetector,
    OnlineLearningAdapter,
    EnsembleAdapter,
    UncertaintyBasedAdapter,
    RetrainingAdapter,
    generate_synthetic_data,
    DriftMetrics,
    AdaptationMetrics,
    ComprehensiveMetrics,
    DriftVisualizer,
)


class TestDriftDetectors:
    """Test drift detection methods."""
    
    def setup_method(self):
        """Set up test data."""
        self.X_ref, self.y_ref = make_classification(
            n_samples=200, n_features=5, n_classes=2, random_state=42
        )
        self.X_test, self.y_test = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=123
        )
    
    def test_psi_detector(self):
        """Test PSI detector."""
        detector = PSIDetector(threshold=0.1)
        detector.fit(self.X_ref)
        
        drift_detected, drift_score = detector.detect_drift(self.X_test)
        
        assert isinstance(drift_detected, bool)
        assert isinstance(drift_score, float)
        assert drift_score >= 0
    
    def test_ks_detector(self):
        """Test KS detector."""
        detector = KSDetector(threshold=0.05)
        detector.fit(self.X_ref)
        
        drift_detected, drift_score = detector.detect_drift(self.X_test)
        
        assert isinstance(drift_detected, bool)
        assert isinstance(drift_score, float)
        assert 0 <= drift_score <= 1
    
    def test_mmd_detector(self):
        """Test MMD detector."""
        detector = MMDDetector(threshold=0.05)
        detector.fit(self.X_ref)
        
        drift_detected, drift_score = detector.detect_drift(self.X_test)
        
        assert isinstance(drift_detected, bool)
        assert isinstance(drift_score, float)
        assert drift_score >= 0
    
    def test_detector_fit_detect(self):
        """Test fit_detect method."""
        detector = PSIDetector(threshold=0.1)
        
        drift_detected, drift_score = detector.fit_detect(self.X_ref, self.X_test)
        
        assert isinstance(drift_detected, bool)
        assert isinstance(drift_score, float)


class TestModelAdapters:
    """Test model adaptation methods."""
    
    def setup_method(self):
        """Set up test data and model."""
        self.X_train, self.y_train = make_classification(
            n_samples=200, n_features=5, n_classes=2, random_state=42
        )
        self.X_test, self.y_test = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=123
        )
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def test_online_learning_adapter(self):
        """Test online learning adapter."""
        adapter = OnlineLearningAdapter(learning_rate=0.01)
        
        adapted_model = adapter.adapt_model(
            self.model, self.X_test[:10], self.y_test[:10]
        )
        
        assert adapted_model is not None
        assert hasattr(adapted_model, 'predict')
    
    def test_ensemble_adapter(self):
        """Test ensemble adapter."""
        adapter = EnsembleAdapter(n_estimators=3, adaptation_weight=0.3)
        
        adapted_model = adapter.adapt_model(
            self.model, self.X_test[:10], self.y_test[:10],
            X_train=self.X_train, y_train=self.y_train
        )
        
        assert adapted_model is not None
        assert hasattr(adapted_model, 'predict')
    
    def test_uncertainty_based_adapter(self):
        """Test uncertainty-based adapter."""
        adapter = UncertaintyBasedAdapter(
            uncertainty_threshold=0.5, adaptation_rate=0.1
        )
        
        adapted_model = adapter.adapt_model(
            self.model, self.X_test[:10], self.y_test[:10]
        )
        
        assert adapted_model is not None
        assert hasattr(adapted_model, 'predict')
    
    def test_retraining_adapter(self):
        """Test retraining adapter."""
        adapter = RetrainingAdapter(retraining_frequency=50)
        
        adapted_model = adapter.adapt_model(
            self.model, self.X_test[:10], self.y_test[:10],
            X_train=self.X_train, y_train=self.y_train
        )
        
        assert adapted_model is not None
        assert hasattr(adapted_model, 'predict')


class TestDataGeneration:
    """Test data generation functions."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        X_train, y_train, X_test, y_test = generate_synthetic_data(
            n_samples=100, n_features=5, drift_type="concept", drift_strength=0.3
        )
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]
    
    def test_generate_synthetic_data_different_drift_types(self):
        """Test different drift types."""
        drift_types = ["concept", "covariate", "label"]
        
        for drift_type in drift_types:
            X_train, y_train, X_test, y_test = generate_synthetic_data(
                n_samples=50, drift_type=drift_type, drift_strength=0.2
            )
            
            assert X_train.shape[0] > 0
            assert X_test.shape[0] > 0


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_drift_metrics(self):
        """Test drift detection metrics."""
        metrics = DriftMetrics()
        
        # Add some test data
        metrics.update(True, True, 0.8)
        metrics.update(False, False, 0.2)
        metrics.update(True, False, 0.6)
        metrics.update(False, True, 0.3)
        
        results = metrics.compute_metrics()
        
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results
        assert 0 <= results["accuracy"] <= 1
    
    def test_adaptation_metrics(self):
        """Test adaptation performance metrics."""
        metrics = AdaptationMetrics()
        
        # Add some test data
        for i in range(10):
            metrics.update_accuracy(0.8 + i * 0.01)
            metrics.update_stability(0.9 - i * 0.01)
        
        results = metrics.compute_metrics()
        
        assert "mean_accuracy" in results
        assert "mean_stability" in results
        assert "accuracy_trend" in results
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics."""
        metrics = ComprehensiveMetrics()
        
        # Add test data
        metrics.update_drift_detection(True, True, 0.8)
        metrics.update_adaptation_performance(0.85, 0.9)
        metrics.update_computational_metrics(0.1, 0.2)
        
        results = metrics.compute_all_metrics()
        
        assert "drift_detection" in results
        assert "adaptation_performance" in results
        assert "computational" in results


class TestVisualization:
    """Test visualization utilities."""
    
    def test_drift_visualizer_initialization(self):
        """Test drift visualizer initialization."""
        visualizer = DriftVisualizer()
        
        assert visualizer.figsize == (12, 8)
        assert len(visualizer.colors) > 0
    
    def test_plot_drift_detection_results(self):
        """Test drift detection results plotting."""
        visualizer = DriftVisualizer()
        
        drift_scores = [0.1, 0.3, 0.2, 0.4, 0.1]
        drift_labels = [False, True, False, True, False]
        
        fig = visualizer.plot_drift_detection_results(
            drift_scores, drift_labels, title="Test Plot"
        )
        
        assert fig is not None
    
    def test_plot_adaptation_performance(self):
        """Test adaptation performance plotting."""
        visualizer = DriftVisualizer()
        
        accuracy_history = [0.8, 0.82, 0.85, 0.87, 0.89]
        
        fig = visualizer.plot_adaptation_performance(
            accuracy_history, title="Test Plot"
        )
        
        assert fig is not None


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_drift_detection(self):
        """Test end-to-end drift detection workflow."""
        # Generate data with drift
        X_train, y_train, X_test, y_test = generate_synthetic_data(
            n_samples=200, drift_type="concept", drift_strength=0.3
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Detect drift
        detector = PSIDetector(threshold=0.1)
        detector.fit(X_train)
        drift_detected, drift_score = detector.detect_drift(X_test)
        
        assert isinstance(drift_detected, bool)
        assert isinstance(drift_score, float)
    
    def test_end_to_end_adaptation(self):
        """Test end-to-end adaptation workflow."""
        # Generate data
        X_train, y_train, X_test, y_test = generate_synthetic_data(
            n_samples=200, drift_type="concept", drift_strength=0.3
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Adapt model
        adapter = OnlineLearningAdapter()
        adapted_model = adapter.adapt_model(
            model, X_test[:20], y_test[:20]
        )
        
        assert adapted_model is not None
        assert hasattr(adapted_model, 'predict')


@pytest.mark.slow
class TestPerformance:
    """Performance tests."""
    
    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Generate larger dataset
        X_train, y_train, X_test, y_test = generate_synthetic_data(
            n_samples=1000, n_features=20, drift_type="concept", drift_strength=0.3
        )
        
        # Test drift detection performance
        detector = PSIDetector(threshold=0.1)
        detector.fit(X_train)
        
        drift_detected, drift_score = detector.detect_drift(X_test)
        
        assert isinstance(drift_detected, bool)
        assert isinstance(drift_score, float)
    
    def test_memory_usage(self):
        """Test memory usage with multiple detectors."""
        X_train, y_train, X_test, y_test = generate_synthetic_data(
            n_samples=500, drift_type="concept", drift_strength=0.3
        )
        
        detectors = [
            PSIDetector(threshold=0.1),
            KSDetector(threshold=0.05),
            MMDDetector(threshold=0.05),
        ]
        
        results = []
        for detector in detectors:
            detector.fit(X_train)
            drift_detected, drift_score = detector.detect_drift(X_test)
            results.append((drift_detected, drift_score))
        
        assert len(results) == len(detectors)
        for result in results:
            assert isinstance(result[0], bool)
            assert isinstance(result[1], float)


if __name__ == "__main__":
    pytest.main([__file__])
