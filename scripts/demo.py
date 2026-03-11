#!/usr/bin/env python3
"""Main demonstration script for drift detection and adaptation techniques.

This script demonstrates the comprehensive drift detection and adaptation
capabilities of the package with various synthetic datasets and scenarios.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from drift_adaptation import (
    DriftDetector,
    PSIDetector,
    KSDetector,
    MMDDetector,
    ModelAdapter,
    OnlineLearningAdapter,
    EnsembleAdapter,
    UncertaintyBasedAdapter,
    RetrainingAdapter,
    generate_synthetic_data,
    DriftMetrics,
    AdaptationMetrics,
    ComprehensiveMetrics,
    DriftVisualizer,
    create_drift_detector,
    create_model_adapter,
)


def run_drift_detection_experiment(
    drift_type: str = "concept",
    drift_strength: float = 0.3,
    n_samples: int = 1000,
    random_state: int = 42
) -> Dict[str, any]:
    """Run a comprehensive drift detection experiment.
    
    Args:
        drift_type: Type of drift to simulate.
        drift_strength: Strength of the drift.
        n_samples: Number of samples to generate.
        random_state: Random seed.
        
    Returns:
        Dictionary containing experiment results.
    """
    print(f"\n=== Running Drift Detection Experiment ===")
    print(f"Drift Type: {drift_type}")
    print(f"Drift Strength: {drift_strength}")
    print(f"Number of Samples: {n_samples}")
    
    # Generate synthetic data with drift
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        n_samples=n_samples,
        drift_type=drift_type,
        drift_strength=drift_strength,
        random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Initialize drift detectors
    detectors = {
        "PSI": PSIDetector(threshold=0.1),
        "KS": KSDetector(threshold=0.05),
        "MMD": MMDDetector(threshold=0.05),
    }
    
    # Train baseline model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate baseline performance
    baseline_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Baseline accuracy: {baseline_accuracy:.3f}")
    
    # Test drift detection
    results = {}
    for name, detector in detectors.items():
        print(f"\nTesting {name} detector...")
        
        start_time = time.time()
        
        # Fit detector on training data
        detector.fit(X_train)
        
        # Detect drift in test data
        drift_detected, drift_score = detector.detect_drift(X_test)
        
        detection_time = time.time() - start_time
        
        results[name] = {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "detection_time": detection_time,
        }
        
        print(f"  Drift detected: {drift_detected}")
        print(f"  Drift score: {drift_score:.3f}")
        print(f"  Detection time: {detection_time:.3f}s")
    
    return {
        "data": (X_train, y_train, X_test, y_test),
        "model": model,
        "baseline_accuracy": baseline_accuracy,
        "detection_results": results,
    }


def run_adaptation_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: RandomForestClassifier,
    random_state: int = 42
) -> Dict[str, any]:
    """Run a comprehensive model adaptation experiment.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        model: Baseline model.
        random_state: Random seed.
        
    Returns:
        Dictionary containing adaptation results.
    """
    print(f"\n=== Running Model Adaptation Experiment ===")
    
    # Initialize adapters
    adapters = {
        "Online Learning": OnlineLearningAdapter(learning_rate=0.01),
        "Ensemble": EnsembleAdapter(n_estimators=5, adaptation_weight=0.3),
        "Uncertainty-Based": UncertaintyBasedAdapter(
            uncertainty_threshold=0.5,
            adaptation_rate=0.1
        ),
        "Retraining": RetrainingAdapter(retraining_frequency=100),
    }
    
    # Simulate streaming data for adaptation
    batch_size = 50
    n_batches = len(X_test) // batch_size
    
    results = {}
    
    for name, adapter in adapters.items():
        print(f"\nTesting {name} adapter...")
        
        start_time = time.time()
        
        # Initialize adapted model
        adapted_model = model
        accuracy_history = []
        
        # Process data in batches
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_test))
            
            X_batch = X_test[start_idx:end_idx]
            y_batch = y_test[start_idx:end_idx]
            
            # Adapt model
            adapted_model = adapter.adapt_model(
                adapted_model,
                X_batch,
                y_batch,
                X_train=X_train,
                y_train=y_train
            )
            
            # Evaluate adapted model
            accuracy = accuracy_score(y_test, adapted_model.predict(X_test))
            accuracy_history.append(accuracy)
        
        adaptation_time = time.time() - start_time
        
        results[name] = {
            "final_accuracy": accuracy_history[-1],
            "accuracy_history": accuracy_history,
            "adaptation_time": adaptation_time,
            "model": adapted_model,
        }
        
        print(f"  Final accuracy: {accuracy_history[-1]:.3f}")
        print(f"  Adaptation time: {adaptation_time:.3f}s")
    
    return results


def run_comprehensive_evaluation(
    detection_results: Dict[str, any],
    adaptation_results: Dict[str, any],
    drift_type: str,
    drift_strength: float
) -> None:
    """Run comprehensive evaluation and create visualizations.
    
    Args:
        detection_results: Results from drift detection experiment.
        adaptation_results: Results from adaptation experiment.
        drift_type: Type of drift that was simulated.
        drift_strength: Strength of the drift.
    """
    print(f"\n=== Comprehensive Evaluation ===")
    
    # Initialize visualizer
    visualizer = DriftVisualizer()
    
    # Create drift detection comparison
    detection_metrics = {}
    for name, result in detection_results["detection_results"].items():
        detection_metrics[name] = {
            "drift_score": result["drift_score"],
            "detection_time": result["detection_time"],
        }
    
    # Create adaptation comparison
    adaptation_metrics = {}
    for name, result in adaptation_results.items():
        adaptation_metrics[name] = {
            "final_accuracy": result["final_accuracy"],
            "adaptation_time": result["adaptation_time"],
        }
    
    # Plot comparisons
    fig1 = visualizer.plot_metrics_comparison(
        detection_metrics,
        metric_name="drift_score",
        title="Drift Detection Scores Comparison"
    )
    visualizer.save_figure(fig1, "assets/drift_detection_comparison.png")
    
    fig2 = visualizer.plot_metrics_comparison(
        adaptation_metrics,
        metric_name="final_accuracy",
        title="Model Adaptation Performance Comparison"
    )
    visualizer.save_figure(fig2, "assets/adaptation_performance_comparison.png")
    
    # Create comprehensive dashboard
    # Simulate drift scores over time
    drift_scores = [result["drift_score"] for result in detection_results["detection_results"].values()]
    accuracy_history = adaptation_results["Online Learning"]["accuracy_history"]
    
    fig3 = visualizer.create_summary_dashboard(
        drift_scores=drift_scores,
        accuracy_history=accuracy_history,
        title=f"Drift Detection and Adaptation Dashboard\n"
              f"Drift Type: {drift_type}, Strength: {drift_strength}"
    )
    visualizer.save_figure(fig3, "assets/comprehensive_dashboard.png")
    
    print("Visualizations saved to assets/ directory")
    
    # Print summary
    print(f"\n=== Experiment Summary ===")
    print(f"Drift Type: {drift_type}")
    print(f"Drift Strength: {drift_strength}")
    print(f"Baseline Accuracy: {detection_results['baseline_accuracy']:.3f}")
    
    print(f"\nDrift Detection Results:")
    for name, result in detection_results["detection_results"].items():
        print(f"  {name}: Score={result['drift_score']:.3f}, "
              f"Detected={result['drift_detected']}, "
              f"Time={result['detection_time']:.3f}s")
    
    print(f"\nAdaptation Results:")
    for name, result in adaptation_results.items():
        print(f"  {name}: Final Accuracy={result['final_accuracy']:.3f}, "
              f"Time={result['adaptation_time']:.3f}s")


def main():
    """Main function to run comprehensive drift detection and adaptation experiments."""
    print("Drift Detection and Adaptation Techniques Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run experiments with different drift scenarios
    drift_scenarios = [
        ("concept", 0.2),
        ("concept", 0.5),
        ("covariate", 0.3),
        ("label", 0.4),
    ]
    
    all_results = {}
    
    for drift_type, drift_strength in drift_scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {drift_type.upper()} DRIFT (strength={drift_strength})")
        print(f"{'='*60}")
        
        # Run drift detection experiment
        detection_results = run_drift_detection_experiment(
            drift_type=drift_type,
            drift_strength=drift_strength,
            n_samples=1000,
            random_state=42
        )
        
        # Run adaptation experiment
        adaptation_results = run_adaptation_experiment(
            *detection_results["data"],
            detection_results["model"],
            random_state=42
        )
        
        # Run comprehensive evaluation
        run_comprehensive_evaluation(
            detection_results,
            adaptation_results,
            drift_type,
            drift_strength
        )
        
        # Store results
        all_results[f"{drift_type}_{drift_strength}"] = {
            "detection": detection_results,
            "adaptation": adaptation_results,
        }
    
    # Create final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    for scenario, results in all_results.items():
        drift_type, drift_strength = scenario.split("_")
        print(f"\n{drift_type.upper()} DRIFT (strength={drift_strength}):")
        
        # Best drift detector
        detection_results = results["detection"]["detection_results"]
        best_detector = max(detection_results.items(), 
                          key=lambda x: x[1]["drift_score"])
        print(f"  Best Detector: {best_detector[0]} (score={best_detector[1]['drift_score']:.3f})")
        
        # Best adapter
        adaptation_results = results["adaptation"]
        best_adapter = max(adaptation_results.items(),
                         key=lambda x: x[1]["final_accuracy"])
        print(f"  Best Adapter: {best_adapter[0]} (accuracy={best_adapter[1]['final_accuracy']:.3f})")
    
    print(f"\nExperiment completed! Check the assets/ directory for visualizations.")
    print(f"Run 'streamlit run demo/app.py' to launch the interactive demo.")


if __name__ == "__main__":
    main()
