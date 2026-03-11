"""Evaluation metrics for drift detection and adaptation."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class DriftMetrics:
    """Metrics for evaluating drift detection performance."""
    
    def __init__(self):
        """Initialize drift metrics."""
        self.drift_predictions = []
        self.drift_labels = []
        self.drift_scores = []
        
    def update(self, prediction: bool, label: bool, score: float) -> None:
        """Update metrics with new prediction.
        
        Args:
            prediction: Predicted drift (True/False).
            label: True drift label (True/False).
            score: Drift detection score.
        """
        self.drift_predictions.append(prediction)
        self.drift_labels.append(label)
        self.drift_scores.append(score)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute drift detection metrics.
        
        Returns:
            Dictionary of metrics.
        """
        if not self.drift_predictions:
            return {}
        
        predictions = np.array(self.drift_predictions)
        labels = np.array(self.drift_labels)
        scores = np.array(self.drift_scores)
        
        # Basic classification metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # ROC AUC if we have scores
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.0
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.drift_predictions = []
        self.drift_labels = []
        self.drift_scores = []


class AdaptationMetrics:
    """Metrics for evaluating model adaptation performance."""
    
    def __init__(self):
        """Initialize adaptation metrics."""
        self.accuracy_history = []
        self.stability_scores = []
        self.convergence_metrics = []
        
    def update_accuracy(self, accuracy: float) -> None:
        """Update accuracy history.
        
        Args:
            accuracy: Model accuracy.
        """
        self.accuracy_history.append(accuracy)
    
    def update_stability(self, stability_score: float) -> None:
        """Update stability score.
        
        Args:
            stability_score: Model stability score.
        """
        self.stability_scores.append(stability_score)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute adaptation metrics.
        
        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        
        if self.accuracy_history:
            accuracies = np.array(self.accuracy_history)
            metrics.update({
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "min_accuracy": np.min(accuracies),
                "max_accuracy": np.max(accuracies),
                "accuracy_trend": self._compute_trend(accuracies),
            })
        
        if self.stability_scores:
            stabilities = np.array(self.stability_scores)
            metrics.update({
                "mean_stability": np.mean(stabilities),
                "std_stability": np.std(stabilities),
                "min_stability": np.min(stabilities),
                "max_stability": np.max(stabilities),
            })
        
        return metrics
    
    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute trend in values.
        
        Args:
            values: Array of values.
            
        Returns:
            Trend coefficient.
        """
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]
        return trend
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.accuracy_history = []
        self.stability_scores = []
        self.convergence_metrics = []


class ComprehensiveMetrics:
    """Comprehensive metrics for drift detection and adaptation."""
    
    def __init__(self):
        """Initialize comprehensive metrics."""
        self.drift_metrics = DriftMetrics()
        self.adaptation_metrics = AdaptationMetrics()
        self.computational_metrics = {}
        
    def update_drift_detection(
        self, 
        prediction: bool, 
        label: bool, 
        score: float
    ) -> None:
        """Update drift detection metrics.
        
        Args:
            prediction: Predicted drift.
            label: True drift label.
            score: Detection score.
        """
        self.drift_metrics.update(prediction, label, score)
    
    def update_adaptation_performance(
        self, 
        accuracy: float, 
        stability: Optional[float] = None
    ) -> None:
        """Update adaptation performance metrics.
        
        Args:
            accuracy: Model accuracy.
            stability: Optional stability score.
        """
        self.adaptation_metrics.update_accuracy(accuracy)
        if stability is not None:
            self.adaptation_metrics.update_stability(stability)
    
    def update_computational_metrics(
        self, 
        detection_time: float, 
        adaptation_time: float,
        memory_usage: Optional[float] = None
    ) -> None:
        """Update computational metrics.
        
        Args:
            detection_time: Time for drift detection.
            adaptation_time: Time for model adaptation.
            memory_usage: Optional memory usage.
        """
        self.computational_metrics.update({
            "detection_time": detection_time,
            "adaptation_time": adaptation_time,
            "total_time": detection_time + adaptation_time,
        })
        
        if memory_usage is not None:
            self.computational_metrics["memory_usage"] = memory_usage
    
    def compute_all_metrics(self) -> Dict[str, Union[Dict[str, float], float]]:
        """Compute all metrics.
        
        Returns:
            Dictionary containing all computed metrics.
        """
        all_metrics = {
            "drift_detection": self.drift_metrics.compute_metrics(),
            "adaptation_performance": self.adaptation_metrics.compute_metrics(),
            "computational": self.computational_metrics.copy(),
        }
        
        return all_metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.drift_metrics.reset()
        self.adaptation_metrics.reset()
        self.computational_metrics = {}
    
    def get_summary(self) -> str:
        """Get a summary of all metrics.
        
        Returns:
            String summary of metrics.
        """
        metrics = self.compute_all_metrics()
        
        summary = "=== Drift Detection and Adaptation Metrics ===\n\n"
        
        # Drift detection metrics
        drift_metrics = metrics["drift_detection"]
        if drift_metrics:
            summary += "Drift Detection:\n"
            summary += f"  Accuracy: {drift_metrics.get('accuracy', 0):.3f}\n"
            summary += f"  Precision: {drift_metrics.get('precision', 0):.3f}\n"
            summary += f"  Recall: {drift_metrics.get('recall', 0):.3f}\n"
            summary += f"  F1-Score: {drift_metrics.get('f1_score', 0):.3f}\n"
            summary += f"  AUC: {drift_metrics.get('auc', 0):.3f}\n\n"
        
        # Adaptation metrics
        adapt_metrics = metrics["adaptation_performance"]
        if adapt_metrics:
            summary += "Adaptation Performance:\n"
            summary += f"  Mean Accuracy: {adapt_metrics.get('mean_accuracy', 0):.3f}\n"
            summary += f"  Accuracy Std: {adapt_metrics.get('std_accuracy', 0):.3f}\n"
            summary += f"  Accuracy Trend: {adapt_metrics.get('accuracy_trend', 0):.3f}\n"
            summary += f"  Mean Stability: {adapt_metrics.get('mean_stability', 0):.3f}\n\n"
        
        # Computational metrics
        comp_metrics = metrics["computational"]
        if comp_metrics:
            summary += "Computational Performance:\n"
            summary += f"  Detection Time: {comp_metrics.get('detection_time', 0):.3f}s\n"
            summary += f"  Adaptation Time: {comp_metrics.get('adaptation_time', 0):.3f}s\n"
            summary += f"  Total Time: {comp_metrics.get('total_time', 0):.3f}s\n"
        
        return summary


def evaluate_drift_detection(
    predictions: List[bool],
    labels: List[bool],
    scores: List[float]
) -> Dict[str, float]:
    """Evaluate drift detection performance.
    
    Args:
        predictions: List of drift predictions.
        labels: List of true drift labels.
        scores: List of detection scores.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    metrics = DriftMetrics()
    
    for pred, label, score in zip(predictions, labels, scores):
        metrics.update(pred, label, score)
    
    return metrics.compute_metrics()


def evaluate_adaptation_performance(
    accuracy_history: List[float],
    stability_history: Optional[List[float]] = None
) -> Dict[str, float]:
    """Evaluate adaptation performance.
    
    Args:
        accuracy_history: List of accuracy values over time.
        stability_history: Optional list of stability scores.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    metrics = AdaptationMetrics()
    
    for acc in accuracy_history:
        metrics.update_accuracy(acc)
    
    if stability_history:
        for stab in stability_history:
            metrics.update_stability(stab)
    
    return metrics.compute_metrics()
