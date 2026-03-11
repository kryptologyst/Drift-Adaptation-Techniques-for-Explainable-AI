"""Visualization utilities for drift detection and adaptation."""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from ..utils import set_seed


class DriftVisualizer:
    """Visualization utilities for drift detection and adaptation."""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """Initialize drift visualizer.
        
        Args:
            style: Matplotlib style.
            figsize: Default figure size.
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_drift_detection_results(
        self,
        drift_scores: List[float],
        drift_labels: List[bool],
        timestamps: Optional[List[int]] = None,
        threshold: float = 0.05,
        title: str = "Drift Detection Results"
    ) -> Figure:
        """Plot drift detection results over time.
        
        Args:
            drift_scores: List of drift detection scores.
            drift_labels: List of true drift labels.
            timestamps: Optional timestamps for x-axis.
            threshold: Detection threshold.
            title: Plot title.
            
        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        if timestamps is None:
            timestamps = list(range(len(drift_scores)))
        
        # Plot drift scores
        ax1.plot(timestamps, drift_scores, 'b-', linewidth=2, label='Drift Score')
        ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        ax1.set_ylabel('Drift Score')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drift labels
        drift_indices = [i for i, label in enumerate(drift_labels) if label]
        ax2.scatter(drift_indices, [1] * len(drift_indices), 
                   c='red', s=50, label='True Drift', alpha=0.7)
        ax2.set_ylabel('Drift Detected')
        ax2.set_xlabel('Time')
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_adaptation_performance(
        self,
        accuracy_history: List[float],
        stability_history: Optional[List[float]] = None,
        timestamps: Optional[List[int]] = None,
        title: str = "Model Adaptation Performance"
    ) -> Figure:
        """Plot model adaptation performance over time.
        
        Args:
            accuracy_history: List of accuracy values.
            stability_history: Optional list of stability scores.
            timestamps: Optional timestamps for x-axis.
            title: Plot title.
            
        Returns:
            Matplotlib figure.
        """
        if timestamps is None:
            timestamps = list(range(len(accuracy_history)))
        
        fig, axes = plt.subplots(2 if stability_history else 1, 1, 
                               figsize=self.figsize, sharex=True)
        
        if stability_history is None:
            axes = [axes]
        
        # Plot accuracy
        axes[0].plot(timestamps, accuracy_history, 'g-', linewidth=2, label='Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot stability if provided
        if stability_history:
            axes[1].plot(timestamps, stability_history, 'b-', linewidth=2, label='Stability')
            axes[1].set_ylabel('Stability Score')
            axes[1].set_xlabel('Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[0].set_xlabel('Time')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_distributions(
        self,
        X_ref: np.ndarray,
        X_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str = "Feature Distribution Comparison"
    ) -> Figure:
        """Plot feature distributions for reference and test data.
        
        Args:
            X_ref: Reference data.
            X_test: Test data.
            feature_names: Optional feature names.
            title: Plot title.
            
        Returns:
            Matplotlib figure.
        """
        n_features = X_ref.shape[1]
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i in range(n_features):
            ax = axes[i]
            
            # Plot histograms
            ax.hist(X_ref[:, i], bins=30, alpha=0.7, label='Reference', color=self.colors[0])
            ax.hist(X_test[:, i], bins=30, alpha=0.7, label='Test', color=self.colors[1])
            
            feature_name = feature_names[i] if feature_names else f'Feature {i}'
            ax.set_title(feature_name)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(
        self,
        y_true: List[bool],
        y_pred: List[bool],
        title: str = "Drift Detection Confusion Matrix"
    ) -> Figure:
        """Plot confusion matrix for drift detection.
        
        Args:
            y_true: True drift labels.
            y_pred: Predicted drift labels.
            title: Plot title.
            
        Returns:
            Matplotlib figure.
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Drift', 'Drift'],
                   yticklabels=['No Drift', 'Drift'])
        
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metric_name: str = "accuracy",
        title: str = "Methods Comparison"
    ) -> Figure:
        """Plot comparison of different methods.
        
        Args:
            metrics_dict: Dictionary of metrics for different methods.
            metric_name: Name of metric to plot.
            title: Plot title.
            
        Returns:
            Matplotlib figure.
        """
        methods = list(metrics_dict.keys())
        values = [metrics_dict[method].get(metric_name, 0) for method in methods]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.bar(methods, values, color=self.colors[:len(methods)])
        ax.set_title(title)
        ax.set_ylabel(metric_name.title())
        ax.set_xlabel('Methods')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_drift_timeline(
        self,
        data: np.ndarray,
        drift_points: List[int],
        feature_idx: int = 0,
        window_size: int = 50,
        title: str = "Drift Timeline"
    ) -> Figure:
        """Plot timeline showing data and drift points.
        
        Args:
            data: Time series data.
            drift_points: List of drift point indices.
            feature_idx: Index of feature to plot.
            window_size: Window size for smoothing.
            title: Plot title.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot data
        ax.plot(data[:, feature_idx], 'b-', alpha=0.7, label='Data')
        
        # Add smoothed line
        if len(data) > window_size:
            smoothed = np.convolve(data[:, feature_idx], 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
            ax.plot(range(window_size-1, len(data)), smoothed, 
                   'r-', linewidth=2, label=f'Smoothed (window={window_size})')
        
        # Mark drift points
        for drift_point in drift_points:
            ax.axvline(x=drift_point, color='red', linestyle='--', alpha=0.7)
            ax.text(drift_point, ax.get_ylim()[1] * 0.9, 'Drift', 
                   rotation=90, ha='right', va='top')
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Feature {feature_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: Figure, filename: str, dpi: int = 300) -> None:
        """Save figure to file.
        
        Args:
            fig: Matplotlib figure.
            filename: Output filename.
            dpi: Resolution for saving.
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    def create_summary_dashboard(
        self,
        drift_scores: List[float],
        accuracy_history: List[float],
        drift_labels: Optional[List[bool]] = None,
        title: str = "Drift Detection and Adaptation Dashboard"
    ) -> Figure:
        """Create a comprehensive dashboard.
        
        Args:
            drift_scores: List of drift detection scores.
            accuracy_history: List of accuracy values.
            drift_labels: Optional list of true drift labels.
            title: Dashboard title.
            
        Returns:
            Matplotlib figure.
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Drift scores plot
        ax1 = fig.add_subplot(gs[0, :])
        timestamps = list(range(len(drift_scores)))
        ax1.plot(timestamps, drift_scores, 'b-', linewidth=2, label='Drift Score')
        ax1.axhline(y=0.05, color='r', linestyle='--', label='Threshold')
        
        if drift_labels:
            drift_indices = [i for i, label in enumerate(drift_labels) if label]
            ax1.scatter(drift_indices, [drift_scores[i] for i in drift_indices],
                       c='red', s=50, label='True Drift', zorder=5)
        
        ax1.set_title('Drift Detection Scores')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(timestamps, accuracy_history, 'g-', linewidth=2, label='Accuracy')
        ax2.set_title('Model Accuracy Over Time')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Statistics
        ax3 = fig.add_subplot(gs[2, 0])
        stats_text = f"""
        Drift Detection Stats:
        Mean Score: {np.mean(drift_scores):.3f}
        Max Score: {np.max(drift_scores):.3f}
        Min Score: {np.min(drift_scores):.3f}
        
        Accuracy Stats:
        Mean Accuracy: {np.mean(accuracy_history):.3f}
        Final Accuracy: {accuracy_history[-1]:.3f}
        Accuracy Trend: {np.polyfit(range(len(accuracy_history)), accuracy_history, 1)[0]:.3f}
        """
        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='center')
        ax3.set_title('Statistics')
        ax3.axis('off')
        
        # Distribution plot
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.hist(drift_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax4.set_title('Drift Score Distribution')
        ax4.set_xlabel('Drift Score')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        return fig
