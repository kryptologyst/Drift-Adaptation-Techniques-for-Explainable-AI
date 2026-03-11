"""Streamlit demo application for drift detection and adaptation."""

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from drift_adaptation import (
    PSIDetector,
    KSDetector,
    MMDDetector,
    OnlineLearningAdapter,
    EnsembleAdapter,
    UncertaintyBasedAdapter,
    RetrainingAdapter,
    generate_synthetic_data,
    DriftVisualizer,
)

# Page configuration
st.set_page_config(
    page_title="Drift Detection & Adaptation Demo",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔄 Drift Detection & Adaptation Demo</h1>', 
            unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p>This demo is for <strong>research and educational purposes only</strong>. 
    Drift detection outputs may be unstable or misleading. 
    <strong>Do not use for regulated decisions without human review.</strong></p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Dataset parameters
st.sidebar.subheader("Dataset Parameters")
n_samples = st.sidebar.slider("Number of samples", 500, 2000, 1000)
n_features = st.sidebar.slider("Number of features", 5, 20, 10)
drift_type = st.sidebar.selectbox(
    "Drift type",
    ["concept", "covariate", "label"],
    help="Type of drift to simulate"
)
drift_strength = st.sidebar.slider(
    "Drift strength", 
    0.1, 0.8, 0.3, 0.1,
    help="Strength of the drift (0-1)"
)
random_state = st.sidebar.number_input("Random seed", 1, 1000, 42)

# Drift detection parameters
st.sidebar.subheader("Drift Detection")
detection_methods = st.sidebar.multiselect(
    "Detection methods",
    ["PSI", "KS", "MMD"],
    default=["PSI", "KS", "MMD"],
    help="Select drift detection methods to compare"
)

# Model adaptation parameters
st.sidebar.subheader("Model Adaptation")
adaptation_methods = st.sidebar.multiselect(
    "Adaptation methods",
    ["Online Learning", "Ensemble", "Uncertainty-Based", "Retraining"],
    default=["Online Learning", "Ensemble"],
    help="Select adaptation methods to compare"
)

# Main content
if st.button("🚀 Run Experiment", type="primary"):
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize results storage
    results = {
        "detection": {},
        "adaptation": {},
        "data": None,
        "model": None,
    }
    
    # Step 1: Generate data
    status_text.text("Generating synthetic data...")
    progress_bar.progress(10)
    
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        drift_type=drift_type,
        drift_strength=drift_strength,
        random_state=random_state
    )
    
    results["data"] = (X_train, y_train, X_test, y_test)
    
    # Step 2: Train baseline model
    status_text.text("Training baseline model...")
    progress_bar.progress(20)
    
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, model.predict(X_test))
    results["model"] = model
    
    # Step 3: Run drift detection
    status_text.text("Running drift detection...")
    progress_bar.progress(40)
    
    detectors = {
        "PSI": PSIDetector(threshold=0.1),
        "KS": KSDetector(threshold=0.05),
        "MMD": MMDDetector(threshold=0.05),
    }
    
    for method in detection_methods:
        if method in detectors:
            detector = detectors[method]
            detector.fit(X_train)
            drift_detected, drift_score = detector.detect_drift(X_test)
            
            results["detection"][method] = {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
            }
    
    # Step 4: Run model adaptation
    status_text.text("Running model adaptation...")
    progress_bar.progress(70)
    
    adapters = {
        "Online Learning": OnlineLearningAdapter(learning_rate=0.01),
        "Ensemble": EnsembleAdapter(n_estimators=5, adaptation_weight=0.3),
        "Uncertainty-Based": UncertaintyBasedAdapter(
            uncertainty_threshold=0.5,
            adaptation_rate=0.1
        ),
        "Retraining": RetrainingAdapter(retraining_frequency=100),
    }
    
    batch_size = 50
    n_batches = len(X_test) // batch_size
    
    for method in adaptation_methods:
        if method in adapters:
            adapter = adapters[method]
            adapted_model = model
            accuracy_history = []
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_test))
                
                X_batch = X_test[start_idx:end_idx]
                y_batch = y_test[start_idx:end_idx]
                
                adapted_model = adapter.adapt_model(
                    adapted_model,
                    X_batch,
                    y_batch,
                    X_train=X_train,
                    y_train=y_train
                )
                
                accuracy = accuracy_score(y_test, adapted_model.predict(X_test))
                accuracy_history.append(accuracy)
            
            results["adaptation"][method] = {
                "final_accuracy": accuracy_history[-1],
                "accuracy_history": accuracy_history,
            }
    
    # Step 5: Generate visualizations
    status_text.text("Generating visualizations...")
    progress_bar.progress(90)
    
    visualizer = DriftVisualizer()
    
    # Create plots
    if results["detection"]:
        detection_scores = [result["drift_score"] for result in results["detection"].values()]
        fig1 = visualizer.plot_drift_detection_results(
            drift_scores=detection_scores,
            drift_labels=[True] * len(detection_scores),  # Simplified
            title="Drift Detection Results"
        )
    
    if results["adaptation"]:
        accuracy_history = list(results["adaptation"].values())[0]["accuracy_history"]
        fig2 = visualizer.plot_adaptation_performance(
            accuracy_history=accuracy_history,
            title="Model Adaptation Performance"
        )
    
    progress_bar.progress(100)
    status_text.text("Experiment completed!")
    
    # Display results
    st.success("✅ Experiment completed successfully!")
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Drift Detection", "🔄 Adaptation", "📈 Visualizations"])
    
    with tab1:
        st.subheader("Experiment Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset Size", f"{n_samples:,}")
            st.metric("Features", n_features)
        
        with col2:
            st.metric("Drift Type", drift_type.title())
            st.metric("Drift Strength", f"{drift_strength:.1f}")
        
        with col3:
            st.metric("Baseline Accuracy", f"{baseline_accuracy:.3f}")
            st.metric("Test Samples", len(X_test))
        
        # Data summary
        st.subheader("Data Summary")
        data_summary = pd.DataFrame({
            "Split": ["Train", "Test"],
            "Samples": [len(X_train), len(X_test)],
            "Features": [X_train.shape[1], X_test.shape[1]],
            "Classes": [len(np.unique(y_train)), len(np.unique(y_test))],
        })
        st.dataframe(data_summary, use_container_width=True)
    
    with tab2:
        st.subheader("Drift Detection Results")
        
        if results["detection"]:
            # Detection metrics
            detection_df = pd.DataFrame([
                {
                    "Method": method,
                    "Drift Detected": result["drift_detected"],
                    "Drift Score": f"{result['drift_score']:.3f}",
                    "Status": "✅ Drift Detected" if result["drift_detected"] else "❌ No Drift"
                }
                for method, result in results["detection"].items()
            ])
            
            st.dataframe(detection_df, use_container_width=True)
            
            # Detection comparison chart
            detection_scores = [result["drift_score"] for result in results["detection"].values()]
            methods = list(results["detection"].keys())
            
            chart_data = pd.DataFrame({
                "Method": methods,
                "Drift Score": detection_scores,
            })
            
            st.bar_chart(chart_data.set_index("Method"))
            
            # Best detector
            best_detector = max(results["detection"].items(), 
                              key=lambda x: x[1]["drift_score"])
            st.success(f"🏆 Best Detector: **{best_detector[0]}** "
                      f"(score: {best_detector[1]['drift_score']:.3f})")
        else:
            st.warning("No drift detection methods selected.")
    
    with tab3:
        st.subheader("Model Adaptation Results")
        
        if results["adaptation"]:
            # Adaptation metrics
            adaptation_df = pd.DataFrame([
                {
                    "Method": method,
                    "Final Accuracy": f"{result['final_accuracy']:.3f}",
                    "Improvement": f"{result['final_accuracy'] - baseline_accuracy:+.3f}",
                    "Status": "✅ Improved" if result['final_accuracy'] > baseline_accuracy else "❌ Degraded"
                }
                for method, result in results["adaptation"].items()
            ])
            
            st.dataframe(adaptation_df, use_container_width=True)
            
            # Adaptation comparison chart
            final_accuracies = [result["final_accuracy"] for result in results["adaptation"].values()]
            methods = list(results["adaptation"].keys())
            
            chart_data = pd.DataFrame({
                "Method": methods,
                "Final Accuracy": final_accuracies,
            })
            
            st.bar_chart(chart_data.set_index("Method"))
            
            # Best adapter
            best_adapter = max(results["adaptation"].items(),
                             key=lambda x: x[1]["final_accuracy"])
            st.success(f"🏆 Best Adapter: **{best_adapter[0]}** "
                      f"(accuracy: {best_adapter[1]['final_accuracy']:.3f})")
            
            # Accuracy over time
            st.subheader("Accuracy Over Time")
            for method, result in results["adaptation"].items():
                accuracy_history = result["accuracy_history"]
                st.line_chart(pd.DataFrame({
                    method: accuracy_history
                }))
        else:
            st.warning("No adaptation methods selected.")
    
    with tab4:
        st.subheader("Visualizations")
        
        if results["detection"] and results["adaptation"]:
            # Display plots
            if 'fig1' in locals():
                st.pyplot(fig1)
            
            if 'fig2' in locals():
                st.pyplot(fig2)
            
            # Feature distribution comparison
            st.subheader("Feature Distribution Comparison")
            X_train, y_train, X_test, y_test = results["data"]
            
            # Select feature to visualize
            feature_idx = st.selectbox(
                "Select feature to visualize",
                range(X_train.shape[1]),
                format_func=lambda x: f"Feature {x}"
            )
            
            fig3 = visualizer.plot_feature_distributions(
                X_train, X_test,
                title=f"Feature {feature_idx} Distribution Comparison"
            )
            st.pyplot(fig3)
        else:
            st.warning("No results available for visualization.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Drift Detection & Adaptation Demo | 
    <a href="https://github.com/example/drift-adaptation-xai" target="_blank">GitHub</a> | 
    <a href="DISCLAIMER.md" target="_blank">Disclaimer</a></p>
</div>
""", unsafe_allow_html=True)
