"""
Anomaly detection module for OceanData.

This module provides classes for detecting anomalies in data using various methods:
- Autoencoder
- Isolation Forest
- DBSCAN
"""

import os
import json
import traceback
from typing import List, Dict, Union, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from oceandata.core.logging import get_logger

logger = get_logger("OceanData.AnomalyDetector")


class AnomalyDetector:
    """Class for detecting anomalies in various data types."""
    
    def __init__(
        self, 
        method: str = "autoencoder", 
        contamination: float = 0.05,
        random_state: int = 42
    ):
        """
        Initialize an anomaly detector.
        
        Args:
            method: Method for anomaly detection ('autoencoder', 'isolation_forest', 'dbscan')
            contamination: Expected proportion of anomalies in the data (for isolation forest)
            random_state: Random seed for reproducibility
        """
        self.method = method.lower()
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dims = None
        self.threshold = None
        
        # Validate method
        valid_methods = ["autoencoder", "isolation_forest", "dbscan"]
        if self.method not in valid_methods:
            raise ValueError(
                f"Method '{method}' not supported. Choose from: {', '.join(valid_methods)}"
            )
        
        logger.info(f"Initialized {self.method} anomaly detector with contamination {contamination}")
    
    def _build_autoencoder(self, input_dim: int) -> Tuple[keras.Model, keras.Model]:
        """
        Build an autoencoder model for anomaly detection.
        
        Args:
            input_dim: Dimensionality of the input data
            
        Returns:
            Tuple of (autoencoder, encoder) models
        """
        encoding_dim = max(1, input_dim // 2)
        hidden_dim = max(1, encoding_dim // 2)
        
        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation="relu")(inputs)
        encoded = layers.Dense(hidden_dim, activation="relu")(encoded)
        
        # Decoder
        decoded = layers.Dense(encoding_dim, activation="relu")(encoded)
        decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(optimizer="adam", loss="mse")
        
        # Encoder model for extracting features
        encoder = keras.Model(inputs, encoded)
        
        return autoencoder, encoder
    
    def fit(
        self, 
        X: pd.DataFrame, 
        categorical_cols: Optional[List[str]] = None,
        **kwargs
    ) -> "AnomalyDetector":
        """
        Train the anomaly detector with data.
        
        Args:
            X: Input data (DataFrame)
            categorical_cols: List of categorical columns that need to be one-hot encoded
            **kwargs: Additional arguments for the underlying model
            
        Returns:
            Trained anomaly detector
        """
        try:
            # Prepare the data
            X_prep = X.copy()
            
            # Handle categorical columns with one-hot encoding
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Convert to float and handle NaN values
            X_prep = X_prep.select_dtypes(include=["number"]).fillna(0)
            
            if X_prep.shape[1] == 0:
                raise ValueError("No numeric features remain after preprocessing")
                
            X_scaled = self.scaler.fit_transform(X_prep)
            self.feature_dims = X_scaled.shape[1]
            
            if self.method == "autoencoder":
                # Autoencoder approach for anomaly detection
                self.model, self.encoder = self._build_autoencoder(self.feature_dims)
                
                # Train the autoencoder
                epochs = kwargs.get("epochs", 50)
                batch_size = kwargs.get("batch_size", 32)
                validation_split = kwargs.get("validation_split", 0.1)
                
                self.model.fit(
                    X_scaled, X_scaled,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=kwargs.get("verbose", 0),
                    validation_split=validation_split
                )
                
                # Calculate reconstruction errors for all training data
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Set threshold based on reconstruction errors
                self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
                
            elif self.method == "isolation_forest":
                # Isolation Forest for outlier detection
                n_estimators = kwargs.get("n_estimators", 100)
                
                self.model = IsolationForest(
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_estimators=n_estimators,
                    **{k: v for k, v in kwargs.items() if k not in ["n_estimators"]}
                )
                self.model.fit(X_scaled)
                
            elif self.method == "dbscan":
                # DBSCAN for clustering-based outlier detection
                eps = kwargs.get("eps", 0.5)
                min_samples = kwargs.get("min_samples", 5)
                
                self.model = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    n_jobs=-1,
                    **{k: v for k, v in kwargs.items() if k not in ["eps", "min_samples"]}
                )
                self.model.fit(X_scaled)
            
            self.is_fitted = True
            logger.info(f"Successfully trained {self.method} anomaly detector on {len(X)} samples")
            return self
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(
        self, 
        X: pd.DataFrame, 
        categorical_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Identify anomalies in the data.
        
        Args:
            X: Data to check for anomalies
            categorical_cols: List of categorical columns to one-hot encode
            
        Returns:
            Array with anomaly scores for each row.
            - For Isolation Forest: -1 for anomalies, 1 for normal data
            - For Autoencoder and DBSCAN: True for anomalies, False for normal data
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first with fit()")
            
        try:
            # Prepare the data
            X_prep = X.copy()
            
            # Handle categorical columns
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Convert and handle NaN values
            X_prep = X_prep.select_dtypes(include=["number"]).fillna(0)
            
            # Handle missing or extra columns by aligning with the trained features
            missing_cols = set(range(self.feature_dims)) - set(range(X_prep.shape[1]))
            if missing_cols:
                for col in missing_cols:
                    X_prep[f"missing_{col}"] = 0
            
            X_prep = X_prep.iloc[:, :self.feature_dims]
            X_scaled = self.scaler.transform(X_prep)
            
            if self.method == "autoencoder":
                # Calculate reconstruction error
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Anomaly if error is above threshold
                return reconstruction_errors > self.threshold
                
            elif self.method == "isolation_forest":
                # -1 for anomalies, 1 for normal data
                return self.model.predict(X_scaled)
                
            elif self.method == "dbscan":
                # Assign new points to nearest clusters
                labels = self.model.fit_predict(X_scaled)
                # -1 are noise points (anomalies)
                return labels == -1
                
        except Exception as e:
            logger.error(f"Error predicting anomalies: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_anomaly_insights(
        self, 
        X: pd.DataFrame, 
        predictions: np.ndarray,
        categorical_cols: Optional[List[str]] = None, 
        top_n_features: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Provide explanations for detected anomalies.
        
        Args:
            X: Original data
            predictions: Results from predict() method
            categorical_cols: List of categorical columns
            top_n_features: Number of most important features to return
            
        Returns:
            List of explanations for each anomaly
        """
        insights = []
        
        try:
            # Determine indices of anomalies based on method
            if self.method == "isolation_forest":
                anomaly_indices = np.where(predictions == -1)[0]
            else:
                anomaly_indices = np.where(predictions)[0]
            
            if len(anomaly_indices) == 0:
                logger.info("No anomalies detected")
                return []
            
            X_prep = X.copy()
            X_orig = X.copy()  # Original data for reports
            
            # Handle categorical columns
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include=["number"]).fillna(0)
            X_scaled = self.scaler.transform(X_prep)
            
            # Calculate general statistics
            mean_values = np.mean(X_scaled, axis=0)
            std_values = np.std(X_scaled, axis=0)
            
            for idx in anomaly_indices:
                if idx >= len(X_scaled):
                    continue
                    
                sample = X_scaled[idx]
                orig_sample = X_orig.iloc[idx]
                
                if self.method == "autoencoder":
                    # Calculate reconstruction error per feature
                    reconstruction = self.model.predict(sample.reshape(1, -1))[0]
                    feature_errors = np.square(sample - reconstruction)
                    
                    # Identify features with the largest errors
                    top_features_idx = np.argsort(feature_errors)[-top_n_features:]
                    top_features = [(X_prep.columns[i], feature_errors[i], sample[i], reconstruction[i]) 
                                    for i in top_features_idx]
                    
                    insight = {
                        "index": int(idx),
                        "anomaly_score": float(np.mean(feature_errors)),
                        "important_features": [
                            {
                                "feature": feat,
                                "error": float(err),
                                "actual_value": float(val),
                                "expected_value": float(rec)
                            } for feat, err, val, rec in top_features
                        ],
                        "original_data": orig_sample.to_dict()
                    }
                    
                else:  # Isolation Forest or DBSCAN
                    # Calculate Z-scores to identify unusual values
                    z_scores = (sample - mean_values) / (std_values + 1e-10)  # Add small epsilon to avoid division by zero
                    
                    # Features with the highest Z-scores (positive or negative)
                    abs_z_scores = np.abs(z_scores)
                    top_features_idx = np.argsort(abs_z_scores)[-top_n_features:]
                    top_features = [(X_prep.columns[i], z_scores[i], sample[i], mean_values[i]) 
                                    for i in top_features_idx]
                    
                    insight = {
                        "index": int(idx),
                        "anomaly_score": float(np.max(abs_z_scores)),
                        "important_features": [
                            {
                                "feature": feat,
                                "z_score": float(z),
                                "actual_value": float(val),
                                "average_value": float(avg)
                            } for feat, z, val, avg in top_features
                        ],
                        "original_data": orig_sample.to_dict()
                    }
                
                insights.append(insight)
            
            logger.info(f"Generated insights for {len(insights)} anomalies")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating anomaly insights: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def visualize_anomalies(
        self, 
        X: pd.DataFrame, 
        predictions: np.ndarray,
        categorical_cols: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Visualize detected anomalies in the data.
        
        Args:
            X: Data
            predictions: Anomaly detection results
            categorical_cols: Categorical columns
            save_path: Path to save the visualization
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure with the visualization
        """
        try:
            X_prep = X.copy()
            
            # Handle categorical columns
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include
