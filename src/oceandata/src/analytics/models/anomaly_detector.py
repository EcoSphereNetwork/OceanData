###########################################
# 1. Anomalieerkennung & Outlier Detection
###########################################

import pandas as pd
import numpy as np
import json
import logging
import os
import matplotlib.pyplot as plt
import traceback
from typing import Dict, List, Any, Union, Tuple
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Konfiguriere Logger
logger = logging.getLogger("OceanData.Analytics.AnomalyDetector")

class AnomalyDetector:
    """Klasse für die Erkennung von Anomalien in verschiedenen Datentypen mit Ensemble-Methoden"""
    
    SUPPORTED_METHODS = [
        'autoencoder',
        'isolation_forest',
        'dbscan',
        'lof',  # Local Outlier Factor
        'ocsvm',  # One-Class SVM
        'ensemble'  # Kombiniert mehrere Methoden
    ]
    
    def __init__(self, method: str = 'ensemble', contamination: float = 0.05, ensemble_config: Dict = None):
        """
        Initialisiert einen Anomaliedetektor.
        
        Args:
            method: Methode für die Anomalieerkennung ('autoencoder', 'isolation_forest', 'dbscan', 'lof', 'ocsvm', 'ensemble')
            contamination: Erwarteter Anteil an Anomalien in den Daten
            ensemble_config: Konfiguration für Ensemble-Methode (optional)
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Nicht unterstützte Methode: {method}. "
                           f"Unterstützte Methoden: {', '.join(self.SUPPORTED_METHODS)}")
        
        self.method = method
        self.contamination = contamination
        self.models = {}  # Für Ensemble-Methode: Methode -> Modell
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dims = None
        self.threshold = None
        
        # Konfiguration für Ensemble-Methode
        self.ensemble_config = ensemble_config or {
            'methods': ['isolation_forest', 'lof', 'autoencoder'] if method == 'ensemble' else [method],
            'voting': 'majority',  # 'majority' oder 'weighted'
            'weights': {  # Gewichte für gewichtete Abstimmung
                'isolation_forest': 1.0,
                'lof': 1.0,
                'autoencoder': 1.0,
                'dbscan': 0.8,
                'ocsvm': 0.9
            }
        }
        
        # Metadaten für Modellversionierung
        self.metadata = {
            'method': method,
            'contamination': contamination,
            'version': '1.0',
            'created_at': None,
            'updated_at': None,
            'training_samples': 0,
            'feature_count': 0,
            'performance_metrics': {}
        }
        
    def _build_autoencoder(self, input_dim):
        """
        Erstellt ein verbessertes Autoencoder-Modell für die Anomalieerkennung.
        
        Args:
            input_dim: Dimension der Eingabedaten
            
        Returns:
            Tuple: (Autoencoder-Modell, Encoder-Modell)
        """
        # Berechne Dimensionen für die versteckten Schichten
        encoding_dim = max(1, input_dim // 2)
        hidden_dim_1 = max(1, encoding_dim // 1.5)
        hidden_dim_2 = max(1, encoding_dim // 2)
        
        # Encoder mit Dropout für bessere Generalisierung
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(encoding_dim, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(hidden_dim_1, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        encoded = layers.Dense(hidden_dim_2, activation='relu')(x)
        
        # Decoder mit symmetrischer Struktur
        x = layers.Dense(hidden_dim_1, activation='relu')(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(encoding_dim, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        decoded = layers.Dense(input_dim, activation='sigmoid')(x)
        
        # Autoencoder-Modell
        autoencoder = keras.Model(inputs, decoded)
        
        # Verwende Adam-Optimizer mit angepasster Lernrate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        # Kompiliere mit Mean Squared Error Loss
        autoencoder.compile(optimizer=optimizer, loss='mse')
        
        # Encoder-Modell für Feature-Extraktion
        encoder = keras.Model(inputs, encoded)
        
        return autoencoder, encoder
        
    def _build_ensemble_models(self, input_dim: int):
        """
        Erstellt alle Modelle für den Ensemble-Ansatz.
        
        Args:
            input_dim: Dimension der Eingabedaten
            
        Returns:
            Dict: Dictionary mit allen erstellten Modellen
        """
        models = {}
        methods = self.ensemble_config['methods']
        
        for method in methods:
            if method == 'autoencoder':
                autoencoder, encoder = self._build_autoencoder(input_dim)
                models[method] = {
                    'model': autoencoder,
                    'encoder': encoder,
                    'threshold': None
                }
            elif method == 'isolation_forest':
                models[method] = {
                    'model': IsolationForest(
                        contamination=self.contamination,
                        random_state=42,
                        n_estimators=100,
                        max_samples='auto',
                        bootstrap=True
                    )
                }
            elif method == 'dbscan':
                # DBSCAN-Parameter werden während des Trainings optimiert
                models[method] = {
                    'model': None,  # Wird während des Trainings erstellt
                    'eps': None,    # Wird während des Trainings optimiert
                    'min_samples': None  # Wird während des Trainings optimiert
                }
            elif method == 'lof':
                models[method] = {
                    'model': LocalOutlierFactor(
                        n_neighbors=20,
                        contamination=self.contamination,
                        novelty=True,  # Für predict-Methode
                        n_jobs=-1
                    )
                }
            elif method == 'ocsvm':
                models[method] = {
                    'model': OneClassSVM(
                        nu=self.contamination,
                        kernel='rbf',
                        gamma='scale'
                    )
                }
                
        return models
    
    def fit(self, X: pd.DataFrame, categorical_cols: List[str] = None, validation_split: float = 0.2):
        """
        Trainiert den Anomaliedetektor mit Daten.

        Args:
            X: Eingabedaten (DataFrame oder numpy array)
            categorical_cols: Liste der kategorischen Spalten, die One-Hot-kodiert werden müssen
            validation_split: Anteil der Daten für die Validierung (nur für Autoencoder)
            
        Returns:
            self: Trainierter Anomaliedetektor
        """
        try:
            # Aktualisiere Metadaten
            from datetime import datetime
            self.metadata['created_at'] = datetime.now().isoformat()
            self.metadata['updated_at'] = datetime.now().isoformat()
            
            # Vorbereiten der Daten
            X_prep = X.copy()

            # Behandlung kategorischer Spalten durch One-Hot-Encoding
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)

            # Konvertieren zu float und Behandlung von NaN-Werten
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)

            if X_prep.shape[1] == 0:
                raise ValueError("Nach der Vorverarbeitung sind keine numerischen Merkmale übrig")

            # Skaliere die Daten
            X_scaled = self.scaler.fit_transform(X_prep)
            self.feature_dims = X_scaled.shape[1]
            
            # Aktualisiere Metadaten
            self.metadata['training_samples'] = X_scaled.shape[0]
            self.metadata['feature_count'] = self.feature_dims

            # Trainiere je nach Methode
            if self.method == 'ensemble':
                # Ensemble-Ansatz: Trainiere alle konfigurierten Modelle
                self._fit_ensemble(X_scaled, validation_split)
            else:
                # Einzelnes Modell trainieren
                self._fit_single_model(X_scaled, validation_split)

            self.is_fitted = True
            
            # Berechne Performance-Metriken
            self._calculate_performance_metrics(X_scaled)
            
            return self

        except Exception as e:
            logger.error(f"Fehler beim Training des Anomaliedetektors: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _fit_single_model(self, X_scaled: np.ndarray, validation_split: float = 0.2):
        """
        Trainiert ein einzelnes Modell.
        
        Args:
            X_scaled: Skalierte Eingabedaten
            validation_split: Anteil der Daten für die Validierung (nur für Autoencoder)
        """
        if self.method == 'autoencoder':
            # Autoencoder-Ansatz für Anomalieerkennung
            autoencoder, encoder = self._build_autoencoder(self.feature_dims)
            
            # Trainiere den Autoencoder mit Early Stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Trainiere den Autoencoder
            history = autoencoder.fit(
                X_scaled, X_scaled,
                epochs=100,  # Mehr Epochen, aber mit Early Stopping
                batch_size=32,
                shuffle=True,
                verbose=0,
                validation_split=validation_split,
                callbacks=[early_stopping]
            )
            
            # Speichere Trainingshistorie in Metadaten
            self.metadata['training_history'] = {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']] if 'val_loss' in history.history else []
            }

            # Berechne Rekonstruktionsfehler für alle Trainingsdaten
            reconstructions = autoencoder.predict(X_scaled)
            reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)

            # Setze Threshold basierend auf den Rekonstruktionsfehlern
            threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
            
            # Speichere Modell und Parameter
            self.models = {
                'autoencoder': {
                    'model': autoencoder,
                    'encoder': encoder,
                    'threshold': threshold,
                    'reconstruction_errors': reconstruction_errors
                }
            }
            
            # Für Abwärtskompatibilität
            self.model = autoencoder
            self.encoder = encoder
            self.threshold = threshold

        elif self.method == 'isolation_forest':
            # Isolation Forest für Outlier-Erkennung
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                bootstrap=True
            )
            model.fit(X_scaled)
            
            # Speichere Modell
            self.models = {
                'isolation_forest': {
                    'model': model
                }
            }
            
            # Für Abwärtskompatibilität
            self.model = model

        elif self.method == 'dbscan':
            # Optimiere DBSCAN-Parameter
            eps, min_samples = self._optimize_dbscan_params(X_scaled)
            
            # DBSCAN für Clustering-basierte Outlier-Erkennung
            model = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                n_jobs=-1
            )
            model.fit(X_scaled)
            
            # Speichere Modell und Parameter
            self.models = {
                'dbscan': {
                    'model': model,
                    'eps': eps,
                    'min_samples': min_samples
                }
            }
            
            # Für Abwärtskompatibilität
            self.model = model
            
        elif self.method == 'lof':
            # Local Outlier Factor für Outlier-Erkennung
            model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1
            )
            model.fit(X_scaled)
            
            # Speichere Modell
            self.models = {
                'lof': {
                    'model': model
                }
            }
            
            # Für Abwärtskompatibilität
            self.model = model
            
        elif self.method == 'ocsvm':
            # One-Class SVM für Outlier-Erkennung
            model = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
            model.fit(X_scaled)
            
            # Speichere Modell
            self.models = {
                'ocsvm': {
                    'model': model
                }
            }
            
            # Für Abwärtskompatibilität
            self.model = model
    
    def _fit_ensemble(self, X_scaled: np.ndarray, validation_split: float = 0.2):
        """
        Trainiert alle Modelle für den Ensemble-Ansatz.
        
        Args:
            X_scaled: Skalierte Eingabedaten
            validation_split: Anteil der Daten für die Validierung (nur für Autoencoder)
        """
        # Erstelle alle Modelle
        self.models = self._build_ensemble_models(self.feature_dims)
        
        # Trainiere jedes Modell
        for method, model_info in self.models.items():
            logger.info(f"Trainiere {method}-Modell für Ensemble")
            
            if method == 'autoencoder':
                # Trainiere Autoencoder
                autoencoder = model_info['model']
                
                # Early Stopping für bessere Generalisierung
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                # Trainiere den Autoencoder
                history = autoencoder.fit(
                    X_scaled, X_scaled,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    verbose=0,
                    validation_split=validation_split,
                    callbacks=[early_stopping]
                )
                
                # Berechne Rekonstruktionsfehler und Threshold
                reconstructions = autoencoder.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
                
                # Aktualisiere Modellinfo
                model_info['threshold'] = threshold
                model_info['reconstruction_errors'] = reconstruction_errors
                model_info['history'] = {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']] if 'val_loss' in history.history else []
                }
                
            elif method == 'isolation_forest':
                # Trainiere Isolation Forest
                model_info['model'].fit(X_scaled)
                
            elif method == 'dbscan':
                # Optimiere DBSCAN-Parameter
                eps, min_samples = self._optimize_dbscan_params(X_scaled)
                
                # Erstelle und trainiere DBSCAN-Modell
                model = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    n_jobs=-1
                )
                model.fit(X_scaled)
                
                # Aktualisiere Modellinfo
                model_info['model'] = model
                model_info['eps'] = eps
                model_info['min_samples'] = min_samples
                
            elif method == 'lof':
                # Trainiere Local Outlier Factor
                model_info['model'].fit(X_scaled)
                
            elif method == 'ocsvm':
                # Trainiere One-Class SVM
                model_info['model'].fit(X_scaled)
    
    def _optimize_dbscan_params(self, X_scaled: np.ndarray) -> Tuple[float, int]:
        """
        Optimiert die Parameter für DBSCAN.
        
        Args:
            X_scaled: Skalierte Eingabedaten
            
        Returns:
            Tuple: (eps, min_samples)
        """
        try:
            # Berechne Distanzmatrix für k-distance Graph
            from sklearn.neighbors import NearestNeighbors
            
            # Verwende einen Subset der Daten, wenn zu viele Datenpunkte vorhanden sind
            max_samples = 1000
            if X_scaled.shape[0] > max_samples:
                indices = np.random.choice(X_scaled.shape[0], max_samples, replace=False)
                X_subset = X_scaled[indices]
            else:
                X_subset = X_scaled
            
            # Berechne k-distance Graph
            k = min(20, X_subset.shape[0] - 1)
            nbrs = NearestNeighbors(n_neighbors=k).fit(X_subset)
            distances, _ = nbrs.kneighbors(X_subset)
            
            # Sortiere Distanzen für den "Elbow"-Punkt
            distances = np.sort(distances[:, -1])
            
            # Finde den "Elbow"-Punkt
            from scipy.signal import argrelextrema
            from scipy.ndimage import gaussian_filter1d
            
            # Glätte die Kurve
            smoothed = gaussian_filter1d(distances, sigma=3)
            
            # Finde lokale Maxima der zweiten Ableitung (Krümmung)
            second_derivative = np.gradient(np.gradient(smoothed))
            local_maxima = argrelextrema(second_derivative, np.greater)[0]
            
            if len(local_maxima) > 0:
                # Wähle den ersten signifikanten "Elbow"-Punkt
                elbow_index = local_maxima[0]
                eps = distances[elbow_index]
            else:
                # Fallback: Verwende Heuristik
                eps = np.percentile(distances, 90)
            
            # Bestimme min_samples basierend auf der Datengröße und Contamination
            min_samples = max(5, int(X_scaled.shape[0] * self.contamination * 0.5))
            
            return eps, min_samples
        except Exception as e:
            logger.error(f"Fehler bei der Optimierung der DBSCAN-Parameter: {str(e)}")
            # Fallback zu Standardwerten
            return 0.5, 5
        
    def _calculate_performance_metrics(self, X_scaled: np.ndarray):
        """
        Berechnet Performance-Metriken für das trainierte Modell.
        
        Args:
            X_scaled: Skalierte Eingabedaten
        """
        try:
            # Vorhersagen für Trainingsdaten
            predictions = self.predict_raw(X_scaled)
            
            # Berechne Anomalie-Rate
            if self.method == 'isolation_forest' or self.method == 'ocsvm':
                anomaly_rate = np.mean(predictions == -1)
            else:
                anomaly_rate = np.mean(predictions)
                
            # Speichere Metriken
            self.metadata['performance_metrics'] = {
                'anomaly_rate': float(anomaly_rate),
                'expected_anomaly_rate': float(self.contamination),
                'training_samples': int(X_scaled.shape[0]),
                'anomaly_count': int(np.sum(predictions == -1) if self.method in ['isolation_forest', 'ocsvm'] else np.sum(predictions))
            }
            
            # Für Ensemble: Berechne Übereinstimmung zwischen Modellen
            if self.method == 'ensemble':
                agreement_metrics = self._calculate_ensemble_agreement(X_scaled)
                self.metadata['performance_metrics']['ensemble_agreement'] = agreement_metrics
                
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Performance-Metriken: {str(e)}")
            
    def _calculate_ensemble_agreement(self, X_scaled: np.ndarray) -> Dict[str, float]:
        """
        Berechnet die Übereinstimmung zwischen den Modellen im Ensemble.
        
        Args:
            X_scaled: Skalierte Eingabedaten
            
        Returns:
            Dict: Übereinstimmungsmetriken
        """
        try:
            # Sammle Vorhersagen aller Modelle
            predictions = {}
            
            for method, model_info in self.models.items():
                if method == 'autoencoder':
                    # Berechne Rekonstruktionsfehler
                    reconstructions = model_info['model'].predict(X_scaled)
                    reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                    # Anomalie, wenn Fehler über dem Threshold
                    preds = reconstruction_errors > model_info['threshold']
                    predictions[method] = preds
                    
                elif method == 'isolation_forest' or method == 'ocsvm':
                    # -1 für Anomalien, 1 für normale Daten
                    preds = model_info['model'].predict(X_scaled)
                    predictions[method] = preds == -1  # Konvertiere zu Boolean
                    
                elif method == 'dbscan':
                    # Ordne neue Punkte den nächsten Clustern zu
                    preds = model_info['model'].fit_predict(X_scaled)
                    # -1 sind Noise-Punkte (Anomalien)
                    predictions[method] = preds == -1
                    
                elif method == 'lof':
                    # -1 für Anomalien, 1 für normale Daten
                    preds = model_info['model'].predict(X_scaled)
                    predictions[method] = preds == -1  # Konvertiere zu Boolean
            
            # Berechne paarweise Übereinstimmung
            methods = list(predictions.keys())
            agreement = {}
            
            for i in range(len(methods)):
                for j in range(i+1, len(methods)):
                    method1 = methods[i]
                    method2 = methods[j]
                    
                    # Berechne Übereinstimmung (Jaccard-Index)
                    pred1 = predictions[method1]
                    pred2 = predictions[method2]
                    
                    intersection = np.sum(np.logical_and(pred1, pred2))
                    union = np.sum(np.logical_or(pred1, pred2))
                    
                    if union > 0:
                        jaccard = intersection / union
                    else:
                        jaccard = 1.0  # Beide leere Mengen
                    
                    agreement[f"{method1}_vs_{method2}"] = float(jaccard)
            
            # Berechne durchschnittliche Übereinstimmung
            if agreement:
                avg_agreement = sum(agreement.values()) / len(agreement)
                agreement["average"] = float(avg_agreement)
            
            return agreement
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Ensemble-Übereinstimmung: {str(e)}")
            return {"error": str(e)}
    
    def predict(self, X: pd.DataFrame, categorical_cols: List[str] = None) -> np.ndarray:
        """
        Identifiziert Anomalien in den Daten.
        
        Args:
            X: Zu prüfende Daten
            categorical_cols: Liste der kategorischen Spalten
            
        Returns:
            Array mit Anomalie-Scores für jede Zeile:
            - Für Isolation Forest/OCSVM: -1 für Anomalien, 1 für normale Daten
            - Für Autoencoder/DBSCAN/LOF/Ensemble: True für Anomalien, False für normale Daten
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Daten
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Konvertieren und Behandlung von NaN-Werten
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            
            # Anpassen der Features, falls nötig
            missing_cols = set(range(self.feature_dims)) - set(range(X_prep.shape[1]))
            if missing_cols:
                for col in missing_cols:
                    X_prep[f'missing_{col}'] = 0
            
            X_prep = X_prep.iloc[:, :self.feature_dims]
            X_scaled = self.scaler.transform(X_prep)
            
            # Vorhersage je nach Methode
            if self.method == 'ensemble':
                # Ensemble-Vorhersage: Kombiniere Vorhersagen aller Modelle
                return self._predict_ensemble(X_scaled)
            else:
                # Einzelnes Modell
                return self._predict_single_model(X_scaled)
            
        except Exception as e:
            logger.error(f"Fehler bei der Anomalievorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def predict_raw(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Identifiziert Anomalien in bereits vorverarbeiteten Daten.
        
        Args:
            X_scaled: Bereits skalierte Daten
            
        Returns:
            Array mit Anomalie-Scores
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorhersage je nach Methode
            if self.method == 'ensemble':
                # Ensemble-Vorhersage: Kombiniere Vorhersagen aller Modelle
                return self._predict_ensemble(X_scaled)
            else:
                # Einzelnes Modell
                return self._predict_single_model(X_scaled)
                
        except Exception as e:
            logger.error(f"Fehler bei der Anomalievorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _predict_single_model(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Identifiziert Anomalien mit einem einzelnen Modell.
        
        Args:
            X_scaled: Skalierte Eingabedaten
            
        Returns:
            Array mit Anomalie-Scores
        """
        if self.method == 'autoencoder':
            # Für Abwärtskompatibilität
            if hasattr(self, 'model') and self.model is not None and hasattr(self, 'threshold'):
                # Berechne Rekonstruktionsfehler
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Identifiziere Anomalien basierend auf dem Threshold
                return reconstruction_errors > self.threshold
            else:
                # Verwende das Modell aus dem models-Dictionary
                model_info = self.models['autoencoder']
                reconstructions = model_info['model'].predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                return reconstruction_errors > model_info['threshold']
                
        elif self.method == 'isolation_forest' or self.method == 'ocsvm':
            # Für Abwärtskompatibilität
            if hasattr(self, 'model') and self.model is not None:
                # Isolation Forest/OCSVM gibt -1 für Anomalien und 1 für normale Daten zurück
                return self.model.predict(X_scaled)
            else:
                # Verwende das Modell aus dem models-Dictionary
                model_info = self.models[self.method]
                return model_info['model'].predict(X_scaled)
                
        elif self.method == 'dbscan':
            # Für Abwärtskompatibilität
            if hasattr(self, 'model') and self.model is not None:
                # DBSCAN gibt Cluster-Labels zurück, -1 bedeutet Noise (Anomalie)
                labels = self.model.fit_predict(X_scaled)
                return labels == -1
            else:
                # Verwende das Modell aus dem models-Dictionary
                model_info = self.models['dbscan']
                labels = model_info['model'].fit_predict(X_scaled)
                return labels == -1
                
        elif self.method == 'lof':
            # Verwende das Modell aus dem models-Dictionary
            model_info = self.models['lof']
            # LOF gibt -1 für Anomalien und 1 für normale Daten zurück
            return model_info['model'].predict(X_scaled) == -1
            
    def _predict_ensemble(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Identifiziert Anomalien mit dem Ensemble-Ansatz.
        
        Args:
            X_scaled: Skalierte Eingabedaten
            
        Returns:
            Array mit Anomalie-Scores (True für Anomalien, False für normale Daten)
        """
        # Sammle Vorhersagen aller Modelle
        predictions = {}
        weights = self.ensemble_config['weights']
        
        for method, model_info in self.models.items():
            if method == 'autoencoder':
                # Berechne Rekonstruktionsfehler
                reconstructions = model_info['model'].predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                # Anomalie, wenn Fehler über dem Threshold
                preds = reconstruction_errors > model_info['threshold']
                predictions[method] = preds
                
            elif method == 'isolation_forest' or method == 'ocsvm':
                # -1 für Anomalien, 1 für normale Daten
                preds = model_info['model'].predict(X_scaled)
                predictions[method] = preds == -1  # Konvertiere zu Boolean
                
            elif method == 'dbscan':
                # Ordne neue Punkte den nächsten Clustern zu
                preds = model_info['model'].fit_predict(X_scaled)
                # -1 sind Noise-Punkte (Anomalien)
                predictions[method] = preds == -1
                
            elif method == 'lof':
                # -1 für Anomalien, 1 für normale Daten
                preds = model_info['model'].predict(X_scaled)
                predictions[method] = preds == -1  # Konvertiere zu Boolean
        
        # Kombiniere Vorhersagen je nach Voting-Methode
        voting_method = self.ensemble_config.get('voting', 'majority')
        
        if voting_method == 'majority':
            # Mehrheitsentscheidung: Anomalie, wenn die Mehrheit der Modelle sie als solche erkennt
            ensemble_predictions = np.zeros(X_scaled.shape[0], dtype=bool)
            
            for i in range(X_scaled.shape[0]):
                votes = 0
                for method, preds in predictions.items():
                    if preds[i]:
                        votes += 1
                
                # Anomalie, wenn mindestens die Hälfte der Modelle sie als solche erkennt
                ensemble_predictions[i] = votes >= len(predictions) / 2
                
            return ensemble_predictions
            
        elif voting_method == 'weighted':
            # Gewichtete Abstimmung: Anomalie, wenn die gewichtete Summe einen Schwellenwert überschreitet
            ensemble_predictions = np.zeros(X_scaled.shape[0], dtype=bool)
            
            for i in range(X_scaled.shape[0]):
                weighted_sum = 0
                total_weight = 0
                
                for method, preds in predictions.items():
                    weight = weights.get(method, 1.0)
                    if preds[i]:
                        weighted_sum += weight
                    total_weight += weight
                
                # Anomalie, wenn die gewichtete Summe mindestens die Hälfte des Gesamtgewichts beträgt
                ensemble_predictions[i] = weighted_sum >= total_weight / 2
                
            return ensemble_predictions
            
        else:
            # Fallback: Verwende das erste Modell
            first_method = list(predictions.keys())[0]
            return predictions[first_method]
    
    def get_anomaly_insights(self, X: pd.DataFrame, predictions: np.ndarray, 
                           categorical_cols: List[str] = None, top_n_features: int = 3) -> List[Dict]:
        """
        Liefert Erklärungen für die erkannten Anomalien.
        
        Args:
            X: Originaldaten
            predictions: Ergebnisse der predict()-Methode
            categorical_cols: Liste der kategorischen Spalten
            top_n_features: Anzahl der wichtigsten Features, die zurückgegeben werden sollen
            
        Returns:
            Liste mit Erklärungen für jede Anomalie
        """
        insights = []
        
        try:
            anomaly_indices = np.where(predictions == -1)[0] if self.method == 'isolation_forest' else np.where(predictions)[0]
            
            if len(anomaly_indices) == 0:
                return []
            
            X_prep = X.copy()
            X_orig = X.copy()  # Original-Daten für Berichte
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            X_scaled = self.scaler.transform(X_prep)
            
            # Allgemeine Statistiken berechnen
            mean_values = np.mean(X_scaled, axis=0)
            std_values = np.std(X_scaled, axis=0)
            
            for idx in anomaly_indices:
                if idx >= len(X_scaled):
                    continue
                    
                sample = X_scaled[idx]
                orig_sample = X_orig.iloc[idx]
                
                if self.method == 'autoencoder':
                    # Rekonstruktionsfehler pro Feature berechnen
                    reconstruction = self.model.predict(sample.reshape(1, -1))[0]
                    feature_errors = np.square(sample - reconstruction)
                    
                    # Features mit den größten Fehlern identifizieren
                    top_features_idx = np.argsort(feature_errors)[-top_n_features:]
                    top_features = [(X_prep.columns[i], feature_errors[i], sample[i], reconstruction[i]) for i in top_features_idx]
                    
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
                    
                else:  # Isolation Forest oder DBSCAN
                    # Z-Scores berechnen, um ungewöhnliche Werte zu identifizieren
                    z_scores = (sample - mean_values) / std_values
                    
                    # Features mit den höchsten Z-Scores (positiv oder negativ)
                    abs_z_scores = np.abs(z_scores)
                    top_features_idx = np.argsort(abs_z_scores)[-top_n_features:]
                    top_features = [(X_prep.columns[i], z_scores[i], sample[i], mean_values[i]) for i in top_features_idx]
                    
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
                
            return insights
            
        except Exception as e:
            logger.error(f"Fehler bei der Generierung von Anomalie-Insights: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def visualize_anomalies(self, X: pd.DataFrame, predictions: np.ndarray, 
                            categorical_cols: List[str] = None, 
                            save_path: str = None) -> plt.Figure:
        """
        Visualisiert erkannte Anomalien in den Daten.
        
        Args:
            X: Daten
            predictions: Anomalieerkennungsergebnisse
            categorical_cols: Kategorische Spalten
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        try:
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            X_scaled = self.scaler.transform(X_prep)
            
            # PCA für Dimensionsreduktion
            if X_scaled.shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
            else:
                X_pca = X_scaled
            
            # Anomalien bestimmen
            if self.method == 'isolation_forest':
                anomalies = (predictions == -1)
            else:
                anomalies = predictions
            
            # Visualisierung erstellen
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Normale Datenpunkte plotten
            ax.scatter(X_pca[~anomalies, 0], X_pca[~anomalies, 1], c='blue', label='Normal', alpha=0.5)
            
            # Anomalien plotten
            if np.any(anomalies):
                ax.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], c='red', label='Anomalien', alpha=0.7)
            
            # Grafik anpassen
            ax.set_title(f'Anomalieerkennung mit {self.method.capitalize()}')
            ax.set_xlabel('Hauptkomponente 1')
            ax.set_ylabel('Hauptkomponente 2')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung von Anomalien: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """
        Speichert das trainierte Modell.
        
        Args:
            path: Pfad zum Speichern des Modells
            
        Returns:
            None
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden")
            
        try:
            # Aktualisiere Metadaten
            from datetime import datetime
            if not hasattr(self, 'metadata'):
                self.metadata = {
                    'method': self.method,
                    'contamination': self.contamination,
                    'version': '1.0',
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'training_samples': 0,
                    'feature_count': self.feature_dims if hasattr(self, 'feature_dims') else 0,
                    'performance_metrics': {}
                }
            else:
                self.metadata['updated_at'] = datetime.now().isoformat()
            
            # Erstelle ein Dictionary mit allen zu speichernden Daten
            model_data = {
                'method': self.method,
                'contamination': self.contamination,
                'feature_dims': self.feature_dims,
                'is_fitted': self.is_fitted,
                'metadata': self.metadata
            }
            
            # Speichere Threshold für Abwärtskompatibilität
            if hasattr(self, 'threshold'):
                model_data['threshold'] = self.threshold
                
            # Speichere Ensemble-Konfiguration, falls vorhanden
            if hasattr(self, 'ensemble_config'):
                model_data['ensemble_config'] = self.ensemble_config
            
            # Erstelle Verzeichnis, falls es nicht existiert
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            
            # Speichere Modelle je nach Methode
            if self.method == 'ensemble':
                # Für Ensemble: Speichere jedes Modell separat
                model_data['ensemble_models'] = {}
                
                for method_name, model_info in self.models.items():
                    if method_name == 'autoencoder':
                        # Für Autoencoder: Speichere Modell in separater Datei
                        autoencoder_path = f"{path}_{method_name}_model"
                        model_info['model'].save(autoencoder_path)
                        
                        # Speichere Encoder, falls vorhanden
                        if 'encoder' in model_info:
                            encoder_path = f"{path}_{method_name}_encoder"
                            model_info['encoder'].save(encoder_path)
                            model_data['ensemble_models'][method_name] = {
                                'model_path': autoencoder_path,
                                'encoder_path': encoder_path,
                                'threshold': model_info['threshold']
                            }
                        else:
                            model_data['ensemble_models'][method_name] = {
                                'model_path': autoencoder_path,
                                'threshold': model_info['threshold']
                            }
                    else:
                        # Für andere Modelle: Speichere in separater Datei
                        model_path = f"{path}_{method_name}_model.joblib"
                        joblib.dump(model_info['model'], model_path)
                        
                        model_data['ensemble_models'][method_name] = {
                            'model_path': model_path
                        }
                        
                        # Speichere zusätzliche Parameter
                        if 'eps' in model_info:
                            model_data['ensemble_models'][method_name]['eps'] = model_info['eps']
                        if 'min_samples' in model_info:
                            model_data['ensemble_models'][method_name]['min_samples'] = model_info['min_samples']
            else:
                # Für einzelnes Modell
                if self.method == 'autoencoder':
                    # Speichere Autoencoder und Encoder
                    self.model.save(f"{path}_autoencoder")
                    if hasattr(self, 'encoder') and self.encoder is not None:
                        self.encoder.save(f"{path}_encoder")
                        model_data['has_encoder'] = True
                else:
                    # Speichere andere Modelle
                    joblib.dump(self.model, f"{path}_model.joblib")
            
            # Speichere Scaler
            joblib.dump(self.scaler, f"{path}_scaler.joblib")
            
            # Speichere Metadaten
            with open(f"{path}_metadata.json", 'w') as f:
                # Konvertiere komplexe Objekte zu JSON-serialisierbaren Typen
                json_data = {}
                for key, value in model_data.items():
                    if key not in ['ensemble_models']:  # Diese enthalten komplexe Objekte
                        if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                            json_data[key] = value
                        else:
                            json_data[key] = str(value)
                
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Modell erfolgreich gespeichert unter {path}")
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    @classmethod
    def load(cls, path: str):
        """
        Lädt ein trainiertes Modell aus einer Datei.
        
        Args:
            path: Pfad zur Modelldatei
            
        Returns:
            AnomalyDetector: Geladenes Modell
        """
        try:
            # Lade Metadaten
            with open(f"{path}_metadata.json", 'r') as f:
                model_data = json.load(f)
            
            # Erstelle neue Instanz
            detector = cls(
                method=model_data['method'],
                contamination=model_data['contamination'],
                ensemble_config=model_data.get('ensemble_config')
            )
            
            # Lade Attribute
            detector.feature_dims = model_data['feature_dims']
            detector.is_fitted = model_data['is_fitted']
            
            # Lade Threshold für Abwärtskompatibilität
            if 'threshold' in model_data:
                detector.threshold = model_data['threshold']
                
            # Lade Metadaten, falls vorhanden
            if 'metadata' in model_data:
                detector.metadata = model_data['metadata']
            else:
                # Erstelle Metadaten für ältere Modelle
                from datetime import datetime
                detector.metadata = {
                    'method': model_data['method'],
                    'contamination': model_data['contamination'],
                    'version': '1.0',
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'training_samples': 0,
                    'feature_count': model_data['feature_dims'],
                    'performance_metrics': {}
                }
            
            # Lade Scaler
            detector.scaler = joblib.load(f"{path}_scaler.joblib")
            
            # Lade Modelle je nach Methode
            if model_data['method'] == 'ensemble':
                # Für Ensemble: Lade jedes Modell
                detector.models = {}
                
                # Prüfe, ob ensemble_models in den Metadaten vorhanden ist
                if 'ensemble_models' in model_data:
                    # Neue Version mit ensemble_models in Metadaten
                    for method_name, model_info in model_data['ensemble_models'].items():
                        if method_name == 'autoencoder':
                            # Lade Autoencoder
                            model = keras.models.load_model(model_info['model_path'])
                            
                            # Lade Encoder, falls vorhanden
                            encoder = None
                            if 'encoder_path' in model_info:
                                encoder = keras.models.load_model(model_info['encoder_path'])
                            
                            detector.models[method_name] = {
                                'model': model,
                                'encoder': encoder,
                                'threshold': model_info['threshold']
                            }
                        else:
                            # Lade andere Modelle
                            model = joblib.load(model_info['model_path'])
                            
                            detector.models[method_name] = {
                                'model': model
                            }
                            
                            # Lade zusätzliche Parameter
                            if 'eps' in model_info:
                                detector.models[method_name]['eps'] = model_info['eps']
                            if 'min_samples' in model_info:
                                detector.models[method_name]['min_samples'] = model_info['min_samples']
                else:
                    # Alte Version ohne ensemble_models in Metadaten
                    # Hier müssten wir die Modelle aus separaten Dateien laden
                    logger.warning("Alte Modellversion ohne ensemble_models in Metadaten")
                    
                    # Versuche, die Modelle aus den Standard-Dateipfaden zu laden
                    for method_name in ['isolation_forest', 'dbscan', 'autoencoder', 'lof', 'ocsvm']:
                        try:
                            if method_name == 'autoencoder':
                                # Versuche, Autoencoder zu laden
                                model_path = f"{path}_{method_name}_model"
                                if os.path.exists(model_path):
                                    model = keras.models.load_model(model_path)
                                    
                                    # Versuche, Encoder zu laden
                                    encoder_path = f"{path}_{method_name}_encoder"
                                    encoder = None
                                    if os.path.exists(encoder_path):
                                        encoder = keras.models.load_model(encoder_path)
                                    
                                    # Verwende Standard-Threshold, falls nicht vorhanden
                                    threshold = 0.1
                                    
                                    detector.models[method_name] = {
                                        'model': model,
                                        'encoder': encoder,
                                        'threshold': threshold
                                    }
                            else:
                                # Versuche, andere Modelle zu laden
                                model_path = f"{path}_{method_name}_model.joblib"
                                if os.path.exists(model_path):
                                    model = joblib.load(model_path)
                                    
                                    detector.models[method_name] = {
                                        'model': model
                                    }
                        except Exception as e:
                            logger.warning(f"Konnte Modell {method_name} nicht laden: {str(e)}")
            else:
                # Für einzelnes Modell
                if model_data['method'] == 'autoencoder':
                    # Lade Autoencoder
                    detector.model = keras.models.load_model(f"{path}_autoencoder")
                    
                    # Lade Encoder, falls vorhanden
                    if model_data.get('has_encoder', True):  # Standardmäßig True für Abwärtskompatibilität
                        try:
                            detector.encoder = keras.models.load_model(f"{path}_encoder")
                        except Exception as e:
                            logger.warning(f"Konnte Encoder nicht laden: {str(e)}")
                            detector.encoder = None
                    
                    # Erstelle auch models-Dictionary für Konsistenz
                    detector.models = {
                        'autoencoder': {
                            'model': detector.model,
                            'encoder': detector.encoder if hasattr(detector, 'encoder') else None,
                            'threshold': detector.threshold
                        }
                    }
                else:
                    # Lade andere Modelle
                    detector.model = joblib.load(f"{path}_model.joblib")
                    
                    # Erstelle auch models-Dictionary für Konsistenz
                    detector.models = {
                        model_data['method']: {
                            'model': detector.model
                        }
                    }
            
            logger.info(f"Modell erfolgreich geladen von {path}")
            return detector
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das Modell zurück.
        
        Returns:
            Dict: Modell-Informationen
        """
        info = {
            'method': self.method,
            'contamination': self.contamination,
            'is_fitted': self.is_fitted,
            'feature_dims': self.feature_dims if self.is_fitted else None
        }
        
        # Füge Metadaten hinzu, falls vorhanden
        if hasattr(self, 'metadata'):
            info['metadata'] = self.metadata
        
        if self.is_fitted:
            # Füge Informationen über die Modelle hinzu
            if self.method == 'ensemble':
                info['ensemble_config'] = self.ensemble_config if hasattr(self, 'ensemble_config') else {}
                info['models'] = list(self.models.keys()) if hasattr(self, 'models') else []
                
                # Füge Informationen über die Performance-Metriken hinzu
                if hasattr(self, 'metadata') and 'performance_metrics' in self.metadata:
                    info['performance'] = self.metadata['performance_metrics']
            else:
                # Für einzelnes Modell
                if self.method == 'autoencoder':
                    info['model_type'] = 'Autoencoder'
                    info['threshold'] = self.threshold if hasattr(self, 'threshold') else None
                elif self.method == 'isolation_forest':
                    info['model_type'] = 'Isolation Forest'
                    if hasattr(self, 'model') and hasattr(self.model, 'n_estimators'):
                        info['n_estimators'] = self.model.n_estimators
                elif self.method == 'dbscan':
                    info['model_type'] = 'DBSCAN'
                    if hasattr(self, 'models') and 'dbscan' in self.models:
                        info['eps'] = self.models['dbscan'].get('eps')
                        info['min_samples'] = self.models['dbscan'].get('min_samples')
                elif self.method == 'lof':
                    info['model_type'] = 'Local Outlier Factor'
                elif self.method == 'ocsvm':
                    info['model_type'] = 'One-Class SVM'
                
                # Füge Informationen über die Performance-Metriken hinzu
                if hasattr(self, 'metadata') and 'performance_metrics' in self.metadata:
                    info['performance'] = self.metadata['performance_metrics']
        
        return info
