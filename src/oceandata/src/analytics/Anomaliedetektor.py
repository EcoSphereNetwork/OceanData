"""
OceanData - Anomaliedetektor

Dieses Modul implementiert verschiedene Algorithmen zur Erkennung von Anomalien in Daten,
was für die Wertschöpfung und für Insights besonders nützlich ist.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
import warnings

# Unterdrücke Warnungen
warnings.filterwarnings('ignore')

# Logging konfigurieren
logger = logging.getLogger("OceanData.Analytics.AnomalyDetector")

class AnomalyDetectionMethod(Enum):
    """Enum für verschiedene Anomalieerkennungsmethoden"""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"
    STATISTICAL = "statistical"

class AnomalyDetector:
    """
    Klasse zur Erkennung von Anomalien in Datensätzen.
    
    Unterstützt verschiedene Algorithmen:
    - Isolation Forest (für allgemeine Ausreißererkennung)
    - Local Outlier Factor (für dichtebasierte Ausreißererkennung)
    - One-Class SVM (für Grenzfallerkennung)
    - Autoencoder (für komplexe, hochdimensionale Datenstrukturen)
    - Statistische Methoden (Z-Score, IQR)
    """
    
    def __init__(self, method: Union[str, AnomalyDetectionMethod] = "isolation_forest", contamination: float = 0.05):
        """
        Initialisiert den Anomaliedetektor.
        
        Args:
            method: Die zu verwendende Methode ('isolation_forest', 'local_outlier_factor', 
                   'one_class_svm', 'autoencoder', 'statistical')
            contamination: Erwarteter Anteil von Anomalien in den Daten (0 bis 0.5)
        """
        if isinstance(method, str):
            try:
                self.method = AnomalyDetectionMethod(method)
            except ValueError:
                logger.warning(f"Unbekannte Methode: {method}, verwende Isolation Forest")
                self.method = AnomalyDetectionMethod.ISOLATION_FOREST
        else:
            self.method = method
            
        self.contamination = max(0.001, min(0.5, contamination))
        logger.info(f"Anomaliedetektor initialisiert mit Methode '{self.method.value}' und Kontaminationsrate {self.contamination}")
        
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_dims = None
        
        # Initialize the model based on the chosen method
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialisiert das Modell basierend auf der gewählten Methode"""
        try:
            if self.method == AnomalyDetectionMethod.ISOLATION_FOREST:
                from sklearn.ensemble import IsolationForest
                self.model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
                
            elif self.method == AnomalyDetectionMethod.LOCAL_OUTLIER_FACTOR:
                from sklearn.neighbors import LocalOutlierFactor
                self.model = LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=self.contamination
                )
                
            elif self.method == AnomalyDetectionMethod.ONE_CLASS_SVM:
                from sklearn.svm import OneClassSVM
                self.model = OneClassSVM(
                    nu=self.contamination,
                    kernel='rbf',
                    gamma='scale'
                )
                
            elif self.method == AnomalyDetectionMethod.AUTOENCODER:
                try:
                    import tensorflow as tf
                    from tensorflow.keras.models import Model
                    from tensorflow.keras.layers import Input, Dense
                    
                    # Dies ist nur ein Platzhalter - das tatsächliche Modell wird in fit() erstellt
                    # sobald wir die Eingabedimensionen kennen
                    self.model = None
                    self.threshold = None
                    
                except ImportError:
                    logger.warning("TensorFlow nicht verfügbar, verwende Isolation Forest stattdessen")
                    self.method = AnomalyDetectionMethod.ISOLATION_FOREST
                    from sklearn.ensemble import IsolationForest
                    self.model = IsolationForest(
                        contamination=self.contamination,
                        random_state=42
                    )
                    
            elif self.method == AnomalyDetectionMethod.STATISTICAL:
                # Für statistische Methoden verwenden wir keinen vordefinierten Algorithmus
                # Stattdessen implementieren wir Z-Score und IQR-basierte Erkennung selbst
                self.model = {
                    "method": "statistical",
                    "z_threshold": 3.0,  # Standardwert für Z-Score
                    "iqr_factor": 1.5    # Standardwert für IQR
                }
            
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des Modells: {str(e)}")
            # Fallback zu einer einfachen Methode
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            self.method = AnomalyDetectionMethod.ISOLATION_FOREST
    
    def fit(self, X: pd.DataFrame, categorical_cols: List[str] = None) -> 'AnomalyDetector':
        """
        Trainiert den Anomaliedetektor mit den gegebenen Daten.
        
        Args:
            X: DataFrame mit Trainingsdaten
            categorical_cols: Liste mit Namen der kategorialen Spalten, die gesondert behandelt werden müssen
            
        Returns:
            AnomalyDetector: Das trainierte Modell zur Verkettung von Aufrufen
        """
        try:
            if X is None or X.empty:
                logger.warning("Leere Daten für Training übergeben")
                return self
            
            # Verarbeite Daten und behandle kategoriale Variablen
            X_processed = self._preprocess_data(X, categorical_cols)
            
            # Speichere die Dimensionalität der Features
            self.feature_dims = X_processed.shape[1]
            
            # Wende den Scaler an
            X_scaled = self.scaler.fit_transform(X_processed)
            
            # Fit the model depending on the method
            if self.method == AnomalyDetectionMethod.ISOLATION_FOREST or \
               self.method == AnomalyDetectionMethod.ONE_CLASS_SVM:
                self.model.fit(X_scaled)
                
            elif self.method == AnomalyDetectionMethod.LOCAL_OUTLIER_FACTOR:
                # LOF wird im Vorhersageschritt ausgeführt (novelty=False)
                pass
                
            elif self.method == AnomalyDetectionMethod.AUTOENCODER:
                # Baue und trainiere das Autoencoder-Modell
                import tensorflow as tf
                from tensorflow.keras.models import Model
                from tensorflow.keras.layers import Input, Dense
                from tensorflow.keras.callbacks import EarlyStopping
                
                # Dimensionen definieren
                input_dim = X_scaled.shape[1]
                encoding_dim = min(input_dim // 2, 32)  # Komprimierte Dimension
                
                # Eingabeschicht
                input_layer = Input(shape=(input_dim,))
                
                # Encoder
                encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
                encoded = Dense(encoding_dim, activation='relu')(encoded)
                
                # Decoder
                decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
                decoded = Dense(input_dim, activation='sigmoid')(decoded)
                
                # Autoencoder
                autoencoder = Model(input_layer, decoded)
                autoencoder.compile(optimizer='adam', loss='mse')
                
                # Trainieren mit Early Stopping
                early_stopping = EarlyStopping(monitor='val_loss', patience=5)
                history = autoencoder.fit(
                    X_scaled, X_scaled,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Rekonstruktionsfehler berechnen
                reconstructions = autoencoder.predict(X_scaled)
                mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
                
                # Threshold für Anomalien bestimmen (z.B. 95. Perzentil)
                self.threshold = np.percentile(mse, 100 * (1 - self.contamination))
                self.model = autoencoder
                
            elif self.method == AnomalyDetectionMethod.STATISTICAL:
                # Berechne Statistiken für jede numerische Spalte
                stats = {}
                for col in X_processed.columns:
                    values = X_processed[col].values
                    stats[col] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'q1': np.percentile(values, 25),
                        'q3': np.percentile(values, 75),
                        'iqr': np.percentile(values, 75) - np.percentile(values, 25)
                    }
                self.model['stats'] = stats
            
            self.is_fitted = True
            logger.info(f"Modell erfolgreich mit {len(X)} Datenpunkten trainiert")
            
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des Modells: {str(e)}")
            return self
    
    def predict(self, X: pd.DataFrame, categorical_cols: List[str] = None) -> np.ndarray:
        """
        Erkennt Anomalien in den gegebenen Daten.
        
        Args:
            X: DataFrame mit zu analysierenden Daten
            categorical_cols: Liste mit Namen der kategorialen Spalten
            
        Returns:
            np.ndarray: Array mit Ergebnissen (je nach Methode unterschiedliches Format)
        """
        if not self.is_fitted:
            logger.error("Modell wurde noch nicht trainiert")
            raise ValueError("Modell wurde noch nicht trainiert. Rufe zuerst fit() auf.")
            
        try:
            # Verarbeite Daten
            X_processed = self._preprocess_data(X, categorical_cols)
            
            # Wende den Scaler an
            X_scaled = self.scaler.transform(X_processed)
            
            # Vorhersage je nach Methode
            if self.method == AnomalyDetectionMethod.ISOLATION_FOREST or \
               self.method == AnomalyDetectionMethod.ONE_CLASS_SVM:
                # Diese Modelle geben -1 für Anomalien, 1 für normale Daten zurück
                predictions = self.model.predict(X_scaled)
                
            elif self.method == AnomalyDetectionMethod.LOCAL_OUTLIER_FACTOR:
                # LOF gibt negative Scores für Anomalien zurück
                scores = self.model.fit_predict(X_scaled)
                predictions = scores
                
            elif self.method == AnomalyDetectionMethod.AUTOENCODER:
                # Rekonstruktionsfehler berechnen
                reconstructions = self.model.predict(X_scaled)
                mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
                
                # Anomalie, wenn Fehler über dem Schwellenwert
                predictions = np.where(mse > self.threshold, -1, 1)
                
            elif self.method == AnomalyDetectionMethod.STATISTICAL:
                # Anomalien mit Z-Score und IQR erkennen
                predictions = np.ones(len(X_processed))  # Standard: 1 (normal)
                
                for col in X_processed.columns:
                    col_stats = self.model['stats'][col]
                    
                    # Z-Score-basierte Anomalien
                    z_scores = np.abs((X_processed[col] - col_stats['mean']) / col_stats['std'])
                    z_anomalies = z_scores > self.model['z_threshold']
                    
                    # IQR-basierte Anomalien
                    iqr_lower = col_stats['q1'] - self.model['iqr_factor'] * col_stats['iqr']
                    iqr_upper = col_stats['q3'] + self.model['iqr_factor'] * col_stats['iqr']
                    iqr_anomalies = (X_processed[col] < iqr_lower) | (X_processed[col] > iqr_upper)
                    
                    # Kombiniere Erkennungen
                    combined_anomalies = z_anomalies | iqr_anomalies
                    
                    # Markiere Anomalien mit -1
                    predictions[combined_anomalies] = -1
            
            logger.info(f"Anomalieerkennung für {len(X)} Datenpunkte abgeschlossen")
            return predictions
            
        except Exception as e:
            logger.error(f"Fehler bei der Anomalieerkennung: {str(e)}")
            # Im Fehlerfall alles als normal markieren
            return np.ones(len(X))
    
    def get_anomaly_scores(self, X: pd.DataFrame, categorical_cols: List[str] = None) -> np.ndarray:
        """
        Berechnet Anomalie-Scores für die gegebenen Daten (höher = anomaler).
        
        Args:
            X: DataFrame mit zu analysierenden Daten
            categorical_cols: Liste mit Namen der kategorialen Spalten
            
        Returns:
            np.ndarray: Array mit Anomalie-Scores
        """
        if not self.is_fitted:
            logger.error("Modell wurde noch nicht trainiert")
            raise ValueError("Modell wurde noch nicht trainiert. Rufe zuerst fit() auf.")
            
        try:
            # Verarbeite Daten
            X_processed = self._preprocess_data(X, categorical_cols)
            
            # Wende den Scaler an
            X_scaled = self.scaler.transform(X_processed)
            
            # Scores je nach Methode
            if self.method == AnomalyDetectionMethod.ISOLATION_FOREST:
                # Negierte Decision-Function (höher = anomaler)
                scores = -self.model.decision_function(X_scaled)
                
            elif self.method == AnomalyDetectionMethod.ONE_CLASS_SVM:
                # Negierte Decision-Function (höher = anomaler)
                scores = -self.model.decision_function(X_scaled)
                
            elif self.method == AnomalyDetectionMethod.LOCAL_OUTLIER_FACTOR:
                # Negierte Scores (höher = anomaler)
                scores = -self.model.negative_outlier_factor_
                
            elif self.method == AnomalyDetectionMethod.AUTOENCODER:
                # Rekonstruktionsfehler (höher = anomaler)
                reconstructions = self.model.predict(X_scaled)
                scores = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
                
            elif self.method == AnomalyDetectionMethod.STATISTICAL:
                # Kombinierte Z-Scores über alle Spalten
                scores = np.zeros(len(X_processed))
                
                for col in X_processed.columns:
                    col_stats = self.model['stats'][col]
                    z_scores = np.abs((X_processed[col] - col_stats['mean']) / col_stats['std'])
                    scores += z_scores
                
                # Normalisieren auf den Bereich [0, 1]
                max_score = np.max(scores) if len(scores) > 0 else 1
                scores = scores / max_score if max_score > 0 else scores
            
            logger.info(f"Anomalie-Scores für {len(X)} Datenpunkte berechnet")
            return scores
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung von Anomalie-Scores: {str(e)}")
            # Im Fehlerfall neutrale Scores zurückgeben
            return np.zeros(len(X))
    
    def get_anomaly_insights(self, X: pd.DataFrame, predictions: np.ndarray) -> List[Dict]:
        """
        Liefert Erkenntnisse über die gefundenen Anomalien.
        
        Args:
            X: DataFrame mit analysierten Daten
            predictions: Ergebnisse der Anomalieerkennung
            
        Returns:
            List[Dict]: Liste mit Erkenntnissen über Anomalien
        """
        try:
            if predictions is None or len(predictions) != len(X):
                logger.error("Ungültige Vorhersagen für Insights")
                return []
                
            # Identifiziere Anomalien
            if self.method in [AnomalyDetectionMethod.ISOLATION_FOREST, 
                             AnomalyDetectionMethod.ONE_CLASS_SVM, 
                             AnomalyDetectionMethod.AUTOENCODER]:
                anomaly_indices = np.where(predictions == -1)[0]
            else:
                # Für andere Methoden: negative Werte sind Anomalien
                anomaly_indices = np.where(predictions < 0)[0]
                
            if len(anomaly_indices) == 0:
                logger.info("Keine Anomalien gefunden")
                return []
                
            # Berechne Anomalie-Scores für alle Punkte
            anomaly_scores = self.get_anomaly_scores(X)
            
            # Extrahiere Anomaliedaten
            anomaly_data = X.iloc[anomaly_indices].copy()
            anomaly_data['anomaly_score'] = anomaly_scores[anomaly_indices]
            
            # Sortiere nach Anomalie-Score (absteigend)
            anomaly_data = anomaly_data.sort_values('anomaly_score', ascending=False)
            
            # Identifiziere die wichtigsten Merkmale für jede Anomalie
            insights = []
            
            for idx, row in anomaly_data.iterrows():
                # Berechne Z-Scores für jede Spalte
                z_scores = {}
                for col in X.columns:
                    if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                        mean = X[col].mean()
                        std = X[col].std()
                        if std > 0:
                            z_scores[col] = abs((row[col] - mean) / std)
                
                # Sortiere Merkmale nach Z-Score (absteigend)
                sorted_features = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Erstelle Insight-Objekt
                insight = {
                    "index": idx,
                    "anomaly_score": float(row['anomaly_score']),
                    "top_features": [{"name": f[0], "z_score": float(f[1])} for f in sorted_features[:3]],
                    "values": {col: row[col] for col in X.columns if col != 'anomaly_score'}
                }
                
                insights.append(insight)
            
            logger.info(f"{len(insights)} Anomalie-Insights erstellt")
            return insights
            
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung von Anomalie-Insights: {str(e)}")
            return []
    
    def _preprocess_data(self, X: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
        """
        Verarbeitet Daten für die Anomalieerkennung vor, einschließlich der Behandlung kategorialer Variablen.
        
        Args:
            X: DataFrame mit zu verarbeitenden Daten
            categorical_cols: Liste mit Namen der kategorialen Spalten
            
        Returns:
            pd.DataFrame: Verarbeitete Daten
        """
        try:
            # Kopiere Daten, um das Original nicht zu verändern
            X_processed = X.copy()
            
            # Entferne nicht-numerische Spalten, wenn nicht als kategorial angegeben
            if categorical_cols is None:
                categorical_cols = []
                
            # Identifiziere automatisch kategoriale Spalten, wenn nicht explizit angegeben
            for col in X_processed.columns:
                if X_processed[col].dtype == 'object' and col not in categorical_cols:
                    categorical_cols.append(col)
            
            # One-Hot-Encoding für kategoriale Variablen
            for col in categorical_cols:
                if col in X_processed.columns:
                    # Erstelle Dummy-Variablen
                    dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=False)
                    
                    # Füge Dummies hinzu und entferne Originalspalte
                    X_processed = pd.concat([X_processed, dummies], axis=1)
                    X_processed = X_processed.drop(col, axis=1)
            
            # Entferne verbleibende nicht-numerische Spalten
            for col in X_processed.columns:
                if X_processed[col].dtype not in [np.float64, np.float32, np.int64, np.int32, bool]:
                    X_processed = X_processed.drop(col, axis=1)
            
            # Behandle boolesche Spalten
            for col in X_processed.columns:
                if X_processed[col].dtype == bool:
                    X_processed[col] = X_processed[col].astype(int)
            
            # Behandle fehlende Werte
            X_processed = X_processed.fillna(X_processed.mean())
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenvorverarbeitung: {str(e)}")
            # Fallback: Nur numerische Spalten verwenden
            numeric_cols = X.select_dtypes(include=['number']).columns
            return X[numeric_cols].fillna(0)
    
    def visualize_anomalies(self, X: pd.DataFrame, predictions: np.ndarray = None, 
                          categorical_cols: List[str] = None):
        """
        Erstellt eine Visualisierung der erkannten Anomalien.
        
        Args:
            X: DataFrame mit analysierten Daten
            predictions: Optional. Vorhersagen aus predict(). Wenn None, werden sie berechnet.
            categorical_cols: Liste mit Namen der kategorialen Spalten
            
        Returns:
            matplotlib.figure.Figure: Visualisierung der Anomalien
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            
            # Berechne Vorhersagen, falls nicht angegeben
            if predictions is None:
                predictions = self.predict(X, categorical_cols)
            
            # Identifiziere Anomalien
            if self.method in [AnomalyDetectionMethod.ISOLATION_FOREST, 
                             AnomalyDetectionMethod.ONE_CLASS_SVM, 
                             AnomalyDetectionMethod.AUTOENCODER]:
                anomaly_indices = np.where(predictions == -1)[0]
                normal_indices = np.where(predictions == 1)[0]
            else:
                # Für andere Methoden: negative Werte sind Anomalien
                anomaly_indices = np.where(predictions < 0)[0]
                normal_indices = np.where(predictions >= 0)[0]
            
            # Verarbeite Daten für die Visualisierung
            X_processed = self._preprocess_data(X, categorical_cols)
            
            # Wähle die besten Features für die Visualisierung
            if len(X_processed.columns) > 2:
                # PCA für Dimensionsreduktion
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(self.scaler.transform(X_processed))
                
                fig = Figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                
                # Plotte normale Punkte
                ax.scatter(X_pca[normal_indices, 0], X_pca[normal_indices, 1], 
                           c='blue', s=30, alpha=0.5, label='Normal')
                
                # Plotte Anomalien
                ax.scatter(X_pca[anomaly_indices, 0], X_pca[anomaly_indices, 1], 
                           c='red', s=50, marker='x', label='Anomalie')
                
                ax.set_title(f'Anomalieerkennung mit {self.method.value} (PCA-Visualisierung)')
                ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} Varianz)')
                ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} Varianz)')
                ax.legend()
                
            else:
                # Direkte Visualisierung wenn nur 1-2 Features
                fig = Figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                
                if len(X_processed.columns) == 1:
                    # 1D-Plot
                    feature = X_processed.columns[0]
                    
                    # Normale Punkte
                    ax.scatter(X_processed.iloc[normal_indices][feature], 
                               np.zeros(len(normal_indices)), 
                               c='blue', s=30, alpha=0.5, label='Normal')
                    
                    # Anomalien
                    ax.scatter(X_processed.iloc[anomaly_indices][feature], 
                               np.zeros(len(anomaly_indices)), 
                               c='red', s=50, marker='x', label='Anomalie')
                    
                    ax.set_title(f'Anomalieerkennung mit {self.method.value}')
                    ax.set_xlabel(feature)
                    ax.set_yticks([])
                    ax.legend()
                    
                else:
                    # 2D-Plot
                    features = X_processed.columns.tolist()
                    
                    # Normale Punkte
                    ax.scatter(X_processed.iloc[normal_indices][features[0]], 
                               X_processed.iloc[normal_indices][features[1]], 
                               c='blue', s=30, alpha=0.5, label='Normal')
                    
                    # Anomalien
                    ax.scatter(X_processed.iloc[anomaly_indices][features[0]], 
                               X_processed.iloc[anomaly_indices][features[1]], 
                               c='red', s=50, marker='x', label='Anomalie')
                    
                    ax.set_title(f'Anomalieerkennung mit {self.method.value}')
                    ax.set_xlabel(features[0])
                    ax.set_ylabel(features[1])
                    ax.legend()
            
            logger.info(f"Anomalie-Visualisierung erstellt: {len(anomaly_indices)} Anomalien, {len(normal_indices)} normale Datenpunkte")
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung von Anomalien: {str(e)}")
            # Leeres Figure zurückgeben
            return Figure()
