###########################################
# 1. Anomalieerkennung & Outlier Detection
###########################################

class AnomalyDetector:
    """Klasse für die Erkennung von Anomalien in verschiedenen Datentypen"""
    
    def __init__(self, method: str = 'autoencoder', contamination: float = 0.05):
        """
        Initialisiert einen Anomaliedetektor.
        
        Args:
            method: Methode für die Anomalieerkennung ('autoencoder', 'isolation_forest', 'dbscan')
            contamination: Erwarteter Anteil an Anomalien in den Daten (für Isolation Forest)
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dims = None
        self.threshold = None
        
    def _build_autoencoder(self, input_dim):
        """Erstellt ein Autoencoder-Modell für die Anomalieerkennung"""
        encoding_dim = max(1, input_dim // 2)
        hidden_dim = max(1, encoding_dim // 2)
        
        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(inputs)
        encoded = layers.Dense(hidden_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Encoder model for extracting features
        encoder = keras.Model(inputs, encoded)
        
        return autoencoder, encoder
    
    def fit(self, X: pd.DataFrame, categorical_cols: List[str] = None):
        """
        Trainiert den Anomaliedetektor mit Daten.
        
        Args:
            X: Eingabedaten (DataFrame oder numpy array)
            categorical_cols: Liste der kategorischen Spalten, die One-Hot-kodiert werden müssen
        """
        try:
            # Vorbereiten der Daten
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten durch One-Hot-Encoding
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Konvertieren zu float und Behandlung von NaN-Werten
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            
            if X_prep.shape[1] == 0:
                raise ValueError("Nach der Vorverarbeitung sind keine numerischen Merkmale übrig")
                
            X_scaled = self.scaler.fit_transform(X_prep)
            self.feature_dims = X_scaled.shape[1]
            
            if self.method == 'autoencoder':
                # Autoencoder-Ansatz für Anomalieerkennung
                self.model, self.encoder = self._build_autoencoder(self.feature_dims)
                
                # Trainiere den Autoencoder
                self.model.fit(
                    X_scaled, X_scaled,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    verbose=0,
                    validation_split=0.1
                )
                
                # Berechne Rekonstruktionsfehler für alle Trainingsdaten
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Setze Threshold basierend auf den Rekonstruktionsfehlern
                self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
                
            elif self.method == 'isolation_forest':
                # Isolation Forest für Outlier-Erkennung
                self.model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
                self.model.fit(X_scaled)
                
            elif self.method == 'dbscan':
                # DBSCAN für Clustering-basierte Outlier-Erkennung
                self.model = DBSCAN(
                    eps=0.5,  # Maximum distance between samples
                    min_samples=5,  # Minimum samples in a cluster
                    n_jobs=-1
                )
                self.model.fit(X_scaled)
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des Anomaliedetektors: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X: pd.DataFrame, categorical_cols: List[str] = None) -> np.ndarray:
        """
        Identifiziert Anomalien in den Daten.
        
        Args:
            X: Zu prüfende Daten
            categorical_cols: Liste der kategorischen Spalten
            
        Returns:
            Array mit Anomalie-Scores für jede Zeile (-1 für Anomalien, 1 für normale Daten bei Isolation Forest;
            True für Anomalien, False für normale Daten bei Autoencoder)
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
            
            if self.method == 'autoencoder':
                # Berechne Rekonstruktionsfehler
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Anomalie, wenn Fehler über dem Threshold
                return reconstruction_errors > self.threshold
                
            elif self.method == 'isolation_forest':
                # -1 für Anomalien, 1 für normale Daten
                return self.model.predict(X_scaled)
                
            elif self.method == 'dbscan':
                # Ordne neue Punkte den nächsten Clustern zu
                labels = self.model.fit_predict(X_scaled)
                # -1 sind Noise-Punkte (Anomalien)
                return labels == -1
                
        except Exception as e:
            logger.error(f"Fehler bei der Anomalievorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
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
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden")
            
        model_data = {
            "method": self.method,
            "contamination": self.contamination,
            "feature_dims": self.feature_dims,
            "threshold": self.threshold,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere das Model-Objekt
        if self.method in ['isolation_forest', 'dbscan']:
            joblib.dump(self.model, f"{path}_model.joblib")
        elif self.method == 'autoencoder':
            self.model.save(f"{path}_autoencoder")
            self.encoder.save(f"{path}_encoder")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        detector = cls(
            method=model_data['method'],
            contamination=model_data['contamination']
        )
        
        detector.feature_dims = model_data['feature_dims']
        detector.threshold = model_data['threshold']
        detector.is_fitted = model_data['is_fitted']
        
        # Lade das Model-Objekt
        if detector.method in ['isolation_forest', 'dbscan']:
            detector.model = joblib.load(f"{path}_model.joblib")
        elif detector.method == 'autoencoder':
            detector.model = keras.models.load_model(f"{path}_autoencoder")
            detector.encoder = keras.models.load_model(f"{path}_encoder")
        
        # Lade den Scaler
        detector.scaler = joblib.load(f"{path}_scaler.joblib")
        
        return detector
