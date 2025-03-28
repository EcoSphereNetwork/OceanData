###########################################
# 3. Prädiktive Modellierung
###########################################

class PredictiveModeler:
    """
    Klasse für die Entwicklung von prädiktiven Modellen, die verschiedene Datentypen
    verarbeiten und Vorhersagen treffen können.
    """
    
    def __init__(self, model_type: str = 'lstm', forecast_horizon: int = 7):
        """
        Initialisiert den Prädiktiven Modellierer.
        
        Args:
            model_type: Typ des zu verwendenden Modells ('lstm', 'transformer', 'gru', 'arima')
            forecast_horizon: Anzahl der Zeitschritte für Vorhersagen in die Zukunft
        """
        self.model_type = model_type.lower()
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.lookback = 10  # Standardwert für die Anzahl der zurückliegenden Zeitschritte
        self.feature_dims = None
        self.target_dims = None
        self.target_scaler = None
        self.history = None
        
    def _build_lstm_model(self, input_shape, output_dim):
        """Erstellt ein LSTM-Modell für Zeitreihenvorhersage"""
        model = keras.Sequential()
        model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(32))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(output_dim))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _build_transformer_model(self, input_shape, output_dim):
        """Erstellt ein Transformer-Modell für Zeitreihenvorhersage"""
        # Einfaches Transformer-Modell für Zeitreihen
        inputs = keras.Input(shape=input_shape)
        
        # Positional encoding layer
        class PositionalEncoding(layers.Layer):
            def __init__(self, position, d_model):
                super(PositionalEncoding, self).__init__()
                self.pos_encoding = self.positional_encoding(position, d_model)
                
            def get_angles(self, position, i, d_model):
                angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
                return position * angles
            
            def positional_encoding(self, position, d_model):
                angle_rads = self.get_angles(
                    position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                    i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                    d_model=d_model
                )
                
                # Apply sine to even indices
                sines = tf.math.sin(angle_rads[:, 0::2])
                # Apply cosine to odd indices
                cosines = tf.math.cos(angle_rads[:, 1::2])
                
                pos_encoding = tf.concat([sines, cosines], axis=-1)
                pos_encoding = pos_encoding[tf.newaxis, ...]
                
                return tf.cast(pos_encoding, tf.float32)
            
            def call(self, inputs):
                return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        
        x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
        
        # Multi-head attention layer
        x = layers.MultiHeadAttention(
            key_dim=input_shape[1], num_heads=4, dropout=0.1
        )(x, x, x, attention_mask=None, training=True)
        
        # Feed-forward network
        x = layers.Dropout(0.1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=1, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv1D(filters=input_shape[1], kernel_size=1)(x)
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Output layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(output_dim)(x)
        
        model = keras.Model(inputs, x)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def _build_gru_model(self, input_shape, output_dim):
        """Erstellt ein GRU-Modell für Zeitreihenvorhersage"""
        model = keras.Sequential()
        model.add(layers.GRU(64, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.GRU(32))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(output_dim))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _create_sequences(self, data, target=None, lookback=None):
        """
        Erstellt Sequenzen für die Zeitreihenmodellierung
        
        Args:
            data: Eingabedaten (numpy array)
            target: Zielvariablen (optional, numpy array)
            lookback: Anzahl der zurückliegenden Zeitschritte (optional)
            
        Returns:
            X: Sequenzen für die Eingabe
            y: Zielwerte (wenn target bereitgestellt wird)
        """
        if lookback is None:
            lookback = self.lookback
        
        X = []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
        X = np.array(X)
        
        if target is not None:
            y = target[lookback:]
            return X, y
        
        return X
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, 
            lookback: int = 10, epochs: int = 50, 
            validation_split: float = 0.2, batch_size: int = 32,
            verbose: int = 1):
        """
        Trainiert das prädiktive Modell mit den gegebenen Daten.
        
        Args:
            X: Eingabedaten (DataFrame)
            y: Zielvariablen (DataFrame, optional für Zeitreihen)
            lookback: Anzahl der zurückliegenden Zeitschritte für Zeitreihenmodelle
            epochs: Anzahl der Trainings-Epochen
            validation_split: Anteil der Daten für die Validierung
            batch_size: Batch-Größe für das Training
            verbose: Ausgabedetailstufe (0, 1, oder 2)
            
        Returns:
            self: Trainiertes Modell
        """
        try:
            self.lookback = lookback
            
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            # Skalierung der Eingabedaten
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_values)
            
            # Vorbereiten der Zielvariablen
            if y is not None:
                if isinstance(y, pd.DataFrame):
                    y_values = y.values
                else:
                    y_values = y
                    
                self.target_scaler = StandardScaler()
                y_scaled = self.target_scaler.fit_transform(y_values)
                self.target_dims = y_scaled.shape[1]
            else:
                # Wenn keine Zielvariablen bereitgestellt werden, nehmen wir an, dass X selbst eine Zeitreihe ist
                y_scaled = X_scaled
                self.target_dims = X_scaled.shape[1]
            
            self.feature_dims = X_scaled.shape[1]
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, lookback)
            
            # Modell basierend auf dem ausgewählten Typ erstellen
            input_shape = (lookback, self.feature_dims)
            output_dim = self.target_dims * self.forecast_horizon
            
            if self.model_type == 'lstm':
                self.model = self._build_lstm_model(input_shape, output_dim)
            elif self.model_type == 'transformer':
                self.model = self._build_transformer_model(input_shape, output_dim)
            elif self.model_type == 'gru':
                self.model = self._build_gru_model(input_shape, output_dim)
            else:
                raise ValueError(f"Nicht unterstützter Modelltyp: {self.model_type}")
            
            # Reshape y_sequences für das Forecast-Horizon
            y_prepared = y_sequences.reshape(y_sequences.shape[0], -1)
            
            # Modell trainieren
            self.history = self.model.fit(
                X_sequences, y_prepared,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des prädiktiven Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X: pd.DataFrame, return_sequences: bool = False) -> np.ndarray:
        """
        Macht Vorhersagen mit dem trainierten Modell.
        
        Args:
            X: Eingabedaten (DataFrame)
            return_sequences: Ob die Vorhersagesequenz zurückgegeben werden soll
            
        Returns:
            Vorhersagen für die Eingabedaten
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_values)
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences = self._create_sequences(X_scaled, lookback=self.lookback)
            
            # Vorhersagen machen
            predictions_scaled = self.model.predict(X_sequences)
            
            # Reshape für die Ausgabe
            predictions_scaled = predictions_scaled.reshape(
                predictions_scaled.shape[0], 
                self.forecast_horizon, 
                self.target_dims
            )
            
            # Rücktransformation
            predictions = np.zeros_like(predictions_scaled)
            for i in range(self.forecast_horizon):
                step_predictions = predictions_scaled[:, i, :]
                # Rücktransformation nur für jeden Zeitschritt
                predictions[:, i, :] = self.target_scaler.inverse_transform(step_predictions)
            
            if return_sequences:
                return predictions
            else:
                # Nur den ersten Vorhersageschritt zurückgeben
                return predictions[:, 0, :]
            
        except Exception as e:
            logger.error(f"Fehler bei der Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def forecast(self, X: pd.DataFrame, steps: int = None) -> np.ndarray:
        """
        Erstellt eine Vorhersage für mehrere Zeitschritte in die Zukunft.
        
        Args:
            X: Letzte bekannte Datenpunkte (mindestens lookback viele)
            steps: Anzahl der vorherzusagenden Schritte (Standard: forecast_horizon)
            
        Returns:
            Vorhersagesequenz für die nächsten 'steps' Zeitschritte
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        if steps is None:
            steps = self.forecast_horizon
            
        try:
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
            
            if len(X_values) < self.lookback:
                raise ValueError(f"Eingabedaten müssen mindestens {self.lookback} Zeitschritte enthalten")
            
            # Verwende nur die letzten 'lookback' Zeitschritte
            X_recent = X_values[-self.lookback:]
            
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_recent)
            X_sequence = X_scaled.reshape(1, self.lookback, self.feature_dims)
            
            # Erstelle die multi-step-Vorhersage
            forecast_values = []
            
            current_sequence = X_sequence.copy()
            
            for _ in range(steps):
                # Mache eine Vorhersage für den nächsten Schritt
                next_step_scaled = self.model.predict(current_sequence)[0]
                next_step_scaled = next_step_scaled.reshape(1, self.target_dims)
                
                # Rücktransformation
                next_step = self.target_scaler.inverse_transform(next_step_scaled)
                forecast_values.append(next_step[0])
                
                # Aktualisiere die Eingabesequenz für den nächsten Schritt
                # Entferne den ersten Zeitschritt und füge den neu vorhergesagten hinzu
                new_sequence = np.zeros_like(current_sequence)
                new_sequence[0, :-1, :] = current_sequence[0, 1:, :]
                new_sequence[0, -1, :] = next_step_scaled
                current_sequence = new_sequence
            
            return np.array(forecast_values)
            
        except Exception as e:
            logger.error(f"Fehler bei der Mehrschritt-Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluiert das Modell mit Testdaten.
        
        Args:
            X_test: Test-Eingabedaten
            y_test: Test-Zielvariablen
            
        Returns:
            Dictionary mit Bewertungsmetriken
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Testdaten
            if isinstance(X_test, pd.DataFrame):
                X_values = X_test.values
            else:
                X_values = X_test
                
            if isinstance(y_test, pd.DataFrame):
                y_values = y_test.values
            else:
                y_values = y_test
            
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_values)
            y_scaled = self.target_scaler.transform(y_values)
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, self.lookback)
            
            # Reshape y_sequences für das Forecast-Horizon
            y_prepared = y_sequences.reshape(y_sequences.shape[0], -1)
            
            # Modell evaluieren
            evaluation = self.model.evaluate(X_sequences, y_prepared, verbose=0)
            
            # Vorhersagen machen für detailliertere Metriken
            predictions = self.predict(X_test)
            
            # Tatsächliche Werte (ohne die ersten lookback Zeitschritte)
            actuals = y_values[self.lookback:]
            
            # Berechne RMSE, MAE, MAPE
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Entsprechende Anzahl an Vorhersagen auswählen
            predictions = predictions[:len(actuals)]
            
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # MAPE (Mean Absolute Percentage Error)
            # Vermeide Division durch Null
            mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
            
            return {
                'loss': evaluation[0],
                'mae': evaluation[1],
                'rmse': rmse,
                'mean_absolute_error': mae,
                'mean_absolute_percentage_error': mape
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Modellbewertung: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def plot_forecast(self, X: pd.DataFrame, y_true: pd.DataFrame = None, 
                     steps: int = None, feature_idx: int = 0,
                     save_path: str = None) -> plt.Figure:
        """
        Visualisiert die Vorhersage des Modells.
        
        Args:
            X: Eingabedaten
            y_true: Tatsächliche zukünftige Werte (optional)
            steps: Anzahl der vorherzusagenden Schritte
            feature_idx: Index der darzustellenden Feature-Dimension
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Standard-Schritte
            if steps is None:
                steps = self.forecast_horizon
            
            # Vorhersage erstellen
            forecast_values = self.forecast(X, steps)
            
            # Historische Daten (letzte lookback Zeitschritte)
            if isinstance(X, pd.DataFrame):
                historical_values = X.values[-self.lookback:, feature_idx]
            else:
                historical_values = X[-self.lookback:, feature_idx]
            
            # Zeitachse erstellen
            time_hist = np.arange(-self.lookback, 0)
            time_future = np.arange(0, steps)
            
            # Visualisierung erstellen
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historische Daten plotten
            ax.plot(time_hist, historical_values, 'b-', label='Historische Daten')
            
            # Vorhersage plotten
            ax.plot(time_future, forecast_values[:, feature_idx], 'r-', label='Vorhersage')
            
            # Tatsächliche zukünftige Werte plotten, falls vorhanden
            if y_true is not None:
                if isinstance(y_true, pd.DataFrame):
                    true_future = y_true.values[:steps, feature_idx]
                else:
                    true_future = y_true[:steps, feature_idx]
                
                ax.plot(time_future[:len(true_future)], true_future, 'g-', label='Tatsächliche Werte')
            
            # Grafik anpassen
            ax.set_title(f'Zeitreihenvorhersage mit {self.model_type.upper()}')
            ax.set_xlabel('Zeitschritte')
            ax.set_ylabel(f'Feature {feature_idx}')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Trennlinie zwischen historischen und Vorhersagedaten
            ax.axvline(x=0, color='k', linestyle='--')
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung der Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        model_data = {
            "model_type": self.model_type,
            "forecast_horizon": self.forecast_horizon,
            "lookback": self.lookback,
            "feature_dims": self.feature_dims,
            "target_dims": self.target_dims,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere das Model-Objekt
        self.model.save(f"{path}_model")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        joblib.dump(self.target_scaler, f"{path}_target_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        predictor = cls(
            model_type=model_data['model_type'],
            forecast_horizon=model_data['forecast_horizon']
        )
        
        predictor.lookback = model_data['lookback']
        predictor.feature_dims = model_data['feature_dims']
        predictor.target_dims = model_data['target_dims']
        predictor.is_fitted = model_data['is_fitted']
        
        # Lade das Model-Objekt
        predictor.model = keras.models.load_model(f"{path}_model")
        
        # Lade die Scaler
        predictor.scaler = joblib.load(f"{path}_scaler.joblib")
        predictor.target_scaler = joblib.load(f"{path}_target_scaler.joblib")
        
        return predictor
