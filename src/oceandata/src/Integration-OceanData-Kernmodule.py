###########################################
# 6. Integration und OceanData-Kernmodule
###########################################

class OceanDataAI:
    """
    Hauptklasse, die alle KI-Module für OceanData integriert und koordiniert.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialisiert den OceanData-KI-Manager.
        
        Args:
            config: Konfigurationsparameter für die KI-Module
        """
        self.config = config or {}
        self.logger = logging.getLogger("OceanData.AI")
        
        # Instanziiere die einzelnen Module
        self.anomaly_detector = AnomalyDetector(
            method=self.config.get('anomaly_detection_method', 'isolation_forest'),
            contamination=self.config.get('anomaly_contamination', 0.05)
        )
        
        self.semantic_analyzer = SemanticAnalyzer(
            model_type=self.config.get('semantic_model', 'bert'),
            model_name=self.config.get('semantic_model_name', 'bert-base-uncased')
        )
        
        self.predictive_modeler = PredictiveModeler(
            model_type=self.config.get('predictive_model', 'lstm'),
            forecast_horizon=self.config.get('forecast_horizon', 7)
        )
        
        self.data_synthesizer = DataSynthesizer(
            categorical_threshold=self.config.get('categorical_threshold', 10),
            noise_dim=self.config.get('gan_noise_dim', 100)
        )
        
        self.c2d_manager = ComputeToDataManager()
        
        # Aktiviere GPU-Beschleunigung, falls verfügbar
        self._setup_hardware_acceleration()
    
    def _setup_hardware_acceleration(self):
        """Konfiguriert Hardware-Beschleunigung für die KI-Module"""
        try:
            # Prüfe, ob GPUs für TensorFlow verfügbar sind
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.logger.info(f"GPU-Beschleunigung aktiviert mit {len(gpus)} GPUs")
                
                # Konfiguriere TensorFlow für bessere GPU-Nutzung
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                self.logger.info("Keine GPUs gefunden, verwende CPU-Computing")
        
        except Exception as e:
            self.logger.warning(f"Fehler bei der GPU-Konfiguration: {str(e)}")
            self.logger.info("Verwende Standard-CPU-Computing")
    
    def analyze_data_source(self, data: pd.DataFrame, source_type: str) -> Dict:
        """
        Führt eine umfassende Analyse einer Datenquelle durch.
        
        Args:
            data: Die zu analysierenden Daten
            source_type: Typ der Datenquelle (browser, calendar, iot, etc.)
            
        Returns:
            Umfassende Analyseergebnisse
        """
        results = {
            'source_type': source_type,
            'timestamp': datetime.now().isoformat(),
            'record_count': len(data),
            'column_count': len(data.columns),
            'analyses': {}
        }
        
        try:
            # 1. Ausreißererkennung
            self.logger.info(f"Führe Ausreißererkennung für {source_type}-Daten durch")
            self.anomaly_detector.fit(data)
            anomaly_predictions = self.anomaly_detector.predict(data)
            anomaly_insights = self.anomaly_detector.get_anomaly_insights(
                data, anomaly_predictions, top_n_features=5
            )
            
            results['analyses']['anomalies'] = {
                'count': int(np.sum(anomaly_predictions == -1) if self.anomaly_detector.method == 'isolation_forest' else np.sum(anomaly_predictions)),
                'percentage': float(np.mean(anomaly_predictions == -1) * 100 if self.anomaly_detector.method == 'isolation_forest' else np.mean(anomaly_predictions) * 100),
                'insights': anomaly_insights[:5]  # Begrenze auf die Top-5-Insights
            }
            
            # 2. Textanalyse für relevante Textspalten
            text_columns = [col for col in data.columns 
                           if data[col].dtype == 'object' 
                           and data[col].str.len().mean() > 20]
            
            if text_columns:
                self.logger.info(f"Führe Textanalyse für {len(text_columns)} Spalten durch")
                text_analyses = {}
                
                for col in text_columns[:3]:  # Begrenze auf die ersten 3 Textspalten
                    # Wähle Texte aus, die nicht leer sind
                    texts = data[col].dropna().astype(str).tolist()
                    if not texts:
                        continue
                        
                    # Stichprobe für Performance
                    sample_size = min(100, len(texts))
                    text_sample = random.sample(texts, sample_size)
                    
                    # Sentimentanalyse
                    sentiments = self.semantic_analyzer.analyze_sentiment(text_sample)
                    
                    # Themenextraktion
                    topics = self.semantic_analyzer.extract_topics(text_sample, num_topics=3)
                    
                    text_analyses[col] = {
                        'sentiment': {
                            'positive': len([s for s in sentiments if s['sentiment'] == 'positive']) / len(sentiments),
                            'neutral': len([s for s in sentiments if s['sentiment'] == 'neutral']) / len(sentiments),
                            'negative': len([s for s in sentiments if s['sentiment'] == 'negative']) / len(sentiments)
                        },
                        'topics': topics
                    }
                
                results['analyses']['text'] = text_analyses
            
            # 3. Zeitreihenanalyse für Zeitstempel-basierte Daten
            datetime_columns = [col for col in data.columns 
                              if pd.api.types.is_datetime64_any_dtype(data[col])]
            
            if datetime_columns and len(data) > 10:
                self.logger.info("Führe Zeitreihenanalyse durch")
                
                # Wähle die erste Zeitstempelspalte für die Analyse
                time_col = datetime_columns[0]
                
                # Sortiere Daten nach Zeit
                sorted_data = data.sort_values(by=time_col)
                
                # Wähle numerische Spalten für die Vorhersage
                numeric_cols = sorted_data.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    # Wähle die ersten 3 numerischen Spalten für die Demonstration
                    target_cols = numeric_cols[:3]
                    
                    # Erstelle Zeitreihen-Features
                    sorted_data['hour'] = sorted_data[time_col].dt.hour
                    sorted_data['day_of_week'] = sorted_data[time_col].dt.dayofweek
                    sorted_data['day_of_month'] = sorted_data[time_col].dt.day
                    sorted_data['month'] = sorted_data[time_col].dt.month
                    
                    # Bilde Feature- und Zielspalten
                    X = sorted_data[['hour', 'day_of_week', 'day_of_month', 'month']]
                    y = sorted_data[target_cols]
                    
                    if len(X) > 20:  # Mindestanzahl für sinnvolle Zeitreihenanalyse
                        # Trainiere ein einfaches Modell
                        self.predictive_modeler.fit(X, y, lookback=5, epochs=20, verbose=0)
                        
                        # Forecasting für die nächsten 7 Tage
                        forecast = self.predictive_modeler.forecast(X, steps=7)
                        
                        results['analyses']['time_series'] = {
                            'forecast_horizon': 7,
                            'forecasted_features': target_cols,
                            'forecasts': {
                                col: forecast[:, i].tolist() 
                                for i, col in enumerate(target_cols)
                            }
                        }
            
            # 4. Datensyntheseanalyse (optional, ressourcenintensiv)
            if self.config.get('enable_data_synthesis', False) and len(data) > 50:
                self.logger.info("Führe Datensynthese-Analyse durch")
                
                # Wähle eine Stichprobe und maximal 10 Spalten für die Demonstration
                sample_size = min(1000, len(data))
                data_sample = data.sample(sample_size)
                
                if len(data.columns) > 10:
                    # Wähle die wichtigsten Spalten aus
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()[:5]
                    cat_cols = data.select_dtypes(exclude=['number']).columns.tolist()[:5]
                    selected_cols = numeric_cols + cat_cols
                    data_sample = data_sample[selected_cols]
                
                # Trainiere das GAN mit einer kleinen Anzahl von Epochen
                self.data_synthesizer.fit(data_sample, epochs=100, verbose=0)
                
                # Generiere einige synthetische Beispiele
                synthetic_examples = self.data_synthesizer.generate(10)
                
                # Bewerte die Qualität
                quality_metrics = self.data_synthesizer.evaluate_quality(100)
                
                results['analyses']['data_synthesis'] = {
                    'quality_score': quality_metrics.get('overall_quality_score', 0),
                    'sample_examples': synthetic_examples.head(3).to_dict()
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenanalyse: {str(e)}")
            self.logger.error(traceback.format_exc())
            results['error'] = str(e)
            return results
    
    def prepare_data_for_monetization(self, data: pd.DataFrame, source_type: str, 
                                    privacy_level: str = 'high') -> Dict:
        """
        Bereitet Daten für die Monetarisierung vor, einschließlich Anonymisierung und Metadatengenerierung.
        
        Args:
            data: Die zu monetarisierenden Daten
            source_type: Typ der Datenquelle
            privacy_level: Datenschutzniveau ('low', 'medium', 'high')
            
        Returns:
            Aufbereitete Daten mit Metadaten für die Monetarisierung
        """
        try:
            self.logger.info(f"Bereite {source_type}-Daten für Monetarisierung vor (Privacy: {privacy_level})")
            
            # 1. Datenschutzmaßnahmen basierend auf dem gewählten Niveau
            anonymized_data = data.copy()
            
            # Identifiziere sensible Spalten
            id_columns = [col for col in data.columns 
                         if 'id' in col.lower() or 'user' in col.lower() or 'email' in col.lower()]
            location_columns = [col for col in data.columns 
                              if 'location' in col.lower() or 'address' in col.lower() or 'gps' in col.lower() or 'ip' in col.lower()]
            datetime_columns = [col for col in data.columns 
                              if pd.api.types.is_datetime64_any_dtype(data[col])]
            
            # Anonymisierung basierend auf Datenschutzniveau
            if privacy_level == 'high':
                # Vollständige Anonymisierung und Entfernung sensibler Daten
                anonymized_data = anonymized_data.drop(columns=id_columns, errors='ignore')
                
                # Anonymisiere Standortdaten
                for col in location_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: f"location_{hashlib.md5(str(x).encode()).hexdigest()[:8]}" if pd.notna(x) else x
                        )
                
                # Vergröbere Zeitstempel (nur auf Tagesebene)
                for col in datetime_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = pd.to_datetime(anonymized_data[col]).dt.floor('D')
                
            elif privacy_level == 'medium':
                # Moderate Anonymisierung
                for col in id_columns:
                    if col in anonymized_data.columns:
                        salt = datetime.now().strftime("%Y%m%d")
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest() if pd.notna(x) else x
                        )
                
                # Vergröbere Standortdaten
                for col in location_columns:
                    if col in anonymized_data.columns and anonymized_data[col].dtype == 'object':