###########################################
# 4. Datensynthese und GAN-basierte Modelle
###########################################

class DataSynthesizer:
    """
    Klasse zur Generierung synthetischer Daten basierend auf realen Beispielen.
    Verwendet GAN (Generative Adversarial Network) für realistische Datensynthese.
    """
    
    def __init__(self, categorical_threshold: int = 10, noise_dim: int = 100):
        """
        Initialisiert den Datensynthetisierer.
        
        Args:
            categorical_threshold: Anzahl eindeutiger Werte, ab der eine Spalte als kategorisch gilt
            noise_dim: Dimension des Rauschvektors für den Generator
        """
        self.categorical_threshold = categorical_threshold
        self.noise_dim = noise_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        
        self.column_types = {}  # Speichert, ob eine Spalte kategorisch oder kontinuierlich ist
        self.categorical_mappings = {}  # Speichert Mappings für kategorische Spalten
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # Für kontinuierliche Variablen
        
        self.is_fitted = False
        self.feature_dims = None
        self.training_data = None
    
    def _identify_column_types(self, data: pd.DataFrame):
        """Identifiziert, ob Spalten kategorisch oder kontinuierlich sind"""
        self.column_types = {}
        
        for col in data.columns:
            n_unique = data[col].nunique()
            
            # Wenn die Anzahl eindeutiger Werte kleiner als der Schwellenwert ist oder
            # der Datentyp ist nicht numerisch, behandle die Spalte als kategorisch
            if n_unique < self.categorical_threshold or not pd.api.types.is_numeric_dtype(data[col]):
                self.column_types[col] = 'categorical'
                
                # Erstelle Mapping von Kategorien zu Zahlen
                categories = data[col].unique()
                self.categorical_mappings[col] = {
                    cat: i for i, cat in enumerate(categories)
                }
                # Umgekehrtes Mapping für die Rücktransformation
                self.categorical_mappings[f"{col}_reverse"] = {
                    i: cat for i, cat in enumerate(categories)
                }
            else:
                self.column_types[col] = 'continuous'
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Vorverarbeitung der Daten für das GAN"""
        processed_data = pd.DataFrame()
        
        for col in data.columns:
            if self.column_types[col] == 'categorical':
                # One-Hot-Encoding für kategorische Spalten
                mapped_col = data[col].map(self.categorical_mappings[col])
                one_hot = pd.get_dummies(mapped_col, prefix=col)
                processed_data = pd.concat([processed_data, one_hot], axis=1)
            else:
                # Skalierung für kontinuierliche Spalten
                processed_data[col] = data[col]
        
        # Skaliere alle Spalten auf [-1, 1]
        return self.scaler.fit_transform(processed_data)
    
    def _postprocess_data(self, generated_data: np.ndarray) -> pd.DataFrame:
        """Nachverarbeitung der generierten Daten zurück in das ursprüngliche Format"""
        # Rücktransformation der Skalierung
        rescaled_data = self.scaler.inverse_transform(generated_data)
        
        # Erstelle einen DataFrame mit den ursprünglichen Spalten
        result = pd.DataFrame()
        
        col_idx = 0
        for col, col_type in self.column_types.items():
            if col_type == 'categorical':
                # Anzahl der eindeutigen Werte für diese kategorische Spalte
                n_categories = len(self.categorical_mappings[col])
                
                # Extrahiere die One-Hot-kodierten Werte
                cat_values = rescaled_data[:, col_idx:col_idx+n_categories]
                
                # Konvertiere von One-Hot zurück zu kategorischen Werten
                # Nehme die Kategorie mit dem höchsten Wert
                cat_indices = np.argmax(cat_values, axis=1)
                
                # Mappe zurück zu den ursprünglichen Kategorien
                result[col] = [self.categorical_mappings[f"{col}_reverse"][idx] for idx in cat_indices]
                
                col_idx += n_categories
            else:
                # Kontinuierliche Spalte einfach übernehmen
                result[col] = rescaled_data[:, col_idx]
                col_idx += 1
        
        return result
    
    def _build_generator(self, output_dim):
        """Erstellt den Generator für das GAN"""
        model = keras.Sequential([
            layers.Dense(256, input_dim=self.noise_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(output_dim, activation='tanh')  # tanh für Output im Bereich [-1, 1]
        ])
        return model
    
    def _build_discriminator(self, input_dim):
        """Erstellt den Diskriminator für das GAN"""
        model = keras.Sequential([
            layers.Dense(512, input_dim=input_dim, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy')
        return model
    
    def _build_gan(self, generator, discriminator):
        """Kombiniert Generator und Diskriminator zum GAN"""
        discriminator.trainable = False  # Diskriminator beim GAN-Training nicht aktualisieren
        
        model = keras.Sequential([
            generator,
            discriminator
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy')
        return model
    
    def fit(self, data: pd.DataFrame, epochs: int = 2000, batch_size: int = 32, 
           sample_interval: int = 100, verbose: int = 1):
        """
        Trainiert das GAN-Modell mit den gegebenen Daten.
        
        Args:
            data: Eingabedaten (DataFrame)
            epochs: Anzahl der Trainings-Epochen
            batch_size: Batch-Größe für das Training
            sample_interval: Intervall für Stichproben der generierten Daten
            verbose: Ausgabedetailstufe (0, 1, oder 2)
            
        Returns:
            self: Trainiertes Modell
        """
        try:
            # Identifiziere Spaltentypen
            self._identify_column_types(data)
            
            # Vorverarbeitung der Daten
            processed_data = self._preprocess_data(data)
            self.feature_dims = processed_data.shape[1]
            
            # Speichere trainierte Daten für spätere Validierung
            self.training_data = data.copy()
            
            # Baue das GAN-Modell
            self.generator = self._build_generator(self.feature_dims)
            self.discriminator = self._build_discriminator(self.feature_dims)
            self.gan = self._build_gan(self.generator, self.discriminator)
            
            # Label für echte und gefälschte Daten
            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            
            # Trainingsschleife
            for epoch in range(epochs):
                # ---------------------
                #  Trainiere Diskriminator
                # ---------------------
                
                # Wähle eine zufällige Batch aus echten Daten
                idx = np.random.randint(0, processed_data.shape[0], batch_size)
                real_data = processed_data[idx]
                
                # Generiere eine Batch aus gefälschten Daten
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                fake_data = self.generator.predict(noise)
                
                # Trainiere den Diskriminator
                d_loss_real = self.discriminator.train_on_batch(real_data, real)
                d_loss_fake = self.discriminator.train_on_batch(fake_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ---------------------
                #  Trainiere Generator
                # ---------------------
                
                # Generiere neue Batch aus Rauschen
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                
                # Trainiere den Generator
                g_loss = self.gan.train_on_batch(noise, real)
                
                # Ausgabe für Fortschrittsüberwachung
                if verbose > 0 and epoch % sample_interval == 0:
                    print(f"Epoch {epoch}/{epochs} [D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}]")
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des GAN-Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def generate(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generiert synthetische Daten.
        
        Args:
            n_samples: Anzahl der zu generierenden Datensätze
            
        Returns:
            DataFrame mit synthetischen Daten im Format der Trainingsdaten
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere Rauschen als Input für den Generator
            noise = np.random.normal(0, 1, (n_samples, self.noise_dim))
            
            # Generiere Daten
            generated_data = self.generator.predict(noise)
            
            # Nachverarbeitung der Daten
            synthetic_data = self._postprocess_data(generated_data)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Fehler bei der Datengenerierung: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate_quality(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Bewertet die Qualität der generierten Daten durch Vergleich mit den Trainingsdaten.
        
        Args:
            n_samples: Anzahl der zu generierenden und bewertenden Datensätze
            
        Returns:
            Dictionary mit Qualitätsmetriken
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere synthetische Daten
            synthetic_data = self.generate(n_samples)
            
            # Statistischer Vergleich zwischen echten und synthetischen Daten
            metrics = {}
            
            # Vergleiche Mittelwerte und Standardabweichungen für kontinuierliche Spalten
            for col, col_type in self.column_types.items():
                if col_type == 'continuous':
                    # Berechne Mittelwert und Standardabweichung für echte Daten
                    real_mean = self.training_data[col].mean()
                    real_std = self.training_data[col].std()
                    
                    # Berechne dieselben Statistiken für synthetische Daten
                    synth_mean = synthetic_data[col].mean()
                    synth_std = synthetic_data[col].std()
                    
                    # Berechne die relative Differenz
                    mean_diff = abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10)
                    std_diff = abs(real_std - synth_std) / (abs(real_std) + 1e-10)
                    
                    metrics[f"{col}_mean_diff"] = float(mean_diff)
                    metrics[f"{col}_std_diff"] = float(std_diff)
                else:
                    # Vergleiche die Verteilung kategorischer Werte
                    real_dist = self.training_data[col].value_counts(normalize=True)
                    synth_dist = synthetic_data[col].value_counts(normalize=True)
                    
                    # Berechne die Jensen-Shannon-Divergenz
                    # (symmetrische Version der KL-Divergenz)
                    js_divergence = 0.0
                    
                    # Stelle sicher, dass beide Verteilungen dieselben Kategorien haben
                    all_categories = set(real_dist.index) | set(synth_dist.index)
                    
                    for cat in all_categories:
                        p = real_dist.get(cat, 0)
                        q = synth_dist.get(cat, 0)
                        
                        # Vermeide Logarithmus von 0
                        if p > 0 and q > 0:
                            m = 0.5 * (p + q)
                            js_divergence += 0.5 * (p * np.log(p / m) + q * np.log(q / m))
                    
                    metrics[f"{col}_js_divergence"] = float(js_divergence)
            
            # Gesamtqualitätsmetrik
            # Durchschnitt der normalisierten Abweichungen (niedriger ist besser)
            continuous_diffs = [v for k, v in metrics.items() if k.endswith('_diff')]
            categorical_diffs = [v for k, v in metrics.items() if k.endswith('_js_divergence')]
            
            if continuous_diffs:
                metrics['continuous_avg_diff'] = float(np.mean(continuous_diffs))
            if categorical_diffs:
                metrics['categorical_avg_diff'] = float(np.mean(categorical_diffs))
            
            # Gesamtbewertung (0 bis 1, höher ist besser)
            overall_score = 1.0
            if continuous_diffs:
                overall_score -= 0.5 * min(1.0, np.mean(continuous_diffs))
            if categorical_diffs:
                overall_score -= 0.5 * min(1.0, np.mean(categorical_diffs))
            
            metrics['overall_quality_score'] = float(overall_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Fehler bei der Qualitätsbewertung: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def plot_comparison(self, n_samples: int = 1000, 
                       features: List[str] = None,
                       save_path: str = None) -> plt.Figure:
        """
        Visualisiert einen Vergleich zwischen echten und synthetischen Daten.
        
        Args:
            n_samples: Anzahl der zu generierenden Datensätze
            features: Liste der darzustellenden Features (Standard: alle)
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere synthetische Daten
            synthetic_data = self.generate(n_samples)
            
            # Wähle die darzustellenden Features aus
            if features is None:
                # Wähle bis zu 6 Features für die Visualisierung
                features = list(self.column_types.keys())[:min(6, len(self.column_types))]
            
            # Bestimme die Anzahl der Zeilen und Spalten für das Subplot-Raster
            n_features = len(features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            # Erstelle die Figur
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
            if n_rows * n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, feature in enumerate(features):
                ax = axes[i]
                
                if self.column_types[feature] == 'continuous':
                    # Histogramm für kontinuierliche Variablen
                    sns.histplot(self.training_data[feature], kde=True, ax=ax, color='blue', alpha=0.5, label='Echte Daten')
                    sns.histplot(synthetic_data[feature], kde=True, ax=ax, color='red', alpha=0.5, label='Synthetische Daten')
                else:
                    # Balkendiagramm für kategorische Variablen
                    real_counts = self.training_data[feature].value_counts(normalize=True)
                    synth_counts = synthetic_data[feature].value_counts(normalize=True)
                    
                    # Kombiniere beide, um alle Kategorien zu erfassen
                    all_cats = sorted(set(real_counts.index) | set(synth_counts.index))
                    
                    # Erstelle ein DataFrame für Seaborn
                    plot_data = []
                    for cat in all_cats:
                        plot_data.append({'Category': cat, 'Frequency': real_counts.get(cat, 0), 'Type': 'Real'})
                        plot_data.append({'Category': cat, 'Frequency': synth_counts.get(cat, 0), 'Type': 'Synthetic'})
                    
                    plot_df = pd.DataFrame(plot_data)
                    
                    # Balkendiagramm
                    sns.barplot(x='Category', y='Frequency', hue='Type', data=plot_df, ax=ax)
                
                ax.set_title(f'Verteilung von {feature}')
                ax.legend()
                
                # Achsen anpassen
                if self.column_types[feature] == 'categorical':
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Verstecke ungenutzte Subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung des Datenvergleichs: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        model_data = {
            "categorical_threshold": self.categorical_threshold,
            "noise_dim": self.noise_dim,
            "feature_dims": self.feature_dims,
            "column_types": self.column_types,
            "categorical_mappings": self.categorical_mappings,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere die Modelle
        self.generator.save(f"{path}_generator")
        self.discriminator.save(f"{path}_discriminator")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten und Mappings
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        synthesizer = cls(
            categorical_threshold=model_data['categorical_threshold'],
            noise_dim=model_data['noise_dim']
        )
        
        synthesizer.feature_dims = model_data['feature_dims']
        synthesizer.column_types = model_data['column_types']
        synthesizer.categorical_mappings = model_data['categorical_mappings']
        synthesizer.is_fitted = model_data['is_fitted']
        
        # Lade die Modelle
        synthesizer.generator = keras.models.load_model(f"{path}_generator")
        synthesizer.discriminator = keras.models.load_model(f"{path}_discriminator")
        
        # Lade den Scaler
        synthesizer.scaler = joblib.load(f"{path}_scaler.joblib")
        
        # Rekonstruiere das GAN
        synthesizer.gan = synthesizer._build_gan(synthesizer.generator, synthesizer.discriminator)
        
        return synthesizer
