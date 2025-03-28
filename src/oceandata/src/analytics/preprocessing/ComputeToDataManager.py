###########################################
# 5. Federated Learning und Compute-to-Data
###########################################

class ComputeToDataManager:
    """
    Klasse zur Verwaltung von Compute-to-Data-Operationen für den Datenschutz.
    Ermöglicht die Ausführung von Berechnungen auf sensiblen Daten ohne direkten Zugriff.
    """
    
    def __init__(self, encryption_key: bytes = None):
        """
        Initialisiert den Compute-to-Data-Manager.
        
        Args:
            encryption_key: Schlüssel für die Verschlüsselung (optional)
        """
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.allowed_operations = {
            'aggregate': self._aggregate,
            'count': self._count,
            'mean': self._mean,
            'sum': self._sum,
            'min': self._min,
            'max': self._max,
            'std': self._std,
            'correlation': self._correlation,
            'histogram': self._histogram,
            'custom_model': self._custom_model_inference
        }
        
        # Datenschutzkonfiguration
        self.privacy_config = {
            'min_group_size': 5,  # Minimale Gruppengröße für Aggregationen
            'noise_level': 0.01,  # Standardrauschen für Differentialprivatsphäre
            'outlier_removal': True  # Entfernung von Ausreißern für sensible Berechnungen
        }
    
    def set_privacy_config(self, min_group_size: int = None, noise_level: float = None, 
                          outlier_removal: bool = None):
        """Aktualisiert die Datenschutzkonfiguration"""
        if min_group_size is not None:
            self.privacy_config['min_group_size'] = min_group_size
        if noise_level is not None:
            self.privacy_config['noise_level'] = noise_level
        if outlier_removal is not None:
            self.privacy_config['outlier_removal'] = outlier_removal
    
    def _encrypt_data(self, data: pd.DataFrame) -> bytes:
        """Verschlüsselt Daten für die sichere Speicherung"""
        serialized = data.to_json().encode()
        return self.cipher_suite.encrypt(serialized)
    
    def _decrypt_data(self, encrypted_data: bytes) -> pd.DataFrame:
        """Entschlüsselt Daten für die Verarbeitung"""
        decrypted = self.cipher_suite.decrypt(encrypted_data)
        return pd.read_json(decrypted.decode())
    
    def _add_differential_privacy_noise(self, result, scale: float = None):
        """Fügt Rauschen für Differential Privacy hinzu"""
        scale = scale or self.privacy_config['noise_level']
        
        if isinstance(result, (int, float)):
            return result + np.random.normal(0, scale * abs(result))
        elif isinstance(result, np.ndarray):
            return result + np.random.normal(0, scale * np.abs(result).mean(), result.shape)
        elif isinstance(result, pd.Series):
            noisy_values = result.values + np.random.normal(0, scale * np.abs(result.values).mean(), result.shape)
            return pd.Series(noisy_values, index=result.index)
        elif isinstance(result, pd.DataFrame):
            noisy_values = result.values + np.random.normal(0, scale * np.abs(result.values).mean(), result.shape)
            return pd.DataFrame(noisy_values, index=result.index, columns=result.columns)
        else:
            return result
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Entfernt Ausreißer aus den Daten"""
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            return data
        
        # Z-Score-basierte Ausreißererkennung
        z_scores = ((numeric_data - numeric_data.mean()) / numeric_data.std()).abs()
        mask = (z_scores < 3).all(axis=1)
        
        return data[mask]
    
    def _validate_group_size(self, data: pd.DataFrame, group_column: str = None, 
                            group_value: Any = None) -> bool:
        """
        Prüft, ob eine Gruppe die Mindestgröße für Datenschutz erfüllt
        
        Args:
            data: Daten
            group_column: Spalte für die Gruppierung
            group_value: Wert für die Filterung
            
        Returns:
            True, wenn die Gruppe groß genug ist
        """
        min_size = self.privacy_config['min_group_size']
        
        if group_column is not None and group_value is not None:
            group_data = data[data[group_column] == group_value]
            return len(group_data) >= min_size
        
        return len(data) >= min_size
    
    def _prepare_data_for_computation(self, data: pd.DataFrame, columns: List[str] = None,
                                    group_column: str = None, group_value: Any = None) -> pd.DataFrame:
        """
        Bereitet Daten für die Berechnung vor, einschließlich Filterung und Datenschutzprüfungen
        
        Args:
            data: Eingabedaten
            columns: Zu verwendende Spalten
            group_column: Spalte für die Gruppierung
            group_value: Wert für die Filterung
            
        Returns:
            Vorbereitete Daten
        """
        # Filtere Daten nach Gruppe, falls angegeben
        if group_column is not None and group_value is not None:
            prepared_data = data[data[group_column] == group_value]
        else:
            prepared_data = data.copy()
        
        # Prüfe Mindestgruppengröße
        if not self._validate_group_size(prepared_data):
            raise ValueError(f"Gruppe zu klein für Datenschutz (min: {self.privacy_config['min_group_size']})")
        
        # Entferne Ausreißer, falls konfiguriert
        if self.privacy_config['outlier_removal']:
            prepared_data = self._remove_outliers(prepared_data)
        
        # Wähle Spalten aus, falls angegeben
        if columns is not None:
            prepared_data = prepared_data[columns]
        
        return prepared_data
    
    # Implementierte Operationen für C2D
    
    def _aggregate(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Aggregiert Daten nach Spalten und Gruppen"""
        columns = params.get('columns')
        group_by = params.get('group_by')
        aggregations = params.get('aggregations', ['count', 'mean'])
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        if group_by:
            grouped = prepared_data.groupby(group_by)
            
            # Prüfe Mindestgröße für jede Gruppe
            group_sizes = grouped.size()
            valid_groups = group_sizes[group_sizes >= self.privacy_config['min_group_size']].index
            
            # Filtere ungültige Gruppen
            results = {}
            for agg in aggregations:
                if agg == 'count':
                    agg_result = grouped.size()
                else:
                    agg_func = getattr(grouped, agg, None)
                    if agg_func is None:
                        continue
                    agg_result = agg_func()
                
                # Behalte nur valide Gruppen
                agg_result = agg_result.loc[valid_groups]
                
                # Füge Rauschen hinzu
                results[agg] = self._add_differential_privacy_noise(agg_result)
        else:
            results = {}
            for agg in aggregations:
                if agg == 'count':
                    agg_result = len(prepared_data)
                else:
                    agg_func = getattr(prepared_data, agg, None)
                    if agg_func is None:
                        continue
                    agg_result = agg_func()
                
                # Füge Rauschen hinzu
                results[agg] = self._add_differential_privacy_noise(agg_result)
        
        # Konvertiere zu serialisierbarem Format
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
                serializable_results[k] = v.to_dict()
            else:
                serializable_results[k] = v
        
        return serializable_results
    
    def _count(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Zählt Datensätze nach Filterkriterien"""
        filter_column = params.get('filter_column')
        filter_value = params.get('filter_value')
        
        if filter_column and filter_value is not None:
            count = len(data[data[filter_column] == filter_value])
        else:
            count = len(data)
        
        # Füge Rauschen hinzu und runde auf ganze Zahlen
        noisy_count = round(self._add_differential_privacy_noise(count))
        
        # Stelle sicher, dass der Wert nicht negativ ist
        return {'count': max(0, noisy_count)}
    
    def _mean(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet den Mittelwert für numerische Spalten"""
        columns = params.get('columns')
        group_column = params.get('group_column')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        if group_column:
            grouped = prepared_data.groupby(group_column)
            
            # Prüfe Mindestgröße für jede Gruppe
            group_sizes = grouped.size()
            valid_groups = group_sizes[group_sizes >= self.privacy_config['min

          'data_uniqueness': {
                    'score': 0.0,
                    'weight': 0.2,
                    'explanation': ''
                },
                'data_applicability': {
                    'score': 0.0,
                    'weight': 0.25,
                    'explanation': ''
                },
                'time_relevance': {
                    'score': 0.0,
                    'weight': 0.1,
                    'explanation': ''
                },
                'enrichment_potential': {
                    'score': 0.0,
                    'weight': 0.1,
                    'explanation': ''
                }
            }
            
            # 1. Datenvolumen
            # Bewerte die Größe des Datensatzes (Zeilen und Spalten)
            rows_score = min(1.0, len(data) / 10000)  # Skaliert bis 10.000 Zeilen
            cols_score = min(1.0, len(data.columns) / 50)  # Skaliert bis 50 Spalten
            volume_score = (rows_score * 0.7) + (cols_score * 0.3)  # Gewichtung: Zeilen wichtiger als Spalten
            
            value_metrics['data_volume']['score'] = volume_score
            value_metrics['data_volume']['explanation'] = f"Dataset contains {len(data)} records and {len(data.columns)} fields. "
            
            if volume_score < 0.3:
                value_metrics['data_volume']['explanation'] += "This is a relatively small dataset, which limits its potential value."
            elif volume_score < 0.7:
                value_metrics['data_volume']['explanation'] += "This is a medium-sized dataset with solid commercial value potential."
            else:
                value_metrics['data_volume']['explanation'] += "This is a large dataset with significant commercial value potential."
            
            # 2. Datenqualität
            # Bewerte die Qualität anhand von fehlenden Werten, Varianz und Konsistenz
            
            # Berechne fehlende Werte
            missing_ratio = data.isna().mean().mean()
            completeness = 1 - missing_ratio
            
            # Berechne Varianz für numerische Spalten
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                normalized_variance = 0
                for col in numeric_cols:
                    # Normalisierte Varianz (0-1)
                    if data[col].std() > 0:
                        normalized_variance += min(1.0, data[col].std() / data[col].mean() if data[col].mean() != 0 else 0)
                
                avg_normalized_variance = normalized_variance / len(numeric_cols)
                variance_score = avg_normalized_variance
            else:
                variance_score = 0.5  # Mittlerer Wert, wenn keine numerischen Spalten vorhanden sind
            
            # Berechne Eindeutigkeit für kategorische Spalten
            categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
            if categorical_cols:
                uniqueness_ratio = 0
                for col in categorical_cols:
                    uniqueness_ratio += data[col].nunique() / len(data)
                
                avg_uniqueness_ratio = uniqueness_ratio / len(categorical_cols)
                uniqueness_score = min(1.0, avg_uniqueness_ratio * 3)  # Skaliert, um niedrigere Werte aufzuwerten
            else:
                uniqueness_score = 0.5  # Mittlerer Wert, wenn keine kategorischen Spalten vorhanden sind
            
            # Kombiniere die Qualitätsmetriken
            quality_score = (completeness * 0.5) + (variance_score * 0.25) + (uniqueness_score * 0.25)
            
            value_metrics['data_quality']['score'] = quality_score
            value_metrics['data_quality']['explanation'] = f"Data completeness: {completeness:.1%}. "
            
            if completeness < 0.8:
                value_metrics['data_quality']['explanation'] += "Missing data impacts overall quality. "
            
            value_metrics['data_quality']['explanation'] += f"Data variability: {variance_score:.1%}. "
            if variance_score < 0.3:
                value_metrics['data_quality']['explanation'] += "Low variability may indicate limited information content. "
            elif variance_score > 0.7:
                value_metrics['data_quality']['explanation'] += "High variability indicates rich information content. "
            
            # 3. Dateneinzigartigkeit
            # Bewerte die Einzigartigkeit basierend auf Spaltentypen und Datenbereichen
            
            # Analysiere Spaltentypen
            column_types = {
                'datetime': len(data.select_dtypes(include=['datetime']).columns),
                'numeric': len(data.select_dtypes(include=['number']).columns),
                'categorical': len(data.select_dtypes(include=['category', 'object']).columns),
                'boolean': len(data.select_dtypes(include=['bool']).columns),
                'text': sum(1 for col in data.select_dtypes(include=['object']).columns 
                          if data[col].astype(str).str.len().mean() > 20)
            }
            
            # Diversitätspunkte für Spaltentypen-Mix
            diversity_score = min(1.0, sum(1 for count in column_types.values() if count > 0) / 5)
            
            # Zusätzliche Punkte für seltene oder wertvolle Datentypen
            value_column_score = 0
            if column_types['datetime'] > 0:
                value_column_score += 0.2  # Zeitreihen sind wertvoll
            if column_types['text'] > 0:
                value_column_score += 0.3  # Textdaten sind wertvoll für NLP
            if column_types['boolean'] > 0 and column_types['numeric'] > 0:
                value_column_score += 0.1  # Gute Kombination für ML
            
            # Quelltyp-spezifischer Wert (wenn vorhanden)
            source_type = metadata.get('source_type', '')
            source_type_scores = {
                'health_data': 0.9,
                'health_insurance': 0.9,
                'smartwatch': 0.8,
                'social_media': 0.7,
                'browser': 0.6,
                'chat': 0.7,
                'streaming': 0.7,
                'calendar': 0.5,
                'iot_security': 0.6,
                'smart_home': 0.5
            }
            source_score = source_type_scores.get(source_type, 0.5)
            
            # Kombiniere Einzigartigkeit und Wert
            uniqueness_score = (diversity_score * 0.3) + (value_column_score * 0.3) + (source_score * 0.4)
            
            value_metrics['data_uniqueness']['score'] = uniqueness_score
            value_metrics['data_uniqueness']['explanation'] = f"Data contains {sum(column_types.values())} columns across {sum(1 for count in column_types.values() if count > 0)} different data types. "
            
            if source_type:
                value_metrics['data_uniqueness']['explanation'] += f"Source type '{source_type}' "
                if source_score > 0.7:
                    value_metrics['data_uniqueness']['explanation'] += "is particularly valuable in the data marketplace. "
                elif source_score > 0.5:
                    value_metrics['data_uniqueness']['explanation'] += "has good value in the data marketplace. "
                else:
                    value_metrics['data_uniqueness']['explanation'] += "has standard value in the data marketplace. "
            
            if column_types['text'] > 0:
                value_metrics['data_uniqueness']['explanation'] += f"Contains {column_types['text']} text columns which add significant value for NLP applications. "
            
            if column_types['datetime'] > 0:
                value_metrics['data_uniqueness']['explanation'] += f"Contains {column_types['datetime']} datetime columns enabling time-series analysis. "
            
            # 4. Anwendbarkeit der Daten
            # Bewerte potenzielle Anwendungsfälle basierend auf Datentyp und Marktanforderungen
            
            # Identifiziere potenzielle Anwendungsfälle basierend auf verfügbaren Daten
            application_scenarios = []
            
            # Prüfe auf typische Anwendungsfälle basierend auf Spaltenkombinationen
            has_location_data = any('location' in col.lower() or 'address' in col.lower() or 'geo' in col.lower() for col in data.columns)
            has_user_data = any('user' in col.lower() or 'customer' in col.lower() or 'id' in col.lower() for col in data.columns)
            has_time_data = column_types['datetime'] > 0
            has_numeric_features = column_types['numeric'] > 2  # Mindestens 3 numerische Spalten
            has_categorical_features = column_types['categorical'] > 0
            has_text_features = column_types['text'] > 0
            
            # Bewerte Anwendbarkeit für Machine Learning
            if has_numeric_features and (has_categorical_features or has_time_data):
                application_scenarios.append({
                    'name': 'Predictive Modeling',
                    'score': 0.8,
                    'description': 'Suitable for building predictive models using machine learning'
                })
            
            # Bewerte Anwendbarkeit für Zeitreihenanalyse
            if has_time_data and has_numeric_features:
                application_scenarios.append({
                    'name': 'Time Series Analysis',
                    'score': 0.9,
                    'description': 'Ideal for trend analysis and forecasting over time'
                })
            
            # Bewerte Anwendbarkeit für Personalisierung
            if has_user_data and (has_categorical_features or has_text_features):
                application_scenarios.append({
                    'name': 'Personalization & Recommendations',
                    'score': 0.85,
                    'description': 'Can be used for user preference modeling and recommendations'
                })
            
            # Bewerte Anwendbarkeit für geografische Analysen
            if has_location_data:
                application_scenarios.append({
                    'name': 'Geospatial Analysis',
                    'score': 0.75,
                    'description': 'Enables location-based insights and targeting'
                })
            
            # Bewerte Anwendbarkeit für NLP
            if has_text_features:
                application_scenarios.append({
                    'name': 'Natural Language Processing',
                    'score': 0.85,
                    'description': 'Text data can be used for sentiment analysis, topic modeling, etc.'
                })
            
            # Bewerte Anwendbarkeit für Clusteranalyse
            if has_numeric_features >= 3:
                application_scenarios.append({
                    'name': 'Clustering & Segmentation',
                    'score': 0.7,
                    'description': 'Can be used to identify natural groupings and segments'
                })
            
            # Berechne den Anwendbarkeitswert basierend auf den identifizierten Szenarien
            if application_scenarios:
                # Durchschnittliche Bewertung über alle Szenarien
                applicability_score = sum(scenario['score'] for scenario in application_scenarios) / len(application_scenarios)
                
                # Bonus für die Anzahl der Szenarien (mehr Szenarien = vielseitigere Daten)
                scenario_count_bonus = min(0.2, (len(application_scenarios) - 1) * 0.05)
                applicability_score += scenario_count_bonus
                
                # Begrenze auf Maximum von 1.0
                applicability_score = min(1.0, applicability_score)
            else:
                # Standard-Score, wenn keine spezifischen Szenarien identifiziert wurden
                applicability_score = 0.3
            
            value_metrics['data_applicability']['score'] = applicability_score
            value_metrics['data_applicability']['explanation'] = f"Identified {len(application_scenarios)} potential application scenarios. "
            
            if application_scenarios:
                top_scenarios = sorted(application_scenarios, key=lambda x: x['score'], reverse=True)[:3]
                value_metrics['data_applicability']['explanation'] += "Top applications: " + ", ".join(s['name'] for s in top_scenarios) + ". "
            
            if applicability_score > 0.7:
                value_metrics['data_applicability']['explanation'] += "Data has high versatility across multiple use cases. "
            elif applicability_score > 0.4:
                value_metrics['data_applicability']['explanation'] += "Data has moderate versatility for specific applications. "
            else:
                value_metrics['data_applicability']['explanation'] += "Data has limited application potential without additional enrichment. "
            
            # 5. Zeitliche Relevanz
            # Bewerte die Aktualität und zeitliche Spanne der Daten
            
            time_relevance_score = 0.5  # Mittlerer Standardwert
            
            # Prüfe auf zeitbezogene Metadaten
            time_range = metadata.get('time_range', {})
            if time_range:
                # Aktualität: Wie nah ist das neueste Datum an heute?
                end_date = time_range.get('end')
                if end_date:
                    try:
                        # Konvertiere ISO-String zu Datum
                        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        days_since_end = (datetime.now() - end_date).days
                        
                        # Berechne Aktualitätspunkte (höher für neuere Daten)
                        recency_score = max(0, 1 - (days_since_end / 365))  # Skaliert über ein Jahr
                    except:
                        recency_score = 0.5  # Standardwert bei Parsing-Fehler
                else:
                    recency_score = 0.5
                
                # Zeitspanne: Längere Zeitreihen sind wertvoller
                duration_days = time_range.get('duration_days', 0)
                if duration_days:
                    # Berechne Punktzahl für Zeitspanne (höher für längere Zeiträume)
                    span_score = min(1.0, duration_days / 365)  # Skaliert bis zu einem Jahr
                else:
                    span_score = 0.5
                
                # Kombiniere Aktualität und Zeitspanne
                time_relevance_score = (recency_score * 0.6) + (span_score * 0.4)  # Aktualität wichtiger als Spanne
            
            value_metrics['time_relevance']['score'] = time_relevance_score
            value_metrics['time_relevance']['explanation'] = ""
            
            if time_range:
                value_metrics['time_relevance']['explanation'] += f"Data spans {time_range.get('duration_days', '?')} days. "
                
                if time_relevance_score > 0.7:
                    value_metrics['time_relevance']['explanation'] += "Data is recent and covers a substantial time period. "
                elif time_relevance_score > 0.4:
                    value_metrics['time_relevance']['explanation'] += "Data has moderate temporal relevance. "
                else:
                    value_metrics['time_relevance']['explanation'] += "Data may be outdated or covers a limited time period. "
            else:
                value_metrics['time_relevance']['explanation'] += "No temporal information available to assess time relevance. "
            
            # 6. Anreicherungspotenzial
            # Bewerte, wie gut sich die Daten mit anderen Quellen kombinieren lassen
            
            # Identifiziere potenzielle Verknüpfungsfelder
            linkage_fields = []
            
            # Typische ID-Felder
            id_fields = [col for col in data.columns if 'id' in col.lower() or 'key' in col.lower()]
            if id_fields:
                linkage_fields.extend(id_fields)
            
            # Standortfelder für geografische Verknüpfung
            location_fields = [col for col in data.columns 
                              if 'location' in col.lower() or 'address' in col.lower() 
                              or 'city' in col.lower() or 'country' in col.lower()]
            if location_fields:
                linkage_fields.extend(location_fields)
            
            # Zeitfelder für zeitliche Verknüpfung
            time_fields = list(data.select_dtypes(include=['datetime']).columns)
            if time_fields:
                linkage_fields.extend(time_fields)
            
            # Kategorienwerte für kontextuelle Verknüpfung
            category_fields = [col for col in data.columns 
                              if data[col].dtype == 'object' and data[col].nunique() < 50]
            if category_fields:
                linkage_fields.extend(category_fields[:3])  # Beschränke auf die ersten drei
            
            # Einzigartige Linkage-Felder
            unique_linkage_fields = list(set(linkage_fields))
            
            # Berechne den Anreicherungswert basierend auf Linkage-Feldern
            if unique_linkage_fields:
                # Mehr Verknüpfungsfelder = höheres Anreicherungspotenzial
                field_count_score = min(1.0, len(unique_linkage_fields) / 5)  # Skaliert bis zu 5 Felder
                
                # Zusätzliche Punkte für besonders wertvolle Verknüpfungstypen
                id_bonus = 0.3 if any('id' in field.lower() for field in unique_linkage_fields) else 0
                location_bonus = 0.2 if any(field in location_fields for field in unique_linkage_fields) else 0
                time_bonus = 0.2 if any(field in time_fields for field in unique_linkage_fields) else 0
                
                # Berechne Gesamtwert mit Begrenzung auf 1.0
                enrichment_score = min(1.0, field_count_score + id_bonus + location_bonus + time_bonus)
            else:
                # Geringer Wert für Daten ohne offensichtliche Verknüpfungsmöglichkeiten
                enrichment_score = 0.2
            
            value_metrics['enrichment_potential']['score'] = enrichment_score
            value_metrics['enrichment_potential']['explanation'] = f"Identified {len(unique_linkage_fields)} potential linkage fields. "
            
            if unique_linkage_fields:
                field_types = []
                if any('id' in field.lower() for field in unique_linkage_fields):
                    field_types.append("IDs")
                if any(field in location_fields for field in unique_linkage_fields):
                    field_types.append("location data")
                if any(field in time_fields for field in unique_linkage_fields):
                    field_types.append("temporal markers")
                if any(field in category_fields for field in unique_linkage_fields):
                    field_types.append("categorical attributes")
                
                value_metrics['enrichment_potential']['explanation'] += "Linkage opportunities include: " + ", ".join(field_types) + ". "
            
            if enrichment_score > 0.7:
                value_metrics['enrichment_potential']['explanation'] += "Data has excellent potential for enrichment with other datasets. "
            elif enrichment_score > 0.4:
                value_metrics['enrichment_potential']['explanation'] += "Data has moderate potential for combination with complementary sources. "
            else:
                value_metrics['enrichment_potential']['explanation'] += "Data has limited enrichment potential without additional identifiers. "
            
            # Berechne den Gesamtwert als gewichtete Summe aller Faktoren
            total_weighted_score = sum(metric['score'] * metric['weight'] for metric in value_metrics.values())
            
            # Konvertiere den normalisierten Score in einen monetären Wert
            # Basierend auf einer konfigurierbaren Wertskala, z.B. 0-10 OCEAN Tokens
            min_value = self.config.get('min_token_value', 1.0)
            max_value = self.config.get('max_token_value', 10.0)
            estimated_token_value = min_value + (total_weighted_score * (max_value - min_value))
            
            # Erstelle die Wertzusammenfassung
            value_summary = {
                'normalized_score': float(total_weighted_score),
                'estimated_token_value': float(estimated_token_value),
                'metrics': value_metrics,
                'summary': f"This dataset has an estimated value of {estimated_token_value:.2f} OCEAN tokens, "
                          f"with its primary strengths being {', '.join(k for k, v in sorted([(k, v['score']) for k, v in value_metrics.items()], key=lambda x: x[1], reverse=True)[:2])}."
            }
            
            return value_summary
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenwertschätzung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'normalized_score': 0.3,  # Standardwert im Fehlerfall
                'estimated_token_value': self.config.get('min_token_value', 1.0),
                'metrics': {k: {'score': 0.3, 'weight': v['weight'], 'explanation': 'Error in calculation'} 
                           for k, v in value_metrics.items()}
            }

    def prepare_for_ocean_tokenization(self, data_asset: Dict) -> Dict:
        """
        Bereitet einen Daten-Asset für die Tokenisierung mit Ocean Protocol vor.
        
        Args:
            data_asset: Der vorzubereitende Daten-Asset mit Metadaten und C2D-Asset
            
        Returns:
            Für Ocean Protocol formatierte Asset-Informationen
        """
        try:
            self.logger.info(f"Bereite Asset für Ocean Protocol Tokenisierung vor")
            
            if 'error' in data_asset:
                raise ValueError(f"Fehlerhafter Daten-Asset: {data_asset['error']}")
            
            if 'c2d_asset' not in data_asset or 'metadata' not in data_asset:
                raise ValueError("Asset enthält nicht die erforderlichen Informationen (c2d_asset oder metadata)")
            
            # Extrahiere relevante Informationen
            asset_id = data_asset['c2d_asset']['asset_id']
            metadata = data_asset['metadata']
            
            # Erstelle ein DDO (Decentralized Data Object) im Ocean-Format
            name = metadata.get('name', f"Dataset {asset_id[:8]}")
            if 'source_type' in metadata:
                name = f"{metadata['source_type'].replace('_', ' ').title()} Dataset {asset_id[:8]}"
            
            description = metadata.get('description', '')
            if not description and 'potential_use_cases' in metadata and metadata['potential_use_cases']:
                # Verwende die Beschreibung des ersten Anwendungsfalls
                use_case = metadata['potential_use_cases'][0]
                description = f"{use_case.get('title', '')}: {use_case.get('description', '')}"
            
            # Kategorien basierend auf dem Quelltyp
            categories = []
            if 'source_type' in metadata:
                source_type = metadata['source_type']
                
                # Mapping von Quelltypen zu Ocean-Kategorien
                category_mapping = {
                    'browser': ['Web', 'Consumer Behavior'],
                    'calendar': ['Productivity', 'Time Management'],
                    'chat': ['Communication', 'Social'],
                    'social_media': ['Social', 'Marketing'],
                    'streaming': ['Entertainment', 'Media'],
                    'health_insurance': ['Healthcare', 'Insurance'],
                    'health_data': ['Healthcare', 'Wellness'],
                    'smartwatch': ['IoT', 'Wearables', 'Fitness'],
                    'iot_vacuum': ['IoT', 'Smart Home'],
                    'iot_thermostat': ['IoT', 'Energy'],
                    'iot_lighting': ['IoT', 'Smart Home'],
                    'iot_security': ['IoT', 'Security'],
                    'smart_home': ['IoT', 'Smart Home']
                }
                categories = category_mapping.get(source_type, ['Other'])
            
            # Erstelle Tags basierend auf Metadaten und Spalteninformationen
            tags = []
            
            # Tags basierend auf der Datenquelle
            if 'source_type' in metadata:
                tags.append(metadata['source_type'])
            
            # Tags basierend auf Spaltenstatistiken
            if 'column_statistics' in metadata:
                column_types = [stats.get('type') for stats in metadata['column_statistics'].values()]
                if 'datetime' in column_types or 'timestamp' in column_types:
                    tags.append('time-series')
                if 'text' in column_types:
                    tags.append('text-data')
                if 'categorical' in column_types:
                    tags.append('categorical-data')
                if any(t == 'numeric' for t in column_types):
                    tags.append('numeric-data')
                if any('location' in col.lower() for col in metadata.get('column_statistics', {}).keys()):
                    tags.append('geospatial')
            
            # Datenschutztags basierend auf dem Privacy-Level
            if 'privacy_level' in metadata:
                tags.append(f"privacy-{metadata['privacy_level']}")
            
            # Füge eine zeitliche Dimension hinzu, falls vorhanden
            if 'time_range' in metadata and metadata['time_range'] and 'duration_days' in metadata['time_range']:
                days = metadata['time_range']['duration_days']
                if days <= 7:
                    tags.append('short-term')
                elif days <= 90:
                    tags.append('medium-term')
                else:
                    tags.append('long-term')
            
            # Erstelle das DDO
            ddo = {
                'id': asset_id,
                'created': datetime.now().isoformat(),
                'updated': datetime.now().isoformat(),
                'type': 'dataset',
                'name': name,
                'description': description,
                'tags': list(set(tags)),  # Entferne Duplikate
                'categories': categories,
                'author': metadata.get('author', 'OceanData User'),
                'license': metadata.get('license', 'No License Specified'),
                'price': metadata.get('estimated_value', 5.0),  # Geschätzter Wert in OCEAN
                'files': [
                    {
                        'type': 'compute-to-data',
                        'method': 'c2d',
                        'schema': 'c2d-schema'
                    }
                ],
                'additionalInformation': {
                    'privacy': metadata.get('privacy_level', 'medium'),
                    'record_count': metadata.get('record_count', 0),
                    'field_count': metadata.get('field_count', 0),
                    'potential_use_cases': metadata.get('potential_use_cases', []),
                    'value_factors': metadata.get('value_factors', {})
                }
            }
            
            # Compute-to-Data-spezifische Informationen
            c2d_config = {
                'asset_id': asset_id,
                'allowed_operations': [
                    'aggregate', 'count', 'mean', 'sum', 'min', 'max', 'std', 
                    'correlation', 'histogram', 'custom_model'
                ],
                'timeout': 3600,  # 1 Stunde Standard-Timeout
                'privacy_config': {
                    'min_group_size': 5,
                    'noise_level': 0.01,
                    'outlier_removal': True
                }
            }
            
            # Preisgestaltung und Token-Informationen
            pricing = {
                'type': 'fixed',
                'baseToken': {
                    'address': '0x967da4048cD07aB37855c090aAF366e4ce1b9F48',  # Ocean Token Adresse
                    'name': 'Ocean Token',
                    'symbol': 'OCEAN'
                },
                'baseTokenAmount': metadata.get('estimated_value', 5.0),
                'datatoken': {
                    'name': f"DT-{name.replace(' ', '-')}",
                    'symbol': f"DT{asset_id[:4]}".upper()
                }
            }
            
            return {
                'ddo': ddo,
                'c2d_config': c2d_config,
                'pricing': pricing,
                'asset_id': asset_id
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Vorbereitung für Ocean Tokenisierung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'asset_id': data_asset.get('c2d_asset', {}).get('asset_id', 'unknown')
            }
    
    def tokenize_with_ocean(self, ocean_asset: Dict) -> Dict:
        """
        Tokenisiert einen Asset mit Ocean Protocol.
        Diese Funktion ist ein Platzhalter, da die tatsächliche Implementierung 
        von der Integration mit dem Ocean Protocol Smart Contract abhängt.
        
        Args:
            ocean_asset: Vorbereitete Asset-Informationen für Ocean Protocol
            
        Returns:
            Tokenisierungsergebnisse mit Transaktionsinformationen
        """
        # Dies ist eine Platzhalterimplementierung, die in einer echten Umgebung
        # einen Aufruf an die Ocean Protocol Bibliothek und Smart Contracts machen würde
        
        self.logger.info(f"Simuliere Tokenisierung mit Ocean Protocol für Asset {ocean_asset.get('asset_id', 'unknown')}")
        
        # Simulierte Tokenisierungsergebnisse
        tx_hash = f"0x{uuid.uuid4().hex                    if col in anonymized_data.columns and anonymized_data[col].dtype == 'object':
                        # Behalte nur den ersten Teil der Adresse/Standorts (z.B. Stadt)
                        anonymized_data[col] = anonymized_data[col].astype(str).apply(
                            lambda x: x.split(',')[0] if ',' in x else x
                        )
                
                # Vergröbere Zeitstempel (auf Stundenebene)
                for col in datetime_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = pd.to_datetime(anonymized_data[col]).dt.floor('H')
                
            elif privacy_level == 'low':
                # Minimale Anonymisierung, nur offensichtlich persönliche Kennung anonymisieren
                for col in id_columns:
                    if col in anonymized_data.columns and 'user' in col.lower():
                        salt = datetime.now().strftime("%Y%m%d")
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest() if pd.notna(x) else x
                        )
            
            # 2. Berechnung des Datenwerts und Erstellung von Metadaten
            
            # a) Berechne Basiskennzahlen
            metadata = {
                'source_type': source_type,
                'privacy_level': privacy_level,
                'record_count': len(anonymized_data),
                'field_count': len(anonymized_data.columns),
                'time_range': None,
                'created_at': datetime.now().isoformat(),
                'data_schema': {
                    col: str(anonymized_data[col].dtype) for col in anonymized_data.columns
                },
                'estimated_value': 0.0,
                'value_factors': {},
                'column_statistics': {},
                'potential_use_cases': []
            }
            
            # b) Berechne Zeitspanne, falls Zeitstempelspalten existieren
            if datetime_columns:
                # Wähle die erste Zeitstempelspalte
                time_col = datetime_columns[0]
                min_date = anonymized_data[time_col].min()
                max_date = anonymized_data[time_col].max()
                
                metadata['time_range'] = {
                    'start': min_date.isoformat() if pd.notna(min_date) else None,
                    'end': max_date.isoformat() if pd.notna(max_date) else None,
                    'duration_days': (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None
                }
            
            # c) Berechne Spaltenstatistiken für numerische und kategorische Spalten
            for col in anonymized_data.columns:
                col_stats = {}
                
                if pd.api.types.is_numeric_dtype(anonymized_data[col]):
                    # Statistiken für numerische Spalten
                    col_stats = {
                        'type': 'numeric',
                        'min': float(anonymized_data[col].min()) if not pd.isna(anonymized_data[col].min()) else None,
                        'max': float(anonymized_data[col].max()) if not pd.isna(anonymized_data[col].max()) else None,
                        'mean': float(anonymized_data[col].mean()) if not pd.isna(anonymized_data[col].mean()) else None,
                        'median': float(anonymized_data[col].median()) if not pd.isna(anonymized_data[col].median()) else None,
                        'std': float(anonymized_data[col].std()) if not pd.isna(anonymized_data[col].std()) else None,
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100)
                    }
                elif pd.api.types.is_string_dtype(anonymized_data[col]) or anonymized_data[col].dtype == 'object':
                    # Statistiken für kategorische/Text-Spalten
                    col_stats = {
                        'type': 'categorical' if anonymized_data[col].nunique() < 30 else 'text',
                        'unique_values': int(anonymized_data[col].nunique()),
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100),
                        'most_common': anonymized_data[col].value_counts().head(5).to_dict() if anonymized_data[col].nunique() < 30 else None
                    }
                    
                    # Wenn es sich um Text handelt, zusätzliche Textstatistiken
                    if col_stats['type'] == 'text':
                        # Durchschnittliche Textlänge
                        col_stats['avg_length'] = float(anonymized_data[col].astype(str).str.len().mean())
                        
                        # Optional: Sentimentanalyse auf Stichprobe
                        if self.config.get('analyze_text_sentiment', True):
                            sample = anonymized_data[col].dropna().sample(min(100, anonymized_data[col].dropna().shape[0])).tolist()
                            if sample:
                                sentiment_results = self.semantic_analyzer.analyze_sentiment(sample)
                                sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
                                for result in sentiment_results:
                                    sentiment_counts[result['sentiment']] += 1
                                
                                col_stats['sentiment'] = {
                                    k: v / len(sentiment_results) for k, v in sentiment_counts.items()
                                }
                else:
                    # Andere Datentypen (Datum, Boolean, etc.)
                    col_stats = {
                        'type': str(anonymized_data[col].dtype),
                        'unique_values': int(anonymized_data[col].nunique()),
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100)
                    }
                
                metadata['column_statistics'][col] = col_stats
            
            # d) Berechnung des geschätzten Datenwerts basierend auf verschiedenen Faktoren
            value_factors = {}
            
            # Faktor: Datenmenge
            data_size_factor = min(1.0, len(anonymized_data) / 10000)  # Skaliert bis 10.000 Datenpunkte
            value_factors['data_size'] = float(data_size_factor)
            
            # Faktor: Datenvollständigkeit
            completeness = 1.0 - anonymized_data.isna().mean().mean()  # Prozentsatz der nicht-fehlenden Werte
            value_factors['completeness'] = float(completeness)
            
            # Faktor: Zeitliche Spanne (falls verfügbar)
            time_factor = 0.5  # Standardwert
            if metadata['time_range'] and metadata['time_range']['duration_days']:
                # Höherer Wert für längere Zeitreihen (skaliert bis zu 1 Jahr)
                days = metadata['time_range']['duration_days']
                time_factor = min(1.0, days / 365)
            value_factors['time_span'] = float(time_factor)
            
            # Faktor: Datenquellentyp (unterschiedliche Gewichtung je nach Quelle)
            source_weights = {
                'browser': 0.7,
                'calendar': 0.6,
                'chat': 0.8,
                'social_media': 0.9,
                'streaming': 0.8,
                'health_insurance': 1.0,
                'health_data': 1.0,
                'smartwatch': 0.9,
                'iot_vacuum': 0.5,
                'iot_thermostat': 0.6,
                'iot_lighting': 0.5,
                'iot_security': 0.8,
                'smart_home': 0.7
            }
            source_factor = source_weights.get(source_type, 0.5)
            value_factors['source_type'] = float(source_factor)
            
            # Faktor: Datenschutzniveau (höherer Datenschutz = geringerer Wert)
            privacy_factors = {
                'low': 1.0,
                'medium': 0.8,
                'high': 0.6
            }
            privacy_factor = privacy_factors.get(privacy_level, 0.7)
            value_factors['privacy_level'] = float(privacy_factor)
            
            # Schätze den Gesamtwert als gewichteten Durchschnitt der Faktoren
            weights = {
                'data_size': 0.3,
                'completeness': 0.2,
                'time_span': 0.15,
                'source_type': 0.25,
                'privacy_level': 0.1
            }
            
            estimated_value = sum(value_factors[factor] * weights[factor] for factor in weights)
            
            # Konvertiere den normierten Wert in einen Tokenwert (angenommene Basis: 1-10 OCEAN)
            base_token_value = self.config.get('base_token_value', 5.0)  # 5 OCEAN als Basiswert
            token_value = base_token_value * estimated_value
            
            metadata['estimated_value'] = float(token_value)
            metadata['value_factors'] = value_factors
            
            # e) Identifiziere potenzielle Anwendungsfälle basierend auf den Daten
            use_cases = []
            
            # Browser-Daten: Surfgewohnheiten, Werbezielgruppen
            if source_type == 'browser':
                use_cases.append({
                    'title': 'Consumer Behavior Analysis',
                    'description': 'Analyze browsing patterns to understand consumer preferences and interests',
                    'value_proposition': 'Optimize marketing strategies and product recommendations'
                })
                use_cases.append({
                    'title': 'Ad Targeting Optimization',
                    'description': 'Use browsing history to create more relevant ad targeting profiles',
                    'value_proposition': 'Improve ad conversion rates and reduce marketing waste'
                })
            
            # Kalender-Daten: Zeitplanung, Produktivitätsanalyse
            elif source_type == 'calendar':
                use_cases.append({
                    'title': 'Productivity Pattern Analysis',
                    'description': 'Identify optimal meeting times and productivity patterns',
                    'value_proposition': 'Improve organizational efficiency and meeting scheduling'
                })
                use_cases.append({
                    'title': 'Work-Life Balance Insights',
                    'description': 'Analyze time allocation between work and personal activities',
                    'value_proposition': 'Develop better wellness and productivity programs'
                })
            
            # Gesundheitsdaten: Medizinische Forschung, Versicherungsanalyse
            elif source_type in ['health_data', 'health_insurance', 'smartwatch']:
                use_cases.append({
                    'title': 'Health Trend Analysis',
                    'description': 'Identify health patterns and correlations in anonymized health data',
                    'value_proposition': 'Support medical research and health intervention programs'
                })
                use_cases.append({
                    'title': 'Wellness Program Optimization',
                    'description': 'Use activity and health data to design better wellness initiatives',
                    'value_proposition': 'Improve health outcomes and reduce healthcare costs'
                })
            
            # IoT-Daten: Smart Home, Energieoptimierung
            elif source_type in ['iot_vacuum', 'iot_thermostat', 'iot_lighting', 'smart_home']:
                use_cases.append({
                    'title': 'Energy Usage Optimization',
                    'description': 'Analyze home energy consumption patterns',
                    'value_proposition': 'Develop more efficient energy management solutions'
                })
                use_cases.append({
                    'title': 'Smart Home Product Development',
                    'description': 'Understand usage patterns of smart home devices',
                    'value_proposition': 'Inform the design of new IoT products and services'
                })
            
            # Social Media und Chat: Stimmungsanalyse, Trendforschung
            elif source_type in ['social_media', 'chat']:
                use_cases.append({
                    'title': 'Sentiment Analysis and Trend Detection',
                    'description': 'Track public opinion and emerging topics in social conversations',
                    'value_proposition': 'Stay ahead of market trends and consumer sentiment shifts'
                })
                use_cases.append({
                    'title': 'Content Strategy Optimization',
                    'description': 'Analyze engagement patterns to optimize content strategy',
                    'value_proposition': 'Improve audience engagement and content performance'
                })
            
            # Streaming: Medienkonsum, Empfehlungen
            elif source_type == 'streaming':
                use_cases.append({
                    'title': 'Content Preference Analysis',
                    'description': 'Understand viewer preferences and consumption patterns',
                    'value_proposition': 'Optimize content creation and acquisition strategies'
                })
                use_cases.append({
                    'title': 'Recommendation Engine Improvement',
                    'description': 'Enhance content recommendation algorithms with viewing pattern data',
                    'value_proposition': 'Increase viewer satisfaction and platform engagement'
                })
            
            # Generische Anwendungsfälle für andere Quellen
            else:
                use_cases.append({
                    'title': 'Pattern Recognition',
                    'description': 'Identify patterns and trends in the data',
                    'value_proposition': 'Gain insights for business strategy and decision making'
                })
                use_cases.append({
                    'title': 'Predictive Analytics',
                    'description': 'Use historical data to predict future trends',
                    'value_proposition': 'Improve forecasting and strategic planning'
                })
            
            metadata['potential_use_cases'] = use_cases
            
            # Erstelle einen Asset für die Compute-to-Data-Funktionalität
            c2d_asset = self.c2d_manager.create_data_asset(
                anonymized_data,
                asset_metadata=metadata
            )
            
            return {
                'anonymized_data': anonymized_data,
                'metadata': metadata,
                'c2d_asset': c2d_asset
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenaufbereitung für Monetarisierung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'anonymized_data': None,
                'metadata': {
                    'source_type': source_type,
                    'privacy_level': privacy_level,
                    'error': str(e)
                }
            }
    
    def combine_data_sources(self, sources: List[Dict], combination_type: str = 'merge') -> Dict:
        """
        Kombiniert mehrere Datenquellen zu einem wertvolleren Datensatz.
        
        Args:
            sources: Liste von Datenquellen mit 'data' und 'metadata' Schlüsseln
            combination_type: Art der Kombination ('merge', 'enrich', 'correlate')
            
        Returns:
            Kombinierte Daten mit verbesserten Metadaten
        """
        try:
            self.logger.info(f"Kombiniere {len(sources)} Datenquellen mit Methode {combination_type}")
            
            # Prüfe, ob gültige Quellen vorhanden sind
            valid_sources = [s for s in sources if 'anonymized_data' in s and s['anonymized_data'] is not None]
            
            if len(valid_sources) < 2:
                raise ValueError("Mindestens zwei gültige Datenquellen werden für die Kombination benötigt")
            
            combined_data = None
            combined_metadata = {
                'combination_type': combination_type,
                'source_count': len(valid_sources),
                'source_types': [s['metadata']['source_type'] for s in valid_sources],
                'created_at': datetime.now().isoformat(),
                'record_count': 0,
                'field_count': 0,
                'estimated_value': 0.0,
                'combination_description': '',
                'source_metadata': [s['metadata'] for s in valid_sources]
            }
            
            # Implementierung je nach Kombinationstyp
            if combination_type == 'merge':
                # Horizontale Zusammenführung (Zeilen anfügen)
                # Geeignet für gleichartige Datensätze
                
                all_data = [s['anonymized_data'] for s in valid_sources]
                
                # Prüfe Spaltenkompatibilität
                common_columns = set.intersection(*[set(df.columns) for df in all_data])
                
                if not common_columns:
                    raise ValueError("Keine gemeinsamen Spalten für die Zusammenführung gefunden")
                
                # Zusammenführung der Daten mit gemeinsamen Spalten
                combined_data = pd.concat([df[list(common_columns)] for df in all_data], ignore_index=True)
                
                combined_metadata['combination_description'] = f"Horizontal merge of {len(valid_sources)} similar datasets, keeping {len(common_columns)} common columns"
                combined_metadata['record_count'] = len(combined_data)
                combined_metadata['field_count'] = len(combined_data.columns)
                
                # Wertschätzung: Summe der Einzelwerte mit einem Multiplikator für die Kombination
                base_value = sum(s['metadata'].get('estimated_value', 0) for s in valid_sources)
                combined_metadata['estimated_value'] = base_value * 1.2  # 20% Mehrwert durch Kombination
                
            elif combination_type == 'enrich':
                # Vertikale Anreicherung (Spalten anfügen)
                # Geeignet für komplementäre Datensätze mit einem gemeinsamen Identifikator
                
                # Erster Datensatz als Basis
                base_data = valid_sources[0]['anonymized_data']
                join_key = None
                
                # Suche nach einem gemeinsamen Schlüssel für die Verknüpfung
                for col in base_data.columns:
                    if all(col in s['anonymized_data'].columns for s in valid_sources[1:]):
                        # Prüfe, ob der Wert als Schlüssel dienen kann (eindeutige Werte)
                        if all(s['anonymized_data'][col].nunique() / len(s['anonymized_data']) > 0.7 for s in valid_sources):
                            join_key = col
                            break
                
                if not join_key:
                    # Falls kein expliziter Schlüssel existiert, erstelle einen künstlichen Schlüssel
                    self.logger.warning("Kein natürlicher Schlüssel gefunden, erstelle künstlichen Index")
                    
                    # Erstelle einen eindeutigen Index für jeden Datensatz
                    for i, source in enumerate(valid_sources):
                        source['anonymized_data']['temp_join_key'] = range(len(source['anonymized_data']))
                    
                    join_key = 'temp_join_key'
                
                # Führe die Daten zusammen
                combined_data = base_data.copy()
                
                for source in valid_sources[1:]:
                    # Finde neue Spalten, die noch nicht existieren
                    new_columns = [col for col in source['anonymized_data'].columns if col not in combined_data.columns]
                    
                    if new_columns:
                        # Verknüpfe über den Schlüssel mit linkem Join (behält alle Zeilen von combined_data)
                        combined_data = pd.merge(
                            combined_data,
                            source['anonymized_data'][new_columns + [join_key]],
                            on=join_key,
                            how='left'
                        )
                
                # Entferne den temporären Schlüssel, falls erstellt
                if join_key == 'temp_join_key' and join_key in combined_data.columns:
                    combined_data = combined_data.drop(columns=[join_key])
                
                combined_metadata['combination_description'] = f"Vertical enrichment using key '{join_key}', adding {len(combined_data.columns) - len(base_data.columns)} new features"
                combined_metadata['record_count'] = len(combined_data)
                combined_metadata['field_count'] = len(combined_data.columns)
                
                # Wertschätzung: Wert der Basis plus zusätzlicher Wert für neue Spalten
                base_value = valid_sources[0]['metadata'].get('estimated_value', 0)
                additional_value = sum(s['metadata'].get('estimated_value', 0) * 0.6 for s in valid_sources[1:])  # 60% des Werts der zusätzlichen Quellen
                combined_metadata['estimated_value'] = base_value + additional_value
                
            elif combination_type == 'correlate':
                # Korrelationsanalyse zwischen verschiedenen Datenquellen
                # Erstellt einen neuen Datensatz mit Korrelationsmaßen und abgeleiteten Features
                
                # Erstelle eine Liste für Korrelationsergebnisse
                correlation_results = []
                feature_pairs = []
                
                # Suche nach numerischen Spalten in jedem Datensatz
                for i, source1 in enumerate(valid_sources):
                    for j in range(i+1, len(valid_sources)):
                        source2 = valid_sources[j]
                        
                        # Extrahiere numerische Spalten
                        num_cols1 = source1['anonymized_data'].select_dtypes(include=['number']).columns
                        num_cols2 = source2['anonymized_data'].select_dtypes(include=['number']).columns
                        
                        # Prüfe, ob genügend numerische Spalten vorhanden sind
                        if len(num_cols1) == 0 or len(num_cols2) == 0:
                            continue
                        
                        # Wähle die Top-5-Spalten pro Quelle für die Analyse
                        select_cols1 = list(num_cols1)[:5]
                        select_cols2 = list(num_cols2)[:5]
                        
                        # Erstelle ein DataFrame mit Korrelationswerten
                        source_type1 = source1['metadata']['source_type']
                        source_type2 = source2['metadata']['source_type']
                        
                        for col1 in select_cols1:
                            for col2 in select_cols2:
                                # Erstelle 100 zufällige Datenpunkte für jede Spalte
                                # (für eine Stichprobenkorrelation ohne Verknüpfung der tatsächlichen Daten)
                                sample1 = source1['anonymized_data'][col1].sample(min(100, len(source1['anonymized_data']))).reset_index(drop=True)
                                sample2 = source2['anonymized_data'][col2].sample(min(100, len(source2['anonymized_data']))).reset_index(drop=True)
                                
                                # Berechne die Korrelation
                                try:
                                    corr = sample1.corr(sample2)
                                    
                                    # Speichere nur signifikante Korrelationen
                                    if abs(corr) > 0.3:
                                        corr_entry = {
                                            'source1': source_type1,
                                            'source2': source_type2,
                                            'feature1': col1,
                                            'feature2': col2,
                                            'correlation': float(corr),
                                            'abs_correlation': float(abs(corr))
                                        }
                                        correlation_results.append(corr_entry)
                                        feature_pairs.append((source1, col1, source2, col2))
                                except:
                                    # Ignoriere Fehler bei der Korrelationsberechnung
                                    pass
                
                # Erstelle ein neues DataFrame mit den Korrelationsergebnissen
                if correlation_results:
                    combined_data = pd.DataFrame(correlation_results)
                    
                    # Füge abgeleitete Features für die Top-Korrelationen hinzu
                    if feature_pairs:
                        # Sortiere nach absoluter Korrelationsstärke
                        sorted_pairs = sorted(list(zip(correlation_results, feature_pairs)), 
                                             key=lambda x: x[0]['abs_correlation'], 
                                             reverse=True)
                        
                        # Wähle die Top-10-Paare (oder weniger, falls nicht genügend vorhanden)
                        top_pairs = sorted_pairs[:min(10, len(sorted_pairs))]
                        
                        # Erstelle für jedes Top-Paar ein kombiniertes Feature
                        for i, (corr_info, (source1, col1, source2, col2)) in enumerate(top_pairs):
                            # Normalisiere die Werte
                            samples1 = source1['anonymized_data'][col1].sample(min(100, len(source1['anonymized_data']))).reset_index(drop=True)
                            samples2 = source2['anonymized_data'][col2].sample(min(100, len(source2['anonymized_data']))).reset_index(drop=True)
                            
                            # Z-Score-Normalisierung
                            normalized1 = (samples1 - samples1.mean()) / samples1.std()
                            normalized2 = (samples2 - samples2.mean()) / samples2.std()
                            
                            # Erstelle ein kombiniertes Feature basierend auf der Korrelationsrichtung
                            if corr_info['correlation'] >= 0:
                                combined_feature = normalized1 + normalized2
                            else:
                                combined_feature = normalized1 - normalized2
                            
                            # Füge das kombinierte Feature dem Ergebnis hinzu
                            feature_name = f"combined_{source1['metadata']['source_type']}_{col1}_{source2['metadata']['source_type']}_{col2}"
                            combined_data[feature_name] = combined_feature
                    
                    combined_metadata['combination_description'] = f"Correlation analysis between {len(valid_sources)} datasets, identifying {len(correlation_results)} significant cross-source relationships"
                    combined_metadata['record_count'] = len(combined_data)
                    combined_metadata['field_count'] = len(combined_data.columns)
                    
                    # Wertschätzung: Basierend auf der Anzahl und Stärke der gefundenen Korrelationen
                    avg_correlation = np.mean([r['abs_correlation'] for r in correlation_results]) if correlation_results else 0
                    correlation_value = len(correlation_results) * avg_correlation * 0.5  # Grundwert pro signifikante Korrelation
                    
                    # Füge Werte der Originalquellen hinzu
                    base_value = sum(s['metadata'].get('estimated_value', 0) * 0.3 for s in valid_sources)  # 30% der Originalwerte
                    combined_metadata['estimated_value'] = base_value + correlation_value
                else:
                    raise ValueError("Keine signifikanten Korrelationen zwischen den Datenquellen gefunden")
            
            else:
                raise ValueError(f"Nicht unterstützter Kombinationstyp: {combination_type}")
            
            # Erstelle einen neuen C2D-Asset für den kombinierten Datensatz
            c2d_asset = self.c2d_manager.create_data_asset(
                combined_data,
                asset_metadata=combined_metadata
            )
            
            return {
                'anonymized_data': combined_data,
                'metadata': combined_metadata,
                'c2d_asset': c2d_asset
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Kombination von Datenquellen: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'anonymized_data': None,
                'metadata': {
                    'combination_type': combination_type,
                    'source_count': len(sources),
                    'error': str(e)
                }
            }
    
    def estimate_data_value(self, data: pd.DataFrame, metadata: Dict = None) -> Dict:
        """
        Berechnet einen detaillierten Wert für einen Datensatz basierend auf mehreren Faktoren.
        
        Args:
            data: Der zu bewertende Datensatz
            metadata: Optionale Metadaten über den Datensatz
            
        Returns:
            Detaillierte Wertschätzung mit Begründungen
        """
        try:
            self.logger.info(f"Berechne detaillierten Datenwert für Datensatz mit {len(data)} Zeilen und {len(data.columns)} Spalten")
            
            if metadata is None:
                metadata = {}
            
            # Basisfaktoren für die Wertschätzung
            value_metrics = {
                'data_volume': {
                    'score': 0.0,
                    'weight': 0.15,
                    'explanation': ''
                },
                'data_quality': {
                    'score': 0.0,
                    'weight': 0.2,
                    'explanation': ''
                },
                'data_uniqueness': {
                    'score': 0._group_size']].index
            
            means = grouped.mean().loc[valid_groups]
            
            # Füge Rauschen hinzu
            noisy_means = self._add_differential_privacy_noise(means)
            
            return {'group_means': noisy_means.to_dict()}
        else:
            means = prepared_data.mean()
            
            # Füge Rauschen hinzu
            noisy_means = self._add_differential_privacy_noise(means)
            
            return {'means': noisy_means.to_dict()}
    
    def _sum(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Summe für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        sums = prepared_data.sum()
        
        # Füge Rauschen hinzu
        noisy_sums = self._add_differential_privacy_noise(sums)
        
        return {'sums': noisy_sums.to_dict()}
    
    def _min(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet das Minimum für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        minimums = prepared_data.min()
        
        # Hier kein Rauschen, da Minimum keine sensiblen Informationen preisgibt
        return {'minimums': minimums.to_dict()}
    
    def _max(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet das Maximum für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        maximums = prepared_data.max()
        
        # Hier kein Rauschen, da Maximum keine sensiblen Informationen preisgibt
        return {'maximums': maximums.to_dict()}
    
    def _std(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Standardabweichung für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        std_devs = prepared_data.std()
        
        # Füge Rauschen hinzu
        noisy_std_devs = self._add_differential_privacy_noise(std_devs)
        
        return {'standard_deviations': noisy_std_devs.to_dict()}
    
    def _correlation(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Korrelation zwischen numerischen Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        # Verwende nur numerische Spalten
        numeric_data = prepared_data.select_dtypes(include=['number'])
        
        if numeric_data.empty or numeric_data.shape[1] < 2:
            return {'error': 'Nicht genügend numerische Spalten für Korrelation'}
        
        corr_matrix = numeric_data.corr()
        
        # Füge Rauschen hinzu
        noisy_corr = self._add_differential_privacy_noise(corr_matrix)
        
        # Stelle sicher, dass Werte im gültigen Bereich liegen
        noisy_corr = noisy_corr.clip(-1, 1)
        
        return {'correlation_matrix': noisy_corr.to_dict()}
    
    def _histogram(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Erstellt ein Histogramm für eine Spalte"""
        column = params.get('column')
        bins = params.get('bins', 10)
        
        if column not in data.columns:
            return {'error': f'Spalte {column} nicht gefunden'}
        
        prepared_data = self._prepare_data_for_computation(data, [column])
        
        if pd.api.types.is_numeric_dtype(prepared_data[column]):
            # Numerisches Histogramm
            hist, bin_edges = np.histogram(prepared_data[column], bins=bins)
            
            # Füge Rauschen hinzu
            noisy_hist = self._add_differential_privacy_noise(hist).astype(int)
            
            # Stelle sicher, dass Werte nicht negativ sind
            noisy_hist = np.maximum(0, noisy_hist)
            
            return {
                'histogram': noisy_hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        else:
            # Kategorisches Histogramm
            value_counts = prepared_data[column].value_counts()
            
            # Prüfe auf zu kleine Gruppen
            value_counts = value_counts[value_counts >= self.privacy_config['min_group_size']]
            
            if value_counts.empty:
                return {'error': 'Keine Kategorien mit ausreichender Größe'}
            
            # Füge Rauschen hinzu
            noisy_counts = self._add_differential_privacy_noise(value_counts).astype(int)
            
            # Stelle sicher, dass Werte nicht negativ sind
            noisy_counts = noisy_counts.clip(0)
            
            return {
                'categories': value_counts.index.tolist(),
                'counts': noisy_counts.tolist()
            }
    
    def _custom_model_inference(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Führt Inferenz mit einem benutzerdefinierten Modell durch, ohne die Daten preiszugeben
        
        Args:
            data: Eingabedaten für die Inferenz
            params: Parameter für die Inferenz, einschließlich des Modelltyps und der Modellparameter
            
        Returns:
            Ergebnisse der Inferenz
        """
        model_type = params.get('model_type')
        model_config = params.get('model_config', {})
        columns = params.get('columns')
        
        if not model_type:
            return {'error': 'Kein Modelltyp angegeben'}
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        try:
            results = None
            
            if model_type == 'anomaly_detection':
                # Anomalieerkennung mit Isolation Forest
                detector = AnomalyDetector(method='isolation_forest', contamination=model_config.get('contamination', 0.05))
                detector.fit(prepared_data)
                predictions = detector.predict(prepared_data)
                
                # Aggregiere Ergebnisse für den Datenschutz
                anomaly_count = np.sum(predictions == -1)
                total_count = len(predictions)
                
                # Füge Rauschen hinzu
                noisy_anomaly_count = self._add_differential_privacy_noise(anomaly_count).astype(int)
                
                results = {
                    'anomaly_percentage': round(100 * noisy_anomaly_count / total_count, 2),
                    'total_records': total_count
                }
            
            elif model_type == 'clustering':
                # K-Means-Clustering
                n_clusters = model_config.get('n_clusters', 3)
                
                # Verarbeite nur numerische Spalten
                numeric_data = prepared_data.select_dtypes(include=['number'])
                
                if numeric_data.empty:
                    return {'error': 'Keine numerischen Spalten für Clustering gefunden'}
                
                # Standardisiere Daten
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                # Cluster-Analyse
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Cluster-Statistiken (datenschutzfreundlich)
                cluster_sizes = {}
                for i in range(n_clusters):
                    size = np.sum(clusters == i)
                    
                    # Prüfe Mindestgruppengröße
                    if size >= self.privacy_config['min_group_size']:
                        # Füge Rauschen hinzu
                        noisy_size = self._add_differential_privacy_noise(size).astype(int)
                        cluster_sizes[f'cluster_{i}'] = max(0, noisy_size)
                
                # Cluster-Zentren (keine persönlichen Daten)
                cluster_centers = kmeans.cluster_centers_
                
                results = {
                    'cluster_sizes': cluster_sizes,
                    'cluster_centers': cluster_centers.tolist()
                }
            
            elif model_type == 'sentiment_analysis':
                # Sentimentanalyse für Textdaten
                text_column = model_config.get('text_column')
                
                if text_column not in prepared_data.columns:
                    return {'error': f'Textspalte {text_column} nicht gefunden'}
                
                # Einfache Sentimentanalyse mit NLTK
                analyzer = SemanticAnalyzer()
                sentiment_results = analyzer.analyze_sentiment(prepared_data[text_column].tolist())
                
                # Aggregierte Ergebnisse für den Datenschutz
                sentiment_counts = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
                
                for result in sentiment_results:
                    sentiment_counts[result['sentiment']] += 1
                
                # Füge Rauschen hinzu
                for sentiment in sentiment_counts:
                    noisy_count = self._add_differential_privacy_noise(sentiment_counts[sentiment]).astype(int)
                    sentiment_counts[sentiment] = max(0, noisy_count)
                
                results = {
                    'sentiment_distribution': sentiment_counts,
                    'total_records': len(sentiment_results)
                }
            
            else:
                return {'error': f'Nicht unterstützter Modelltyp: {model_type}'}
            
            return results
            
        except Exception as e:
            logger.error(f"Fehler bei der Modell-Inferenz: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Modell-Inferenz: {str(e)}'}
    
    def execute_operation(self, encrypted_data: bytes, operation: str, params: Dict) -> Dict:
        """
        Führt eine Operation auf verschlüsselten Daten aus, ohne sie zu entschlüsseln
        
        Args:
            encrypted_data: Verschlüsselte Daten
            operation: Name der auszuführenden Operation
            params: Parameter für die Operation
            
        Returns:
            Ergebnisse der Operation
        """
        try:
            if operation not in self.allowed_operations:
                return {'error': f'Nicht unterstützte Operation: {operation}'}
            
            # Entschlüssele die Daten temporär im Speicher
            data = self._decrypt_data(encrypted_data)
            
            # Führe die Operation aus
            operation_func = self.allowed_operations[operation]
            results = operation_func(data, params)
            
            return results
            
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung der Operation {operation}: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Ausführung: {str(e)}'}
    
    def create_data_asset(self, data: pd.DataFrame, asset_metadata: Dict = None) -> Dict:
        """
        Erstellt einen verschlüsselten Daten-Asset mit Metadaten für den Marktplatz
        
        Args:
            data: Zu verschlüsselnde Daten
            asset_metadata: Metadaten für den Asset
            
        Returns:
            Asset-Details mit ID und Metadaten
        """
        try:
            # Erstelle eine eindeutige ID für den Asset
            asset_id = uuid.uuid4().hex
            
            # Verschlüssele die Daten
            encrypted_data = self._encrypt_data(data)
            
            # Erstelle Metadaten mit datenschutzfreundlichen Statistiken
            stats = {}
            
            # Anzahl der Datensätze
            stats['record_count'] = len(data)
            
            # Spalteninformationen
            stats['columns'] = []
            for col in data.columns:
                col_info = {
                    'name': col,
                    'type': str(data[col].dtype)
                }
                
                # Füge datenschutzfreundliche Statistiken hinzu
                if pd.api.types.is_numeric_dtype(data[col]):
                    # Füge Rauschen hinzu, um Datenschutz zu gewährleisten
                    col_info['min'] = float(self._add_differential_privacy_noise(data[col].min()))
                    col_info['max'] = float(self._add_differential_privacy_noise(data[col].max()))
                    col_info['mean'] = float(self._add_differential_privacy_noise(data[col].mean()))
                    col_info['std'] = float(self._add_differential_privacy_noise(data[col].std()))
                else:
                    # Für kategorische Daten nur die Anzahl eindeutiger Werte
                    col_info['unique_values'] = int(self._add_differential_privacy_noise(data[col].nunique()))
                
                stats['columns'].append(col_info)
            
            # Zusammenführen mit benutzerdefinierten Metadaten
            asset_info = {
                'asset_id': asset_id,
                'created_at': datetime.now().isoformat(),
                'statistics': stats
            }
            
            if asset_metadata:
                asset_info.update(asset_metadata)
            
            return {
                'asset_id': asset_id,
                'metadata': asset_info,
                'encrypted_data': encrypted_data
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung des Daten-Assets: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Asset-Erstellung: {str(e)}'}
    
    def generate_access_token(self, asset_id: str, allowed_operations: List[str], 
                            expiration_time: int = 3600) -> Dict:
        """
        Generiert ein temporäres Zugriffstoken für einen Daten-Asset
        
        Args:
            asset_id: ID des Daten-Assets
            allowed_operations: Liste der erlaubten Operationen
            expiration_time: Ablaufzeit in Sekunden
            
        Returns:
            Token-Details mit Token und Ablaufzeit
        """
        # Prüfe, ob alle angeforderten Operationen unterstützt werden
        unsupported_ops = [op for op in allowed_operations if op not in self.allowed_operations]
        if unsupported_ops:
            return {'error': f'Nicht unterstützte Operationen: {", ".join(unsupported_ops)}'}
        
        # Erstelle Token-Daten
        token_data = {
            'asset_id': asset_id,
            'allowed_operations': allowed_operations,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=expiration_time)).isoformat(),
            'token_id': uuid.uuid4().hex
        }
        
        # Verschlüssele Token-Daten
        token_json = json.dumps(token_data).encode()
        encrypted_token = self.cipher_suite.encrypt(token_json)
        
        return {
            'token': encrypted_token.decode(),
            'token_id': token_data['token_id'],
            'expires_at': token_data['expires_at']
        }
    
    def validate_access_token(self, token: str, operation: str) -> bool:
        """
        Validiert ein Zugriffstoken für eine bestimmte Operation
        
        Args:
            token: Zugriffstoken
            operation: Angeforderte Operation
            
        Returns:
            True, wenn das Token gültig ist und die Operation erlaubt ist
        """
        try:
            # Entschlüssele das Token
            decrypted_token = self.cipher_suite.decrypt(token.encode())
            token_data = json.loads(decrypted_token.decode())
            
            # Prüfe, ob das Token abgelaufen ist
            expiry_time = datetime.fromisoformat(token_data['expires_at'])
            if datetime.now() > expiry_time:
                logger.warning(f"Zugriffstoken abgelaufen: {token_data['token_id']}")
                return False
            
            # Prüfe, ob die Operation erlaubt ist
            if operation not in token_data['allowed_operations']:
                logger.warning(f"Operation {operation} nicht erlaubt für Token {token_data['token_id']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Token-Validierung: {str(e)}")
            return False