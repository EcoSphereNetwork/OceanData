"""
OceanData - Hauptklasse OceanDataAI

Diese Klasse bildet das Herzstück der OceanData-Plattform und integriert
alle Komponenten zu einer kohärenten Anwendung für die Datenmonetarisierung.
"""

import pandas as pd
import logging
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

# Import der einzelnen Komponenten
from oceandata.data_integration.base import DataSource, DataCategory, PrivacyLevel
from oceandata.analytics.models.anomaly_detector import AnomalyDetector
from oceandata.privacy.compute_to_data import ComputeToDataManager
from oceandata.blockchain.ocean_integration import OceanIntegration

# Logging konfigurieren
logger = logging.getLogger("OceanData.Core")

class OceanDataAI:
    """
    Hauptklasse der OceanData-Plattform.
    
    Diese Klasse orchestriert alle Komponenten der Plattform:
    - Datenerfassung und -integration
    - KI-gestützte Analyse
    - Datenschutz und Compute-to-Data
    - Monetarisierung über Ocean Protocol
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialisiert die OceanDataAI-Plattform.
        
        Args:
            config: Konfigurationsdaten für die Plattform
        """
        self.config = config or {}
        
        # Initialisiere Komponenten
        self.anomaly_detector = AnomalyDetector(
            method=self.config.get('anomaly_detection_method', 'isolation_forest'),
            contamination=self.config.get('anomaly_contamination', 0.05)
        )
        
        # In einer vollständigen Implementierung würden wir weitere Komponenten initialisieren:
        # self.semantic_analyzer = SemanticAnalyzer()
        # self.predictive_modeler = PredictiveModeler()
        # self.data_synthesizer = DataSynthesizer()
        
        # Datenschutz und C2D
        self.c2d_manager = ComputeToDataManager(
            privacy_config=self.config.get('privacy_config')
        )
        
        # Ocean Protocol Integration
        self.ocean = OceanIntegration(
            config=self.config.get('ocean_config')
        )
        
        logger.info("OceanDataAI-Plattform initialisiert")
    
    def analyze_data_source(self, data: pd.DataFrame, source_type: str) -> Dict:
        """
        Führt eine umfassende Analyse einer Datenquelle durch.
        
        Args:
            data: DataFrame mit den zu analysierenden Daten
            source_type: Art der Datenquelle (z.B. 'browser', 'smartwatch')
            
        Returns:
            Dict: Analyseergebnisse
        """
        try:
            if data is None or data.empty:
                logger.warning(f"Leere Daten für die Analyse von {source_type}")
                return {
                    'source_type': source_type,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': 'Leere Daten'
                }
            
            logger.info(f"Analyse von {source_type}-Daten mit {len(data)} Datensätzen gestartet")
            
            # Grundlegende Datenstatistiken
            stats = {
                'record_count': len(data),
                'column_count': len(data.columns),
                'columns': list(data.columns),
            }
            
            # Anomalieerkennung
            numeric_data = data.select_dtypes(include=['number'])
            if not numeric_data.empty:
                self.anomaly_detector.fit(numeric_data)
                anomaly_predictions = self.anomaly_detector.predict(numeric_data)
                anomaly_scores = self.anomaly_detector.get_anomaly_scores(numeric_data)
                
                # Erkenntnisse über Anomalien
                anomaly_insights = self.anomaly_detector.get_anomaly_insights(numeric_data, anomaly_predictions)
                
                # Anomaliestatistiken
                anomaly_count = sum(1 for pred in anomaly_predictions if pred == -1)
                anomaly_percentage = (anomaly_count / len(data)) * 100
                
                anomaly_analysis = {
                    'count': anomaly_count,
                    'percentage': anomaly_percentage,
                    'insights': anomaly_insights[:5]  # Limitiere auf Top-5-Erkenntnisse
                }
            else:
                anomaly_analysis = {
                    'count': 0,
                    'percentage': 0,
                    'insights': []
                }
            
            # In einer vollständigen Implementierung würden wir weitere Analysen durchführen:
            # - Semantic Analysis für Text
            # - Zeitreihenanalyse für zeitbasierte Daten
            # - Predictive Modeling
            
            # Zeitbasierte Analyse, falls Zeitstempel vorhanden sind
            time_series_analysis = {}
            if 'timestamp' in data.columns or any(col.lower().endswith('time') or col.lower().endswith('date') for col in data.columns):
                # Identifiziere die Zeitstempelspalte
                time_col = 'timestamp'
                if 'timestamp' not in data.columns:
                    for col in data.columns:
                        if col.lower().endswith('time') or col.lower().endswith('date'):
                            time_col = col
                            break
                
                # Einfache Zeitreihenanalyse
                if time_col in data.columns:
                    data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
                    valid_times = data.dropna(subset=[time_col])
                    
                    if not valid_times.empty:
                        # Zeiträume
                        time_range = {
                            'start': valid_times[time_col].min().isoformat(),
                            'end': valid_times[time_col].max().isoformat(),
                            'duration_hours': (valid_times[time_col].max() - valid_times[time_col].min()).total_seconds() / 3600
                        }
                        
                        time_series_analysis = {
                            'time_column': time_col,
                            'time_range': time_range,
                            'granularity': self._detect_time_granularity(valid_times[time_col]),
                            'is_periodic': self._check_periodicity(valid_times[time_col]),
                            'forecast_horizon': 7  # Standard-Vorhersagehorizont
                        }
            
            # Spezifische Analysen je nach Datenquellentyp
            source_specific_analysis = {}
            
            if source_type == 'browser':
                # Browser-spezifische Analyse
                source_specific_analysis = self._analyze_browser_data(data)
            elif source_type == 'smartwatch' or source_type == 'health_data':
                # Gesundheitsdaten-spezifische Analyse
                source_specific_analysis = self._analyze_health_data(data)
            elif source_type == 'social_media':
                # Social-Media-spezifische Analyse
                source_specific_analysis = self._analyze_social_media_data(data)
            
            # Kombiniere alle Analyseergebnisse
            result = {
                'source_type': source_type,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'statistics': stats,
                'analyses': {
                    'anomalies': anomaly_analysis,
                    'time_series': time_series_analysis,
                    'source_specific': source_specific_analysis
                }
            }
            
            logger.info(f"Analyse von {source_type}-Daten abgeschlossen")
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenanalyse: {str(e)}")
            return {
                'source_type': source_type,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def prepare_data_for_monetization(self, data: pd.DataFrame, source_type: str, 
                                    privacy_level: str = 'medium') -> Dict:
        """
        Bereitet Daten für die Monetarisierung vor.
        
        Args:
            data: DataFrame mit den zu monetarisierenden Daten
            source_type: Art der Datenquelle
            privacy_level: Datenschutzniveau ('low', 'medium', 'high')
            
        Returns:
            Dict: Vorbereitete Daten für die Monetarisierung
        """
        try:
            if data is None or data.empty:
                logger.warning(f"Leere Daten für die Monetarisierung von {source_type}")
                return {
                    'status': 'error',
                    'error': 'Leere Daten'
                }
            
            logger.info(f"Vorbereitung von {source_type}-Daten für Monetarisierung mit Privacy-Level {privacy_level}")
            
            # Konvertiere Datenschutzniveau in PrivacyLevel-Enum
            privacy_mapping = {
                'low': PrivacyLevel.PUBLIC,
                'medium': PrivacyLevel.ANONYMIZED,
                'high': PrivacyLevel.ENCRYPTED
            }
            privacy_enum = privacy_mapping.get(privacy_level.lower(), PrivacyLevel.ANONYMIZED)
            
            # Bestimme zu schützende Felder basierend auf Datenquellentyp und Datenschutzniveau
            protected_fields = self._get_protected_fields(data, source_type, privacy_enum)
            
            # Wende Datenschutzmaßnahmen an
            anonymized_data = data.copy()
            
            for field, level in protected_fields.items():
                if field in anonymized_data.columns:
                    if level == PrivacyLevel.ANONYMIZED:
                        # Anonymisiere das Feld
                        anonymized_data = self._anonymize_field(anonymized_data, field)
                    elif level == PrivacyLevel.ENCRYPTED:
                        # Verschlüssele oder entferne das Feld
                        anonymized_data = anonymized_data.drop(field, axis=1)
                    elif level == PrivacyLevel.SENSITIVE:
                        # Entferne sensitive Daten
                        anonymized_data = anonymized_data.drop(field, axis=1)
            
            # Erstelle C2D-Asset für geschützte Daten
            c2d_asset = self.c2d_manager.create_data_asset(data, {
                'source_type': source_type,
                'privacy_level': privacy_level,
                'original_columns': list(data.columns),
                'preserved_columns': list(anonymized_data.columns)
            })
            
            # Schätze den Wert der Daten basierend auf Typus und Umfang
            analysis_result = self.analyze_data_source(data, source_type)
            value_estimation = self.estimate_data_value(data, {
                'source_type': source_type,
                'analysis_result': analysis_result
            })
            
            # Berücksichtige Datenschutzniveau bei der Wertschätzung
            privacy_factor = {
                'low': 1.2,     # Höherer Wert bei geringem Datenschutz
                'medium': 1.0,  # Standardwert
                'high': 0.8     # Geringerer Wert bei hohem Datenschutz (weniger nutzbare Daten)
            }.get(privacy_level, 1.0)
            
            adjusted_value = value_estimation['estimated_token_value'] * privacy_factor
            
            # Erstelle Metadaten für die Monetarisierung
            metadata = {
                'source_type': source_type,
                'privacy_level': privacy_level,
                'record_count': len(data),
                'field_count': len(anonymized_data.columns),
                'original_field_count': len(data.columns),
                'estimated_value': adjusted_value,
                'adjusted_privacy_factor': privacy_factor,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"Daten für Monetarisierung vorbereitet: {len(anonymized_data)} Datensätze, geschätzter Wert: {adjusted_value}")
            
            return {
                'anonymized_data': anonymized_data,
                'metadata': metadata,
                'c2d_asset': c2d_asset,
                'value_estimation': value_estimation
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenvorbereitung für Monetarisierung: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def combine_data_sources(self, sources: List[Dict], combination_type: str = 'merge') -> Dict:
        """
        Kombiniert mehrere Datenquellen zu einem wertvolleren Asset.
        
        Args:
            sources: Liste der zu kombinierenden Datenquellen
            combination_type: Art der Kombination ('merge', 'enrich', 'correlate')
            
        Returns:
            Dict: Kombiniertes Daten-Asset
        """
        try:
            if not sources:
                logger.warning("Keine Datenquellen zum Kombinieren angegeben")
                return {
                    'status': 'error',
                    'error': 'Keine Datenquellen angegeben'
                }
            
            logger.info(f"Kombination von {len(sources)} Datenquellen mit Methode '{combination_type}'")
            
            # Extrahiere Daten und Metadaten
            data_frames = []
            metadata_list = []
            
            for source in sources:
                if 'anonymized_data' in source and source['anonymized_data'] is not None:
                    data_frames.append(source['anonymized_data'])
                if 'metadata' in source:
                    metadata_list.append(source['metadata'])
            
            if not data_frames:
                logger.warning("Keine gültigen Daten in den Datenquellen")
                return {
                    'status': 'error',
                    'error': 'Keine gültigen Daten in den Datenquellen'
                }
            
            # Kombiniere die Daten je nach Kombinationstyp
            if combination_type == 'merge':
                # Einfaches vertikales Anfügen (mehr Datensätze)
                combined_data = pd.concat(data_frames, ignore_index=True)
                
            elif combination_type == 'enrich':
                # Horizontale Anreicherung (mehr Features)
                # Nur die ersten beiden Quellen werden angereichert
                combined_data = data_frames[0].copy()
                
                if len(data_frames) > 1:

# Horizontale Anreicherung (mehr Features)
                # Nur die ersten beiden Quellen werden angereichert
                combined_data = data_frames[0].copy()
                
                if len(data_frames) > 1:
                    # Wähle Spalten von der zweiten Quelle, die nicht in der ersten vorhanden sind
                    additional_columns = [col for col in data_frames[1].columns if col not in combined_data.columns]
                    
                    if additional_columns and len(data_frames[1]) == len(combined_data):
                        # Füge neue Spalten hinzu
                        for col in additional_columns:
                            combined_data[col] = data_frames[1][col].values
            
            elif combination_type == 'correlate':
                # Korrelationsanalyse zwischen Datenquellen
                # Hierfür benötigen wir einen gemeinsamen Schlüssel
                common_keys = set.intersection(*[set(df.columns) for df in data_frames])
                
                if not common_keys:
                    logger.warning("Keine gemeinsamen Schlüssel für Korrelationsanalyse gefunden")
                    return {
                        'status': 'error',
                        'error': 'Keine gemeinsamen Schlüssel für Korrelationsanalyse gefunden'
                    }
                
                # Verwende den ersten gemeinsamen Schlüssel
                key = list(common_keys)[0]
                
                # Erstelle ein kombiniertes DataFrame durch Zusammenführen über den Schlüssel
                combined_data = data_frames[0]
                
                for i, df in enumerate(data_frames[1:], 1):
                    suffix = f'_{i}'
                    combined_data = pd.merge(combined_data, df, on=key, suffixes=('', suffix))
            
            else:
                logger.warning(f"Nicht unterstützter Kombinationstyp: {combination_type}")
                return {
                    'status': 'error',
                    'error': f'Nicht unterstützter Kombinationstyp: {combination_type}'
                }
            
            # Erstelle C2D-Asset für die kombinierten Daten
            c2d_asset = self.c2d_manager.create_data_asset(combined_data, {
                'combination_type': combination_type,
                'source_count': len(sources),
                'source_types': [source.get('metadata', {}).get('source_type', 'unknown') for source in sources],
                'original_data_total_records': sum(source.get('metadata', {}).get('record_count', 0) for source in sources)
            })
            
            # Schätze den Wert des kombinierten Datensatzes
            source_values = [source.get('metadata', {}).get('estimated_value', 0) for source in sources]
            
            if combination_type == 'merge':
                # Bei Merge: Summe der Werte mit 20% Bonus
                combined_value = sum(source_values) * 1.2
            elif combination_type == 'enrich':
                # Bei Enrich: Höchster Wert plus 60% der Werte der weiteren Quellen
                combined_value = max(source_values) + sum(source_values[1:]) * 0.6 if len(source_values) > 1 else max(source_values)
            elif combination_type == 'correlate':
                # Bei Correlate: 30% der Originalwerte plus Bonus pro Quelle
                combined_value = sum(source_values) * 0.3 + len(sources) * 0.5
            else:
                combined_value = sum(source_values)
            
            # Erstelle Metadaten für die kombinierten Daten
            metadata = {
                'combination_type': combination_type,
                'source_count': len(sources),
                'source_types': [source.get('metadata', {}).get('source_type', 'unknown') for source in sources],
                'record_count': len(combined_data),
                'field_count': len(combined_data.columns),
                'estimated_value': combined_value,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"Datenquellen erfolgreich kombiniert: {len(combined_data)} Datensätze, geschätzter Wert: {combined_value}")
            
            return {
                'anonymized_data': combined_data,
                'metadata': metadata,
                'c2d_asset': c2d_asset,
                'source_assets': [source.get('c2d_asset', {}) for source in sources if 'c2d_asset' in source]
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Kombination von Datenquellen: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def estimate_data_value(self, data: pd.DataFrame, metadata: Dict = None) -> Dict:
        """
        Schätzt den Wert eines Datensatzes für die Monetarisierung.
        
        Args:
            data: DataFrame mit den zu bewertenden Daten
            metadata: Zusätzliche Metadaten für die Bewertung
            
        Returns:
            Dict: Geschätzter Wert und Bewertungsfaktoren
        """
        try:
            if data is None or data.empty:
                logger.warning("Leere Daten für die Wertschätzung")
                return {
                    'normalized_score': 0.0,
                    'estimated_token_value': 0.0,
                    'metrics': {},
                    'summary': "Keine Daten zur Bewertung"
                }
            
            logger.info(f"Wertschätzung für Datensatz mit {len(data)} Datensätzen")
            
            metadata = metadata or {}
            
            # Basiswert nach Datengröße
            base_value = min(1.0, len(data) / 1000) * 5  # Skalieren nach Größe, max 5 OCEAN
            
            # Faktor nach Datenqualität (weniger fehlende Werte = höherer Wert)
            data_quality = 1.0 - data.isna().mean().mean()
            
            # Faktor nach Spaltenanzahl
            data_diversity = min(1.0, len(data.columns) / 10) * 0.5 + 0.5  # 0.5 bis 1.0 basierend auf Spaltenanzahl
            
            # Spezielle Werterhöhung für bestimmte Datentypen
            source_type = metadata.get('source_type', '')
            source_bonus = {
                'health_data': 1.5,
                'smartwatch': 1.3,
                'browser': 1.2,
                'social_media': 1.4,
                'streaming': 1.1,
                'iot': 1.2
            }.get(source_type, 1.0)
            
            # Zeitliche Relevanz berechnen
            time_relevance = 0.8  # Standardwert
            
            # Prüfe auf Zeitstempelspalten
            time_cols = [col for col in data.columns if col.lower() in ['timestamp', 'date', 'time'] 
                       or 'time' in col.lower() or 'date' in col.lower()]
            
            if time_cols:
                # Berechne zeitliche Relevanz basierend auf der Aktualität der Daten
                time_col = time_cols[0]
                try:
                    data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
                    latest_date = data[time_col].max()
                    if pd.notna(latest_date):
                        days_since_latest = (datetime.now() - latest_date).days
                        time_relevance = max(0.2, min(1.0, 1.0 - (days_since_latest / 365)))
                except:
                    pass  # Bei Fehler Standardwert verwenden
            
            # Wertfaktoren
            volume_factor = min(1.0, len(data) / 1000)
            quality_factor = data_quality
            uniqueness_factor = data_diversity
            relevance_factor = time_relevance
            
            # Gewichte für die Faktoren
            weights = {
                'volume': 0.3,
                'quality': 0.3,
                'uniqueness': 0.2,
                'relevance': 0.2
            }
            
            # Gewichteter Gesamtwert
            value_factors = {
                'data_volume': volume_factor,
                'data_quality': quality_factor,
                'data_uniqueness': uniqueness_factor,
                'time_relevance': relevance_factor
            }
            
            weighted_sum = sum(factor * weights[name.split('_')[1]] for name, factor in value_factors.items())
            
            # Normalisierte Bewertung (0-1)
            normalized_score = weighted_sum / sum(weights.values())
            
            # Geschätzter Token-Wert basierend auf der Bewertung und dem Datentyp
            estimated_value = base_value * (0.5 + 0.5 * normalized_score) * source_bonus
            
            # Erklärungen für die Faktoren
            explanations = {
                'data_volume': ("Datenmenge ist ein wichtiger Faktor für die Wertsteigerung. "
                             f"Mit {len(data)} Datensätzen erreicht dieser Datensatz "
                             f"einen Volumenfaktor von {volume_factor:.2f}."),
                             
                'data_quality': ("Datenqualität beeinflusst direkt die Nutzbarkeit. "
                              f"Dieser Datensatz hat eine Qualitätsbewertung von {quality_factor:.2f}, "
                              "basierend auf der Vollständigkeit der Daten."),
                              
                'data_uniqueness': ("Die Vielfalt der Datenpunkte bestimmt den Informationsgehalt. "
                                 f"Mit {len(data.columns)} Spalten erreicht dieser Datensatz "
                                 f"einen Diversitätsfaktor von {uniqueness_factor:.2f}."),
                                 
                'time_relevance': ("Die zeitliche Relevanz bewertet die Aktualität der Daten. "
                                f"Dieser Datensatz hat einen Relevanzfaktor von {relevance_factor:.2f}.")
            }
            
            # Erstelle Metrics-Objekt mit Detail-Informationen
            metrics = {}
            for name, factor in value_factors.items():
                metrics[name] = {
                    'score': factor,
                    'weight': weights[name.split('_')[1]],
                    'explanation': explanations[name]
                }
            
            # Erstelle Zusammenfassung
            if source_type:
                summary = (f"Dieser {source_type}-Datensatz hat einen geschätzten Wert von {estimated_value:.2f} OCEAN-Token. "
                        f"Mit {len(data)} Datensätzen und {len(data.columns)} Features erreicht er eine Bewertung von {normalized_score:.2f}.")
            else:
                summary = (f"Dieser Datensatz hat einen geschätzten Wert von {estimated_value:.2f} OCEAN-Token. "
                        f"Mit {len(data)} Datensätzen und {len(data.columns)} Features erreicht er eine Bewertung von {normalized_score:.2f}.")
            
            logger.info(f"Datenwert geschätzt: {estimated_value:.2f} OCEAN (Score: {normalized_score:.2f})")
            
            return {
                'normalized_score': float(normalized_score),
                'estimated_token_value': float(estimated_value),
                'metrics': metrics,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenwertschätzung: {str(e)}")
            return {
                'normalized_score': 0.5,  # Standardwert bei Fehler
                'estimated_token_value': 3.0,  # Standardwert bei Fehler
                'metrics': {},
                'summary': f"Fehler bei der Wertschätzung: {str(e)}"
            }
    
    def prepare_for_ocean_tokenization(self, data_asset: Dict) -> Dict:
        """
        Bereitet ein Daten-Asset für die Tokenisierung mit Ocean Protocol vor.
        
        Args:
            data_asset: Das zu tokenisierende Daten-Asset
            
        Returns:
            Dict: Vorbereitetes Asset für Ocean Protocol
        """
        try:
            if 'metadata' not in data_asset:
                logger.warning("Keine Metadaten im Daten-Asset für Tokenisierung")
                return {
                    'status': 'error',
                    'error': 'Keine Metadaten im Daten-Asset'
                }
            
            metadata = data_asset['metadata']
            asset_id = data_asset.get('c2d_asset', {}).get('asset_id', str(uuid.uuid4()))
            
            logger.info(f"Vorbereitung von Asset {asset_id} für Ocean-Tokenisierung")
            
            # Erstelle einen Asset-Namen
            asset_name = metadata.get('name', f"Dataset {asset_id[:8]}")
            
            # Erstelle eine Beschreibung
            description = metadata.get('description', f"Data asset of type {metadata.get('source_type', 'unknown')}")
            
            # Erstelle ein Token-Symbol
            symbol = metadata.get('symbol', f"DT{datetime.now().strftime('%m%d')}")
            
            # Erstelle DDO-Metadaten
            ddo_metadata = {
                'name': asset_name,
                'type': "dataset",
                'description': description,
                'author': metadata.get('author', "OceanData User"),
                'license': metadata.get('license', "CC BY-NC-SA"),
                'tags': metadata.get('tags', [metadata.get('source_type', 'data')]),
                'additionalInformation': {
                    'source_type': metadata.get('source_type', 'unknown'),
                    'record_count': metadata.get('record_count', 0),
                    'field_count': metadata.get('field_count', 0),
                    'privacy_level': metadata.get('privacy_level', 'medium'),
                    'created_at': metadata.get('created_at', datetime.now().isoformat())
                }
            }
            
            # Bestimme, ob Compute-to-Data aktiviert werden soll
            use_c2d = 'privacy_level' in metadata and metadata['privacy_level'] in ['medium', 'high']
            
            # Erstelle Service-Definition
            service_type = "compute" if use_c2d else "access"
            
            # Erstelle Preisinformationen
            pricing = {
                'type': "fixed",
                'baseTokenAmount': metadata.get('estimated_value', 5.0),
                'marketplace': "OCEANMARKET",
            }
            
            # Erstelle Ocean Asset
            ocean_asset = {
                'ddo': {
                    'id': asset_id,
                    'created': datetime.now().isoformat(),
                    'updated': datetime.now().isoformat(),
                    'type': "dataset",
                    'name': asset_name,
                    'description': description,
                    'tags': ddo_metadata['tags'],
                    'price': metadata.get('estimated_value', 5.0)
                },
                'pricing': pricing,
                'service_type': service_type,
                'metadata': ddo_metadata,
                'asset_id': asset_id
            }
            
            logger.info(f"Asset {asset_id} für Ocean-Tokenisierung vorbereitet")
            
            return ocean_asset
            
        except Exception as e:
            logger.error(f"Fehler bei der Vorbereitung für Ocean-Tokenisierung: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def tokenize_with_ocean(self, ocean_asset: Dict, wallet_address: str = None) -> Dict:
        """
        Tokenisiert ein Daten-Asset mit Ocean Protocol.
        
        Args:
            ocean_asset: Das vorbereitete Asset für Ocean Protocol
            wallet_address: Adresse der Wallet für die Transaktion (optional)
            
        Returns:
            Dict: Ergebnis der Tokenisierung
        """
        try:
            if 'asset_id' not in ocean_asset or 'metadata' not in ocean_asset:
                logger.warning("Ungültiges Ocean-Asset für Tokenisierung")
                return {
                    'success': False,
                    'error': 'Ungültiges Ocean-Asset'
                }
            
            asset_id = ocean_asset['asset_id']
            metadata = ocean_asset['metadata']
            
            logger.info(f"Tokenisierung von Asset {asset_id} mit Ocean Protocol")
            
            # Erstelle Dataset-Info für die Ocean-Integration
            dataset_info = {
                'name': metadata['name'],
                'symbol': metadata.get('symbol', f"DT{datetime.now().strftime('%m%d')}"),
                'description': metadata['description'],
                'author': metadata['author'],
                'license': metadata['license'],
                'type': metadata['type'],
                'tags': metadata['tags'],
                'compute_to_data': ocean_asset['service_type'] == 'compute',
                'files': []  # In einer realen Implementierung würden hier Datei-Referenzen stehen
            }
            
            # Erstelle Pricing-Options
            pricing_options = {
                'type': 'fixed',
                'price': ocean_asset['pricing']['baseTokenAmount']
            }
            
            # Tokenisiere den Datensatz
            tokenization_result = self.ocean.tokenize_dataset(dataset_info, pricing_options, wallet_address)
            
            if 'error' in tokenization_result:
                logger.error(f"Fehler bei der Ocean-Tokenisierung: {tokenization_result['error']}")
                return {
                    'success': False,
                    'error': tokenization_result['error']
                }
            
            logger.info(f"Asset {asset_id} erfolgreich mit Ocean Protocol tokenisiert")
            
            # Erstelle ein einheitliches Ergebnisformat
            result = {
                'success': True,
                'asset_id': asset_id,
                'token_address': tokenization_result['datatoken']['address'],
                'token_symbol': tokenization_result['datatoken']['symbol'],
                'token_name': tokenization_result['datatoken']['name'],
                'token_price': tokenization_result['pricing'].get('price', 
                                                               tokenization_result['pricing'].get('initial_price', 0)),
                'transaction_hash': tokenization_result.get('transaction_hash', ''),
                'owner': tokenization_result.get('owner', wallet_address),
                'marketplace_url': tokenization_result.get('marketplace_url', '')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei der Ocean-Tokenisierung: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_protected_fields(self, data: pd.DataFrame, source_type: str, privacy_level: PrivacyLevel) -> Dict:
        """
        Bestimmt zu schützende Felder basierend auf Datenquellentyp und Datenschutzniveau.
        
        Args:
            data: DataFrame mit den Daten
            source_type: Art der Datenquelle
            privacy_level: Datenschutzniveau
            
        Returns:
            Dict: Zu schützende Felder und ihre Schutzniveaus
        """
        protected_fields = {}
        
        # Allgemeine schützenswerte Felder
        sensitive_patterns = ['user', 'name', 'email', 'phone', 'address', 'location', 'ip', 'password', 'ssn', 'id']
        
        # Bestimme die zu schützenden Felder basierend auf Spaltenname
        for column in data.columns:
            col_lower = column.lower()
            
            # Prüfe auf sensitive Muster im Spaltennamen
            if any(pattern in col_lower for pattern in sensitive_patterns):
                if privacy_level == PrivacyLevel.PUBLIC:
                    protected_fields[column] = PrivacyLevel.ANONYMIZED
                else:
                    protected_fields[column] = privacy_level
            
        # Spezifische Felder je nach Datenquellentyp
        if source_type == 'browser':
            browser_sensitive = ['url', 'search_term', 'download_path']
            for column in data.columns:
                col_lower = column.lower()
                if any(field in col_lower for field in browser_sensitive):
                    protected_fields[column] = privacy_level
        
        elif source_type in ['smartwatch', 'health_data']:
            health_sensitive = ['heart_rate', 'blood', 'pressure', 'sleep', 'diagnosis', 'medication']
            health_very_sensitive = ['disease', 'condition', 'symptom', 'treatment', 'ecg', 'ekg']
            
            for column in data.columns:
                col_lower = column.lower()
                if any(field in col_lower for field in health_sensitive):
                    protected_fields[column] = privacy_level
                elif any(field in col_lower for field in health_very_sensitive):
                    # Sehr sensitive Gesundheitsdaten immer strenger schützen
                    protected_fields[column] = PrivacyLevel.SENSITIVE
        
        elif source_type == 'social_media':
            social_sensitive = ['post', 'message', 'friend', 'connection', 'like', 'comment']
            for column in data.columns:
                col_lower = column.lower()
                if any(field in col_lower for field in social_sensitive):
                    protected_fields[column] = privacy_level
        
        return protected_fields
    
    def _anonymize_field(self, data: pd.DataFrame, field: str) -> pd.DataFrame:
        """
        Anonymisiert ein bestimmtes Feld in einem DataFrame.
        
        Args:
            data: DataFrame mit den Daten
            field: Name des zu anonymisierenden Feldes
            
        Returns:
            pd.DataFrame: DataFrame mit anonymisiertem Feld
        """
        anonymized_data = data.copy()
        
        # Verschiedene Anonymisierungsstrategien je nach Datentyp
        if data[field].dtype == 'object':
            # Text oder kategoriale Daten
            if 'id' in field.lower() or 'user' in field.lower() or 'name' in field.lower() or 'email' in field.lower():
                # Hash-basierte Anonymisierung
                salt = datetime.now().strftime("%Y%m%d")
                anonymized_data[field] = anonymized_data[field].apply(
                    lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest() if pd.notna(x) else x
                )
            elif 'location' in field.lower() or 'address' in field.lower() or 'ip' in field.lower():
                # Grobe Kategorisierung
                anonymized_data[field] = anonymized_data[field].apply(
                    lambda x: f"anonymized_{hashlib.md5(str(x).encode()).hexdigest()[:8]}" if pd.notna(x) else x
                )
            else:
                # Allgemeine Kategorisierung für Text
                categories = pd.Categorical(anonymized_data[field]).codes
                anonymized_data[field] = [f"category_{code}" if code >= 0 else None for code in categories]
        
        elif pd.api.types.is_numeric_dtype(data[field]):
            # Numerische Daten
            # Perzentil-basierte Bucketing (Binning)
            quantiles = [0, 0.25, 0.5, 0.75, 1.0]
            bins = data[field].quantile(quantiles).values
            labels = [f"Q{i+1}" for i in range(len(bins)-1)]
            anonymized_data[field] = pd.cut(data[field], bins=bins, labels=labels, include_lowest=True)
        
        return anonymized_data
    
    def _detect_time_granularity(self, time_series: pd.Series) -> str:
        """
        Erkennt die Granularität einer Zeitreihe.
        
        Args:
            time_series: Serie mit Zeitstempeln
            
        Returns:
            str: Erkannte Granularität ('hourly', 'daily', 'weekly', 'monthly')
        """
        # Sortiere die Zeitreihe
        time_series = time_series.sort_values()
        
        # Berechne Zeitdifferenzen
        if len(time_series) < 2:
            return 'unknown'
            
        time_diffs = time_series.diff().dropna()
        
        # Berechne durchschnittliche Zeitdifferenz in Stunden
        avg_diff_hours = time_diffs.mean().total_seconds() / 3600
        
        # Bestimme Granularität basierend auf der durchschnittlichen Differenz
        if avg_diff_hours <= 1:
            return 'minutely'
        elif avg_diff_hours <= 3:
            return 'hourly'
        elif avg_diff_hours <= 36:
            return 'daily'
        elif avg_diff_hours <= 8 * 24:
            return 'weekly'
        else:
            return 'monthly'
    
    def _check_periodicity(self, time_series: pd.Series) -> bool:
        """
        Prüft, ob eine Zeitreihe periodisch ist.
        
        Args:
            time_series: Serie mit Zeitstempeln
            
        Returns:
            bool: True, wenn die Zeitreihe periodisch ist, sonst False
        """
        # Eine einfache Heuristik: Prüfe die Varianz der Zeitdifferenzen
        if len(time_series) < 3:
            return False
            
        time_diffs = time_series.diff().dropna()
        
        # Berechne Variationskoeffizient (Standardabweichung / Mittelwert)
        cv = time_diffs.std() / time_diffs.mean() if time_diffs.mean().total_seconds() > 0 else float('inf')
        
        # Wenn der Variationskoeffizient klein ist, sind die Zeitdifferenzen regelmäßig
        return cv < 0.5
    
    def _analyze_browser_data(self, data: pd.DataFrame) -> Dict:
        """
        Führt spezifische Analysen für Browser-Daten durch.
        
        Args:
            data: DataFrame mit Browser-Daten
            
        Returns:
            Dict: Browser-spezifische Analyseergebnisse
        """
        analysis = {}
        
        # Domain-Analyse
        if 'url' in data.columns:
            # Extrahiere Domain aus URL
            try:
                data['domain'] = data['url'].str.extract(r'(?:https?://)?(?:www\.)?([^/]+)', expand=False)
                domain_counts = data['domain'].value_counts()
                
                analysis['top_domains'] = domain_counts.head(5).to_dict()
                analysis['domain_diversity'] = len(domain_counts)
            except:
                pass
        
        # Suchterm-Analyse
        if 'search_term' in data.columns:
            search_data = data.dropna(subset=['search_term'])
            if not search_data.empty:
                # Einfache Wortzählung
                all_terms = ' '.join(search_data['search_term'].astype(str))
                words = all_terms.lower().split()
                word_counts = pd.Series(words).value_counts()
                
                analysis['top_search_terms'] = word_counts.head(10).to_dict()
                analysis['search_term_count'] = len(search_data)
        
        # Zeitliche Nutzungsmuster
        if 'timestamp' in data.columns:
            try:
                data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
                data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
                
                hourly_activity = data.groupby('hour').size()
                daily_activity = data.groupby('day_of_week').size()
                
                analysis['hourly_activity'] = hourly_activity.to_dict()
                analysis['daily_activity'] = daily_activity.to_dict()
                analysis['peak_hour'] = hourly_activity.idxmax()
                analysis['peak_day'] = daily_activity.idxmax()
            except:
                pass
        
        return analysis
    
    def _analyze_health_data(self, data: pd.DataFrame) -> Dict:
        """
        Führt spezifische Analysen für Gesundheitsdaten durch.
        
        Args:
            data: DataFrame mit Gesundheitsdaten
            
        Returns:
            Dict: Gesundheitsdaten-spezifische Analyseergebnisse
        """
        analysis = {}
        
        # Herzfrequenz-Analyse
        if 'heart_rate' in data.columns:
            hr_data = data.dropna(subset=['heart_rate'])
            if not hr_data.empty:
                analysis['heart_rate'] = {
                    'mean': float(hr_data['heart_rate'].mean()),
                    'min': float(hr_data['heart_rate'].min()),
                    'max': float(hr_data['heart_rate'].max()),
                    'std': float(hr_data['heart_rate'].std())
                }
        
        # Schritte-Analyse
        if 'steps' in data.columns:
            steps_data = data.dropna(subset=['steps'])
            if not steps_data.empty:
                analysis['steps'] = {
                    'total': int(steps_data['steps'].sum()),
                    'daily_average': float(steps_data['steps'].mean()),
                    'max_daily': float(steps_data['steps'].max())
                }
        
        # Schlaf-Analyse
        if 'sleep_state' in data.columns:
            sleep_data = data.dropna(subset=['sleep_state'])
            if not sleep_data.empty:
                sleep_counts = sleep_data['sleep_state'].value_counts()
                
                analysis['sleep'] = {
                    'states': sleep_counts.to_dict(),
                    'deep_sleep_ratio': sleep_counts.get('deep', 0) / len(sleep_data) if 'deep' in sleep_counts else 0,
                    'rem_sleep_ratio': sleep_counts.get('rem', 0) / len(sleep_data) if 'rem' in sleep_counts else 0
                }
        
        # Aktivitätsanalyse
        if 'activity_type' in data.columns:
            activity_data = data.dropna(subset=['activity_type'])
            if not activity_data.empty:
                activity_counts = activity_data['activity_type'].value_counts()
                
                analysis['activity'] = {
                    'types': activity_counts.to_dict(),
                    'active_ratio': sum(activity_counts.get(t, 0) for t in ['walking', 'running', 'cycling']) / len(activity_data)
                }
        
        return analysis
    
    def _analyze_social_media_data(self, data: pd.DataFrame) -> Dict:
        """
        Führt spezifische Analysen für Social-Media-Daten durch.
        
        Args:
            data: DataFrame mit Social-Media-Daten
            
        Returns:
            Dict: Social-Media-spezifische Analyseergebnisse
        """
        analysis = {}
        
        # Interaktionsanalyse
        interaction_cols = ['likes', 'comments', 'shares', 'views']
        for col in interaction_cols:
            if col in data.columns:
                int_data = data.dropna(subset=[col])
                if not int_data.empty:
                    analysis[f'{col}_stats'] = {
                        'total': int(int_data[col].sum()),
                        'mean': float(int_data[col].mean()),
                        'median': float(int_data[col].median()),
                        'max': float(int_data[col].max())
                    }
        
        # Content-Typ-Analyse
        if 'content_type' in data.columns:
            content_data = data.dropna(subset=['content_type'])
            if not content_data.empty:
                type_counts = content_data['content_type'].value_counts()
                
                analysis['content_types'] = {
                    'counts': type_counts.to_dict(),
                    'most_common': type_counts.idxmax()
                }
        
        # Hashtag-Analyse
        if 'hashtags' in data.columns:
            hashtag_data = data.dropna(subset=['hashtags'])
            if not hashtag_data.empty:
                # Annahme: Hashtags sind als Liste oder kommaseparierter String gespeichert
                all_hashtags = []
                for tags in hashtag_data['hashtags']:
                    if isinstance(tags, list):
                        all_hashtags.extend(tags)
                    elif isinstance(tags, str):
                        all_hashtags.extend([tag.strip() for tag in tags.split(',')])
                
                hashtag_counts = pd.Series(all_hashtags).value_counts()
                
                analysis['hashtags'] = {
                    'top_hashtags': hashtag_counts.head(10).to_dict(),
                    'unique_hashtags': len(hashtag_counts)
                }
        
        # Zeitliche Analyse
        if 'timestamp' in data.columns:
            try:
                data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
                data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
                
                hourly_activity = data.groupby('hour').size()
                daily_activity = data.groupby('day_of_week').size()
                
                analysis['posting_times'] = {
                    'hourly': hourly_activity.to_dict(),
                    'daily': daily_activity.to_dict(),
                    'peak_hour': hourly_activity.idxmax(),
                    'peak_day': daily_activity.idxmax()
                }
            except:
                pass
        
        return analysis
"""
OceanData - Hauptklasse OceanDataAI

Diese Klasse bildet das Herzstück der OceanData-Plattform und integriert
alle Komponenten zu einer kohärenten Anwendung für die Datenmonetarisierung.
"""

import pandas as pd
import logging
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

# Import der einzelnen Komponenten
from oceandata.data_integration.base import DataSource, DataCategory, PrivacyLevel
from oceandata.analytics.models.anomaly_detector import AnomalyDetector
from oceandata.privacy.compute_to_data import ComputeToDataManager
from oceandata.blockchain.ocean_integration import OceanIntegration

# Logging konfigurieren
logger = logging.getLogger("OceanData.Core")

class OceanDataAI:
    """
    Hauptklasse der OceanData-Plattform.
    
    Diese Klasse orchestriert alle Komponenten der Plattform:
    - Datenerfassung und -integration
    - KI-gestützte Analyse
    - Datenschutz und Compute-to-Data
    - Monetarisierung über Ocean Protocol
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialisiert die OceanDataAI-Plattform.
        
        Args:
            config: Konfigurationsdaten für die Plattform
        """
        self.config = config or {}
        
        # Initialisiere Komponenten
        self.anomaly_detector = AnomalyDetector(
            method=self.config.get('anomaly_detection_method', 'isolation_forest'),
            contamination=self.config.get('anomaly_contamination', 0.05)
        )
        
        # In einer vollständigen Implementierung würden wir weitere Komponenten initialisieren:
        # self.semantic_analyzer = SemanticAnalyzer()
        # self.predictive_modeler = PredictiveModeler()
        # self.data_synthesizer = DataSynthesizer()
        
        # Datenschutz und C2D
        self.c2d_manager = ComputeToDataManager(
            privacy_config=self.config.get('privacy_config')
        )
        
        # Ocean Protocol Integration
        self.ocean = OceanIntegration(
            config=self.config.get('ocean_config')
        )
        
        logger.info("OceanDataAI-Plattform initialisiert")
    
    def analyze_data_source(self, data: pd.DataFrame, source_type: str) -> Dict:
        """
        Führt eine umfassende Analyse einer Datenquelle durch.
        
        Args:
            data: DataFrame mit den zu analysierenden Daten
            source_type: Art der Datenquelle (z.B. 'browser', 'smartwatch')
            
        Returns:
            Dict: Analyseergebnisse
        """
        try:
            if data is None or data.empty:
                logger.warning(f"Leere Daten für die Analyse von {source_type}")
                return {
                    'source_type': source_type,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': 'Leere Daten'
                }
            
            logger.info(f"Analyse von {source_type}-Daten mit {len(data)} Datensätzen gestartet")
            
            # Grundlegende Datenstatistiken
            stats = {
                'record_count': len(data),
                'column_count': len(data.columns),
                'columns': list(data.columns),
            }
            
            # Anomalieerkennung
            numeric_data = data.select_dtypes(include=['number'])
            if not numeric_data.empty:
                self.anomaly_detector.fit(numeric_data)
                anomaly_predictions = self.anomaly_detector.predict(numeric_data)
                anomaly_scores = self.anomaly_detector.get_anomaly_scores(numeric_data)
                
                # Erkenntnisse über Anomalien
                anomaly_insights = self.anomaly_detector.get_anomaly_insights(numeric_data, anomaly_predictions)
                
                # Anomaliestatistiken
                anomaly_count = sum(1 for pred in anomaly_predictions if pred == -1)
                anomaly_percentage = (anomaly_count / len(data)) * 100
                
                anomaly_analysis = {
                    'count': anomaly_count,
                    'percentage': anomaly_percentage,
                    'insights': anomaly_insights[:5]  # Limitiere auf Top-5-Erkenntnisse
                }
            else:
                anomaly_analysis = {
                    'count': 0,
                    'percentage': 0,
                    'insights': []
                }
            
            # In einer vollständigen Implementierung würden wir weitere Analysen durchführen:
            # - Semantic Analysis für Text
            # - Zeitreihenanalyse für zeitbasierte Daten
            # - Predictive Modeling
            
            # Zeitbasierte Analyse, falls Zeitstempel vorhanden sind
            time_series_analysis = {}
            if 'timestamp' in data.columns or any(col.lower().endswith('time') or col.lower().endswith('date') for col in data.columns):
                # Identifiziere die Zeitstempelspalte
                time_col = 'timestamp'
                if 'timestamp' not in data.columns:
                    for col in data.columns:
                        if col.lower().endswith('time') or col.lower().endswith('date'):
                            time_col = col
                            break
                
                # Einfache Zeitreihenanalyse
                if time_col in data.columns:
                    data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
                    valid_times = data.dropna(subset=[time_col])
                    
                    if not valid_times.empty:
                        # Zeiträume
                        time_range = {
                            'start': valid_times[time_col].min().isoformat(),
                            'end': valid_times[time_col].max().isoformat(),
                            'duration_hours': (valid_times[time_col].max() - valid_times[time_col].min()).total_seconds() / 3600
                        }
                        
                        time_series_analysis = {
                            'time_column': time_col,
                            'time_range': time_range,
                            'granularity': self._detect_time_granularity(valid_times[time_col]),
                            'is_periodic': self._check_periodicity(valid_times[time_col]),
                            'forecast_horizon': 7  # Standard-Vorhersagehorizont
                        }
            
            # Spezifische Analysen je nach Datenquellentyp
            source_specific_analysis = {}
            
            if source_type == 'browser':
                # Browser-spezifische Analyse
                source_specific_analysis = self._analyze_browser_data(data)
            elif source_type == 'smartwatch' or source_type == 'health_data':
                # Gesundheitsdaten-spezifische Analyse
                source_specific_analysis = self._analyze_health_data(data)
            elif source_type == 'social_media':
                # Social-Media-spezifische Analyse
                source_specific_analysis = self._analyze_social_media_data(data)
            
            # Kombiniere alle Analyseergebnisse
            result = {
                'source_type': source_type,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'statistics': stats,
                'analyses': {
                    'anomalies': anomaly_analysis,
                    'time_series': time_series_analysis,
                    'source_specific': source_specific_analysis
                }
            }
            
            logger.info(f"Analyse von {source_type}-Daten abgeschlossen")
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenanalyse: {str(e)}")
            return {
                'source_type': source_type,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def prepare_data_for_monetization(self, data: pd.DataFrame, source_type: str, 
                                    privacy_level: str = 'medium') -> Dict:
        """
        Bereitet Daten für die Monetarisierung vor.
        
        Args:
            data: DataFrame mit den zu monetarisierenden Daten
            source_type: Art der Datenquelle
            privacy_level: Datenschutzniveau ('low', 'medium', 'high')
            
        Returns:
            Dict: Vorbereitete Daten für die Monetarisierung
        """
        try:
            if data is None or data.empty:
                logger.warning(f"Leere Daten für die Monetarisierung von {source_type}")
                return {
                    'status': 'error',
                    'error': 'Leere Daten'
                }
            
            logger.info(f"Vorbereitung von {source_type}-Daten für Monetarisierung mit Privacy-Level {privacy_level}")
            
            # Konvertiere Datenschutzniveau in PrivacyLevel-Enum
            privacy_mapping = {
                'low': PrivacyLevel.PUBLIC,
                'medium': PrivacyLevel.ANONYMIZED,
                'high': PrivacyLevel.ENCRYPTED
            }
            privacy_enum = privacy_mapping.get(privacy_level.lower(), PrivacyLevel.ANONYMIZED)
            
            # Bestimme zu schützende Felder basierend auf Datenquellentyp und Datenschutzniveau
            protected_fields = self._get_protected_fields(data, source_type, privacy_enum)
            
            # Wende Datenschutzmaßnahmen an
            anonymized_data = data.copy()
            
            for field, level in protected_fields.items():
                if field in anonymized_data.columns:
                    if level == PrivacyLevel.ANONYMIZED:
                        # Anonymisiere das Feld
                        anonymized_data = self._anonymize_field(anonymized_data, field)
                    elif level == PrivacyLevel.ENCRYPTED:
                        # Verschlüssele oder entferne das Feld
                        anonymized_data = anonymized_data.drop(field, axis=1)
                    elif level == PrivacyLevel.SENSITIVE:
                        # Entferne sensitive Daten
                        anonymized_data = anonymized_data.drop(field, axis=1)
            
            # Erstelle C2D-Asset für geschützte Daten
            c2d_asset = self.c2d_manager.create_data_asset(data, {
                'source_type': source_type,
                'privacy_level': privacy_level,
                'original_columns': list(data.columns),
                'preserved_columns': list(anonymized_data.columns)
            })
            
            # Schätze den Wert der Daten basierend auf Typus und Umfang
            analysis_result = self.analyze_data_source(data, source_type)
            value_estimation = self.estimate_data_value(data, {
                'source_type': source_type,
                'analysis_result': analysis_result
            })
            
            # Berücksichtige Datenschutzniveau bei der Wertschätzung
            privacy_factor = {
                'low': 1.2,     # Höherer Wert bei geringem Datenschutz
                'medium': 1.0,  # Standardwert
                'high': 0.8     # Geringerer Wert bei hohem Datenschutz (weniger nutzbare Daten)
            }.get(privacy_level, 1.0)
            
            adjusted_value = value_estimation['estimated_token_value'] * privacy_factor
            
            # Erstelle Metadaten für die Monetarisierung
            metadata = {
                'source_type': source_type,
                'privacy_level': privacy_level,
                'record_count': len(data),
                'field_count': len(anonymized_data.columns),
                'original_field_count': len(data.columns),
                'estimated_value': adjusted_value,
                'adjusted_privacy_factor': privacy_factor,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"Daten für Monetarisierung vorbereitet: {len(anonymized_data)} Datensätze, geschätzter Wert: {adjusted_value}")
            
            return {
                'anonymized_data': anonymized_data,
                'metadata': metadata,
                'c2d_asset': c2d_asset,
                'value_estimation': value_estimation
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenvorbereitung für Monetarisierung: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def combine_data_sources(self, sources: List[Dict], combination_type: str = 'merge') -> Dict:
        """
        Kombiniert mehrere Datenquellen zu einem wertvolleren Asset.
        
        Args:
            sources: Liste der zu kombinierenden Datenquellen
            combination_type: Art der Kombination ('merge', 'enrich', 'correlate')
            
        Returns:
            Dict: Kombiniertes Daten-Asset
        """
        try:
            if not sources:
                logger.warning("Keine Datenquellen zum Kombinieren angegeben")
                return {
                    'status': 'error',
                    'error': 'Keine Datenquellen angegeben'
                }
            
            logger.info(f"Kombination von {len(sources)} Datenquellen mit Methode '{combination_type}'")
            
            # Extrahiere Daten und Metadaten
            data_frames = []
            metadata_list = []
            
            for source in sources:
                if 'anonymized_data' in source and source['anonymized_data'] is not None:
                    data_frames.append(source['anonymized_data'])
                if 'metadata' in source:
                    metadata_list.append(source['metadata'])
            
            if not data_frames:
                logger.warning("Keine gültigen Daten in den Datenquellen")
                return {
                    'status': 'error',
                    'error': 'Keine gültigen Daten in den Datenquellen'
                }
            
            # Kombiniere die Daten je nach Kombinationstyp
            if combination_type == 'merge':
                # Einfaches vertikales Anfügen (mehr Datensätze)
                combined_data = pd.concat(data_frames, ignore_index=True)
                
            elif combination_type == 'enrich':
                # Horizontale Anreicherung (mehr Features)
                # Nur die ersten beiden Quellen werden angereichert
                combined_data = data_frames[0].copy()
                
                if len(data_frames) > 1:
