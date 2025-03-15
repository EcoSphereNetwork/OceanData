"""
OceanData - Compute-to-Data Manager

Dieses Modul implementiert Compute-to-Data-Funktionalität für OceanData.
Dadurch wird die sichere und private Nutzung sensibler Daten ermöglicht, 
ohne dass die Daten direkt mit Dritten geteilt werden müssen.
"""

import pandas as pd
import numpy as np
import json
import logging
import os
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from io import StringIO
from cryptography.fernet import Fernet

# Logging konfigurieren
logger = logging.getLogger("OceanData.Privacy.ComputeToData")

class ComputeToDataManager:
    """
    Manager für Compute-to-Data-Funktionalität.
    
    Diese Klasse ermöglicht:
    - Sichere Speicherung und Verschlüsselung von Daten
    - Ausführung von Berechnungen auf verschlüsselten Daten, ohne sie direkt offenzulegen
    - Generierung und Validierung von Zugriffstoken
    - Verwaltung von erlaubten Operationen und Zugriffsrechten
    """
    
    def __init__(self, encryption_key=None, privacy_config=None):
        """
        Initialisiert den Compute-to-Data-Manager.
        
        Args:
            encryption_key: Optional. Schlüssel für die Verschlüsselung von Daten.
            privacy_config: Optional. Konfiguration für Datenschutzeinstellungen.
        """
        # Verschlüsselung initialisieren
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Standardoperationen definieren
        self.allowed_operations = {
            'aggregate': self._aggregate,
            'count': self._count,
            'mean': self._mean,
            'sum': self._sum,
            'min': self._min,
            'max': self._max,
            'correlation': self._correlation,
            'histogram': self._histogram,
            'distribution': self._distribution,
            'custom_model': self._custom_model_inference
        }
        
        # Datenschutzkonfiguration
        self.privacy_config = privacy_config or {
            'min_group_size': 5,  # Minimale Gruppengröße für Aggregationen
            'noise_level': 0.01,  # Rauschpegel für Differentiellen Datenschutz
            'outlier_removal': True,  # Ausreißer entfernen für Differentiellen Datenschutz
            'max_query_per_token': 10  # Maximale Anzahl von Abfragen pro Token
        }
        
        # Asset-Speicher
        self.assets = {}  # asset_id -> asset_info
        self.tokens = {}  # token_id -> token_info
        
        logger.info("Compute-to-Data-Manager initialisiert")
    
    def _encrypt_data(self, data: pd.DataFrame) -> bytes:
        """
        Verschlüsselt Daten für die sichere Speicherung.
        
        Args:
            data: DataFrame mit zu verschlüsselnden Daten
            
        Returns:
            bytes: Verschlüsselte Daten
        """
        try:
            # Konvertiere DataFrame zu JSON
            serialized = data.to_json().encode()
            
            # Verschlüssele die Daten
            encrypted = self.cipher_suite.encrypt(serialized)
            
            return encrypted
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenverschlüsselung: {str(e)}")
            raise
    
    def _decrypt_data(self, encrypted_data: bytes) -> pd.DataFrame:
        """
        Entschlüsselt Daten für die Verarbeitung.
        
        Args:
            encrypted_data: Verschlüsselte Daten
            
        Returns:
            pd.DataFrame: Entschlüsselte Daten als DataFrame
        """
        try:
            # Entschlüssele die Daten
            decrypted = self.cipher_suite.decrypt(encrypted_data)
            
            # Konvertiere JSON zu DataFrame
            data = pd.read_json(StringIO(decrypted.decode()))
            
            return data
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenentschlüsselung: {str(e)}")
            raise
    
    def _add_privacy_noise(self, result: Any) -> Any:
        """
        Fügt Rauschen zu Ergebnissen hinzu, um Differentiellen Datenschutz zu gewährleisten.
        
        Args:
            result: Das ursprüngliche Ergebnis
            
        Returns:
            Any: Das Ergebnis mit Rauschen
        """
        try:
            noise_level = self.privacy_config['noise_level']
            
            if isinstance(result, (int, float)):
                # Füge normalverteiltes Rauschen hinzu
                noise = np.random.normal(0, noise_level * abs(result) + 1e-6)
                return result + noise
                
            elif isinstance(result, dict):
                # Durchlaufe rekursiv alle Werte im Dictionary
                noisy_result = {}
                for key, value in result.items():
                    noisy_result[key] = self._add_privacy_noise(value)
                return noisy_result
                
            elif isinstance(result, list):
                # Durchlaufe rekursiv alle Elemente in der Liste
                return [self._add_privacy_noise(item) for item in result]
                
            elif isinstance(result, np.ndarray):
                # Füge Rauschen zu jedem Element im Array hinzu
                noise = np.random.normal(0, noise_level * np.abs(result) + 1e-6, result.shape)
                return result + noise
                
            else:
                # Nicht-numerische Typen unverändert lassen
                return result
                
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Privatsphäre-Rauschen: {str(e)}")
            return result
    
    def _check_min_group_size(self, data: pd.DataFrame, params: Dict) -> bool:
        """
        Prüft, ob gruppierte Daten die Mindestgruppengröße erfüllen.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation
            
        Returns:
            bool: True wenn die Mindestgruppengröße erfüllt ist, sonst False
        """
        min_size = self.privacy_config['min_group_size']
        
        try:
            # Wenn keine Gruppierung angegeben ist, prüfe die Gesamtgröße
            if 'group_by' not in params:
                return len(data) >= min_size
                
            # Bei Gruppierung: Prüfe jede Gruppe
            group_cols = params['group_by']
            if not isinstance(group_cols, list):
                group_cols = [group_cols]
                
            grouped = data.groupby(group_cols).size()
            return grouped.min() >= min_size
            
        except Exception as e:
            logger.error(f"Fehler bei der Prüfung der Mindestgruppengröße: {str(e)}")
            return False
    
    # Implementierung von erlaubten Operationen
    
    def _aggregate(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Aggregiert Daten nach Spalten und Gruppen.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'group_by', 'aggregates')
            
        Returns:
            Dict: Aggregierte Daten
        """
        try:
            # Extrahiere Parameter
            group_by = params.get('group_by', None)
            aggregates = params.get('aggregates', {'count': 'count'})
            
            # Prüfe die Mindestgruppengröße
            if not self._check_min_group_size(data, params):
                return {'error': 'Mindestgruppengröße nicht erfüllt'}
            
            # Führe die Aggregation durch
            if group_by:
                if not isinstance(group_by, list):
                    group_by = [group_by]
                
                # Gruppierte Aggregation
                result = data.groupby(group_by).agg(aggregates)
                
                # Konvertiere das Ergebnis in eine serialisierbare Form
                result_dict = {}
                for group, values in result.to_dict('index').items():
                    if isinstance(group, tuple):
                        group_key = '_'.join([str(g) for g in group])
                    else:
                        group_key = str(group)
                    result_dict[group_key] = values
                
                return {'aggregated_data': result_dict}
            else:
                # Globale Aggregation
                result = data.agg(aggregates)
                
                # Konvertiere das Ergebnis in eine serialisierbare Form
                return {'aggregated_data': result.to_dict()}
                
        except Exception as e:
            logger.error(f"Fehler bei der Aggregation: {str(e)}")
            return {'error': f'Fehler bei der Aggregation: {str(e)}'}
    
    def _count(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Zählt Datensätze nach Filterkriterien.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'filter_column', 'filter_value')
            
        Returns:
            Dict: Anzahl der Datensätze
        """
        try:
            filter_column = params.get('filter_column')
            filter_value = params.get('filter_value')
            
            if filter_column and filter_value is not None:
                # Gefilterte Zählung
                filtered_data = data[data[filter_column] == filter_value]
                
                # Prüfe die Mindestgruppengröße
                if len(filtered_data) < self.privacy_config['min_group_size']:
                    return {'error': 'Mindestgruppengröße nicht erfüllt'}
                    
                count = len(filtered_data)
            else:
                # Gesamtzählung
                count = len(data)
            
            # Füge Rauschen hinzu
            noisy_count = self._add_privacy_noise(count)
            
            return {'count': round(noisy_count)}
            
        except Exception as e:
            logger.error(f"Fehler bei der Zählung: {str(e)}")
            return {'error': f'Fehler bei der Zählung: {str(e)}'}
    
    def _mean(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Berechnet den Mittelwert für numerische Spalten.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'columns')
            
        Returns:
            Dict: Mittelwerte
        """
        try:
            # Extrahiere Parameter
            columns = params.get('columns')
            
            # Wenn keine Spalten angegeben, verwende alle numerischen
            if not columns:
                columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # Berechne Mittelwerte
            means = {}
            for col in columns:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    # Entferne Ausreißer, wenn konfiguriert
                    if self.privacy_config['outlier_removal']:
                        q1 = data[col].quantile(0.25)
                        q3 = data[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        filtered_data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                    else:
                        filtered_data = data
                    
                    # Prüfe die Mindestgruppengröße
                    if len(filtered_data) < self.privacy_config['min_group_size']:
                        means[col] = None
                        continue
                    
                    # Berechne Mittelwert
                    mean_value = float(filtered_data[col].mean())
                    
                    # Füge Rauschen hinzu
                    noisy_mean = self._add_privacy_noise(mean_value)
                    means[col] = noisy_mean
            
            return {'means': means}
            
        except Exception as e:
            logger.error(f"Fehler bei der Mittelwertberechnung: {str(e)}")
            return {'error': f'Fehler bei der Mittelwertberechnung: {str(e)}'}
    
    def _sum(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Berechnet die Summe für numerische Spalten.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'columns')
            
        Returns:
            Dict: Summen
        """
        try:
            # Extrahiere Parameter
            columns = params.get('columns')
            
            # Wenn keine Spalten angegeben, verwende alle numerischen
            if not columns:
                columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # Berechne Summen
            sums = {}
            for col in columns:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    # Entferne Ausreißer, wenn konfiguriert
                    if self.privacy_config['outlier_removal']:
                        q1 = data[col].quantile(0.25)
                        q3 = data[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        filtered_data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                    else:
                        filtered_data = data
                    
                    # Prüfe die Mindestgruppengröße
                    if len(filtered_data) < self.privacy_config['min_group_size']:
                        sums[col] = None
                        continue
                    
                    # Berechne Summe
                    sum_value = float(filtered_data[col].sum())
                    
                    # Füge Rauschen hinzu
                    noisy_sum = self._add_privacy_noise(sum_value)
                    sums[col] = noisy_sum
            
            return {'sums': sums}
            
        except Exception as e:
            logger.error(f"Fehler bei der Summenberechnung: {str(e)}")
            return {'error': f'Fehler bei der Summenberechnung: {str(e)}'}
    
    def _min(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Berechnet das Minimum für numerische Spalten.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'columns')
            
        Returns:
            Dict: Minimalwerte
        """
        try:
            # Extrahiere Parameter
            columns = params.get('columns')
            
            # Wenn keine Spalten angegeben, verwende alle numerischen
            if not columns:
                columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # Berechne Minimalwerte
            minimums = {}
            for col in columns:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    # Entferne Ausreißer, wenn konfiguriert
                    if self.privacy_config['outlier_removal']:
                        q1 = data[col].quantile(0.25)
                        q3 = data[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        filtered_data = data[data[col] >= lower_bound]
                    else:
                        filtered_data = data
                    
                    # Prüfe die Mindestgruppengröße
                    if len(filtered_data) < self.privacy_config['min_group_size']:
                        minimums[col] = None
                        continue
                    
                    # Berechne Minimum
                    min_value = float(filtered_data[col].min())
                    
                    # Füge Rauschen hinzu (vorsichtig, um den Minimalwert nicht zu stark zu verfälschen)
                    noisy_min = min_value - abs(self._add_privacy_noise(0))  # Addiere negatives Rauschen
                    minimums[col] = noisy_min
            
            return {'minimums': minimums}
            
        except Exception as e:
            logger.error(f"Fehler bei der Minimalwertberechnung: {str(e)}")
            return {'error': f'Fehler bei der Minimalwertberechnung: {str(e)}'}
    
    def _max(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Berechnet das Maximum für numerische Spalten.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'columns')
            
        Returns:
            Dict: Maximalwerte
        """
        try:
            # Extrahiere Parameter
            columns = params.get('columns')
            
            # Wenn keine Spalten angegeben, verwende alle numerischen
            if not columns:
                columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # Berechne Maximalwerte
            maximums = {}
            for col in columns:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    # Entferne Ausreißer, wenn konfiguriert
                    if self.privacy_config['outlier_removal']:
                        q1 = data[col].quantile(0.25)
                        q3 = data[col].quantile(0.75)
                        iqr = q3 - q1
                        upper_bound = q3 + 1.5 * iqr
                        filtered_data = data[data[col] <= upper_bound]
                    else:
                        filtered_data = data
                    
                    # Prüfe die Mindestgruppengröße
                    if len(filtered_data) < self.privacy_config['min_group_size']:
                        maximums[col] = None
                        continue
                    
                    # Berechne Maximum
                    max_value = float(filtered_data[col].max())
                    
                    # Füge Rauschen hinzu (vorsichtig, um den Maximalwert nicht zu stark zu verfälschen)
                    noisy_max = max_value + abs(self._add_privacy_noise(0))  # Addiere positives Rauschen
                    maximums[col] = noisy_max
            
            return {'maximums': maximums}
            
        except Exception as e:
            logger.error(f"Fehler bei der Maximalwertberechnung: {str(e)}")
            return {'error': f'Fehler bei der Maximalwertberechnung: {str(e)}'}
    
    def _correlation(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Berechnet die Korrelationsmatrix für numerische Spalten.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'columns', 'method')
            
        Returns:
            Dict: Korrelationsmatrix
        """
        try:
            # Extrahiere Parameter
            columns = params.get('columns')
            method = params.get('method', 'pearson')
            
            # Wenn keine Spalten angegeben, verwende alle numerischen
            if not columns:
                columns = data.select_dtypes(include=['number']).columns.tolist()
            else:
                # Filtere nicht-numerische Spalten
                columns = [col for col in columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            
            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}
            
            # Berechne Korrelationsmatrix
            correlation_matrix = data[columns].corr(method=method).round(4)
            
            # Konvertiere die Matrix in ein Dictionary
            correlation_dict = correlation_matrix.to_dict()
            
            # Füge Rauschen hinzu
            noisy_correlation = {}
            for col1, values in correlation_dict.items():
                noisy_correlation[col1] = {}
                for col2, value in values.items():
                    if col1 == col2:
                        # Selbstkorrelation immer 1
                        noisy_correlation[col1][col2] = 1.0
                    else:
                        # Füge Rauschen hinzu und begrenze auf [-1, 1]
                        noisy_value = self._add_privacy_noise(value)
                        noisy_correlation[col1][col2] = max(-1.0, min(1.0, noisy_value))
            
            return {'correlation_matrix': noisy_correlation}
            
        except Exception as e:
            logger.error(f"Fehler bei der Korrelationsberechnung: {str(e)}")
            return {'error': f'Fehler bei der Korrelationsberechnung: {str(e)}'}
    
    def _histogram(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Erstellt ein Histogramm für eine Spalte.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'column', 'bins')
            
        Returns:
            Dict: Histogrammdaten
        """
        try:
            # Extrahiere Parameter
            column = params.get('column')
            bins = params.get('bins', 10)
            
            if column not in data.columns:
                return {'error': f'Spalte {column} nicht gefunden'}
            
            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}
            
            if pd.api.types.is_numeric_dtype(data[column]):
                # Numerische Spalte: Histogramm mit Bins
                hist, bin_edges = np.histogram(data[column].dropna(), bins=bins)
                
                # Füge Rauschen zu den Häufigkeiten hinzu
                noisy_hist = [self._add_privacy_noise(count) for count in hist]
                
                return {
                    'histogram': [max(0, round(count)) for count in noisy_hist],  # Keine negativen Häufigkeiten
                    'bin_edges': bin_edges.tolist()
                }
            else:
                # Kategoriale Spalte: Häufigkeitszählung
                value_counts = data[column].value_counts()
                
                # Füge Rauschen zu den Häufigkeiten hinzu
                noisy_counts = {}
                for value, count in value_counts.items():
                    if count >= self.privacy_config['min_group_size']:
                        noisy_counts[str(value)] = max(0, round(self._add_privacy_noise(count)))
                
                return {
                    'categories': list(noisy_counts.keys()),
                    'counts': list(noisy_counts.values())
                }
            
        except Exception as e:
            logger.error(f"Fehler bei der Histogrammerstellung: {str(e)}")
            return {'error': f'Fehler bei der Histogrammerstellung: {str(e)}'}
    
    def _distribution(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Berechnet statistische Verteilungskennzahlen für eine Spalte.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'column')
            
        Returns:
            Dict: Verteilungskennzahlen
        """
        try:
            # Extrahiere Parameter
            column = params.get('column')
            
            if column not in data.columns:
                return {'error': f'Spalte {column} nicht gefunden'}
            
            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}
            
            if pd.api.types.is_numeric_dtype(data[column]):
                # Numerische Spalte: Deskriptive Statistiken
                stats = {
                    'count': len(data[column].dropna()),
                    'mean': float(data[column].mean()),
                    'std': float(data[column].std()),
                    'min': float(data[column].min()),
                    '25%': float(data[column].quantile(0.25)),
                    'median': float(data[column].median()),
                    '75%': float(data[column].quantile(0.75)),
                    'max': float(data[column].max())
                }
                
                # Füge Rauschen hinzu
                noisy_stats = {}
                for key, value in stats.items():
                    if key == 'count':
                        noisy_stats[key] = max(0, round(self._add_privacy_noise(value)))
                    else:
                        noisy_stats[key] = self._add_privacy_noise(value)
                
                return {'distribution': noisy_stats}
            else:
                # Kategoriale Spalte: Häufigkeitsverteilung
                value_counts = data[column].value_counts(normalize=True)
                
                # Konvertiere in Dictionary mit gerundeten Werten
                distribution = {}
                for value, freq in value_counts.items():
                    if data[column].value_counts()[value] >= self.privacy_config['min_group_size']:
                        noisy_freq = self._add_privacy_noise(freq)
                        distribution[str(value)] = max(0, min(1, noisy_freq))  # Begrenze auf [0, 1]
                
                # Normalisiere, damit die Summe wieder 1 ergibt
                total = sum(distribution.values())
                if total > 0:
                    distribution = {key: value / total for key, value in distribution.items()}
                
                return {'distribution': distribution}
            
        except Exception as e:
            logger.error(f"Fehler bei der Verteilungsberechnung: {str(e)}")
            return {'error': f'Fehler bei der Verteilungsberechnung: {str(e)}'}
    
    def _custom_model_inference(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Führt Inferenz mit einem benutzerdefinierten Modell durch.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'model_type', 'model_params')
            
        Returns:
            Dict: Ergebnisse der Modellinferenz
        """
        try:
            # Extrahiere Parameter
            model_type = params.get('model_type')
            model_params = params.get('model_params', {})
            
            if model_type not in ['linear_regression', 'random_forest', 'clustering']:
                return {'error': f'Nicht unterstützter Modelltyp: {model_type}'}
            
            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}
            
            # Je nach Modelltyp unterschiedliche Inferenz durchführen
            if model_type == 'linear_regression':
                # Einfache lineare Regression
                from sklearn.linear_model import LinearRegression
                
                # Extrahiere Feature- und Zielspalten
                feature_cols = model_params.get('feature_cols', [])
                target_col = model_params.get('target_col')
                
                if not feature_cols or not target_col or target_col not in data.columns:
                    return {'error': 'Ungültige Feature- oder Zielspalten'}
                
                # Filtere nur vorhandene numerische Spalten
                feature_cols = [col for col in feature_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
                
                # Trainiere das Modell
                model = LinearRegression()
                model.fit(data[feature_cols], data[target_col])
                
                # Extrahiere Koeffizienten und füge Rauschen hinzu
                coefficients = {}
                for i, col in enumerate(feature_cols):
                    coefficients[col] = self._add_privacy_noise(float(model.coef_[i]))
                
                intercept = self._add_privacy_noise(float(model.intercept_))
                
                return {
                    'model_type': 'linear_regression',
                    'coefficients': coefficients,
                    'intercept': intercept,
                    'r2_score': self._add_privacy_noise(float(model.score(data[feature_cols], data[target_col])))
                }
                
            elif model_type == 'clustering':
                # K-Means-Clustering
                from sklearn.cluster import KMeans
                
                # Extrahiere Feature-Spalten und Clusterzahl
                feature_cols = model_params.get('feature_cols', [])
                n_clusters = model_params.get('n_clusters', 3)
                
                # Filtere nur vorhandene numerische Spalten
                feature_cols = [col for col in feature_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
                
                if not feature_cols:
                    return {'error': 'Keine gültigen Feature-Spalten angegeben'}
                
                # Trainiere das Modell
                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(data[feature_cols])
                
                # Zähle die Größe jedes Clusters
                cluster_sizes = {}
                for i in range(n_clusters):
                    size = np.sum(clusters == i)
                    # Nur Cluster mit Mindestgröße einbeziehen
                    if size >= self.privacy_config['min_group_size']:
                        cluster_sizes[f'cluster_{i}'] = self._add_privacy_noise(int(size))
                
                # Berechne Cluster-Zentren und füge Rauschen hinzu
                cluster_centers = {}
                for i, center in enumerate(model.cluster_centers_):
                    if f'cluster_{i}' in cluster_sizes:  # Nur für Cluster mit ausreichender Größe
                        cluster_centers[f'cluster_{i}'] = {
                            feature_cols[j]: self._add_privacy_noise(float(center[j]))
                            for j in range(len(feature_cols))
                        }
                
                return {
                    'model_type': 'clustering',
                    'n_clusters': len(cluster_sizes),
                    'cluster_sizes': cluster_sizes,
                    'cluster_centers': cluster_centers
                }
                
            elif model_type == 'random_forest':
                # Vereinfachte Random-Forest-Analyse
                from sklearn.ensemble import RandomForestRegressor
                
                # Extrahiere Feature- und Zielspalten
                feature_cols = model_params.get('feature_cols', [])
                target_col = model_params.get('target_col')
                
                if not feature_cols or not target_col or target_col not in data.columns:
                    return {'error': 'Ungültige Feature- oder Zielspalten'}
                
                # Filtere nur vorhandene numerische Spalten
                feature_cols = [col for col in feature_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
                
                # Trainiere das Modell
                model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
                model.fit(data[feature_cols], data[target_col])
                
                # Berechne Feature-Importance und füge Rauschen hinzu
                feature_importance = {}
                for i, col in enumerate(feature_cols):
                    importance = float(model.feature_importances_[i])
                    # Normalisiere auf [0, 1]
                    feature_importance[col] = max(0, min(1, self._add_privacy_noise(importance)))
                
                # Normalisiere Feature-Importance, damit die Summe 1 ergibt
                total_importance = sum(feature_importance.values())
                if total_importance > 0:
                    feature_importance = {k: v / total_importance for k, v in feature_importance.items()}
                
                return {
                    'model_type': 'random_forest',
                    'feature_importance': feature_importance,
                    'r2_score': self._add_privacy_noise(float(model.score(data[feature_cols], data[target_col])))
                }
            
        except Exception as e:
            logger.error(f"Fehler bei der Modellinferenz: {str(e)}")
            return {'error': f'Fehler bei der Modellinferenz: {str(e)}'}
    
    def execute_operation(self, encrypted_data: bytes, operation: str, params: Dict = None) -> Dict:
        """
        Führt eine Operation auf verschlüsselten Daten aus.
        
        Args:
            encrypted_data: Verschlüsselte Daten
            operation: Name der auszuführenden Operation
            params: Parameter für die Operation
            
        Returns:
            Dict: Ergebnis der Operation
        """
        params = params or {}
        
        try:
            if operation not in self.allowed_operations:
                return {'error': f'Nicht unterstützte Operation: {operation}'}
            
            # Entschlüssele die Daten temporär im Speicher
            data = self._decrypt_data(encrypted_data)
            
            # Führe die Operation aus
            operation_func = self.allowed_operations[operation]
            result = operation_func(data, params)
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung der Operation '{operation}': {str(e)}")
            return {'error': f'Fehler bei der Ausführung: {str(e)}'}
    
    def create_data_asset(self, data: pd.DataFrame, asset_metadata: Dict = None) -> Dict:
        """
        Erstellt einen verschlüsselten Daten-Asset mit Metadaten.
        
        Args:
            data: DataFrame mit den Daten
            asset_metadata: Zusätzliche Metadaten für den Asset
            
        Returns:
            Dict: Informationen über den erstellten Asset
        """
        try:
            # Erstelle eine eindeutige ID für den Asset
            asset_id = str(uuid.uuid4())
            
            # Verschlüssele die Daten
            encrypted_data = self._encrypt_data(data)
            
            # Erstelle Basisstatistiken
            stats = {
                'record_count': len(data),
                'column_count': len(data.columns),
                'columns': [{"name": col, "type": str(data[col].dtype)} for col in data.columns]
            }
            
            # Erstelle Metadaten
            metadata = asset_metadata or {}
            
            # Füge Basisinformationen hinzu
            asset_info = {
                'asset_id': asset_id,
                'created_at': datetime.now().isoformat(),
                'statistics': stats,
                'metadata': metadata
            }
            
            # Speichere den Asset
            self.assets[asset_id] = {
                'asset_info': asset_info,
                'encrypted_data': encrypted_data
            }
            
            logger.info(f"Daten-Asset {asset_id} erstellt mit {len(data)} Datensätzen")
            
            # Gib nur die öffentlichen Informationen zurück
            return {
                'asset_id': asset_id,
                'asset_info': asset_info
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung des Daten-Assets: {str(e)}")
            return {'error': f'Fehler bei der Asset-Erstellung: {str(e)}'}
    
    def generate_access_token(self, asset_id: str, allowed_operations: List[str], 
                            expiration_time: int = 3600, token_metadata: Dict = None) -> Dict:
        """
        Generiert ein temporäres Zugriffstoken für einen Daten-Asset.
        
        Args:
            asset_id: ID des Daten-Assets
            allowed_operations: Liste der erlaubten Operationen
            expiration_time: Gültigkeitsdauer des Tokens in Sekunden
            token_metadata: Zusätzliche Metadaten für das Token
            
        Returns:
            Dict: Informationen über das generierte Token
        """
        try:
            if asset_id not in self.assets:
                return {'error': f'Daten-Asset {asset_id} nicht gefunden'}
            
            # Validiere die Operationen
            valid_operations = [op for op in allowed_operations if op in self.allowed_operations]
            
            if not valid_operations:
                return {'error': 'Keine gültigen Operationen angegeben'}
            
            # Generiere eine eindeutige Token-ID
            token_id = str(uuid.uuid4())
            
            # Erstelle Token-Daten
            token_data = {
                'token_id': token_id,
                'asset_id': asset_id,
                'allowed_operations': valid_operations,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(seconds=expiration_time)).isoformat(),
                'max_queries': self.privacy_config['max_query_per_token'],
                'remaining_queries': self.privacy_config['max_query_per_token'],
                'metadata': token_metadata or {}
            }
            
            # Verschlüssele das Token
            token_json = json.dumps(token_data).encode()
            encrypted_token = self.cipher_suite.encrypt(token_json)
            
            # Speichere das Token
            self.tokens[token_id] = {
                'token_data': token_data,
                'encrypted_token': encrypted_token
            }
            
            logger.info(f"Zugriffstoken {token_id} für Asset {asset_id} erstellt")
            
            # Gib das Token zurück
            return {
                'token': encrypted_token.decode(),
                'token_id': token_id,
                'asset_id': asset_id,
                'allowed_operations': valid_operations,
                'expires_at': token_data['expires_at'],
                'max_queries': token_data['max_queries']
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Generierung des Zugriffstokens: {str(e)}")
            return {'error': f'Fehler bei der Token-Generierung: {str(e)}'}
    
    def validate_access_token(self, token: str, operation: str) -> Dict:
        """
        Validiert ein Zugriffstoken für eine bestimmte Operation.
        
        Args:
            token: Das zu validierende Zugriffstoken
            operation: Die auszuführende Operation
            
        Returns:
            Dict: Validierungsergebnis
        """
        try:
            # Entschlüssele das Token
            decrypted_token = self.cipher_suite.decrypt(token.encode())
            token_data = json.loads(decrypted_token.decode())
            
            # Extrahiere Token-ID und prüfe, ob das Token bekannt ist
            token_id = token_data.get('token_id')
            if token_id not in self.tokens:
                return {'valid': False, 'reason': 'Unbekanntes Token'}
            
            # Prüfe, ob das Token abgelaufen ist
            expiry_time = datetime.fromisoformat(token_data['expires_at'])
            if datetime.now() > expiry_time:
                return {'valid': False, 'reason': 'Token abgelaufen'}
            
            # Prüfe, ob die Operation erlaubt ist
            if operation not in token_data['allowed_operations']:
                return {'valid': False, 'reason': 'Operation nicht erlaubt'}
            
            # Prüfe, ob noch Abfragen übrig sind
            if token_data['remaining_queries'] <= 0:
                return {'valid': False, 'reason': 'Keine Abfragen mehr übrig'}
            
            # Token ist gültig
            return {
                'valid': True,
                'token_id': token_id,
                'asset_id': token_data['asset_id'],
                'remaining_queries': token_data['remaining_queries']
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Token-Validierung: {str(e)}")
            return {'valid': False, 'reason': f'Validierungsfehler: {str(e)}'}
    
    def process_query_with_token(self, token: str, operation: str, params: Dict = None) -> Dict:
        """
        Verarbeitet eine Abfrage mit einem Zugriffstoken.
        
        Args:
            token: Zugriffstoken
            operation: Auszuführende Operation
            params: Parameter für die Operation
            
        Returns:
            Dict: Ergebnis der Operation
        """
        try:
            # Validiere das Token
            validation = self.validate_access_token(token, operation)
            
            if not validation['valid']:
                return {'error': f'Ungültiges Token: {validation.get("reason", "Unbekannter Grund")}'}
            
            # Extrahiere Asset-ID und Token-ID
            asset_id = validation['asset_id']
            token_id = validation['token_id']
            
            # Prüfe, ob der Asset existiert
            if asset_id not in self.assets:
                return {'error': f'Daten-Asset {asset_id} nicht gefunden'}
            
            # Hole verschlüsselte Daten
            encrypted_data = self.assets[asset_id]['encrypted_data']
            
            # Führe die Operation aus
            result = self.execute_operation(encrypted_data, operation, params)
            
            # Reduziere die verbleibenden Abfragen
            self.tokens[token_id]['token_data']['remaining_queries'] -= 1
            
            # Füge Nutzungsinformationen zum Ergebnis hinzu
            result['usage_info'] = {
                'token_id': token_id,
                'remaining_queries': self.tokens[token_id]['token_data']['remaining_queries'],
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Abfrage mit Token {token_id} für Asset {asset_id} verarbeitet")
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung der Abfrage: {str(e)}")
            return {'error': f'Fehler bei der Abfrageverarbeitung: {str(e)}'}
    
    def revoke_token(self, token_id: str) -> Dict:
        """
        Widerruft ein Zugriffstoken.
        
        Args:
            token_id: ID des zu widerrufenden Tokens
            
        Returns:
            Dict: Ergebnis des Widerrufs
        """
        try:
            if token_id not in self.tokens:
                return {'error': f'Token {token_id} nicht gefunden'}
            
            # Entferne das Token
            del self.tokens[token_id]
            
            logger.info(f"Token {token_id} widerrufen")
            
            return {'success': True, 'message': f'Token {token_id} erfolgreich widerrufen'}
            
        except Exception as e:
            logger.error(f"Fehler beim Widerrufen des Tokens: {str(e)}")
            return {'error': f'Fehler beim Token-Widerruf: {str(e)}'}
    
    def delete_asset(self, asset_id: str) -> Dict:
        """
        Löscht einen Daten-Asset.
        
        Args:
            asset_id: ID des zu löschenden Assets
            
        Returns:
            Dict: Ergebnis des Löschvorgangs
        """
        try:
            if asset_id not in self.assets:
                return {'error': f'Asset {asset_id} nicht gefunden'}
            
            # Entferne den Asset
            del self.assets[asset_id]
            
            # Entferne alle zugehörigen Tokens
            token_ids_to_remove = []
            for token_id, token_info in self.tokens.items():
                if token_info['token_data']['asset_id'] == asset_id:
                    token_ids_to_remove.append(token_id)
            
            for token_id in token_ids_to_remove:
                del self.tokens[token_id]
            
            logger.info(f"Asset {asset_id} und {len(token_ids_to_remove)} zugehörige Tokens gelöscht")
            
            return {
                'success': True,
                'message': f'Asset {asset_id} erfolgreich gelöscht',
                'removed_tokens': token_ids_to_remove
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Löschen des Assets: {str(e)}")
            return {'error': f'Fehler beim Asset-Löschen: {str(e)}'}
    
    def get_asset_info(self, asset_id: str) -> Dict:
        """
        Gibt Informationen über einen Daten-Asset zurück.
        
        Args:
            asset_id: ID des Assets
            
        Returns:
            Dict: Informationen über den Asset
        """
        try:
            if asset_id not in self.assets:
                return {'error': f'Asset {asset_id} nicht gefunden'}
            
            # Hole Asset-Informationen (ohne verschlüsselte Daten)
            asset_info = self.assets[asset_id]['asset_info']
            
            # Zähle aktive Tokens für diesen Asset
            active_tokens = 0
            for token_info in self.tokens.values():
                if token_info['token_data']['asset_id'] == asset_id:
                    active_tokens += 1
            
            # Füge Tokeninformationen hinzu
            asset_info['active_tokens'] = active_tokens
            
            return asset_info
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Asset-Informationen: {str(e)}")
            return {'error': f'Fehler beim Abrufen der Informationen: {str(e)}'}
    
    def list_assets(self) -> Dict:
        """
        Listet alle verfügbaren Daten-Assets auf.
        
        Returns:
            Dict: Liste der Assets mit Grundinformationen
        """
        try:
            asset_list = []
            
            for asset_id, asset_data in self.assets.items():
                asset_info = asset_data['asset_info']
                
                # Erstelle eine vereinfachte Zusammenfassung
                summary = {
                    'asset_id': asset_id,
                    'created_at': asset_info['created_at'],
                    'record_count': asset_info['statistics']['record_count'],
                    'column_count': asset_info['statistics']['column_count']
                }
                
                # Füge Metadaten hinzu, falls vorhanden
                if 'metadata' in asset_info:
                    summary.update({
                        'name': asset_info['metadata'].get('name', f'Asset {asset_id}'),
                        'description': asset_info['metadata'].get('description', ''),
                        'owner': asset_info['metadata'].get('owner', ''),
                        'data_type': asset_info['metadata'].get('data_type', '')
                    })
                
                asset_list.append(summary)
            
            return {'assets': asset_list, 'count': len(asset_list)}
            
        except Exception as e:
            logger.error(f"Fehler beim Auflisten der Assets: {str(e)}")
            return {'error': f'Fehler beim Auflisten der Assets: {str(e)}'}
"""
OceanData - Compute-to-Data Manager

Dieses Modul implementiert Compute-to-Data-Funktionalität für OceanData.
Dadurch wird die sichere und private Nutzung sensibler Daten ermöglicht, 
ohne dass die Daten direkt mit Dritten geteilt werden müssen.
"""

import pandas as pd
import numpy as np
import json
import logging
import os
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from io import StringIO
from cryptography.fernet import Fernet

# Logging konfigurieren
logger = logging.getLogger("OceanData.Privacy.ComputeToData")

class ComputeToDataManager:
    """
    Manager für Compute-to-Data-Funktionalität.
    
    Diese Klasse ermöglicht:
    - Sichere Speicherung und Verschlüsselung von Daten
    - Ausführung von Berechnungen auf verschlüsselten Daten, ohne sie direkt offenzulegen
    - Generierung und Validierung von Zugriffstoken
    - Verwaltung von erlaubten Operationen und Zugriffsrechten
    """
    
    def __init__(self, encryption_key=None, privacy_config=None):
        """
        Initialisiert den Compute-to-Data-Manager.
        
        Args:
            encryption_key: Optional. Schlüssel für die Verschlüsselung von Daten.
            privacy_config: Optional. Konfiguration für Datenschutzeinstellungen.
        """
        # Verschlüsselung initialisieren
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Standardoperationen definieren
        self.allowed_operations = {
            'aggregate': self._aggregate,
            'count': self._count,
            'mean': self._mean,
            'sum': self._sum,
            'min': self._min,
            'max': self._max,
            'correlation': self._correlation,
            'histogram': self._histogram,
            'distribution': self._distribution,
            'custom_model': self._custom_model_inference
        }
        
        # Datenschutzkonfiguration
        self.privacy_config = privacy_config or {
            'min_group_size': 5,  # Minimale Gruppengröße für Aggregationen
            'noise_level': 0.01,  # Rauschpegel für Differentiellen Datenschutz
            'outlier_removal': True,  # Ausreißer entfernen für Differentiellen Datenschutz
            'max_query_per_token': 10  # Maximale Anzahl von Abfragen pro Token
        }
        
        # Asset-Speicher
        self.assets = {}  # asset_id -> asset_info
        self.tokens = {}  # token_id -> token_info
        
        logger.info("Compute-to-Data-Manager initialisiert")
    
    def _encrypt_data(self, data: pd.DataFrame) -> bytes:
        """
        Verschlüsselt Daten für die sichere Speicherung.
        
        Args:
            data: DataFrame mit zu verschlüsselnden Daten
            
        Returns:
            bytes: Verschlüsselte Daten
        """
        try:
            # Konvertiere DataFrame zu JSON
            serialized = data.to_json().encode()
            
            # Verschlüssele die Daten
            encrypted = self.cipher_suite.encrypt(serialized)
            
            return encrypted
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenverschlüsselung: {str(e)}")
            raise
    
    def _decrypt_data(self, encrypted_data: bytes) -> pd.DataFrame:
        """
        Entschlüsselt Daten für die Verarbeitung.
        
        Args:
            encrypted_data: Verschlüsselte Daten
            
        Returns:
            pd.DataFrame: Entschlüsselte Daten als DataFrame
        """
        try:
            # Entschlüssele die Daten
            decrypted = self.cipher_suite.decrypt(encrypted_data)
            
            # Konvertiere JSON zu DataFrame
            data = pd.read_json(StringIO(decrypted.decode()))
            
            return data
            
        except Exception as e:
            logger.error(f"Fehler bei der Datenentschlüsselung: {str(e)}")
            raise
    
    def _add_privacy_noise(self, result: Any) -> Any:
        """
        Fügt Rauschen zu Ergebnissen hinzu, um Differentiellen Datenschutz zu gewährleisten.
        
        Args:
            result: Das ursprüngliche Ergebnis
            
        Returns:
            Any: Das Ergebnis mit Rauschen
        """
        try:
            noise_level = self.privacy_config['noise_level']
            
            if isinstance(result, (int, float)):
                # Füge normalverteiltes Rauschen hinzu
                noise = np.random.normal(0, noise_level * abs(result) + 1e-6)
                return result + noise
                
            elif isinstance(result, dict):
                # Durchlaufe rekursiv alle Werte im Dictionary
                noisy_result = {}
                for key, value in result.items():
                    noisy_result[key] = self._add_privacy_noise(value)
                return noisy_result
                
            elif isinstance(result, list):
                # Durchlaufe rekursiv alle Elemente in der Liste
                return [self._add_privacy_noise(item) for item in result]
                
            elif isinstance(result, np.ndarray):
                # Füge Rauschen zu jedem Element im Array hinzu
                noise = np.random.normal(0, noise_level * np.abs(result) + 1e-6, result.shape)
                return result + noise
                
            else:
                # Nicht-numerische Typen unverändert lassen
                return result
                
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Privatsphäre-Rauschen: {str(e)}")
            return result
    
    def _check_min_group_size(self, data: pd.DataFrame, params: Dict) -> bool:
        """
        Prüft, ob gruppierte Daten die Mindestgruppengröße erfüllen.
        
        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation
            
        Returns:
            bool: True wenn die Mindestgruppengröße erfüllt ist, sonst False
        """
        min_size = self.privacy_config['min_group_size']
        
        try:
            # Wenn keine Gruppierung angegeben ist, prüfe die Gesamtgröße
            if 'group_by' not in params:
                return len(data) >= min_size
                
            # Bei Gruppierung: Prüfe jede Gruppe
            group_cols = params['group_by']
            if not isinstance(group_cols, list):
                group_cols = [group_cols]
                
            grouped = data.groupby(group_cols).size()
            return grouped.min() >= min_size
            
        except Exception as e:
            logger.error(f"Fehler bei der Prüfung der Mindestgruppengröße: {str(e)}")
            return False
    
    # Implementierung von erlaubten Operationen
    
    def _aggregate(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Aggregiert Daten nach


    
