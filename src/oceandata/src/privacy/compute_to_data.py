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
import time
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from io import StringIO
import importlib.util
import sys
import traceback
import threading
import queue
import tempfile
import shutil
import subprocess
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
import base64

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
    - Sichere Ausführungsumgebung für benutzerdefinierte Berechnungen
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
        
        # Asymmetrische Schlüssel für sichere Kommunikation
        self._generate_key_pair()

        # Standardoperationen definieren
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
            'distribution': self._distribution,
            'custom_model': self._custom_model_inference
        }

        # Datenschutzkonfiguration
        self.privacy_config = privacy_config or {
            'min_group_size': 5,  # Minimale Gruppengröße für Aggregationen
            'noise_level': 0.01,  # Rauschpegel für Differentiellen Datenschutz
            'outlier_removal': True,  # Ausreißer entfernen für Differentiellen Datenschutz
            'max_query_per_token': 10,  # Maximale Anzahl von Abfragen pro Token
            'token_expiry_hours': 24,  # Gültigkeitsdauer der Token in Stunden
            'max_execution_time': 60,  # Maximale Ausführungszeit für benutzerdefinierte Berechnungen in Sekunden
            'max_memory_usage': 1024,  # Maximale Speichernutzung in MB
            'allowed_packages': [      # Erlaubte Python-Pakete für benutzerdefinierte Berechnungen
                'numpy', 'pandas', 'scipy', 'sklearn', 'statsmodels'
            ],
            'audit_logging': True,     # Audit-Logging aktivieren
            'secure_execution': True   # Sichere Ausführungsumgebung aktivieren
        }

        # Asset-Speicher
        self.assets = {}  # asset_id -> asset_info
        self.tokens = {}  # token_id -> token_info
        
        # Audit-Log
        self.audit_log = []
        
        # Sperren für Thread-Sicherheit
        self.asset_lock = threading.RLock()
        self.token_lock = threading.RLock()
        self.audit_lock = threading.RLock()
        
        # Zähler für Anfragen
        self.request_counter = 0
        
        logger.info("Compute-to-Data-Manager initialisiert")
        
    def _generate_key_pair(self):
        """Generiert ein asymmetrisches Schlüsselpaar für sichere Kommunikation"""
        try:
            # Generiere RSA-Schlüsselpaar
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            
            # Speichere Schlüssel
            self.private_key = private_key
            self.public_key = public_key
            
            # Exportiere öffentlichen Schlüssel für externe Nutzung
            self.public_key_pem = public_key.public_bytes(
                encoding=Encoding.PEM,
                format=PublicFormat.SubjectPublicKeyInfo
            )
            
            logger.info("Asymmetrisches Schlüsselpaar generiert")
            
        except Exception as e:
            logger.error(f"Fehler bei der Generierung des Schlüsselpaars: {str(e)}")
            # Fallback: Verwende nur symmetrische Verschlüsselung
            self.private_key = None
            self.public_key = None
            self.public_key_pem = None
            logger.warning("Verwende nur symmetrische Verschlüsselung als Fallback")

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

    def _add_privacy_noise(self, result: Any, scale: float = None) -> Any:
        """
        Fügt Rauschen zu Ergebnissen hinzu, um Differentiellen Datenschutz zu gewährleisten.

        Args:
            result: Das ursprüngliche Ergebnis
            scale: Skalierungsfaktor für das Rauschen (optional)

        Returns:
            Any: Das Ergebnis mit Rauschen
        """
        try:
            noise_level = scale or self.privacy_config['noise_level']

            if isinstance(result, (int, float)):
                # Füge normalverteiltes Rauschen hinzu
                noise = np.random.normal(0, noise_level * abs(result) + 1e-6)
                return result + noise

            elif isinstance(result, dict):
                # Durchlaufe rekursiv alle Werte im Dictionary
                noisy_result = {}
                for key, value in result.items():
                    noisy_result[key] = self._add_privacy_noise(value, scale)
                return noisy_result

            elif isinstance(result, list):
                # Durchlaufe rekursiv alle Elemente in der Liste
                return [self._add_privacy_noise(item, scale) for item in result]

            elif isinstance(result, np.ndarray):
                # Füge Rauschen zu jedem Element im Array hinzu
                noise = np.random.normal(0, noise_level * np.abs(result).mean() + 1e-6, result.shape)
                return result + noise

            elif isinstance(result, pd.Series):
                # Füge Rauschen zu jeder Zelle in der Series hinzu
                if pd.api.types.is_numeric_dtype(result):
                    noise = np.random.normal(0, noise_level * result.abs().mean() + 1e-6, result.shape)
                    return result + noise
                return result

            elif isinstance(result, pd.DataFrame):
                # Füge Rauschen zu jeder numerischen Spalte im DataFrame hinzu
                noisy_df = result.copy()
                for col in noisy_df.select_dtypes(include=['number']).columns:
                    noise = np.random.normal(0, noise_level * noisy_df[col].abs().mean() + 1e-6, noisy_df[col].shape)
                    noisy_df[col] = noisy_df[col] + noise
                return noisy_df

            else:
                # Nicht-numerische Typen unverändert lassen
                return result

        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Privatsphäre-Rauschen: {str(e)}")
            return result

    def _remove_outliers(self, data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Entfernt Ausreißer aus den Daten für besseren Datenschutz.

        Args:
            data: DataFrame mit den Daten
            columns: Spalten, für die Ausreißer entfernt werden sollen (optional)

        Returns:
            pd.DataFrame: Daten ohne Ausreißer
        """
        if not self.privacy_config['outlier_removal']:
            return data
            
        try:
            # Wenn keine Spalten angegeben, verwende alle numerischen
            if columns is None:
                columns = data.select_dtypes(include=['number']).columns.tolist()
                
            # Kopie erstellen, um die Originaldaten nicht zu verändern
            filtered_data = data.copy()
            
            # Für jede Spalte Ausreißer entfernen
            for col in columns:
                if col in filtered_data.columns and pd.api.types.is_numeric_dtype(filtered_data[col]):
                    # Berechne Q1, Q3 und IQR
                    q1 = filtered_data[col].quantile(0.25)
                    q3 = filtered_data[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    # Definiere Grenzen für Ausreißer
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Filtere Ausreißer
                    mask = (filtered_data[col] >= lower_bound) & (filtered_data[col] <= upper_bound)
                    filtered_data = filtered_data[mask]
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Fehler beim Entfernen von Ausreißern: {str(e)}")
            return data

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

    def _log_audit_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Protokolliert ein Audit-Ereignis.

        Args:
            event_type: Typ des Ereignisses
            details: Details zum Ereignis
        """
        if not self.privacy_config.get('audit_logging', True):
            return
            
        try:
            with self.audit_lock:
                event = {
                    'timestamp': datetime.now().isoformat(),
                    'event_type': event_type,
                    'request_id': self.request_counter,
                    'details': details
                }
                
                self.audit_log.append(event)
                
                # Log auch in die Logdatei
                logger.info(f"Audit: {event_type} - {json.dumps(details)}")
                
        except Exception as e:
            logger.error(f"Fehler beim Protokollieren des Audit-Ereignisses: {str(e)}")

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
            
            # Entferne Ausreißer, wenn konfiguriert
            if self.privacy_config['outlier_removal']:
                data = self._remove_outliers(data)

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

                # Füge Rauschen hinzu
                noisy_result = self._add_privacy_noise({'aggregated_data': result_dict})
                return noisy_result
            else:
                # Globale Aggregation
                result = data.agg(aggregates)

                # Konvertiere das Ergebnis in eine serialisierbare Form
                # Füge Rauschen hinzu
                noisy_result = self._add_privacy_noise({'aggregated_data': result.to_dict()})
                return noisy_result

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
                
            # Entferne Ausreißer, wenn konfiguriert
            if self.privacy_config['outlier_removal']:
                data = self._remove_outliers(data, columns)

            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}

            # Berechne Mittelwerte
            means = {}
            for col in columns:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    # Berechne Mittelwert
                    mean_value = float(data[col].mean())
                    means[col] = mean_value

            # Füge Rauschen hinzu
            noisy_means = self._add_privacy_noise({'means': means})
            return noisy_means

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
                
            # Entferne Ausreißer, wenn konfiguriert
            if self.privacy_config['outlier_removal']:
                data = self._remove_outliers(data, columns)

            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}

            # Berechne Summen
            sums = {}
            for col in columns:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    # Berechne Summe
                    sum_value = float(data[col].sum())
                    sums[col] = sum_value

            # Füge Rauschen hinzu
            noisy_sums = self._add_privacy_noise({'sums': sums})
            return noisy_sums

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
                
            # Entferne Ausreißer, wenn konfiguriert
            if self.privacy_config['outlier_removal']:
                data = self._remove_outliers(data, columns)

            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}

            # Berechne Minimalwerte
            minimums = {}
            for col in columns:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    # Berechne Minimum
                    min_value = float(data[col].min())
                    minimums[col] = min_value

            # Füge Rauschen hinzu
            noisy_minimums = self._add_privacy_noise({'minimums': minimums})
            return noisy_minimums

        except Exception as e:
            logger.error(f"Fehler bei der Minimumberechnung: {str(e)}")
            return {'error': f'Fehler bei der Minimumberechnung: {str(e)}'}

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
                
            # Entferne Ausreißer, wenn konfiguriert
            if self.privacy_config['outlier_removal']:
                data = self._remove_outliers(data, columns)

            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}

            # Berechne Maximalwerte
            maximums = {}
            for col in columns:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    # Berechne Maximum
                    max_value = float(data[col].max())
                    maximums[col] = max_value

            # Füge Rauschen hinzu
            noisy_maximums = self._add_privacy_noise({'maximums': maximums})
            return noisy_maximums

        except Exception as e:
            logger.error(f"Fehler bei der Maximumberechnung: {str(e)}")
            return {'error': f'Fehler bei der Maximumberechnung: {str(e)}'}

    def _std(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Berechnet die Standardabweichung für numerische Spalten.

        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'columns')

        Returns:
            Dict: Standardabweichungen
        """
        try:
            # Extrahiere Parameter
            columns = params.get('columns')

            # Wenn keine Spalten angegeben, verwende alle numerischen
            if not columns:
                columns = data.select_dtypes(include=['number']).columns.tolist()
                
            # Entferne Ausreißer, wenn konfiguriert
            if self.privacy_config['outlier_removal']:
                data = self._remove_outliers(data, columns)

            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}

            # Berechne Standardabweichungen
            stds = {}
            for col in columns:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    # Berechne Standardabweichung
                    std_value = float(data[col].std())
                    stds[col] = std_value

            # Füge Rauschen hinzu
            noisy_stds = self._add_privacy_noise({'standard_deviations': stds})
            return noisy_stds

        except Exception as e:
            logger.error(f"Fehler bei der Standardabweichungsberechnung: {str(e)}")
            return {'error': f'Fehler bei der Standardabweichungsberechnung: {str(e)}'}

    def _correlation(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Berechnet die Korrelation zwischen numerischen Spalten.

        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'columns', 'method')

        Returns:
            Dict: Korrelationsmatrix
        """
        try:
            # Extrahiere Parameter
            columns = params.get('columns')
            method = params.get('method', 'pearson')  # pearson, kendall, spearman

            # Wenn keine Spalten angegeben, verwende alle numerischen
            if not columns:
                columns = data.select_dtypes(include=['number']).columns.tolist()
                
            # Entferne Ausreißer, wenn konfiguriert
            if self.privacy_config['outlier_removal']:
                data = self._remove_outliers(data, columns)

            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}

            # Berechne Korrelation
            correlation = data[columns].corr(method=method)

            # Konvertiere zu Dictionary
            corr_dict = correlation.to_dict()

            # Füge Rauschen hinzu
            noisy_correlation = self._add_privacy_noise({'correlation': corr_dict, 'method': method})
            return noisy_correlation

        except Exception as e:
            logger.error(f"Fehler bei der Korrelationsberechnung: {str(e)}")
            return {'error': f'Fehler bei der Korrelationsberechnung: {str(e)}'}

    def _histogram(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Erstellt ein Histogramm für numerische Spalten.

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
            
            if not column:
                return {'error': 'Keine Spalte angegeben'}
                
            if column not in data.columns:
                return {'error': f'Spalte {column} nicht gefunden'}
                
            if not pd.api.types.is_numeric_dtype(data[column]):
                return {'error': f'Spalte {column} ist nicht numerisch'}
                
            # Entferne Ausreißer, wenn konfiguriert
            if self.privacy_config['outlier_removal']:
                data = self._remove_outliers(data, [column])

            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}

            # Berechne Histogramm
            hist, bin_edges = np.histogram(data[column].dropna(), bins=bins)
            
            # Erstelle Ergebnis
            histogram_data = {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'column': column,
                'bins': bins
            }

            # Füge Rauschen hinzu
            noisy_histogram = self._add_privacy_noise({'histogram': histogram_data})
            return noisy_histogram

        except Exception as e:
            logger.error(f"Fehler bei der Histogrammerstellung: {str(e)}")
            return {'error': f'Fehler bei der Histogrammerstellung: {str(e)}'}

    def _distribution(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Berechnet Verteilungsstatistiken für numerische Spalten.

        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'columns', 'percentiles')

        Returns:
            Dict: Verteilungsstatistiken
        """
        try:
            # Extrahiere Parameter
            columns = params.get('columns')
            percentiles = params.get('percentiles', [0.25, 0.5, 0.75])
            
            # Wenn keine Spalten angegeben, verwende alle numerischen
            if not columns:
                columns = data.select_dtypes(include=['number']).columns.tolist()
                
            # Entferne Ausreißer, wenn konfiguriert
            if self.privacy_config['outlier_removal']:
                data = self._remove_outliers(data, columns)

            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}

            # Berechne Verteilungsstatistiken
            distribution = {}
            for col in columns:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    col_data = data[col].dropna()
                    
                    # Berechne Statistiken
                    stats = {
                        'count': len(col_data),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'percentiles': {
                            str(int(p * 100)): float(col_data.quantile(p))
                            for p in percentiles
                        }
                    }
                    
                    distribution[col] = stats

            # Füge Rauschen hinzu
            noisy_distribution = self._add_privacy_noise({'distribution': distribution})
            return noisy_distribution

        except Exception as e:
            logger.error(f"Fehler bei der Verteilungsberechnung: {str(e)}")
            return {'error': f'Fehler bei der Verteilungsberechnung: {str(e)}'}

    def _custom_model_inference(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Führt benutzerdefinierte Modellberechnungen auf den Daten aus.

        Args:
            data: DataFrame mit den Daten
            params: Parameter für die Operation (z.B. 'model_code', 'input_columns', 'output_format')

        Returns:
            Dict: Ergebnisse der Modellberechnung
        """
        try:
            # Extrahiere Parameter
            model_code = params.get('model_code')
            input_columns = params.get('input_columns')
            output_format = params.get('output_format', 'json')
            
            if not model_code:
                return {'error': 'Kein Modellcode angegeben'}
                
            # Prüfe die Mindestgruppengröße
            if len(data) < self.privacy_config['min_group_size']:
                return {'error': 'Mindestgruppengröße nicht erfüllt'}
                
            # Filtere Daten nach Eingabespalten, falls angegeben
            if input_columns:
                # Prüfe, ob alle angegebenen Spalten existieren
                missing_columns = [col for col in input_columns if col not in data.columns]
                if missing_columns:
                    return {'error': f'Spalten nicht gefunden: {", ".join(missing_columns)}'}
                    
                input_data = data[input_columns].copy()
            else:
                input_data = data.copy()
                
            # Führe den benutzerdefinierten Code in einer sicheren Umgebung aus
            if self.privacy_config.get('secure_execution', True):
                result = self._run_in_secure_environment(model_code, input_data)
            else:
                # Unsichere Ausführung (nicht empfohlen)
                result = self._run_unsafe(model_code, input_data)
                
            # Prüfe das Ergebnis
            if isinstance(result, dict) and 'error' in result:
                return result
                
            # Konvertiere das Ergebnis in das gewünschte Format
            if output_format == 'json':
                # Versuche, das Ergebnis in JSON zu konvertieren
                try:
                    if isinstance(result, pd.DataFrame):
                        result_json = result.to_dict(orient='records')
                    elif isinstance(result, pd.Series):
                        result_json = result.to_dict()
                    elif isinstance(result, np.ndarray):
                        result_json = result.tolist()
                    else:
                        result_json = result
                        
                    # Füge Rauschen hinzu
                    noisy_result = self._add_privacy_noise({'result': result_json})
                    return noisy_result
                    
                except Exception as e:
                    return {'error': f'Fehler bei der Konvertierung des Ergebnisses: {str(e)}'}
            else:
                return {'error': f'Nicht unterstütztes Ausgabeformat: {output_format}'}

        except Exception as e:
            logger.error(f"Fehler bei der benutzerdefinierten Modellberechnung: {str(e)}")
            return {'error': f'Fehler bei der benutzerdefinierten Modellberechnung: {str(e)}'}

    def _run_in_secure_environment(self, code: str, data: pd.DataFrame) -> Any:
        """
        Führt benutzerdefinierten Code in einer sicheren Umgebung aus.

        Args:
            code: Auszuführender Python-Code
            data: Eingabedaten als DataFrame

        Returns:
            Any: Ergebnis der Codeausführung
        """
        try:
            # Erstelle ein temporäres Verzeichnis für die Ausführung
            with tempfile.TemporaryDirectory() as temp_dir:
                # Erstelle eine Datei mit dem Code
                code_file = os.path.join(temp_dir, 'model_code.py')
                with open(code_file, 'w') as f:
                    f.write(code)
                
                # Speichere die Daten als CSV
                data_file = os.path.join(temp_dir, 'input_data.csv')
                data.to_csv(data_file, index=False)
                
                # Erstelle ein Wrapper-Skript, das den Code ausführt und das Ergebnis zurückgibt
                wrapper_file = os.path.join(temp_dir, 'wrapper.py')
                with open(wrapper_file, 'w') as f:
                    f.write("""
import pandas as pd
import numpy as np
import json
import sys
import os
import importlib.util
import traceback

# Lade die Daten
data = pd.read_csv('input_data.csv')

try:
    # Lade den Code als Modul
    spec = importlib.util.spec_from_file_location("model_code", "model_code.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    # Führe die run-Funktion aus, falls vorhanden
    if hasattr(model_module, 'run'):
        result = model_module.run(data)
        
        # Konvertiere das Ergebnis in JSON
        if isinstance(result, pd.DataFrame):
            result_json = result.to_dict(orient='records')
        elif isinstance(result, pd.Series):
            result_json = result.to_dict()
        elif isinstance(result, np.ndarray):
            result_json = result.tolist()
        else:
            result_json = result
            
        # Gib das Ergebnis aus
        print(json.dumps({'result': result_json}))
    else:
        print(json.dumps({'error': 'Keine run-Funktion im Code gefunden'}))
except Exception as e:
    print(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}))
""")
                
                # Führe den Code in einem separaten Prozess aus
                cmd = [
                    sys.executable,
                    wrapper_file
                ]
                
                # Setze Zeitlimit
                timeout = self.privacy_config.get('max_execution_time', 60)
                
                # Führe den Prozess aus
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    
                    # Prüfe, ob der Prozess erfolgreich war
                    if result.returncode != 0:
                        return {'error': f'Fehler bei der Ausführung: {result.stderr}'}
                    
                    # Parse das Ergebnis
                    try:
                        return json.loads(result.stdout)
                    except json.JSONDecodeError:
                        return {'error': f'Ungültiges JSON-Ergebnis: {result.stdout}'}
                        
                except subprocess.TimeoutExpired:
                    return {'error': f'Zeitlimit von {timeout} Sekunden überschritten'}
                
        except Exception as e:
            logger.error(f"Fehler bei der sicheren Ausführung: {str(e)}")
            return {'error': f'Fehler bei der sicheren Ausführung: {str(e)}'}

    def _run_unsafe(self, code: str, data: pd.DataFrame) -> Any:
        """
        Führt benutzerdefinierten Code direkt aus (unsicher, nur für Entwicklung).

        Args:
            code: Auszuführender Python-Code
            data: Eingabedaten als DataFrame

        Returns:
            Any: Ergebnis der Codeausführung
        """
        try:
            # Erstelle ein temporäres Modul
            module_name = f"custom_model_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            
            # Füge Abhängigkeiten hinzu
            module.__dict__['pd'] = pd
            module.__dict__['np'] = np
            module.__dict__['data'] = data
            
            # Führe den Code aus
            exec(code, module.__dict__)
            
            # Rufe die run-Funktion auf, falls vorhanden
            if hasattr(module, 'run'):
                return module.run(data)
            else:
                return {'error': 'Keine run-Funktion im Code gefunden'}
                
        except Exception as e:
            logger.error(f"Fehler bei der unsicheren Ausführung: {str(e)}")
            return {'error': f'Fehler bei der unsicheren Ausführung: {str(e)}'}

    # Asset-Verwaltung

    def register_asset(self, data: pd.DataFrame, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Registriert einen neuen Daten-Asset.

        Args:
            data: Die zu registrierenden Daten
            metadata: Metadaten zum Asset (optional)

        Returns:
            Dict: Informationen zum registrierten Asset
        """
        try:
            # Generiere eine eindeutige Asset-ID
            asset_id = f"asset_{uuid.uuid4().hex}"
            
            # Verschlüssele die Daten
            encrypted_data = self._encrypt_data(data)
            
            # Erstelle Asset-Informationen
            asset_info = {
                "asset_id": asset_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "owner": metadata.get("owner", "unknown"),
                "statistics": {
                    "record_count": len(data),
                    "column_count": len(data.columns),
                    "columns": list(data.columns)
                },
                "metadata": metadata or {}
            }
            
            # Speichere den Asset
            with self.asset_lock:
                self.assets[asset_id] = {
                    "asset_info": asset_info,
                    "encrypted_data": encrypted_data
                }
            
            # Protokolliere das Ereignis
            self._log_audit_event("asset_registered", {
                "asset_id": asset_id,
                "owner": asset_info["owner"],
                "record_count": asset_info["statistics"]["record_count"]
            })
            
            logger.info(f"Asset {asset_id} registriert mit {len(data)} Datensätzen")
            
            return {
                "success": True,
                "asset_id": asset_id,
                "asset_info": asset_info
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Asset-Registrierung: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler bei der Asset-Registrierung: {str(e)}"
            }

    def get_asset(self, asset_id: str) -> Dict[str, Any]:
        """
        Gibt Informationen zu einem Asset zurück (ohne die Daten).

        Args:
            asset_id: ID des Assets

        Returns:
            Dict: Informationen zum Asset
        """
        try:
            with self.asset_lock:
                if asset_id not in self.assets:
                    return {
                        "success": False,
                        "error": f"Asset {asset_id} nicht gefunden"
                    }
                
                asset_info = self.assets[asset_id]["asset_info"]
                
                # Protokolliere das Ereignis
                self._log_audit_event("asset_accessed", {
                    "asset_id": asset_id,
                    "access_type": "info"
                })
                
                return {
                    "success": True,
                    "asset_info": asset_info
                }
                
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Assets: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Abrufen des Assets: {str(e)}"
            }

    def update_asset_metadata(self, asset_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aktualisiert die Metadaten eines Assets.

        Args:
            asset_id: ID des Assets
            metadata: Neue Metadaten

        Returns:
            Dict: Aktualisierte Asset-Informationen
        """
        try:
            with self.asset_lock:
                if asset_id not in self.assets:
                    return {
                        "success": False,
                        "error": f"Asset {asset_id} nicht gefunden"
                    }
                
                # Aktualisiere die Metadaten
                self.assets[asset_id]["asset_info"]["metadata"].update(metadata)
                self.assets[asset_id]["asset_info"]["updated_at"] = datetime.now().isoformat()
                
                # Protokolliere das Ereignis
                self._log_audit_event("asset_updated", {
                    "asset_id": asset_id,
                    "update_type": "metadata"
                })
                
                return {
                    "success": True,
                    "asset_info": self.assets[asset_id]["asset_info"]
                }
                
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der Asset-Metadaten: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Aktualisieren der Asset-Metadaten: {str(e)}"
            }

    def delete_asset(self, asset_id: str) -> Dict[str, Any]:
        """
        Löscht einen Daten-Asset.

        Args:
            asset_id: ID des zu löschenden Assets

        Returns:
            Dict: Ergebnis des Löschvorgangs
        """
        try:
            with self.asset_lock:
                if asset_id not in self.assets:
                    return {
                        "success": False,
                        "error": f"Asset {asset_id} nicht gefunden"
                    }
                
                # Entferne den Asset
                del self.assets[asset_id]
            
            # Entferne alle zugehörigen Tokens
            with self.token_lock:
                token_ids_to_remove = []
                for token_id, token_info in self.tokens.items():
                    if token_info['token_data']['asset_id'] == asset_id:
                        token_ids_to_remove.append(token_id)
                
                for token_id in token_ids_to_remove:
                    del self.tokens[token_id]
            
            # Protokolliere das Ereignis
            self._log_audit_event("asset_deleted", {
                "asset_id": asset_id,
                "removed_tokens": token_ids_to_remove
            })
            
            logger.info(f"Asset {asset_id} und {len(token_ids_to_remove)} zugehörige Tokens gelöscht")
            
            return {
                "success": True,
                "message": f"Asset {asset_id} erfolgreich gelöscht",
                "removed_tokens": token_ids_to_remove
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Löschen des Assets: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Asset-Löschen: {str(e)}"
            }

    def list_assets(self) -> Dict[str, Any]:
        """
        Listet alle verfügbaren Daten-Assets auf.

        Returns:
            Dict: Liste der Assets mit Grundinformationen
        """
        try:
            with self.asset_lock:
                asset_list = []
                
                for asset_id, asset_data in self.assets.items():
                    asset_info = asset_data["asset_info"]
                    
                    # Erstelle eine vereinfachte Zusammenfassung
                    summary = {
                        "asset_id": asset_id,
                        "created_at": asset_info["created_at"],
                        "updated_at": asset_info["updated_at"],
                        "owner": asset_info["owner"],
                        "record_count": asset_info["statistics"]["record_count"],
                        "column_count": asset_info["statistics"]["column_count"]
                    }
                    
                    # Füge Metadaten hinzu, falls vorhanden
                    if "metadata" in asset_info:
                        summary.update({
                            "name": asset_info["metadata"].get("name", f"Asset {asset_id}"),
                            "description": asset_info["metadata"].get("description", ""),
                            "data_type": asset_info["metadata"].get("data_type", "")
                        })
                    
                    asset_list.append(summary)
            
            # Protokolliere das Ereignis
            self._log_audit_event("assets_listed", {
                "count": len(asset_list)
            })
            
            return {
                "success": True,
                "assets": asset_list,
                "count": len(asset_list)
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Auflisten der Assets: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Auflisten der Assets: {str(e)}"
            }

    # Token-Verwaltung

    def create_access_token(self, asset_id: str, allowed_operations: List[str] = None,
                          max_queries: int = None, expiry_hours: int = None) -> Dict[str, Any]:
        """
        Erstellt ein Zugriffstoken für einen Asset.

        Args:
            asset_id: ID des Assets
            allowed_operations: Liste der erlaubten Operationen (optional)
            max_queries: Maximale Anzahl von Abfragen (optional)
            expiry_hours: Gültigkeitsdauer in Stunden (optional)

        Returns:
            Dict: Informationen zum erstellten Token
        """
        try:
            with self.asset_lock:
                if asset_id not in self.assets:
                    return {
                        "success": False,
                        "error": f"Asset {asset_id} nicht gefunden"
                    }
            
            # Standardwerte verwenden, falls nicht angegeben
            if allowed_operations is None:
                allowed_operations = list(self.allowed_operations.keys())
            else:
                # Prüfe, ob alle angegebenen Operationen unterstützt werden
                unsupported_ops = [op for op in allowed_operations if op not in self.allowed_operations]
                if unsupported_ops:
                    return {
                        "success": False,
                        "error": f"Nicht unterstützte Operationen: {', '.join(unsupported_ops)}"
                    }
            
            if max_queries is None:
                max_queries = self.privacy_config.get('max_query_per_token', 10)
                
            if expiry_hours is None:
                expiry_hours = self.privacy_config.get('token_expiry_hours', 24)
            
            # Generiere Token-ID
            token_id = f"token_{uuid.uuid4().hex}"
            
            # Berechne Ablaufzeit
            expires_at = (datetime.now() + timedelta(hours=expiry_hours)).isoformat()
            
            # Erstelle Token-Daten
            token_data = {
                "token_id": token_id,
                "asset_id": asset_id,
                "created_at": datetime.now().isoformat(),
                "expires_at": expires_at,
                "allowed_operations": allowed_operations,
                "max_queries": max_queries,
                "remaining_queries": max_queries
            }
            
            # Erstelle JWT-Token
            if self.private_key:
                # Verwende JWT mit asymmetrischer Verschlüsselung
                token_jwt = jwt.encode(
                    token_data,
                    self.private_key,
                    algorithm="RS256"
                )
            else:
                # Fallback: Verwende symmetrische Verschlüsselung
                token_jwt = self.cipher_suite.encrypt(json.dumps(token_data).encode()).decode()
            
            # Speichere Token-Informationen
            with self.token_lock:
                self.tokens[token_id] = {
                    "token_data": token_data,
                    "token_jwt": token_jwt
                }
            
            # Protokolliere das Ereignis
            self._log_audit_event("token_created", {
                "token_id": token_id,
                "asset_id": asset_id,
                "allowed_operations": allowed_operations,
                "max_queries": max_queries,
                "expires_at": expires_at
            })
            
            logger.info(f"Zugriffstoken {token_id} für Asset {asset_id} erstellt")
            
            return {
                "success": True,
                "token_id": token_id,
                "token": token_jwt,
                "token_data": token_data
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Token-Erstellung: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler bei der Token-Erstellung: {str(e)}"
            }

    def validate_access_token(self, token: str, operation: str) -> Dict[str, Any]:
        """
        Validiert ein Zugriffstoken für eine Operation.

        Args:
            token: Zugriffstoken
            operation: Die auszuführende Operation

        Returns:
            Dict: Validierungsergebnis
        """
        try:
            # Versuche, das Token zu decodieren
            token_data = None
            
            if self.public_key:
                # Versuche JWT-Decodierung mit asymmetrischem Schlüssel
                try:
                    token_data = jwt.decode(
                        token,
                        self.public_key,
                        algorithms=["RS256"]
                    )
                except jwt.InvalidTokenError:
                    pass
            
            if token_data is None:
                # Versuche symmetrische Entschlüsselung als Fallback
                try:
                    decrypted_token = self.cipher_suite.decrypt(token.encode())
                    token_data = json.loads(decrypted_token.decode())
                except Exception:
                    return {
                        "valid": False,
                        "reason": "Ungültiges Token-Format"
                    }
            
            # Extrahiere Token-ID und prüfe, ob das Token bekannt ist
            token_id = token_data.get('token_id')
            
            with self.token_lock:
                if token_id not in self.tokens:
                    return {
                        "valid": False,
                        "reason": "Unbekanntes Token"
                    }
                
                # Prüfe, ob das Token abgelaufen ist
                expiry_time = datetime.fromisoformat(token_data['expires_at'])
                if datetime.now() > expiry_time:
                    return {
                        "valid": False,
                        "reason": "Token abgelaufen"
                    }
                
                # Prüfe, ob die Operation erlaubt ist
                if operation not in token_data['allowed_operations']:
                    return {
                        "valid": False,
                        "reason": "Operation nicht erlaubt"
                    }
                
                # Prüfe, ob noch Abfragen übrig sind
                if token_data['remaining_queries'] <= 0:
                    return {
                        "valid": False,
                        "reason": "Keine Abfragen mehr übrig"
                    }
                
                # Token ist gültig
                return {
                    "valid": True,
                    "token_id": token_id,
                    "asset_id": token_data['asset_id'],
                    "remaining_queries": token_data['remaining_queries']
                }
                
        except Exception as e:
            logger.error(f"Fehler bei der Token-Validierung: {str(e)}")
            return {
                "valid": False,
                "reason": f"Validierungsfehler: {str(e)}"
            }

    def revoke_token(self, token_id: str) -> Dict[str, Any]:
        """
        Widerruft ein Zugriffstoken.

        Args:
            token_id: ID des zu widerrufenden Tokens

        Returns:
            Dict: Ergebnis des Widerrufs
        """
        try:
            with self.token_lock:
                if token_id not in self.tokens:
                    return {
                        "success": False,
                        "error": f"Token {token_id} nicht gefunden"
                    }
                
                # Entferne das Token
                token_info = self.tokens[token_id]
                del self.tokens[token_id]
            
            # Protokolliere das Ereignis
            self._log_audit_event("token_revoked", {
                "token_id": token_id,
                "asset_id": token_info["token_data"]["asset_id"]
            })
            
            logger.info(f"Token {token_id} widerrufen")
            
            return {
                "success": True,
                "message": f"Token {token_id} erfolgreich widerrufen"
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Widerrufen des Tokens: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Token-Widerruf: {str(e)}"
            }

    def list_tokens(self, asset_id: str = None) -> Dict[str, Any]:
        """
        Listet alle Zugriffstoken auf, optional gefiltert nach Asset-ID.

        Args:
            asset_id: ID des Assets (optional)

        Returns:
            Dict: Liste der Token mit Grundinformationen
        """
        try:
            with self.token_lock:
                token_list = []
                
                for token_id, token_info in self.tokens.items():
                    token_data = token_info["token_data"]
                    
                    # Filtere nach Asset-ID, falls angegeben
                    if asset_id and token_data["asset_id"] != asset_id:
                        continue
                    
                    # Erstelle eine vereinfachte Zusammenfassung
                    summary = {
                        "token_id": token_id,
                        "asset_id": token_data["asset_id"],
                        "created_at": token_data["created_at"],
                        "expires_at": token_data["expires_at"],
                        "allowed_operations": token_data["allowed_operations"],
                        "remaining_queries": token_data["remaining_queries"],
                        "max_queries": token_data["max_queries"]
                    }
                    
                    token_list.append(summary)
            
            # Protokolliere das Ereignis
            self._log_audit_event("tokens_listed", {
                "count": len(token_list),
                "filtered_by_asset": asset_id is not None
            })
            
            return {
                "success": True,
                "tokens": token_list,
                "count": len(token_list)
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Auflisten der Tokens: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Auflisten der Tokens: {str(e)}"
            }

    # Hauptfunktionen

    def execute_operation(self, encrypted_data: bytes, operation: str, params: Dict = None) -> Dict[str, Any]:
        """
        Führt eine Operation auf verschlüsselten Daten aus.

        Args:
            encrypted_data: Verschlüsselte Daten
            operation: Auszuführende Operation
            params: Parameter für die Operation

        Returns:
            Dict: Ergebnis der Operation
        """
        try:
            # Prüfe, ob die Operation unterstützt wird
            if operation not in self.allowed_operations:
                return {
                    "success": False,
                    "error": f"Operation {operation} wird nicht unterstützt"
                }
            
            # Entschlüssele die Daten
            data = self._decrypt_data(encrypted_data)
            
            # Führe die Operation aus
            operation_func = self.allowed_operations[operation]
            result = operation_func(data, params or {})
            
            # Protokolliere das Ereignis
            self._log_audit_event("operation_executed", {
                "operation": operation,
                "record_count": len(data),
                "success": "error" not in result
            })
            
            return {
                "success": "error" not in result,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung der Operation {operation}: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler bei der Ausführung der Operation: {str(e)}"
            }

    def process_query_with_token(self, token: str, operation: str, params: Dict = None) -> Dict[str, Any]:
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
            # Inkrementiere den Anfragezähler
            self.request_counter += 1
            
            # Validiere das Token
            validation = self.validate_access_token(token, operation)
            
            if not validation["valid"]:
                # Protokolliere das Ereignis
                self._log_audit_event("token_validation_failed", {
                    "reason": validation["reason"],
                    "operation": operation
                })
                
                return {
                    "success": False,
                    "error": f"Ungültiges Token: {validation.get('reason', 'Unbekannter Grund')}"
                }
            
            # Extrahiere Asset-ID und Token-ID
            asset_id = validation["asset_id"]
            token_id = validation["token_id"]
            
            # Prüfe, ob der Asset existiert
            with self.asset_lock:
                if asset_id not in self.assets:
                    return {
                        "success": False,
                        "error": f"Daten-Asset {asset_id} nicht gefunden"
                    }
                
                # Hole verschlüsselte Daten
                encrypted_data = self.assets[asset_id]["encrypted_data"]
            
            # Führe die Operation aus
            result = self.execute_operation(encrypted_data, operation, params)
            
            # Reduziere die verbleibenden Abfragen
            with self.token_lock:
                self.tokens[token_id]['token_data']['remaining_queries'] -= 1
                remaining_queries = self.tokens[token_id]['token_data']['remaining_queries']
            
            # Protokolliere das Ereignis
            self._log_audit_event("query_processed", {
                "token_id": token_id,
                "asset_id": asset_id,
                "operation": operation,
                "success": result["success"],
                "remaining_queries": remaining_queries
            })
            
            # Füge Nutzungsinformationen zum Ergebnis hinzu
            result["usage_info"] = {
                "token_id": token_id,
                "remaining_queries": remaining_queries,
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "request_id": self.request_counter
            }
            
            logger.info(f"Abfrage mit Token {token_id} für Asset {asset_id} verarbeitet")
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung der Abfrage: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler bei der Abfrageverarbeitung: {str(e)}"
            }

    def get_audit_log(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Gibt das Audit-Log zurück, optional gefiltert.

        Args:
            filters: Filter für das Audit-Log (optional)

        Returns:
            Dict: Audit-Log-Einträge
        """
        try:
            with self.audit_lock:
                # Kopiere das Audit-Log, um es nicht zu verändern
                log_entries = self.audit_log.copy()
            
            # Filtere das Log, falls Filter angegeben sind
            if filters:
                filtered_entries = []
                
                for entry in log_entries:
                    # Prüfe, ob der Eintrag alle Filter erfüllt
                    matches = True
                    
                    for key, value in filters.items():
                        # Unterstütze verschachtelte Filter mit Punktnotation (z.B. "details.asset_id")
                        keys = key.split('.')
                        entry_value = entry
                        
                        for k in keys:
                            if k in entry_value:
                                entry_value = entry_value[k]
                            else:
                                matches = False
                                break
                        
                        if not matches or entry_value != value:
                            matches = False
                            break
                    
                    if matches:
                        filtered_entries.append(entry)
                
                log_entries = filtered_entries
            
            return {
                "success": True,
                "log_entries": log_entries,
                "count": len(log_entries)
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Audit-Logs: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Abrufen des Audit-Logs: {str(e)}"
            }

    def clear_audit_log(self) -> Dict[str, Any]:
        """
        Löscht das Audit-Log.

        Returns:
            Dict: Ergebnis des Löschvorgangs
        """
        try:
            with self.audit_lock:
                entry_count = len(self.audit_log)
                self.audit_log = []
            
            logger.info(f"Audit-Log gelöscht ({entry_count} Einträge)")
            
            return {
                "success": True,
                "message": f"Audit-Log erfolgreich gelöscht ({entry_count} Einträge)"
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Löschen des Audit-Logs: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Löschen des Audit-Logs: {str(e)}"
            }

    def get_public_key(self) -> Dict[str, Any]:
        """
        Gibt den öffentlichen Schlüssel für die sichere Kommunikation zurück.

        Returns:
            Dict: Öffentlicher Schlüssel
        """
        if self.public_key_pem:
            return {
                "success": True,
                "public_key": self.public_key_pem.decode(),
                "algorithm": "RSA-2048"
            }
        else:
            return {
                "success": False,
                "error": "Kein öffentlicher Schlüssel verfügbar"
            }

    def update_privacy_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aktualisiert die Datenschutzkonfiguration.

        Args:
            config_updates: Zu aktualisierende Konfigurationsparameter

        Returns:
            Dict: Aktualisierte Konfiguration
        """
        try:
            # Aktualisiere die Konfiguration
            self.privacy_config.update(config_updates)
            
            # Protokolliere das Ereignis
            self._log_audit_event("privacy_config_updated", {
                "updated_params": list(config_updates.keys())
            })
            
            logger.info(f"Datenschutzkonfiguration aktualisiert: {config_updates}")
            
            return {
                "success": True,
                "privacy_config": self.privacy_config
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Aktualisierung der Datenschutzkonfiguration: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler bei der Aktualisierung der Datenschutzkonfiguration: {str(e)}"
            }

    def get_system_info(self) -> Dict[str, Any]:
        """
        Gibt Systeminformationen zurück.

        Returns:
            Dict: Systeminformationen
        """
        try:
            info = {
                "version": "1.0.0",
                "assets_count": len(self.assets),
                "tokens_count": len(self.tokens),
                "audit_log_entries": len(self.audit_log),
                "request_counter": self.request_counter,
                "privacy_config": self.privacy_config,
                "supported_operations": list(self.allowed_operations.keys()),
                "has_encryption": True,
                "has_asymmetric_encryption": self.public_key is not None,
                "system_time": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "system_info": info
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Systeminformationen: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Abrufen der Systeminformationen: {str(e)}"
            }