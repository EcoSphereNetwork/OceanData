"""
OceanData - Modularer Datenintegrationsalgorithmus

Dieser Algorithmus ist der Kern der OceanData-Plattform und ermöglicht:
1. Datenerfassung aus verschiedenen Quellen
2. Datenbereinigung und -vorbereitung
3. Datenpseudonymisierung für Datenschutz
4. Tokenisierung der Daten mittels Ocean Protocol
5. Wertschätzung und Analyse der Daten
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import json
import logging
import uuid
import os
import math
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from abc import ABC, abstractmethod

# Konfiguriere Logger
logger = logging.getLogger("OceanData.DataIntegration")

# Enums für Datentypen und Kategorien
class DataCategory(Enum):
    """Kategorien von Daten für die Monetarisierung."""
    BROWSING = "browsing"
    HEALTH = "health"
    LOCATION = "location"
    FINANCIAL = "financial"
    SOCIAL = "social"
    IOT = "iot"
    CUSTOM = "custom"

class PrivacyLevel(Enum):
    """Datenschutzniveaus für die Datenmonetarisierung."""
    LOW = "low"         # Minimale Anonymisierung
    MEDIUM = "medium"   # Standard-Anonymisierung
    HIGH = "high"       # Maximale Anonymisierung
    COMPUTE_ONLY = "compute_only"  # Nur Compute-to-Data, keine Datenfreigabe

# Basisklasse für Datenkonnektoren
class DataConnector(ABC):
    """Abstrakte Basisklasse für alle Datenkonnektoren."""
    
    def __init__(self, source_id: str, user_id: str, category: DataCategory = DataCategory.CUSTOM):
        """
        Initialisiert einen Datenkonnektor.
        
        Args:
            source_id: Eindeutige ID der Datenquelle
            user_id: ID des Benutzers, dessen Daten abgerufen werden
            category: Kategorie der Daten (z.B. BROWSING, HEALTH)
        """
        self.source_id = source_id
        self.user_id = user_id
        self.category = category
        self.data = None
        self.metadata = {
            "source_id": source_id,
            "user_id": user_id,
            "category": category.value,
            "created_at": datetime.now().isoformat(),
            "record_count": 0,
            "fields": [],
            "data_quality_score": 0.0,
            "estimated_value": 0.0
        }
        self.connection_status = False
        self.last_sync = None
        
        logger.info(f"Datenkonnektor für {source_id} (Kategorie: {category.value}) initialisiert")
        
    @abstractmethod
    def connect(self) -> bool:
        """
        Verbindung zur Datenquelle herstellen.
        
        Returns:
            bool: True, wenn die Verbindung erfolgreich hergestellt wurde, sonst False
        """
        pass
    
    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """
        Daten aus der Quelle abrufen.
        
        Returns:
            pd.DataFrame: Die abgerufenen Daten als DataFrame
        """
        pass
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Grundlegende Datenvorverarbeitung.
        
        Args:
            data: Die zu verarbeitenden Daten
            
        Returns:
            pd.DataFrame: Die vorverarbeiteten Daten
        """
        if not isinstance(data, pd.DataFrame):
            logger.warning(f"Daten für {self.source_id} sind kein DataFrame, überspringe Vorverarbeitung")
            return data
            
        # Ursprüngliche Größe protokollieren
        original_size = len(data)
        original_columns = list(data.columns)
        
        # Entferne Nullwerte
        data = data.dropna(how='all')
        
        # Entferne Duplikate
        if 'timestamp' in data.columns:
            data = data.drop_duplicates(subset=['timestamp', 'user_id'])
        else:
            data = data.drop_duplicates()
            
        # Konvertiere Zeitstempel
        if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            except Exception as e:
                logger.error(f"Fehler beim Konvertieren des Zeitstempels: {e}")
        
        # Aktualisiere Metadaten
        self.metadata["fields"] = list(data.columns)
        self.metadata["record_count"] = len(data)
        self.metadata["preprocessing"] = {
            "original_size": original_size,
            "processed_size": len(data),
            "removed_records": original_size - len(data),
            "original_columns": original_columns
        }
        
        # Berechne einfachen Datenqualitätsscore (0-1)
        if original_size > 0:
            completeness = len(data) / original_size
            field_count = len(data.columns) / max(len(original_columns), 1)
            self.metadata["data_quality_score"] = round((completeness * 0.7 + field_count * 0.3), 2)
        
        logger.info(f"Vorverarbeitung für {self.source_id} abgeschlossen: {len(data)} Datensätze")
        return data
    
    def anonymize(self, data: pd.DataFrame, privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM) -> pd.DataFrame:
        """
        Datenpseudonmymisierung für Datenschutz.
        
        Args:
            data: Die zu anonymisierenden Daten
            privacy_level: Gewünschtes Datenschutzniveau
            
        Returns:
            pd.DataFrame: Die anonymisierten Daten
        """
        if not isinstance(data, pd.DataFrame):
            logger.warning(f"Daten für {self.source_id} sind kein DataFrame, überspringe Anonymisierung")
            return data
            
        # Kopie erstellen, um die Originaldaten nicht zu verändern
        anonymized_data = data.copy()
        
        # Benutzer-ID hashen, um Anonymität zu gewährleisten
        if 'user_id' in anonymized_data.columns:
            # Verwende einen konsistenten Salt für den gleichen Tag
            salt = datetime.now().strftime("%Y%m%d")
            anonymized_data['user_id'] = anonymized_data['user_id'].apply(
                lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest()
            )
        
        # Je nach Datenschutzniveau weitere Anonymisierungsschritte durchführen
        if privacy_level == PrivacyLevel.MEDIUM or privacy_level == PrivacyLevel.HIGH:
            # Entferne direkte Identifikatoren
            direct_identifiers = ['name', 'email', 'phone', 'address', 'ip_address', 'device_id']
            for col in direct_identifiers:
                if col in anonymized_data.columns:
                    anonymized_data.drop(columns=[col], inplace=True)
            
            # Bei hohem Datenschutzniveau weitere Maßnahmen
            if privacy_level == PrivacyLevel.HIGH:
                # Zeitstempel auf Tagesebene aggregieren
                if 'timestamp' in anonymized_data.columns:
                    anonymized_data['timestamp'] = anonymized_data['timestamp'].dt.floor('D')
                
                # Standortdaten vergröbern (falls vorhanden)
                for col in ['latitude', 'longitude', 'lat', 'lon', 'coords']:
                    if col in anonymized_data.columns and pd.api.types.is_numeric_dtype(anonymized_data[col]):
                        # Runde auf 1 Dezimalstelle (ca. 11km Genauigkeit)
                        anonymized_data[col] = anonymized_data[col].round(1)
        
        # Bei Compute-Only keine Daten zurückgeben, nur Metadaten
        if privacy_level == PrivacyLevel.COMPUTE_ONLY:
            # Erstelle ein leeres DataFrame mit den gleichen Spalten
            anonymized_data = pd.DataFrame(columns=anonymized_data.columns)
            logger.info(f"Compute-Only-Modus für {self.source_id}: Keine Daten werden zurückgegeben")
        
        # Aktualisiere Metadaten
        self.metadata["privacy"] = {
            "level": privacy_level.value,
            "applied_at": datetime.now().isoformat(),
            "original_columns": list(data.columns),
            "anonymized_columns": list(anonymized_data.columns),
            "removed_columns": [col for col in data.columns if col not in anonymized_data.columns]
        }
        
        logger.info(f"Anonymisierung für {self.source_id} mit Level {privacy_level.value} abgeschlossen")
        return anonymized_data
    
    def get_data(self, privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM) -> Dict[str, Any]:
        """
        Vollständige Pipeline zum Abrufen und Vorverarbeiten von Daten.
        
        Args:
            privacy_level: Gewünschtes Datenschutzniveau
            
        Returns:
            Dict: Ein Dictionary mit den Daten und Metadaten
        """
        try:
            # Verbindung herstellen
            self.connection_status = self.connect()
            if not self.connection_status:
                logger.error(f"Verbindung zu {self.source_id} konnte nicht hergestellt werden")
                return {
                    "status": "error",
                    "message": f"Verbindung zu {self.source_id} konnte nicht hergestellt werden",
                    "data": None,
                    "metadata": self.metadata
                }
            
            # Daten abrufen
            start_time = datetime.now()
            raw_data = self.fetch_data()
            fetch_duration = (datetime.now() - start_time).total_seconds()
            
            # Vorverarbeitung
            processed_data = self.preprocess(raw_data)
            
            # Anonymisierung
            anonymized_data = self.anonymize(processed_data, privacy_level)
            
            # Metadaten aktualisieren
            self.last_sync = datetime.now()
            self.metadata.update({
                "last_sync": self.last_sync.isoformat(),
                "fetch_duration_seconds": fetch_duration,
                "privacy_level": privacy_level.value
            })
            
            # Schätze den Wert der Daten basierend auf verschiedenen Faktoren
            self._estimate_data_value(anonymized_data)
            
            # Speichere die Daten
            self.data = anonymized_data
            
            logger.info(f"Daten für {self.source_id} erfolgreich abgerufen und verarbeitet")
            return {
                "status": "success",
                "message": "Daten erfolgreich abgerufen",
                "data": anonymized_data,
                "metadata": self.metadata
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Daten für {self.source_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Fehler beim Abrufen der Daten: {str(e)}",
                "data": None,
                "metadata": self.metadata
            }
    
    def _estimate_data_value(self, data: pd.DataFrame) -> float:
        """
        Schätzt den Wert der Daten basierend auf verschiedenen Faktoren.
        
        Args:
            data: Die zu bewertenden Daten
            
        Returns:
            float: Der geschätzte Wert der Daten in OCEAN-Token
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            self.metadata["estimated_value"] = 0.0
            return 0.0
        
        # Basiswert basierend auf der Datenkategorie
        category_values = {
            DataCategory.BROWSING: 1.0,
            DataCategory.HEALTH: 2.5,
            DataCategory.LOCATION: 2.0,
            DataCategory.FINANCIAL: 3.0,
            DataCategory.SOCIAL: 1.5,
            DataCategory.IOT: 1.8,
            DataCategory.CUSTOM: 1.0
        }
        
        base_value = category_values.get(self.category, 1.0)
        
        # Faktoren für die Wertberechnung
        record_count_factor = min(len(data) / 1000, 5.0)  # Max 5x Multiplikator für große Datensätze
        field_count_factor = min(len(data.columns) / 5, 2.0)  # Max 2x Multiplikator für viele Felder
        quality_factor = self.metadata.get("data_quality_score", 0.5)
        
        # Zeitfaktor: Neuere Daten sind wertvoller
        time_factor = 1.0
        if 'timestamp' in data.columns:
            latest_date = data['timestamp'].max()
            if isinstance(latest_date, pd.Timestamp):
                days_old = (datetime.now() - latest_date.to_pydatetime()).days
                time_factor = max(0.5, 1.0 - (days_old / 365))  # Reduziere Wert für ältere Daten
        
        # Berechne Gesamtwert
        estimated_value = base_value * record_count_factor * field_count_factor * quality_factor * time_factor
        
        # Runde auf 2 Dezimalstellen
        estimated_value = round(estimated_value, 2)
        
        # Aktualisiere Metadaten
        self.metadata["estimated_value"] = estimated_value
        self.metadata["value_factors"] = {
            "base_value": base_value,
            "record_count_factor": round(record_count_factor, 2),
            "field_count_factor": round(field_count_factor, 2),
            "quality_factor": round(quality_factor, 2),
            "time_factor": round(time_factor, 2)
        }
        
        logger.info(f"Geschätzter Wert für {self.source_id}: {estimated_value} OCEAN")
        return estimated_value

# Spezifische Konnektoren für verschiedene Datenquellen
class BrowserDataConnector(DataConnector):
    """Konnektor für Browser-Daten."""
    
    def __init__(self, user_id: str, browser_type: str = 'chrome'):
        """
        Initialisiert einen Browser-Datenkonnektor.
        
        Args:
            user_id: ID des Benutzers
            browser_type: Typ des Browsers (chrome, firefox, safari, etc.)
        """
        super().__init__('browser', user_id, DataCategory.BROWSING)
        self.browser_type = browser_type
        self.metadata.update({
            "browser_type": browser_type,
            "source_details": {
                "name": f"{browser_type.capitalize()} Browser History",
                "version": "1.0",
                "description": f"Browsing history and activity data from {browser_type.capitalize()} browser"
            }
        })
    
    def connect(self) -> bool:
        """
        Verbindung zum Browser-Verlauf herstellen.
        
        Returns:
            bool: True, wenn die Verbindung erfolgreich hergestellt wurde
        """
        try:
            # In einer echten Implementierung würden wir hier die Verbindung zum Browser herstellen
            # Für das MVP simulieren wir eine erfolgreiche Verbindung
            logger.info(f"Verbindung zu {self.browser_type}-Daten für Benutzer {self.user_id} hergestellt")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Verbinden mit {self.browser_type}: {str(e)}")
            return False
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Browser-Verlaufsdaten abrufen.
        
        Returns:
            pd.DataFrame: Die abgerufenen Browser-Daten
        """
        # In einer echten Implementierung würden wir hier die tatsächlichen Daten abrufen
        # Für das MVP generieren wir realistische Beispieldaten
        
        # Generiere zufällige Zeitstempel der letzten 30 Tage
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, periods=100)
        
        # Liste von häufig besuchten Websites
        common_websites = [
            'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'wikipedia.org',
            'twitter.com', 'instagram.com', 'linkedin.com', 'github.com', 'stackoverflow.com',
            'reddit.com', 'netflix.com', 'twitch.tv', 'ebay.com', 'nytimes.com'
        ]
        
        # Generiere realistische Browsing-Daten
        num_records = 100
        websites = np.random.choice(common_websites, num_records)
        timestamps = np.random.choice(dates, num_records)
        durations = np.random.randint(10, 1800, num_records)  # 10 Sekunden bis 30 Minuten
        
        # Füge Kategorien hinzu
        categories = {
            'google.com': 'search', 'youtube.com': 'video', 'facebook.com': 'social',
            'amazon.com': 'shopping', 'wikipedia.org': 'education', 'twitter.com': 'social',
            'instagram.com': 'social', 'linkedin.com': 'professional', 'github.com': 'development',
            'stackoverflow.com': 'development', 'reddit.com': 'social', 'netflix.com': 'entertainment',
            'twitch.tv': 'entertainment', 'ebay.com': 'shopping', 'nytimes.com': 'news'
        }
        site_categories = [categories.get(site, 'other') for site in websites]
        
        # Erstelle DataFrame
        sample_data = {
            'website': websites,
            'timestamp': timestamps,
            'duration': durations,
            'user_id': [self.user_id] * num_records,
            'browser': [self.browser_type] * num_records,
            'category': site_categories
        }
        
        df = pd.DataFrame(sample_data)
        
        # Sortiere nach Zeitstempel
        df = df.sort_values('timestamp')
        
        logger.info(f"Browser-Daten abgerufen: {len(df)} Einträge")
        return df
    
    def extract_features(self) -> pd.DataFrame:
        """
        Extrahiere spezifische Features aus Browser-Daten für erweiterte Analysen.
        
        Returns:
            pd.DataFrame: Die Daten mit extrahierten Features
        """
        if self.data is None or not isinstance(self.data, pd.DataFrame) or self.data.empty:
            logger.warning("Keine Daten für Feature-Extraktion verfügbar")
            return pd.DataFrame()
        
        # Kopie erstellen, um die Originaldaten nicht zu verändern
        df = self.data.copy()
        
        # Zeitbasierte Features
        if 'timestamp' in df.columns:
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            df['is_working_hours'] = df['hour_of_day'].apply(lambda x: 1 if 9 <= x <= 17 else 0)
        
        # Kategoriebasierte Features
        if 'category' in df.columns:
            # One-hot encoding für Kategorien
            category_dummies = pd.get_dummies(df['category'], prefix='category')
            df = pd.concat([df, category_dummies], axis=1)
        
        # Domainbasierte Features
        if 'website' in df.columns:
            # Extrahiere Domain-Teile
            df['domain_tld'] = df['website'].apply(lambda x: x.split('.')[-1] if '.' in x else '')
        
        # Nutzungsintensität
        if 'duration' in df.columns:
            # Kategorisiere Dauer
            df['duration_category'] = pd.cut(
                df['duration'],
                bins=[0, 60, 300, 900, 1800, float('inf')],
                labels=['very_short', 'short', 'medium', 'long', 'very_long']
            )
        
        logger.info(f"Features extrahiert: {len(df.columns)} Spalten")
        return df

class SmartwatchDataConnector(DataConnector):
    """Konnektor für Smartwatch-Daten."""
    
    def __init__(self, user_id: str, device_brand: str = 'fitbit'):
        """
        Initialisiert einen Smartwatch-Datenkonnektor.
        
        Args:
            user_id: ID des Benutzers
            device_brand: Marke der Smartwatch (fitbit, apple, garmin, etc.)
        """
        super().__init__('smartwatch', user_id, DataCategory.HEALTH)
        self.device_brand = device_brand
        self.metadata.update({
            "device_brand": device_brand,
            "source_details": {
                "name": f"{device_brand.capitalize()} Health Data",
                "version": "1.0",
                "description": f"Health and activity data from {device_brand.capitalize()} smartwatch"
            }
        })
    
    def connect(self) -> bool:
        """
        Verbindung zur Smartwatch-API herstellen.
        
        Returns:
            bool: True, wenn die Verbindung erfolgreich hergestellt wurde
        """
        try:
            # In einer echten Implementierung würden wir hier die Verbindung zur API herstellen
            # Für das MVP simulieren wir eine erfolgreiche Verbindung
            logger.info(f"Verbindung zu {self.device_brand}-API für Benutzer {self.user_id} hergestellt")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Verbinden mit {self.device_brand}-API: {str(e)}")
            return False
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Smartwatch-Daten abrufen.
        
        Returns:
            pd.DataFrame: Die abgerufenen Gesundheitsdaten
        """
        # In einer echten Implementierung würden wir hier die tatsächlichen Daten abrufen
        # Für das MVP generieren wir realistische Beispieldaten
        
        # Generiere Zeitstempel für die letzten 7 Tage mit stündlichen Messungen
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        num_records = len(timestamps)
        
        # Generiere realistische Gesundheitsdaten
        # Herzfrequenz: 60-100 bpm mit täglichem Muster
        base_heart_rate = 70
        heart_rates = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            # Niedrigere Herzfrequenz nachts, höher tagsüber
            if 0 <= hour < 6:
                hr = base_heart_rate - 10 + np.random.randint(-5, 5)
            elif 6 <= hour < 9:
                hr = base_heart_rate + 10 + np.random.randint(-5, 10)
            elif 9 <= hour < 18:
                hr = base_heart_rate + 5 + np.random.randint(-5, 15)
            elif 18 <= hour < 22:
                hr = base_heart_rate + np.random.randint(-5, 10)
            else:
                hr = base_heart_rate - 5 + np.random.randint(-5, 5)
            heart_rates.append(max(50, min(120, hr)))
        
        # Schritte: 0-1000 pro Stunde mit täglichem Muster
        steps = []
        for ts in timestamps:
            hour = ts.hour
            # Weniger Schritte nachts, mehr tagsüber
            if 0 <= hour < 6:
                s = np.random.randint(0, 20)
            elif 6 <= hour < 9:
                s = np.random.randint(500, 2000)
            elif 9 <= hour < 18:
                s = np.random.randint(200, 1000)
            elif 18 <= hour < 22:
                s = np.random.randint(300, 1500)
            else:
                s = np.random.randint(0, 200)
            steps.append(s)
        
        # Kalorienverbrauch: Korreliert mit Schritten und Herzfrequenz
        calories = []
        for i in range(num_records):
            base_calories = steps[i] * 0.05  # Ungefähr 0.05 Kalorien pro Schritt
            hr_factor = heart_rates[i] / 70.0  # Herzfrequenzfaktor
            calories.append(int(base_calories * hr_factor + np.random.randint(0, 20)))
        
        # Schlafphasen: Nur für Nachtstunden
        sleep_states = []
        for ts in timestamps:
            hour = ts.hour
            if 0 <= hour < 6:
                # Tiefschlaf, REM, leichter Schlaf
                sleep_states.append(np.random.choice(['deep', 'rem', 'light'], p=[0.3, 0.2, 0.5]))
            elif 22 <= hour < 24:
                # Hauptsächlich leichter Schlaf am frühen Abend
                sleep_states.append(np.random.choice(['awake', 'light'], p=[0.2, 0.8]))
            else:
                sleep_states.append('awake')
        
        # Erstelle DataFrame
        sample_data = {
            'timestamp': timestamps,
            'heart_rate': heart_rates,
            'steps': steps,
            'calories': calories,
            'sleep_state': sleep_states,
            'user_id': [self.user_id] * num_records,
            'device': [self.device_brand] * num_records
        }
        
        df = pd.DataFrame(sample_data)
        
        logger.info(f"Smartwatch-Daten abgerufen: {len(df)} Einträge")
        return df
    
    def extract_features(self) -> pd.DataFrame:
        """
        Extrahiere spezifische Features aus Smartwatch-Daten für erweiterte Analysen.
        
        Returns:
            pd.DataFrame: Die Daten mit extrahierten Features
        """
        if self.data is None or not isinstance(self.data, pd.DataFrame) or self.data.empty:
            logger.warning("Keine Daten für Feature-Extraktion verfügbar")
            return pd.DataFrame()
        
        # Kopie erstellen, um die Originaldaten nicht zu verändern
        df = self.data.copy()
        
        # Zeitbasierte Features
        if 'timestamp' in df.columns:
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            df['date'] = df['timestamp'].dt.date
        
        # Tägliche Aggregate
        if 'date' in df.columns:
            # Tägliche Schritte
            if 'steps' in df.columns:
                daily_steps = df.groupby('date')['steps'].sum().reset_index()
                daily_steps.columns = ['date', 'daily_steps']
                df = pd.merge(df, daily_steps, on='date', how='left')
            
            # Tägliche Kalorien
            if 'calories' in df.columns:
                daily_calories = df.groupby('date')['calories'].sum().reset_index()
                daily_calories.columns = ['date', 'daily_calories']
                df = pd.merge(df, daily_calories, on='date', how='left')
            
            # Durchschnittliche Herzfrequenz
            if 'heart_rate' in df.columns:
                daily_hr_avg = df.groupby('date')['heart_rate'].mean().reset_index()
                daily_hr_avg.columns = ['date', 'daily_avg_hr']
                df = pd.merge(df, daily_hr_avg, on='date', how='left')
                
                # Ruheherzfrequenz (Durchschnitt während des Schlafs)
                if 'sleep_state' in df.columns:
                    sleep_mask = df['sleep_state'] != 'awake'
                    if sleep_mask.any():
                        sleep_hr = df[sleep_mask].groupby('date')['heart_rate'].mean().reset_index()
                        sleep_hr.columns = ['date', 'resting_hr']
                        df = pd.merge(df, sleep_hr, on='date', how='left')
        
        # Schlafanalyse
        if 'sleep_state' in df.columns:
            # One-hot encoding für Schlafzustände
            sleep_dummies = pd.get_dummies(df['sleep_state'], prefix='sleep')
            df = pd.concat([df, sleep_dummies], axis=1)
            
            # Schlafqualität (vereinfachte Berechnung)
            if 'sleep_deep' in df.columns and 'sleep_rem' in df.columns:
                df['sleep_quality'] = (df['sleep_deep'] * 1.0 + df['sleep_rem'] * 0.8 + 
                                      df['sleep_light'] * 0.6 + df['sleep_awake'] * 0.0)
        
        logger.info(f"Features extrahiert: {len(df.columns)} Spalten")
        return df

class IoTDeviceConnector(DataConnector):
    """Konnektor für IoT-Gerätedaten."""
    
    def __init__(self, user_id: str, device_type: str, device_id: str = None):
        """
        Initialisiert einen IoT-Geräte-Datenkonnektor.
        
        Args:
            user_id: ID des Benutzers
            device_type: Typ des IoT-Geräts (thermostat, camera, speaker, etc.)
            device_id: Eindeutige ID des Geräts (falls vorhanden)
        """
        super().__init__('iot_' + device_type, user_id, DataCategory.IOT)
        self.device_type = device_type
        self.device_id = device_id or f"{device_type}_{uuid.uuid4().hex[:8]}"
        self.metadata.update({
            "device_type": device_type,
            "device_id": self.device_id,
            "source_details": {
                "name": f"{device_type.capitalize()} IoT Data",
                "version": "1.0",
                "description": f"Usage and sensor data from {device_type} IoT device"
            }
        })
    
    def connect(self) -> bool:
        """
        Verbindung zum IoT-Gerät herstellen.
        
        Returns:
            bool: True, wenn die Verbindung erfolgreich hergestellt wurde
        """
        try:
            # In einer echten Implementierung würden wir hier die Verbindung zum Gerät herstellen
            # Für das MVP simulieren wir eine erfolgreiche Verbindung
            logger.info(f"Verbindung zu {self.device_type} (ID: {self.device_id}) für Benutzer {self.user_id} hergestellt")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Verbinden mit {self.device_type}: {str(e)}")
            return False
    
    def fetch_data(self) -> pd.DataFrame:
        """
        IoT-Gerätedaten abrufen.
        
        Returns:
            pd.DataFrame: Die abgerufenen IoT-Daten
        """
        # In einer echten Implementierung würden wir hier die tatsächlichen Daten abrufen
        # Für das MVP generieren wir realistische Beispieldaten basierend auf dem Gerätetyp
        
        # Generiere Zeitstempel für die letzten 7 Tage mit stündlichen Messungen
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        num_records = len(timestamps)
        
        # Gemeinsame Felder für alle Gerätetypen
        common_data = {
            'timestamp': timestamps,
            'user_id': [self.user_id] * num_records,
            'device_id': [self.device_id] * num_records,
            'device_type': [self.device_type] * num_records,
            'connection_status': np.random.choice(['connected', 'disconnected'], num_records, p=[0.95, 0.05])
        }
        
        # Gerätespezifische Daten
        if self.device_type == 'thermostat':
            # Thermostat-Daten: Temperatur, Zieltemperatur, Heizungsstatus
            device_data = {
                'temperature': [round(20 + np.random.normal(0, 2), 1) for _ in range(num_records)],
                'target_temperature': [round(21 + np.random.normal(0, 1), 1) for _ in range(num_records)],
                'heating_status': np.random.choice(['on', 'off'], num_records, p=[0.3, 0.7]),
                'humidity': [round(40 + np.random.normal(0, 5), 1) for _ in range(num_records)]
            }
        elif self.device_type == 'light':
            # Smart-Light-Daten: Status, Helligkeit, Farbe
            device_data = {
                'status': np.random.choice(['on', 'off'], num_records, p=[0.4, 0.6]),
                'brightness': [round(np.random.randint(10, 100), -1) for _ in range(num_records)],
                'color_temperature': [np.random.randint(2700, 6500) for _ in range(num_records)]
            }
        elif self.device_type == 'security_camera':
            # Sicherheitskamera-Daten: Status, Bewegungserkennung
            device_data = {
                'status': np.random.choice(['recording', 'standby', 'off'], num_records, p=[0.2, 0.7, 0.1]),
                'motion_detected': np.random.choice([True, False], num_records, p=[0.1, 0.9]),
                'storage_used_percent': [min(100, max(0, np.random.normal(50, 10))) for _ in range(num_records)]
            }
        else:
            # Generische IoT-Daten für andere Gerätetypen
            device_data = {
                'status': np.random.choice(['active', 'standby', 'off'], num_records, p=[0.5, 0.3, 0.2]),
                'power_consumption': [round(np.random.uniform(0.5, 10.0), 2) for _ in range(num_records)]
            }
        
        # Kombiniere gemeinsame und gerätespezifische Daten
        sample_data = {**common_data, **device_data}
        df = pd.DataFrame(sample_data)
        
        logger.info(f"IoT-Daten für {self.device_type} abgerufen: {len(df)} Einträge")
        return df

# Datenintegrator-Klasse
class DataIntegrator:
    """Klasse für die Integration verschiedener Datenquellen."""
    
    def __init__(self, user_id: str):
        """
        Initialisiert einen Datenintegrator für einen Benutzer.
        
        Args:
            user_id: ID des Benutzers
        """
        self.user_id = user_id
        self.connectors = {}
        self.integrated_data = {}
        self.metadata = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": None,
            "sources": [],
            "total_records": 0,
            "total_fields": 0,
            "estimated_total_value": 0.0,
            "privacy_level": None
        }
        logger.info(f"Datenintegrator für Benutzer {user_id} initialisiert")
    
    def add_connector(self, connector: DataConnector) -> bool:
        """
        Fügt einen neuen Datenkonnektor hinzu.
        
        Args:
            connector: Der hinzuzufügende Datenkonnektor
            
        Returns:
            bool: True, wenn der Konnektor erfolgreich hinzugefügt wurde
        """
        if not isinstance(connector, DataConnector):
            logger.error(f"Ungültiger Konnektor-Typ: {type(connector)}")
            return False
            
        self.connectors[connector.source_id] = connector
        logger.info(f"Konnektor für {connector.source_id} hinzugefügt")
        
        # Aktualisiere Metadaten
        if connector.source_id not in self.metadata["sources"]:
            self.metadata["sources"].append(connector.source_id)
            
        return True
    
    def collect_all_data(self, privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM) -> Dict[str, Any]:
        """
        Sammelt Daten von allen registrierten Konnektoren.
        
        Args:
            privacy_level: Datenschutzniveau für die Datensammlung
            
        Returns:
            Dict: Ein Dictionary mit den gesammelten Daten und Metadaten
        """
        start_time = datetime.now()
        self.integrated_data = {}
        total_records = 0
        total_fields = set()
        total_value = 0.0
        
        # Sammle Daten von allen Konnektoren
        for source_id, connector in self.connectors.items():
            logger.info(f"Sammle Daten von {source_id}")
            result = connector.get_data(privacy_level)
            
            if result["status"] == "success" and result["data"] is not None:
                self.integrated_data[source_id] = result["data"]
                
                # Aktualisiere Zähler
                total_records += len(result["data"])
                if isinstance(result["data"], pd.DataFrame):
                    total_fields.update(result["data"].columns)
                
                # Summiere geschätzten Wert
                total_value += result["metadata"].get("estimated_value", 0.0)
            else:
                logger.warning(f"Fehler beim Sammeln von Daten von {source_id}: {result.get('message', 'Unbekannter Fehler')}")
        
        # Aktualisiere Metadaten
        self.metadata.update({
            "last_updated": datetime.now().isoformat(),
            "total_records": total_records,
            "total_fields": len(total_fields),
            "fields_list": list(total_fields),
            "estimated_total_value": round(total_value, 2),
            "privacy_level": privacy_level.value,
            "collection_time_seconds": (datetime.now() - start_time).total_seconds(),
            "sources_collected": list(self.integrated_data.keys()),
            "sources_failed": [source_id for source_id in self.connectors.keys() if source_id not in self.integrated_data]
        })
        
        logger.info(f"Datensammlung abgeschlossen: {total_records} Datensätze aus {len(self.integrated_data)} Quellen")
        
        return {
            "status": "success" if len(self.integrated_data) > 0 else "error",
            "message": f"Daten von {len(self.integrated_data)} Quellen gesammelt",
            "data": self.integrated_data,
            "metadata": self.metadata
        }
    
    def get_dataset_metadata(self) -> Dict[str, Any]:
        """
        Erstellt Metadaten für den Datensatz zur Verwendung mit Ocean Protocol.
        
        Returns:
            Dict: Metadaten für den Datensatz
        """
        # Basis-Metadaten
        metadata = {
            'name': f'User Data Bundle - {self.user_id}',
            'description': 'Integrierte Benutzerdaten aus verschiedenen Quellen',
            'author': f'User-{self.user_id}',
            'created': datetime.now().isoformat(),
            'sources': list(self.integrated_data.keys()),
            'records_count': {
                source: len(data) if isinstance(data, pd.DataFrame) else 0 
                for source, data in self.integrated_data.items()
            },
            'total_records': self.metadata["total_records"],
            'estimated_value': self.metadata["estimated_total_value"],
            'privacy_level': self.metadata.get("privacy_level", "unknown")
        }
        
        # Erweiterte Metadaten für Ocean Protocol
        ocean_metadata = {
            'main': {
                'type': 'dataset',
                'name': metadata['name'],
                'dateCreated': metadata['created'],
                'author': metadata['author'],
                'license': 'CC-BY',
                'files': [
                    {
                        'index': 0,
                        'contentType': 'application/json',
                        'checksum': self._calculate_checksum(),
                        'checksumType': 'MD5',
                        'contentLength': self._calculate_content_size(),
                        'url': ''  # Wird später von Ocean Protocol ausgefüllt
                    }
                ]
            },
            'additionalInformation': {
                'description': metadata['description'],
                'categories': [connector.category.value for connector in self.connectors.values()],
                'tags': self._generate_tags(),
                'privacy': {
                    'level': metadata['privacy_level'],
                    'complianceStandards': ['GDPR', 'CCPA'],
                    'anonymizationMethods': ['pseudonymization', 'aggregation']
                },
                'qualityMetrics': {
                    'completeness': self._calculate_completeness(),
                    'accuracy': 0.85,  # Beispielwert
                    'timeliness': self._calculate_timeliness()
                }
            }
        }
        
        return {
            'basic': metadata,
            'ocean': ocean_metadata
        }
    
    def _calculate_checksum(self) -> str:
        """Berechnet einen Checksum für die integrierten Daten."""
        combined_hash = hashlib.md5()
        
        for source, data in sorted(self.integrated_data.items()):
            if isinstance(data, pd.DataFrame):
                # Konvertiere DataFrame in JSON für Hashing
                data_json = data.to_json()
                combined_hash.update(f"{source}:{data_json}".encode())
        
        return combined_hash.hexdigest()
    
    def _calculate_content_size(self) -> int:
        """Schätzt die Größe der integrierten Daten in Bytes."""
        total_size = 0
        
        for data in self.integrated_data.values():
            if isinstance(data, pd.DataFrame):
                # Grobe Schätzung der DataFrame-Größe
                total_size += data.memory_usage(deep=True).sum()
        
        return total_size
    
    def _generate_tags(self) -> List[str]:
        """Generiert Tags basierend auf den Datenquellen."""
        tags = []
        
        # Füge Kategorien als Tags hinzu
        for connector in self.connectors.values():
            tags.append(connector.category.value)
            
            # Füge gerätespezifische Tags hinzu
            if hasattr(connector, 'browser_type'):
                tags.append(f"browser:{connector.browser_type}")
            elif hasattr(connector, 'device_brand'):
                tags.append(f"device:{connector.device_brand}")
            elif hasattr(connector, 'device_type'):
                tags.append(f"iot:{connector.device_type}")
        
        # Entferne Duplikate und sortiere
        return sorted(list(set(tags)))
    
    def _calculate_completeness(self) -> float:
        """Berechnet die Vollständigkeit der Daten (0-1)."""
        if not self.integrated_data:
            return 0.0
            
        completeness_scores = []
        
        for data in self.integrated_data.values():
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Berechne den Anteil der nicht-null Werte
                non_null_ratio = 1.0 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
                completeness_scores.append(non_null_ratio)
        
        if not completeness_scores:
            return 0.0
            
        return round(sum(completeness_scores) / len(completeness_scores), 2)
    
    def _calculate_timeliness(self) -> float:
        """Berechnet die Aktualität der Daten (0-1)."""
        if not self.integrated_data:
            return 0.0
            
        now = datetime.now()
        timeliness_scores = []
        
        for data in self.integrated_data.values():
            if isinstance(data, pd.DataFrame) and 'timestamp' in data.columns:
                # Berechne die durchschnittliche Aktualität (1.0 für heute, abnehmend für ältere Daten)
                latest_date = data['timestamp'].max()
                if isinstance(latest_date, pd.Timestamp):
                    days_old = (now - latest_date.to_pydatetime()).days
                    # Exponentieller Abfall über 30 Tage
                    timeliness = max(0.1, min(1.0, math.exp(-days_old / 30)))
                    timeliness_scores.append(timeliness)
        
        if not timeliness_scores:
            return 0.5  # Standardwert, wenn keine Zeitstempel verfügbar sind
            
        return round(sum(timeliness_scores) / len(timeliness_scores), 2)
    
    def prepare_for_tokenization(self, privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM) -> Dict[str, Any]:
        """
        Bereitet Daten für die Tokenisierung mit Ocean Protocol vor.
        
        Args:
            privacy_level: Datenschutzniveau für die Tokenisierung
            
        Returns:
            Dict: Ein Tokenisierungspaket mit Daten und Metadaten
        """
        # Stelle sicher, dass Daten mit dem angegebenen Datenschutzniveau gesammelt wurden
        if self.metadata.get("privacy_level") != privacy_level.value:
            logger.info(f"Sammle Daten mit Datenschutzniveau {privacy_level.value}")
            self.collect_all_data(privacy_level)
        
        # Für jeden Datensatz einen Hash erstellen, um die Integrität zu gewährleisten
        dataset_hashes = {}
        for source, data in self.integrated_data.items():
            if isinstance(data, pd.DataFrame):
                # Konvertiere DataFrame in JSON für Hashing
                data_json = data.to_json()
                # Erstelle Hash für die Daten
                data_hash = hashlib.sha256(data_json.encode()).hexdigest()
                dataset_hashes[source] = data_hash
        
        # Erstelle Metadaten
        metadata = self.get_dataset_metadata()
        
        # Erstelle das Tokenisierungspaket
        tokenization_package = {
            'metadata': metadata,
            'dataset_hashes': dataset_hashes,
            'timestamp': datetime.now().isoformat(),
            'privacy_level': privacy_level.value,
            'user_id': self.user_id,
            'estimated_value': self.metadata["estimated_total_value"],
            'data_sample': self._create_data_sample()
        }
        
        logger.info(f"Tokenisierungspaket erstellt mit geschätztem Wert von {self.metadata['estimated_total_value']} OCEAN")
        return tokenization_package
    
    def _create_data_sample(self, sample_size: int = 5) -> Dict[str, Any]:
        """
        Erstellt eine Stichprobe der Daten für die Vorschau.
        
        Args:
            sample_size: Anzahl der Datensätze pro Quelle
            
        Returns:
            Dict: Eine Stichprobe der Daten
        """
        sample = {}
        
        for source, data in self.integrated_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Nehme die ersten N Zeilen als Stichprobe
                sample[source] = data.head(sample_size).to_dict(orient='records')
        
        return sample
    
    def combine_data_sources(self, combination_type: str = 'merge') -> pd.DataFrame:
        """
        Kombiniert mehrere Datenquellen zu einem einzigen DataFrame.
        
        Args:
            combination_type: Art der Kombination ('merge', 'append', 'correlate')
            
        Returns:
            pd.DataFrame: Die kombinierten Daten
        """
        if not self.integrated_data:
            logger.warning("Keine Daten zum Kombinieren vorhanden")
            return pd.DataFrame()
            
        if len(self.integrated_data) == 1:
            # Nur eine Datenquelle, gib diese zurück
            return list(self.integrated_data.values())[0]
        
        # Extrahiere DataFrames
        dataframes = {}
        for source, data in self.integrated_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                dataframes[source] = data
        
        if not dataframes:
            logger.warning("Keine gültigen DataFrames zum Kombinieren vorhanden")
            return pd.DataFrame()
        
        # Kombiniere basierend auf dem angegebenen Typ
        if combination_type == 'append':
            # Füge alle DataFrames vertikal zusammen
            combined_df = pd.concat(dataframes.values(), ignore_index=True)
            logger.info(f"Datenquellen vertikal kombiniert: {len(combined_df)} Datensätze")
            return combined_df
            
        elif combination_type == 'merge':
            # Versuche, DataFrames horizontal zu kombinieren (basierend auf gemeinsamen Schlüsseln)
            # Wir nehmen an, dass 'user_id' und 'timestamp' gemeinsame Schlüssel sind
            dfs = list(dataframes.values())
            result = dfs[0]
            
            for df in dfs[1:]:
                common_columns = set(result.columns).intersection(set(df.columns))
                if 'user_id' in common_columns:
                    # Wenn Zeitstempel vorhanden sind, verwende diese für ein genaueres Matching
                    if 'timestamp' in common_columns:
                        result = pd.merge(
                            result, df, on=['user_id', 'timestamp'], 
                            how='outer', suffixes=('', f'_{list(dataframes.keys())[dfs.index(df)]}')
                        )
                    else:
                        result = pd.merge(
                            result, df, on='user_id', 
                            how='outer', suffixes=('', f'_{list(dataframes.keys())[dfs.index(df)]}')
                        )
                else:
                    # Wenn keine gemeinsamen Schlüssel vorhanden sind, füge einfach hinzu
                    logger.warning(f"Keine gemeinsamen Schlüssel für Merge gefunden, führe Append durch")
                    return self.combine_data_sources('append')
            
            logger.info(f"Datenquellen horizontal kombiniert: {len(result)} Datensätze, {len(result.columns)} Felder")
            return result
            
        elif combination_type == 'correlate':
            # Korreliere Daten basierend auf Zeitstempeln
            # Wir aggregieren Daten auf Stundenbasis und kombinieren sie dann
            aggregated_dfs = {}
            
            for source, df in dataframes.items():
                if 'timestamp' in df.columns:
                    # Runde Zeitstempel auf Stunden
                    df = df.copy()
                    df['hour'] = df['timestamp'].dt.floor('H')
                    
                    # Aggregiere numerische Spalten
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        agg_dict = {col: 'mean' for col in numeric_cols if col != 'user_id'}
                        if 'user_id' in df.columns:
                            agg_dict['user_id'] = 'first'
                        
                        aggregated = df.groupby('hour').agg(agg_dict).reset_index()
                        aggregated.rename(columns={'hour': 'timestamp'}, inplace=True)
                        
                        # Füge Quellspalte hinzu
                        aggregated['source'] = source
                        
                        aggregated_dfs[source] = aggregated
            
            if not aggregated_dfs:
                logger.warning("Keine Daten mit Zeitstempeln für Korrelation gefunden")
                return pd.DataFrame()
                
            # Kombiniere die aggregierten DataFrames
            combined_df = pd.concat(aggregated_dfs.values(), ignore_index=True)
            
            logger.info(f"Datenquellen korreliert: {len(combined_df)} Datensätze")
            return combined_df
        
        else:
            logger.error(f"Ungültiger Kombinationstyp: {combination_type}")
            return pd.DataFrame()

# Datenanalyse-Klasse für Wertsteigerung der Daten
class DataAnalyzer:
    """Analyse-Algorithmen für verschiedene Datentypen."""
    
    def __init__(self, data_integrator: DataIntegrator):
        """
        Initialisiert einen Datenanalyzer.
        
        Args:
            data_integrator: Der Datenintegrator mit den zu analysierenden Daten
        """
        self.data_integrator = data_integrator
        self.analytics_results = {}
        self.metadata = {
            "user_id": data_integrator.user_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": None,
            "analyzed_sources": [],
            "analysis_version": "1.0.0"
        }
        logger.info(f"Datenanalyzer für Benutzer {data_integrator.user_id} initialisiert")
    
    def analyze_browser_data(self) -> Dict[str, Any]:
        """
        Analysiert Browser-Daten für wertvolle Erkenntnisse.
        
        Returns:
            Dict: Analyseergebnisse für Browser-Daten
        """
        browser_sources = [source for source in self.data_integrator.integrated_data.keys() 
                          if source.startswith('browser')]
        
        if not browser_sources:
            logger.warning("Keine Browser-Daten für Analyse gefunden")
            return {}
        
        # Kombiniere alle Browser-Datenquellen
        browser_results = {}
        
        for source in browser_sources:
            browser_data = self.data_integrator.integrated_data[source]
            
            if not isinstance(browser_data, pd.DataFrame) or browser_data.empty:
                logger.warning(f"Ungültige Browser-Daten für {source}")
                continue
                
            logger.info(f"Analysiere Browser-Daten von {source}: {len(browser_data)} Datensätze")
            
            # Grundlegende Statistiken
            stats = {
                "record_count": len(browser_data),
                "date_range": {
                    "start": browser_data['timestamp'].min().isoformat() if 'timestamp' in browser_data.columns else None,
                    "end": browser_data['timestamp'].max().isoformat() if 'timestamp' in browser_data.columns else None
                },
                "field_count": len(browser_data.columns)
            }
            
            # Website-Analyse
            website_analysis = {}
            if 'website' in browser_data.columns:
                # Top-Domains
                site_frequency = browser_data['website'].value_counts().head(10).to_dict()
                website_analysis["top_domains"] = site_frequency
                
                # Domain-Kategorien
                if 'category' in browser_data.columns:
                    category_counts = browser_data['category'].value_counts().to_dict()
                    website_analysis["category_distribution"] = category_counts
            
            # Zeitliche Analyse
            temporal_analysis = {}
            if 'timestamp' in browser_data.columns:
                # Stunden des Tages
                browser_data['hour'] = browser_data['timestamp'].dt.hour
                hourly_activity = browser_data.groupby('hour').size().to_dict()
                temporal_analysis["hourly_activity"] = hourly_activity
                
                # Tage der Woche
                browser_data['day_of_week'] = browser_data['timestamp'].dt.dayofweek
                daily_activity = browser_data.groupby('day_of_week').size().to_dict()
                temporal_analysis["daily_activity"] = {str(day): count for day, count in daily_activity.items()}
            
            # Nutzungsanalyse
            usage_analysis = {}
            if 'duration' in browser_data.columns:
                usage_analysis["avg_duration"] = float(browser_data['duration'].mean())
                usage_analysis["total_duration"] = float(browser_data['duration'].sum())
                usage_analysis["duration_distribution"] = {
                    "min": float(browser_data['duration'].min()),
                    "25%": float(browser_data['duration'].quantile(0.25)),
                    "median": float(browser_data['duration'].median()),
                    "75%": float(browser_data['duration'].quantile(0.75)),
                    "max": float(browser_data['duration'].max())
                }
            
            # Anomalieerkennung
            anomalies = self._detect_anomalies(browser_data, 'duration')
            
            # Kombiniere alle Analysen
            source_results = {
                "statistics": stats,
                "website_analysis": website_analysis,
                "temporal_analysis": temporal_analysis,
                "usage_analysis": usage_analysis,
                "anomalies": anomalies
            }
            
            browser_results[source] = source_results
            
            # Füge zur Liste der analysierten Quellen hinzu
            if source not in self.metadata["analyzed_sources"]:
                self.metadata["analyzed_sources"].append(source)
        
        # Speichere Ergebnisse
        self.analytics_results['browser'] = browser_results
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        logger.info(f"Browser-Datenanalyse abgeschlossen für {len(browser_results)} Quellen")
        return browser_results
    
    def analyze_health_data(self) -> Dict[str, Any]:
        """
        Analysiert Gesundheitsdaten (z.B. von Smartwatches) für wertvolle Erkenntnisse.
        
        Returns:
            Dict: Analyseergebnisse für Gesundheitsdaten
        """
        health_sources = [source for source in self.data_integrator.integrated_data.keys() 
                         if source.startswith('smartwatch') or source == 'health']
        
        if not health_sources:
            logger.warning("Keine Gesundheitsdaten für Analyse gefunden")
            return {}
        
        # Kombiniere alle Gesundheitsdatenquellen
        health_results = {}
        
        for source in health_sources:
            health_data = self.data_integrator.integrated_data[source]
            
            if not isinstance(health_data, pd.DataFrame) or health_data.empty:
                logger.warning(f"Ungültige Gesundheitsdaten für {source}")
                continue
                
            logger.info(f"Analysiere Gesundheitsdaten von {source}: {len(health_data)} Datensätze")
            
            # Grundlegende Statistiken
            stats = {
                "record_count": len(health_data),
                "date_range": {
                    "start": health_data['timestamp'].min().isoformat() if 'timestamp' in health_data.columns else None,
                    "end": health_data['timestamp'].max().isoformat() if 'timestamp' in health_data.columns else None
                },
                "field_count": len(health_data.columns)
            }
            
            # Herzfrequenzanalyse
            heart_rate_analysis = {}
            if 'heart_rate' in health_data.columns:
                heart_rate_analysis = {
                    "mean": float(health_data['heart_rate'].mean()),
                    "min": float(health_data['heart_rate'].min()),
                    "max": float(health_data['heart_rate'].max()),
                    "std": float(health_data['heart_rate'].std()),
                    "distribution": {
                        "low": int(health_data[health_data['heart_rate'] < 60].shape[0]),
                        "normal": int(health_data[(health_data['heart_rate'] >= 60) & (health_data['heart_rate'] <= 100)].shape[0]),
                        "elevated": int(health_data[health_data['heart_rate'] > 100].shape[0])
                    }
                }
                
                # Zeitliche Analyse der Herzfrequenz
                if 'timestamp' in health_data.columns:
                    health_data['hour'] = health_data['timestamp'].dt.hour
                    hourly_hr = health_data.groupby('hour')['heart_rate'].mean().to_dict()
                    heart_rate_analysis["hourly_average"] = {str(hour): float(avg) for hour, avg in hourly_hr.items()}
            
            # Schrittanalyse
            steps_analysis = {}
            if 'steps' in health_data.columns:
                # Gruppiere nach Datum für tägliche Schritte
                if 'timestamp' in health_data.columns:
                    health_data['date'] = health_data['timestamp'].dt.date
                    daily_steps = health_data.groupby('date')['steps'].sum()
                    
                    steps_analysis = {
                        "total": int(health_data['steps'].sum()),
                        "daily_average": float(daily_steps.mean()),
                        "max_daily": int(daily_steps.max()),
                        "min_daily": int(daily_steps.min()),
                        "days_above_10k": int((daily_steps >= 10000).sum())
                    }
            
            # Schlafanalyse
            sleep_analysis = {}
            if 'sleep_state' in health_data.columns:
                sleep_counts = health_data['sleep_state'].value_counts().to_dict()
                
                # Berechne Schlafqualität, falls möglich
                if 'sleep_quality' in health_data.columns:
                    sleep_analysis["average_quality"] = float(health_data['sleep_quality'].mean())
                
                sleep_analysis["state_distribution"] = sleep_counts
                
                # Berechne Schlafzeit pro Tag
                if 'timestamp' in health_data.columns:
                    health_data['date'] = health_data['timestamp'].dt.date
                    sleep_mask = health_data['sleep_state'] != 'awake'
                    if sleep_mask.any():
                        sleep_hours = health_data[sleep_mask].groupby('date').size() / 24.0  # Annahme: stündliche Messungen
                        sleep_analysis["average_sleep_hours"] = float(sleep_hours.mean())
            
            # Kalorienanalyse
            calories_analysis = {}
            if 'calories' in health_data.columns:
                if 'timestamp' in health_data.columns:
                    health_data['date'] = health_data['timestamp'].dt.date
                    daily_calories = health_data.groupby('date')['calories'].sum()
                    
                    calories_analysis = {
                        "total": int(health_data['calories'].sum()),
                        "daily_average": float(daily_calories.mean()),
                        "max_daily": int(daily_calories.max()),
                        "min_daily": int(daily_calories.min())
                    }
            
            # Anomalieerkennung
            anomalies = {}
            if 'heart_rate' in health_data.columns:
                anomalies["heart_rate"] = self._detect_anomalies(health_data, 'heart_rate')
            if 'steps' in health_data.columns and 'date' in health_data.columns:
                daily_steps_df = health_data.groupby('date')['steps'].sum().reset_index()
                anomalies["daily_steps"] = self._detect_anomalies(daily_steps_df, 'steps')
            
            # Kombiniere alle Analysen
            source_results = {
                "statistics": stats,
                "heart_rate": heart_rate_analysis,
                "steps": steps_analysis,
                "sleep": sleep_analysis,
                "calories": calories_analysis,
                "anomalies": anomalies
            }
            
            health_results[source] = source_results
            
            # Füge zur Liste der analysierten Quellen hinzu
            if source not in self.metadata["analyzed_sources"]:
                self.metadata["analyzed_sources"].append(source)
        
        # Speichere Ergebnisse
        self.analytics_results['health'] = health_results
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        logger.info(f"Gesundheitsdatenanalyse abgeschlossen für {len(health_results)} Quellen")
        return health_results
    
    def analyze_iot_data(self) -> Dict[str, Any]:
        """
        Analysiert IoT-Gerätedaten für wertvolle Erkenntnisse.
        
        Returns:
            Dict: Analyseergebnisse für IoT-Daten
        """
        iot_sources = [source for source in self.data_integrator.integrated_data.keys() 
                      if source.startswith('iot_')]
        
        if not iot_sources:
            logger.warning("Keine IoT-Daten für Analyse gefunden")
            return {}
        
        # Kombiniere alle IoT-Datenquellen
        iot_results = {}
        
        for source in iot_sources:
            iot_data = self.data_integrator.integrated_data[source]
            
            if not isinstance(iot_data, pd.DataFrame) or iot_data.empty:
                logger.warning(f"Ungültige IoT-Daten für {source}")
                continue
                
            logger.info(f"Analysiere IoT-Daten von {source}: {len(iot_data)} Datensätze")
            
            # Extrahiere Gerätetyp aus Quell-ID
            device_type = source.replace('iot_', '')
            
            # Grundlegende Statistiken
            stats = {
                "record_count": len(iot_data),
                "date_range": {
                    "start": iot_data['timestamp'].min().isoformat() if 'timestamp' in iot_data.columns else None,
                    "end": iot_data['timestamp'].max().isoformat() if 'timestamp' in iot_data.columns else None
                },
                "device_type": device_type,
                "field_count": len(iot_data.columns)
            }
            
            # Gerätespezifische Analysen
            device_analysis = {}
            
            # Thermostat-Analyse
            if device_type == 'thermostat':
                if 'temperature' in iot_data.columns:
                    device_analysis["temperature"] = {
                        "mean": float(iot_data['temperature'].mean()),
                        "min": float(iot_data['temperature'].min()),
                        "max": float(iot_data['temperature'].max()),
                        "std": float(iot_data['temperature'].std())
                    }
                
                if 'target_temperature' in iot_data.columns:
                    device_analysis["target_temperature"] = {
                        "mean": float(iot_data['target_temperature'].mean()),
                        "min": float(iot_data['target_temperature'].min()),
                        "max": float(iot_data['target_temperature'].max())
                    }
                    
                    # Berechne Differenz zwischen tatsächlicher und Zieltemperatur
                    if 'temperature' in iot_data.columns:
                        iot_data['temp_diff'] = iot_data['temperature'] - iot_data['target_temperature']
                        device_analysis["temperature_difference"] = {
                            "mean": float(iot_data['temp_diff'].mean()),
                            "abs_mean": float(iot_data['temp_diff'].abs().mean())
                        }
                
                if 'heating_status' in iot_data.columns:
                    status_counts = iot_data['heating_status'].value_counts().to_dict()
                    device_analysis["heating_status_distribution"] = status_counts
                    
                    # Berechne Heizungsnutzung pro Tag
                    if 'timestamp' in iot_data.columns:
                        iot_data['date'] = iot_data['timestamp'].dt.date
                        daily_heating = iot_data[iot_data['heating_status'] == 'on'].groupby('date').size()
                        if not daily_heating.empty:
                            device_analysis["daily_heating_hours"] = {
                                "mean": float(daily_heating.mean()),
                                "max": int(daily_heating.max())
                            }
            
            # Smart-Light-Analyse
            elif device_type == 'light':
                if 'status' in iot_data.columns:
                    status_counts = iot_data['status'].value_counts().to_dict()
                    device_analysis["status_distribution"] = status_counts
                    
                    # Berechne Nutzungsstunden pro Tag
                    if 'timestamp' in iot_data.columns:
                        iot_data['date'] = iot_data['timestamp'].dt.date
                        daily_usage = iot_data[iot_data['status'] == 'on'].groupby('date').size()
                        if not daily_usage.empty:
                            device_analysis["daily_usage_hours"] = {
                                "mean": float(daily_usage.mean()),
                                "max": int(daily_usage.max())
                            }
                
                if 'brightness' in iot_data.columns:
                    device_analysis["brightness"] = {
                        "mean": float(iot_data['brightness'].mean()),
                        "distribution": iot_data['brightness'].value_counts().to_dict()
                    }
            
            # Sicherheitskamera-Analyse
            elif device_type == 'security_camera':
                if 'motion_detected' in iot_data.columns:
                    motion_counts = iot_data['motion_detected'].value_counts().to_dict()
                    device_analysis["motion_detection"] = {
                        "total_detections": int(iot_data['motion_detected'].sum()),
                        "detection_rate": float(iot_data['motion_detected'].mean())
                    }
                    
                    # Zeitliche Analyse der Bewegungserkennung
                    if 'timestamp' in iot_data.columns:
                        iot_data['hour'] = iot_data['timestamp'].dt.hour
                        hourly_motion = iot_data[iot_data['motion_detected']].groupby('hour').size()
                        device_analysis["hourly_motion_distribution"] = hourly_motion.to_dict()
            
            # Generische Analyse für andere Gerätetypen
            else:
                if 'status' in iot_data.columns:
                    status_counts = iot_data['status'].value_counts().to_dict()
                    device_analysis["status_distribution"] = status_counts
                
                if 'power_consumption' in iot_data.columns:
                    device_analysis["power_consumption"] = {
                        "total": float(iot_data['power_consumption'].sum()),
                        "mean": float(iot_data['power_consumption'].mean()),
                        "max": float(iot_data['power_consumption'].max())
                    }
                    
                    # Täglicher Energieverbrauch
                    if 'timestamp' in iot_data.columns:
                        iot_data['date'] = iot_data['timestamp'].dt.date
                        daily_power = iot_data.groupby('date')['power_consumption'].sum()
                        device_analysis["daily_power_consumption"] = {
                            "mean": float(daily_power.mean()),
                            "max": float(daily_power.max()),
                            "min": float(daily_power.min())
                        }
            
            # Zeitliche Analyse
            temporal_analysis = {}
            if 'timestamp' in iot_data.columns:
                # Stunden des Tages
                iot_data['hour'] = iot_data['timestamp'].dt.hour
                hourly_activity = iot_data.groupby('hour').size().to_dict()
                temporal_analysis["hourly_activity"] = {str(hour): count for hour, count in hourly_activity.items()}
                
                # Tage der Woche
                iot_data['day_of_week'] = iot_data['timestamp'].dt.dayofweek
                daily_activity = iot_data.groupby('day_of_week').size().to_dict()
                temporal_analysis["daily_activity"] = {str(day): count for day, count in daily_activity.items()}
            
            # Anomalieerkennung
            anomalies = {}
            numeric_columns = iot_data.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                if col not in ['user_id', 'hour', 'day_of_week']:
                    anomalies[col] = self._detect_anomalies(iot_data, col)
            
            # Kombiniere alle Analysen
            source_results = {
                "statistics": stats,
                "device_analysis": device_analysis,
                "temporal_analysis": temporal_analysis,
                "anomalies": anomalies
            }
            
            iot_results[source] = source_results
            
            # Füge zur Liste der analysierten Quellen hinzu
            if source not in self.metadata["analyzed_sources"]:
                self.metadata["analyzed_sources"].append(source)
        
        # Speichere Ergebnisse
        self.analytics_results['iot'] = iot_results
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        logger.info(f"IoT-Datenanalyse abgeschlossen für {len(iot_results)} Quellen")
        return iot_results
    
    def run_all_analytics(self) -> Dict[str, Any]:
        """
        Führt alle verfügbaren Analysen durch.
        
        Returns:
            Dict: Alle Analyseergebnisse
        """
        start_time = datetime.now()
        logger.info("Starte umfassende Datenanalyse")
        
        # Führe alle Analysen durch
        browser_results = self.analyze_browser_data()
        health_results = self.analyze_health_data()
        iot_results = self.analyze_iot_data()
        
        # Aktualisiere Metadaten
        self.metadata.update({
            "last_updated": datetime.now().isoformat(),
            "analysis_duration_seconds": (datetime.now() - start_time).total_seconds(),
            "analyzed_categories": {
                "browser": len(browser_results) > 0,
                "health": len(health_results) > 0,
                "iot": len(iot_results) > 0
            },
            "total_analyzed_sources": len(self.metadata["analyzed_sources"])
        })
        
        logger.info(f"Datenanalyse abgeschlossen: {len(self.metadata['analyzed_sources'])} Quellen analysiert")
        return {
            "results": self.analytics_results,
            "metadata": self.metadata
        }
    
    def _detect_anomalies(self, data: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Erkennt Anomalien in einer Datenspalte.
        
        Args:
            data: DataFrame mit den Daten
            column: Name der zu analysierenden Spalte
            
        Returns:
            Dict: Erkannte Anomalien und Statistiken
        """
        if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
            return {"count": 0, "percentage": 0.0, "details": []}
        
        try:
            # Einfache statistische Anomalieerkennung
            # Wir betrachten Werte außerhalb von 3 Standardabweichungen als Anomalien
            mean = data[column].mean()
            std = data[column].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            anomalies = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            anomaly_count = len(anomalies)
            anomaly_percentage = (anomaly_count / len(data)) * 100 if len(data) > 0 else 0
            
            # Erstelle Details für die ersten 5 Anomalien
            details = []
            for _, row in anomalies.head(5).iterrows():
                detail = {
                    "value": float(row[column]),
                    "expected_range": [float(lower_bound), float(upper_bound)],
                    "deviation": float(abs(row[column] - mean) / std) if std > 0 else 0
                }
                
                # Füge Zeitstempel hinzu, falls vorhanden
                if 'timestamp' in row:
                    detail["timestamp"] = row['timestamp'].isoformat()
                
                details.append(detail)
            
            return {
                "count": anomaly_count,
                "percentage": round(anomaly_percentage, 2),
                "threshold": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound)
                },
                "details": details
            }
        except Exception as e:
            logger.error(f"Fehler bei der Anomalieerkennung für {column}: {str(e)}")
            return {"count": 0, "percentage": 0.0, "details": [], "error": str(e)}

# Beispiel für die Verwendung des Algorithmus
def demo_integration():
    """
    Demonstriert die Datenintegration und -analyse.
    
    Returns:
        Dict: Ergebnisse der Integration und Analyse
    """
    user_id = "user123"
    
    # Erstelle einen Datenintegrator
    integrator = DataIntegrator(user_id)
    
    # Füge verschiedene Datenquellen hinzu
    browser_connector = BrowserDataConnector(user_id, 'chrome')
    smartwatch_connector = SmartwatchDataConnector(user_id, 'fitbit')
    thermostat_connector = IoTDeviceConnector(user_id, 'thermostat')
    
    integrator.add_connector(browser_connector)
    integrator.add_connector(smartwatch_connector)
    integrator.add_connector(thermostat_connector)
    
    # Sammle alle Daten mit mittlerem Datenschutzniveau
    integrated_data = integrator.collect_all_data(PrivacyLevel.MEDIUM)
    
    # Bereite Daten für Tokenisierung vor
    tokenization_package = integrator.prepare_for_tokenization(PrivacyLevel.MEDIUM)
    
    # Führe Analysen durch, um den Datenwert zu steigern
    analyzer = DataAnalyzer(integrator)
    analytics_results = analyzer.run_all_analytics()
    
    # Kombiniere Daten für erweiterte Analysen
    combined_data = integrator.combine_data_sources('correlate')
    
    # Erstelle eine Zusammenfassung der Ergebnisse
    summary = {
        'user_id': user_id,
        'data_sources': list(integrator.integrated_data.keys()),
        'total_records': integrator.metadata.get('total_records', 0),
        'estimated_value': integrator.metadata.get('estimated_total_value', 0.0),
        'privacy_level': integrator.metadata.get('privacy_level', 'unknown'),
        'analyzed_sources': analyzer.metadata.get('analyzed_sources', []),
        'combined_data_size': len(combined_data) if isinstance(combined_data, pd.DataFrame) else 0
    }
    
    return {
        'integrated_data': integrated_data,
        'tokenization_package': tokenization_package,
        'analytics_results': analytics_results,
        'combined_data': combined_data,
        'summary': summary
    }

def run_demo():
    """
    Führt die Demo aus und gibt die Ergebnisse aus.
    """
    print("OceanData - Datenintegration und -monetarisierung Demo")
    print("=====================================================")
    
    # Führe die Demo aus
    start_time = datetime.now()
    demo_results = demo_integration()
    duration = (datetime.now() - start_time).total_seconds()
    
    # Zeige Zusammenfassung
    summary = demo_results['summary']
    print("\n📊 ZUSAMMENFASSUNG")
    print(f"Benutzer-ID: {summary['user_id']}")
    print(f"Datenquellen: {', '.join(summary['data_sources'])}")
    print(f"Gesamtdatensätze: {summary['total_records']}")
    print(f"Geschätzter Wert: {summary['estimated_value']:.2f} OCEAN")
    print(f"Datenschutzniveau: {summary['privacy_level']}")
    print(f"Analysierte Quellen: {len(summary['analyzed_sources'])}")
    print(f"Kombinierte Datensatzgröße: {summary['combined_data_size']}")
    print(f"Verarbeitungszeit: {duration:.2f} Sekunden")
    
    # Zeige Tokenisierungspaket
    token_pkg = demo_results['tokenization_package']
    print("\n💰 TOKENISIERUNGSPAKET")
    print(f"Zeitstempel: {token_pkg['timestamp']}")
    print(f"Geschätzter Wert: {token_pkg['estimated_value']:.2f} OCEAN")
    print(f"Datenschutzniveau: {token_pkg['privacy_level']}")
    
    # Zeige Metadaten
    metadata = token_pkg['metadata']['basic']
    print("\n📋 METADATEN")
    print(f"Name: {metadata['name']}")
    print(f"Beschreibung: {metadata['description']}")
    print(f"Autor: {metadata['author']}")
    print(f"Erstellt: {metadata['created']}")
    print(f"Quellen: {', '.join(metadata['sources'])}")
    print(f"Gesamtdatensätze: {metadata['total_records']}")
    
    # Zeige Analyseergebnisse
    analytics = demo_results['analytics_results']['results']
    print("\n🔍 ANALYSEN")
    
    if 'browser' in analytics:
        browser_sources = list(analytics['browser'].keys())
        if browser_sources:
            browser_source = browser_sources[0]
            browser_data = analytics['browser'][browser_source]
            print("\n🌐 Browser-Daten:")
            print(f"  Datensätze: {browser_data['statistics']['record_count']}")
            
            if 'website_analysis' in browser_data and 'top_domains' in browser_data['website_analysis']:
                print("  Top-Domains:")
                for domain, count in list(browser_data['website_analysis']['top_domains'].items())[:3]:
                    print(f"    - {domain}: {count}")
            
            if 'usage_analysis' in browser_data and 'avg_duration' in browser_data['usage_analysis']:
                print(f"  Durchschnittliche Nutzungsdauer: {browser_data['usage_analysis']['avg_duration']:.1f} Sekunden")
    
    if 'health' in analytics:
        health_sources = list(analytics['health'].keys())
        if health_sources:
            health_source = health_sources[0]
            health_data = analytics['health'][health_source]
            print("\n❤️ Gesundheitsdaten:")
            print(f"  Datensätze: {health_data['statistics']['record_count']}")
            
            if 'heart_rate' in health_data and health_data['heart_rate']:
                hr = health_data['heart_rate']
                print(f"  Herzfrequenz: Ø {hr['mean']:.1f} (Min: {hr['min']}, Max: {hr['max']})")
            
            if 'steps' in health_data and health_data['steps']:
                steps = health_data['steps']
                print(f"  Schritte: Gesamt {steps['total']}, Ø {steps['daily_average']:.1f} pro Tag")
    
    if 'iot' in analytics:
        iot_sources = list(analytics['iot'].keys())
        if iot_sources:
            iot_source = iot_sources[0]
            iot_data = analytics['iot'][iot_source]
            print(f"\n🏠 IoT-Daten ({iot_data['statistics']['device_type']}):")
            print(f"  Datensätze: {iot_data['statistics']['record_count']}")
            
            if 'device_analysis' in iot_data:
                device = iot_data['device_analysis']
                if 'temperature' in device:
                    print(f"  Temperatur: Ø {device['temperature']['mean']:.1f}°C (Min: {device['temperature']['min']}°C, Max: {device['temperature']['max']}°C)")
    
    print("\n✅ Demo abgeschlossen!")

if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Führe die Demo aus
    run_demo()
