"""
OceanData - Umfassender Datenerfassungs- und Analysealgorithmus

Dieser fortschrittliche Algorithmus bildet das Herzstück der OceanData-Plattform und ermöglicht:
1. Umfassende Datenerfassung aus diversen Quellen (Browser, Chat, Social Media, IoT, Gesundheit, etc.)
2. Datentransformation und -harmonisierung für die einheitliche Verarbeitung
3. Datenprivacy durch fortschrittliche Anonymisierungstechniken
4. Wertschöpfung durch KI-basierte Datenanalyse
5. Vorbereitung der Daten für die Tokenisierung über Ocean Protocol
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import uuid
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import tensorflow as tf
from cryptography.fernet import Fernet

# Konfiguration des Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("OceanData")

# Erzeugung eines Verschlüsselungsschlüssels für sensible Daten
def generate_encryption_key():
    """Generiert einen symmetrischen Schlüssel für die Verschlüsselung sensibler Daten."""
    return Fernet.generate_key()

ENCRYPTION_KEY = generate_encryption_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

class DataCategory(Enum):
    """Enumeration für verschiedene Datenkategorien"""
    BROWSER = "browser"
    CALENDAR = "calendar"
    CHAT = "chat"
    SOCIAL_MEDIA = "social_media"
    STREAMING = "streaming"
    HEALTH_INSURANCE = "health_insurance"
    HEALTH_DATA = "health_data"
    SMARTWATCH = "smartwatch"
    SPORTS = "sports"
    IOT_VACUUM = "iot_vacuum"
    IOT_THERMOSTAT = "iot_thermostat"
    IOT_LIGHTING = "iot_lighting"
    IOT_SECURITY = "iot_security"
    SMART_HOME = "smart_home"
    IOT_GENERAL = "iot_general"

class PrivacyLevel(Enum):
    """Datenschutzstufen für verschiedene Datentypen"""
    PUBLIC = "public"         # Daten können im Klartext geteilt werden
    ANONYMIZED = "anonymized" # Daten werden anonymisiert geteilt
    ENCRYPTED = "encrypted"   # Daten werden verschlüsselt und für C2D verwendet
    SENSITIVE = "sensitive"   # Höchst sensible Daten (medizinisch), erfordern besondere Maßnahmen

class DataSource(ABC):
    """Abstrakte Basisklasse für alle Datenquellen"""
    
    def __init__(self, source_id: str, user_id: str, category: DataCategory):
        """
        Initialisiert eine Datenquelle.
        
        Args:
            source_id: Eindeutige ID für die Datenquelle
            user_id: ID des Benutzers, dem die Daten gehören
            category: Kategorie der Datenquelle
        """
        self.source_id = source_id
        self.user_id = user_id
        self.category = category
        self.last_sync = None
        self.data = None
        self.metadata = {
            "source_id": source_id,
            "user_id": user_id,
            "category": category.value,
            "created": datetime.now().isoformat(),
            "privacy_fields": {},  # Wird mit Feldern und deren Datenschutzstufen gefüllt
        }
        self.logger = logging.getLogger(f"OceanData.{category.value}.{source_id}")
    
    def set_privacy_level(self, field: str, level: PrivacyLevel):
        """Setzt die Datenschutzstufe für ein bestimmtes Feld"""
        self.metadata["privacy_fields"][field] = level.value
    
    def get_privacy_level(self, field: str) -> PrivacyLevel:
        """Gibt die Datenschutzstufe für ein bestimmtes Feld zurück"""
        level_str = self.metadata["privacy_fields"].get(field, PrivacyLevel.ANONYMIZED.value)
        return PrivacyLevel(level_str)
    
    @abstractmethod
    def connect(self) -> bool:
        """Verbindung zur Datenquelle herstellen"""
        pass
    
    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """Daten von der Quelle abrufen"""
        pass
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Verarbeitet die Daten basierend auf den festgelegten Datenschutzstufen:
        - PUBLIC: keine Verarbeitung
        - ANONYMIZED: Daten werden anonymisiert
        - ENCRYPTED: Daten werden verschlüsselt
        - SENSITIVE: Daten werden stark verschlüsselt und mit zusätzlichen Schutzmaßnahmen versehen
        """
        if data is None or data.empty:
            return pd.DataFrame()
            
        processed_data = data.copy()
        
        # Durchlaufe alle Spalten und wende entsprechende Verarbeitung an
        for column in processed_data.columns:
            privacy_level = self.get_privacy_level(column)
            
            if privacy_level == PrivacyLevel.PUBLIC:
                # Keine Verarbeitung notwendig
                continue
                
            elif privacy_level == PrivacyLevel.ANONYMIZED:
                # Anonymisiere die Daten basierend auf dem Datentyp
                if processed_data[column].dtype == 'object':
                    # Text oder kategoriale Daten
                    if 'id' in column.lower() or 'user' in column.lower() or 'name' in column.lower():
                        salt = datetime.now().strftime("%Y%m%d")
                        processed_data[column] = processed_data[column].apply(
                            lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest() if pd.notna(x) else x
                        )
                    elif 'location' in column.lower() or 'address' in column.lower() or 'ip' in column.lower():
                        # Standortdaten verschleiern
                        processed_data[column] = processed_data[column].apply(
                            lambda x: f"anonymized_{hashlib.md5(str(x).encode()).hexdigest()[:8]}" if pd.notna(x) else x
                        )
                
            elif privacy_level == PrivacyLevel.ENCRYPTED:
                # Verschlüssele die Daten
                processed_data[column] = processed_data[column].apply(
                    lambda x: cipher_suite.encrypt(str(x).encode()).decode() if pd.notna(x) else x
                )
                
            elif privacy_level == PrivacyLevel.SENSITIVE:
                # Für hochsensible Daten: doppelte Verschlüsselung und Markierung
                processed_data[column] = processed_data[column].apply(
                    lambda x: "SENSITIVE_DATA_PROTECTED" if pd.notna(x) else x
                )
                # In echten Implementierungen würden wir hier besondere Schutzmechanismen anwenden
        
        return processed_data
    
    def extract_metadata(self, data: pd.DataFrame) -> Dict:
        """Extrahiert Metadaten aus den Daten für den Datensatz"""
        if data is None or data.empty:
            return {}
            
        metadata = {
            "rows": len(data),
            "columns": list(data.columns),
            "data_types": {col: str(data[col].dtype) for col in data.columns},
            "last_updated": datetime.now().isoformat(),
            "data_summary": {}
        }
        
        # Erstelle eine Zusammenfassung der Daten (mit Datenschutz im Hinterkopf)
        for col in data.columns:
            privacy_level = self.get_privacy_level(col)
            if privacy_level == PrivacyLevel.PUBLIC or privacy_level == PrivacyLevel.ANONYMIZED:
                if data[col].dtype == 'object':
                    metadata["data_summary"][col] = {
                        "unique_values": data[col].nunique(),
                        "sample_values": [] if privacy_level == PrivacyLevel.ANONYMIZED else data[col].dropna().sample(min(3, data[col].nunique())).tolist()
                    }
                else:
                    metadata["data_summary"][col] = {
                        "min": data[col].min() if not pd.api.types.is_numeric_dtype(data[col]) else float(data[col].min()),
                        "max": data[col].max() if not pd.api.types.is_numeric_dtype(data[col]) else float(data[col].max()),
                        "mean": float(data[col].mean()) if pd.api.types.is_numeric_dtype(data[col]) else None
                    }
        
        return metadata
    
    def get_data(self) -> Dict[str, Any]:
        """
        Führt den vollständigen Datenabruf- und Verarbeitungsprozess durch.
        
        Returns:
            Dictionary mit verarbeiteten Daten und zugehörigen Metadaten
        """
        try:
            self.logger.info(f"Starte Datenabruf für Quelle {self.source_id}")
            
            if not self.connect():
                self.logger.error(f"Verbindung zu Quelle {self.source_id} fehlgeschlagen")
                return {"data": None, "metadata": self.metadata, "status": "error"}
            
            raw_data = self.fetch_data()
            if raw_data is None or raw_data.empty:
                self.logger.warning(f"Keine Daten von Quelle {self.source_id} abgerufen")
                return {"data": None, "metadata": self.metadata, "status": "empty"}
            
            processed_data = self.process_data(raw_data)
            metadata = self.extract_metadata(raw_data)  # Metadaten aus Rohdaten extrahieren
            
            self.last_sync = datetime.now()
            self.data = processed_data
            self.metadata.update(metadata)
            self.metadata["last_sync"] = self.last_sync.isoformat()
            
            self.logger.info(f"Datenabruf für Quelle {self.source_id} erfolgreich: {len(processed_data)} Datensätze")
            
            return {
                "data": processed_data,
                "metadata": self.metadata,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Datenabruf für Quelle {self.source_id}: {str(e)}")
            return {"data": None, "metadata": self.metadata, "status": "error", "error": str(e)}

# Implementierung verschiedener Datenkonnektoren

class BrowserDataConnector(DataSource):
    """
    Konnektor für Browserdaten, der folgende Informationen erfasst:
    - Besuchte Websites und Zeitstempel
    - Suchverlauf
    - Download-Historie
    - Lesezeichen
    - Browsing-Dauer pro Website
    - Gerätetyp und Betriebssystem
    - IP-Adresse und geografischer Standort
    - Verwendete Browser-Erweiterungen
    """
    
    def __init__(self, user_id: str, browser_type: str = 'chrome'):
        super().__init__(f"browser_{browser_type}", user_id, DataCategory.BROWSER)
        self.browser_type = browser_type
        
        # Setze Datenschutzstufen für verschiedene Felder
        self.set_privacy_level("url", PrivacyLevel.ANONYMIZED)
        self.set_privacy_level("search_term", PrivacyLevel.ANONYMIZED)
        self.set_privacy_level("download_path", PrivacyLevel.ENCRYPTED)
        self.set_privacy_level("ip_address", PrivacyLevel.ENCRYPTED)
        self.set_privacy_level("user_id", PrivacyLevel.ENCRYPTED)
        self.set_privacy_level("extension_id", PrivacyLevel.ANONYMIZED)
        
    def connect(self) -> bool:
        """Verbindung zum Browser-Verlauf herstellen"""
        self.logger.info(f"Verbindung zu {self.browser_type}-Daten für Benutzer {self.user_id} hergestellt")
        # In realer Implementierung: Verbindung zur Browser-History-API oder Dateien
        return True
    
    def fetch_data(self) -> pd.DataFrame:
        """Browser-Verlaufsdaten abrufen"""
        # Simuliere Browserdaten mit verschiedenen Datentypen
        websites = ['example.com', 'github.com', 'docs.python.org', 'mail.google.com', 'youtube.com']
        search_terms = ['python tutorial', 'javascript frameworks', 'data privacy', 'machine learning']
        browser_extensions = ['adblock_plus', 'grammarly', 'lastpass', 'dark_reader']
        ips = ['192.168.1.1', '10.0.0.15', '172.16.254.1', '127.0.0.1']
        locations = ['New York', 'Berlin', 'Tokyo', 'London', 'Paris']
        
        # Erzeuge 30-50 zufällige Einträge
        entries = np.random.randint(30, 50)
        
        now = datetime.now()
        data = {
            "user_id": [self.user_id] * entries,
            "browser": [self.browser_type] * entries,
            "timestamp": [now - timedelta(minutes=np.random.randint(1, 60*24*7)) for _ in range(entries)],
            "url": [np.random.choice(websites) for _ in range(entries)],
            "duration": [np.random.randint(10, 1800) for _ in range(entries)],  # 10 Sek. bis 30 Min.
            "search_term": [np.random.choice(search_terms) if np.random.random() > 0.7 else None for _ in range(entries)],
            "is_download": [np.random.random() > 0.9 for _ in range(entries)],
            "download_path": [f"/downloads/file_{i}.pdf" if np.random.random() > 0.9 else None for i in range(entries)],
            "is_bookmark": [np.random.random() > 0.8 for _ in range(entries)],
            "device_type": [np.random.choice(["desktop", "mobile", "tablet"]) for _ in range(entries)],
            "os": [np.random.choice(["Windows", "MacOS", "Linux", "iOS", "Android"]) for _ in range(entries)],
            "ip_address": [np.random.choice(ips) for _ in range(entries)],
            "location": [np.random.choice(locations) for _ in range(entries)],
            "extension_id": [np.random.choice(browser_extensions) if np.random.random() > 0.6 else None for _ in range(entries)]
        }
        
        return pd.DataFrame(data)
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extrahiere erweiterte Features aus Browser-Daten"""
        if data is None or data.empty:
            return pd.DataFrame()
            
        features = pd.DataFrame()
        
        # User-ID beibehalten
        features['user_id'] = data['user_id'].iloc[0]
        
        # Browsing-Gewohnheiten
        if 'timestamp' in data.columns:
            data['hour'] = data['timestamp'].apply(lambda x: x.hour)
            data['day_of_week'] = data['timestamp'].apply(lambda x: x.dayofweek)
            
            # Aktivitätsmuster nach Tageszeit
            activity_by_hour = data.groupby('hour').size()
            features['peak_activity_hour'] = activity_by_hour.idxmax()
            features['night_browsing_ratio'] = data[data['hour'].between(0, 5)].shape[0] / data.shape[0]
            
            # Aktivitätsmuster nach Wochentag
            activity_by_dow = data.groupby('day_of_week').size()
            features['peak_activity_day'] = activity_by_dow.idxmax()
            features['weekend_browsing_ratio'] = data[data['day_of_week'].isin([5, 6])].shape[0] / data.shape[0]
        
        # Browsing-Präferenzen
        if 'url' in data.columns:
            # Top-Domains
            data['domain'] = data['url'].apply(lambda url: url.split('/')[0])
            domain_counts = data.groupby('domain').size()
            features['top_domain'] = domain_counts.idxmax()
            features['domain_diversity'] = len(domain_counts)
        
        # Suchpräferenzen
        if 'search_term' in data.columns:
            search_data = data.dropna(subset=['search_term'])
            if not search_data.empty:
                # Einfaches Topic Modeling: Wortzählung
                all_terms = ' '.join(search_data['search_term'])
                common_words = pd.Series(all_terms.lower().split()).value_counts().head(5).index.tolist()
                features['search_topics'] = json.dumps(common_words)
        
        # Nutzung von Browser-Funktionen
        if 'is_bookmark' in data.columns:
            features['bookmark_ratio'] = data['is_bookmark'].mean()
        
        if 'is_download' in data.columns:
            features['download_ratio'] = data['is_download'].mean()
        
        # Gerätenutzung
        if 'device_type' in data.columns:
            device_counts = data['device_type'].value_counts(normalize=True)
            for device in device_counts.index:
                features[f'device_{device}_ratio'] = device_counts.get(device, 0)
        
        return pd.DataFrame([features])

class CalendarDataConnector(DataSource):
    """
    Konnektor für Kalenderdaten, der folgende Informationen erfasst:
    - Termine und Ereignisse mit Datum, Uhrzeit und Dauer
    - Teilnehmer an Ereignissen
    - Wiederholende Termine
    - Erinnerungen und Benachrichtigungen
    - Standorte von Ereignissen
    - Kategorien oder Tags für Termine
    - Synchronisationsstatus mit anderen Geräten
    - Änderungshistorie von Terminen
    """
    
    def __init__(self, user_id: str, calendar_type: str = 'google'):
        super().__init__(f"calendar_{calendar_type}", user_id, DataCategory.CALENDAR)
        self.calendar_type = calendar_type
        
        # Setze Datenschutzstufen für verschiedene Felder
        self.set_privacy_level("event_title", PrivacyLevel.ANONYMIZED)
        self.set_privacy_level("event_description", PrivacyLevel.ENCRYPTED)
        self.set_privacy_level("location", PrivacyLevel.ANONYMIZED)
        self.set_privacy_level("attendees", PrivacyLevel.ENCRYPTED)
        self.set_privacy_level("user_id", PrivacyLevel.ENCRYPTED)
        
    def connect(self) -> bool:
        """Verbindung zum Kalender herstellen"""
        self.logger.info(f"Verbindung zu {self.calendar_type}-Kalenderdaten für Benutzer {self.user_id} hergestellt")
        # In realer Implementierung: Verbindung zur Kalender-API
        return True
    
    def fetch_data(self) -> pd.DataFrame:
        """Kalenderdaten abrufen"""
        # Simuliere Kalenderdaten mit verschiedenen Datentypen
        event_titles = ['Team Meeting', 'Project Deadline', 'Doctor Appointment', 'Lunch with Client', 'Conference Call']
        event_descriptions = ['Weekly team sync', 'Submit final project report', 'Annual check-up', 'Business lunch with potential client', 'Call with overseas team']
        locations = ['Conference Room A', 'Office', 'Medical Center', 'Restaurant Downtown', 'Home Office']
        categories = ['Work', 'Personal', 'Health', 'Social', 'Family']
        
        # Erzeuge 15-25 zufällige Einträge
        entries = np.random.randint(15, 25)
        
        now = datetime.now()
        start_dates = [now + timedelta(days=np.random.randint(-30, 30)) for _ in range(entries)]
        
        data = {
            "user_id": [self.user_id] * entries,
            "calendar_type": [self.calendar_type] * entries,
            "event_id": [f"evt_{uuid.uuid4()}" for _ in range(entries)],
            "event_title": [np.random.choice(event_titles) for _ in range(entries)],
            "event_description": [np.random.choice(event_descriptions) for _ in range(entries)],
            "start_time": start_dates,
            "end_time": [start_date + timedelta(minutes=np.random.choice([30, 60, 90, 120])) for start_date in start_dates],
            "is_recurring": [np.random.random() > 0.7 for _ in range(entries)],
            "recurrence_pattern": [np.random.choice(["daily", "weekly", "monthly", "yearly"]) if np.random.random() > 0.7 else None for _ in range(entries)],
            "has_reminder": [np.random.random() > 0.5 for _ in range(entries)],
            "reminder_time": [np.random.choice([5, 10, 15, 30, 60]) if np.random.random() > 0.5 else None for _ in range(entries)],
            "location": [np.random.choice(locations) for _ in range(entries)],
            "attendees": [[f"person_{i}@example.com" for i in range(np.random.randint(0, 5))] for _ in range(entries)],
            "category": [np.random.choice(categories) for _ in range(entries)],
            "is_synced": [np.random.random() > 0.2 for _ in range(entries)],
            "last_modified": [now - timedelta(days=np.random.randint(0, 10)) for _ in range(entries)]
        }
        
        return pd.DataFrame(data)
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extrahiere erweiterte Features aus Kalenderdaten"""
        if data is None or data.empty:
            return pd.DataFrame()
            
        features = pd.DataFrame()
        
        # User-ID beibehalten
        features['user_id'] = data['user_id'].iloc[0]
        
        # Analysiere Terminplanung
        if 'start_time' in data.columns:
            # Termine nach Tageszeit
            data['hour'] = data['start_time'].apply(lambda x: x.hour)
            data['day_of_week'] = data['start_time'].apply(lambda x: x.dayofweek)
            
            # Bevorzugte Besprechungszeiten
            meeting_hours = data.groupby('hour').size()
            features['preferred_meeting_hour'] = meeting_hours.idxmax() if not meeting_hours.empty else None
            features['early_meetings_ratio'] = data[data['hour'] < 10].shape[0] / data.shape[0] if data.shape[0] > 0 else 0
            
            # Arbeitsgewohnheiten
            features['weekend_meetings_ratio'] = data[data['day_of_week'].isin([5, 6])].shape[0] / data.shape[0] if data.shape[0] > 0 else 0
        
        # Termintypen und -kategorien
        if 'category' in data.columns:
            category_counts = data['category'].value_counts(normalize=True)
            for category in category_counts.index:
                features[f'category_{category.lower()}_ratio'] = category_counts.get(category, 0)
        
        # Analyse der Termindauer
        if 'start_time' in data.columns and 'end_time' in data.columns:
            data['duration_minutes'] = data.apply(lambda row: (row['end_time'] - row['start_time']).total_seconds() / 60, axis=1)
            features['avg_meeting_duration'] = data['duration_minutes'].mean()
            features['short_meetings_ratio'] = data[data['duration_minutes'] <= 30].shape[0] / data.shape[0] if data.shape[0] > 0 else 0
            features['long_meetings_ratio'] = data[data['duration_minutes'] >= 90].shape[0] / data.shape[0] if data.shape[0] > 0 else 0
        
        # Analyse der Terminwiederholung
        if 'is_recurring' in data.columns:
            features['recurring_events_ratio'] = data['is_recurring'].mean()
        
        # Analyse der Teilnehmer
        if 'attendees' in data.columns:
            data['attendee_count'] = data['attendees'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            features['avg_attendees_per_meeting'] = data['attendee_count'].mean()
            features['solo_meetings_ratio'] = data[data['attendee_count'] == 0].shape[0] / data.shape[0] if data.shape[0] > 0 else 0
            features['large_meetings_ratio'] = data[data['attendee_count'] > 3].shape[0] / data.shape[0] if data.shape[0] > 0 else 0
        
        return pd.DataFrame([features])

class SmartDeviceDataConnector(DataSource):
    """
    Konnektor für Smart Device Daten (Smartwatches, IoT-Geräte), der folgende Informationen erfassen kann:
    - Gesundheits- und Fitness-Daten (Schritte, Herzfrequenz, etc.)
    - Standortdaten und Bewegungsmuster
    - Sensordaten von Smart Home Geräten
    - Energieverbrauchsdaten
    - Gerätestatus und Nutzungsmuster
    """
    
    def __init__(self, user_id: str, device_type: str, device_id: str = None):
        self.device_type = device_type.lower()
        self.device_id = device_id or f"{device_type}_{uuid.uuid4().hex[:8]}"
        
        # Bestimme die richtige Kategorie basierend auf dem Gerätetyp
        if self.device_type == 'smartwatch':
            category = DataCategory.SMARTWATCH
        elif self.device_type in ['thermostat', 'temperature_sensor']:
            category = DataCategory.IOT_THERMOSTAT
        elif self.device_type in ['vacuum', 'robot_vacuum']:
            category = DataCategory.IOT_VACUUM
        elif self.device_type in ['smart_light', 'lighting']:
            category = DataCategory.IOT_LIGHTING
        elif self.device_type in ['security_camera', 'doorbell', 'motion_sensor']:
            category = DataCategory.IOT_SECURITY
        elif self.device_type in ['smart_speaker', 'hub', 'smart_tv']:
            category = DataCategory.SMART_HOME
        else:
            category = DataCategory.IOT_GENERAL
            
        super().__init__(f"{category.value}_{self.device_id}", user_id, category)
        
        # Setze Datenschutzstufen für verschiedene Felder basierend auf der Gerätekategorie
        if category == DataCategory.SMARTWATCH:
            self.set_privacy_level("heart_rate", PrivacyLevel.ANONYMIZED)
            self.set_privacy_level("steps", PrivacyLevel.PUBLIC)
            self.set_privacy_level("sleep_data", PrivacyLevel.ANONYMIZED)
            self.set_privacy_level("location", PrivacyLevel.ENCRYPTED)
        elif category in [DataCategory.IOT_VACUUM, DataCategory.IOT_THERMOSTAT, DataCategory.IOT_LIGHTING]:
            self.set_privacy_level("room_layout", PrivacyLevel.ENCRYPTED)
            self.set_privacy_level("usage_pattern", PrivacyLevel.ANONYMIZED)
            self.set_privacy_level("energy_consumption", PrivacyLevel.PUBLIC)
        elif category == DataCategory.IOT_SECURITY:
            self.set_privacy_level("video_data", PrivacyLevel.SENSITIVE)
            self.set_privacy_level("motion_events", PrivacyLevel.ENCRYPTED)
            self.set_privacy_level("person_detected", PrivacyLevel.ENCRYPTED)
        
        # Gemeinsame Datenschutzstufen für alle Geräte
        self.set_privacy_level("device_id", PrivacyLevel.ANONYMIZED)
        self.set_privacy_level("user_id", PrivacyLevel.ENCRYPTED)
        self.set_privacy_level("ip_address", PrivacyLevel.ENCRYPTED)
    
    def connect(self) -> bool:
        """Verbindung zum Smart Device herstellen"""
        self.logger.info(f"Verbindung zu {self.device_type} (ID: {self.device_id}) für Benutzer {self
