"""
OceanData - Browser-Datenkonnektor

Dieser Konnektor ist für die Erfassung von Browser-Daten verantwortlich, einschließlich:
- Besuchte Websites und Zeitstempel
- Suchverlauf
- Download-Historie
- Lesezeichen
- Browsing-Dauer pro Website
- Gerätetyp und Betriebssystem
- IP-Adresse und geografischer Standort
- Verwendete Browser-Erweiterungen
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import platform
import socket
import random
import json
import logging
from typing import Dict, List, Optional, Union, Any

# Importiere die Basisklassen
from oceandata.data_integration.base import DataSource, DataCategory, PrivacyLevel

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
    
    def __init__(self, user_id: str, browser_type: str = 'chrome', encryption_key=None):
        """
        Initialisiert den Browser-Datenkonnektor.
        
        Args:
            user_id: ID des Benutzers, dem die Daten gehören
            browser_type: Art des Browsers (z.B. 'chrome', 'firefox', 'safari')
            encryption_key: Optional. Schlüssel für die Verschlüsselung sensibler Daten.
        """
        super().__init__(f"browser_{browser_type}", user_id, DataCategory.BROWSER, encryption_key)
        self.browser_type = browser_type
        
        # Browser-API-Verbindungsdetails (würden in einer realen Implementierung eingestellt)
        self.api_settings = {
            'chrome': {
                'history_path': f"~/.config/google-chrome/Default/History",
                'bookmarks_path': f"~/.config/google-chrome/Default/Bookmarks",
                'extensions_path': f"~/.config/google-chrome/Default/Extensions"
            },
            'firefox': {
                'history_path': f"~/.mozilla/firefox/*.default/places.sqlite",
                'bookmarks_path': f"~/.mozilla/firefox/*.default/bookmarkbackups/",
                'extensions_path': f"~/.mozilla/firefox/*.default/extensions/"
            },
            'safari': {
                'history_path': f"~/Library/Safari/History.db",
                'bookmarks_path': f"~/Library/Safari/Bookmarks.plist",
                'extensions_path': f"~/Library/Safari/Extensions/"
            }
        }
        
        # Setze Datenschutzstufen für verschiedene Felder
        self.set_privacy_level("url", PrivacyLevel.ANONYMIZED)
        self.set_privacy_level("search_term", PrivacyLevel.ANONYMIZED)
        self.set_privacy_level("download_path", PrivacyLevel.ENCRYPTED)
        self.set_privacy_level("ip_address", PrivacyLevel.ENCRYPTED)
        self.set_privacy_level("user_id", PrivacyLevel.ENCRYPTED)
        self.set_privacy_level("extension_id", PrivacyLevel.ANONYMIZED)
        
    def connect(self) -> bool:
        """
        Verbindung zum Browser-Verlauf herstellen
        
        In einer realen Implementierung würde diese Methode:
        1. Prüfen, ob der Browser installiert ist
        2. Die entsprechenden Verlaufsdateien oder APIs lokalisieren
        3. Die erforderlichen Berechtigungen prüfen
        
        Returns:
            bool: True wenn die Verbindung erfolgreich ist, sonst False
        """
        try:
            self.logger.info(f"Verbindung zu {self.browser_type}-Daten für Benutzer {self.user_id} hergestellt")
            
            # TODO: In einer realen Implementierung würden wir hier tatsächlich:
            # 1. Den Browsertyp erkennen
            # 2. Die entsprechenden Dateipfade für die Browserdaten lokalisieren
            # 3. Die Zugriffsberechtigungen überprüfen
            # 4. Eine Verbindung zur Browser-History-API oder -Datenbank herstellen
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Verbinden mit {self.browser_type}: {str(e)}")
            return False
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Browser-Verlaufsdaten abrufen
        
        In einer realen Implementierung würde diese Methode:
        1. Die Browserdatenbank oder API abfragen
        2. Die Browserdaten in ein einheitliches Format konvertieren
        
        Returns:
            pd.DataFrame: Die abgerufenen Browserdaten
        """
        # TODO: Ersetze diese simulierten Daten mit echten Browserabfragen
        # Im Moment simulieren wir Browserdaten zu Demonstrationszwecken
        
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
        """
        Extrahiere erweiterte Features aus Browser-Daten
        
        Diese Methode analysiert die Rohdaten und erstellt höherwertige Merkmalsdaten,
        die für die Analyse und Monetarisierung nützlich sind.
        
        Args:
            data: Die Rohdaten aus dem Browser
            
        Returns:
            pd.DataFrame: DataFrame mit extrahierten Features
        """
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
