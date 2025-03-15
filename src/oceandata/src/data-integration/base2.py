"""
OceanData - Abstrakte Basis-Klasse für Datenquellen

Diese Klasse definiert die grundlegende Struktur für alle Datenquellen-Konnektoren,
die in der OceanData-Plattform verwendet werden.
"""

from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
import numpy as np
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from cryptography.fernet import Fernet

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("OceanData")

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

def generate_encryption_key():
    """Generiert einen symmetrischen Schlüssel für die Verschlüsselung sensibler Daten."""
    return Fernet.generate_key()

class DataSource(ABC):
    """Abstrakte Basisklasse für alle Datenquellen"""
    
    def __init__(self, source_id: str, user_id: str, category: DataCategory, encryption_key=None):
        """
        Initialisiert eine Datenquelle.
        
        Args:
            source_id: Eindeutige ID für die Datenquelle
            user_id: ID des Benutzers, dem die Daten gehören
            category: Kategorie der Datenquelle
            encryption_key: Optional. Schlüssel für die Verschlüsselung sensibler Daten.
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
        
        # Verschlüsselung initialisieren
        self.encryption_key = encryption_key or generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Logger für diese Datenquelle konfigurieren
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
        """
        Verbindung zur Datenquelle herstellen
        
        Returns:
            bool: True, wenn die Verbindung erfolgreich hergestellt wurde, sonst False.
        """
        pass
    
    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """
        Daten von der Quelle abrufen
        
        Returns:
            pd.DataFrame: Die abgerufenen Daten als DataFrame.
        """
        pass
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Verarbeitet die Daten basierend auf den festgelegten Datenschutzstufen:
        - PUBLIC: keine Verarbeitung
        - ANONYMIZED: Daten werden anonymisiert
        - ENCRYPTED: Daten werden verschlüsselt
        - SENSITIVE: Daten werden stark verschlüsselt und mit zusätzlichen Schutzmaßnahmen versehen
        
        Args:
            data: Die zu verarbeitenden Daten
            
        Returns:
            pd.DataFrame: Die verarbeiteten Daten
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
                    lambda x: self.cipher_suite.encrypt(str(x).encode()).decode() if pd.notna(x) else x
                )
                
            elif privacy_level == PrivacyLevel.SENSITIVE:
                # Für hochsensible Daten: doppelte Verschlüsselung und Markierung
                processed_data[column] = processed_data[column].apply(
                    lambda x: "SENSITIVE_DATA_PROTECTED" if pd.notna(x) else x
                )
                # In echten Implementierungen würden wir hier besondere Schutzmechanismen anwenden
        
        return processed_data
    
    def extract_metadata(self, data: pd.DataFrame) -> Dict:
        """
        Extrahiert Metadaten aus den Daten für den Datensatz
        
        Args:
            data: Die Daten, aus denen Metadaten extrahiert werden sollen
            
        Returns:
            Dict: Metadaten über den Datensatz
        """
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
