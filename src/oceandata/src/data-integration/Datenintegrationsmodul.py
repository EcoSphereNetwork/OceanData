"""
OceanData - Datenintegrationsmodul

Dieses Modul ist für die Integration und Vereinheitlichung von Daten aus verschiedenen Quellen verantwortlich.
Es bildet die Grundlage für die Datenanalyse und -monetarisierung.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import uuid
import os
from typing import Dict, List, Optional, Union, Any, Tuple
from cryptography.fernet import Fernet

# Importiere die Basisklassen
from oceandata.data_integration.base import DataSource, DataCategory, PrivacyLevel

logger = logging.getLogger("OceanData.Integration")

class DataIntegrator:
    """
    Klasse für die Integration von Daten aus verschiedenen Quellen.
    
    Aufgaben:
    - Registrierung verschiedener Datenquellen
    - Sammlung und Vereinheitlichung von Daten
    - Zeitliche Abstimmung von Daten
    - Bereinigung und Transformation
    - Vorbereitung für Analyse und Monetarisierung
    """
    
    def __init__(self, user_id: str, encryption_key=None):
        """
        Initialisiert den Datenintegrator.
        
        Args:
            user_id: ID des Benutzers, dessen Daten integriert werden
            encryption_key: Optional. Schlüssel für die Verschlüsselung sensibler Daten.
        """
        self.user_id = user_id
        self.connectors = {}  # Dictionary mit Datenquellen (source_id -> DataSource)
        self.integrated_data = {}  # Integrierte Daten (source_id -> DataFrame)
        self.metadata = {}  # Metadaten (source_id -> Dict)
        
        # Verschlüsselung für integrierte Daten
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        logger.info(f"Datenintegrator für Benutzer {user_id} initialisiert")
    
    def add_connector(self, connector: DataSource) -> bool:
        """
        Fügt einen Datenkonnektor hinzu
        
        Args:
            connector: Der hinzuzufügende Datenkonnektor
            
        Returns:
            bool: True wenn erfolgreich, sonst False
        """
        try:
            if connector.user_id != self.user_id:
                logger.warning(f"Konnektor {connector.source_id} gehört nicht zu Benutzer {self.user_id}")
                return False
                
            self.connectors[connector.source_id] = connector
            logger.info(f"Konnektor {connector.source_id} für Benutzer {self.user_id} hinzugefügt")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Konnektor {connector.source_id}: {str(e)}")
            return False
    
    def remove_connector(self, source_id: str) -> bool:
        """
        Entfernt einen Datenkonnektor
        
        Args:
            source_id: ID der zu entfernenden Datenquelle
            
        Returns:
            bool: True wenn erfolgreich, sonst False
        """
        try:
            if source_id in self.connectors:
                del self.connectors[source_id]
                logger.info(f"Konnektor {source_id} für Benutzer {self.user_id} entfernt")
                
                # Entferne auch zugehörige Daten
                if source_id in self.integrated_data:
                    del self.integrated_data[source_id]
                if source_id in self.metadata:
                    del self.metadata[source_id]
                
                return True
            else:
                logger.warning(f"Konnektor {source_id} nicht gefunden")
                return False
                
        except Exception as e:
            logger.error(f"Fehler beim Entfernen von Konnektor {source_id}: {str(e)}")
            return False
    
    def get_connectors(self) -> Dict[str, dict]:
        """
        Gibt eine Liste aller registrierten Konnektoren zurück
        
        Returns:
            Dict: Informationen über alle Konnektoren (source_id -> info_dict)
        """
        connector_info = {}
        
        for source_id, connector in self.connectors.items():
            info = {
                "source_id": source_id,
                "category": connector.category.value,
                "last_sync": connector.last_sync.isoformat() if connector.last_sync else None,
                "status": "active" if connector.last_sync else "inactive"
            }
            connector_info[source_id] = info
            
        return connector_info
    
    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Sammelt Daten von allen registrierten Konnektoren
        
        Returns:
            Dict: Integrierte Daten (source_id -> DataFrame)
        """
        for source_id, connector in self.connectors.items():
            try:
                logger.info(f"Sammle Daten von {source_id}")
                
                # Daten abrufen
                result = connector.get_data()
                
                if result["status"] == "success" and result["data"] is not None:
                    self.integrated_data[source_id] = result["data"]
                    self.metadata[source_id] = result["metadata"]
                    logger.info(f"Daten von {source_id} erfolgreich gesammelt: {len(result['data'])} Einträge")
                else:
                    logger.warning(f"Keine Daten von {source_id} gesammelt: {result.get('error', 'Unbekannter Fehler')}")
                    
            except Exception as e:
                logger.error(f"Fehler beim Sammeln von Daten von {source_id}: {str(e)}")
        
        return self.integrated_data
    
    def get_dataset_metadata(self) -> Dict:
        """
        Erstellt Metadaten für den gesamten integrierten Datensatz
        
        Returns:
            Dict: Metadaten über den integrierten Datensatz
        """
        dataset_metadata = {
            "user_id": self.user_id,
            "name": f"User Data Bundle - {self.user_id}",
            "description": "Integrierte Benutzerdaten aus verschiedenen Quellen",
            "created": datetime.now().isoformat(),
            "sources": list(self.integrated_data.keys()),
            "records_count": {
                source: len(data) for source, data in self.integrated_data.items()
            },
            "total_records": sum(len(data) for data in self.integrated_data.values()),
            "data_categories": list(set(self.connectors[source_id].category.value 
                                      for source_id in self.integrated_data.keys()))
        }
        
        return dataset_metadata
    
    def combine_datasets(self, source_ids: List[str] = None, method: str = "append") -> pd.DataFrame:
        """
        Kombiniert Datensätze aus verschiedenen Quellen zu einem DataFrame
        
        Args:
            source_ids: Liste der zu kombinierenden Quellen-IDs (None für alle)
            method: Methode zur Kombination ('append', 'merge', 'pivot')
            
        Returns:
            pd.DataFrame: Kombinierter Datensatz
        """
        if source_ids is None:
            source_ids = list(self.integrated_data.keys())
        
        if not source_ids:
            logger.warning("Keine Datenquellen zum Kombinieren angegeben")
            return pd.DataFrame()
        
        # Filtere nur verfügbare Quellen
        available_sources = [s for s in source_ids if s in self.integrated_data]
        if not available_sources:
            logger.warning("Keine der angegebenen Datenquellen ist verfügbar")
            return pd.DataFrame()
        
        try:
            if method == "append":
                # Einfaches Verketten der Datensätze
                combined_data = pd.concat(
                    [self.integrated_data[source_id] for source_id in available_sources],
                    ignore_index=True
                )
                
                # Ursprungsquelle hinzufügen
                combined_data['data_source'] = [source_id for source_id in available_sources 
                                             for _ in range(len(self.integrated_data[source_id]))]
                
            elif method == "merge":
                # Versuche, Datensätze basierend auf gemeinsamen Schlüsseln zu vereinen
                # Dies erfordert eine komplexere Logik und hängt stark von den Datensätzen ab
                logger.warning("Merge-Methode ist komplex und erfordert weitere Implementierung")
                combined_data = pd.DataFrame()  # Platzhalter
                
            elif method == "pivot":
                # Pivot-Tabelle mit Benutzer als Schlüssel
                # Dies setzt voraus, dass alle Datensätze einen gemeinsamen Benutzer-ID haben
                logger.warning("Pivot-Methode ist komplex und erfordert weitere Implementierung")
                combined_data = pd.DataFrame()  # Platzhalter
                
            else:
                logger.error(f"Unbekannte Kombinationsmethode: {method}")
                return pd.DataFrame()
                
            logger.info(f"Datensätze erfolgreich mit Methode '{method}' kombiniert: {len(combined_data)} Einträge")
            return combined_data
            
        except Exception as e:
            logger.error(f"Fehler beim Kombinieren von Datensätzen: {str(e)}")
            return pd.DataFrame()
    
    def prepare_for_tokenization(self) -> Dict:
        """
        Bereitet die integrierten Daten für die Tokenisierung mit Ocean Protocol vor
        
        Returns:
            Dict: Tokenisierungspaket mit Metadaten und Hash-Verifikation
        """
        # Für jeden Datensatz einen Hash erstellen, um die Integrität zu gewährleisten
        dataset_hashes = {}
        for source, data in self.integrated_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Konvertiere DataFrame in JSON für Hashing
                data_json = data.to_json()
                
                # Erstelle Hash für die Daten
                data_hash = hashlib.sha256(data_json.encode()).hexdigest()
                dataset_hashes[source] = data_hash
        
        tokenization_package = {
            'metadata': self.get_dataset_metadata(),
            'dataset_hashes': dataset_hashes,
            'timestamp': datetime.now().isoformat(),
            'asset_id': str(uuid.uuid4())
        }
        
        logger.info(f"Tokenisierungspaket für Benutzer {self.user_id} erstellt")
        return tokenization_package
    
    def save_to_disk(self, directory: str) -> bool:
        """
        Speichert die integrierten Daten auf der Festplatte
        
        Args:
            directory: Verzeichnis zum Speichern der Daten
            
        Returns:
            bool: True wenn erfolgreich, sonst False
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Speichere jede Datenquelle als separate CSV-Datei
            for source_id, data in self.integrated_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    file_path = os.path.join(directory, f"{source_id}.csv")
                    data.to_csv(file_path, index=False)
                    logger.info(f"Daten von {source_id} in {file_path} gespeichert")
            
            # Speichere Metadaten als JSON
            metadata_path = os.path.join(directory, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    'dataset_metadata': self.get_dataset_metadata(),
                    'source_metadata': self.metadata
                }, f, indent=2)
                
            logger.info(f"Metadaten in {metadata_path} gespeichert")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Daten: {str(e)}")
            return False
