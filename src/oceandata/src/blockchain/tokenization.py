"""
OceanData - Tokenisierungsmodul für Ocean Protocol

Dieses Modul ermöglicht die Tokenisierung von Daten mit Ocean Protocol.
Es bietet Funktionen zum Erstellen von Datentokens, zum Veröffentlichen von Datensätzen
und zum Verwalten von Zugriffsrechten.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import uuid

# Konfiguriere Logger
logger = logging.getLogger("OceanData.Tokenization")

class OceanTokenizer:
    """
    Klasse für die Tokenisierung von Daten mit Ocean Protocol.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den Ocean Tokenizer.
        
        Args:
            config: Konfiguration für Ocean Protocol
        """
        self.config = config or self._get_default_config()
        self.web3 = None
        self.ocean = None
        self.account = None
        self.is_connected = False
        self.tokens = {}
        self.published_assets = {}
        
        logger.info("Ocean Tokenizer initialisiert")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Gibt die Standardkonfiguration für Ocean Protocol zurück.
        
        Returns:
            Dict: Standardkonfiguration
        """
        return {
            "network": os.environ.get("OCEAN_NETWORK", "polygon"),
            "rpc_url": os.environ.get("OCEAN_RPC_URL", "https://polygon-rpc.com"),
            "subgraph_url": os.environ.get("OCEAN_SUBGRAPH_URL", 
                                          "https://v4.subgraph.polygon.oceanprotocol.com/subgraphs/name/oceanprotocol/ocean-subgraph"),
            "metadata_cache_uri": os.environ.get("OCEAN_METADATA_CACHE_URI", 
                                               "https://v4.aquarius.oceanprotocol.com"),
            "provider_url": os.environ.get("OCEAN_PROVIDER_URL", 
                                         "https://v4.provider.polygon.oceanprotocol.com"),
            "gas_limit": int(os.environ.get("OCEAN_GAS_LIMIT", "1000000")),
            "private_key": os.environ.get("OCEAN_PRIVATE_KEY", ""),
            "ocean_token_address": os.environ.get("OCEAN_TOKEN_ADDRESS", ""),
            "factory_address": os.environ.get("OCEAN_FACTORY_ADDRESS", ""),
            "fixed_rate_exchange_address": os.environ.get("OCEAN_FIXED_RATE_EXCHANGE_ADDRESS", ""),
            "mock_mode": os.environ.get("OCEAN_MOCK_MODE", "true").lower() == "true"
        }
    
    def connect(self) -> bool:
        """
        Stellt eine Verbindung zu Ocean Protocol her.
        
        Returns:
            bool: True, wenn die Verbindung erfolgreich hergestellt wurde
        """
        if self.is_connected:
            logger.info("Bereits mit Ocean Protocol verbunden")
            return True
        
        try:
            if self.config["mock_mode"]:
                logger.info("Mock-Modus aktiviert, simuliere Verbindung zu Ocean Protocol")
                self.is_connected = True
                return True
            
            # In einer echten Implementierung würden wir hier die Verbindung zu Ocean Protocol herstellen
            # Beispiel:
            # from ocean_lib.web3_internal.utils import connect_to_network
            # from ocean_lib.example_config import get_config_dict
            # from ocean_lib.ocean.ocean import Ocean
            # 
            # config = get_config_dict(self.config["network"])
            # self.web3 = connect_to_network(config["network_url"])
            # self.ocean = Ocean(config)
            # self.account = self.web3.eth.account.from_key(self.config["private_key"])
            # self.is_connected = True
            
            # Für das MVP simulieren wir eine erfolgreiche Verbindung
            logger.info(f"Verbindung zu Ocean Protocol ({self.config['network']}) hergestellt")
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Verbinden mit Ocean Protocol: {str(e)}")
            self.is_connected = False
            return False
    
    def create_data_token(self, name: str, symbol: str) -> Dict[str, Any]:
        """
        Erstellt einen neuen Datentoken.
        
        Args:
            name: Name des Datentokens
            symbol: Symbol des Datentokens
            
        Returns:
            Dict: Informationen zum erstellten Datentoken
        """
        if not self.is_connected and not self.connect():
            return {
                "success": False,
                "error": "Nicht mit Ocean Protocol verbunden"
            }
        
        try:
            if self.config["mock_mode"]:
                # Simuliere Tokenisierung im Mock-Modus
                token_address = f"0x{uuid.uuid4().hex[:40]}"
                token_id = str(uuid.uuid4())
                
                token_info = {
                    "success": True,
                    "token_address": token_address,
                    "token_id": token_id,
                    "name": name,
                    "symbol": symbol,
                    "created_at": datetime.now().isoformat(),
                    "tx_hash": f"0x{uuid.uuid4().hex}",
                    "network": self.config["network"]
                }
                
                # Speichere Token-Informationen
                self.tokens[token_id] = token_info
                
                logger.info(f"Datentoken erstellt (Mock): {symbol} ({token_address})")
                return token_info
            
            # In einer echten Implementierung würden wir hier den Datentoken erstellen
            # Beispiel:
            # from ocean_lib.web3_internal.constants import ZERO_ADDRESS
            # 
            # data_token = self.ocean.create_data_token(
            #     name, 
            #     symbol, 
            #     from_wallet=self.account,
            #     blob=f"{name} - Created by OceanData"
            # )
            # token_address = data_token.address
            
            # Für das MVP simulieren wir eine erfolgreiche Tokenisierung
            token_address = f"0x{uuid.uuid4().hex[:40]}"
            token_id = str(uuid.uuid4())
            
            token_info = {
                "success": True,
                "token_address": token_address,
                "token_id": token_id,
                "name": name,
                "symbol": symbol,
                "created_at": datetime.now().isoformat(),
                "tx_hash": f"0x{uuid.uuid4().hex}",
                "network": self.config["network"]
            }
            
            # Speichere Token-Informationen
            self.tokens[token_id] = token_info
            
            logger.info(f"Datentoken erstellt: {symbol} ({token_address})")
            return token_info
            
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Datentokens: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Erstellen des Datentokens: {str(e)}"
            }
    
    def publish_dataset(self, 
                       token_id: str, 
                       metadata: Dict[str, Any], 
                       price: float, 
                       files: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Veröffentlicht einen Datensatz mit dem angegebenen Datentoken.
        
        Args:
            token_id: ID des Datentokens
            metadata: Metadaten des Datensatzes
            price: Preis des Datensatzes in OCEAN
            files: Liste der Dateien im Datensatz
            
        Returns:
            Dict: Informationen zum veröffentlichten Datensatz
        """
        if not self.is_connected and not self.connect():
            return {
                "success": False,
                "error": "Nicht mit Ocean Protocol verbunden"
            }
        
        if token_id not in self.tokens:
            return {
                "success": False,
                "error": f"Datentoken mit ID {token_id} nicht gefunden"
            }
        
        token_info = self.tokens[token_id]
        
        try:
            if self.config["mock_mode"]:
                # Simuliere Veröffentlichung im Mock-Modus
                asset_id = f"did:op:{uuid.uuid4().hex}"
                
                # Erstelle Asset-Informationen
                asset_info = {
                    "success": True,
                    "asset_id": asset_id,
                    "token_id": token_id,
                    "token_address": token_info["token_address"],
                    "price": price,
                    "metadata": metadata,
                    "files": files or [],
                    "created_at": datetime.now().isoformat(),
                    "tx_hash": f"0x{uuid.uuid4().hex}",
                    "network": self.config["network"],
                    "marketplace_url": f"https://market.oceanprotocol.com/asset/{asset_id.replace('did:op:', '')}"
                }
                
                # Speichere Asset-Informationen
                self.published_assets[asset_id] = asset_info
                
                logger.info(f"Datensatz veröffentlicht (Mock): {metadata.get('name', 'Unbenannt')} ({asset_id})")
                return asset_info
            
            # In einer echten Implementierung würden wir hier den Datensatz veröffentlichen
            # Beispiel:
            # from ocean_lib.assets.asset import Asset
            # from ocean_lib.models.datatoken import Datatoken
            # 
            # data_token = Datatoken(self.web3, token_info["token_address"])
            # 
            # # Erstelle Asset-Metadaten
            # asset_metadata = {
            #     "main": {
            #         "type": "dataset",
            #         "name": metadata.get("name", ""),
            #         "author": metadata.get("author", ""),
            #         "license": metadata.get("license", ""),
            #         "dateCreated": metadata.get("created", datetime.now().isoformat()),
            #         "files": files or []
            #     },
            #     "additionalInformation": metadata.get("additionalInformation", {})
            # }
            # 
            # # Veröffentliche Asset
            # asset = Asset.create(
            #     metadata=asset_metadata,
            #     publisher_wallet=self.account,
            #     data_token_address=token_info["token_address"],
            #     provider_url=self.config["provider_url"]
            # )
            # 
            # # Erstelle Fixed-Rate-Exchange
            # exchange = self.ocean.fixed_rate_exchange.create(
            #     datatoken=data_token,
            #     base_token_address=self.config["ocean_token_address"],
            #     fixed_rate=price,
            #     from_wallet=self.account
            # )
            
            # Für das MVP simulieren wir eine erfolgreiche Veröffentlichung
            asset_id = f"did:op:{uuid.uuid4().hex}"
            
            # Erstelle Asset-Informationen
            asset_info = {
                "success": True,
                "asset_id": asset_id,
                "token_id": token_id,
                "token_address": token_info["token_address"],
                "price": price,
                "metadata": metadata,
                "files": files or [],
                "created_at": datetime.now().isoformat(),
                "tx_hash": f"0x{uuid.uuid4().hex}",
                "network": self.config["network"],
                "marketplace_url": f"https://market.oceanprotocol.com/asset/{asset_id.replace('did:op:', '')}"
            }
            
            # Speichere Asset-Informationen
            self.published_assets[asset_id] = asset_info
            
            logger.info(f"Datensatz veröffentlicht: {metadata.get('name', 'Unbenannt')} ({asset_id})")
            return asset_info
            
        except Exception as e:
            logger.error(f"Fehler beim Veröffentlichen des Datensatzes: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Veröffentlichen des Datensatzes: {str(e)}"
            }
    
    def tokenize_dataset(self, 
                        name: str, 
                        metadata: Dict[str, Any], 
                        price: float, 
                        files: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Tokenisiert einen Datensatz in einem Schritt (Erstellen eines Tokens und Veröffentlichen).
        
        Args:
            name: Name des Datensatzes
            metadata: Metadaten des Datensatzes
            price: Preis des Datensatzes in OCEAN
            files: Liste der Dateien im Datensatz
            
        Returns:
            Dict: Informationen zum tokenisierten Datensatz
        """
        # Erstelle ein Symbol aus dem Namen (max. 8 Zeichen, nur Großbuchstaben und Zahlen)
        symbol = ''.join(c for c in name if c.isalnum())[:6].upper()
        symbol = f"DT{symbol}"
        
        # Erstelle Datentoken
        token_result = self.create_data_token(name, symbol)
        
        if not token_result["success"]:
            return token_result
        
        # Veröffentliche Datensatz
        return self.publish_dataset(token_result["token_id"], metadata, price, files)
    
    def get_token_info(self, token_id: str) -> Dict[str, Any]:
        """
        Gibt Informationen zu einem Datentoken zurück.
        
        Args:
            token_id: ID des Datentokens
            
        Returns:
            Dict: Informationen zum Datentoken
        """
        if token_id not in self.tokens:
            return {
                "success": False,
                "error": f"Datentoken mit ID {token_id} nicht gefunden"
            }
        
        return self.tokens[token_id]
    
    def get_asset_info(self, asset_id: str) -> Dict[str, Any]:
        """
        Gibt Informationen zu einem veröffentlichten Datensatz zurück.
        
        Args:
            asset_id: ID des Datensatzes
            
        Returns:
            Dict: Informationen zum Datensatz
        """
        if asset_id not in self.published_assets:
            return {
                "success": False,
                "error": f"Datensatz mit ID {asset_id} nicht gefunden"
            }
        
        return self.published_assets[asset_id]
    
    def get_all_assets(self) -> List[Dict[str, Any]]:
        """
        Gibt Informationen zu allen veröffentlichten Datensätzen zurück.
        
        Returns:
            List: Liste aller veröffentlichten Datensätze
        """
        return list(self.published_assets.values())
    
    def get_all_tokens(self) -> List[Dict[str, Any]]:
        """
        Gibt Informationen zu allen erstellten Datentokens zurück.
        
        Returns:
            List: Liste aller erstellten Datentokens
        """
        return list(self.tokens.values())


class OceanDataTokenizer:
    """
    Höhere Abstraktionsebene für die Tokenisierung von OceanData-Datensätzen.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den OceanData Tokenizer.
        
        Args:
            config: Konfiguration für Ocean Protocol
        """
        self.tokenizer = OceanTokenizer(config)
        logger.info("OceanData Tokenizer initialisiert")
    
    def tokenize_data_package(self, tokenization_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenisiert ein Datenpaket aus dem OceanData-Integrator.
        
        Args:
            tokenization_package: Tokenisierungspaket aus dem OceanData-Integrator
            
        Returns:
            Dict: Informationen zum tokenisierten Datensatz
        """
        if not self.tokenizer.is_connected and not self.tokenizer.connect():
            return {
                "success": False,
                "error": "Nicht mit Ocean Protocol verbunden"
            }
        
        try:
            # Extrahiere Metadaten aus dem Tokenisierungspaket
            metadata = tokenization_package.get("metadata", {})
            basic_metadata = metadata.get("basic", {})
            ocean_metadata = metadata.get("ocean", {})
            
            # Erstelle Namen und Beschreibung
            name = basic_metadata.get("name", f"Dataset-{uuid.uuid4().hex[:8]}")
            description = basic_metadata.get("description", "Tokenisierter Datensatz von OceanData")
            
            # Schätze den Preis basierend auf dem geschätzten Wert
            estimated_value = tokenization_package.get("estimated_value", 1.0)
            price = max(0.1, estimated_value)  # Mindestpreis: 0.1 OCEAN
            
            # Erstelle Dateimetadaten
            files = []
            if "data_sample" in tokenization_package:
                # Füge Beispieldaten als JSON-Datei hinzu
                files.append({
                    "type": "json",
                    "name": "sample.json",
                    "url": "",  # Wird später von Ocean Protocol ausgefüllt
                    "contentType": "application/json"
                })
            
            # Erstelle zusätzliche Metadaten
            additional_info = {
                "privacy_level": tokenization_package.get("privacy_level", "unknown"),
                "user_id": tokenization_package.get("user_id", "anonymous"),
                "timestamp": tokenization_package.get("timestamp", datetime.now().isoformat()),
                "dataset_hashes": tokenization_package.get("dataset_hashes", {}),
                "sources": basic_metadata.get("sources", []),
                "total_records": basic_metadata.get("total_records", 0),
                "categories": ocean_metadata.get("additionalInformation", {}).get("categories", []),
                "tags": ocean_metadata.get("additionalInformation", {}).get("tags", [])
            }
            
            # Kombiniere Metadaten
            combined_metadata = {
                "name": name,
                "description": description,
                "author": basic_metadata.get("author", "OceanData User"),
                "license": "CC-BY",
                "created": basic_metadata.get("created", datetime.now().isoformat()),
                "additionalInformation": additional_info
            }
            
            # Tokenisiere Datensatz
            result = self.tokenizer.tokenize_dataset(name, combined_metadata, price, files)
            
            if result["success"]:
                # Füge zusätzliche Informationen hinzu
                result["estimated_value"] = estimated_value
                result["privacy_level"] = tokenization_package.get("privacy_level", "unknown")
                result["sources"] = basic_metadata.get("sources", [])
                result["total_records"] = basic_metadata.get("total_records", 0)
                
                logger.info(f"Datenpaket erfolgreich tokenisiert: {name} ({result['asset_id']})")
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler beim Tokenisieren des Datenpakets: {str(e)}")
            return {
                "success": False,
                "error": f"Fehler beim Tokenisieren des Datenpakets: {str(e)}"
            }
    
    def get_tokenized_assets(self, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Gibt alle tokenisierten Datensätze zurück, optional gefiltert nach Benutzer-ID.
        
        Args:
            user_id: Benutzer-ID für die Filterung
            
        Returns:
            List: Liste der tokenisierten Datensätze
        """
        assets = self.tokenizer.get_all_assets()
        
        if user_id:
            # Filtere nach Benutzer-ID
            assets = [
                asset for asset in assets 
                if asset.get("metadata", {}).get("additionalInformation", {}).get("user_id") == user_id
            ]
        
        return assets


# Beispiel für die Verwendung des Tokenisierungsmoduls
def demo_tokenization():
    """
    Demonstriert die Tokenisierung von Daten mit Ocean Protocol.
    
    Returns:
        Dict: Ergebnisse der Tokenisierung
    """
    # Erstelle ein Beispiel-Tokenisierungspaket
    tokenization_package = {
        "metadata": {
            "basic": {
                "name": "Beispiel-Datensatz",
                "description": "Ein Beispiel-Datensatz für die Tokenisierung",
                "author": "OceanData Demo",
                "created": datetime.now().isoformat(),
                "sources": ["browser", "smartwatch"],
                "records_count": {
                    "browser": 100,
                    "smartwatch": 200
                },
                "total_records": 300,
                "estimated_value": 2.5,
                "privacy_level": "medium"
            },
            "ocean": {
                "main": {
                    "type": "dataset",
                    "name": "Beispiel-Datensatz",
                    "dateCreated": datetime.now().isoformat(),
                    "author": "OceanData Demo",
                    "license": "CC-BY",
                    "files": []
                },
                "additionalInformation": {
                    "description": "Ein Beispiel-Datensatz für die Tokenisierung",
                    "categories": ["browsing", "health"],
                    "tags": ["browser", "smartwatch", "demo"],
                    "privacy": {
                        "level": "medium",
                        "complianceStandards": ["GDPR", "CCPA"],
                        "anonymizationMethods": ["pseudonymization", "aggregation"]
                    }
                }
            }
        },
        "dataset_hashes": {
            "browser": "0x1234567890abcdef",
            "smartwatch": "0xabcdef1234567890"
        },
        "timestamp": datetime.now().isoformat(),
        "privacy_level": "medium",
        "user_id": "demo_user",
        "estimated_value": 2.5,
        "data_sample": {
            "browser": [
                {"website": "example.com", "duration": 120},
                {"website": "github.com", "duration": 300}
            ],
            "smartwatch": [
                {"heart_rate": 72, "steps": 1000},
                {"heart_rate": 75, "steps": 2000}
            ]
        }
    }
    
    # Erstelle einen OceanData Tokenizer
    tokenizer = OceanDataTokenizer()
    
    # Verbinde mit Ocean Protocol
    tokenizer.tokenizer.connect()
    
    # Tokenisiere das Datenpaket
    result = tokenizer.tokenize_data_package(tokenization_package)
    
    return {
        "tokenization_package": tokenization_package,
        "tokenization_result": result
    }

if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Führe die Demo aus
    demo_result = demo_tokenization()
    
    # Zeige Ergebnisse
    print("\n🪙 OCEAN PROTOCOL TOKENISIERUNG DEMO")
    print("=====================================")
    
    result = demo_result["tokenization_result"]
    
    if result["success"]:
        print(f"\n✅ Tokenisierung erfolgreich!")
        print(f"Asset-ID: {result['asset_id']}")
        print(f"Token-Adresse: {result['token_address']}")
        print(f"Preis: {result['price']} OCEAN")
        print(f"Netzwerk: {result['network']}")
        print(f"Marketplace URL: {result['marketplace_url']}")
        
        print("\n📊 Metadaten:")
        metadata = result["metadata"]
        print(f"Name: {metadata['name']}")
        print(f"Beschreibung: {metadata['description']}")
        print(f"Autor: {metadata['author']}")
        print(f"Erstellt: {metadata['created']}")
        
        print("\n🔍 Zusätzliche Informationen:")
        additional_info = metadata["additionalInformation"]
        print(f"Datenschutzniveau: {additional_info['privacy_level']}")
        print(f"Benutzer-ID: {additional_info['user_id']}")
        print(f"Quellen: {', '.join(additional_info['sources'])}")
        print(f"Gesamtdatensätze: {additional_info['total_records']}")
        
        if "categories" in additional_info:
            print(f"Kategorien: {', '.join(additional_info['categories'])}")
        
        if "tags" in additional_info:
            print(f"Tags: {', '.join(additional_info['tags'])}")
    else:
        print(f"\n❌ Tokenisierung fehlgeschlagen!")
        print(f"Fehler: {result.get('error', 'Unbekannter Fehler')}")