"""
OceanData Konfigurationsdatei

Diese Datei enthält alle Konfigurationseinstellungen für die OceanData-Plattform.
"""

import os
import logging
from typing import Dict, Any

# Basis-Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class OceanDataConfig:
    """Konfigurationsklasse für die OceanData-Plattform"""
    
    def __init__(self, env: str = None):
        """
        Initialisiert die Konfiguration basierend auf der Umgebung.
        
        Args:
            env: Umgebung ('dev', 'test', 'prod'), falls None wird aus OCEAN_DATA_ENV gelesen
        """
        self.env = env or os.environ.get('OCEAN_DATA_ENV', 'dev')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Lädt die Konfiguration basierend auf der Umgebung"""
        base_config = self._get_base_config()
        
        # Umgebungsspezifische Konfigurationen
        if self.env == 'dev':
            return {**base_config, **self._get_dev_config()}
        elif self.env == 'test':
            return {**base_config, **self._get_test_config()}
        elif self.env == 'prod':
            return {**base_config, **self._get_prod_config()}
        else:
            # Bei unbekannter Umgebung Entwicklungskonfiguration verwenden
            logging.warning(f"Unbekannte Umgebung: {self.env}, verwende Entwicklungskonfiguration")
            return {**base_config, **self._get_dev_config()}
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Basis-Konfiguration für alle Umgebungen"""
        return {
            # Allgemeine Konfiguration
            'app_name': 'OceanData',
            'app_version': '0.1.0',
            'debug': True,
            
            # Logging-Konfiguration
            'logging': {
                'level': logging.INFO,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_path': None  # Kein Logging in Datei in der Basis-Konfiguration
            },
            
            # Datenbankverbindungen
            'database': {
                'type': 'sqlite',
                'path': ':memory:'  # In-Memory-SQLite-Datenbank
            },
            
            # Datenerfassung-Konfiguration
            'data_collection': {
                'batch_size': 100,
                'sync_interval': 3600,  # Synchronisationintervall in Sekunden (1 Stunde)
                'max_retries': 3
            },
            
            # KI-Konfiguration
            'ai': {
                'anomaly_detection_method': 'isolation_forest',
                'anomaly_contamination': 0.05,
                'sentiment_analysis_model': 'default',
                'predictive_model_type': 'lstm',
                'forecast_horizon': 7
            },
            
            # Datenschutz-Konfiguration
            'privacy': {
                'min_group_size': 5,
                'noise_level': 0.01,
                'outlier_removal': True,
                'max_query_per_token': 10,
                'default_token_expiry': 3600  # Token-Ablaufzeit in Sekunden (1 Stunde)
            },
            
            # Ocean Protocol-Konfiguration
            'ocean': {
                'network': 'polygon',  # oder 'ethereum', 'bsc', usw.
                'rpc_url': 'https://polygon-rpc.com',
                'subgraph_url': 'https://v4.subgraph.polygon.oceanprotocol.com/subgraphs/name/oceanprotocol/ocean-subgraph',
                'metadata_cache_uri': 'https://v4.aquarius.oceanprotocol.com',
                'provider_url': 'https://v4.provider.polygon.oceanprotocol.com'
            },
            
            # Web-Server-Konfiguration
            'server': {
                'host': '127.0.0.1',
                'port': 5000,
                'workers': 4,
                'timeout': 60
            },
            
            # Frontend-Konfiguration
            'frontend': {
                'api_url': 'http://localhost:5000/api',
                'theme': 'light',
                'language': 'de'
            }
        }
    
    def _get_dev_config(self) -> Dict[str, Any]:
        """Entwicklungsumgebung-Konfiguration"""
        return {
            'debug': True,
            'logging': {
                'level': logging.DEBUG,
                'file_path': 'logs/oceandata-dev.log'
            },
            'server': {
                'host': '127.0.0.1',
                'port': 5000,
                'workers': 1,  # Nur ein Worker für Entwicklung
                'reload': True  # Auto-Reload bei Codeänderungen
            },
            # Mocking für Entwicklung
            'mock': {
                'enabled': True,
                'blockchain': True,
                'ocean_api': True
            }
        }
    
    def _get_test_config(self) -> Dict[str, Any]:
        """Testumgebung-Konfiguration"""
        return {
            'debug': True,
            'database': {
                'type': 'sqlite',
                'path': 'test_db.sqlite'  # Separate Datenbank für Tests
            },
            'logging': {
                'level': logging.INFO,
                'file_path': 'logs/oceandata-test.log'
            },
            'server': {
                'host': '127.0.0.1',
                'port': 5001,  # Anderer Port für Testumgebung
                'workers': 2
            },
            # Mocking für Tests
            'mock': {
                'enabled': True,
                'blockchain': True,
                'ocean_api': True
            },
            # Spezifische Test-Konfiguration
            'testing': {
                'test_data_path': 'tests/data',
                'performance_metrics_enabled': True
            }
        }
    
    def _get_prod_config(self) -> Dict[str, Any]:
        """Produktionsumgebung-Konfiguration"""
        return {
            'debug': False,
            'database': {
                'type': 'postgresql',
                # Aus Umgebungsvariablen oder sicheren Quellen laden
                'host': os.environ.get('DB_HOST', 'localhost'),
                'port': int(os.environ.get('DB_PORT', 5432)),
                'name': os.environ.get('DB_NAME', 'oceandata'),
                'user': os.environ.get('DB_USER', 'postgres'),
                'password': os.environ.get('DB_PASSWORD', ''),
                'ssl_mode': 'require'
            },
            'logging': {
                'level': logging.WARNING,
                'file_path': '/var/log/oceandata/oceandata-prod.log'
            },
            'server': {
                'host': '0.0.0.0',  # Auf allen Interfaces lauschen
                'port': int(os.environ.get('PORT', 8000)),
                'workers': int(os.environ.get('WORKERS', 4)),
                'timeout': 120,
                'reload': False
            },
            # Produktionseinstellungen für Ocean
            'ocean': {
                'network': os.environ.get('OCEAN_NETWORK', 'polygon'),
                'rpc_url': os.environ.get('OCEAN_RPC_URL', 'https://polygon-rpc.com'),
                'subgraph_url': os.environ.get('OCEAN_SUBGRAPH_URL', 'https://v4.subgraph.polygon.oceanprotocol.com/subgraphs/name/oceanprotocol/ocean-subgraph'),
                'metadata_cache_uri': os.environ.get('OCEAN_METADATA_CACHE_URI', 'https://v4.aquarius.oceanprotocol.com'),
                'provider_url': os.environ.get('OCEAN_PROVIDER_URL', 'https://v4.provider.polygon.oceanprotocol.com')
            },
            # Security settings
            'security': {
                'secret_key': os.environ.get('SECRET_KEY', 'generate-a-secure-key'),
                'jwt_secret': os.environ.get('JWT_SECRET', 'generate-a-secure-jwt-secret'),
                'csrf_protection': True,
                'session_secure': True,
                'cors_origins': os.environ.get('CORS_ORIGINS', '*').split(',')
            },
            # Keine Mocks in Produktion
            'mock': {
                'enabled': False,
                'blockchain': False,
                'ocean_api': False
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Holt einen Konfigurationswert.
        
        Args:
            key: Schlüssel des Konfigurationswerts
            default: Standardwert, falls Schlüssel nicht gefunden
            
        Returns:
            Any: Konfigurationswert oder Standardwert
        """
        # Unterstützt verschachtelte Schlüssel wie 'server.host'
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Setzt einen Konfigurationswert.
        
        Args:
            key: Schlüssel des Konfigurationswerts
            value: Neuer Wert
        """
        # Unterstützt verschachtelte Schlüssel wie 'server.host'
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __getitem__(self, key: str) -> Any:
        """Erlaubt dict-ähnlichen Zugriff auf Konfiguration"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Erlaubt dict-ähnliches Setzen von Konfigurationswerten"""
        self.set(key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Gibt die gesamte Konfiguration als Dictionary zurück"""
        return self.config.copy()


# Singleton-Instanz für die Konfiguration
_instance = None

def get_config(env: str = None) -> OceanDataConfig:
    """
    Holt die Singleton-Instanz der Konfiguration.
    
    Args:
        env: Umgebung ('dev', 'test', 'prod'), falls None wird aus OCEAN_DATA_ENV gelesen
        
    Returns:
        OceanDataConfig: Konfigurationsinstanz
    """
    global _instance
    if _instance is None:
        _instance = OceanDataConfig(env)
    return _instance


# Beispielverwendung
if __name__ == "__main__":
    # Konfiguration für Entwicklungsumgebung abrufen
    config = get_config('dev')
    print(f"Server-Host: {config.get('server.host')}")
    print(f"Server-Port: {config.get('server.port')}")
    
    # Konfiguration für Testumgebung abrufen
    test_config = get_config('test')
    print(f"Test-Server-Port: {test_config.get('server.port')}")
    
    # Konfiguration ändern
    config.set('server.port', 8080)
    print(f"Neuer Server-Port: {config.get('server.port')}")
