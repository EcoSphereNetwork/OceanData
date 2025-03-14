"""
OceanData - Modularer Datenintegrationsalgorithmus

Dieser Algorithmus ist der Kern der OceanData-Plattform und ermöglicht:
1. Datenerfassung aus verschiedenen Quellen
2. Datenbereinigung und -vorbereitung
3. Datenpseudonymisierung für Datenschutz
4. Tokenisierung der Daten mittels Ocean Protocol
"""

import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import json
from abc import ABC, abstractmethod

# Basisklasse für Datenkonnektoren
class DataConnector(ABC):
    """Abstrakte Basisklasse für alle Datenkonnektoren."""
    
    def __init__(self, source_id, user_id):
        self.source_id = source_id
        self.user_id = user_id
        self.data = None
        
    @abstractmethod
    def connect(self):
        """Verbindung zur Datenquelle herstellen."""
        pass
    
    @abstractmethod
    def fetch_data(self):
        """Daten aus der Quelle abrufen."""
        pass
    
    def preprocess(self, data):
        """Grundlegende Datenvorverarbeitung."""
        # Entferne Nullwerte
        if isinstance(data, pd.DataFrame):
            data = data.dropna()
        return data
    
    def anonymize(self, data):
        """Datenpseudonmymisierung für Datenschutz."""
        # Benutzer-ID hashen, um Anonymität zu gewährleisten
        if isinstance(data, pd.DataFrame) and 'user_id' in data.columns:
            salt = datetime.now().strftime("%Y%m%d")
            data['user_id'] = data['user_id'].apply(
                lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest()
            )
        return data
    
    def get_data(self):
        """Vollständige Pipeline zum Abrufen und Vorverarbeiten von Daten."""
        self.connect()
        raw_data = self.fetch_data()
        processed_data = self.preprocess(raw_data)
        anonymized_data = self.anonymize(processed_data)
        self.data = anonymized_data
        return self.data

# Spezifische Konnektoren für verschiedene Datenquellen
class BrowserDataConnector(DataConnector):
    """Konnektor für Browser-Daten."""
    
    def __init__(self, user_id, browser_type='chrome'):
        super().__init__('browser', user_id)
        self.browser_type = browser_type
    
    def connect(self):
        """Verbindung zum Browser-Verlauf herstellen."""
        print(f"Verbindung zu {self.browser_type}-Daten für Benutzer {self.user_id} hergestellt")
        return True
    
    def fetch_data(self):
        """Browser-Verlaufsdaten abrufen."""
        # Hier würde die tatsächliche Implementierung folgen, für das MVP simulieren wir
        sample_data = {
            'website': ['example.com', 'github.com', 'docs.python.org'],
            'timestamp': [datetime.now() for _ in range(3)],
            'duration': [120, 300, 450],
            'user_id': [self.user_id for _ in range(3)]
        }
        return pd.DataFrame(sample_data)
    
    def extract_features(self):
        """Extrahiere spezifische Features aus Browser-Daten."""
        if self.data is not None:
            # Feature-Engineering für Browser-Daten
            self.data['day_of_week'] = self.data['timestamp'].apply(lambda x: x.dayofweek)
            self.data['hour_of_day'] = self.data['timestamp'].apply(lambda x: x.hour)
        return self.data

class IoTDataConnector(DataConnector):
    """Konnektor für IoT-Gerätedaten."""
    
    def __init__(self, user_id, device_type):
        super().__init__('iot', user_id)
        self.device_type = device_type
    
    def connect(self):
        """Verbindung zu IoT-Gerät herstellen."""
        print(f"Verbindung zu {self.device_type} für Benutzer {self.user_id} hergestellt")
        return True
    
    def fetch_data(self):
        """IoT-Gerätedaten abrufen."""
        # Für MVP simulieren wir Smartwatch-Daten
        if self.device_type == 'smartwatch':
            sample_data = {
                'heart_rate': [72, 75, 80, 78, 76],
                'steps': [100, 200, 300, 400, 500],
                'timestamp': [datetime.now() for _ in range(5)],
                'user_id': [self.user_id for _ in range(5)]
            }
        else:
            sample_data = {
                'device_state': ['on', 'off', 'on', 'standby'],
                'timestamp': [datetime.now() for _ in range(4)],
                'user_id': [self.user_id for _ in range(4)]
            }
        return pd.DataFrame(sample_data)

# Datenintegrator-Klasse
class DataIntegrator:
    """Klasse für die Integration verschiedener Datenquellen."""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.connectors = {}
        self.integrated_data = {}
    
    def add_connector(self, connector):
        """Füge einen neuen Datenkonnektor hinzu."""
        self.connectors[connector.source_id] = connector
    
    def collect_all_data(self):
        """Sammle Daten von allen registrierten Konnektoren."""
        for source_id, connector in self.connectors.items():
            self.integrated_data[source_id] = connector.get_data()
        return self.integrated_data
    
    def get_dataset_metadata(self):
        """Erstelle Metadaten für den Datensatz zur Verwendung mit Ocean Protocol."""
        metadata = {
            'name': f'User Data Bundle - {self.user_id}',
            'description': 'Integrierte Benutzerdaten aus verschiedenen Quellen',
            'author': f'User-{self.user_id}',
            'created': datetime.now().isoformat(),
            'sources': list(self.integrated_data.keys()),
            'records_count': {
                source: len(data) for source, data in self.integrated_data.items()
            }
        }
        return metadata
    
    def prepare_for_tokenization(self):
        """Bereite Daten für die Tokenisierung mit Ocean Protocol vor."""
        # Für jeden Datensatz einen Hash erstellen, um die Integrität zu gewährleisten
        dataset_hashes = {}
        for source, data in self.integrated_data.items():
            if isinstance(data, pd.DataFrame):
                # Konvertiere DataFrame in JSON für Hashing
                data_json = data.to_json()
                # Erstelle Hash für die Daten
                data_hash = hashlib.sha256(data_json.encode()).hexdigest()
                dataset_hashes[source] = data_hash
        
        tokenization_package = {
            'metadata': self.get_dataset_metadata(),
            'dataset_hashes': dataset_hashes,
            'timestamp': datetime.now().isoformat()
        }
        
        return tokenization_package

# Datenanalyse-Klasse für Wertsteigerung der Daten
class DataAnalyzer:
    """Analyse-Algorithmen für verschiedene Datentypen."""
    
    def __init__(self, data_integrator):
        self.data_integrator = data_integrator
        self.analytics_results = {}
    
    def analyze_browser_data(self):
        """Analysiere Browser-Daten für wertvolle Erkenntnisse."""
        if 'browser' in self.data_integrator.integrated_data:
            browser_data = self.data_integrator.integrated_data['browser']
            
            # Einfache Analysen
            if isinstance(browser_data, pd.DataFrame) and 'website' in browser_data.columns:
                site_frequency = browser_data['website'].value_counts().to_dict()
                avg_duration = browser_data['duration'].mean() if 'duration' in browser_data.columns else None
                
                self.analytics_results['browser'] = {
                    'site_frequency': site_frequency,
                    'avg_duration': avg_duration
                }
            
        return self.analytics_results.get('browser', {})
    
    def analyze_iot_data(self):
        """Analysiere IoT-Daten für wertvolle Erkenntnisse."""
        if 'iot' in self.data_integrator.integrated_data:
            iot_data = self.data_integrator.integrated_data['iot']
            
            # Einfache Analysen für Smartwatch-Daten
            if isinstance(iot_data, pd.DataFrame):
                if 'heart_rate' in iot_data.columns:
                    avg_heart_rate = iot_data['heart_rate'].mean()
                    max_heart_rate = iot_data['heart_rate'].max()
                    
                    self.analytics_results['iot'] = {
                        'avg_heart_rate': avg_heart_rate,
                        'max_heart_rate': max_heart_rate
                    }
                    
                    if 'steps' in iot_data.columns:
                        total_steps = iot_data['steps'].sum()
                        self.analytics_results['iot']['total_steps'] = total_steps
            
        return self.analytics_results.get('iot', {})
    
    def run_all_analytics(self):
        """Führe alle verfügbaren Analysen durch."""
        self.analyze_browser_data()
        self.analyze_iot_data()
        return self.analytics_results

# Beispiel für die Verwendung des Algorithmus
def demo_integration():
    """Demonstriere die Datenintegration."""
    user_id = "user123"
    
    # Erstelle einen Datenintegrator
    integrator = DataIntegrator(user_id)
    
    # Füge verschiedene Datenquellen hinzu
    browser_connector = BrowserDataConnector(user_id)
    iot_connector = IoTDataConnector(user_id, 'smartwatch')
    
    integrator.add_connector(browser_connector)
    integrator.add_connector(iot_connector)
    
    # Sammle alle Daten
    integrated_data = integrator.collect_all_data()
    
    # Bereite Daten für Tokenisierung vor
    tokenization_package = integrator.prepare_for_tokenization()
    
    # Führe Analysen durch, um den Datenwert zu steigern
    analyzer = DataAnalyzer(integrator)
    analytics_results = analyzer.run_all_analytics()
    
    return {
        'integrated_data': integrated_data,
        'tokenization_package': tokenization_package,
        'analytics_results': analytics_results
    }

if __name__ == "__main__":
    demo_results = demo_integration()
    print("Tokenisierungspaket:", json.dumps(demo_results['tokenization_package'], indent=2))
    print("Analyseergebnisse:", json.dumps(demo_results['analytics_results'], indent=2))
