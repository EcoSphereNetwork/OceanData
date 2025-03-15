"""
OceanData - Hauptanwendung

Dies ist der Einstiegspunkt für die OceanData-Plattform.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any

# Füge Projektverzeichnis zum Pfad hinzu
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Importiere Konfiguration
from oceandata.config import get_config

# Importiere Hauptkomponenten
from oceandata.data_integration.base import DataSource, DataCategory, PrivacyLevel
from oceandata.analytics.models.anomaly_detector import AnomalyDetector
from oceandata.privacy.compute_to_data import ComputeToDataManager
from oceandata.blockchain.ocean_integration import OceanIntegration
from oceandata.core.ocean_data_ai import OceanDataAI

# Importiere Datenkonnektoren
from oceandata.data_integration.connectors.browser_connector import BrowserDataConnector
from oceandata.data_integration.connectors.smartdevice_connector import SmartwatchDataConnector

# Importiere Web-Server (optional, wenn verwendet)
try:
    from oceandata.server import create_app
except ImportError:
    create_app = None

# Logger konfigurieren
logger = logging.getLogger("OceanData")

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Konfiguriert das Logging basierend auf der Konfiguration.
    
    Args:
        config: Konfigurationsobjekt
    """
    log_level = config.get('logging.level', logging.INFO)
    log_format = config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('logging.file_path')
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Stelle sicher, dass das Verzeichnis existiert
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    logger.info(f"Logging konfiguriert mit Level {logging.getLevelName(log_level)}")

def create_ocean_data_instance(config: Dict[str, Any]) -> OceanDataAI:
    """
    Erstellt eine Instanz der OceanDataAI-Klasse.
    
    Args:
        config: Konfigurationsobjekt
        
    Returns:
        OceanDataAI: Instanz der OceanDataAI-Klasse
    """
    # Erstelle Instanz mit Konfiguration
    ocean_data = OceanDataAI(config)
    
    logger.info("OceanDataAI-Instanz erstellt")
    
    return ocean_data

def run_web_server(config: Dict[str, Any], ocean_data: OceanDataAI) -> None:
    """
    Startet den Web-Server für die OceanData-Plattform.
    
    Args:
        config: Konfigurationsobjekt
        ocean_data: Instanz der OceanDataAI-Klasse
    """
    if create_app is None:
        logger.error("Web-Server-Modul nicht verfügbar")
        return
    
    host = config.get('server.host', '127.0.0.1')
    port = config.get('server.port', 5000)
    workers = config.get('server.workers', 1)
    reload = config.get('server.reload', False)
    
    # Erstelle Flask-App
    app = create_app(ocean_data)
    
    logger.info(f"Web-Server wird gestartet auf {host}:{port} mit {workers} Workern")
    
    # Starte Server mit Uvicorn oder Gunicorn (falls installiert)
    try:
        import uvicorn
        uvicorn.run(app, host=host, port=port, workers=workers, reload=reload)
    except ImportError:
        try:
            from gunicorn.app.base import BaseApplication
            
            class StandaloneApplication(BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
                
                def load_config(self):
                    for key, value in self.options.items():
                        if key in self.cfg.settings and value is not None:
                            self.cfg.set(key, value)
                
                def load(self):
                    return self.application
            
            options = {
                'bind': f"{host}:{port}",
                'workers': workers,
                'reload': reload
            }
            
            StandaloneApplication(app, options).run()
        except ImportError:
            # Fallback: Eingebauter Flask-Server
            app.run(host=host, port=port, debug=config.get('debug', False))

def run_demo_mode(ocean_data: OceanDataAI) -> None:
    """
    Führt eine Demo-Analyse mit simulierten Daten durch.
    
    Args:
        ocean_data: Instanz der OceanDataAI-Klasse
    """
    logger.info("Demo-Modus wird gestartet")
    
    # Erstelle Beispiel-Konnektoren
    browser_connector = BrowserDataConnector("demo_user", "chrome")
    smartwatch_connector = SmartwatchDataConnector("demo_user", "fitbit")
    
    # Daten abrufen
    browser_data_result = browser_connector.get_data()
    smartwatch_data_result = smartwatch_connector.get_data()
    
    if browser_data_result['status'] == 'success' and browser_data_result['data'] is not None:
        browser_data = browser_data_result['data']
        logger.info(f"Browser-Daten erfolgreich abgerufen: {len(browser_data)} Datensätze")
        
        # Analysiere Browser-Daten
        browser_analysis = ocean_data.analyze_data_source(browser_data, 'browser')
        print("\n=== Browser-Datenanalyse ===")
        print(f"Datensätze: {browser_analysis['statistics']['record_count']}")
        print(f"Anomalien: {browser_analysis['analyses']['anomalies']['count']} ({browser_analysis['analyses']['anomalies']['percentage']:.1f}%)")
        if 'source_specific' in browser_analysis['analyses'] and 'top_domains' in browser_analysis['analyses']['source_specific']:
            print("Top-Domains:")
            for domain, count in list(browser_analysis['analyses']['source_specific']['top_domains'].items())[:3]:
                print(f"  - {domain}: {count}")
        
        # Vorbereiten für Monetarisierung
        browser_monetization = ocean_data.prepare_data_for_monetization(browser_data, 'browser', 'medium')
        print("\n=== Browser-Daten Monetarisierung ===")
        print(f"Geschätzter Wert: {browser_monetization['metadata']['estimated_value']:.2f} OCEAN")
        print(f"Datenschutzniveau: {browser_monetization['metadata']['privacy_level']}")
        print(f"Erhaltene Felder: {len(browser_monetization['anonymized_data'].columns)} (von {browser_monetization['metadata']['original_field_count']})")
    
    if smartwatch_data_result['status'] == 'success' and smartwatch_data_result['data'] is not None:
        smartwatch_data = smartwatch_data_result['data']
        logger.info(f"Smartwatch-Daten erfolgreich abgerufen: {len(smartwatch_data)} Datensätze")
        
        # Analysiere Smartwatch-Daten
        smartwatch_analysis = ocean_data.analyze_data_source(smartwatch_data, 'smartwatch')
        print("\n=== Smartwatch-Datenanalyse ===")
        print(f"Datensätze: {smartwatch_analysis['statistics']['record_count']}")
        
        if 'source_specific' in smartwatch_analysis['analyses']:
            if 'heart_rate' in smartwatch_analysis['analyses']['source_specific']:
                hr = smartwatch_analysis['analyses']['source_specific']['heart_rate']
                print(f"Herzfrequenz: Ø {hr['mean']:.1f} (Min: {hr['min']}, Max: {hr['max']})")
            
            if 'steps' in smartwatch_analysis['analyses']['source_specific']:
                steps = smartwatch_analysis['analyses']['source_specific']['steps']
                print(f"Schritte: Gesamt {steps['total']}, Ø {steps['daily_average']:.1f} pro Tag")
        
        # Vorbereiten für Monetarisierung
        smartwatch_monetization = ocean_data.prepare_data_for_monetization(smartwatch_data, 'smartwatch', 'high')
        print("\n=== Smartwatch-Daten Monetarisierung ===")
        print(f"Geschätzter Wert: {smartwatch_monetization['metadata']['estimated_value']:.2f} OCEAN")
        print(f"Datenschutzniveau: {smartwatch_monetization['metadata']['privacy_level']}")
        print(f"Erhaltene Felder: {len(smartwatch_monetization['anonymized_data'].columns)} (von {smartwatch_monetization['metadata']['original_field_count']})")
    
    # Kombiniere Datenquellen für höheren Wert
    if ('anonymized_data' in browser_monetization and browser_monetization['anonymized_data'] is not None and
       'anonymized_data' in smartwatch_monetization and smartwatch_monetization['anonymized_data'] is not None):
        
        combined_data = ocean_data.combine_data_sources(
            [browser_monetization, smartwatch_monetization],
            'correlate'
        )
        
        print("\n=== Kombinierte Daten ===")
        print(f"Kombinationstyp: {combined_data['metadata']['combination_type']}")
        print(f"Geschätzter Wert: {combined_data['metadata']['estimated_value']:.2f} OCEAN")
        print(f"Datensätze: {combined_data['metadata']['record_count']}")
        print(f"Felder: {combined_data['metadata']['field_count']}")
    
    # Ocean Tokenisierung (simuliert)
    ocean_asset = ocean_data.prepare_for_ocean_tokenization(browser_monetization)
    tokenization_result = ocean_data.tokenize_with_ocean(ocean_asset)
    
    print("\n=== Ocean Protocol Tokenisierung ===")
    if tokenization_result.get('success', False):
        print(f"Tokenisierung erfolgreich: {tokenization_result['token_symbol']} ({tokenization_result['token_name']})")
        print(f"Token-Adresse: {tokenization_result['token_address']}")
        print(f"Token-Preis: {tokenization_result['token_price']:.2f} OCEAN")
        print(f"Marketplace URL: {tokenization_result['marketplace_url']}")
    else:
        print(f"Tokenisierung fehlgeschlagen: {tokenization_result.get('error', 'Unbekannter Fehler')}")
    
    logger.info("Demo-Modus beendet")

def main():
    """Hauptfunktion für die OceanData-Plattform"""
    parser = argparse.ArgumentParser(description="OceanData Platform")
    parser.add_argument('--env', choices=['dev', 'test', 'prod'], default=None,
                      help='Umgebung (dev, test, prod)')
    parser.add_argument('--server', action='store_true',
                      help='Web-Server starten')
    parser.add_argument('--demo', action='store_true',
                      help='Demo-Modus ausführen')
    args = parser.parse_args()
    
    # Konfiguration laden
    config = get_config(args.env)
    
    # Logging einrichten
    setup_logging(config)
    
    logger.info(f"OceanData wird gestartet in {config['env']}-Umgebung")
    
    # OceanData-Instanz erstellen
    ocean_data = create_ocean_data_instance(config)
    
    # Ausführungsmodus bestimmen
    if args.server:
        run_web_server(config, ocean_data)
    elif args.demo:
        run_demo_mode(ocean_data)
    else:
        # Standardverhalten: Demo-Modus ausführen, wenn keine Option angegeben
        run_demo_mode(ocean_data)

if __name__ == "__main__":
    main()
