"""
OceanData - Vollständige Demo

Diese Demo zeigt den gesamten Workflow von OceanData:
1. Datenerfassung aus verschiedenen Quellen
2. Datenintegration und -analyse
3. Tokenisierung mit Ocean Protocol
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List

# Importiere OceanData-Module
import sys
import os

# Füge das Verzeichnis zum Pythonpfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from oceandata.src.data_integration.base import (
    DataIntegrator, BrowserDataConnector, SmartwatchDataConnector, 
    IoTDeviceConnector, DataAnalyzer, PrivacyLevel, DataCategory
)
from oceandata.src.blockchain.tokenization import OceanDataTokenizer

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OceanData.Demo")

class OceanDataDemo:
    """
    Vollständige Demo für OceanData.
    """
    
    def __init__(self):
        """
        Initialisiert die OceanData-Demo.
        """
        self.user_id = f"user_{int(time.time())}"
        self.integrator = None
        self.analyzer = None
        self.tokenizer = None
        
        logger.info(f"OceanData-Demo initialisiert für Benutzer {self.user_id}")
    
    def setup(self):
        """
        Richtet die Demo ein.
        """
        # Erstelle Datenintegrator
        self.integrator = DataIntegrator(self.user_id)
        
        # Erstelle Datenanalyzer
        self.analyzer = DataAnalyzer(self.integrator)
        
        # Erstelle Tokenizer
        self.tokenizer = OceanDataTokenizer()
        
        logger.info("OceanData-Demo eingerichtet")
    
    def add_data_sources(self):
        """
        Fügt Datenquellen hinzu.
        """
        # Füge Browser-Datenquelle hinzu
        browser_connector = BrowserDataConnector(self.user_id, 'chrome')
        self.integrator.add_connector(browser_connector)
        
        # Füge Smartwatch-Datenquelle hinzu
        smartwatch_connector = SmartwatchDataConnector(self.user_id, 'fitbit')
        self.integrator.add_connector(smartwatch_connector)
        
        # Füge IoT-Datenquelle hinzu
        thermostat_connector = IoTDeviceConnector(self.user_id, 'thermostat')
        self.integrator.add_connector(thermostat_connector)
        
        logger.info("Datenquellen hinzugefügt")
    
    def collect_and_analyze_data(self, privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM):
        """
        Sammelt und analysiert Daten.
        
        Args:
            privacy_level: Datenschutzniveau für die Datensammlung
        
        Returns:
            Dict: Ergebnisse der Datensammlung und -analyse
        """
        # Sammle Daten
        logger.info(f"Sammle Daten mit Datenschutzniveau {privacy_level.value}")
        integration_result = self.integrator.collect_all_data(privacy_level)
        
        if integration_result["status"] != "success":
            logger.error(f"Fehler bei der Datensammlung: {integration_result.get('message', 'Unbekannter Fehler')}")
            return {
                "success": False,
                "error": integration_result.get('message', 'Fehler bei der Datensammlung')
            }
        
        # Analysiere Daten
        logger.info("Analysiere Daten")
        analytics_result = self.analyzer.run_all_analytics()
        
        return {
            "success": True,
            "integration_result": integration_result,
            "analytics_result": analytics_result
        }
    
    def tokenize_data(self, privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM):
        """
        Tokenisiert die gesammelten Daten.
        
        Args:
            privacy_level: Datenschutzniveau für die Tokenisierung
        
        Returns:
            Dict: Ergebnis der Tokenisierung
        """
        # Bereite Daten für Tokenisierung vor
        logger.info(f"Bereite Daten für Tokenisierung mit Datenschutzniveau {privacy_level.value} vor")
        tokenization_package = self.integrator.prepare_for_tokenization(privacy_level)
        
        # Tokenisiere Daten
        logger.info("Tokenisiere Daten mit Ocean Protocol")
        tokenization_result = self.tokenizer.tokenize_data_package(tokenization_package)
        
        return {
            "success": tokenization_result.get("success", False),
            "tokenization_package": tokenization_package,
            "tokenization_result": tokenization_result
        }
    
    def run_full_demo(self, privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM):
        """
        Führt die vollständige Demo aus.
        
        Args:
            privacy_level: Datenschutzniveau für die Demo
        
        Returns:
            Dict: Ergebnisse der Demo
        """
        start_time = datetime.now()
        
        # Richte Demo ein
        self.setup()
        
        # Füge Datenquellen hinzu
        self.add_data_sources()
        
        # Sammle und analysiere Daten
        data_result = self.collect_and_analyze_data(privacy_level)
        if not data_result["success"]:
            return {
                "success": False,
                "error": data_result.get("error", "Fehler bei der Datensammlung und -analyse"),
                "duration_seconds": (datetime.now() - start_time).total_seconds()
            }
        
        # Tokenisiere Daten
        token_result = self.tokenize_data(privacy_level)
        
        # Erstelle Zusammenfassung
        summary = {
            "user_id": self.user_id,
            "privacy_level": privacy_level.value,
            "data_sources": list(self.integrator.integrated_data.keys()),
            "total_records": self.integrator.metadata.get("total_records", 0),
            "estimated_value": self.integrator.metadata.get("estimated_total_value", 0.0),
            "analyzed_sources": self.analyzer.metadata.get("analyzed_sources", []),
            "tokenization_success": token_result.get("success", False),
            "asset_id": token_result.get("tokenization_result", {}).get("asset_id", None),
            "marketplace_url": token_result.get("tokenization_result", {}).get("marketplace_url", None),
            "duration_seconds": (datetime.now() - start_time).total_seconds()
        }
        
        return {
            "success": token_result.get("success", False),
            "data_result": data_result,
            "token_result": token_result,
            "summary": summary
        }
    
    def print_demo_results(self, results: Dict[str, Any]):
        """
        Gibt die Ergebnisse der Demo aus.
        
        Args:
            results: Ergebnisse der Demo
        """
        if not results["success"]:
            print("\n❌ DEMO FEHLGESCHLAGEN")
            print(f"Fehler: {results.get('error', 'Unbekannter Fehler')}")
            return
        
        summary = results["summary"]
        token_result = results["token_result"]["tokenization_result"]
        
        print("\n🌊 OCEANDATA VOLLSTÄNDIGE DEMO")
        print("=============================")
        
        print(f"\n📊 ZUSAMMENFASSUNG")
        print(f"Benutzer-ID: {summary['user_id']}")
        print(f"Datenquellen: {', '.join(summary['data_sources'])}")
        print(f"Gesamtdatensätze: {summary['total_records']}")
        print(f"Geschätzter Wert: {summary['estimated_value']:.2f} OCEAN")
        print(f"Datenschutzniveau: {summary['privacy_level']}")
        print(f"Analysierte Quellen: {len(summary['analyzed_sources'])}")
        print(f"Verarbeitungszeit: {summary['duration_seconds']:.2f} Sekunden")
        
        print(f"\n🔍 DATENANALYSE")
        analytics = results["data_result"]["analytics_result"]["results"]
        
        if "browser" in analytics:
            browser_sources = list(analytics["browser"].keys())
            if browser_sources:
                browser_source = browser_sources[0]
                browser_data = analytics["browser"][browser_source]
                print("\n🌐 Browser-Daten:")
                print(f"  Datensätze: {browser_data['statistics']['record_count']}")
                
                if "website_analysis" in browser_data and "top_domains" in browser_data["website_analysis"]:
                    print("  Top-Domains:")
                    for domain, count in list(browser_data["website_analysis"]["top_domains"].items())[:3]:
                        print(f"    - {domain}: {count}")
                
                if "usage_analysis" in browser_data and "avg_duration" in browser_data["usage_analysis"]:
                    print(f"  Durchschnittliche Nutzungsdauer: {browser_data['usage_analysis']['avg_duration']:.1f} Sekunden")
        
        if "health" in analytics:
            health_sources = list(analytics["health"].keys())
            if health_sources:
                health_source = health_sources[0]
                health_data = analytics["health"][health_source]
                print("\n❤️ Gesundheitsdaten:")
                print(f"  Datensätze: {health_data['statistics']['record_count']}")
                
                if "heart_rate" in health_data and health_data["heart_rate"]:
                    hr = health_data["heart_rate"]
                    print(f"  Herzfrequenz: Ø {hr['mean']:.1f} (Min: {hr['min']}, Max: {hr['max']})")
                
                if "steps" in health_data and health_data["steps"]:
                    steps = health_data["steps"]
                    print(f"  Schritte: Gesamt {steps['total']}, Ø {steps['daily_average']:.1f} pro Tag")
        
        if "iot" in analytics:
            iot_sources = list(analytics["iot"].keys())
            if iot_sources:
                iot_source = iot_sources[0]
                iot_data = analytics["iot"][iot_source]
                print(f"\n🏠 IoT-Daten ({iot_data['statistics']['device_type']}):")
                print(f"  Datensätze: {iot_data['statistics']['record_count']}")
                
                if "device_analysis" in iot_data:
                    device = iot_data["device_analysis"]
                    if "temperature" in device:
                        print(f"  Temperatur: Ø {device['temperature']['mean']:.1f}°C (Min: {device['temperature']['min']}°C, Max: {device['temperature']['max']}°C)")
        
        print(f"\n💰 TOKENISIERUNG")
        print(f"Asset-ID: {token_result['asset_id']}")
        print(f"Token-Adresse: {token_result['token_address']}")
        print(f"Preis: {token_result['price']} OCEAN")
        print(f"Netzwerk: {token_result['network']}")
        print(f"Marketplace URL: {token_result['marketplace_url']}")
        
        print(f"\n✅ DEMO ERFOLGREICH ABGESCHLOSSEN!")


def run_demo():
    """
    Führt die OceanData-Demo aus.
    """
    print("OceanData - Vollständige Demo")
    print("=============================")
    print("Starte Demo...")
    
    # Erstelle und führe Demo aus
    demo = OceanDataDemo()
    results = demo.run_full_demo(PrivacyLevel.MEDIUM)
    
    # Zeige Ergebnisse
    demo.print_demo_results(results)


if __name__ == "__main__":
    run_demo()