# OceanData

Eine modulare Plattform zur Erfassung, Analyse und Monetarisierung von Benutzerdaten mit Hilfe von Ocean Protocol.

## Über OceanData

OceanData ist eine umfassende Lösung zur sicheren und privatsphärengerechten Monetarisierung von Benutzerdaten. Die Plattform ermöglicht es Nutzern, Daten aus verschiedenen Quellen (Browser, Smartwatch, Kalender, etc.) zu sammeln, zu analysieren und mittels Ocean Protocol zu tokenisieren und zu monetarisieren.

### Hauptfunktionen

- **Datenerfassung**: Unterstützung für diverse Datenquellen wie Browser, Smartwatches, Kalender, Social Media und mehr
- **KI-gestützte Analyse**: Anomalieerkennung, Zeitreihenanalyse und datenquellenspezifische Analysen
- **Datenschutz**: Implementierung von Compute-to-Data für sensible Daten
- **Monetarisierung**: Integration mit Ocean Protocol für Tokenisierung und Handeln von Daten

## Architektur

Die Plattform ist modular aufgebaut und besteht aus folgenden Hauptkomponenten:

1. **Datenerfassungsmodule**: Konnektoren für verschiedene Datenquellen
2. **Datenintegrationsschicht**: Vereinheitlichung heterogener Daten
3. **Analytics-Module**: KI-gestützte Datenanalyse
4. **Datenschutzmodule**: Anonymisierung und Compute-to-Data
5. **Blockchain/Tokenisierungsmodule**: Integration mit Ocean Protocol
6. **Marktplatz/Frontend**: Benutzeroberfläche (optional)

## Erste Schritte

### Installation

```bash
# Repository klonen
git clone https://github.com/yourusername/oceandata.git
cd oceandata

# Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Konfiguration

Die Konfiguration erfolgt über die Datei `oceandata/config.py`. Sie können die Umgebung über die Umgebungsvariable `OCEAN_DATA_ENV` oder beim Start der Anwendung festlegen:

```bash
# Umgebungsvariable setzen
export OCEAN_DATA_ENV=dev

# Oder bei Start angeben
python app.py --env dev
```

### Demo-Modus

Führen Sie den Demo-Modus aus, um die Funktionen der Plattform mit simulierten Daten zu testen:

```bash
python app.py --demo
```

### Web-Server starten

Starten Sie den eingebauten Web-Server, um die Plattform über eine Weboberfläche zu nutzen:

```bash
python app.py --server
```

## Verwendung

### Datenquellen verbinden

```python
from oceandata.data_integration.connectors.browser_connector import BrowserDataConnector
from oceandata.data_integration.connectors.smartdevice_connector import SmartwatchDataConnector

# Konnektoren erstellen
browser_connector = BrowserDataConnector("user123", "chrome")
smartwatch_connector = SmartwatchDataConnector("user123", "fitbit")

# Daten abrufen
browser_data = browser_connector.get_data()
smartwatch_data = smartwatch_connector.get_data()
```

### Datenanalyse

```python
from oceandata.core.ocean_data_ai import OceanDataAI

# OceanDataAI-Instanz erstellen
ocean_data = OceanDataAI()

# Daten analysieren
analysis_result = ocean_data.analyze_data_source(browser_data['data'], 'browser')
```

### Daten monetarisieren

```python
# Daten für Monetarisierung vorbereiten
monetization_result = ocean_data.prepare_data_for_monetization(browser_data['data'], 'browser', 'medium')

# Für Ocean Protocol vorbereiten
ocean_asset = ocean_data.prepare_for_ocean_tokenization(monetization_result)

# Tokenisieren
tokenization_result = ocean_data.tokenize_with_ocean(ocean_asset)
```

## Entwicklungsanleitung

### Neue Datenquelle hinzufügen

1. Erstellen Sie eine neue Konnektor-Klasse in `oceandata/data_integration/connectors/`
2. Erweitern Sie die abstrakte Klasse `DataSource`
3. Implementieren Sie die Methoden `connect()` und `fetch_data()`
4. Fügen Sie gegebenenfalls eine `extract_features()`-Methode hinzu

Beispiel:

```python
from oceandata.data_integration.base import DataSource, DataCategory, PrivacyLevel

class MyCustomConnector(DataSource):
    def __init__(self, user_id, custom_param):
        super().__init__(f"custom_{custom_param}", user_id, DataCategory.CUSTOM)
        self.custom_param = custom_param
        
    def connect(self):
        # Implementierung
        return True
        
    def fetch_data(self):
        # Implementierung
        return data_frame
```

### Tests ausführen

```bash
# Alle Tests ausführen
python -m tests.run_tests

# Spezifische Tests ausführen
python -m tests.run_tests --unit
python -m tests.run_tests --integration
```

## Projektstruktur

```
oceandata/
├── data_integration/           # Datenintegration und -erfassung
│   ├── base.py                 # Abstrakte Basisklassen
│   ├── connectors/             # Datenquellenkonnektoren
│   ├── privacy/                # Datenschutzmodule
│   └── transforms/             # Datentransformationen
├── analytics/                  # Datenanalyse
│   ├── models/                 # Analysemodelle
│   ├── preprocessing/          # Datenvorverarbeitung
│   └── visualization/          # Visualisierungsmodule
├── privacy/                    # Datenschutzmodule
│   └── compute_to_data.py      # Compute-to-Data-Manager
├── blockchain/                 # Blockchain-Integration
│   ├── ocean_integration.py    # Ocean Protocol Integration
│   ├── services/               # Blockchain-Dienste
│   └── contracts/              # Smart Contracts
├── marketplace/                # Marktplatz-Frontend
│   ├── frontend/               # Frontend-Code
│   └── backend/                # Backend-API
├── core/                       # Kernmodule
│   └── ocean_data_ai.py        # Hauptanwendungsklasse
├── config.py                   # Konfigurationsmodul
├── app.py                      # Hauptanwendung
└── server.py                   # Web-Server (optional)
```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe die [LICENSE](LICENSE) Datei für Details.

## Mitwirkende

- [Ihr Name](https://github.com/yourusername)
