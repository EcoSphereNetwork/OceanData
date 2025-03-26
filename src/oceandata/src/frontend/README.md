# OceanData Frontend

Dieses Verzeichnis enthält die Frontend-Anwendung für OceanData, eine Plattform zur Datenmonetarisierung mit Ocean Protocol.

## Funktionen

- **Dashboard**: Übersicht über Datenquellen, Analysen und tokenisierte Datensätze
- **Datenquellenmanagement**: Hinzufügen, Konfigurieren und Überwachen von Datenquellen
- **Analysevisualisierungen**: Interaktive Diagramme und Grafiken für Datenanalysen
- **Tokenisierungsworkflow**: Benutzerfreundliche Schnittstelle für den Tokenisierungsprozess
- **Marktplatzintegration**: Anzeige und Verwaltung von tokenisierten Datensätzen

## Technologien

- **React**: Frontend-Framework
- **TypeScript**: Typsicheres JavaScript
- **Vite**: Build-Tool
- **Tailwind CSS**: Styling
- **Smolitux-UI**: UI-Komponentenbibliothek
- **Recharts**: Diagrammbibliothek
- **Flask**: Backend-Server für die Auslieferung der Anwendung

## Erste Schritte

### Voraussetzungen

- Node.js (v14 oder höher)
- npm (v6 oder höher)
- Python (v3.8 oder höher)

### Installation und Start

1. Führen Sie das Build- und Start-Skript aus:

```bash
./build_and_run.sh
```

Dieses Skript führt folgende Schritte aus:
- Installiert Node.js-Abhängigkeiten
- Baut die Frontend-Anwendung
- Installiert Python-Abhängigkeiten
- Startet den Frontend-Server

2. Öffnen Sie die Anwendung in Ihrem Browser:

```
http://localhost:3000
```

### Entwicklung

Für die Entwicklung können Sie den Entwicklungsserver starten:

```bash
npm run dev
```

## Projektstruktur

```
frontend/
├── src/                  # Quellcode
│   ├── components/       # Wiederverwendbare Komponenten
│   ├── context/          # React Context für globalen Zustand
│   ├── hooks/            # Benutzerdefinierte React Hooks
│   ├── pages/            # Seitenkomponenten
│   ├── utils/            # Hilfsfunktionen
│   ├── App.tsx           # Hauptanwendungskomponente
│   ├── main.tsx          # Einstiegspunkt
│   └── server.py         # Flask-Server für die Auslieferung
├── public/               # Statische Dateien
├── index.html            # HTML-Template
├── package.json          # npm-Konfiguration
├── tsconfig.json         # TypeScript-Konfiguration
├── vite.config.ts        # Vite-Konfiguration
└── build_and_run.sh      # Build- und Start-Skript
```

## Integration mit OceanData Backend

Die Frontend-Anwendung kommuniziert mit dem OceanData Backend über eine REST-API. Die API-Endpunkte sind in der Datei `src/utils/api.ts` definiert.

## Anpassung

Sie können die Anwendung anpassen, indem Sie die Konfigurationsdateien bearbeiten:

- **Styling**: Bearbeiten Sie `tailwind.config.js` für Designanpassungen
- **Routing**: Bearbeiten Sie `src/App.tsx` für Routenanpassungen
- **API-Endpunkte**: Bearbeiten Sie `src/utils/api.ts` für API-Anpassungen