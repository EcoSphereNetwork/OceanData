#!/bin/bash

# Verzeichnis zum Frontend-Ordner wechseln
cd "$(dirname "$0")"

# Abhängigkeiten installieren
echo "Installiere Abhängigkeiten..."
npm install

# Frontend bauen
echo "Baue Frontend-Anwendung..."
npm run build

# Python-Abhängigkeiten installieren
echo "Installiere Python-Abhängigkeiten..."
pip install flask flask-cors

# Server starten
echo "Starte Frontend-Server..."
python src/server.py