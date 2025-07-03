# AGENTS

Diese Datei beschreibt den Einsatz von spezialisierten Codex-Agenten zur Weiterentwicklung des OceanData-Projekts. OceanData ist eine dezentrale Daten-Monetarisierungsplattform mit modularer Architektur. Die Agenten unterstützen Entwicklungsschritte in Bereichen wie Datenerfassung, Blockchain-Integration, Analyse und Visualisierung sowie Qualitätssicherung und Deployment.

## Daten-Integrations-Agent (API-Integration)

Rolle: Implementiert Schnittstellen zu vielfältigen Datenquellen (Browser, Gesundheits-Apps, IoT, Social Media, Streaming etc.). Bereitet die Rohdaten auf und wendet Datenschutz-Transformationen an. Unterstützt den Aufbau des einheitlichen Datenmodells und Echtzeit-Streaming.

Aufgaben: Entwickelt und pflegt Connector-Module (z. B. browser_connector.py, calendar_connector.py, socialmedia_connector.py in src/data-integration/connectors). Erstellt Daten-Transformations- und Anonymisierungsfunktionen (data_transform.py, anonymizer.py, encryption.py). Implementiert Compute-to-Data (C2D)-Funktionen zum Analysieren verschlüsselter Daten.

Komponenten im Repo: Alles unter src/data-integration/ (Ordner connectors, transforms, privacy sowie base.py). Außerdem ggf. Skripte unter scripts/data_import/.

Technologien: Python 3.x, REST-APIs, ggf. Streaming-Plattformen (Kafka). Beachtet Sicherheitsstandards (z.B. DSGVO) wie in der Datenschutzkonzeption beschrieben.

Beispiel: In der Projektbeschreibung wird die Datenerfassung explizit genannt: „Entwicklung von API-Schnittstellen für verschiedene Datenquellen“.


## Datenanalyse-Agent (KI/ML)

Rolle: Entwickelt und trainiert Machine-Learning-Modelle zur Mustererkennung und Vorhersage (Anomalieerkennung, semantische Analyse, prädiktive Modelle, Datensynthese). Nutzt die integrierten Daten aus dem Data-Integration-Layer zur Wertschöpfung.

Aufgaben: Implementiert Algorithmen in src/analytics/models/ (z.B. anomaly_detector.py, semantic_analyzer.py, predictive_modeler.py, data_synthesizer.py). Baut Datenvorverarbeitungs-Pipelines (data_cleaner.py, feature_extraction.py). Generiert Visualisierungen aus den Analyseergebnissen (anomaly_plots.py, trend_plots.py).

Komponenten im Repo: src/analytics/ einschließlich Unterverzeichnisse models, preprocessing, visualization. Unit-Tests in tests/unit/analytics/.

Technologien: Python (TensorFlow, PyTorch, Scikit-Learn, Pandas usw.). Nutzt Datenschutztechniken (z.B. differenzieller Datenschutz oder k-Anonymität) in Abstimmung mit dem Data-Integration-Agenten.

Beispiel: Die Spezifikation nennt „Machine-Learning-Modelle zur Erkennung wertvoller Datenmuster“, was diese Agentenarbeit konkretisiert.


## Visualisierungs- und Frontend-Agent

Rolle: Entwickelt die Benutzeroberfläche für den Datenmarktplatz und Visualisierungen zur Analyse. Realisiert Dashboards und interaktive Komponenten für Datenbesitzer und Käufer.

Aufgaben: Implementiert React-Komponenten und Seiten im Ordner src/marketplace/frontend/ (z.B. DataTokenizationDashboard.jsx, AnalysisVisualizer.jsx, Seiten wie Marketplace.jsx, Wallet.jsx). Sorgt für ansprechendes UI/UX-Design, Einhaltung des Style-Guides und Barrierefreiheit.

Komponenten im Repo: Frontend-Code in src/marketplace/frontend/ (Ordner components und pages), sowie Backend-Anbindung über src/marketplace/app.py und Services (z.B. im Ordner backend/services). Unit-Tests für React-Komponenten liegen evtl. in tests/components/.

Technologien: JavaScript/TypeScript mit React.js. Nutzt Smolitux-UI oder ein vergleichbares UI-Framework für Konsistenz. Entwickelt visualisierte Auswertungen (Charts, Graphen) ggf. mit Bibliotheken wie D3.js oder Chart.js.

Beispiel: Die Projektbeschreibung fordert „intuitive Benutzeroberfläche“ und Dashboards zur Überwachung von Datenverkäufen. Der Visualisierungs-Agent setzt dies in Code um (siehe DataTokenizationDashboard.jsx etc. in src/marketplace/frontend).


## Blockchain-Agent (Smart Contracts)

Rolle: Implementiert die Blockchain- und Tokenisierungsebene der Plattform. Erstellt Smart Contracts für Daten-Token (DataToken) und Compute-to-Data (C2D). Sorgt für sichere Transaktionen und Wallet-Integration.

Aufgaben: Entwickelt Solidity-Verträge (DataToken.sol, ComputeToData.sol in src/blockchain/contracts/). Baut Backend-Dienste zur Kommunikation mit Ocean Protocol und Web3 (z.B. ocean_api.py, token_service.py). Testet Smart Contracts mit Truffle/Hardhat. Arbeitet eng mit Frontend-Agent (Wallet-Interaktion) und Data-Integration-Agent (Erstellung von C2D-Aufträgen) zusammen.

Komponenten im Repo: Alles unter src/blockchain/: contracts und services. Unit-Tests in tests/unit/blockchain/.

Technologien: Solidity, Ethereum/Ocean Protocol. Backendentwicklung in Python oder Node.js je nach Modul. Befolgt Smart-Contract-Sicherheitsstandards (Audits, OpenZeppelin-Bibliotheken, klare Zugriffskontrollen).

Beispiel: Im Pflichtenheft steht „Integration des Ocean Protocol für die Erstellung von Datentokens“. Der Blockchain-Agent setzt dies technisch um.


## QA-Agent (Qualitätssicherung)

Rolle: Sichert die Software-Qualität durch Testing und Reviews. Kontinuierliche Überprüfung des Codes auf Fehlerfreiheit und Konformität.

Aufgaben: Entwickelt und pflegt Testfälle (Unit- und Integrationstests in tests/), z. B. unter tests/unit/data_integration/, tests/unit/analytics/ etc. Führt automatisierte Tests durch (z.B. mit pytest, Jest). Überwacht Code-Coverage und Performance-Tests (tests/performance/). Prüft Pull Requests auf Stil (z.B. PEP8, ESLint) und Funktionalität.

Komponenten im Repo: Die gesamte Test-Struktur tests/ ist relevant (Unit-Tests, tests/integration/, tests/components/ usw.). Außerdem CI-Konfigurationsdateien (z.B. GitHub Actions) sobald vorhanden.

Technologien: pytest für Python, Jest/Mocha für JavaScript, Linter (flake8, ESLint). Code-Coverage-Tools. Regelmäßige Code Reviews im Team.

Beispiel: Der Test-Framework-Teil des Generators zeigt viele Testdateien in tests/ vor (z.B. test_connectors.py, test_anomaly_detector.py usw.), die der QA-Agent implementiert und erweitert.


## CI/CD-Agent (DevOps)

Rolle: Errichtet und verwaltet automatisierte Build-, Test- und Deployment-Pipelines. Sorgt für Continuous Integration und Delivery.

Aufgaben: Erstellt CI-Workflows (z.B. GitHub Actions) für Builds, Tests und Deployment. Konfiguriert Docker-Umgebungen (Dockerfile, docker-compose.yml) und Deployment-Skripte (scripts/deployment/deploy.sh, update.sh). Automatisiert das Setup (npm run setup, pip install, copy .env). Koordiniert Staging- und Produktionsumgebungen (z.B. mit Kubernetes).

Komponenten im Repo: Skripte in scripts/deployment/, das Dockerfile, docker-compose.yml, .nvmrc und .env.example. GitHub Actions (sofern eingerichtet) oder andere CI-Konfigurationsdateien.

Technologien: Docker, Kubernetes, CI-Tools (GitHub Actions, Jenkins, GitLab CI). Deployment auf Cloud-Plattformen (AWS, GCP) gemäß Spezifikation. Branching- und Release-Strategien (z.B. Gitflow, Trunk-Based Development).

Beispiel: Der Generator legt u.a. docker-compose.yml und Deployment-Skripte an, die der CI/CD-Agent mit Leben füllt.


## Dokumentations-Agent

Rolle: Pflegt die Projektdokumentation, macht sie aktuell und zugänglich. Erstellt Benutzer- und Entwicklertutorials.

Aufgaben: Schreibt und strukturiert Inhalte in docs/ (z. B. docs/docs/OceanData/Projektbeschreibung.md, Anforderungsanalyse.md etc.). Pflegt README.md und interne Wikis. Dokumentiert APIs, Datenmodelle und Architektur. Baut die Dokumentation (z.B. mit Docusaurus) und stellt sie Online (GitHub Pages, Wiki).

Komponenten im Repo: Das Verzeichnis docs/, insbesondere Unterverzeichnisse docs/static/img/ und docs/docs/OceanData/ mit Markdown-Dateien. Außerdem CONTRIBUTING.md, LICENSE und CHANGELOG falls vorhanden.

Technologien: Markdown, ggf. Docusaurus oder MkDocs. Grafiken/Schemas (Tools: draw.io, PlantUML). Vorlage und Styleguide für einheitliche Dokumentation.

Beispiel: Im Projekt sind viele DOC-Dateien vorgerüstet (Projektbeschreibung.md, Datenerfassung.md etc.); der Dokumentations-Agent ergänzt und aktualisiert diese.


## Koordination der Agents

Zur effizienten Arbeitsteilung empfiehlt sich eine hierarchische Agenten-Architektur: Ein übergeordneter Supervisor-Agent verteilt Aufgaben an spezialisierte Worker-Agents (z.B. Integrations-, Analyse-, Frontend-Agent). Regelmäßige Synchronisation (Daily Standup, gemeinsame Retrospektiven) stellt sicher, dass z.B. der Daten-Integrations-Agent neue Datenmodelle liefert, auf deren Basis der Analyse-Agent arbeitet. Abhängigkeiten (z.B. Frontend benötigt neue API-Endpunkte) werden in Backlog-Tickets erfasst und priorisiert. Jeder Agent sollte klare Verantwortungsbereiche haben, bei Bedarf aber teamübergreifend kommunizieren (z.B. Datenschutz- und Blockchain-Agent abstimmen C2D-Implementierung).

## Entwicklungsprozesse und Standards

Das OceanData-Projekt nutzt eine modulare Technologie-Stack: Python für Backend/Datenverarbeitung, Node.js/Express für Backend-Services, React für Frontend, Solidity/Ethereum für Smart Contracts. Empfohlene Praktiken sind:

Versionskontrolle: Git (Pull Requests, Code Reviews). Klare Branching-Strategie (z.B. main, dev, Feature-Branches).

Qualitätskontrolle: Unit- und Integrationstests (pytest/Jest). Statische Code-Analyse (flake8, ESLint) und Code Coverage.

CI/CD: Automatisierte Pipelines zur Prüfung jeder Änderung. Containerisierung mit Docker und Deployment via Kubernetes oder vergleichbare Plattform.

Dokumentation: Laufend aktualisierte Architektur- und API-Dokumentation. Kommentierung des Codes (Docstrings). Verwendung eines einheitlichen Style-Guides.

Sicherheit: Sichere Entwicklungsrichtlinien (OWASP, Best Practices für Solidity). Kontinuierliche Überprüfung von Abhängigkeiten (Dependabot o. Ä.).

Prozess: Agile Methodik (Scrum/Kanban), definiertes Issue- und Release-Management, tägliche Standups.


Durch diese Maßnahmen wird sichergestellt, dass die Agents ihre Teilaufgaben nachvollziehbar und konsistent umsetzen können. So kann z. B. jeder Agent eigenständig Komponenten im angegebenen Verzeichnis bearbeiten (z. B. src/data-integration/ für den Integrations-Agent) und alle zusammen arbeiten koordiniert am Gesamtprojekt.
