AGENTS

Diese Datei beschreibt den Einsatz von spezialisierten Codex-Agenten zur Weiterentwicklung des OceanData-Projekts. OceanData ist eine dezentrale Daten-Monetarisierungsplattform mit modularer Architektur. Die Agenten unterstützen Entwicklungsschritte in Bereichen wie Datenerfassung, Blockchain-Integration, Interoperabilität, Analyse und Visualisierung sowie Qualitätssicherung und Deployment.

Die vollständige Kompatibilität mit Ocean Protocol, DataUnion, Datalatte und Streamr ist zentrales Entwicklungsziel. Der technische Plan dazu ist dokumentiert in: 📄 docs/OCEANDATA-PLATTFORM-KOMPATIBILITÄT.md


---

Interop-Agent (Ocean & Web3-Kompatibilität)

Rolle: Setzt die vollständige Integration der Ocean Protocol SDKs, DataUnion-Logik, Datalatte-Features und Streamr-Schnittstellen um. Vereinheitlicht Tokenisierung, Datenzugriff und Monetarisierung über mehrere Web3-Protokolle hinweg.

Aufgaben:

Integration von Ocean Protocol SDKs:

Python: ocean.py – 📘 Quickstart & Docs

JavaScript: ocean.js


DataUnion:

SDK & Smart Contracts: 📁 dataunion-app Monorepo

Foundation: 🌐 dataunion.foundation


Datalatte:

Repos: 📁 datalatte-ai GitHub

Konzept & Demo: 🌐 datalatte.xyz


Streamr:

SDK: 📘 streamr-client

Netzwerk: 🌐 streamr.network


Integration von Biconomy für gasloses Minting: 📘 biconomy.io


Komponenten im Repo: src/interop/: ocean_client.py, dataunion_service.py, streamr_connector.py, datalatte_adapter.py, mit Tests unter tests/unit/interop/

Technologien: Python, Web3.py, ocean.py, ocean.js, Solidity, Streamr SDK, REST/WebSocket APIs, Biconomy Relayer

Beispiel: Reale Ocean-DID-Registrierung, Live-Streams via Streamr, Gasless Onboarding – umgesetzt gemäß OCEANDATA-PLATTFORM-KOMPATIBILITÄT.md


---

Daten-Integrations-Agent (API-Integration)

Rolle: Implementiert Schnittstellen zu vielfältigen Datenquellen (Browser, Gesundheits-Apps, IoT, Social Media, Streaming etc.). Bereitet die Rohdaten auf und wendet Datenschutz-Transformationen an.

Aufgaben:

Module in src/data-integration/connectors/ und transforms/

Privacy-Funktionen in privacy/

C2D-Prozesse über compute_handler.py


Technologien: Python 3.x, REST-APIs, ggf. Streamr/GraphQL/Kafka

Beispiel: Entwicklung von Connectors für Streamingverlauf, Kalenderdaten, Sensordaten


---

Datenanalyse-Agent (KI/ML)

Rolle: Entwickelt und trainiert ML-Modelle zur Erkennung und Vorhersage wertvoller Datenmuster.

Aufgaben:

Modelle in src/analytics/models/

Visualisierungen in visualization/

Nutzung gemeinsamer Preprocessing-Module mit Integration-Agent


Technologien: Python, PyTorch, Scikit-Learn, Pandas, Matplotlib

Beispiel: Modell „semantic_analyzer.py“ analysiert Textdaten auf thematische Relevanz


---

Visualisierungs- und Frontend-Agent

Rolle: Entwickelt Frontend-Komponenten für Datenmarktplatz, Tokenisierung und Analyse.

Aufgaben:

UI in src/marketplace/frontend/components/

React-Pages in pages/

Styleguide-Anpassung mit Smolitux-UI


Technologien: React, TypeScript, Tailwind, Chart.js/D3

Beispiel: DataTokenizationDashboard.jsx stellt Status und Verlauf von Datenverkäufen dar


---

Blockchain-Agent (Smart Contracts)

Rolle: Implementiert Smart Contracts für DataNFT, Datatoken, C2D und Revenue-Sharing.

Aufgaben:

Solidity-Verträge unter src/blockchain/contracts/

Integration mit Web3 via ocean.py, Hardhat oder Truffle

Tokenverteilung an mehrere Contributor (DataUnion)


Technologien: Solidity, OpenZeppelin, Hardhat, Web3.py

Beispiel: DataUnionRevenue.sol regelt Verteilung bei kollektiven Verkäufen


---

QA-Agent (Qualitätssicherung)

Rolle: Führt automatische Tests, Style-Prüfungen und Reviews durch.

Aufgaben:

Teststruktur in tests/unit/, tests/integration/

Testdaten und Benchmarks in tests/data/

CI-Test-Abdeckung mit Coverage-Badge


Technologien: Pytest, Jest, flake8, ESLint, GitHub Actions

Beispiel: Test-Framework deckt alle Schnittstellen der Interop-Module ab


---

CI/CD-Agent (DevOps)

Rolle: Automatisiert Setup, Tests und Deployment via Pipelines

Aufgaben:

GitHub Actions Workflows (.github/workflows/)

Docker-Skripte unter scripts/deployment/

Environments für Staging & Production via k8s/ oder compose/


Technologien: Docker, Kubernetes, GitHub Actions, Gitflow

Beispiel: deploy.sh installiert OceanData samt Abhängigkeiten automatisch


---

Dokumentations-Agent

Rolle: Verfasst, strukturiert und veröffentlicht technische Doku

Aufgaben:

Dateien unter docs/

Links und Guides zu allen SDKs und externen Repos:

Ocean Protocol Docs 
https://docs.oceanprotocol.com

Ocean.py auf GitHub
https://github.com/oceanprotocol
https://github.com/oceanprotocol/pdr-backend
https://github.com/oceanprotocol/ocean.js
https://github.com/oceanprotocol/docs
https://github.com/oceanprotocol/ocean.py

DataUnion Foundation
https://github.com/DataUnion-app
https://github.com/DataUnion-app/du-ocean-contracts
https://github.com/DataUnion-app/market
https://github.com/DataUnion-app/Squid


Streamr Github
https://github.com/streamr-dev

Datalatte GitHub

Biconomy SDK



Technologien: Markdown, Docusaurus, PlantUML, Mermaid

Beispiel: docs/interop/ocean_integration.md enthält vollständige SDK-Setup-Anleitung


---

Koordination der Agents

Ein Supervisor-Agent verwaltet Aufgabenverteilung, synchronisiert Spezifikationen, validiert Kompatibilitäten und dokumentiert Querverbindungen (z. B. Streamr als Transportlayer, Ocean als Monetarisierungsschicht).


---

Entwicklungsprozesse und Standards

Stack: Python, React/TypeScript, Solidity, Web3, Ocean SDKs

Standards:

Branching: main, dev, Feature-Branches

Tests: pytest/Jest, Coverage >90 %

Security: OWASP, Audit-Ready Contracts

Prozess: Scrum-basiert, Daily Standups, Agile Board (z. B. GitHub Projects)


Durch diese Struktur arbeiten die Agenten autonom, aber integriert – mit klaren Übergabepunkten, geteilten Standards und gemeinsamer Zielsetzung: die vollumfängliche, modulare Interoperabilität von OceanData im Web3-Datenökosystem.

