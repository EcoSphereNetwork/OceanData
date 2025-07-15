# OceanData Architektur

Dieses Dokument gibt einen kurzen Überblick über die wichtigsten Module der Plattform.

## Kernbereiche

- **src/interop** – Schnittstellen zu Ocean Protocol, DataUnion, Datalatte und Streamr. Die Klassen arbeiten wahlweise im Mock-Modus oder mit echten Endpunkten.
- **ocean_sdk** – vereinfachte Helper-Funktionen und CLI-Kommandos.
- **tests/unit** – Unit-Tests inklusive neuer Tests für die Interop-Module.

Diese Struktur ermöglicht es, schrittweise echte Web3-Funktionalität anzubinden und dennoch eine lauffähige Demo ohne Blockchain-Zugang bereitzustellen.
