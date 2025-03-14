#!/usr/bin/env python
"""
OceanData Test Suite Runner

Dieses Skript führt alle Tests für die OceanData-Plattform aus
und generiert einen Testbericht.
"""

import unittest
import pytest
import os
import sys
import argparse
import time
import json
from datetime import datetime
from unittest import TestLoader, TextTestRunner, TestSuite
from coverage import Coverage

# Pfade für Imports konfigurieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import der Testmodule
try:
    from tests.test_oceandata import (
        TestDataSource, TestAnomalyDetector, TestSemanticAnalyzer,
        TestPredictiveModeler, TestDataSynthesizer, TestComputeToDataManager,
        TestOceanDataAI, TestIntegrationOceanData, TestPerformanceOceanData
    )
except ImportError as e:
    print(f"Fehler beim Import der Testmodule: {e}")
    print("Stelle sicher, dass die Tests im 'tests'-Verzeichnis verfügbar sind.")
    sys.exit(1)

def parse_args():
    """Kommandozeilenargumente parsen"""
    parser = argparse.ArgumentParser(description="OceanData Test Suite Runner")
    parser.add_argument("--unit", action="store_true", help="Nur Unit-Tests ausführen")
    parser.add_argument("--integration", action="store_true", help="Nur Integrationstests ausführen")
    parser.add_argument("--performance", action="store_true", help="Nur Performance-Tests ausführen")
    parser.add_argument("--coverage", action="store_true", help="Testabdeckung berechnen")
    parser.add_argument("--html-report", action="store_true", help="HTML-Bericht generieren")
    parser.add_argument("--output", default="test_results", help="Ausgabeverzeichnis für Berichte")
    return parser.parse_args()

def run_unit_tests():
    """Unit-Tests ausführen"""
    print("=== Ausführung der Unit-Tests ===")
    loader = TestLoader()
    suite = TestSuite()
    
    # Testklassen hinzufügen
    test_cases = [
        TestDataSource,
        TestAnomalyDetector,
        TestSemanticAnalyzer,
        TestPredictiveModeler,
        TestDataSynthesizer,
        TestComputeToDataManager,
        TestOceanDataAI
    ]
    
    for test_class in test_cases:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Tests ausführen
    runner = TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_integration_tests():
    """Integrationstests ausführen"""
    print("=== Ausführung der Integrationstests ===")
    
    loader = TestLoader()
    suite = TestSuite()
    
    # Integrationstests hinzufügen
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationOceanData))
    
    # React-Komponententests mit pytest
    # Diese Tests müssen separat behandelt werden, da sie pytest verwenden
    print("React-Komponententests werden übersprungen (erfordern spezielle Testumgebung)")
    
    # Tests ausführen
    runner = TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_performance_tests():
    """Performance-Tests ausführen"""
    print("=== Ausführung der Performance-Tests ===")
    
    loader = TestLoader()
    suite = TestSuite()
    
    # Performance-Tests hinzufügen
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceOceanData))
    
    # Tests ausführen
    runner = TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_with_coverage(test_func, source_packages):
    """Tests mit Abdeckungsmessung ausführen"""
    cov = Coverage(source=source_packages)
    cov.start()
    
    result = test_func()
    
    cov.stop()
    cov.save()
    
    print("\n=== Testabdeckungsbericht ===")
    cov.report()
    
    return cov, result

def generate_html_report(coverage, output_dir):
    """HTML-Bericht für die Testabdeckung generieren"""
    os.makedirs(output_dir, exist_ok=True)
    coverage.html_report(directory=os.path.join(output_dir, 'coverage_html'))
    print(f"HTML-Abdeckungsbericht wurde in {os.path.join(output_dir, 'coverage_html')} gespeichert.")

def generate_test_summary(results, output_dir):
    """Zusammenfassung der Testergebnisse generieren"""
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "results": {}
    }
    
    for test_type, result in results.items():
        if result:
            summary["results"][test_type] = {
                "total": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
                "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
            }
    
    with open(os.path.join(output_dir, 'test_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Testzusammenfassung wurde in {os.path.join(output_dir, 'test_summary.json')} gespeichert.")

def main():
    """Hauptfunktion zum Ausführen der Tests"""
    args = parse_args()
    start_time = time.time()
    
    source_packages = [
        'oceandata.data_integration',
        'oceandata.ai',
        'oceandata.privacy',
        'oceandata.monetization'
    ]
    
    results = {}
    coverage_data = None
    
    # Bestimmen, welche Tests ausgeführt werden sollen
    run_all = not (args.unit or args.integration or args.performance)
    
    if run_all or args.unit:
        if args.coverage:
            coverage_data, results['unit'] = run_with_coverage(run_unit_tests, source_packages)
        else:
            results['unit'] = run_unit_tests()
    
    if run_all or args.integration:
        results['integration'] = run_integration_tests()
    
    if run_all or args.performance:
        results['performance'] = run_performance_tests()
    
    # Berichte generieren
    if args.html_report and coverage_data:
        generate_html_report(coverage_data, args.output)
    
    generate_test_summary(results, args.output)
    
    # Ausführungszeit ausgeben
    execution_time = time.time() - start_time
    print(f"\nGesamtausführungszeit: {execution_time:.2f} Sekunden")
    
    # Bestimmen, ob alle Tests erfolgreich waren
    success = all(
        result.wasSuccessful()
        for result in results.values()
        if result is not None
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
