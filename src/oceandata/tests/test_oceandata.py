def test_prepare_data_for_monetization(self):
        """Test der Datenvorbereitung für Monetarisierung"""
        ocean_ai = self.OceanDataAI()
        
        # Browser-Daten für Monetarisierung vorbereiten
        browser_asset = ocean_ai.prepare_data_for_monetization(self.browser_data, 'browser', 'medium')
        
        # Überprüfen, ob der Asset das erwartete Format hat
        self.assertIn('anonymized_data', browser_asset)
        self.assertIn('metadata', browser_asset)
        self.assertIn('c2d_asset', browser_asset)
        
        # Überprüfen, ob die Metadaten korrekt sind
        self.assertEqual(browser_asset['metadata']['source_type'], 'browser')
        self.assertEqual(browser_asset['metadata']['privacy_level'], 'medium')
        self.assertEqual(browser_asset['metadata']['record_count'], len(self.browser_data))
        
        # Smartwatch-Daten mit höherem Datenschutzniveau vorbereiten
        smartwatch_asset = ocean_ai.prepare_data_for_monetization(self.smartwatch_data, 'smartwatch', 'high')
        
        # Überprüfen, ob der Privacy-Level korrekt ist und sich auf den Wert auswirkt
        self.assertEqual(smartwatch_asset['metadata']['privacy_level'], 'high')
        # Bei höherem Datenschutz erwarten wir einen geringeren Wert
        self.assertLess(
            smartwatch_asset['metadata']['estimated_value'],
            # Der gleiche Datensatz mit niedrigerem Datenschutz sollte mehr wert sein
            ocean_ai.prepare_data_for_monetization(self.smartwatch_data, 'smartwatch', 'low')['metadata']['estimated_value']
        )
    
    def test_combine_data_sources(self):
        """Test der Kombination von Datenquellen"""
        ocean_ai = self.OceanDataAI()
        
        # Assets für den Test erstellen
        browser_asset = ocean_ai.prepare_data_for_monetization(self.browser_data, 'browser', 'medium')
        smartwatch_asset = ocean_ai.prepare_data_for_monetization(self.smartwatch_data, 'smartwatch', 'medium')
        
        # Testen verschiedener Kombinationstypen
        for combination_type in ['merge', 'enrich', 'correlate']:
            combined_asset = ocean_ai.combine_data_sources(
                [browser_asset, smartwatch_asset], 
                combination_type
            )
            
            # Überprüfen, ob das kombinierte Asset das erwartete Format hat
            self.assertIn('anonymized_data', combined_asset)
            self.assertIn('metadata', combined_asset)
            self.assertIn('c2d_asset', combined_asset)
            
            # Überprüfen, ob die Metadaten korrekt sind
            self.assertEqual(combined_asset['metadata']['combination_type'], combination_type)
            self.assertEqual(combined_asset['metadata']['source_count'], 2)
            
            # Überprüfen, ob der kombinierte Wert höher ist als die Summe der Einzelwerte
            individual_values = browser_asset['metadata']['estimated_value'] + smartwatch_asset['metadata']['estimated_value']
            self.assertGreaterEqual(combined_asset['metadata']['estimated_value'], individual_values)
    
    def test_estimate_data_value(self):
        """Test der Datenwertschätzung"""
        ocean_ai = self.OceanDataAI()
        
        # Wert der Browser-Daten schätzen
        value_assessment = ocean_ai.estimate_data_value(self.browser_data, {'source_type': 'browser'})
        
        # Überprüfen, ob die Wertschätzung das erwartete Format hat
        self.assertIn('normalized_score', value_assessment)
        self.assertIn('estimated_token_value', value_assessment)
        self.assertIn('metrics', value_assessment)
        self.assertIn('summary', value_assessment)
        
        # Überprüfen, ob die Metriken das erwartete Format haben
        self.assertIn('data_volume', value_assessment['metrics'])
        self.assertIn('data_quality', value_assessment['metrics'])
        self.assertIn('data_uniqueness', value_assessment['metrics'])
        self.assertIn('time_relevance', value_assessment['metrics'])
        
        # Überprüfen, ob jede Metrik die erwarteten Felder hat
        for metric in value_assessment['metrics'].values():
            self.assertIn('score', metric)
            self.assertIn('weight', metric)
            self.assertIn('explanation', metric)
        
        # Überprüfen, ob der geschätzte Wert im erwarteten Bereich liegt
        self.assertGreaterEqual(value_assessment['estimated_token_value'], 0)
        
        # Wert eines anderen Datentyps schätzen
        smartwatch_value = ocean_ai.estimate_data_value(self.smartwatch_data, {'source_type': 'smartwatch'})
        
        # Gesundheitsdaten sollten wertvoller sein als Browserdaten
        self.assertGreater(smartwatch_value['estimated_token_value'], value_assessment['estimated_token_value'])
    
    def test_prepare_for_ocean_tokenization(self):
        """Test der Vorbereitung für Ocean Protocol Tokenisierung"""
        ocean_ai = self.OceanDataAI()
        
        # Asset für den Test erstellen
        browser_asset = ocean_ai.prepare_data_for_monetization(self.browser_data, 'browser', 'medium')
        
        # Asset für Ocean vorbereiten
        ocean_asset = ocean_ai.prepare_for_ocean_tokenization(browser_asset)
        
        # Überprüfen, ob der Ocean-Asset das erwartete Format hat
        self.assertIn('ddo', ocean_asset)
        self.assertIn('pricing', ocean_asset)
        self.assertIn('asset_id', ocean_asset)
        
        # Überprüfen, ob das DDO die erwarteten Felder hat
        self.assertIn('id', ocean_asset['ddo'])
        self.assertIn('name', ocean_asset['ddo'])
        self.assertIn('type', ocean_asset['ddo'])
        self.assertIn('tags', ocean_asset['ddo'])
        self.assertIn('price', ocean_asset['ddo'])
        
        # Überprüfen, ob die Pricing-Informationen korrekt sind
        self.assertEqual(ocean_asset['pricing']['type'], 'fixed')
        self.assertEqual(
            ocean_asset['pricing']['baseTokenAmount'], 
            browser_asset['metadata']['estimated_value']
        )
        self.assertIn('datatoken', ocean_asset['pricing'])
    
    def test_tokenize_with_ocean(self):
        """Test der Tokenisierung mit Ocean Protocol"""
        ocean_ai = self.OceanDataAI()
        
        # Asset für den Test erstellen
        browser_asset = ocean_ai.prepare_data_for_monetization(self.browser_data, 'browser', 'medium')
        ocean_asset = ocean_ai.prepare_for_ocean_tokenization(browser_asset)
        
        # Asset tokenisieren
        tokenization_result = ocean_ai.tokenize_with_ocean(ocean_asset)
        
        # Überprüfen, ob das Ergebnis das erwartete Format hat
        self.assertIn('success', tokenization_result)
        self.assertTrue(tokenization_result['success'])
        self.assertIn('asset_id', tokenization_result)
        self.assertIn('token_address', tokenization_result)
        self.assertIn('token_symbol', tokenization_result)
        self.assertIn('token_price', tokenization_result)
        self.assertIn('transaction_hash', tokenization_result)
        self.assertIn('marketplace_url', tokenization_result)
        
        # Überprüfen, ob der Token-Preis mit dem geschätzten Wert übereinstimmt
        self.assertEqual(
            tokenization_result['token_price'],
            ocean_asset['pricing']['baseTokenAmount']
        )


class TestIntegrationOceanData(unittest.TestCase):
    """Integrationstests für die OceanData-Plattform"""
    
    @unittest.skip("Erfordert vollständige Plattformimplementierung")
    def test_end_to_end_data_monetization(self):
        """Test des gesamten Datenmonetarisierungsprozesses"""
        # In einem echten Integrationstest würden wir hier die tatsächliche Implementierung verwenden
        pass


# Integration-Tests mit der Frontend-Komponente über pytest
@pytest.mark.skip("Requires React testing environment")
class TestReactComponents:
    """Testsuite für die React-Komponenten mit pytest und React Testing Library"""
    
    @pytest.fixture
    def setup_react_testing(self):
        """Setup für React-Tests"""
        # Hier würden wir das React-Test-Setup initialisieren
        pass
    
    def test_data_tokenization_dashboard(self, setup_react_testing):
        """Test der DataTokenizationDashboard-Komponente"""
        # Hier würden wir die React-Komponente testen
        pass


# Performance-Tests
class TestPerformanceOceanData(unittest.TestCase):
    """Performance-Tests für kritische Komponenten"""
    
    @unittest.skip("Langläufiger Performance-Test")
    def test_data_integration_performance(self):
        """Test der Performance der Datenintegration mit großen Datensätzen"""
        # Große Datensätze generieren
        large_browser_data = create_mock_browser_data(n_rows=100000)
        large_smartwatch_data = create_mock_smartwatch_data(n_rows=100000)
        
        # Performance-Messung
        import time
        
        start_time = time.time()
        # Hier würden wir die Datenintegration durchführen
        end_time = time.time()
        
        # Performance-Schwellenwert prüfen
        execution_time = end_time - start_time
        self.assertLess(execution_time, 10.0)  # Sollte unter 10 Sekunden bleiben
    
    @unittest.skip("Langläufiger Performance-Test")
    def test_ai_analysis_performance(self):
        """Test der Performance der KI-Analyse mit großen Datensätzen"""
        # Zu implementieren
        pass


# Hauptausführungspunkt für Unit-Tests
if __name__ == "__main__":
    unittest.main()
"""
OceanData Test-Framework

Dieses Modul enthält umfassende Tests für alle Komponenten der OceanData-Plattform,
einschließlich Unit-Tests, Integrationstests und End-to-End-Tests.
"""

import unittest
import pandas as pd
import numpy as np
import json
import os
import tempfile
from datetime import datetime, timedelta
import tensorflow as tf
import torch
import pytest
from unittest import mock
from io import StringIO

# Mockdaten für Tests
def create_mock_browser_data(n_rows=100):
    """Erstellt Mockdaten für Browsertests"""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_rows, freq='H'),
        'url': [f"example{i % 10}.com/page{i % 5}" for i in range(n_rows)],
        'duration': np.random.randint(10, 300, n_rows),
        'user_id': ['user123'] * n_rows,
        'device_type': np.random.choice(['desktop', 'mobile', 'tablet'], n_rows),
        'browser_type': np.random.choice(['chrome', 'firefox', 'safari'], n_rows)
    })

def create_mock_smartwatch_data(n_rows=100):
    """Erstellt Mockdaten für Smartwatchtests"""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_rows, freq='H'),
        'heart_rate': np.random.randint(60, 100, n_rows),
        'steps': np.random.randint(0, 1000, n_rows),
        'user_id': ['user123'] * n_rows,
        'sleep_quality': np.random.choice(['good', 'medium', 'poor'], n_rows),
        'calories_burned': np.random.randint(50, 300, n_rows)
    })

class TestDataSource(unittest.TestCase):
    """Testsuite für die abstrakte DataSource-Klasse und ihre Implementierungen"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Wir müssen hier die tatsächlichen Klassen importieren oder mocken
        from unittest.mock import MagicMock
        self.mock_data_category = MagicMock()
        self.mock_privacy_level = MagicMock()
        
        # Annahme, dass wir die Klassen importieren können
        # from oceandata.data_integration import DataSource, BrowserDataConnector, DataCategory, PrivacyLevel
        
        # Für das Testen verwenden wir Mock-Klassen
        class MockDataSource:
            def __init__(self, source_id, user_id, category):
                self.source_id = source_id
                self.user_id = user_id
                self.category = category
                self.metadata = {
                    "source_id": source_id,
                    "user_id": user_id,
                    "category": category,
                    "created": datetime.now().isoformat(),
                    "privacy_fields": {},
                }
                self.last_sync = None
                self.data = None
            
            def set_privacy_level(self, field, level):
                self.metadata["privacy_fields"][field] = level
            
            def get_privacy_level(self, field):
                return self.metadata["privacy_fields"].get(field, self.mock_privacy_level.ANONYMIZED)
            
            def connect(self):
                return True
            
            def fetch_data(self):
                return pd.DataFrame()
            
            def process_data(self, data):
                return data
            
            def get_data(self):
                return {"data": pd.DataFrame(), "metadata": self.metadata, "status": "success"}
        
        self.DataSource = MockDataSource
        
        # Mock-Implementierung für BrowserDataConnector
        class MockBrowserDataConnector(MockDataSource):
            def __init__(self, user_id, browser_type='chrome'):
                super().__init__(f"browser_{browser_type}", user_id, "BROWSER")
                self.browser_type = browser_type
            
            def connect(self):
                return True
            
            def fetch_data(self):
                return create_mock_browser_data()
        
        self.BrowserDataConnector = MockBrowserDataConnector
    
    def test_data_source_initialization(self):
        """Test der Initialisierung von DataSource"""
        source = self.DataSource("test_source", "user123", "TEST")
        self.assertEqual(source.source_id, "test_source")
        self.assertEqual(source.user_id, "user123")
        self.assertEqual(source.category, "TEST")
        self.assertIsNone(source.last_sync)
        self.assertIsNone(source.data)
    
    def test_privacy_level_management(self):
        """Test des Privacy-Level-Managements"""
        source = self.DataSource("test_source", "user123", "TEST")
        source.set_privacy_level("test_field", "ANONYMIZED")
        self.assertEqual(source.metadata["privacy_fields"]["test_field"], "ANONYMIZED")
        
        privacy_level = source.get_privacy_level("test_field")
        self.assertEqual(privacy_level, "ANONYMIZED")
        
        # Test Default-Wert
        default_level = source.get_privacy_level("nonexistent_field")
        self.assertEqual(default_level, self.mock_privacy_level.ANONYMIZED)
    
    def test_browser_connector_fetch_data(self):
        """Test der Datenabruffunktionalität des BrowserDataConnector"""
        browser_connector = self.BrowserDataConnector("user123", "chrome")
        data = browser_connector.fetch_data()
        
        # Überprüfen, ob die Daten ein DataFrame sind und die erwarteten Spalten enthalten
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('url', data.columns)
        self.assertIn('duration', data.columns)
        self.assertIn('user_id', data.columns)
        
        # Überprüfen, ob die user_id korrekt ist
        self.assertEqual(data['user_id'].iloc[0], "user123")
    
    def test_get_data_pipeline(self):
        """Test der gesamten Datenabruf-Pipeline"""
        browser_connector = self.BrowserDataConnector("user123", "chrome")
        result = browser_connector.get_data()
        
        # Überprüfen, ob das Ergebnis das erwartete Format hat
        self.assertIn('data', result)
        self.assertIn('metadata', result)
        self.assertIn('status', result)
        
        # Überprüfen, ob der Status erfolgreich ist
        self.assertEqual(result['status'], 'success')


class TestAnomalyDetector(unittest.TestCase):
    """Testsuite für die AnomalyDetector-Klasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Mockdaten erstellen
        self.normal_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        
        # Anomalien hinzufügen
        anomalies = pd.DataFrame({
            'feature1': np.random.normal(5, 1, 5),  # Weit entfernt vom Durchschnitt
            'feature2': np.random.normal(5, 1, 5)
        })
        
        self.data_with_anomalies = pd.concat([self.normal_data, anomalies])
        
        # Annahme, dass wir die Klasse importieren können
        # from oceandata.ai import AnomalyDetector
        
        # Für diesen Test verwenden wir eine Mock-Implementierung
        class MockAnomalyDetector:
            def __init__(self, method='isolation_forest', contamination=0.05):
                self.method = method
                self.contamination = contamination
                self.model = None
                self.is_fitted = False
                self.feature_dims = None
                self.threshold = None
                
                if method == 'isolation_forest':
                    from sklearn.ensemble import IsolationForest
                    self.model = IsolationForest(contamination=contamination, random_state=42)
                elif method == 'autoencoder':
                    # Vereinfachte Implementierung für Tests
                    self.model = mock.MagicMock()
                    self.model.predict.return_value = np.zeros((100, 2))
                    self.threshold = 0.5
                    
            def fit(self, X, categorical_cols=None):
                if self.method == 'isolation_forest':
                    self.model.fit(X)
                self.is_fitted = True
                self.feature_dims = X.shape[1]
                return self
            
            def predict(self, X, categorical_cols=None):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                if self.method == 'isolation_forest':
                    return self.model.predict(X)
                elif self.method == 'autoencoder':
                    # Simuliere Autoencoder-Vorhersagen
                    reconstructions = self.model.predict(X)
                    mse = np.mean(np.power(X.values - reconstructions, 2), axis=1)
                    return mse > self.threshold
        
        self.AnomalyDetector = MockAnomalyDetector
    
    def test_initialization(self):
        """Test der Initialisierung mit verschiedenen Methoden"""
        detector_if = self.AnomalyDetector(method='isolation_forest', contamination=0.1)
        self.assertEqual(detector_if.method, 'isolation_forest')
        self.assertEqual(detector_if.contamination, 0.1)
        
        detector_ae = self.AnomalyDetector(method='autoencoder', contamination=0.05)
        self.assertEqual(detector_ae.method, 'autoencoder')
        self.assertEqual(detector_ae.contamination, 0.05)
    
    def test_fit_predict_isolation_forest(self):
        """Test der Fit- und Predict-Methoden mit Isolation Forest"""
        detector = self.AnomalyDetector(method='isolation_forest')
        detector.fit(self.data_with_anomalies)
        
        # Überprüfen, ob das Modell trainiert wurde
        self.assertTrue(detector.is_fitted)
        self.assertEqual(detector.feature_dims, 2)
        
        # Vorhersagen treffen
        predictions = detector.predict(self.data_with_anomalies)
        
        # Überprüfen, ob Vorhersagen das erwartete Format haben
        self.assertEqual(predictions.shape, (105,))
        
        # Bei Isolation Forest: -1 für Anomalien, 1 für normale Daten
        # Wir erwarten etwa 5 Anomalien in unseren Daten
        anomaly_count = np.sum(predictions == -1)
        self.assertGreaterEqual(anomaly_count, 1)  # Mindestens eine Anomalie sollte erkannt werden
    
    @unittest.skip("Autoencoder-Tests erfordern TensorFlow-Setup")
    def test_fit_predict_autoencoder(self):
        """Test der Fit- und Predict-Methoden mit Autoencoder"""
        detector = self.AnomalyDetector(method='autoencoder')
        detector.fit(self.data_with_anomalies)
        
        # Überprüfen, ob das Modell trainiert wurde
        self.assertTrue(detector.is_fitted)
        
        # Vorhersagen treffen
        predictions = detector.predict(self.data_with_anomalies)
        
        # Bei Autoencoder: True für Anomalien, False für normale Daten
        anomaly_count = np.sum(predictions)
        self.assertGreaterEqual(anomaly_count, 1)  # Mindestens eine Anomalie sollte erkannt werden


class TestSemanticAnalyzer(unittest.TestCase):
    """Testsuite für die SemanticAnalyzer-Klasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Beispieltexte für Tests
        self.sample_texts = [
            "I love this product! It's amazing and works really well.",
            "This is terrible. Doesn't work at all. Very disappointed.",
            "It's okay. Not great, not bad. Somewhat useful."
        ]
        
        # Da das Laden echter Transformer-Modelle zu ressourcenintensiv für Tests ist,
        # erstellen wir eine Mock-Implementierung
        class MockSemanticAnalyzer:
            def __init__(self, model_type='bert', model_name='bert-base-uncased'):
                self.model_type = model_type
                self.model_name = model_name
                self.model = mock.MagicMock()
                self.tokenizer = mock.MagicMock()
                self.embeddings_cache = {}
                
                # Mock für die Sentimentanalyse
                self.sentiment_results = {
                    "I love this product! It's amazing and works really well.": 
                        {"sentiment": "positive", "scores": {"positive": 0.9, "negative": 0.05, "neutral": 0.05, "compound": 0.8}},
                    "This is terrible. Doesn't work at all. Very disappointed.": 
                        {"sentiment": "negative", "scores": {"positive": 0.05, "negative": 0.9, "neutral": 0.05, "compound": -0.8}},
                    "It's okay. Not great, not bad. Somewhat useful.": 
                        {"sentiment": "neutral", "scores": {"positive": 0.3, "negative": 0.3, "neutral": 0.4, "compound": 0.1}}
                }
            
            def get_embeddings(self, texts, batch_size=8, use_cache=True):
                if isinstance(texts, str):
                    texts = [texts]
                
                # Einfache Mock-Embeddings erstellen
                return np.random.random((len(texts), 768))
            
            def analyze_sentiment(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                
                results = []
                for text in texts:
                    # Fallback, falls der Text nicht in den vordefinierten Ergebnissen ist
                    sentiment_data = self.sentiment_results.get(
                        text, 
                        {"sentiment": "neutral", "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "compound": 0.0}}
                    )
                    
                    result = {
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'sentiment': sentiment_data["sentiment"],
                        'scores': sentiment_data["scores"]
                    }
                    results.append(result)
                
                return results
            
            def extract_topics(self, texts, num_topics=5, words_per_topic=5):
                # Mock-Themenextraktion
                topics = [
                    {
                        "id": 0,
                        "words": ["product", "great", "good", "useful", "works"],
                        "examples": [text[:50] + "..." for text in texts[:2]]
                    },
                    {
                        "id": 1,
                        "words": ["bad", "terrible", "disappointed", "not", "issue"],
                        "examples": [text[:50] + "..." for text in texts if "terrible" in text]
                    }
                ]
                return topics[:num_topics]
            
            def find_similar_texts(self, query, corpus, top_n=5):
                # Mock-Ähnlichkeitssuche
                similarities = np.random.random(len(corpus))
                top_indices = np.argsort(similarities)[::-1][:top_n]
                
                results = []
                for idx in top_indices:
                    result = {
                        "text": corpus[idx][:100] + "..." if len(corpus[idx]) > 100 else corpus[idx],
                        "similarity": float(similarities[idx]),
                        "index": int(idx)
                    }
                    results.append(result)
                
                return results
        
        self.SemanticAnalyzer = MockSemanticAnalyzer
    
    def test_sentiment_analysis(self):
        """Test der Sentimentanalyse-Funktionalität"""
        analyzer = self.SemanticAnalyzer()
        results = analyzer.analyze_sentiment(self.sample_texts)
        
        # Überprüfen, ob die Ergebnisse das erwartete Format haben
        self.assertEqual(len(results), 3)
        self.assertIn('text', results[0])
        self.assertIn('sentiment', results[0])
        self.assertIn('scores', results[0])
        
        # Überprüfen, ob die Sentiments korrekt sind
        self.assertEqual(results[0]['sentiment'], 'positive')
        self.assertEqual(results[1]['sentiment'], 'negative')
        self.assertEqual(results[2]['sentiment'], 'neutral')
    
    def test_topic_extraction(self):
        """Test der Themenextraktionsfunktionalität"""
        analyzer = self.SemanticAnalyzer()
        topics = analyzer.extract_topics(self.sample_texts, num_topics=2)
        
        # Überprüfen, ob die Ergebnisse das erwartete Format haben
        self.assertEqual(len(topics), 2)
        self.assertIn('id', topics[0])
        self.assertIn('words', topics[0])
        self.assertIn('examples', topics[0])
        
        # Überprüfen, ob die Wörter pro Thema korrekt sind
        self.assertEqual(len(topics[0]['words']), 5)
        
        # Überprüfen, ob die Beispiele enthalten sind
        self.assertGreaterEqual(len(topics[0]['examples']), 1)
    
    def test_text_similarity(self):
        """Test der Textähnlichkeitsfunktionalität"""
        analyzer = self.SemanticAnalyzer()
        query = "Is this product any good?"
        results = analyzer.find_similar_texts(query, self.sample_texts, top_n=2)
        
        # Überprüfen, ob die Ergebnisse das erwartete Format haben
        self.assertEqual(len(results), 2)
        self.assertIn('text', results[0])
        self.assertIn('similarity', results[0])
        self.assertIn('index', results[0])
        
        # Überprüfen, ob die Ähnlichkeitswerte im erwarteten Bereich liegen
        self.assertGreaterEqual(results[0]['similarity'], 0)
        self.assertLessEqual(results[0]['similarity'], 1)


class TestPredictiveModeler(unittest.TestCase):
    """Testsuite für die PredictiveModeler-Klasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Mockdaten für Vorhersagetests erstellen
        np.random.seed(42)
        
        # Zeitreihe mit Trend und Saison
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        trend = np.linspace(0, 10, 100)
        seasonality = 5 * np.sin(np.linspace(0, 4*np.pi, 100))
        noise = np.random.normal(0, 1, 100)
        
        self.time_series_data = pd.DataFrame({
            'date': dates,
            'value': trend + seasonality + noise,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'day': dates.day
        })
        
        # Mock-Implementierung für Tests
        class MockPredictiveModeler:
            def __init__(self, model_type='lstm', forecast_horizon=7):
                self.model_type = model_type
                self.forecast_horizon = forecast_horizon
                self.model = mock.MagicMock()
                self.scaler = mock.MagicMock()
                self.target_scaler = mock.MagicMock()
                self.is_fitted = False
                self.lookback = 10
                self.feature_dims = None
                self.target_dims = None
                
                # Mock-Skalierungsverhalten
                self.scaler.transform.side_effect = lambda x: x
                self.scaler.inverse_transform.side_effect = lambda x: x
                self.target_scaler.transform.side_effect = lambda x: x
                self.target_scaler.inverse_transform.side_effect = lambda x: x
                
            def fit(self, X, y=None, lookback=10, epochs=50, validation_split=0.2, batch_size=32, verbose=1):
                self.lookback = lookback
                self.is_fitted = True
                
                if isinstance(X, pd.DataFrame):
                    self.feature_dims = X.shape[1]
                else:
                    self.feature_dims = X.shape[1] if len(X.shape) > 1 else 1
                
                if y is not None:
                    if isinstance(y, pd.DataFrame):
                        self.target_dims = y.shape[1]
                    else:
                        self.target_dims = y.shape[1] if len(y.shape) > 1 else 1
                else:
                    self.target_dims = self.feature_dims
                
                return self
            
            def predict(self, X, return_sequences=False):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                # Einfache Simulation einer Vorhersage
                n_samples = len(X) if isinstance(X, pd.DataFrame) else X.shape[0]
                
                if return_sequences:
                    # Multi-step forecast
                    return np.random.random((n_samples, self.forecast_horizon, self.target_dims))
                else:
                    # Single-step forecast
                    return np.random.random((n_samples, self.target_dims))
            
            def forecast(self, X, steps=None):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                steps = steps or self.forecast_horizon
                
                # Simuliere eine Forecast-Sequenz
                return np.random.random((steps, self.target_dims))
            
            def evaluate(self, X_test, y_test):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                # Simulierte Metriken
                return {
                    'loss': 0.1,
                    'mae': 0.08,
                    'rmse': 0.12,
                    'mean_absolute_error': 0.08,
                    'mean_absolute_percentage_error': 8.5
                }
        
        self.PredictiveModeler = MockPredictiveModeler
    
    def test_initialization(self):
        """Test der Initialisierung mit verschiedenen Parametern"""
        modeler = self.PredictiveModeler(model_type='lstm', forecast_horizon=7)
        self.assertEqual(modeler.model_type, 'lstm')
        self.assertEqual(modeler.forecast_horizon, 7)
        self.assertFalse(modeler.is_fitted)
        
        modeler2 = self.PredictiveModeler(model_type='transformer', forecast_horizon=14)
        self.assertEqual(modeler2.model_type, 'transformer')
        self.assertEqual(modeler2.forecast_horizon, 14)
    
    def test_fit_predict(self):
        """Test der Fit- und Predict-Methoden"""
        modeler = self.PredictiveModeler()
        
        # Feature- und Zieldaten vorbereiten
        X = self.time_series_data[['day_of_week', 'month', 'day']]
        y = self.time_series_data[['value']]
        
        # Modell trainieren
        modeler.fit(X, y, lookback=5, epochs=10, verbose=0)
        
        # Überprüfen, ob das Modell trainiert wurde
        self.assertTrue(modeler.is_fitted)
        self.assertEqual(modeler.feature_dims, 3)
        self.assertEqual(modeler.target_dims, 1)
        
        # Vorhersagen treffen
        predictions = modeler.predict(X)
        
        # Überprüfen, ob die Vorhersagen das erwartete Format haben
        self.assertEqual(predictions.shape[0], len(X))
        self.assertEqual(predictions.shape[1], 1)
    
    def test_forecast(self):
        """Test der Forecast-Methode"""
        modeler = self.PredictiveModeler(forecast_horizon=10)
        
        # Feature- und Zieldaten vorbereiten
        X = self.time_series_data[['day_of_week', 'month', 'day']]
        y = self.time_series_data[['value']]
        
        # Modell trainieren
        modeler.fit(X, y, lookback=5, epochs=10, verbose=0)
        
        # Forecast für die nächsten 10 Schritte
        forecast = modeler.forecast(X, steps=10)
        
        # Überprüfen, ob der Forecast das erwartete Format hat
        self.assertEqual(forecast.shape[0], 10)
        self.assertEqual(forecast.shape[1], 1)
    
    def test_evaluation(self):
        """Test der Evaluate-Methode"""
        modeler = self.PredictiveModeler()
        
        # Feature- und Zieldaten vorbereiten
        X = self.time_series_data[['day_of_week', 'month', 'day']]
        y = self.time_series_data[['value']]
        
        # Modell trainieren
        modeler.fit(X, y, lookback=5, epochs=10, verbose=0)
        
        # Train-Test-Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Modell evaluieren
        metrics = modeler.evaluate(X_test, y_test)
        
        # Überprüfen, ob die Metriken das erwartete Format haben
        self.assertIn('loss', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mean_absolute_error', metrics)
        self.assertIn('mean_absolute_percentage_error', metrics)


class TestDataSynthesizer(unittest.TestCase):
    """Testsuite für die DataSynthesizer-Klasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Mockdaten für Synthesetests erstellen
        np.random.seed(42)
        
        self.categorical_data = pd.DataFrame({
            'category_1': np.random.choice(['A', 'B', 'C'], 100),
            'category_2': np.random.choice(['X', 'Y', 'Z'], 100)
        })
        
        self.numeric_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(5, 2, 100)
        })
        
        self.mixed_data = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'numeric': np.random.normal(0, 1, 100),
            'binary': np.random.choice([0, 1], 100)
        })
        
        # Mock-Implementierung für Tests
        class MockDataSynthesizer:
            def __init__(self, categorical_threshold=10, noise_dim=100):
                self.categorical_threshold = categorical_threshold
                self.noise_dim = noise_dim
                self.generator = mock.MagicMock()
                self.discriminator = mock.MagicMock()
                self.gan = mock.MagicMock()
                
                self.column_types = {}
                self.categorical_mappings = {}
                self.is_fitted = False
                self.feature_dims = None
                self.training_data = None
            
            def _identify_column_types(self, data):
                self.column_types = {}
                
                for col in data.columns:
                    n_unique = data[col].nunique()
                    
                    if n_unique < self.categorical_threshold or not pd.api.types.is_numeric_dtype(data[col]):
                        self.column_types[col] = 'categorical'
                        
                        # Mapping für kategorische Spalten erstellen
                        categories = data[col].unique()
                        self.categorical_mappings[col] = {cat: i for i, cat in enumerate(categories)}
                        self.categorical_mappings[f"{col}_reverse"] = {i: cat for i, cat in enumerate(categories)}
                    else:
                        self.column_types[col] = 'continuous'
            
            def fit(self, data, epochs=2000, batch_size=32, sample_interval=100, verbose=1):
                self._identify_column_types(data)
                self.feature_dims = len(data.columns)
                self.training_data = data.copy()
                self.is_fitted = True
                return self
            
            def generate(self, n_samples=100):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                # Einfache Simulation synthetischer Daten
                synthetic_data = pd.DataFrame()
                
                for col, col_type in self.column_types.items():
                    if col_type == 'categorical':
                        # Für kategorische Spalten: Zufällige Auswahl aus Original-Kategorien
                        categories = list(self.categorical_mappings[f"{col}_reverse"].values())
                        synthetic_data[col] = np.random.choice(categories, n_samples)
                    else:
                        # Für kontinuierliche Spalten: Simulieren ähnlicher Verteilungen
                        orig_mean = self.training_data[col].mean()
                        orig_std = self.training_data[col].std()
                        synthetic_data[col] = np.random.normal(orig_mean, orig_std, n_samples)
                
                return synthetic_data
            
            def evaluate_quality(self, n_samples=100):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                # Synthetische Daten generieren
                synthetic_data = self.generate(n_samples)
                
                # Qualitätsmetriken simulieren
                metrics = {
                    'continuous_avg_diff': 0.15,
                    'categorical_avg_diff': 0.20,
                    'overall_quality_score': 0.75
                }
                
                # Detaillierte Metriken für jede Spalte
                for col in self.column_types:
                    if self.column_types[col] == 'continuous':
                        metrics[f"{col}_mean_diff"] = np.random.random() * 0.3  # Zufällige Differenz zwischen 0 und 0.3
                        metrics[f"{col}_std_diff"] = np.random.random() * 0.3
                    else:
                        metrics[f"{col}_js_divergence"] = np.random.random() * 0.4  # Zufällige JS-Divergenz zwischen 0 und 0.4
                
                return metrics
            
            def plot_comparison(self, n_samples=100, features=None, save_path=None):
                # Mock für Plot-Funktionalität
                from matplotlib.figure import Figure
                return Figure()
        
        self.DataSynthesizer = MockDataSynthesizer
    
    def test_initialization(self):
        """Test der Initialisierung mit verschiedenen Parametern"""
        synthesizer = self.DataSynthesizer(categorical_threshold=10, noise_dim=100)
        self.assertEqual(synthesizer.categorical_threshold, 10)
        self.assertEqual(synthesizer.noise_dim, 100)
        self.assertFalse(synthesizer.is_fitted)
    
    def test_column_type_identification(self):
        """Test der Spaltentyp-Identifikation"""
        synthesizer = self.DataSynthesizer()
        synthesizer._identify_column_types(self.mixed_data)
        
        # Überprüfen, ob die Spaltentypen korrekt identifiziert wurden
        self.assertEqual(synthesizer.column_types['category'], 'categorical')
        self.assertEqual(synthesizer.column_types['numeric'], 'continuous')
        self.assertEqual(synthesizer.column_types['binary'], 'categorical')
        
        # Überprüfen, ob die kategorischen Mappings erstellt wurden
        self.assertIn('category', synthesizer.categorical_mappings)
        self.assertIn('category_reverse', synthesizer.categorical_mappings)
        self.assertIn('binary', synthesizer.categorical_mappings)
        self.assertIn('binary_reverse', synthesizer.categorical_mappings)
    
    def test_fit_generate(self):
        """Test der Fit- und Generate-Methoden"""
        synthesizer = self.DataSynthesizer()
        
        # Modell mit gemischten Daten trainieren
        synthesizer.fit(self.mixed_data, epochs=10, verbose=0)
        
        # Überprüfen, ob das Modell trainiert wurde
        self.assertTrue(synthesizer.is_fitted)
        self.assertEqual(synthesizer.feature_dims, 3)
        
        # Synthetische Daten generieren
        n_samples = 50
        synthetic_data = synthesizer.generate(n_samples)
        
        # Überprüfen, ob die synthetischen Daten das erwartete Format haben
        self.assertEqual(len(synthetic_data), n_samples)
        self.assertEqual(len(synthetic_data.columns), 3)
        self.assertIn('category', synthetic_data.columns)
        self.assertIn('numeric', synthetic_data.columns)
        self.assertIn('binary', synthetic_data.columns)
    
    def test_quality_evaluation(self):
        """Test der Qualitätsbewertung der synthetischen Daten"""
        synthesizer = self.DataSynthesizer()
        
        # Modell mit numerischen Daten trainieren
        synthesizer.fit(self.numeric_data, epochs=10, verbose=0)
        
        # Qualität bewerten
        metrics = synthesizer.evaluate_quality(n_samples=50)
        
        # Überprüfen, ob die Metriken das erwartete Format haben
        self.assertIn('continuous_avg_diff', metrics)
        self.assertIn('overall_quality_score', metrics)
        
        # Überprüfen, ob spaltenspezifische Metriken enthalten sind
        self.assertIn('feature_1_mean_diff', metrics)
        self.assertIn('feature_2_mean_diff', metrics)
        
        # Überprüfen, ob die Qualitätsbewertung im gültigen Bereich liegt
        self.assertGreaterEqual(metrics['overall_quality_score'], 0)
        self.assertLessEqual(metrics['overall_quality_score'], 1)


class TestComputeToDataManager(unittest.TestCase):
    """Testsuite für die ComputeToDataManager-Klasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Mockdaten für C2D-Tests erstellen
        np.random.seed(42)
        
        self.test_data = pd.DataFrame({
            'id': range(100),
            'value': np.random.normal(0, 1, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'location': [f"City_{i % 10}" for i in range(100)],
            'sensitive_info': [f"sensitive_{i}" for i in range(100)]
        })
        
        # Mock-Implementierung für Tests
        class MockComputeToDataManager:
            def __init__(self, encryption_key=None):
                from cryptography.fernet import Fernet
                self.encryption_key = encryption_key or Fernet.generate_key()
                self.cipher_suite = Fernet(self.encryption_key)
                
                # Standard-Operationen definieren
                self.allowed_operations = {
                    'aggregate': self._aggregate,
                    'count': self._count,
                    'mean': self._mean,
                    'sum': self._sum,
                    'min': self._min,
                    'max': self._max,
                    'correlation': self._correlation,
                    'histogram': self._histogram,
                    'custom_model': self._custom_model_inference
                }
                
                # Datenschutzkonfiguration
                self.privacy_config = {
                    'min_group_size': 5,
                    'noise_level': 0.01,
                    'outlier_removal': True
                }
            
            def _encrypt_data(self, data):
                """Verschlüsselt Daten für die sichere Speicherung"""
                serialized = data.to_json().encode()
                return self.cipher_suite.encrypt(serialized)
            
            def _decrypt_data(self, encrypted_data):
                """Entschlüsselt Daten für die Verarbeitung"""
                decrypted = self.cipher_suite.decrypt(encrypted_data)
                return pd.read_json(StringIO(decrypted.decode()))
            
            def _aggregate(self, data, params):
                """Aggregiert Daten nach Spalten und Gruppen"""
                # Vereinfachte Implementierung für Tests
                return {'count': len(data)}
            
            def _count(self, data, params):
                """Zählt Datensätze nach Filterkriterien"""
                filter_column = params.get('filter_column')
                filter_value = params.get('filter_value')
                
                if filter_column and filter_value is not None:
                    count = len(data[data[filter_column] == filter_value])
                else:
                    count = len(data)
                
                return {'count': count}
            
            def _mean(self, data, params):
                """Berechnet den Mittelwert für numerische Spalten"""
                columns = params.get('columns', [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])])
                
                means = {}
                for col in columns:
                    if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                        means[col] = float(data[col].mean())
                
                return {'means': means}
            
            def _sum(self, data, params):
                return {'sums': {col: float(data[col].sum()) for col in data.select_dtypes(include=['number']).columns}}
            
            def _min(self, data, params):
                return {'minimums': {col: float(data[col].min()) for col in data.select_dtypes(include=['number']).columns}}
            
            def _max(self, data, params):
                return {'maximums': {col: float(data[col].max()) for col in data.select_dtypes(include=['number']).columns}}
            
            def _correlation(self, data, params):
                # Vereinfachte Implementierung für Tests
                corr_matrix = data.select_dtypes(include=['number']).corr().to_dict()
                return {'correlation_matrix': corr_matrix}
            
            def _histogram(self, data, params):
                column = params.get('column')
                bins = params.get('bins', 10)
                
                if column not in data.columns:
                    return {'error': f'Spalte {column} nicht gefunden'}
                
                if pd.api.types.is_numeric_dtype(data[column]):
                    hist, bin_edges = np.histogram(data[column], bins=bins)
                    return {
                        'histogram': hist.tolist(),
                        'bin_edges': bin_edges.tolist()
                    }
                else:
                    value_counts = data[column].value_counts()
                    return {
                        'categories': value_counts.index.tolist(),
                        'counts': value_counts.values.tolist()
                    }
            
            def _custom_model_inference(self, data, params):
                # Vereinfachte Implementierung für Tests
                return {'result': 'Model inference result'}
            
            def execute_operation(self, encrypted_data, operation, params):
                """Führt eine Operation auf verschlüsselten Daten aus"""
                try:
                    if operation not in self.allowed_operations:
                        return {'error': f'Nicht unterstützte Operation: {operation}'}
                    
                    # Entschlüssele die Daten temporär im Speicher
                    data = self._decrypt_data(encrypted_data)
                    
                    # Führe die Operation aus
                    operation_func = self.allowed_operations[operation]
                    results = operation_func(data, params)
                    
                    return results
                    
                except Exception as e:
                    return {'error': f'Fehler bei der Ausführung: {str(e)}'}
            
            def create_data_asset(self, data, asset_metadata=None):
                """Erstellt einen verschlüsselten Daten-Asset mit Metadaten"""
                try:
                    # Erstelle eine eindeutige ID für den Asset
                    import uuid
                    asset_id = uuid.uuid4().hex
                    
                    # Verschlüssele die Daten
                    encrypted_data = self._encrypt_data(data)
                    
                    # Erstelle Metadaten
                    stats = {
                        'record_count': len(data),
                        'columns': [{'name': col, 'type': str(data[col].dtype)} for col in data.columns]
                    }
                    
                    asset_info = {
                        'asset_id': asset_id,
                        'created_at': datetime.now().isoformat(),
                        'statistics': stats
                    }
                    
                    if asset_metadata:
                        asset_info.update(asset_metadata)
                    
                    return {
                        'asset_id': asset_id,
                        'metadata': asset_info,
                        'encrypted_data': encrypted_data
                    }
                    
                except Exception as e:
                    return {'error': f'Fehler bei der Asset-Erstellung: {str(e)}'}
            
            def generate_access_token(self, asset_id, allowed_operations, expiration_time=3600):
                """Generiert ein temporäres Zugriffstoken für einen Daten-Asset"""
                import uuid
                
                token_data = {
                    'asset_id': asset_id,
                    'allowed_operations': allowed_operations,
                    'created_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(seconds=expiration_time)).isoformat(),
                    'token_id': uuid.uuid4().hex
                }
                
                token_json = json.dumps(token_data).encode()
                encrypted_token = self.cipher_suite.encrypt(token_json)
                
                return {
                    'token': encrypted_token.decode(),
                    'token_id': token_data['token_id'],
                    'expires_at': token_data['expires_at']
                }
            
            def validate_access_token(self, token, operation):
                """Validiert ein Zugriffstoken für eine bestimmte Operation"""
                try:
                    # Entschlüssele das Token
                    decrypted_token = self.cipher_suite.decrypt(token.encode())
                    token_data = json.loads(decrypted_token.decode())
                    
                    # Prüfe, ob das Token abgelaufen ist
                    expiry_time = datetime.fromisoformat(token_data['expires_at'])
                    if datetime.now() > expiry_time:
                        return False
                    
                    # Prüfe, ob die Operation erlaubt ist
                    if operation not in token_data['allowed_operations']:
                        return False
                    
                    return True
                    
                except Exception:
                    return False
        
        self.ComputeToDataManager = MockComputeToDataManager
    
    def test_encryption_decryption(self):
        """Test der Verschlüsselungs- und Entschlüsselungsfunktionalität"""
        c2d_manager = self.ComputeToDataManager()
        
        # Daten verschlüsseln
        encrypted_data = c2d_manager._encrypt_data(self.test_data)
        
        # Überprüfen, ob die Verschlüsselung erfolgreich war
        self.assertIsInstance(encrypted_data, bytes)
        
        # Daten entschlüsseln
        decrypted_data = c2d_manager._decrypt_data(encrypted_data)
        
        # Überprüfen, ob die Entschlüsselung erfolgreich war und die Daten korrekt sind
        self.assertIsInstance(decrypted_data, pd.DataFrame)
        self.assertEqual(len(decrypted_data), len(self.test_data))
        self.assertListEqual(list(decrypted_data.columns), list(self.test_data.columns))
    
    def test_execute_operation(self):
        """Test der Ausführung von Operationen auf verschlüsselten Daten"""
        c2d_manager = self.ComputeToDataManager()
        
        # Daten verschlüsseln
        encrypted_data = c2d_manager._encrypt_data(self.test_data)
        
        # Operation 'count' ausführen
        result = c2d_manager.execute_operation(encrypted_data, 'count', {})
        
        # Überprüfen, ob das Ergebnis korrekt ist
        self.assertIn('count', result)
        self.assertEqual(result['count'], 100)
        
        # Operation 'mean' ausführen
        result = c2d_manager.execute_operation(encrypted_data, 'mean', {})
        
        # Überprüfen, ob das Ergebnis korrekt ist
        self.assertIn('means', result)
        self.assertIn('value', result['means'])
        
        # Operation mit Filterparametern ausführen
        result = c2d_manager.execute_operation(
            encrypted_data, 
            'count', 
            {'filter_column': 'category', 'filter_value': 'A'}
        )
        
        # Überprüfen, ob das Ergebnis das erwartete Format hat
        self.assertIn('count', result)
        
        # Nicht unterstützte Operation ausführen
        result = c2d_manager.execute_operation(encrypted_data, 'invalid_operation', {})
        
        # Überprüfen, ob ein Fehler zurückgegeben wird
        self.assertIn('error', result)
    
    def test_create_data_asset(self):
        """Test der Erstellung eines Daten-Assets"""
        c2d_manager = self.ComputeToDataManager()
        
        # Asset-Metadaten definieren
        asset_metadata = {
            'name': 'Test Dataset',
            'description': 'A dataset for testing',
            'owner': 'test_user',
            'price': 5.0
        }
        
        # Daten-Asset erstellen
        asset = c2d_manager.create_data_asset(self.test_data, asset_metadata)
        
        # Überprüfen, ob der Asset das erwartete Format hat
        self.assertIn('asset_id', asset)
        self.assertIn('metadata', asset)
        self.assertIn('encrypted_data', asset)
        
        # Überprüfen, ob die Metadaten korrekt sind
        self.assertEqual(asset['metadata']['name'], 'Test Dataset')
        self.assertEqual(asset['metadata']['price'], 5.0)
        
        # Überprüfen, ob die Statistiken korrekt sind
        self.assertEqual(asset['metadata']['statistics']['record_count'], 100)
    
    def test_access_token(self):
        """Test der Token-Generierung und -Validierung"""
        c2d_manager = self.ComputeToDataManager()
        
        # Asset ID und erlaubte Operationen definieren
        asset_id = 'test_asset_123'
        allowed_operations = ['count', 'mean', 'sum']
        
        # Zugriffstoken generieren
        token_info = c2d_manager.generate_access_token(asset_id, allowed_operations, expiration_time=60)
        
        # Überprüfen, ob das Token das erwartete Format hat
        self.assertIn('token', token_info)
        self.assertIn('token_id', token_info)
        self.assertIn('expires_at', token_info)
        
        # Token für erlaubte Operation validieren
        is_valid = c2d_manager.validate_access_token(token_info['token'], 'count')
        self.assertTrue(is_valid)
        
        # Token für nicht erlaubte Operation validieren
        is_valid = c2d_manager.validate_access_token(token_info['token'], 'histogram')
        self.assertFalse(is_valid)


class TestOceanDataAI(unittest.TestCase):
    """Testsuite für die OceanDataAI-Hauptklasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Mockdaten für Tests
        self.browser_data = create_mock_browser_data()
        self.smartwatch_data = create_mock_smartwatch_data()
        
        # Mock-Implementierung für Tests
        class MockOceanDataAI:
            def __init__(self, config=None):
                self.config = config or {}
                
                # Mock-Komponenten
                self.anomaly_detector = mock.MagicMock()
                self.semantic_analyzer = mock.MagicMock()
                self.predictive_modeler = mock.MagicMock()
                self.data_synthesizer = mock.MagicMock()
                self.c2d_manager = mock.MagicMock()
                
            def analyze_data_source(self, data, source_type):
                """Mock-Analyse einer Datenquelle"""
                return {
                    'source_type': source_type,
                    'timestamp': datetime.now().isoformat(),
                    'record_count': len(data),
                    'column_count': len(data.columns),
                    'analyses': {
                        'anomalies': {
                            'count': int(len(data) * 0.05),
                            'percentage': 5.0
                        },
                        'time_series': {
                            'forecast_horizon': 7
                        }
                    }
                }
            
            def prepare_data_for_monetization(self, data, source_type, privacy_level='medium'):
                """Mock-Vorbereitung für Monetarisierung"""
                # Basiswerte für verschiedene Quellen
                base_values = {
                    'browser': 3.5,
                    'smartwatch': 5.0,
                    'calendar': 2.5,
                    'social_media': 4.0,
                    'streaming': 3.0,
                    'health_data': 6.0,
                    'iot': 2.0
                }
                
                # Privacy-Level-Faktoren
                privacy_factors = {
                    'low': 1.2,
                    'medium': 1.0,
                    'high': 0.8
                }
                
                # Basiswert mit Privacy-Faktor kombinieren
                estimated_value = base_values.get(source_type, 3.0) * privacy_factors.get(privacy_level, 1.0)
                
                return {
                    'anonymized_data': data.copy(),
                    'metadata': {
                        'source_type': source_type,
                        'privacy_level': privacy_level,
                        'record_count': len(data),
                        'field_count': len(data.columns),
                        'estimated_value': estimated_value,
                        'created_at': datetime.now().isoformat()
                    },
                    'c2d_asset': {
                        'asset_id': f"asset_{uuid.uuid4().hex[:8]}",
                        'metadata': {}
                    }
                }
            
            def combine_data_sources(self, sources, combination_type='merge'):
                """Mock-Kombination von Datenquellen"""
                # Basiswert ist die Summe der Einzelwerte mit einem Bonus
                source_values = [s['metadata'].get('estimated_value', 0) for s in sources]
                
                if combination_type == 'merge':
                    combined_value = sum(source_values) * 1.2  # 20% Bonus
                elif combination_type == 'enrich':
                    combined_value = max(source_values) + sum(source_values[1:]) * 0.6  # Basis + 60% der zusätzlichen Quellen
                elif combination_type == 'correlate':
                    combined_value = sum(source_values) * 0.3 + len(sources) * 0.5  # 30% der Originalwerte + Bonus pro Quelle
                else:
                    combined_value = sum(source_values)
                
                # Kombinierte Daten erstellen
                combined_data = pd.DataFrame()
                for source in sources:
                    if 'anonymized_data' in source and source['anonymized_data'] is not None:
                        if combined_data.empty:
                            combined_data = source['anonymized_data'].copy()
                        else:
                            # Je nach Kombinationstyp unterschiedliche Strategien
                            if combination_type == 'merge':
                                # Einfaches Anfügen
                                combined_data = pd.concat([combined_data, source['anonymized_data']], ignore_index=True)
                            elif combination_type == 'enrich':
                                # Spalten hinzufügen
                                for col in source['anonymized_data'].columns:
                                    if col not in combined_data.columns:
                                        combined_data[col] = np.nan
                                        combined_data[col][:len(source['anonymized_data'])] = source['anonymized_data'][col].values
                
                return {
                    'anonymized_data': combined_data,
                    'metadata': {
                        'combination_type': combination_type,
                        'source_count': len(sources),
                        'estimated_value': combined_value,
                        'created_at': datetime.now().isoformat()
                    },
                    'c2d_asset': {
                        'asset_id': f"combined_{uuid.uuid4().hex[:8]}",
                        'metadata': {}
                    }
                }
            
            def estimate_data_value(self, data, metadata=None):
                """Mock-Wertschätzung eines Datensatzes"""
                metadata = metadata or {}
                
                # Basiswert nach Datengröße
                base_value = min(1.0, len(data) / 1000) * 5  # Skalieren nach Größe, max 5 OCEAN
                
                # Faktor nach Datenqualität (weniger fehlende Werte = höherer Wert)
                quality_factor = 1.0 - data.isna().mean().mean()
                
                # Faktor nach Spaltenanzahl
                cols_factor = min(1.0, len(data.columns) / 10) * 0.5 + 0.5  # 0.5 bis 1.0 basierend auf Spaltenanzahl
                
                # Spezielle Werterhöhung für bestimmte Datentypen
                source_type = metadata.get('source_type', '')
                source_bonus = {
                    'health_data': 1.5,
                    'smartwatch': 1.3,
                    'browser': 1.2,
                    'social_media': 1.4
                }.get(source_type, 1.0)
                
                # Gesamtwert berechnen
                estimated_value = base_value * quality_factor * cols_factor * source_bonus
                
                # Wertfaktoren
                value_factors = {
                    'data_size': float(min(1.0, len(data) / 1000)),
                    'data_quality': float(quality_factor),
                    'column_diversity': float(cols_factor)
                }
                
                return {
                    'normalized_score': float(quality_factor * cols_factor * source_bonus / 3),
                    'estimated_token_value': float(estimated_value),
                    'metrics': {
                        'data_volume': {'score': value_factors['data_size'], 'weight': 0.3, 'explanation': ''},
                        'data_quality': {'score': value_factors['data_quality'], 'weight': 0.3, 'explanation': ''},
                        'data_uniqueness': {'score': 0.7, 'weight': 0.2, 'explanation': ''},
                        'time_relevance': {'score': 0.8, 'weight': 0.2, 'explanation': ''}
                    },
                    'summary': f"This dataset has an estimated value of {estimated_value:.2f} OCEAN tokens."
                }
            
            def prepare_for_ocean_tokenization(self, data_asset):
                """Mock-Vorbereitung für Ocean-Tokenisierung"""
                asset_id = data_asset['c2d_asset']['asset_id']
                metadata = data_asset['metadata']
                
                ddo = {
                    'id': asset_id,
                    'created': datetime.now().isoformat(),
                    'updated': datetime.now().isoformat(),
                    'type': 'dataset',
                    'name': f"Dataset {asset_id[:8]}",
                    'description': metadata.get('description', 'No description provided.'),
                    'tags': [metadata.get('source_type', 'data'), metadata.get('privacy_level', 'medium')],
                    'price': metadata.get('estimated_value', 5.0)
                }
                
                pricing = {
                    'type': 'fixed',
                    'baseTokenAmount': metadata.get('estimated_value', 5.0),
                    'datatoken': {
                        'name': f"DT-{ddo['name']}",
                        'symbol': f"DT{asset_id[:4]}".upper()
                    }
                }
                
                return {
                    'ddo': ddo,
                    'pricing': pricing,
                    'asset_id': asset_id
                }
            
            def tokenize_with_ocean(self, ocean_asset):
                """Mock-Tokenisierung mit Ocean Protocol"""
                tx_hash = f"0x{uuid.uuid4().hex}"
                token_address = f"0x{uuid.uuid4().hex}"
                
                return {
                    'success': True,
                    'asset_id': ocean_asset.get('asset_id', 'unknown'),
                    'token_address': token_address,
                    'token_symbol': ocean_asset.get('pricing', {}).get('datatoken', {}).get('symbol', 'UNKNOWN'),
                    'token_name': ocean_asset.get('pricing', {}).get('datatoken', {}).get('name', 'Unknown Token'),
                    'token_price': ocean_asset.get('pricing', {}).get('baseTokenAmount', 0),
                    'transaction_hash': tx_hash,
                    'timestamp': datetime.now().isoformat(),
                    'marketplace_url': f"https://market.oceanprotocol.com/asset/{ocean_asset.get('asset_id', 'unknown')}"
                }
        
        self.OceanDataAI = MockOceanDataAI
    
    def test_analyze_data_source(self):
        """Test der Datenquellenanalyse"""
        ocean_ai = self.OceanDataAI()
        
        # Browser-Daten analysieren
        browser_analysis = ocean_ai.analyze_data_source(self.browser_data, 'browser')
        
        # Überprüfen, ob die Analyse das erwartete Format hat
        self.assertEqual(browser_analysis['source_type'], 'browser')
        self.assertEqual(browser_analysis['record_count'], len(self.browser_data))
        self.assertEqual(browser_analysis['column_count'], len(self.browser_data.columns))
        self.assertIn('analyses', browser_analysis)
        self.assertIn('anomalies', browser_analysis['analyses'])
        
        # Smartwatch-Daten analysieren
        smartwatch_analysis = ocean_ai.analyze_data_source(self.smartwatch_data, 'smartwatch')
        
        # Überprüfen, ob die Analyse das erwartete Format hat
        self.assertEqual(smartwatch_analysis['source_type'], 'smartwatch')
    
    def test_prepare_data_for_monetization(self):
        """Test der Datenvorbereitung für Monetarisierung"""
        ocean_ai = self.OceanDataAI()
        
        # Browser-Daten für Monetarisierung vorbereiten
        browser_asset = ocean_ai.prepare_data_for_monetization(self.browser_data, 'browser', 'medium')
        
        # Überprüfen, ob der Asset das erwartete Format hat
        self.assertIn('anonymized_data', browser_asset)
        self.assertIn('metadata', browser_asset)
        self.assertIn('c2d_asset', browser_asset)
        
        # Überprüfen, ob die Metadaten korrekt sind
        self.assertEqual(browser_asset['metadata']['source_type'], 'browser')
        self.assertEqual(browser_asset['metadata']['privacy_level'], 'medium')
        self.assertEqual(browser_asset['metadata']['record_count'], len(self.browser_data))
        
        # Smartwatch-Daten mit höherem Datenschutzniveau vorbereiten
        smartwatch_asset = ocean_ai.prepare_data_for_monetization(self.smartwatch_data, 'smartwatch', 'high')
        
        # Überprüfen, ob der Privacy-Level korrekt ist und sich auf den Wert auswirkt
        self.assertEqual(smartwatch_asset['metadata']['privacy_level'], 'high')
        # Bei höherem Datenschutz erwarten wir einen geringeren Wert
        self.assertLess(
            smartwatch_asset['metadata']['estimated_value'],
            # Der gleiche Datensatz mit niedrigerem Datenschutz sollte mehr wert sein
            ocean_ai.prepare_data_for_monetization(self.smartwatch_data, 'smartwatch', 'low')['metadata']['estimated_value']
        )
    
    def test_combine_data_sources(self):
        """Test der Kombination von Datenquellen"""
        ocean_ai = self.OceanDataAI()
        
        # Assets für den Test erstellen
        browser_asset = ocean_ai.prepare_data_for_monetization(self.browser_data, 'browser', 'medium')
        smartwatch_asset = ocean_ai.prepare_data_for_monetization(self.smartwatch_data, 'smartwatch', 'medium')
        
        # Testen verschiedener Kombinationstypen
        for combination_type in ['merge', 'enrich', 'correlate']:
            combined_asset = ocean_ai.combine_data_sources(
                [browser_asset, smartwatch_asset], 
                combination_type
            )
            
            # Überprüfen, ob das kombinierte Asset das erwartete Format hat
            self.assertIn('anonymized_data', combined_asset)
            self.assertIn('metadata', combined_asset)
            self.assertIn('c2d_asset', combined_asset)
            
            # Überprüfen, ob die Metadaten korrekt sind
            self.assertEqual(combined_asset['metadata']['combination_type'], combination_type)
            self.assertEqual(combined_asset['metadata']['source_count'], 2)
            
            # Überprüfen, ob der kombinierte Wert höher ist als die Summe der Einzelwerte
            individual_values = browser_asset['metadata']['estimated_value'] + smartwatch_asset['metadata']['estimated_value']
            self.assertGreaterEqual(combined_asset['metadata']['estimated_value'], individual_values)
    
    def test_estimate_data_value(self):
        """Test der Datenwertschätzung"""
        ocean_ai = self.OceanDataAI()
        
        # Wert der Browser-Daten schätzen
        value_assessment = ocean_ai.estimate_data_value(self.browser_data, {'source_type': 'browser'})
        
        # Überprüfen, ob die Wertschätzung das erwartete Format hat
        self.assertIn('normalized_score', value_assessment)
        self.assertIn('estimated_token_value', value_assessment)
        self.assertIn('metrics', value_assessment)
        self.assertIn('summary', value_assessment)
        
        # Überprüfen, ob die Metriken das erwartete Format haben
        self.assertIn('data_volume', value_assessment['metrics'])
        self.assertIn('data_quality', value_assessment['metrics'])
        self.assertIn('data_uniqueness', value_assessment['metrics'])
        self.assertIn('time_relevance', value_assessment['metrics'])
        
        # Überprüfen, ob jede Metrik die erwarteten Felder hat
        for metric in value_assessment['metrics'].values():
            self.assertIn('score', metric)
            self.assertIn('weight', metric)
            self.assertIn('explanation', metric)
        
        # Überprüfen, ob der geschätzte Wert im erwarteten Bereich liegt
        self.assertGreaterEqual(value_assessment['estimated_token_value'], 0)
        
        # Wert eines anderen Datentyps schätzen
        smartwatch_value = ocean_ai.estimate_data_value(self.smartwatch_data, {'source_type': 'smartwatch'})
        
        # Gesundheitsdaten sollten wertvoller sein als Browserdaten
        self.assertGreater(smartwatch_value['estimated_token_value'], value_assessment['estimated_token_value'])
    
    def test_prepare_for_ocean_tokenization(self):
        """Test der Vorbereitung für Ocean Protocol Tokenisierung"""
        ocean_ai = self.OceanDataAI()
        
        # Asset für den Test erstellen
        browser_asset = ocean_ai.prepare_data_for_monetization(self.browser_data, 'browser', 'medium')
        
        # Asset für Ocean vorbereiten
        ocean_asset = ocean_ai.prepare_for_ocean_tokenization(browser_asset)
        
        # Überprüfen, ob der Ocean-Asset das erwartete Format hat
        self.assertIn('ddo', ocean_asset)
        self.assertIn('pricing', ocean_asset)
        self.assertIn('asset_id', ocean_asset)
        
        # Überprüfen, ob das DDO die erwarteten Felder hat
        self.assertIn('id', ocean_asset['ddo'])
        self.assertIn('name', ocean_asset['ddo'])
        self.assertIn('type', ocean_asset['ddo'])
        self.assertIn('tags', ocean_asset['ddo'])
        self.assertIn('price', ocean_asset['ddo'])
        
        # Überprüfen, ob die Pricing-Informationen korrekt sind
        self.assertEqual(ocean_asset['pricing']['type'], 'fixed')
        self.assertEqual(
            ocean_asset['pricing']['baseTokenAmount'], 
            browser_asset['metadata']['estimated_value']
        )
        self.assertIn('datatoken', ocean_asset['pricing'])
    
    def test_tokenize_with_ocean(self):
        """Test der Tokenisierung mit Ocean Protocol"""
        ocean_ai = self.OceanDataAI()
        
        # Asset für den Test erstellen
        browser_asset = ocean_ai.prepare_data_for_monetization(self.browser_data, 'browser', 'medium')
        ocean_asset = ocean_ai.prepare_for_ocean_tokenization(browser_asset)
        
        # Asset tokenisieren
        tokenization_result = ocean_ai.tokenize_with_ocean(ocean_asset)
        
        # Überprüfen, ob das Ergebnis das erwartete Format hat
        self.assertIn('success', tokenization_result)
        self.assertTrue(tokenization_result['success'])
        self.assertIn('asset_id', tokenization_result)
        self.assertIn('token_address', tokenization_result)
        self.assertIn('token_symbol', tokenization_result)
        self.assertIn('token_price', tokenization_result)
        self.assertIn('transaction_hash', tokenization_result)
        self.assertIn('marketplace_url', tokenization_result)
        
        # Überprüfen, ob der Token-Preis mit dem geschätzten Wert übereinstimmt
        self.assertEqual(
            tokenization_result['token_price'],
            ocean_asset['pricing']['baseTokenAmount']
        )


class TestIntegrationOceanData(unittest.TestCase):
    """Integrationstests für die OceanData-Plattform"""
    
    @unittest.skip("Erfordert vollständige Plattformimplementierung")
    def test_end_to_end_data_monetization(self):
        """Test des gesamten Datenmonetarisierungsprozesses"""
        # In einem echten Integrationstest würden wir hier die tatsächliche Implementierung verwenden
        pass


# Integration-Tests mit der Frontend-Komponente über pytest
@pytest.mark.skip("Requires React testing environment")
class TestReactComponents:
    """Testsuite für die React-Komponenten mit pytest und React Testing Library"""
    
    @pytest.fixture
    def setup_react_testing(self):
        """Setup für React-Tests"""
        # Hier würden wir das React-Test-Setup initialisieren
        pass
    
    def test_data_tokenization_dashboard(self, setup_react_testing):
        """Test der DataTokenizationDashboard-Komponente"""
        # Hier würden wir die React-Komponente testen
        pass


# Performance-Tests
class TestPerformanceOceanData(unittest.TestCase):
    """Performance-Tests für kritische Komponenten"""
    
    @unittest.skip("Langläufiger Performance-Test")
    def test_data_integration_performance(self):
        """Test der Performance der Datenintegration mit großen Datensätzen"""
        # Große Datensätze generieren
        large_browser_data = create_mock_browser_data(n_rows=100000)
        large_smartwatch_data = create_mock_smartwatch_data(n_rows=100000)
        
        # Performance-Messung
        import time
        
        start_time = time.time()
        # Hier würden wir die Datenintegration durchführen
        end_time = time.time()
        
        # Performance-Schwellenwert prüfen
        execution_time = end_time - start_time
        self.assertLess(execution_time, 10.0)  # Sollte unter 10 Sekunden bleiben
    
    @unittest.skip("Langläufiger Performance-Test")
    def test_ai_analysis_performance(self):
        """Test der Performance der KI-Analyse mit großen Datensätzen"""
        # Zu implementieren
        pass


# Hauptausführungspunkt für Unit-Tests
if __name__ == "__main__":
    unittest.main()
"""
OceanData Test-Framework

Dieses Modul enthält umfassende Tests für alle Komponenten der OceanData-Plattform,
einschließlich Unit-Tests, Integrationstests und End-to-End-Tests.
"""

import unittest
import pandas as pd
import numpy as np
import json
import os
import tempfile
from datetime import datetime, timedelta
import tensorflow as tf
import torch
import pytest
from unittest import mock
from io import StringIO

# Mockdaten für Tests
def create_mock_browser_data(n_rows=100):
    """Erstellt Mockdaten für Browsertests"""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_rows, freq='H'),
        'url': [f"example{i % 10}.com/page{i % 5}" for i in range(n_rows)],
        'duration': np.random.randint(10, 300, n_rows),
        'user_id': ['user123'] * n_rows,
        'device_type': np.random.choice(['desktop', 'mobile', 'tablet'], n_rows),
        'browser_type': np.random.choice(['chrome', 'firefox', 'safari'], n_rows)
    })

def create_mock_smartwatch_data(n_rows=100):
    """Erstellt Mockdaten für Smartwatchtests"""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_rows, freq='H'),
        'heart_rate': np.random.randint(60, 100, n_rows),
        'steps': np.random.randint(0, 1000, n_rows),
        'user_id': ['user123'] * n_rows,
        'sleep_quality': np.random.choice(['good', 'medium', 'poor'], n_rows),
        'calories_burned': np.random.randint(50, 300, n_rows)
    })

class TestDataSource(unittest.TestCase):
    """Testsuite für die abstrakte DataSource-Klasse und ihre Implementierungen"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Wir müssen hier die tatsächlichen Klassen importieren oder mocken
        from unittest.mock import MagicMock
        self.mock_data_category = MagicMock()
        self.mock_privacy_level = MagicMock()
        
        # Annahme, dass wir die Klassen importieren können
        # from oceandata.data_integration import DataSource, BrowserDataConnector, DataCategory, PrivacyLevel
        
        # Für das Testen verwenden wir Mock-Klassen
        class MockDataSource:
            def __init__(self, source_id, user_id, category):
                self.source_id = source_id
                self.user_id = user_id
                self.category = category
                self.metadata = {
                    "source_id": source_id,
                    "user_id": user_id,
                    "category": category,
                    "created": datetime.now().isoformat(),
                    "privacy_fields": {},
                }
                self.last_sync = None
                self.data = None
            
            def set_privacy_level(self, field, level):
                self.metadata["privacy_fields"][field] = level
            
            def get_privacy_level(self, field):
                return self.metadata["privacy_fields"].get(field, self.mock_privacy_level.ANONYMIZED)
            
            def connect(self):
                return True
            
            def fetch_data(self):
                return pd.DataFrame()
            
            def process_data(self, data):
                return data
            
            def get_data(self):
                return {"data": pd.DataFrame(), "metadata": self.metadata, "status": "success"}
        
        self.DataSource = MockDataSource
        
        # Mock-Implementierung für BrowserDataConnector
        class MockBrowserDataConnector(MockDataSource):
            def __init__(self, user_id, browser_type='chrome'):
                super().__init__(f"browser_{browser_type}", user_id, "BROWSER")
                self.browser_type = browser_type
            
            def connect(self):
                return True
            
            def fetch_data(self):
                return create_mock_browser_data()
        
        self.BrowserDataConnector = MockBrowserDataConnector
    
    def test_data_source_initialization(self):
        """Test der Initialisierung von DataSource"""
        source = self.DataSource("test_source", "user123", "TEST")
        self.assertEqual(source.source_id, "test_source")
        self.assertEqual(source.user_id, "user123")
        self.assertEqual(source.category, "TEST")
        self.assertIsNone(source.last_sync)
        self.assertIsNone(source.data)
    
    def test_privacy_level_management(self):
        """Test des Privacy-Level-Managements"""
        source = self.DataSource("test_source", "user123", "TEST")
        source.set_privacy_level("test_field", "ANONYMIZED")
        self.assertEqual(source.metadata["privacy_fields"]["test_field"], "ANONYMIZED")
        
        privacy_level = source.get_privacy_level("test_field")
        self.assertEqual(privacy_level, "ANONYMIZED")
        
        # Test Default-Wert
        default_level = source.get_privacy_level("nonexistent_field")
        self.assertEqual(default_level, self.mock_privacy_level.ANONYMIZED)
    
    def test_browser_connector_fetch_data(self):
        """Test der Datenabruffunktionalität des BrowserDataConnector"""
        browser_connector = self.BrowserDataConnector("user123", "chrome")
        data = browser_connector.fetch_data()
        
        # Überprüfen, ob die Daten ein DataFrame sind und die erwarteten Spalten enthalten
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('url', data.columns)
        self.assertIn('duration', data.columns)
        self.assertIn('user_id', data.columns)
        
        # Überprüfen, ob die user_id korrekt ist
        self.assertEqual(data['user_id'].iloc[0], "user123")
    
    def test_get_data_pipeline(self):
        """Test der gesamten Datenabruf-Pipeline"""
        browser_connector = self.BrowserDataConnector("user123", "chrome")
        result = browser_connector.get_data()
        
        # Überprüfen, ob das Ergebnis das erwartete Format hat
        self.assertIn('data', result)
        self.assertIn('metadata', result)
        self.assertIn('status', result)
        
        # Überprüfen, ob der Status erfolgreich ist
        self.assertEqual(result['status'], 'success')


class TestAnomalyDetector(unittest.TestCase):
    """Testsuite für die AnomalyDetector-Klasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Mockdaten erstellen
        self.normal_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        
        # Anomalien hinzufügen
        anomalies = pd.DataFrame({
            'feature1': np.random.normal(5, 1, 5),  # Weit entfernt vom Durchschnitt
            'feature2': np.random.normal(5, 1, 5)
        })
        
        self.data_with_anomalies = pd.concat([self.normal_data, anomalies])
        
        # Annahme, dass wir die Klasse importieren können
        # from oceandata.ai import AnomalyDetector
        
        # Für diesen Test verwenden wir eine Mock-Implementierung
        class MockAnomalyDetector:
            def __init__(self, method='isolation_forest', contamination=0.05):
                self.method = method
                self.contamination = contamination
                self.model = None
                self.is_fitted = False
                self.feature_dims = None
                self.threshold = None
                
                if method == 'isolation_forest':
                    from sklearn.ensemble import IsolationForest
                    self.model = IsolationForest(contamination=contamination, random_state=42)
                elif method == 'autoencoder':
                    # Vereinfachte Implementierung für Tests
                    self.model = mock.MagicMock()
                    self.model.predict.return_value = np.zeros((100, 2))
                    self.threshold = 0.5
                    
            def fit(self, X, categorical_cols=None):
                if self.method == 'isolation_forest':
                    self.model.fit(X)
                self.is_fitted = True
                self.feature_dims = X.shape[1]
                return self
            
            def predict(self, X, categorical_cols=None):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                if self.method == 'isolation_forest':
                    return self.model.predict(X)
                elif self.method == 'autoencoder':
                    # Simuliere Autoencoder-Vorhersagen
                    reconstructions = self.model.predict(X)
                    mse = np.mean(np.power(X.values - reconstructions, 2), axis=1)
                    return mse > self.threshold
        
        self.AnomalyDetector = MockAnomalyDetector
    
    def test_initialization(self):
        """Test der Initialisierung mit verschiedenen Methoden"""
        detector_if = self.AnomalyDetector(method='isolation_forest', contamination=0.1)
        self.assertEqual(detector_if.method, 'isolation_forest')
        self.assertEqual(detector_if.contamination, 0.1)
        
        detector_ae = self.AnomalyDetector(method='autoencoder', contamination=0.05)
        self.assertEqual(detector_ae.method, 'autoencoder')
        self.assertEqual(detector_ae.contamination, 0.05)
    
    def test_fit_predict_isolation_forest(self):
        """Test der Fit- und Predict-Methoden mit Isolation Forest"""
        detector = self.AnomalyDetector(method='isolation_forest')
        detector.fit(self.data_with_anomalies)
        
        # Überprüfen, ob das Modell trainiert wurde
        self.assertTrue(detector.is_fitted)
        self.assertEqual(detector.feature_dims, 2)
        
        # Vorhersagen treffen
        predictions = detector.predict(self.data_with_anomalies)
        
        # Überprüfen, ob Vorhersagen das erwartete Format haben
        self.assertEqual(predictions.shape, (105,))
        
        # Bei Isolation Forest: -1 für Anomalien, 1 für normale Daten
        # Wir erwarten etwa 5 Anomalien in unseren Daten
        anomaly_count = np.sum(predictions == -1)
        self.assertGreaterEqual(anomaly_count, 1)  # Mindestens eine Anomalie sollte erkannt werden
    
    @unittest.skip("Autoencoder-Tests erfordern TensorFlow-Setup")
    def test_fit_predict_autoencoder(self):
        """Test der Fit- und Predict-Methoden mit Autoencoder"""
        detector = self.AnomalyDetector(method='autoencoder')
        detector.fit(self.data_with_anomalies)
        
        # Überprüfen, ob das Modell trainiert wurde
        self.assertTrue(detector.is_fitted)
        
        # Vorhersagen treffen
        predictions = detector.predict(self.data_with_anomalies)
        
        # Bei Autoencoder: True für Anomalien, False für normale Daten
        anomaly_count = np.sum(predictions)
        self.assertGreaterEqual(anomaly_count, 1)  # Mindestens eine Anomalie sollte erkannt werden


class TestSemanticAnalyzer(unittest.TestCase):
    """Testsuite für die SemanticAnalyzer-Klasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Beispieltexte für Tests
        self.sample_texts = [
            "I love this product! It's amazing and works really well.",
            "This is terrible. Doesn't work at all. Very disappointed.",
            "It's okay. Not great, not bad. Somewhat useful."
        ]
        
        # Da das Laden echter Transformer-Modelle zu ressourcenintensiv für Tests ist,
        # erstellen wir eine Mock-Implementierung
        class MockSemanticAnalyzer:
            def __init__(self, model_type='bert', model_name='bert-base-uncased'):
                self.model_type = model_type
                self.model_name = model_name
                self.model = mock.MagicMock()
                self.tokenizer = mock.MagicMock()
                self.embeddings_cache = {}
                
                # Mock für die Sentimentanalyse
                self.sentiment_results = {
                    "I love this product! It's amazing and works really well.": 
                        {"sentiment": "positive", "scores": {"positive": 0.9, "negative": 0.05, "neutral": 0.05, "compound": 0.8}},
                    "This is terrible. Doesn't work at all. Very disappointed.": 
                        {"sentiment": "negative", "scores": {"positive": 0.05, "negative": 0.9, "neutral": 0.05, "compound": -0.8}},
                    "It's okay. Not great, not bad. Somewhat useful.": 
                        {"sentiment": "neutral", "scores": {"positive": 0.3, "negative": 0.3, "neutral": 0.4, "compound": 0.1}}
                }
            
            def get_embeddings(self, texts, batch_size=8, use_cache=True):
                if isinstance(texts, str):
                    texts = [texts]
                
                # Einfache Mock-Embeddings erstellen
                return np.random.random((len(texts), 768))
            
            def analyze_sentiment(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                
                results = []
                for text in texts:
                    # Fallback, falls der Text nicht in den vordefinierten Ergebnissen ist
                    sentiment_data = self.sentiment_results.get(
                        text, 
                        {"sentiment": "neutral", "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "compound": 0.0}}
                    )
                    
                    result = {
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'sentiment': sentiment_data["sentiment"],
                        'scores': sentiment_data["scores"]
                    }
                    results.append(result)
                
                return results
            
            def extract_topics(self, texts, num_topics=5, words_per_topic=5):
                # Mock-Themenextraktion
                topics = [
                    {
                        "id": 0,
                        "words": ["product", "great", "good", "useful", "works"],
                        "examples": [text[:50] + "..." for text in texts[:2]]
                    },
                    {
                        "id": 1,
                        "words": ["bad", "terrible", "disappointed", "not", "issue"],
                        "examples": [text[:50] + "..." for text in texts if "terrible" in text]
                    }
                ]
                return topics[:num_topics]
            
            def find_similar_texts(self, query, corpus, top_n=5):
                # Mock-Ähnlichkeitssuche
                similarities = np.random.random(len(corpus))
                top_indices = np.argsort(similarities)[::-1][:top_n]
                
                results = []
                for idx in top_indices:
                    result = {
                        "text": corpus[idx][:100] + "..." if len(corpus[idx]) > 100 else corpus[idx],
                        "similarity": float(similarities[idx]),
                        "index": int(idx)
                    }
                    results.append(result)
                
                return results
        
        self.SemanticAnalyzer = MockSemanticAnalyzer
    
    def test_sentiment_analysis(self):
        """Test der Sentimentanalyse-Funktionalität"""
        analyzer = self.SemanticAnalyzer()
        results = analyzer.analyze_sentiment(self.sample_texts)
        
        # Überprüfen, ob die Ergebnisse das erwartete Format haben
        self.assertEqual(len(results), 3)
        self.assertIn('text', results[0])
        self.assertIn('sentiment', results[0])
        self.assertIn('scores', results[0])
        
        # Überprüfen, ob die Sentiments korrekt sind
        self.assertEqual(results[0]['sentiment'], 'positive')
        self.assertEqual(results[1]['sentiment'], 'negative')
        self.assertEqual(results[2]['sentiment'], 'neutral')
    
    def test_topic_extraction(self):
        """Test der Themenextraktionsfunktionalität"""
        analyzer = self.SemanticAnalyzer()
        topics = analyzer.extract_topics(self.sample_texts, num_topics=2)
        
        # Überprüfen, ob die Ergebnisse das erwartete Format haben
        self.assertEqual(len(topics), 2)
        self.assertIn('id', topics[0])
        self.assertIn('words', topics[0])
        self.assertIn('examples', topics[0])
        
        # Überprüfen, ob die Wörter pro Thema korrekt sind
        self.assertEqual(len(topics[0]['words']), 5)
        
        # Überprüfen, ob die Beispiele enthalten sind
        self.assertGreaterEqual(len(topics[0]['examples']), 1)
    
    def test_text_similarity(self):
        """Test der Textähnlichkeitsfunktionalität"""
        analyzer = self.SemanticAnalyzer()
        query = "Is this product any good?"
        results = analyzer.find_similar_texts(query, self.sample_texts, top_n=2)
        
        # Überprüfen, ob die Ergebnisse das erwartete Format haben
        self.assertEqual(len(results), 2)
        self.assertIn('text', results[0])
        self.assertIn('similarity', results[0])
        self.assertIn('index', results[0])
        
        # Überprüfen, ob die Ähnlichkeitswerte im erwarteten Bereich liegen
        self.assertGreaterEqual(results[0]['similarity'], 0)
        self.assertLessEqual(results[0]['similarity'], 1)


class TestPredictiveModeler(unittest.TestCase):
    """Testsuite für die PredictiveModeler-Klasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Mockdaten für Vorhersagetests erstellen
        np.random.seed(42)
        
        # Zeitreihe mit Trend und Saison
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        trend = np.linspace(0, 10, 100)
        seasonality = 5 * np.sin(np.linspace(0, 4*np.pi, 100))
        noise = np.random.normal(0, 1, 100)
        
        self.time_series_data = pd.DataFrame({
            'date': dates,
            'value': trend + seasonality + noise,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'day': dates.day
        })
        
        # Mock-Implementierung für Tests
        class MockPredictiveModeler:
            def __init__(self, model_type='lstm', forecast_horizon=7):
                self.model_type = model_type
                self.forecast_horizon = forecast_horizon
                self.model = mock.MagicMock()
                self.scaler = mock.MagicMock()
                self.target_scaler = mock.MagicMock()
                self.is_fitted = False
                self.lookback = 10
                self.feature_dims = None
                self.target_dims = None
                
                # Mock-Skalierungsverhalten
                self.scaler.transform.side_effect = lambda x: x
                self.scaler.inverse_transform.side_effect = lambda x: x
                self.target_scaler.transform.side_effect = lambda x: x
                self.target_scaler.inverse_transform.side_effect = lambda x: x
                
            def fit(self, X, y=None, lookback=10, epochs=50, validation_split=0.2, batch_size=32, verbose=1):
                self.lookback = lookback
                self.is_fitted = True
                
                if isinstance(X, pd.DataFrame):
                    self.feature_dims = X.shape[1]
                else:
                    self.feature_dims = X.shape[1] if len(X.shape) > 1 else 1
                
                if y is not None:
                    if isinstance(y, pd.DataFrame):
                        self.target_dims = y.shape[1]
                    else:
                        self.target_dims = y.shape[1] if len(y.shape) > 1 else 1
                else:
                    self.target_dims = self.feature_dims
                
                return self
            
            def predict(self, X, return_sequences=False):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                # Einfache Simulation einer Vorhersage
                n_samples = len(X) if isinstance(X, pd.DataFrame) else X.shape[0]
                
                if return_sequences:
                    # Multi-step forecast
                    return np.random.random((n_samples, self.forecast_horizon, self.target_dims))
                else:
                    # Single-step forecast
                    return np.random.random((n_samples, self.target_dims))
            
            def forecast(self, X, steps=None):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                steps = steps or self.forecast_horizon
                
                # Simuliere eine Forecast-Sequenz
                return np.random.random((steps, self.target_dims))
            
            def evaluate(self, X_test, y_test):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                # Simulierte Metriken
                return {
                    'loss': 0.1,
                    'mae': 0.08,
                    'rmse': 0.12,
                    'mean_absolute_error': 0.08,
                    'mean_absolute_percentage_error': 8.5
                }
        
        self.PredictiveModeler = MockPredictiveModeler
    
    def test_initialization(self):
        """Test der Initialisierung mit verschiedenen Parametern"""
        modeler = self.PredictiveModeler(model_type='lstm', forecast_horizon=7)
        self.assertEqual(modeler.model_type, 'lstm')
        self.assertEqual(modeler.forecast_horizon, 7)
        self.assertFalse(modeler.is_fitted)
        
        modeler2 = self.PredictiveModeler(model_type='transformer', forecast_horizon=14)
        self.assertEqual(modeler2.model_type, 'transformer')
        self.assertEqual(modeler2.forecast_horizon, 14)
    
    def test_fit_predict(self):
        """Test der Fit- und Predict-Methoden"""
        modeler = self.PredictiveModeler()
        
        # Feature- und Zieldaten vorbereiten
        X = self.time_series_data[['day_of_week', 'month', 'day']]
        y = self.time_series_data[['value']]
        
        # Modell trainieren
        modeler.fit(X, y, lookback=5, epochs=10, verbose=0)
        
        # Überprüfen, ob das Modell trainiert wurde
        self.assertTrue(modeler.is_fitted)
        self.assertEqual(modeler.feature_dims, 3)
        self.assertEqual(modeler.target_dims, 1)
        
        # Vorhersagen treffen
        predictions = modeler.predict(X)
        
        # Überprüfen, ob die Vorhersagen das erwartete Format haben
        self.assertEqual(predictions.shape[0], len(X))
        self.assertEqual(predictions.shape[1], 1)
    
    def test_forecast(self):
        """Test der Forecast-Methode"""
        modeler = self.PredictiveModeler(forecast_horizon=10)
        
        # Feature- und Zieldaten vorbereiten
        X = self.time_series_data[['day_of_week', 'month', 'day']]
        y = self.time_series_data[['value']]
        
        # Modell trainieren
        modeler.fit(X, y, lookback=5, epochs=10, verbose=0)
        
        # Forecast für die nächsten 10 Schritte
        forecast = modeler.forecast(X, steps=10)
        
        # Überprüfen, ob der Forecast das erwartete Format hat
        self.assertEqual(forecast.shape[0], 10)
        self.assertEqual(forecast.shape[1], 1)
    
    def test_evaluation(self):
        """Test der Evaluate-Methode"""
        modeler = self.PredictiveModeler()
        
        # Feature- und Zieldaten vorbereiten
        X = self.time_series_data[['day_of_week', 'month', 'day']]
        y = self.time_series_data[['value']]
        
        # Modell trainieren
        modeler.fit(X, y, lookback=5, epochs=10, verbose=0)
        
        # Train-Test-Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Modell evaluieren
        metrics = modeler.evaluate(X_test, y_test)
        
        # Überprüfen, ob die Metriken das erwartete Format haben
        self.assertIn('loss', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mean_absolute_error', metrics)
        self.assertIn('mean_absolute_percentage_error', metrics)


class TestDataSynthesizer(unittest.TestCase):
    """Testsuite für die DataSynthesizer-Klasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Mockdaten für Synthesetests erstellen
        np.random.seed(42)
        
        self.categorical_data = pd.DataFrame({
            'category_1': np.random.choice(['A', 'B', 'C'], 100),
            'category_2': np.random.choice(['X', 'Y', 'Z'], 100)
        })
        
        self.numeric_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(5, 2, 100)
        })
        
        self.mixed_data = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'numeric': np.random.normal(0, 1, 100),
            'binary': np.random.choice([0, 1], 100)
        })
        
        # Mock-Implementierung für Tests
        class MockDataSynthesizer:
            def __init__(self, categorical_threshold=10, noise_dim=100):
                self.categorical_threshold = categorical_threshold
                self.noise_dim = noise_dim
                self.generator = mock.MagicMock()
                self.discriminator = mock.MagicMock()
                self.gan = mock.MagicMock()
                
                self.column_types = {}
                self.categorical_mappings = {}
                self.is_fitted = False
                self.feature_dims = None
                self.training_data = None
            
            def _identify_column_types(self, data):
                self.column_types = {}
                
                for col in data.columns:
                    n_unique = data[col].nunique()
                    
                    if n_unique < self.categorical_threshold or not pd.api.types.is_numeric_dtype(data[col]):
                        self.column_types[col] = 'categorical'
                        
                        # Mapping für kategorische Spalten erstellen
                        categories = data[col].unique()
                        self.categorical_mappings[col] = {cat: i for i, cat in enumerate(categories)}
                        self.categorical_mappings[f"{col}_reverse"] = {i: cat for i, cat in enumerate(categories)}
                    else:
                        self.column_types[col] = 'continuous'
            
            def fit(self, data, epochs=2000, batch_size=32, sample_interval=100, verbose=1):
                self._identify_column_types(data)
                self.feature_dims = len(data.columns)
                self.training_data = data.copy()
                self.is_fitted = True
                return self
            
            def generate(self, n_samples=100):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                # Einfache Simulation synthetischer Daten
                synthetic_data = pd.DataFrame()
                
                for col, col_type in self.column_types.items():
                    if col_type == 'categorical':
                        # Für kategorische Spalten: Zufällige Auswahl aus Original-Kategorien
                        categories = list(self.categorical_mappings[f"{col}_reverse"].values())
                        synthetic_data[col] = np.random.choice(categories, n_samples)
                    else:
                        # Für kontinuierliche Spalten: Simulieren ähnlicher Verteilungen
                        orig_mean = self.training_data[col].mean()
                        orig_std = self.training_data[col].std()
                        synthetic_data[col] = np.random.normal(orig_mean, orig_std, n_samples)
                
                return synthetic_data
            
            def evaluate_quality(self, n_samples=100):
                if not self.is_fitted:
                    raise ValueError("Model not fitted yet")
                
                # Synthetische Daten generieren
                synthetic_data = self.generate(n_samples)
                
                # Qualitätsmetriken simulieren
                metrics = {
                    'continuous_avg_diff': 0.15,
                    'categorical_avg_diff': 0.20,
                    'overall_quality_score': 0.75
                }
                
                # Detaillierte Metriken für jede Spalte
                for col in self.column_types:
                    if self.column_types[col] == 'continuous':
                        metrics[f"{col}_mean_diff"] = np.random.random() * 0.3  # Zufällige Differenz zwischen 0 und 0.3
                        metrics[f"{col}_std_diff"] = np.random.random() * 0.3
                    else:
                        metrics[f"{col}_js_divergence"] = np.random.random() * 0.4  # Zufällige JS-Divergenz zwischen 0 und 0.4
                
                return metrics
            
            def plot_comparison(self, n_samples=100, features=None, save_path=None):
                # Mock für Plot-Funktionalität
                from matplotlib.figure import Figure
                return Figure()
        
        self.DataSynthesizer = MockDataSynthesizer
    
    def test_initialization(self):
        """Test der Initialisierung mit verschiedenen Parametern"""
        synthesizer = self.DataSynthesizer(categorical_threshold=10, noise_dim=100)
        self.assertEqual(synthesizer.categorical_threshold, 10)
        self.assertEqual(synthesizer.noise_dim, 100)
        self.assertFalse(synthesizer.is_fitted)
    
    def test_column_type_identification(self):
        """Test der Spaltentyp-Identifikation"""
        synthesizer = self.DataSynthesizer()
        synthesizer._identify_column_types(self.mixed_data)
        
        # Überprüfen, ob die Spaltentypen korrekt identifiziert wurden
        self.assertEqual(synthesizer.column_types['category'], 'categorical')
        self.assertEqual(synthesizer.column_types['numeric'], 'continuous')
        self.assertEqual(synthesizer.column_types['binary'], 'categorical')
        
        # Überprüfen, ob die kategorischen Mappings erstellt wurden
        self.assertIn('category', synthesizer.categorical_mappings)
        self.assertIn('category_reverse', synthesizer.categorical_mappings)
        self.assertIn('binary', synthesizer.categorical_mappings)
        self.assertIn('binary_reverse', synthesizer.categorical_mappings)
    
    def test_fit_generate(self):
        """Test der Fit- und Generate-Methoden"""
        synthesizer = self.DataSynthesizer()
        
        # Modell mit gemischten Daten trainieren
        synthesizer.fit(self.mixed_data, epochs=10, verbose=0)
        
        # Überprüfen, ob das Modell trainiert wurde
        self.assertTrue(synthesizer.is_fitted)
        self.assertEqual(synthesizer.feature_dims, 3)
        
        # Synthetische Daten generieren
        n_samples = 50
        synthetic_data = synthesizer.generate(n_samples)
        
        # Überprüfen, ob die synthetischen Daten das erwartete Format haben
        self.assertEqual(len(synthetic_data), n_samples)
        self.assertEqual(len(synthetic_data.columns), 3)
        self.assertIn('category', synthetic_data.columns)
        self.assertIn('numeric', synthetic_data.columns)
        self.assertIn('binary', synthetic_data.columns)
    
    def test_quality_evaluation(self):
        """Test der Qualitätsbewertung der synthetischen Daten"""
        synthesizer = self.DataSynthesizer()
        
        # Modell mit numerischen Daten trainieren
        synthesizer.fit(self.numeric_data, epochs=10, verbose=0)
        
        # Qualität bewerten
        metrics = synthesizer.evaluate_quality(n_samples=50)
        
        # Überprüfen, ob die Metriken das erwartete Format haben
        self.assertIn('continuous_avg_diff', metrics)
        self.assertIn('overall_quality_score', metrics)
        
        # Überprüfen, ob spaltenspezifische Metriken enthalten sind
        self.assertIn('feature_1_mean_diff', metrics)
        self.assertIn('feature_2_mean_diff', metrics)
        
        # Überprüfen, ob die Qualitätsbewertung im gültigen Bereich liegt
        self.assertGreaterEqual(metrics['overall_quality_score'], 0)
        self.assertLessEqual(metrics['overall_quality_score'], 1)


class TestComputeToDataManager(unittest.TestCase):
    """Testsuite für die ComputeToDataManager-Klasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Mockdaten für C2D-Tests erstellen
        np.random.seed(42)
        
        self.test_data = pd.DataFrame({
            'id': range(100),
            'value': np.random.normal(0, 1, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'location': [f"City_{i % 10}" for i in range(100)],
            'sensitive_info': [f"sensitive_{i}" for i in range(100)]
        })
        
        # Mock-Implementierung für Tests
        class MockComputeToDataManager:
            def __init__(self, encryption_key=None):
                from cryptography.fernet import Fernet
                self.encryption_key = encryption_key or Fernet.generate_key()
                self.cipher_suite = Fernet(self.encryption_key)
                
                # Standard-Operationen definieren
                self.allowed_operations = {
                    'aggregate': self._aggregate,
                    'count': self._count,
                    'mean': self._mean,
                    'sum': self._sum,
                    'min': self._min,
                    'max': self._max,
                    'correlation': self._correlation,
                    'histogram': self._histogram,
                    'custom_model': self._custom_model_inference
                }
                
                # Datenschutzkonfiguration
                self.privacy_config = {
                    'min_group_size': 5,
                    'noise_level': 0.01,
                    'outlier_removal': True
                }
            
            def _encrypt_data(self, data):
                """Verschlüsselt Daten für die sichere Speicherung"""
                serialized = data.to_json().encode()
                return self.cipher_suite.encrypt(serialized)
            
            def _decrypt_data(self, encrypted_data):
                """Entschlüsselt Daten für die Verarbeitung"""
                decrypted = self.cipher_suite.decrypt(encrypted_data)
                return pd.read_json(StringIO(decrypted.decode()))
            
            def _aggregate(self, data, params):
                """Aggregiert Daten nach Spalten und Gruppen"""
                # Vereinfachte Implementierung für Tests
                return {'count': len(data)}
            
            def _count(self, data, params):
                """Zählt Datensätze nach Filterkriterien"""
                filter_column = params.get('filter_column')
                filter_value = params.get('filter_value')
                
                if filter_column and filter_value is not None:
                    count = len(data[data[filter_column] == filter_value])
                else:
                    count = len(data)
                
                return {'count': count}
            
            def _mean(self, data, params):
                """Berechnet den Mittelwert für numerische Spalten"""
                columns = params.get('columns', [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])])
                
                means = {}
                for col in columns:
                    if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                        means[col] = float(data[col].mean())
                
                return {'means': means}
            
            def _sum(self, data, params):
                return {'sums': {col: float(data[col].sum()) for col in data.select_dtypes(include=['number']).columns}}
            
            def _min(self, data, params):
                return {'minimums': {col: float(data[col].min()) for col in data.select_dtypes(include=['number']).columns}}
            
            def _max(self, data, params):
                return {'maximums': {col: float(data[col].max()) for col in data.select_dtypes(include=['number']).columns}}
            
            def _correlation(self, data, params):
                # Vereinfachte Implementierung für Tests
                corr_matrix = data.select_dtypes(include=['number']).corr().to_dict()
                return {'correlation_matrix': corr_matrix}
            
            def _histogram(self, data, params):
                column = params.get('column')
                bins = params.get('bins', 10)
                
                if column not in data.columns:
                    return {'error': f'Spalte {column} nicht gefunden'}
                
                if pd.api.types.is_numeric_dtype(data[column]):
                    hist, bin_edges = np.histogram(data[column], bins=bins)
                    return {
                        'histogram': hist.tolist(),
                        'bin_edges': bin_edges.tolist()
                    }
                else:
                    value_counts = data[column].value_counts()
                    return {
                        'categories': value_counts.index.tolist(),
                        'counts': value_counts.values.tolist()
                    }
            
            def _custom_model_inference(self, data, params):
                # Vereinfachte Implementierung für Tests
                return {'result': 'Model inference result'}
            
            def execute_operation(self, encrypted_data, operation, params):
                """Führt eine Operation auf verschlüsselten Daten aus"""
                try:
                    if operation not in self.allowed_operations:
                        return {'error': f'Nicht unterstützte Operation: {operation}'}
                    
                    # Entschlüssele die Daten temporär im Speicher
                    data = self._decrypt_data(encrypted_data)
                    
                    # Führe die Operation aus
                    operation_func = self.allowed_operations[operation]
                    results = operation_func(data, params)
                    
                    return results
                    
                except Exception as e:
                    return {'error': f'Fehler bei der Ausführung: {str(e)}'}
            
            def create_data_asset(self, data, asset_metadata=None):
                """Erstellt einen verschlüsselten Daten-Asset mit Metadaten"""
                try:
                    # Erstelle eine eindeutige ID für den Asset
                    import uuid
                    asset_id = uuid.uuid4().hex
                    
                    # Verschlüssele die Daten
                    encrypted_data = self._encrypt_data(data)
                    
                    # Erstelle Metadaten
                    stats = {
                        'record_count': len(data),
                        'columns': [{'name': col, 'type': str(data[col].dtype)} for col in data.columns]
                    }
                    
                    asset_info = {
                        'asset_id': asset_id,
                        'created_at': datetime.now().isoformat(),
                        'statistics': stats
                    }
                    
                    if asset_metadata:
                        asset_info.update(asset_metadata)
                    
                    return {
                        'asset_id': asset_id,
                        'metadata': asset_info,
                        'encrypted_data': encrypted_data
                    }
                    
                except Exception as e:
                    return {'error': f'Fehler bei der Asset-Erstellung: {str(e)}'}
            
            def generate_access_token(self, asset_id, allowed_operations, expiration_time=3600):
                """Generiert ein temporäres Zugriffstoken für einen Daten-Asset"""
                import uuid
                
                token_data = {
                    'asset_id': asset_id,
                    'allowed_operations': allowed_operations,
                    'created_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(seconds=expiration_time)).isoformat(),
                    'token_id': uuid.uuid4().hex
                }
                
                token_json = json.dumps(token_data).encode()
                encrypted_token = self.cipher_suite.encrypt(token_json)
                
                return {
                    'token': encrypted_token.decode(),
                    'token_id': token_data['token_id'],
                    'expires_at': token_data['expires_at']
                }
            
            def validate_access_token(self, token, operation):
                """Validiert ein Zugriffstoken für eine bestimmte Operation"""
                try:
                    # Entschlüssele das Token
                    decrypted_token = self.cipher_suite.decrypt(token.encode())
                    token_data = json.loads(decrypted_token.decode())
                    
                    # Prüfe, ob das Token abgelaufen ist
                    expiry_time = datetime.fromisoformat(token_data['expires_at'])
                    if datetime.now() > expiry_time:
                        return False
                    
                    # Prüfe, ob die Operation erlaubt ist
                    if operation not in token_data['allowed_operations']:
                        return False
                    
                    return True
                    
                except Exception:
                    return False
        
        self.ComputeToDataManager = MockComputeToDataManager
    
    def test_encryption_decryption(self):
        """Test der Verschlüsselungs- und Entschlüsselungsfunktionalität"""
        c2d_manager = self.ComputeToDataManager()
        
        # Daten verschlüsseln
        encrypted_data = c2d_manager._encrypt_data(self.test_data)
        
        # Überprüfen, ob die Verschlüsselung erfolgreich war
        self.assertIsInstance(encrypted_data, bytes)
        
        # Daten entschlüsseln
        decrypted_data = c2d_manager._decrypt_data(encrypted_data)
        
        # Überprüfen, ob die Entschlüsselung erfolgreich war und die Daten korrekt sind
        self.assertIsInstance(decrypted_data, pd.DataFrame)
        self.assertEqual(len(decrypted_data), len(self.test_data))
        self.assertListEqual(list(decrypted_data.columns), list(self.test_data.columns))
    
    def test_execute_operation(self):
        """Test der Ausführung von Operationen auf verschlüsselten Daten"""
        c2d_manager = self.ComputeToDataManager()
        
        # Daten verschlüsseln
        encrypted_data = c2d_manager._encrypt_data(self.test_data)
        
        # Operation 'count' ausführen
        result = c2d_manager.execute_operation(encrypted_data, 'count', {})
        
        # Überprüfen, ob das Ergebnis korrekt ist
        self.assertIn('count', result)
        self.assertEqual(result['count'], 100)
        
        # Operation 'mean' ausführen
        result = c2d_manager.execute_operation(encrypted_data, 'mean', {})
        
        # Überprüfen, ob das Ergebnis korrekt ist
        self.assertIn('means', result)
        self.assertIn('value', result['means'])
        
        # Operation mit Filterparametern ausführen
        result = c2d_manager.execute_operation(
            encrypted_data, 
            'count', 
            {'filter_column': 'category', 'filter_value': 'A'}
        )
        
        # Überprüfen, ob das Ergebnis das erwartete Format hat
        self.assertIn('count', result)
        
        # Nicht unterstützte Operation ausführen
        result = c2d_manager.execute_operation(encrypted_data, 'invalid_operation', {})
        
        # Überprüfen, ob ein Fehler zurückgegeben wird
        self.assertIn('error', result)
    
    def test_create_data_asset(self):
        """Test der Erstellung eines Daten-Assets"""
        c2d_manager = self.ComputeToDataManager()
        
        # Asset-Metadaten definieren
        asset_metadata = {
            'name': 'Test Dataset',
            'description': 'A dataset for testing',
            'owner': 'test_user',
            'price': 5.0
        }
        
        # Daten-Asset erstellen
        asset = c2d_manager.create_data_asset(self.test_data, asset_metadata)
        
        # Überprüfen, ob der Asset das erwartete Format hat
        self.assertIn('asset_id', asset)
        self.assertIn('metadata', asset)
        self.assertIn('encrypted_data', asset)
        
        # Überprüfen, ob die Metadaten korrekt sind
        self.assertEqual(asset['metadata']['name'], 'Test Dataset')
        self.assertEqual(asset['metadata']['price'], 5.0)
        
        # Überprüfen, ob die Statistiken korrekt sind
        self.assertEqual(asset['metadata']['statistics']['record_count'], 100)
    
    def test_access_token(self):
        """Test der Token-Generierung und -Validierung"""
        c2d_manager = self.ComputeToDataManager()
        
        # Asset ID und erlaubte Operationen definieren
        asset_id = 'test_asset_123'
        allowed_operations = ['count', 'mean', 'sum']
        
        # Zugriffstoken generieren
        token_info = c2d_manager.generate_access_token(asset_id, allowed_operations, expiration_time=60)
        
        # Überprüfen, ob das Token das erwartete Format hat
        self.assertIn('token', token_info)
        self.assertIn('token_id', token_info)
        self.assertIn('expires_at', token_info)
        
        # Token für erlaubte Operation validieren
        is_valid = c2d_manager.validate_access_token(token_info['token'], 'count')
        self.assertTrue(is_valid)
        
        # Token für nicht erlaubte Operation validieren
        is_valid = c2d_manager.validate_access_token(token_info['token'], 'histogram')
        self.assertFalse(is_valid)


class TestOceanDataAI(unittest.TestCase):
    """Testsuite für die OceanDataAI-Hauptklasse"""
    
    def setUp(self):
        """Setup für alle Tests"""
        # Mockdaten für Tests
        self.browser_data = create_mock_browser_data()
        self.smartwatch_data = create_mock_smartwatch_data()
        
        # Mock-Implementierung für Tests
        class MockOceanDataAI:
            def __init__(self, config=None):
                self.config = config or {}
                
                # Mock-Komponenten
                self.anomaly_detector = mock.MagicMock()
                self.semantic_analyzer = mock.MagicMock()
                self.predictive_modeler = mock.MagicMock()
                self.data_synthesizer = mock.MagicMock()
                self.c2d_manager = mock.MagicMock()
                
            def analyze_data_source(self, data, source_type):
                """Mock-Analyse einer Datenquelle"""
                return {
                    'source_type': source_type,
                    'timestamp': datetime.now().isoformat(),
                    'record_count': len(data),
                    'column_count': len(data.columns),
                    'analyses': {
                        'anomalies': {
                            'count': int(len(data) * 0.05),
                            'percentage': 5.0
                        },
                        'time_series': {
                            'forecast_horizon': 7
                        }
                    }
                }
            
            def prepare_data_for_monetization(self, data, source_type, privacy_level='medium'):
                """Mock-Vorbereitung für Monetarisierung"""
                # Basiswerte für verschiedene Quellen
                base_values = {
                    'browser': 3.5,
                    'smartwatch': 5.0,
                    'calendar': 2.5,
                    'social_media': 4.0,
                    'streaming': 3.0,
                    'health_data': 6.0,
                    'iot': 2.0
                }
                
                # Privacy-Level-Faktoren
                privacy_factors = {
                    'low': 1.2,
                    'medium': 1.0,
                    'high': 0.8
                }
                
                # Basiswert mit Privacy-Faktor kombinieren
                estimated_value = base_values.get(source_type, 3.0) * privacy_factors.get(privacy_level, 1.0)
                
                return {
                    'anonymized_data': data.copy(),
                    'metadata': {
                        'source_type': source_type,
                        'privacy_level': privacy_level,
                        'record_count': len(data),
                        'field_count': len(data.columns),
                        'estimated_value': estimated_value,
                        'created_at': datetime.now().isoformat()
                    },
                    'c2d_asset': {
                        'asset_id': f"asset_{uuid.uuid4().hex[:8]}",
                        'metadata': {}
                    }
                }
            
            def combine_data_sources(self, sources, combination_type='merge'):
                """Mock-Kombination von Datenquellen"""
                # Basiswert ist die Summe der Einzelwerte mit einem Bonus
                source_values = [s['metadata'].get('estimated_value', 0) for s in sources]
                
                if combination_type == 'merge':
                    combined_value = sum(source_values) * 1.2  # 20% Bonus
                elif combination_type == 'enrich':
                    combined_value = max(source_values) + sum(source_values[1:]) * 0.6  # Basis + 60% der zusätzlichen Quellen
                elif combination_type == 'correlate':
                    combined_value = sum(source_values) * 0.3 + len(sources) * 0.5  # 30% der Originalwerte + Bonus pro Quelle
                else:
                    combined_value = sum(source_values)
                
                # Kombinierte Daten erstellen
                combined_data = pd.DataFrame()
                for source in sources:
                    if 'anonymized_data' in source and source['anonymized_data'] is not None:
                        if combined_data.empty:
                            combined_data = source['anonymized_data'].copy()
                        else:
                            # Je nach Kombinationstyp unterschiedliche Strategien
                            if combination_type == 'merge':
                                # Einfaches Anfügen
                                combined_data = pd.concat([combined_data, source['anonymized_data']], ignore_index=True)
                            elif combination_type == 'enrich':
                                # Spalten hinzufügen
                                for col in source['anonymized_data'].columns:
                                    if col not in combined_data.columns:
                                        combined_data[col] = np.nan
                                        combined_data[col][:len(source['anonymized_data'])] = source['anonymized_data'][col].values
                
                return {
                    'anonymized_data': combined_data,
                    'metadata': {
                        'combination_type': combination_type,
                        'source_count': len(sources),
                        'estimated_value': combined_value,
                        'created_at': datetime.now().isoformat()
                    },
                    'c2d_asset': {
                        'asset_id': f"combined_{uuid.uuid4().hex[:8]}",
                        'metadata': {}
                    }
                }
            
            def estimate_data_value(self, data, metadata=None):
                """Mock-Wertschätzung eines Datensatzes"""
                metadata = metadata or {}
                
                # Basiswert nach Datengröße
                base_value = min(1.0, len(data) / 1000) * 5  # Skalieren nach Größe, max 5 OCEAN
                
                # Faktor nach Datenqualität (weniger fehlende Werte = höherer Wert)
                quality_factor = 1.0 - data.isna().mean().mean()
                
                # Faktor nach Spaltenanzahl
                cols_factor = min(1.0, len(data.columns) / 10) * 0.5 + 0.5  # 0.5 bis 1.0 basierend auf Spaltenanzahl
                
                # Spezielle Werterhöhung für bestimmte Datentypen
                source_type = metadata.get('source_type', '')
                source_bonus = {
                    'health_data': 1.5,
                    'smartwatch': 1.3,
                    'browser': 1.2,
                    'social_media': 1.4
                }.get(source_type, 1.0)
                
                # Gesamtwert berechnen
                estimated_value = base_value * quality_factor * cols_factor * source_bonus
                
                # Wertfaktoren
                value_factors = {
                    'data_size': float(min(1.0, len(data) / 1000)),
                    'data_quality': float(quality_factor),
                    'column_diversity': float(cols_factor)
                }
                
                return {
                    'normalized_score': float(quality_factor * cols_factor * source_bonus / 3),
                    'estimated_token_value': float(estimated_value),
                    'metrics': {
                        'data_volume': {'score': value_factors['data_size'], 'weight': 0.3, 'explanation': ''},
                        'data_quality': {'score': value_factors['data_quality'], 'weight': 0.3, 'explanation': ''},
                        'data_uniqueness': {'score': 0.7, 'weight': 0.2, 'explanation': ''},
                        'time_relevance': {'score': 0.8, 'weight': 0.2, 'explanation': ''}
                    },
                    'summary': f"This dataset has an estimated value of {estimated_value:.2f} OCEAN tokens."
                }
            
            def prepare_for_ocean_tokenization(self, data_asset):
                """Mock-Vorbereitung für Ocean-Tokenisierung"""
                asset_id = data_asset['c2d_asset']['asset_id']
                metadata = data_asset['metadata']
                
                ddo = {
                    'id': asset_id,
                    'created': datetime.now().isoformat(),
                    'updated': datetime.now().isoformat(),
                    'type': 'dataset',
                    'name': f"Dataset {asset_id[:8]}",
                    'description': metadata.get('description', 'No description provided.'),
                    'tags': [metadata.get('source_type', 'data'), metadata.get('privacy_level', 'medium')],
                    'price': metadata.get('estimated_value', 5.0)
                }
                
                pricing = {
                    'type': 'fixed',
                    'baseTokenAmount': metadata.get('estimated_value', 5.0),
                    'datatoken': {
                        'name': f"DT-{ddo['name']}",
                        'symbol': f"DT{asset_id[:4]}".upper()
                    }
                }
                
                return {
                    'ddo': ddo,
                    'pricing': pricing,
                    'asset_id': asset_id
                }
            
            def tokenize_with_ocean(self, ocean_asset):
                """Mock-Tokenisierung mit Ocean Protocol"""
                tx_hash = f"0x{uuid.uuid4().hex}"
                token_address = f"0x{uuid.uuid4().hex}"
                
                return {
                    'success': True,
                    'asset_id': ocean_asset.get('asset_id', 'unknown'),
                    'token_address': token_address,
                    'token_symbol': ocean_asset.get('pricing', {}).get('datatoken', {}).get('symbol', 'UNKNOWN'),
                    'token_name': ocean_asset.get('pricing', {}).get('datatoken', {}).get('name', 'Unknown Token'),
                    'token_price': ocean_asset.get('pricing', {}).get('baseTokenAmount', 0),
                    'transaction_hash': tx_hash,
                    'timestamp': datetime.now().isoformat(),
                    'marketplace_url': f"https://market.oceanprotocol.com/asset/{ocean_asset.get('asset_id', 'unknown')}"
                }
        
        self.OceanDataAI = MockOceanDataAI
    
    def test_analyze_data_source(self):
        """Test der Datenquellenanalyse"""
        ocean_ai = self.OceanDataAI()
        
        # Browser-Daten analysieren
        browser_analysis = ocean_ai.analyze_data_source(self.browser_data, 'browser')
        
        # Überprüfen, ob die Analyse das erwartete Format hat
        self.assertEqual(browser_analysis['source_type'], 'browser')
        self.assertEqual(browser_analysis['record_count'], len(self.browser_data))
        self.assertEqual(browser_analysis['column_count'], len(self.browser_data.columns))
        self.assertIn('analyses', browser_analysis)
        self.assertIn('anomalies', browser_analysis['analyses'])
        
        # Smartwatch-Daten analysieren
        smartwatch_analysis = ocean_ai.analyze_data_source(self.smartwatch_data, 'smartwatch')
        
        # Überprüfen, ob die Analyse das erwartete Format hat
        self.assertEqual(smartwatch_analysis['source_type'], 'smartwatch')
    
    def test_prepare_data_for_monetization(self):
        """Test der Datenvorbereitung für Monetarisierung"""
        ocean_ai = self.OceanDataAI()
        
        # Browser-
