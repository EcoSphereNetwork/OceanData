"""
OceanData - Fortgeschrittene KI-Module für Datenanalyse und -monetarisierung

Diese Module erweitern den Kernalgorithmus von OceanData um fortschrittliche KI-Funktionen:
1. Anomalieerkennung
2. Semantische Datenanalyse 
3. Prädiktive Modellierung
4. Datensynthese und Erweiterung
5. Multimodale Analyse
6. Federated Learning und Compute-to-Data
7. Kontinuierliches Lernen und Modellanpassung
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertTokenizer, BertModel, TFBertModel
from transformers import GPT2Tokenizer, GPT2Model, TFGPT2Model
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import json
from typing import Dict, List, Union, Any, Tuple, Optional
import logging
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import h5py
import uuid
import traceback
import warnings

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Ignoriere TensorFlow und PyTorch Warnungen
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger("OceanData.AI")

###########################################
# 1. Anomalieerkennung & Outlier Detection
###########################################

class AnomalyDetector:
    """Klasse für die Erkennung von Anomalien in verschiedenen Datentypen"""
    
    def __init__(self, method: str = 'autoencoder', contamination: float = 0.05):
        """
        Initialisiert einen Anomaliedetektor.
        
        Args:
            method: Methode für die Anomalieerkennung ('autoencoder', 'isolation_forest', 'dbscan')
            contamination: Erwarteter Anteil an Anomalien in den Daten (für Isolation Forest)
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dims = None
        self.threshold = None
        
    def _build_autoencoder(self, input_dim):
        """Erstellt ein Autoencoder-Modell für die Anomalieerkennung"""
        encoding_dim = max(1, input_dim // 2)
        hidden_dim = max(1, encoding_dim // 2)
        
        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(inputs)
        encoded = layers.Dense(hidden_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Encoder model for extracting features
        encoder = keras.Model(inputs, encoded)
        
        return autoencoder, encoder
    
    def fit(self, X: pd.DataFrame, categorical_cols: List[str] = None):
        """
        Trainiert den Anomaliedetektor mit Daten.
        
        Args:
            X: Eingabedaten (DataFrame oder numpy array)
            categorical_cols: Liste der kategorischen Spalten, die One-Hot-kodiert werden müssen
        """
        try:
            # Vorbereiten der Daten
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten durch One-Hot-Encoding
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Konvertieren zu float und Behandlung von NaN-Werten
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            
            if X_prep.shape[1] == 0:
                raise ValueError("Nach der Vorverarbeitung sind keine numerischen Merkmale übrig")
                
            X_scaled = self.scaler.fit_transform(X_prep)
            self.feature_dims = X_scaled.shape[1]
            
            if self.method == 'autoencoder':
                # Autoencoder-Ansatz für Anomalieerkennung
                self.model, self.encoder = self._build_autoencoder(self.feature_dims)
                
                # Trainiere den Autoencoder
                self.model.fit(
                    X_scaled, X_scaled,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    verbose=0,
                    validation_split=0.1
                )
                
                # Berechne Rekonstruktionsfehler für alle Trainingsdaten
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Setze Threshold basierend auf den Rekonstruktionsfehlern
                self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
                
            elif self.method == 'isolation_forest':
                # Isolation Forest für Outlier-Erkennung
                self.model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
                self.model.fit(X_scaled)
                
            elif self.method == 'dbscan':
                # DBSCAN für Clustering-basierte Outlier-Erkennung
                self.model = DBSCAN(
                    eps=0.5,  # Maximum distance between samples
                    min_samples=5,  # Minimum samples in a cluster
                    n_jobs=-1
                )
                self.model.fit(X_scaled)
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des Anomaliedetektors: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X: pd.DataFrame, categorical_cols: List[str] = None) -> np.ndarray:
        """
        Identifiziert Anomalien in den Daten.
        
        Args:
            X: Zu prüfende Daten
            categorical_cols: Liste der kategorischen Spalten
            
        Returns:
            Array mit Anomalie-Scores für jede Zeile (-1 für Anomalien, 1 für normale Daten bei Isolation Forest;
            True für Anomalien, False für normale Daten bei Autoencoder)
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Daten
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Konvertieren und Behandlung von NaN-Werten
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            
            # Anpassen der Features, falls nötig
            missing_cols = set(range(self.feature_dims)) - set(range(X_prep.shape[1]))
            if missing_cols:
                for col in missing_cols:
                    X_prep[f'missing_{col}'] = 0
            
            X_prep = X_prep.iloc[:, :self.feature_dims]
            X_scaled = self.scaler.transform(X_prep)
            
            if self.method == 'autoencoder':
                # Berechne Rekonstruktionsfehler
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Anomalie, wenn Fehler über dem Threshold
                return reconstruction_errors > self.threshold
                
            elif self.method == 'isolation_forest':
                # -1 für Anomalien, 1 für normale Daten
                return self.model.predict(X_scaled)
                
            elif self.method == 'dbscan':
                # Ordne neue Punkte den nächsten Clustern zu
                labels = self.model.fit_predict(X_scaled)
                # -1 sind Noise-Punkte (Anomalien)
                return labels == -1
                
        except Exception as e:
            logger.error(f"Fehler bei der Anomalievorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_anomaly_insights(self, X: pd.DataFrame, predictions: np.ndarray, 
                           categorical_cols: List[str] = None, top_n_features: int = 3) -> List[Dict]:
        """
        Liefert Erklärungen für die erkannten Anomalien.
        
        Args:
            X: Originaldaten
            predictions: Ergebnisse der predict()-Methode
            categorical_cols: Liste der kategorischen Spalten
            top_n_features: Anzahl der wichtigsten Features, die zurückgegeben werden sollen
            
        Returns:
            Liste mit Erklärungen für jede Anomalie
        """
        insights = []
        
        try:
            anomaly_indices = np.where(predictions == -1)[0] if self.method == 'isolation_forest' else np.where(predictions)[0]
            
            if len(anomaly_indices) == 0:
                return []
            
            X_prep = X.copy()
            X_orig = X.copy()  # Original-Daten für Berichte
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            X_scaled = self.scaler.transform(X_prep)
            
            # Allgemeine Statistiken berechnen
            mean_values = np.mean(X_scaled, axis=0)
            std_values = np.std(X_scaled, axis=0)
            
            for idx in anomaly_indices:
                if idx >= len(X_scaled):
                    continue
                    
                sample = X_scaled[idx]
                orig_sample = X_orig.iloc[idx]
                
                if self.method == 'autoencoder':
                    # Rekonstruktionsfehler pro Feature berechnen
                    reconstruction = self.model.predict(sample.reshape(1, -1))[0]
                    feature_errors = np.square(sample - reconstruction)
                    
                    # Features mit den größten Fehlern identifizieren
                    top_features_idx = np.argsort(feature_errors)[-top_n_features:]
                    top_features = [(X_prep.columns[i], feature_errors[i], sample[i], reconstruction[i]) for i in top_features_idx]
                    
                    insight = {
                        "index": int(idx),
                        "anomaly_score": float(np.mean(feature_errors)),
                        "important_features": [
                            {
                                "feature": feat,
                                "error": float(err),
                                "actual_value": float(val),
                                "expected_value": float(rec)
                            } for feat, err, val, rec in top_features
                        ],
                        "original_data": orig_sample.to_dict()
                    }
                    
                else:  # Isolation Forest oder DBSCAN
                    # Z-Scores berechnen, um ungewöhnliche Werte zu identifizieren
                    z_scores = (sample - mean_values) / std_values
                    
                    # Features mit den höchsten Z-Scores (positiv oder negativ)
                    abs_z_scores = np.abs(z_scores)
                    top_features_idx = np.argsort(abs_z_scores)[-top_n_features:]
                    top_features = [(X_prep.columns[i], z_scores[i], sample[i], mean_values[i]) for i in top_features_idx]
                    
                    insight = {
                        "index": int(idx),
                        "anomaly_score": float(np.max(abs_z_scores)),
                        "important_features": [
                            {
                                "feature": feat,
                                "z_score": float(z),
                                "actual_value": float(val),
                                "average_value": float(avg)
                            } for feat, z, val, avg in top_features
                        ],
                        "original_data": orig_sample.to_dict()
                    }
                
                insights.append(insight)
                
            return insights
            
        except Exception as e:
            logger.error(f"Fehler bei der Generierung von Anomalie-Insights: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def visualize_anomalies(self, X: pd.DataFrame, predictions: np.ndarray, 
                            categorical_cols: List[str] = None, 
                            save_path: str = None) -> plt.Figure:
        """
        Visualisiert erkannte Anomalien in den Daten.
        
        Args:
            X: Daten
            predictions: Anomalieerkennungsergebnisse
            categorical_cols: Kategorische Spalten
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        try:
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            X_scaled = self.scaler.transform(X_prep)
            
            # PCA für Dimensionsreduktion
            if X_scaled.shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
            else:
                X_pca = X_scaled
            
            # Anomalien bestimmen
            if self.method == 'isolation_forest':
                anomalies = (predictions == -1)
            else:
                anomalies = predictions
            
            # Visualisierung erstellen
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Normale Datenpunkte plotten
            ax.scatter(X_pca[~anomalies, 0], X_pca[~anomalies, 1], c='blue', label='Normal', alpha=0.5)
            
            # Anomalien plotten
            if np.any(anomalies):
                ax.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], c='red', label='Anomalien', alpha=0.7)
            
            # Grafik anpassen
            ax.set_title(f'Anomalieerkennung mit {self.method.capitalize()}')
            ax.set_xlabel('Hauptkomponente 1')
            ax.set_ylabel('Hauptkomponente 2')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung von Anomalien: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden")
            
        model_data = {
            "method": self.method,
            "contamination": self.contamination,
            "feature_dims": self.feature_dims,
            "threshold": self.threshold,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere das Model-Objekt
        if self.method in ['isolation_forest', 'dbscan']:
            joblib.dump(self.model, f"{path}_model.joblib")
        elif self.method == 'autoencoder':
            self.model.save(f"{path}_autoencoder")
            self.encoder.save(f"{path}_encoder")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        detector = cls(
            method=model_data['method'],
            contamination=model_data['contamination']
        )
        
        detector.feature_dims = model_data['feature_dims']
        detector.threshold = model_data['threshold']
        detector.is_fitted = model_data['is_fitted']
        
        # Lade das Model-Objekt
        if detector.method in ['isolation_forest', 'dbscan']:
            detector.model = joblib.load(f"{path}_model.joblib")
        elif detector.method == 'autoencoder':
            detector.model = keras.models.load_model(f"{path}_autoencoder")
            detector.encoder = keras.models.load_model(f"{path}_encoder")
        
        # Lade den Scaler
        detector.scaler = joblib.load(f"{path}_scaler.joblib")
        
        return detector


###########################################
# 2. Semantische Datenanalyse
###########################################

class SemanticAnalyzer:
    """Klasse für semantische Analyse von Text und anderen Daten mit Deep Learning"""
    
    def __init__(self, model_type: str = 'bert', model_name: str = 'bert-base-uncased'):
        """
        Initialisiert den semantischen Analysator.
        
        Args:
            model_type: Typ des zu verwendenden Modells ('bert', 'gpt2', 'custom')
            model_name: Name oder Pfad des vortrainierten Modells
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.max_length = 512  # Standard für BERT
        self.embeddings_cache = {}  # Cache für Texteinbettungen
        
        # Modell und Tokenizer laden
        self._load_model()
    
    def _load_model(self):
        """Lädt das Modell und den Tokenizer basierend auf dem ausgewählten Typ"""
        try:
            if self.model_type == 'bert':
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                if tf.test.is_gpu_available():
                    self.model = TFBertModel.from_pretrained(self.model_name)
                else:
                    self.model = BertModel.from_pretrained(self.model_name)
                self.max_length = 512
            
            elif self.model_type == 'gpt2':
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 hat kein Padding-Token
                if tf.test.is_gpu_available():
                    self.model = TFGPT2Model.from_pretrained(self.model_name)
                else:
                    self.model = GPT2Model.from_pretrained(self.model_name)
                self.max_length = 1024
            
            else:
                raise ValueError(f"Nicht unterstützter Modelltyp: {self.model_type}")
        
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells {self.model_name}: {str(e)}")
            # Fallback zu kleineren Modellen bei Speicher- oder Download-Problemen
            if self.model_type == 'bert':
                logger.info("Verwende ein kleineres BERT-Modell als Fallback")
                self.model_name = 'distilbert-base-uncased'
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertModel.from_pretrained(self.model_name)
            elif self.model_type == 'gpt2':
                logger.info("Verwende ein kleineres GPT-2-Modell als Fallback")
                self.model_name = 'distilgpt2'
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = GPT2Model.from_pretrained(self.model_name)
    
    def get_embeddings(self, texts: Union[str, List[str]], 
                      batch_size: int = 8, use_cache: bool = True) -> np.ndarray:
        """
        Erzeugt Einbettungen (Embeddings) für Texte.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            batch_size: Größe der Batches für die Verarbeitung
            use_cache: Ob bereits berechnete Einbettungen wiederverwendet werden sollen
            
        Returns:
            Array mit Einbettungen (shape: [n_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Initialisiere ein Array für die Ergebnisse
        all_embeddings = []
        texts_to_process = []
        texts_indices = []
        
        # Prüfe Cache für jede Anfrage
        for i, text in enumerate(texts):
            if use_cache and text in self.embeddings_cache:
                all_embeddings.append(self.embeddings_cache[text])
            else:
                texts_to_process.append(text)
                texts_indices.append(i)
        
        if texts_to_process:
            # Verarbeite Texte in Batches
            for i in range(0, len(texts_to_process), batch_size):
                batch_texts = texts_to_process[i:i+batch_size]
                batch_indices = texts_indices[i:i+batch_size]
                
                # Tokenisierung
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt" if isinstance(self.model, nn.Module) else "tf",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Modelloutput berechnen
                with torch.no_grad() if isinstance(self.model, nn.Module) else tf.device('/CPU:0'):
                    outputs = self.model(**inputs)
                
                # Embeddings aus dem letzten Hidden State extrahieren
                if self.model_type == 'bert':
                    # Verwende [CLS]-Token-Ausgabe als Satzrepräsentation (erstes Token)
                    if isinstance(self.model, nn.Module):
                        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                    else:
                        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                elif self.model_type == 'gpt2':
                    # Verwende den Durchschnitt aller Token-Repräsentationen
                    if isinstance(self.model, nn.Module):
                        embeddings = torch.mean(outputs.last_hidden_state, dim=1).numpy()
                    else:
                        embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
                
                # Füge die Embeddings an der richtigen Position ein
                for j, (idx, text, embedding) in enumerate(zip(batch_indices, batch_texts, embeddings)):
                    # Zum Cache hinzufügen
                    if use_cache:
                        self.embeddings_cache[text] = embedding
                    
                    # Aktualisiere Ergebnisarray an der richtigen Position
                    if idx >= len(all_embeddings):
                        all_embeddings.extend([None] * (idx - len(all_embeddings) + 1))
                    all_embeddings[idx] = embedding
        
        # Konvertiere zu NumPy-Array
        return np.vstack(all_embeddings)
    
    def analyze_sentiment(self, texts: Union[str, List[str]]) -> List[Dict]:
        """
        Führt eine Stimmungsanalyse für Texte durch.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            
        Returns:
            Liste mit Sentiment-Analysen für jeden Text (positive, negative, neutral)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Verwende NLTK für grundlegende Sentimentanalyse
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            
            analyzer = SentimentIntensityAnalyzer()
            results = []
            
            for text in texts:
                scores = analyzer.polarity_scores(text)
                
                # Bestimme die dominante Stimmung
                if scores['compound'] >= 0.05:
                    sentiment = 'positive'
                elif scores['compound'] <= -0.05:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': sentiment,
                    'scores': {
                        'positive': scores['pos'],
                        'negative': scores['neg'],
                        'neutral': scores['neu'],
                        'compound': scores['compound']
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Fehler bei der Sentimentanalyse: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback zu einem einfacheren Ansatz
            return [{'text': t[:100] + '...' if len(t) > 100 else t, 
                    'sentiment': 'unknown', 
                    'scores': {'positive': 0, 'negative': 0, 'neutral': 0, 'compound': 0}} 
                    for t in texts]
    
    def extract_topics(self, texts: Union[str, List[str]], num_topics: int = 5, 
                       words_per_topic: int = 5) -> List[Dict]:
        """
        Extrahiert Themen (Topics) aus Texten.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            num_topics: Anzahl der zu extrahierenden Themen
            words_per_topic: Anzahl der Wörter pro Thema
            
        Returns:
            Liste mit Themen und zugehörigen Top-Wörtern
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Tokenisiere und bereinige die Texte
            try:
                nltk.data.find('stopwords')
                nltk.data.find('punkt')
            except LookupError:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
            
            stop_words = set(stopwords.words('english'))
            
            # Texte vorverarbeiten
            processed_texts = []
            for text in texts:
                # Tokenisieren und Stopwords entfernen
                tokens = [w.lower() for w in word_tokenize(text) 
                         if w.isalpha() and w.lower() not in stop_words]
                processed_texts.append(' '.join(tokens))
            
            # Verwende Transformers für Themenmodellierung
            embeddings = self.get_embeddings(processed_texts)
            
            # Verwende K-Means-Clustering auf Embeddings
            kmeans = KMeans(n_clusters=min(num_topics, len(processed_texts)), random_state=42)
            kmeans.fit(embeddings)
            
            # Finde repräsentative Wörter für jedes Cluster
            topics = []
            
            # Alle Wörter aus allen Texten zusammenfassen
            all_words = []
            for text in processed_texts:
                all_words.extend(text.split())
            
            # Eindeutige Wörter
            unique_words = list(set(all_words))
            
            # Für jedes Wort ein Embedding berechnen
            if len(unique_words) > 0:
                word_embeddings = self.get_embeddings(unique_words)
                
                # Für jedes Cluster die nächsten Wörter bestimmen, die dem Clusterzentrum am nächsten sind
                for cluster_idx in range(kmeans.n_clusters):
                    center = kmeans.cluster_centers_[cluster_idx]
                    
                    # Berechne Distanzen zwischen Zentrum und Wort-Embeddings
                    distances = np.linalg.norm(word_embeddings - center, axis=1)
                    
                    # Finde die nächsten Wörter
                    closest_indices = np.argsort(distances)[:words_per_topic]
                    top_words = [unique_words[i] for i in closest_indices]
                    
                    # Beispieltexte für dieses Cluster finden
                    cluster_texts = [texts[i][:100] + "..." 
                                    for i, label in enumerate(kmeans.labels_) 
                                    if label == cluster_idx][:3]  # Maximal 3 Beispiele
                    
                    topic = {
                        "id": cluster_idx,
                        "words": top_words,
                        "examples": cluster_texts
                    }
                    topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Fehler bei der Themenextraktion: {str(e)}")
            logger.error(traceback.format_exc())
            return [{"id": 0, "words": ["error", "processing", "topics"], "examples": []}]
    
    def find_similar_texts(self, query: str, corpus: List[str], top_n: int = 5) -> List[Dict]:
        """
        Findet ähnliche Texte zu einer Anfrage in einem Korpus.
        
        Args:
            query: Anfrage-Text
            corpus: Liste von Texten, in denen gesucht werden soll
            top_n: Anzahl der zurückzugebenden ähnlichsten Texte
            
        Returns:
            Liste der ähnlichsten Texte mit Ähnlichkeitswerten
        """
        try:
            # Einbettungen für Anfrage und Korpus erzeugen
            query_embedding = self.get_embeddings(query).reshape(1, -1)
            corpus_embeddings = self.get_embeddings(corpus)
            
            # Kosinus-Ähnlichkeiten berechnen
            similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
            
            # Top-N ähnlichste Texte finden
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
            
        except Exception as e:
            logger.error(f"Fehler beim Finden ähnlicher Texte: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def generate_text_summary(self, text: str, max_length: int = 150) -> str:
        """
        Erzeugt eine Zusammenfassung eines längeren Textes.
        
        Args:
            text: Text, der zusammengefasst werden soll
            max_length: Maximale Länge der Zusammenfassung in Zeichen
            
        Returns:
            Zusammenfassung des Textes
        """
        try:
            # Vorverarbeitung des Textes
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) <= 1:
                return text[:max_length] + "..." if len(text) > max_length else text
            
            # Einbettungen für alle Sätze erzeugen
            sentence_embeddings = self.get_embeddings(sentences)
            
            # Durchschnittliche Einbettung berechnen (repräsentiert den Gesamttext)
            mean_embedding = np.mean(sentence_embeddings, axis=0).reshape(1, -1)
            
            # Ähnlichkeit jedes Satzes zum Durchschnitt berechnen
            similarities = np.dot(sentence_embeddings, mean_embedding.T).flatten()
            
            # Sätze nach Ähnlichkeit sortieren
            ranked_sentences = [(sentences[i], float(similarities[i])) for i in range(len(sentences))]
            ranked_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Top-Sätze auswählen, bis max_length erreicht ist
            summary = ""
            for sentence, _ in ranked_sentences:
                if len(summary) + len(sentence) <= max_length:
                    summary += sentence + " "
                else:
                    break
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Fehler bei der Textzusammenfassung: {str(e)}")
            logger.error(traceback.format_exc())
            return text[:max_length] + "..." if len(text) > max_length else text

###########################################
# 3. Prädiktive Modellierung
###########################################

class PredictiveModeler:
    """
    Klasse für die Entwicklung von prädiktiven Modellen, die verschiedene Datentypen
    verarbeiten und Vorhersagen treffen können.
    """
    
    def __init__(self, model_type: str = 'lstm', forecast_horizon: int = 7):
        """
        Initialisiert den Prädiktiven Modellierer.
        
        Args:
            model_type: Typ des zu verwendenden Modells ('lstm', 'transformer', 'gru', 'arima')
            forecast_horizon: Anzahl der Zeitschritte für Vorhersagen in die Zukunft
        """
        self.model_type = model_type.lower()
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.lookback = 10  # Standardwert für die Anzahl der zurückliegenden Zeitschritte
        self.feature_dims = None
        self.target_dims = None
        self.target_scaler = None
        self.history = None
        
    def _build_lstm_model(self, input_shape, output_dim):
        """Erstellt ein LSTM-Modell für Zeitreihenvorhersage"""
        model = keras.Sequential()
        model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(32))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(output_dim))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _build_transformer_model(self, input_shape, output_dim):
        """Erstellt ein Transformer-Modell für Zeitreihenvorhersage"""
        # Einfaches Transformer-Modell für Zeitreihen
        inputs = keras.Input(shape=input_shape)
        
        # Positional encoding layer
        class PositionalEncoding(layers.Layer):
            def __init__(self, position, d_model):
                super(PositionalEncoding, self).__init__()
                self.pos_encoding = self.positional_encoding(position, d_model)
                
            def get_angles(self, position, i, d_model):
                angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
                return position * angles
            
            def positional_encoding(self, position, d_model):
                angle_rads = self.get_angles(
                    position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                    i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                    d_model=d_model
                )
                
                # Apply sine to even indices
                sines = tf.math.sin(angle_rads[:, 0::2])
                # Apply cosine to odd indices
                cosines = tf.math.cos(angle_rads[:, 1::2])
                
                pos_encoding = tf.concat([sines, cosines], axis=-1)
                pos_encoding = pos_encoding[tf.newaxis, ...]
                
                return tf.cast(pos_encoding, tf.float32)
            
            def call(self, inputs):
                return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        
        x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
        
        # Multi-head attention layer
        x = layers.MultiHeadAttention(
            key_dim=input_shape[1], num_heads=4, dropout=0.1
        )(x, x, x, attention_mask=None, training=True)
        
        # Feed-forward network
        x = layers.Dropout(0.1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=1, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv1D(filters=input_shape[1], kernel_size=1)(x)
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Output layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(output_dim)(x)
        
        model = keras.Model(inputs, x)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def _build_gru_model(self, input_shape, output_dim):
        """Erstellt ein GRU-Modell für Zeitreihenvorhersage"""
        model = keras.Sequential()
        model.add(layers.GRU(64, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.GRU(32))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(output_dim))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _create_sequences(self, data, target=None, lookback=None):
        """
        Erstellt Sequenzen für die Zeitreihenmodellierung
        
        Args:
            data: Eingabedaten (numpy array)
            target: Zielvariablen (optional, numpy array)
            lookback: Anzahl der zurückliegenden Zeitschritte (optional)
            
        Returns:
            X: Sequenzen für die Eingabe
            y: Zielwerte (wenn target bereitgestellt wird)
        """
        if lookback is None:
            lookback = self.lookback
        
        X = []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
        X = np.array(X)
        
        if target is not None:
            y = target[lookback:]
            return X, y
        
        return X
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, 
            lookback: int = 10, epochs: int = 50, 
            validation_split: float = 0.2, batch_size: int = 32,
            verbose: int = 1):
        """
        Trainiert das prädiktive Modell mit den gegebenen Daten.
        
        Args:
            X: Eingabedaten (DataFrame)
            y: Zielvariablen (DataFrame, optional für Zeitreihen)
            lookback: Anzahl der zurückliegenden Zeitschritte für Zeitreihenmodelle
            epochs: Anzahl der Trainings-Epochen
            validation_split: Anteil der Daten für die Validierung
            batch_size: Batch-Größe für das Training
            verbose: Ausgabedetailstufe (0, 1, oder 2)
            
        Returns:
            self: Trainiertes Modell
        """
        try:
            self.lookback = lookback
            
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            # Skalierung der Eingabedaten
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_values)
            
            # Vorbereiten der Zielvariablen
            if y is not None:
                if isinstance(y, pd.DataFrame):
                    y_values = y.values
                else:
                    y_values = y
                    
                self.target_scaler = StandardScaler()
                y_scaled = self.target_scaler.fit_transform(y_values)
                self.target_dims = y_scaled.shape[1]
            else:
                # Wenn keine Zielvariablen bereitgestellt werden, nehmen wir an, dass X selbst eine Zeitreihe ist
                y_scaled = X_scaled
                self.target_dims = X_scaled.shape[1]
            
            self.feature_dims = X_scaled.shape[1]
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, lookback)
            
            # Modell basierend auf dem ausgewählten Typ erstellen
            input_shape = (lookback, self.feature_dims)
            output_dim = self.target_dims * self.forecast_horizon
            
            if self.model_type == 'lstm':
                self.model = self._build_lstm_model(input_shape, output_dim)
            elif self.model_type == 'transformer':
                self.model = self._build_transformer_model(input_shape, output_dim)
            elif self.model_type == 'gru':
                self.model = self._build_gru_model(input_shape, output_dim)
            else:
                raise ValueError(f"Nicht unterstützter Modelltyp: {self.model_type}")
            
            # Reshape y_sequences für das Forecast-Horizon
            y_prepared = y_sequences.reshape(y_sequences.shape[0], -1)
            
            # Modell trainieren
            self.history = self.model.fit(
                X_sequences, y_prepared,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des prädiktiven Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X: pd.DataFrame, return_sequences: bool = False) -> np.ndarray:
        """
        Macht Vorhersagen mit dem trainierten Modell.
        
        Args:
            X: Eingabedaten (DataFrame)
            return_sequences: Ob die Vorhersagesequenz zurückgegeben werden soll
            
        Returns:
            Vorhersagen für die Eingabedaten
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_values)
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences = self._create_sequences(X_scaled, lookback=self.lookback)
            
            # Vorhersagen machen
            predictions_scaled = self.model.predict(X_sequences)
            
            # Reshape für die Ausgabe
            predictions_scaled = predictions_scaled.reshape(
                predictions_scaled.shape[0], 
                self.forecast_horizon, 
                self.target_dims
            )
            
            # Rücktransformation
            predictions = np.zeros_like(predictions_scaled)
            for i in range(self.forecast_horizon):
                step_predictions = predictions_scaled[:, i, :]
                # Rücktransformation nur für jeden Zeitschritt
                predictions[:, i, :] = self.target_scaler.inverse_transform(step_predictions)
            
            if return_sequences:
                return predictions
            else:
                # Nur den ersten Vorhersageschritt zurückgeben
                return predictions[:, 0, :]
            
        except Exception as e:
            logger.error(f"Fehler bei der Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def forecast(self, X: pd.DataFrame, steps: int = None) -> np.ndarray:
        """
        Erstellt eine Vorhersage für mehrere Zeitschritte in die Zukunft.
        
        Args:
            X: Letzte bekannte Datenpunkte (mindestens lookback viele)
            steps: Anzahl der vorherzusagenden Schritte (Standard: forecast_horizon)
            
        Returns:
            Vorhersagesequenz für die nächsten 'steps' Zeitschritte
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        if steps is None:
            steps = self.forecast_horizon
            
        try:
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
            
            if len(X_values) < self.lookback:
                raise ValueError(f"Eingabedaten müssen mindestens {self.lookback} Zeitschritte enthalten")
            
            # Verwende nur die letzten 'lookback' Zeitschritte
            X_recent = X_values[-self.lookback:]
            
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_recent)
            X_sequence = X_scaled.reshape(1, self.lookback, self.feature_dims)
            
            # Erstelle die multi-step-Vorhersage
            forecast_values = []
            
            current_sequence = X_sequence.copy()
            
            for _ in range(steps):
                # Mache eine Vorhersage für den nächsten Schritt
                next_step_scaled = self.model.predict(current_sequence)[0]
                next_step_scaled = next_step_scaled.reshape(1, self.target_dims)
                
                # Rücktransformation
                next_step = self.target_scaler.inverse_transform(next_step_scaled)
                forecast_values.append(next_step[0])
                
                # Aktualisiere die Eingabesequenz für den nächsten Schritt
                # Entferne den ersten Zeitschritt und füge den neu vorhergesagten hinzu
                new_sequence = np.zeros_like(current_sequence)
                new_sequence[0, :-1, :] = current_sequence[0, 1:, :]
                new_sequence[0, -1, :] = next_step_scaled
                current_sequence = new_sequence
            
            return np.array(forecast_values)
            
        except Exception as e:
            logger.error(f"Fehler bei der Mehrschritt-Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluiert das Modell mit Testdaten.
        
        Args:
            X_test: Test-Eingabedaten
            y_test: Test-Zielvariablen
            
        Returns:
            Dictionary mit Bewertungsmetriken
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Testdaten
            if isinstance(X_test, pd.DataFrame):
                X_values = X_test.values
            else:
                X_values = X_test
                
            if isinstance(y_test, pd.DataFrame):
                y_values = y_test.values
            else:
                y_values = y_test
            
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_values)
            y_scaled = self.target_scaler.transform(y_values)
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, self.lookback)
            
            # Reshape y_sequences für das Forecast-Horizon
            y_prepared = y_sequences.reshape(y_sequences.shape[0], -1)
            
            # Modell evaluieren
            evaluation = self.model.evaluate(X_sequences, y_prepared, verbose=0)
            
            # Vorhersagen machen für detailliertere Metriken
            predictions = self.predict(X_test)
            
            # Tatsächliche Werte (ohne die ersten lookback Zeitschritte)
            actuals = y_values[self.lookback:]
            
            # Berechne RMSE, MAE, MAPE
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Entsprechende Anzahl an Vorhersagen auswählen
            predictions = predictions[:len(actuals)]
            
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # MAPE (Mean Absolute Percentage Error)
            # Vermeide Division durch Null
            mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
            
            return {
                'loss': evaluation[0],
                'mae': evaluation[1],
                'rmse': rmse,
                'mean_absolute_error': mae,
                'mean_absolute_percentage_error': mape
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Modellbewertung: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def plot_forecast(self, X: pd.DataFrame, y_true: pd.DataFrame = None, 
                     steps: int = None, feature_idx: int = 0,
                     save_path: str = None) -> plt.Figure:
        """
        Visualisiert die Vorhersage des Modells.
        
        Args:
            X: Eingabedaten
            y_true: Tatsächliche zukünftige Werte (optional)
            steps: Anzahl der vorherzusagenden Schritte
            feature_idx: Index der darzustellenden Feature-Dimension
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Standard-Schritte
            if steps is None:
                steps = self.forecast_horizon
            
            # Vorhersage erstellen
            forecast_values = self.forecast(X, steps)
            
            # Historische Daten (letzte lookback Zeitschritte)
            if isinstance(X, pd.DataFrame):
                historical_values = X.values[-self.lookback:, feature_idx]
            else:
                historical_values = X[-self.lookback:, feature_idx]
            
            # Zeitachse erstellen
            time_hist = np.arange(-self.lookback, 0)
            time_future = np.arange(0, steps)
            
            # Visualisierung erstellen
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historische Daten plotten
            ax.plot(time_hist, historical_values, 'b-', label='Historische Daten')
            
            # Vorhersage plotten
            ax.plot(time_future, forecast_values[:, feature_idx], 'r-', label='Vorhersage')
            
            # Tatsächliche zukünftige Werte plotten, falls vorhanden
            if y_true is not None:
                if isinstance(y_true, pd.DataFrame):
                    true_future = y_true.values[:steps, feature_idx]
                else:
                    true_future = y_true[:steps, feature_idx]
                
                ax.plot(time_future[:len(true_future)], true_future, 'g-', label='Tatsächliche Werte')
            
            # Grafik anpassen
            ax.set_title(f'Zeitreihenvorhersage mit {self.model_type.upper()}')
            ax.set_xlabel('Zeitschritte')
            ax.set_ylabel(f'Feature {feature_idx}')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Trennlinie zwischen historischen und Vorhersagedaten
            ax.axvline(x=0, color='k', linestyle='--')
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung der Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        model_data = {
            "model_type": self.model_type,
            "forecast_horizon": self.forecast_horizon,
            "lookback": self.lookback,
            "feature_dims": self.feature_dims,
            "target_dims": self.target_dims,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere das Model-Objekt
        self.model.save(f"{path}_model")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        joblib.dump(self.target_scaler, f"{path}_target_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        predictor = cls(
            model_type=model_data['model_type'],
            forecast_horizon=model_data['forecast_horizon']
        )
        
        predictor.lookback = model_data['lookback']
        predictor.feature_dims = model_data['feature_dims']
        predictor.target_dims = model_data['target_dims']
        predictor.is_fitted = model_data['is_fitted']
        
        # Lade das Model-Objekt
        predictor.model = keras.models.load_model(f"{path}_model")
        
        # Lade die Scaler
        predictor.scaler = joblib.load(f"{path}_scaler.joblib")
        predictor.target_scaler = joblib.load(f"{path}_target_scaler.joblib")
        
        return predictor

###########################################
# 4. Datensynthese und GAN-basierte Modelle
###########################################

class DataSynthesizer:
    """
    Klasse zur Generierung synthetischer Daten basierend auf realen Beispielen.
    Verwendet GAN (Generative Adversarial Network) für realistische Datensynthese.
    """
    
    def __init__(self, categorical_threshold: int = 10, noise_dim: int = 100):
        """
        Initialisiert den Datensynthetisierer.
        
        Args:
            categorical_threshold: Anzahl eindeutiger Werte, ab der eine Spalte als kategorisch gilt
            noise_dim: Dimension des Rauschvektors für den Generator
        """
        self.categorical_threshold = categorical_threshold
        self.noise_dim = noise_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        
        self.column_types = {}  # Speichert, ob eine Spalte kategorisch oder kontinuierlich ist
        self.categorical_mappings = {}  # Speichert Mappings für kategorische Spalten
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # Für kontinuierliche Variablen
        
        self.is_fitted = False
        self.feature_dims = None
        self.training_data = None
    
    def _identify_column_types(self, data: pd.DataFrame):
        """Identifiziert, ob Spalten kategorisch oder kontinuierlich sind"""
        self.column_types = {}
        
        for col in data.columns:
            n_unique = data[col].nunique()
            
            # Wenn die Anzahl eindeutiger Werte kleiner als der Schwellenwert ist oder
            # der Datentyp ist nicht numerisch, behandle die Spalte als kategorisch
            if n_unique < self.categorical_threshold or not pd.api.types.is_numeric_dtype(data[col]):
                self.column_types[col] = 'categorical'
                
                # Erstelle Mapping von Kategorien zu Zahlen
                categories = data[col].unique()
                self.categorical_mappings[col] = {
                    cat: i for i, cat in enumerate(categories)
                }
                # Umgekehrtes Mapping für die Rücktransformation
                self.categorical_mappings[f"{col}_reverse"] = {
                    i: cat for i, cat in enumerate(categories)
                }
            else:
                self.column_types[col] = 'continuous'
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Vorverarbeitung der Daten für das GAN"""
        processed_data = pd.DataFrame()
        
        for col in data.columns:
            if self.column_types[col] == 'categorical':
                # One-Hot-Encoding für kategorische Spalten
                mapped_col = data[col].map(self.categorical_mappings[col])
                one_hot = pd.get_dummies(mapped_col, prefix=col)
                processed_data = pd.concat([processed_data, one_hot], axis=1)
            else:
                # Skalierung für kontinuierliche Spalten
                processed_data[col] = data[col]
        
        # Skaliere alle Spalten auf [-1, 1]
        return self.scaler.fit_transform(processed_data)
    
    def _postprocess_data(self, generated_data: np.ndarray) -> pd.DataFrame:
        """Nachverarbeitung der generierten Daten zurück in das ursprüngliche Format"""
        # Rücktransformation der Skalierung
        rescaled_data = self.scaler.inverse_transform(generated_data)
        
        # Erstelle einen DataFrame mit den ursprünglichen Spalten
        result = pd.DataFrame()
        
        col_idx = 0
        for col, col_type in self.column_types.items():
            if col_type == 'categorical':
                # Anzahl der eindeutigen Werte für diese kategorische Spalte
                n_categories = len(self.categorical_mappings[col])
                
                # Extrahiere die One-Hot-kodierten Werte
                cat_values = rescaled_data[:, col_idx:col_idx+n_categories]
                
                # Konvertiere von One-Hot zurück zu kategorischen Werten
                # Nehme die Kategorie mit dem höchsten Wert
                cat_indices = np.argmax(cat_values, axis=1)
                
                # Mappe zurück zu den ursprünglichen Kategorien
                result[col] = [self.categorical_mappings[f"{col}_reverse"][idx] for idx in cat_indices]
                
                col_idx += n_categories
            else:
                # Kontinuierliche Spalte einfach übernehmen
                result[col] = rescaled_data[:, col_idx]
                col_idx += 1
        
        return result
    
    def _build_generator(self, output_dim):
        """Erstellt den Generator für das GAN"""
        model = keras.Sequential([
            layers.Dense(256, input_dim=self.noise_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(output_dim, activation='tanh')  # tanh für Output im Bereich [-1, 1]
        ])
        return model
    
    def _build_discriminator(self, input_dim):
        """Erstellt den Diskriminator für das GAN"""
        model = keras.Sequential([
            layers.Dense(512, input_dim=input_dim, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy')
        return model
    
    def _build_gan(self, generator, discriminator):
        """Kombiniert Generator und Diskriminator zum GAN"""
        discriminator.trainable = False  # Diskriminator beim GAN-Training nicht aktualisieren
        
        model = keras.Sequential([
            generator,
            discriminator
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy')
        return model
    
    def fit(self, data: pd.DataFrame, epochs: int = 2000, batch_size: int = 32, 
           sample_interval: int = 100, verbose: int = 1):
        """
        Trainiert das GAN-Modell mit den gegebenen Daten.
        
        Args:
            data: Eingabedaten (DataFrame)
            epochs: Anzahl der Trainings-Epochen
            batch_size: Batch-Größe für das Training
            sample_interval: Intervall für Stichproben der generierten Daten
            verbose: Ausgabedetailstufe (0, 1, oder 2)
            
        Returns:
            self: Trainiertes Modell
        """
        try:
            # Identifiziere Spaltentypen
            self._identify_column_types(data)
            
            # Vorverarbeitung der Daten
            processed_data = self._preprocess_data(data)
            self.feature_dims = processed_data.shape[1]
            
            # Speichere trainierte Daten für spätere Validierung
            self.training_data = data.copy()
            
            # Baue das GAN-Modell
            self.generator = self._build_generator(self.feature_dims)
            self.discriminator = self._build_discriminator(self.feature_dims)
            self.gan = self._build_gan(self.generator, self.discriminator)
            
            # Label für echte und gefälschte Daten
            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            
            # Trainingsschleife
            for epoch in range(epochs):
                # ---------------------
                #  Trainiere Diskriminator
                # ---------------------
                
                # Wähle eine zufällige Batch aus echten Daten
                idx = np.random.randint(0, processed_data.shape[0], batch_size)
                real_data = processed_data[idx]
                
                # Generiere eine Batch aus gefälschten Daten
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                fake_data = self.generator.predict(noise)
                
                # Trainiere den Diskriminator
                d_loss_real = self.discriminator.train_on_batch(real_data, real)
                d_loss_fake = self.discriminator.train_on_batch(fake_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ---------------------
                #  Trainiere Generator
                # ---------------------
                
                # Generiere neue Batch aus Rauschen
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                
                # Trainiere den Generator
                g_loss = self.gan.train_on_batch(noise, real)
                
                # Ausgabe für Fortschrittsüberwachung
                if verbose > 0 and epoch % sample_interval == 0:
                    print(f"Epoch {epoch}/{epochs} [D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}]")
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des GAN-Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def generate(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generiert synthetische Daten.
        
        Args:
            n_samples: Anzahl der zu generierenden Datensätze
            
        Returns:
            DataFrame mit synthetischen Daten im Format der Trainingsdaten
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere Rauschen als Input für den Generator
            noise = np.random.normal(0, 1, (n_samples, self.noise_dim))
            
            # Generiere Daten
            generated_data = self.generator.predict(noise)
            
            # Nachverarbeitung der Daten
            synthetic_data = self._postprocess_data(generated_data)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Fehler bei der Datengenerierung: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate_quality(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Bewertet die Qualität der generierten Daten durch Vergleich mit den Trainingsdaten.
        
        Args:
            n_samples: Anzahl der zu generierenden und bewertenden Datensätze
            
        Returns:
            Dictionary mit Qualitätsmetriken
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere synthetische Daten
            synthetic_data = self.generate(n_samples)
            
            # Statistischer Vergleich zwischen echten und synthetischen Daten
            metrics = {}
            
            # Vergleiche Mittelwerte und Standardabweichungen für kontinuierliche Spalten
            for col, col_type in self.column_types.items():
                if col_type == 'continuous':
                    # Berechne Mittelwert und Standardabweichung für echte Daten
                    real_mean = self.training_data[col].mean()
                    real_std = self.training_data[col].std()
                    
                    # Berechne dieselben Statistiken für synthetische Daten
                    synth_mean = synthetic_data[col].mean()
                    synth_std = synthetic_data[col].std()
                    
                    # Berechne die relative Differenz
                    mean_diff = abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10)
                    std_diff = abs(real_std - synth_std) / (abs(real_std) + 1e-10)
                    
                    metrics[f"{col}_mean_diff"] = float(mean_diff)
                    metrics[f"{col}_std_diff"] = float(std_diff)
                else:
                    # Vergleiche die Verteilung kategorischer Werte
                    real_dist = self.training_data[col].value_counts(normalize=True)
                    synth_dist = synthetic_data[col].value_counts(normalize=True)
                    
                    # Berechne die Jensen-Shannon-Divergenz
                    # (symmetrische Version der KL-Divergenz)
                    js_divergence = 0.0
                    
                    # Stelle sicher, dass beide Verteilungen dieselben Kategorien haben
                    all_categories = set(real_dist.index) | set(synth_dist.index)
                    
                    for cat in all_categories:
                        p = real_dist.get(cat, 0)
                        q = synth_dist.get(cat, 0)
                        
                        # Vermeide Logarithmus von 0
                        if p > 0 and q > 0:
                            m = 0.5 * (p + q)
                            js_divergence += 0.5 * (p * np.log(p / m) + q * np.log(q / m))
                    
                    metrics[f"{col}_js_divergence"] = float(js_divergence)
            
            # Gesamtqualitätsmetrik
            # Durchschnitt der normalisierten Abweichungen (niedriger ist besser)
            continuous_diffs = [v for k, v in metrics.items() if k.endswith('_diff')]
            categorical_diffs = [v for k, v in metrics.items() if k.endswith('_js_divergence')]
            
            if continuous_diffs:
                metrics['continuous_avg_diff'] = float(np.mean(continuous_diffs))
            if categorical_diffs:
                metrics['categorical_avg_diff'] = float(np.mean(categorical_diffs))
            
            # Gesamtbewertung (0 bis 1, höher ist besser)
            overall_score = 1.0
            if continuous_diffs:
                overall_score -= 0.5 * min(1.0, np.mean(continuous_diffs))
            if categorical_diffs:
                overall_score -= 0.5 * min(1.0, np.mean(categorical_diffs))
            
            metrics['overall_quality_score'] = float(overall_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Fehler bei der Qualitätsbewertung: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def plot_comparison(self, n_samples: int = 1000, 
                       features: List[str] = None,
                       save_path: str = None) -> plt.Figure:
        """
        Visualisiert einen Vergleich zwischen echten und synthetischen Daten.
        
        Args:
            n_samples: Anzahl der zu generierenden Datensätze
            features: Liste der darzustellenden Features (Standard: alle)
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere synthetische Daten
            synthetic_data = self.generate(n_samples)
            
            # Wähle die darzustellenden Features aus
            if features is None:
                # Wähle bis zu 6 Features für die Visualisierung
                features = list(self.column_types.keys())[:min(6, len(self.column_types))]
            
            # Bestimme die Anzahl der Zeilen und Spalten für das Subplot-Raster
            n_features = len(features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            # Erstelle die Figur
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
            if n_rows * n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, feature in enumerate(features):
                ax = axes[i]
                
                if self.column_types[feature] == 'continuous':
                    # Histogramm für kontinuierliche Variablen
                    sns.histplot(self.training_data[feature], kde=True, ax=ax, color='blue', alpha=0.5, label='Echte Daten')
                    sns.histplot(synthetic_data[feature], kde=True, ax=ax, color='red', alpha=0.5, label='Synthetische Daten')
                else:
                    # Balkendiagramm für kategorische Variablen
                    real_counts = self.training_data[feature].value_counts(normalize=True)
                    synth_counts = synthetic_data[feature].value_counts(normalize=True)
                    
                    # Kombiniere beide, um alle Kategorien zu erfassen
                    all_cats = sorted(set(real_counts.index) | set(synth_counts.index))
                    
                    # Erstelle ein DataFrame für Seaborn
                    plot_data = []
                    for cat in all_cats:
                        plot_data.append({'Category': cat, 'Frequency': real_counts.get(cat, 0), 'Type': 'Real'})
                        plot_data.append({'Category': cat, 'Frequency': synth_counts.get(cat, 0), 'Type': 'Synthetic'})
                    
                    plot_df = pd.DataFrame(plot_data)
                    
                    # Balkendiagramm
                    sns.barplot(x='Category', y='Frequency', hue='Type', data=plot_df, ax=ax)
                
                ax.set_title(f'Verteilung von {feature}')
                ax.legend()
                
                # Achsen anpassen
                if self.column_types[feature] == 'categorical':
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Verstecke ungenutzte Subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung des Datenvergleichs: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        model_data = {
            "categorical_threshold": self.categorical_threshold,
            "noise_dim": self.noise_dim,
            "feature_dims": self.feature_dims,
            "column_types": self.column_types,
            "categorical_mappings": self.categorical_mappings,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere die Modelle
        self.generator.save(f"{path}_generator")
        self.discriminator.save(f"{path}_discriminator")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten und Mappings
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        synthesizer = cls(
            categorical_threshold=model_data['categorical_threshold'],
            noise_dim=model_data['noise_dim']
        )
        
        synthesizer.feature_dims = model_data['feature_dims']
        synthesizer.column_types = model_data['column_types']
        synthesizer.categorical_mappings = model_data['categorical_mappings']
        synthesizer.is_fitted = model_data['is_fitted']
        
        # Lade die Modelle
        synthesizer.generator = keras.models.load_model(f"{path}_generator")
        synthesizer.discriminator = keras.models.load_model(f"{path}_discriminator")
        
        # Lade den Scaler
        synthesizer.scaler = joblib.load(f"{path}_scaler.joblib")
        
        # Rekonstruiere das GAN
        synthesizer.gan = synthesizer._build_gan(synthesizer.generator, synthesizer.discriminator)
        
        return synthesizer

###########################################
# 5. Federated Learning und Compute-to-Data
###########################################

class ComputeToDataManager:
    """
    Klasse zur Verwaltung von Compute-to-Data-Operationen für den Datenschutz.
    Ermöglicht die Ausführung von Berechnungen auf sensiblen Daten ohne direkten Zugriff.
    """
    
    def __init__(self, encryption_key: bytes = None):
        """
        Initialisiert den Compute-to-Data-Manager.
        
        Args:
            encryption_key: Schlüssel für die Verschlüsselung (optional)
        """
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.allowed_operations = {
            'aggregate': self._aggregate,
            'count': self._count,
            'mean': self._mean,
            'sum': self._sum,
            'min': self._min,
            'max': self._max,
            'std': self._std,
            'correlation': self._correlation,
            'histogram': self._histogram,
            'custom_model': self._custom_model_inference
        }
        
        # Datenschutzkonfiguration
        self.privacy_config = {
            'min_group_size': 5,  # Minimale Gruppengröße für Aggregationen
            'noise_level': 0.01,  # Standardrauschen für Differentialprivatsphäre
            'outlier_removal': True  # Entfernung von Ausreißern für sensible Berechnungen
        }
    
    def set_privacy_config(self, min_group_size: int = None, noise_level: float = None, 
                          outlier_removal: bool = None):
        """Aktualisiert die Datenschutzkonfiguration"""
        if min_group_size is not None:
            self.privacy_config['min_group_size'] = min_group_size
        if noise_level is not None:
            self.privacy_config['noise_level'] = noise_level
        if outlier_removal is not None:
            self.privacy_config['outlier_removal'] = outlier_removal
    
    def _encrypt_data(self, data: pd.DataFrame) -> bytes:
        """Verschlüsselt Daten für die sichere Speicherung"""
        serialized = data.to_json().encode()
        return self.cipher_suite.encrypt(serialized)
    
    def _decrypt_data(self, encrypted_data: bytes) -> pd.DataFrame:
        """Entschlüsselt Daten für die Verarbeitung"""
        decrypted = self.cipher_suite.decrypt(encrypted_data)
        return pd.read_json(decrypted.decode())
    
    def _add_differential_privacy_noise(self, result, scale: float = None):
        """Fügt Rauschen für Differential Privacy hinzu"""
        scale = scale or self.privacy_config['noise_level']
        
        if isinstance(result, (int, float)):
            return result + np.random.normal(0, scale * abs(result))
        elif isinstance(result, np.ndarray):
            return result + np.random.normal(0, scale * np.abs(result).mean(), result.shape)
        elif isinstance(result, pd.Series):
            noisy_values = result.values + np.random.normal(0, scale * np.abs(result.values).mean(), result.shape)
            return pd.Series(noisy_values, index=result.index)
        elif isinstance(result, pd.DataFrame):
            noisy_values = result.values + np.random.normal(0, scale * np.abs(result.values).mean(), result.shape)
            return pd.DataFrame(noisy_values, index=result.index, columns=result.columns)
        else:
            return result
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Entfernt Ausreißer aus den Daten"""
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            return data
        
        # Z-Score-basierte Ausreißererkennung
        z_scores = ((numeric_data - numeric_data.mean()) / numeric_data.std()).abs()
        mask = (z_scores < 3).all(axis=1)
        
        return data[mask]
    
    def _validate_group_size(self, data: pd.DataFrame, group_column: str = None, 
                            group_value: Any = None) -> bool:
        """
        Prüft, ob eine Gruppe die Mindestgröße für Datenschutz erfüllt
        
        Args:
            data: Daten
            group_column: Spalte für die Gruppierung
            group_value: Wert für die Filterung
            
        Returns:
            True, wenn die Gruppe groß genug ist
        """
        min_size = self.privacy_config['min_group_size']
        
        if group_column is not None and group_value is not None:
            group_data = data[data[group_column] == group_value]
            return len(group_data) >= min_size
        
        return len(data) >= min_size
    
    def _prepare_data_for_computation(self, data: pd.DataFrame, columns: List[str] = None,
                                    group_column: str = None, group_value: Any = None) -> pd.DataFrame:
        """
        Bereitet Daten für die Berechnung vor, einschließlich Filterung und Datenschutzprüfungen
        
        Args:
            data: Eingabedaten
            columns: Zu verwendende Spalten
            group_column: Spalte für die Gruppierung
            group_value: Wert für die Filterung
            
        Returns:
            Vorbereitete Daten
        """
        # Filtere Daten nach Gruppe, falls angegeben
        if group_column is not None and group_value is not None:
            prepared_data = data[data[group_column] == group_value]
        else:
            prepared_data = data.copy()
        
        # Prüfe Mindestgruppengröße
        if not self._validate_group_size(prepared_data):
            raise ValueError(f"Gruppe zu klein für Datenschutz (min: {self.privacy_config['min_group_size']})")
        
        # Entferne Ausreißer, falls konfiguriert
        if self.privacy_config['outlier_removal']:
            prepared_data = self._remove_outliers(prepared_data)
        
        # Wähle Spalten aus, falls angegeben
        if columns is not None:
            prepared_data = prepared_data[columns]
        
        return prepared_data
    
    # Implementierte Operationen für C2D
    
    def _aggregate(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Aggregiert Daten nach Spalten und Gruppen"""
        columns = params.get('columns')
        group_by = params.get('group_by')
        aggregations = params.get('aggregations', ['count', 'mean'])
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        if group_by:
            grouped = prepared_data.groupby(group_by)
            
            # Prüfe Mindestgröße für jede Gruppe
            group_sizes = grouped.size()
            valid_groups = group_sizes[group_sizes >= self.privacy_config['min_group_size']].index
            
            # Filtere ungültige Gruppen
            results = {}
            for agg in aggregations:
                if agg == 'count':
                    agg_result = grouped.size()
                else:
                    agg_func = getattr(grouped, agg, None)
                    if agg_func is None:
                        continue
                    agg_result = agg_func()
                
                # Behalte nur valide Gruppen
                agg_result = agg_result.loc[valid_groups]
                
                # Füge Rauschen hinzu
                results[agg] = self._add_differential_privacy_noise(agg_result)
        else:
            results = {}
            for agg in aggregations:
                if agg == 'count':
                    agg_result = len(prepared_data)
                else:
                    agg_func = getattr(prepared_data, agg, None)
                    if agg_func is None:
                        continue
                    agg_result = agg_func()
                
                # Füge Rauschen hinzu
                results[agg] = self._add_differential_privacy_noise(agg_result)
        
        # Konvertiere zu serialisierbarem Format
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
                serializable_results[k] = v.to_dict()
            else:
                serializable_results[k] = v
        
        return serializable_results
    
    def _count(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Zählt Datensätze nach Filterkriterien"""
        filter_column = params.get('filter_column')
        filter_value = params.get('filter_value')
        
        if filter_column and filter_value is not None:
            count = len(data[data[filter_column] == filter_value])
        else:
            count = len(data)
        
        # Füge Rauschen hinzu und runde auf ganze Zahlen
        noisy_count = round(self._add_differential_privacy_noise(count))
        
        # Stelle sicher, dass der Wert nicht negativ ist
        return {'count': max(0, noisy_count)}
    
    def _mean(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet den Mittelwert für numerische Spalten"""
        columns = params.get('columns')
        group_column = params.get('group_column')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        if group_column:
            grouped = prepared_data.groupby(group_column)
            
            # Prüfe Mindestgröße für jede Gruppe
            group_sizes = grouped.size()
            valid_groups = group_sizes[group_sizes >= self.privacy_config['min


          if col in anonymized_data.columns and anonymized_data[col].dtype == 'object':
                        # Behalte nur den ersten Teil der Adresse/Standorts (z.B. Stadt)
                        anonymized_data[col] = anonymized_data[col].astype(str).apply(
                            lambda x: x.split(',')[0] if ',' in x else x
                        )
                
                # Vergröbere Zeitstempel (auf Stundenebene)
                for col in datetime_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = pd.to_datetime(anonymized_data[col]).dt.floor('H')
                
            elif privacy_level == 'low':
                # Minimale Anonymisierung, nur offensichtlich persönliche Kennung anonymisieren
                for col in id_columns:
                    if col in anonymized_data.columns and 'user' in col.lower():
                        salt = datetime.now().strftime("%Y%m%d")
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest() if pd.notna(x) else x
                        )
            
            # 2. Berechnung des Datenwerts und Erstellung von Metadaten
            
            # a) Berechne Basiskennzahlen
            metadata = {
                'source_type': source_type,
                'privacy_level': privacy_level,
                'record_count': len(anonymized_data),
                'field_count': len(anonymized_data.columns),
                'time_range': None,
                'created_at': datetime.now().isoformat(),
                'data_schema': {
                    col: str(anonymized_data[col].dtype) for col in anonymized_data.columns
                },
                'estimated_value': 0.0,
                'value_factors': {},
                'column_statistics': {},
                'potential_use_cases': []
            }
            
            # b) Berechne Zeitspanne, falls Zeitstempelspalten existieren
            if datetime_columns:
                # Wähle die erste Zeitstempelspalte
                time_col = datetime_columns[0]
                min_date = anonymized_data[time_col].min()
                max_date = anonymized_data[time_col].max()
                
                metadata['time_range'] = {
                    'start': min_date.isoformat() if pd.notna(min_date) else None,
                    'end': max_date.isoformat() if pd.notna(max_date) else None,
                    'duration_days': (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None
                }
            
            # c) Berechne Spaltenstatistiken für numerische und kategorische Spalten
            for col in anonymized_data.columns:
                col_stats = {}
                
                if pd.api.types.is_numeric_dtype(anonymized_data[col]):
                    # Statistiken für numerische Spalten
                    col_stats = {
                        'type': 'numeric',
                        'min': float(anonymized_data[col].min()) if not pd.isna(anonymized_data[col].min()) else None,
                        'max': float(anonymized_data[col].max()) if not pd.isna(anonymized_data[col].max()) else None,
                        'mean': float(anonymized_data[col].mean()) if not pd.isna(anonymized_data[col].mean()) else None,
                        'median': float(anonymized_data[col].median()) if not pd.isna(anonymized_data[col].median()) else None,
                        'std': float(anonymized_data[col].std()) if not pd.isna(anonymized_data[col].std()) else None,
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100)
                    }
                elif pd.api.types.is_string_dtype(anonymized_data[col]) or anonymized_data[col].dtype == 'object':
                    # Statistiken für kategorische/Text-Spalten
                    col_stats = {
                        'type': 'categorical' if anonymized_data[col].nunique() < 30 else 'text',
                        'unique_values': int(anonymized_data[col].nunique()),
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100),
                        'most_common': anonymized_data[col].value_counts().head(5).to_dict() if anonymized_data[col].nunique() < 30 else None
                    }
                    
                    # Wenn es sich um Text handelt, zusätzliche Textstatistiken
                    if col_stats['type'] == 'text':
                        # Durchschnittliche Textlänge
                        col_stats['avg_length'] = float(anonymized_data[col].astype(str).str.len().mean())
                        
                        # Optional: Sentimentanalyse auf Stichprobe
                        if self.config.get('analyze_text_sentiment', True):
                            sample = anonymized_data[col].dropna().sample(min(100, anonymized_data[col].dropna().shape[0])).tolist()
                            if sample:
                                sentiment_results = self.semantic_analyzer.analyze_sentiment(sample)
                                sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
                                for result in sentiment_results:
                                    sentiment_counts[result['sentiment']] += 1
                                
                                col_stats['sentiment'] = {
                                    k: v / len(sentiment_results) for k, v in sentiment_counts.items()
                                }
                else:
                    # Andere Datentypen (Datum, Boolean, etc.)
                    col_stats = {
                        'type': str(anonymized_data[col].dtype),
                        'unique_values': int(anonymized_data[col].nunique()),
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100)
                    }
                
                metadata['column_statistics'][col] = col_stats
            
            # d) Berechnung des geschätzten Datenwerts basierend auf verschiedenen Faktoren
            value_factors = {}
            
            # Faktor: Datenmenge
            data_size_factor = min(1.0, len(anonymized_data) / 10000)  # Skaliert bis 10.000 Datenpunkte
            value_factors['data_size'] = float(data_size_factor)
            
            # Faktor: Datenvollständigkeit
            completeness = 1.0 - anonymized_data.isna().mean().mean()  # Prozentsatz der nicht-fehlenden Werte
            value_factors['completeness'] = float(completeness)
            
            # Faktor: Zeitliche Spanne (falls verfügbar)
            time_factor = 0.5  # Standardwert
            if metadata['time_range'] and metadata['time_range']['duration_days']:
                # Höherer Wert für längere Zeitreihen (skaliert bis zu 1 Jahr)
                days = metadata['time_range']['duration_days']
                time_factor = min(1.0, days / 365)
            value_factors['time_span'] = float(time_factor)
            
            # Faktor: Datenquellentyp (unterschiedliche Gewichtung je nach Quelle)
            source_weights = {
                'browser': 0.7,
                'calendar': 0.6,
                'chat': 0.8,
                'social_media': 0.9,
                'streaming': 0.8,
                'health_insurance': 1.0,
                'health_data': 1.0,
                'smartwatch': 0.9,
                'iot_vacuum': 0.5,
                'iot_thermostat': 0.6,
                'iot_lighting': 0.5,
                'iot_security': 0.8,
                'smart_home': 0.7
            }
            source_factor = source_weights.get(source_type, 0.5)
            value_factors['source_type'] = float(source_factor)
            
            # Faktor: Datenschutzniveau (höherer Datenschutz = geringerer Wert)
            privacy_factors = {
                'low': 1.0,
                'medium': 0.8,
                'high': 0.6
            }
            privacy_factor = privacy_factors.get(privacy_level, 0.7)
            value_factors['privacy_level'] = float(privacy_factor)
            
            # Schätze den Gesamtwert als gewichteten Durchschnitt der Faktoren
            weights = {
                'data_size': 0.3,
                'completeness': 0.2,
                'time_span': 0.15,
                'source_type': 0.25,
                'privacy_level': 0.1
            }
            
            estimated_value = sum(value_factors[factor] * weights[factor] for factor in weights)
            
            # Konvertiere den normierten Wert in einen Tokenwert (angenommene Basis: 1-10 OCEAN)
            base_token_value = self.config.get('base_token_value', 5.0)  # 5 OCEAN als Basiswert
            token_value = base_token_value * estimated_value
            
            metadata['estimated_value'] = float(token_value)
            metadata['value_factors'] = value_factors
            
            # e) Identifiziere potenzielle Anwendungsfälle basierend auf den Daten
            use_cases = []
            
            # Browser-Daten: Surfgewohnheiten, Werbezielgruppen
            if source_type == 'browser':
                use_cases.append({
                    'title': 'Consumer Behavior Analysis',
                    'description': 'Analyze browsing patterns to understand consumer preferences and interests',
                    'value_proposition': 'Optimize marketing strategies and product recommendations'
                })
                use_cases.append({
                    'title': 'Ad Targeting Optimization',
                    'description': 'Use browsing history to create more relevant ad targeting profiles',
                    'value_proposition': 'Improve ad conversion rates and reduce marketing waste'
                })
            
            # Kalender-Daten: Zeitplanung, Produktivitätsanalyse
            elif source_type == 'calendar':
                use_cases.append({
                    'title': 'Productivity Pattern Analysis',
                    'description': 'Identify optimal meeting times and productivity patterns',
                    'value_proposition': 'Improve organizational efficiency and meeting scheduling'
                })
                use_cases.append({
                    'title': 'Work-Life Balance Insights',
                    'description': 'Analyze time allocation between work and personal activities',
                    'value_proposition': 'Develop better wellness and productivity programs'
                })
            
            # Gesundheitsdaten: Medizinische Forschung, Versicherungsanalyse
            elif source_type in ['health_data', 'health_insurance', 'smartwatch']:
                use_cases.append({
                    'title': 'Health Trend Analysis',
                    'description': 'Identify health patterns and correlations in anonymized health data',
                    'value_proposition': 'Support medical research and health intervention programs'
                })
                use_cases.append({
                    'title': 'Wellness Program Optimization',
                    'description': 'Use activity and health data to design better wellness initiatives',
                    'value_proposition': 'Improve health outcomes and reduce healthcare costs'
                })
            
            # IoT-Daten: Smart Home, Energieoptimierung
            elif source_type in ['iot_vacuum', 'iot_thermostat', 'iot_lighting', 'smart_home']:
                use_cases.append({
                    'title': 'Energy Usage Optimization',
                    'description': 'Analyze home energy consumption patterns',
                    'value_proposition': 'Develop more efficient energy management solutions'
                })
                use_cases.append({
                    'title': 'Smart Home Product Development',
                    'description': 'Understand usage patterns of smart home devices',
                    'value_proposition': 'Inform the design of new IoT products and services'
                })
            
            # Social Media und Chat: Stimmungsanalyse, Trendforschung
            elif source_type in ['social_media', 'chat']:
                use_cases.append({
                    'title': 'Sentiment Analysis and Trend Detection',
                    'description': 'Track public opinion and emerging topics in social conversations',
                    'value_proposition': 'Stay ahead of market trends and consumer sentiment shifts'
                })
                use_cases.append({
                    'title': 'Content Strategy Optimization',
                    'description': 'Analyze engagement patterns to optimize content strategy',
                    'value_proposition': 'Improve audience engagement and content performance'
                })
            
            # Streaming: Medienkonsum, Empfehlungen
            elif source_type == 'streaming':
                use_cases.append({
                    'title': 'Content Preference Analysis',
                    'description': 'Understand viewer preferences and consumption patterns',
                    'value_proposition': 'Optimize content creation and acquisition strategies'
                })
                use_cases.append({
                    'title': 'Recommendation Engine Improvement',
                    'description': 'Enhance content recommendation algorithms with viewing pattern data',
                    'value_proposition': 'Increase viewer satisfaction and platform engagement'
                })
            
            # Generische Anwendungsfälle für andere Quellen
            else:
                use_cases.append({
                    'title': 'Pattern Recognition',
                    'description': 'Identify patterns and trends in the data',
                    'value_proposition': 'Gain insights for business strategy and decision making'
                })
                use_cases.append({
                    'title': 'Predictive Analytics',
                    'description': 'Use historical data to predict future trends',
                    'value_proposition': 'Improve forecasting and strategic planning'
                })
            
            metadata['potential_use_cases'] = use_cases
            
            # Erstelle einen Asset für die Compute-to-Data-Funktionalität
            c2d_asset = self.c2d_manager.create_data_asset(
                anonymized_data,
                asset_metadata=metadata
            )
            
            return {
                'anonymized_data': anonymized_data,
                'metadata': metadata,
                'c2d_asset': c2d_asset
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenaufbereitung für Monetarisierung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'anonymized_data': None,
                'metadata': {
                    'source_type': source_type,
                    'privacy_level': privacy_level,
                    'error': str(e)
                }
            }
    
    def combine_data_sources(self, sources: List[Dict], combination_type: str = 'merge') -> Dict:
        """
        Kombiniert mehrere Datenquellen zu einem wertvolleren Datensatz.
        
        Args:
            sources: Liste von Datenquellen mit 'data' und 'metadata' Schlüsseln
            combination_type: Art der Kombination ('merge', 'enrich', 'correlate')
            
        Returns:
            Kombinierte Daten mit verbesserten Metadaten
        """
        try:
            self.logger.info(f"Kombiniere {len(sources)} Datenquellen mit Methode {combination_type}")
            
            # Prüfe, ob gültige Quellen vorhanden sind
            valid_sources = [s for s in sources if 'anonymized_data' in s and s['anonymized_data'] is not None]
            
            if len(valid_sources) < 2:
                raise ValueError("Mindestens zwei gültige Datenquellen werden für die Kombination benötigt")
            
            combined_data = None
            combined_metadata = {
                'combination_type': combination_type,
                'source_count': len(valid_sources),
                'source_types': [s['metadata']['source_type'] for s in valid_sources],
                'created_at': datetime.now().isoformat(),
                'record_count': 0,
                'field_count': 0,
                'estimated_value': 0.0,
                'combination_description': '',
                'source_metadata': [s['metadata'] for s in valid_sources]
            }
            
            # Implementierung je nach Kombinationstyp
            if combination_type == 'merge':
                # Horizontale Zusammenführung (Zeilen anfügen)
                # Geeignet für gleichartige Datensätze
                
                all_data = [s['anonymized_data'] for s in valid_sources]
                
                # Prüfe Spaltenkompatibilität
                common_columns = set.intersection(*[set(df.columns) for df in all_data])
                
                if not common_columns:
                    raise ValueError("Keine gemeinsamen Spalten für die Zusammenführung gefunden")
                
                # Zusammenführung der Daten mit gemeinsamen Spalten
                combined_data = pd.concat([df[list(common_columns)] for df in all_data], ignore_index=True)
                
                combined_metadata['combination_description'] = f"Horizontal merge of {len(valid_sources)} similar datasets, keeping {len(common_columns)} common columns"
                combined_metadata['record_count'] = len(combined_data)
                combined_metadata['field_count'] = len(combined_data.columns)
                
                # Wertschätzung: Summe der Einzelwerte mit einem Multiplikator für die Kombination
                base_value = sum(s['metadata'].get('estimated_value', 0) for s in valid_sources)
                combined_metadata['estimated_value'] = base_value * 1.2  # 20% Mehrwert durch Kombination
                
            elif combination_type == 'enrich':
                # Vertikale Anreicherung (Spalten anfügen)
                # Geeignet für komplementäre Datensätze mit einem gemeinsamen Identifikator
                
                # Erster Datensatz als Basis
                base_data = valid_sources[0]['anonymized_data']
                join_key = None
                
                # Suche nach einem gemeinsamen Schlüssel für die Verknüpfung
                for col in base_data.columns:
                    if all(col in s['anonymized_data'].columns for s in valid_sources[1:]):
                        # Prüfe, ob der Wert als Schlüssel dienen kann (eindeutige Werte)
                        if all(s['anonymized_data'][col].nunique() / len(s['anonymized_data']) > 0.7 for s in valid_sources):
                            join_key = col
                            break
                
                if not join_key:
                    # Falls kein expliziter Schlüssel existiert, erstelle einen künstlichen Schlüssel
                    self.logger.warning("Kein natürlicher Schlüssel gefunden, erstelle künstlichen Index")
                    
                    # Erstelle einen eindeutigen Index für jeden Datensatz
                    for i, source in enumerate(valid_sources):
                        source['anonymized_data']['temp_join_key'] = range(len(source['anonymized_data']))
                    
                    join_key = 'temp_join_key'
                
                # Führe die Daten zusammen
                combined_data = base_data.copy()
                
                for source in valid_sources[1:]:
                    # Finde neue Spalten, die noch nicht existieren
                    new_columns = [col for col in source['anonymized_data'].columns if col not in combined_data.columns]
                    
                    if new_columns:
                        # Verknüpfe über den Schlüssel mit linkem Join (behält alle Zeilen von combined_data)
                        combined_data = pd.merge(
                            combined_data,
                            source['anonymized_data'][new_columns + [join_key]],
                            on=join_key,
                            how='left'
                        )
                
                # Entferne den temporären Schlüssel, falls erstellt
                if join_key == 'temp_join_key' and join_key in combined_data.columns:
                    combined_data = combined_data.drop(columns=[join_key])
                
                combined_metadata['combination_description'] = f"Vertical enrichment using key '{join_key}', adding {len(combined_data.columns) - len(base_data.columns)} new features"
                combined_metadata['record_count'] = len(combined_data)
                combined_metadata['field_count'] = len(combined_data.columns)
                
                # Wertschätzung: Wert der Basis plus zusätzlicher Wert für neue Spalten
                base_value = valid_sources[0]['metadata'].get('estimated_value', 0)
                additional_value = sum(s['metadata'].get('estimated_value', 0) * 0.6 for s in valid_sources[1:])  # 60% des Werts der zusätzlichen Quellen
                combined_metadata['estimated_value'] = base_value + additional_value
                
            elif combination_type == 'correlate':
                # Korrelationsanalyse zwischen verschiedenen Datenquellen
                # Erstellt einen neuen Datensatz mit Korrelationsmaßen und abgeleiteten Features
                
                # Erstelle eine Liste für Korrelationsergebnisse
                correlation_results = []
                feature_pairs = []
                
                # Suche nach numerischen Spalten in jedem Datensatz
                for i, source1 in enumerate(valid_sources):
                    for j in range(i+1, len(valid_sources)):
                        source2 = valid_sources[j]
                        
                        # Extrahiere numerische Spalten
                        num_cols1 = source1['anonymized_data'].select_dtypes(include=['number']).columns
                        num_cols2 = source2['anonymized_data'].select_dtypes(include=['number']).columns
                        
                        # Prüfe, ob genügend numerische Spalten vorhanden sind
                        if len(num_cols1) == 0 or len(num_cols2) == 0:
                            continue
                        
                        # Wähle die Top-5-Spalten pro Quelle für die Analyse
                        select_cols1 = list(num_cols1)[:5]
                        select_cols2 = list(num_cols2)[:5]
                        
                        # Erstelle ein DataFrame mit Korrelationswerten
                        source_type1 = source1['metadata']['source_type']
                        source_type2 = source2['metadata']['source_type']
                        
                        for col1 in select_cols1:
                            for col2 in select_cols2:
                                # Erstelle 100 zufällige Datenpunkte für jede Spalte
                                # (für eine Stichprobenkorrelation ohne Verknüpfung der tatsächlichen Daten)
                                sample1 = source1['anonymized_data'][col1].sample(min(100, len(source1['anonymized_data']))).reset_index(drop=True)
                                sample2 = source2['anonymized_data'][col2].sample(min(100, len(source2['anonymized_data']))).reset_index(drop=True)
                                
                                # Berechne die Korrelation
                                try:
                                    corr = sample1.corr(sample2)
                                    
                                    # Speichere nur signifikante Korrelationen
                                    if abs(corr) > 0.3:
                                        corr_entry = {
                                            'source1': source_type1,
                                            'source2': source_type2,
                                            'feature1': col1,
                                            'feature2': col2,
                                            'correlation': float(corr),
                                            'abs_correlation': float(abs(corr))
                                        }
                                        correlation_results.append(corr_entry)
                                        feature_pairs.append((source1, col1, source2, col2))
                                except:
                                    # Ignoriere Fehler bei der Korrelationsberechnung
                                    pass
                
                # Erstelle ein neues DataFrame mit den Korrelationsergebnissen
                if correlation_results:
                    combined_data = pd.DataFrame(correlation_results)
                    
                    # Füge abgeleitete Features für die Top-Korrelationen hinzu
                    if feature_pairs:
                        # Sortiere nach absoluter Korrelationsstärke
                        sorted_pairs = sorted(list(zip(correlation_results, feature_pairs)), 
                                             key=lambda x: x[0]['abs_correlation'], 
                                             reverse=True)
                        
                        # Wähle die Top-10-Paare (oder weniger, falls nicht genügend vorhanden)
                        top_pairs = sorted_pairs[:min(10, len(sorted_pairs))]
                        
                        # Erstelle für jedes Top-Paar ein kombiniertes Feature
                        for i, (corr_info, (source1, col1, source2, col2)) in enumerate(top_pairs):
                            # Normalisiere die Werte
                            samples1 = source1['anonymized_data'][col1].sample(min(100, len(source1['anonymized_data']))).reset_index(drop=True)
                            samples2 = source2['anonymized_data'][col2].sample(min(100, len(source2['anonymized_data']))).reset_index(drop=True)
                            
                            # Z-Score-Normalisierung
                            normalized1 = (samples1 - samples1.mean()) / samples1.std()
                            normalized2 = (samples2 - samples2.mean()) / samples2.std()
                            
                            # Erstelle ein kombiniertes Feature basierend auf der Korrelationsrichtung
                            if corr_info['correlation'] >= 0:
                                combined_feature = normalized1 + normalized2
                            else:
                                combined_feature = normalized1 - normalized2
                            
                            # Füge das kombinierte Feature dem Ergebnis hinzu
                            feature_name = f"combined_{source1['metadata']['source_type']}_{col1}_{source2['metadata']['source_type']}_{col2}"
                            combined_data[feature_name] = combined_feature
                    
                    combined_metadata['combination_description'] = f"Correlation analysis between {len(valid_sources)} datasets, identifying {len(correlation_results)} significant cross-source relationships"
                    combined_metadata['record_count'] = len(combined_data)
                    combined_metadata['field_count'] = len(combined_data.columns)
                    
                    # Wertschätzung: Basierend auf der Anzahl und Stärke der gefundenen Korrelationen
                    avg_correlation = np.mean([r['abs_correlation'] for r in correlation_results]) if correlation_results else 0
                    correlation_value = len(correlation_results) * avg_correlation * 0.5  # Grundwert pro signifikante Korrelation
                    
                    # Füge Werte der Originalquellen hinzu
                    base_value = sum(s['metadata'].get('estimated_value', 0) * 0.3 for s in valid_sources)  # 30% der Originalwerte
                    combined_metadata['estimated_value'] = base_value + correlation_value
                else:
                    raise ValueError("Keine signifikanten Korrelationen zwischen den Datenquellen gefunden")
            
            else:
                raise ValueError(f"Nicht unterstützter Kombinationstyp: {combination_type}")
            
            # Erstelle einen neuen C2D-Asset für den kombinierten Datensatz
            c2d_asset = self.c2d_manager.create_data_asset(
                combined_data,
                asset_metadata=combined_metadata
            )
            
            return {
                'anonymized_data': combined_data,
                'metadata': combined_metadata,
                'c2d_asset': c2d_asset
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Kombination von Datenquellen: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'anonymized_data': None,
                'metadata': {
                    'combination_type': combination_type,
                    'source_count': len(sources),
                    'error': str(e)
                }
            }
    
    def estimate_data_value(self, data: pd.DataFrame, metadata: Dict = None) -> Dict:
        """
        Berechnet einen detaillierten Wert für einen Datensatz basierend auf mehreren Faktoren.
        
        Args:
            data: Der zu bewertende Datensatz
            metadata: Optionale Metadaten über den Datensatz
            
        Returns:
            Detaillierte Wertschätzung mit Begründungen
        """
        try:
            self.logger.info(f"Berechne detaillierten Datenwert für Datensatz mit {len(data)} Zeilen und {len(data.columns)} Spalten")
            
            if metadata is None:
                metadata = {}
            
            # Basisfaktoren für die Wertschätzung
            value_metrics = {
                'data_volume': {
                    'score': 0.0,
                    'weight': 0.15,
                    'explanation': ''
                },
                'data_quality': {
                    'score': 0.0,
                    'weight': 0.2,
                    'explanation': ''
                },
                'data_uniqueness': {
                    'score': 0._group_size']].index
            
            means = grouped.mean().loc[valid_groups]
            
            # Füge Rauschen hinzu
            noisy_means = self._add_differential_privacy_noise(means)
            
            return {'group_means': noisy_means.to_dict()}
        else:
            means = prepared_data.mean()
            
            # Füge Rauschen hinzu
            noisy_means = self._add_differential_privacy_noise(means)
            
            return {'means': noisy_means.to_dict()}
    
    def _sum(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Summe für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        sums = prepared_data.sum()
        
        # Füge Rauschen hinzu
        noisy_sums = self._add_differential_privacy_noise(sums)
        
        return {'sums': noisy_sums.to_dict()}
    
    def _min(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet das Minimum für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        minimums = prepared_data.min()
        
        # Hier kein Rauschen, da Minimum keine sensiblen Informationen preisgibt
        return {'minimums': minimums.to_dict()}
    
    def _max(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet das Maximum für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        maximums = prepared_data.max()
        
        # Hier kein Rauschen, da Maximum keine sensiblen Informationen preisgibt
        return {'maximums': maximums.to_dict()}
    
    def _std(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Standardabweichung für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        std_devs = prepared_data.std()
        
        # Füge Rauschen hinzu
        noisy_std_devs = self._add_differential_privacy_noise(std_devs)
        
        return {'standard_deviations': noisy_std_devs.to_dict()}
    
    def _correlation(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Korrelation zwischen numerischen Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        # Verwende nur numerische Spalten
        numeric_data = prepared_data.select_dtypes(include=['number'])
        
        if numeric_data.empty or numeric_data.shape[1] < 2:
            return {'error': 'Nicht genügend numerische Spalten für Korrelation'}
        
        corr_matrix = numeric_data.corr()
        
        # Füge Rauschen hinzu
        noisy_corr = self._add_differential_privacy_noise(corr_matrix)
        
        # Stelle sicher, dass Werte im gültigen Bereich liegen
        noisy_corr = noisy_corr.clip(-1, 1)
        
        return {'correlation_matrix': noisy_corr.to_dict()}
    
    def _histogram(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Erstellt ein Histogramm für eine Spalte"""
        column = params.get('column')
        bins = params.get('bins', 10)
        
        if column not in data.columns:
            return {'error': f'Spalte {column} nicht gefunden'}
        
        prepared_data = self._prepare_data_for_computation(data, [column])
        
        if pd.api.types.is_numeric_dtype(prepared_data[column]):
            # Numerisches Histogramm
            hist, bin_edges = np.histogram(prepared_data[column], bins=bins)
            
            # Füge Rauschen hinzu
            noisy_hist = self._add_differential_privacy_noise(hist).astype(int)
            
            # Stelle sicher, dass Werte nicht negativ sind
            noisy_hist = np.maximum(0, noisy_hist)
            
            return {
                'histogram': noisy_hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        else:
            # Kategorisches Histogramm
            value_counts = prepared_data[column].value_counts()
            
            # Prüfe auf zu kleine Gruppen
            value_counts = value_counts[value_counts >= self.privacy_config['min_group_size']]
            
            if value_counts.empty:
                return {'error': 'Keine Kategorien mit ausreichender Größe'}
            
            # Füge Rauschen hinzu
            noisy_counts = self._add_differential_privacy_noise(value_counts).astype(int)
            
            # Stelle sicher, dass Werte nicht negativ sind
            noisy_counts = noisy_counts.clip(0)
            
            return {
                'categories': value_counts.index.tolist(),
                'counts': noisy_counts.tolist()
            }
    
    def _custom_model_inference(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Führt Inferenz mit einem benutzerdefinierten Modell durch, ohne die Daten preiszugeben
        
        Args:
            data: Eingabedaten für die Inferenz
            params: Parameter für die Inferenz, einschließlich des Modelltyps und der Modellparameter
            
        Returns:
            Ergebnisse der Inferenz
        """
        model_type = params.get('model_type')
        model_config = params.get('model_config', {})
        columns = params.get('columns')
        
        if not model_type:
            return {'error': 'Kein Modelltyp angegeben'}
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        try:
            results = None
            
            if model_type == 'anomaly_detection':
                # Anomalieerkennung mit Isolation Forest
                detector = AnomalyDetector(method='isolation_forest', contamination=model_config.get('contamination', 0.05))
                detector.fit(prepared_data)
                predictions = detector.predict(prepared_data)
                
                # Aggregiere Ergebnisse für den Datenschutz
                anomaly_count = np.sum(predictions == -1)
                total_count = len(predictions)
                
                # Füge Rauschen hinzu
                noisy_anomaly_count = self._add_differential_privacy_noise(anomaly_count).astype(int)
                
                results = {
                    'anomaly_percentage': round(100 * noisy_anomaly_count / total_count, 2),
                    'total_records': total_count
                }
            
            elif model_type == 'clustering':
                # K-Means-Clustering
                n_clusters = model_config.get('n_clusters', 3)
                
                # Verarbeite nur numerische Spalten
                numeric_data = prepared_data.select_dtypes(include=['number'])
                
                if numeric_data.empty:
                    return {'error': 'Keine numerischen Spalten für Clustering gefunden'}
                
                # Standardisiere Daten
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                # Cluster-Analyse
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Cluster-Statistiken (datenschutzfreundlich)
                cluster_sizes = {}
                for i in range(n_clusters):
                    size = np.sum(clusters == i)
                    
                    # Prüfe Mindestgruppengröße
                    if size >= self.privacy_config['min_group_size']:
                        # Füge Rauschen hinzu
                        noisy_size = self._add_differential_privacy_noise(size).astype(int)
                        cluster_sizes[f'cluster_{i}'] = max(0, noisy_size)
                
                # Cluster-Zentren (keine persönlichen Daten)
                cluster_centers = kmeans.cluster_centers_
                
                results = {
                    'cluster_sizes': cluster_sizes,
                    'cluster_centers': cluster_centers.tolist()
                }
            
            elif model_type == 'sentiment_analysis':
                # Sentimentanalyse für Textdaten
                text_column = model_config.get('text_column')
                
                if text_column not in prepared_data.columns:
                    return {'error': f'Textspalte {text_column} nicht gefunden'}
                
                # Einfache Sentimentanalyse mit NLTK
                analyzer = SemanticAnalyzer()
                sentiment_results = analyzer.analyze_sentiment(prepared_data[text_column].tolist())
                
                # Aggregierte Ergebnisse für den Datenschutz
                sentiment_counts = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
                
                for result in sentiment_results:
                    sentiment_counts[result['sentiment']] += 1
                
                # Füge Rauschen hinzu
                for sentiment in sentiment_counts:
                    noisy_count = self._add_differential_privacy_noise(sentiment_counts[sentiment]).astype(int)
                    sentiment_counts[sentiment] = max(0, noisy_count)
                
                results = {
                    'sentiment_distribution': sentiment_counts,
                    'total_records': len(sentiment_results)
                }
            
            else:
                return {'error': f'Nicht unterstützter Modelltyp: {model_type}'}
            
            return results
            
        except Exception as e:
            logger.error(f"Fehler bei der Modell-Inferenz: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Modell-Inferenz: {str(e)}'}
    
    def execute_operation(self, encrypted_data: bytes, operation: str, params: Dict) -> Dict:
        """
        Führt eine Operation auf verschlüsselten Daten aus, ohne sie zu entschlüsseln
        
        Args:
            encrypted_data: Verschlüsselte Daten
            operation: Name der auszuführenden Operation
            params: Parameter für die Operation
            
        Returns:
            Ergebnisse der Operation
        """
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
            logger.error(f"Fehler bei der Ausführung der Operation {operation}: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Ausführung: {str(e)}'}
    
    def create_data_asset(self, data: pd.DataFrame, asset_metadata: Dict = None) -> Dict:
        """
        Erstellt einen verschlüsselten Daten-Asset mit Metadaten für den Marktplatz
        
        Args:
            data: Zu verschlüsselnde Daten
            asset_metadata: Metadaten für den Asset
            
        Returns:
            Asset-Details mit ID und Metadaten
        """
        try:
            # Erstelle eine eindeutige ID für den Asset
            asset_id = uuid.uuid4().hex
            
            # Verschlüssele die Daten
            encrypted_data = self._encrypt_data(data)
            
            # Erstelle Metadaten mit datenschutzfreundlichen Statistiken
            stats = {}
            
            # Anzahl der Datensätze
            stats['record_count'] = len(data)
            
            # Spalteninformationen
            stats['columns'] = []
            for col in data.columns:
                col_info = {
                    'name': col,
                    'type': str(data[col].dtype)
                }
                
                # Füge datenschutzfreundliche Statistiken hinzu
                if pd.api.types.is_numeric_dtype(data[col]):
                    # Füge Rauschen hinzu, um Datenschutz zu gewährleisten
                    col_info['min'] = float(self._add_differential_privacy_noise(data[col].min()))
                    col_info['max'] = float(self._add_differential_privacy_noise(data[col].max()))
                    col_info['mean'] = float(self._add_differential_privacy_noise(data[col].mean()))
                    col_info['std'] = float(self._add_differential_privacy_noise(data[col].std()))
                else:
                    # Für kategorische Daten nur die Anzahl eindeutiger Werte
                    col_info['unique_values'] = int(self._add_differential_privacy_noise(data[col].nunique()))
                
                stats['columns'].append(col_info)
            
            # Zusammenführen mit benutzerdefinierten Metadaten
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
            logger.error(f"Fehler bei der Erstellung des Daten-Assets: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Asset-Erstellung: {str(e)}'}
    
    def generate_access_token(self, asset_id: str, allowed_operations: List[str], 
                            expiration_time: int = 3600) -> Dict:
        """
        Generiert ein temporäres Zugriffstoken für einen Daten-Asset
        
        Args:
            asset_id: ID des Daten-Assets
            allowed_operations: Liste der erlaubten Operationen
            expiration_time: Ablaufzeit in Sekunden
            
        Returns:
            Token-Details mit Token und Ablaufzeit
        """
        # Prüfe, ob alle angeforderten Operationen unterstützt werden
        unsupported_ops = [op for op in allowed_operations if op not in self.allowed_operations]
        if unsupported_ops:
            return {'error': f'Nicht unterstützte Operationen: {", ".join(unsupported_ops)}'}
        
        # Erstelle Token-Daten
        token_data = {
            'asset_id': asset_id,
            'allowed_operations': allowed_operations,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=expiration_time)).isoformat(),
            'token_id': uuid.uuid4().hex
        }
        
        # Verschlüssele Token-Daten
        token_json = json.dumps(token_data).encode()
        encrypted_token = self.cipher_suite.encrypt(token_json)
        
        return {
            'token': encrypted_token.decode(),
            'token_id': token_data['token_id'],
            'expires_at': token_data['expires_at']
        }
    
    def validate_access_token(self, token: str, operation: str) -> bool:
        """
        Validiert ein Zugriffstoken für eine bestimmte Operation
        
        Args:
            token: Zugriffstoken
            operation: Angeforderte Operation
            
        Returns:
            True, wenn das Token gültig ist und die Operation erlaubt ist
        """
        try:
            # Entschlüssele das Token
            decrypted_token = self.cipher_suite.decrypt(token.encode())
            token_data = json.loads(decrypted_token.decode())
            
            # Prüfe, ob das Token abgelaufen ist
            expiry_time = datetime.fromisoformat(token_data['expires_at'])
            if datetime.now() > expiry_time:
                logger.warning(f"Zugriffstoken abgelaufen: {token_data['token_id']}")
                return False
            
            # Prüfe, ob die Operation erlaubt ist
            if operation not in token_data['allowed_operations']:
                logger.warning(f"Operation {operation} nicht erlaubt für Token {token_data['token_id']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Token-Validierung: {str(e)}")
            return False

###########################################
# 6. Integration und OceanData-Kernmodule
###########################################

class OceanDataAI:
    """
    Hauptklasse, die alle KI-Module für OceanData integriert und koordiniert.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialisiert den OceanData-KI-Manager.
        
        Args:
            config: Konfigurationsparameter für die KI-Module
        """
        self.config = config or {}
        self.logger = logging.getLogger("OceanData.AI")
        
        # Instanziiere die einzelnen Module
        self.anomaly_detector = AnomalyDetector(
            method=self.config.get('anomaly_detection_method', 'isolation_forest'),
            contamination=self.config.get('anomaly_contamination', 0.05)
        )
        
        self.semantic_analyzer = SemanticAnalyzer(
            model_type=self.config.get('semantic_model', 'bert'),
            model_name=self.config.get('semantic_model_name', 'bert-base-uncased')
        )
        
        self.predictive_modeler = PredictiveModeler(
            model_type=self.config.get('predictive_model', 'lstm'),
            forecast_horizon=self.config.get('forecast_horizon', 7)
        )
        
        self.data_synthesizer = DataSynthesizer(
            categorical_threshold=self.config.get('categorical_threshold', 10),
            noise_dim=self.config.get('gan_noise_dim', 100)
        )
        
        self.c2d_manager = ComputeToDataManager()
        
        # Aktiviere GPU-Beschleunigung, falls verfügbar
        self._setup_hardware_acceleration()
    
    def _setup_hardware_acceleration(self):
        """Konfiguriert Hardware-Beschleunigung für die KI-Module"""
        try:
            # Prüfe, ob GPUs für TensorFlow verfügbar sind
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.logger.info(f"GPU-Beschleunigung aktiviert mit {len(gpus)} GPUs")
                
                # Konfiguriere TensorFlow für bessere GPU-Nutzung
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                self.logger.info("Keine GPUs gefunden, verwende CPU-Computing")
        
        except Exception as e:
            self.logger.warning(f"Fehler bei der GPU-Konfiguration: {str(e)}")
            self.logger.info("Verwende Standard-CPU-Computing")
    
    def analyze_data_source(self, data: pd.DataFrame, source_type: str) -> Dict:
        """
        Führt eine umfassende Analyse einer Datenquelle durch.
        
        Args:
            data: Die zu analysierenden Daten
            source_type: Typ der Datenquelle (browser, calendar, iot, etc.)
            
        Returns:
            Umfassende Analyseergebnisse
        """
        results = {
            'source_type': source_type,
            'timestamp': datetime.now().isoformat(),
            'record_count': len(data),
            'column_count': len(data.columns),
            'analyses': {}
        }
        
        try:
            # 1. Ausreißererkennung
            self.logger.info(f"Führe Ausreißererkennung für {source_type}-Daten durch")
            self.anomaly_detector.fit(data)
            anomaly_predictions = self.anomaly_detector.predict(data)
            anomaly_insights = self.anomaly_detector.get_anomaly_insights(
                data, anomaly_predictions, top_n_features=5
            )
            
            results['analyses']['anomalies'] = {
                'count': int(np.sum(anomaly_predictions == -1) if self.anomaly_detector.method == 'isolation_forest' else np.sum(anomaly_predictions)),
                'percentage': float(np.mean(anomaly_predictions == -1) * 100 if self.anomaly_detector.method == 'isolation_forest' else np.mean(anomaly_predictions) * 100),
                'insights': anomaly_insights[:5]  # Begrenze auf die Top-5-Insights
            }
            
            # 2. Textanalyse für relevante Textspalten
            text_columns = [col for col in data.columns 
                           if data[col].dtype == 'object' 
                           and data[col].str.len().mean() > 20]
            
            if text_columns:
                self.logger.info(f"Führe Textanalyse für {len(text_columns)} Spalten durch")
                text_analyses = {}
                
                for col in text_columns[:3]:  # Begrenze auf die ersten 3 Textspalten
                    # Wähle Texte aus, die nicht leer sind
                    texts = data[col].dropna().astype(str).tolist()
                    if not texts:
                        continue
                        
                    # Stichprobe für Performance
                    sample_size = min(100, len(texts))
                    text_sample = random.sample(texts, sample_size)
                    
                    # Sentimentanalyse
                    sentiments = self.semantic_analyzer.analyze_sentiment(text_sample)
                    
                    # Themenextraktion
                    topics = self.semantic_analyzer.extract_topics(text_sample, num_topics=3)
                    
                    text_analyses[col] = {
                        'sentiment': {
                            'positive': len([s for s in sentiments if s['sentiment'] == 'positive']) / len(sentiments),
                            'neutral': len([s for s in sentiments if s['sentiment'] == 'neutral']) / len(sentiments),
                            'negative': len([s for s in sentiments if s['sentiment'] == 'negative']) / len(sentiments)
                        },
                        'topics': topics
                    }
                
                results['analyses']['text'] = text_analyses
            
            # 3. Zeitreihenanalyse für Zeitstempel-basierte Daten
            datetime_columns = [col for col in data.columns 
                              if pd.api.types.is_datetime64_any_dtype(data[col])]
            
            if datetime_columns and len(data) > 10:
                self.logger.info("Führe Zeitreihenanalyse durch")
                
                # Wähle die erste Zeitstempelspalte für die Analyse
                time_col = datetime_columns[0]
                
                # Sortiere Daten nach Zeit
                sorted_data = data.sort_values(by=time_col)
                
                # Wähle numerische Spalten für die Vorhersage
                numeric_cols = sorted_data.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    # Wähle die ersten 3 numerischen Spalten für die Demonstration
                    target_cols = numeric_cols[:3]
                    
                    # Erstelle Zeitreihen-Features
                    sorted_data['hour'] = sorted_data[time_col].dt.hour
                    sorted_data['day_of_week'] = sorted_data[time_col].dt.dayofweek
                    sorted_data['day_of_month'] = sorted_data[time_col].dt.day
                    sorted_data['month'] = sorted_data[time_col].dt.month
                    
                    # Bilde Feature- und Zielspalten
                    X = sorted_data[['hour', 'day_of_week', 'day_of_month', 'month']]
                    y = sorted_data[target_cols]
                    
                    if len(X) > 20:  # Mindestanzahl für sinnvolle Zeitreihenanalyse
                        # Trainiere ein einfaches Modell
                        self.predictive_modeler.fit(X, y, lookback=5, epochs=20, verbose=0)
                        
                        # Forecasting für die nächsten 7 Tage
                        forecast = self.predictive_modeler.forecast(X, steps=7)
                        
                        results['analyses']['time_series'] = {
                            'forecast_horizon': 7,
                            'forecasted_features': target_cols,
                            'forecasts': {
                                col: forecast[:, i].tolist() 
                                for i, col in enumerate(target_cols)
                            }
                        }
            
            # 4. Datensyntheseanalyse (optional, ressourcenintensiv)
            if self.config.get('enable_data_synthesis', False) and len(data) > 50:
                self.logger.info("Führe Datensynthese-Analyse durch")
                
                # Wähle eine Stichprobe und maximal 10 Spalten für die Demonstration
                sample_size = min(1000, len(data))
                data_sample = data.sample(sample_size)
                
                if len(data.columns) > 10:
                    # Wähle die wichtigsten Spalten aus
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()[:5]
                    cat_cols = data.select_dtypes(exclude=['number']).columns.tolist()[:5]
                    selected_cols = numeric_cols + cat_cols
                    data_sample = data_sample[selected_cols]
                
                # Trainiere das GAN mit einer kleinen Anzahl von Epochen
                self.data_synthesizer.fit(data_sample, epochs=100, verbose=0)
                
                # Generiere einige synthetische Beispiele
                synthetic_examples = self.data_synthesizer.generate(10)
                
                # Bewerte die Qualität
                quality_metrics = self.data_synthesizer.evaluate_quality(100)
                
                results['analyses']['data_synthesis'] = {
                    'quality_score': quality_metrics.get('overall_quality_score', 0),
                    'sample_examples': synthetic_examples.head(3).to_dict()
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenanalyse: {str(e)}")
            self.logger.error(traceback.format_exc())
            results['error'] = str(e)
            return results
    
    def prepare_data_for_monetization(self, data: pd.DataFrame, source_type: str, 
                                    privacy_level: str = 'high') -> Dict:
        """
        Bereitet Daten für die Monetarisierung vor, einschließlich Anonymisierung und Metadatengenerierung.
        
        Args:
            data: Die zu monetarisierenden Daten
            source_type: Typ der Datenquelle
            privacy_level: Datenschutzniveau ('low', 'medium', 'high')
            
        Returns:
            Aufbereitete Daten mit Metadaten für die Monetarisierung
        """
        try:
            self.logger.info(f"Bereite {source_type}-Daten für Monetarisierung vor (Privacy: {privacy_level})")
            
            # 1. Datenschutzmaßnahmen basierend auf dem gewählten Niveau
            anonymized_data = data.copy()
            
            # Identifiziere sensible Spalten
            id_columns = [col for col in data.columns 
                         if 'id' in col.lower() or 'user' in col.lower() or 'email' in col.lower()]
            location_columns = [col for col in data.columns 
                              if 'location' in col.lower() or 'address' in col.lower() or 'gps' in col.lower() or 'ip' in col.lower()]
            datetime_columns = [col for col in data.columns 
                              if pd.api.types.is_datetime64_any_dtype(data[col])]
            
            # Anonymisierung basierend auf Datenschutzniveau
            if privacy_level == 'high':
                # Vollständige Anonymisierung und Entfernung sensibler Daten
                anonymized_data = anonymized_data.drop(columns=id_columns, errors='ignore')
                
                # Anonymisiere Standortdaten
                for col in location_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: f"location_{hashlib.md5(str(x).encode()).hexdigest()[:8]}" if pd.notna(x) else x
                        )
                
                # Vergröbere Zeitstempel (nur auf Tagesebene)
                for col in datetime_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = pd.to_datetime(anonymized_data[col]).dt.floor('D')
                
            elif privacy_level == 'medium':
                # Moderate Anonymisierung
                for col in id_columns:
                    if col in anonymized_data.columns:
                        salt = datetime.now().strftime("%Y%m%d")
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest() if pd.notna(x) else x
                        )
                
                # Vergröbere Standortdaten
                for col in location_columns:
                    if col in anonymized_data.columns and anonymized_data[col].dtype == 'object':
                """
OceanData - Fortgeschrittene KI-Module für Datenanalyse und -monetarisierung

Diese Module erweitern den Kernalgorithmus von OceanData um fortschrittliche KI-Funktionen:
1. Anomalieerkennung
2. Semantische Datenanalyse 
3. Prädiktive Modellierung
4. Datensynthese und Erweiterung
5. Multimodale Analyse
6. Federated Learning und Compute-to-Data
7. Kontinuierliches Lernen und Modellanpassung
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertTokenizer, BertModel, TFBertModel
from transformers import GPT2Tokenizer, GPT2Model, TFGPT2Model
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import json
from typing import Dict, List, Union, Any, Tuple, Optional
import logging
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import h5py
import uuid
import traceback
import warnings

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Ignoriere TensorFlow und PyTorch Warnungen
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger("OceanData.AI")

###########################################
# 1. Anomalieerkennung & Outlier Detection
###########################################

class AnomalyDetector:
    """Klasse für die Erkennung von Anomalien in verschiedenen Datentypen"""
    
    def __init__(self, method: str = 'autoencoder', contamination: float = 0.05):
        """
        Initialisiert einen Anomaliedetektor.
        
        Args:
            method: Methode für die Anomalieerkennung ('autoencoder', 'isolation_forest', 'dbscan')
            contamination: Erwarteter Anteil an Anomalien in den Daten (für Isolation Forest)
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dims = None
        self.threshold = None
        
    def _build_autoencoder(self, input_dim):
        """Erstellt ein Autoencoder-Modell für die Anomalieerkennung"""
        encoding_dim = max(1, input_dim // 2)
        hidden_dim = max(1, encoding_dim // 2)
        
        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(inputs)
        encoded = layers.Dense(hidden_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Encoder model for extracting features
        encoder = keras.Model(inputs, encoded)
        
        return autoencoder, encoder
    
    def fit(self, X: pd.DataFrame, categorical_cols: List[str] = None):
        """
        Trainiert den Anomaliedetektor mit Daten.
        
        Args:
            X: Eingabedaten (DataFrame oder numpy array)
            categorical_cols: Liste der kategorischen Spalten, die One-Hot-kodiert werden müssen
        """
        try:
            # Vorbereiten der Daten
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten durch One-Hot-Encoding
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Konvertieren zu float und Behandlung von NaN-Werten
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            
            if X_prep.shape[1] == 0:
                raise ValueError("Nach der Vorverarbeitung sind keine numerischen Merkmale übrig")
                
            X_scaled = self.scaler.fit_transform(X_prep)
            self.feature_dims = X_scaled.shape[1]
            
            if self.method == 'autoencoder':
                # Autoencoder-Ansatz für Anomalieerkennung
                self.model, self.encoder = self._build_autoencoder(self.feature_dims)
                
                # Trainiere den Autoencoder
                self.model.fit(
                    X_scaled, X_scaled,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    verbose=0,
                    validation_split=0.1
                )
                
                # Berechne Rekonstruktionsfehler für alle Trainingsdaten
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Setze Threshold basierend auf den Rekonstruktionsfehlern
                self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
                
            elif self.method == 'isolation_forest':
                # Isolation Forest für Outlier-Erkennung
                self.model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
                self.model.fit(X_scaled)
                
            elif self.method == 'dbscan':
                # DBSCAN für Clustering-basierte Outlier-Erkennung
                self.model = DBSCAN(
                    eps=0.5,  # Maximum distance between samples
                    min_samples=5,  # Minimum samples in a cluster
                    n_jobs=-1
                )
                self.model.fit(X_scaled)
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des Anomaliedetektors: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X: pd.DataFrame, categorical_cols: List[str] = None) -> np.ndarray:
        """
        Identifiziert Anomalien in den Daten.
        
        Args:
            X: Zu prüfende Daten
            categorical_cols: Liste der kategorischen Spalten
            
        Returns:
            Array mit Anomalie-Scores für jede Zeile (-1 für Anomalien, 1 für normale Daten bei Isolation Forest;
            True für Anomalien, False für normale Daten bei Autoencoder)
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Daten
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Konvertieren und Behandlung von NaN-Werten
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            
            # Anpassen der Features, falls nötig
            missing_cols = set(range(self.feature_dims)) - set(range(X_prep.shape[1]))
            if missing_cols:
                for col in missing_cols:
                    X_prep[f'missing_{col}'] = 0
            
            X_prep = X_prep.iloc[:, :self.feature_dims]
            X_scaled = self.scaler.transform(X_prep)
            
            if self.method == 'autoencoder':
                # Berechne Rekonstruktionsfehler
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Anomalie, wenn Fehler über dem Threshold
                return reconstruction_errors > self.threshold
                
            elif self.method == 'isolation_forest':
                # -1 für Anomalien, 1 für normale Daten
                return self.model.predict(X_scaled)
                
            elif self.method == 'dbscan':
                # Ordne neue Punkte den nächsten Clustern zu
                labels = self.model.fit_predict(X_scaled)
                # -1 sind Noise-Punkte (Anomalien)
                return labels == -1
                
        except Exception as e:
            logger.error(f"Fehler bei der Anomalievorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_anomaly_insights(self, X: pd.DataFrame, predictions: np.ndarray, 
                           categorical_cols: List[str] = None, top_n_features: int = 3) -> List[Dict]:
        """
        Liefert Erklärungen für die erkannten Anomalien.
        
        Args:
            X: Originaldaten
            predictions: Ergebnisse der predict()-Methode
            categorical_cols: Liste der kategorischen Spalten
            top_n_features: Anzahl der wichtigsten Features, die zurückgegeben werden sollen
            
        Returns:
            Liste mit Erklärungen für jede Anomalie
        """
        insights = []
        
        try:
            anomaly_indices = np.where(predictions == -1)[0] if self.method == 'isolation_forest' else np.where(predictions)[0]
            
            if len(anomaly_indices) == 0:
                return []
            
            X_prep = X.copy()
            X_orig = X.copy()  # Original-Daten für Berichte
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            X_scaled = self.scaler.transform(X_prep)
            
            # Allgemeine Statistiken berechnen
            mean_values = np.mean(X_scaled, axis=0)
            std_values = np.std(X_scaled, axis=0)
            
            for idx in anomaly_indices:
                if idx >= len(X_scaled):
                    continue
                    
                sample = X_scaled[idx]
                orig_sample = X_orig.iloc[idx]
                
                if self.method == 'autoencoder':
                    # Rekonstruktionsfehler pro Feature berechnen
                    reconstruction = self.model.predict(sample.reshape(1, -1))[0]
                    feature_errors = np.square(sample - reconstruction)
                    
                    # Features mit den größten Fehlern identifizieren
                    top_features_idx = np.argsort(feature_errors)[-top_n_features:]
                    top_features = [(X_prep.columns[i], feature_errors[i], sample[i], reconstruction[i]) for i in top_features_idx]
                    
                    insight = {
                        "index": int(idx),
                        "anomaly_score": float(np.mean(feature_errors)),
                        "important_features": [
                            {
                                "feature": feat,
                                "error": float(err),
                                "actual_value": float(val),
                                "expected_value": float(rec)
                            } for feat, err, val, rec in top_features
                        ],
                        "original_data": orig_sample.to_dict()
                    }
                    
                else:  # Isolation Forest oder DBSCAN
                    # Z-Scores berechnen, um ungewöhnliche Werte zu identifizieren
                    z_scores = (sample - mean_values) / std_values
                    
                    # Features mit den höchsten Z-Scores (positiv oder negativ)
                    abs_z_scores = np.abs(z_scores)
                    top_features_idx = np.argsort(abs_z_scores)[-top_n_features:]
                    top_features = [(X_prep.columns[i], z_scores[i], sample[i], mean_values[i]) for i in top_features_idx]
                    
                    insight = {
                        "index": int(idx),
                        "anomaly_score": float(np.max(abs_z_scores)),
                        "important_features": [
                            {
                                "feature": feat,
                                "z_score": float(z),
                                "actual_value": float(val),
                                "average_value": float(avg)
                            } for feat, z, val, avg in top_features
                        ],
                        "original_data": orig_sample.to_dict()
                    }
                
                insights.append(insight)
                
            return insights
            
        except Exception as e:
            logger.error(f"Fehler bei der Generierung von Anomalie-Insights: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def visualize_anomalies(self, X: pd.DataFrame, predictions: np.ndarray, 
                            categorical_cols: List[str] = None, 
                            save_path: str = None) -> plt.Figure:
        """
        Visualisiert erkannte Anomalien in den Daten.
        
        Args:
            X: Daten
            predictions: Anomalieerkennungsergebnisse
            categorical_cols: Kategorische Spalten
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        try:
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            X_scaled = self.scaler.transform(X_prep)
            
            # PCA für Dimensionsreduktion
            if X_scaled.shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
            else:
                X_pca = X_scaled
            
            # Anomalien bestimmen
            if self.method == 'isolation_forest':
                anomalies = (predictions == -1)
            else:
                anomalies = predictions
            
            # Visualisierung erstellen
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Normale Datenpunkte plotten
            ax.scatter(X_pca[~anomalies, 0], X_pca[~anomalies, 1], c='blue', label='Normal', alpha=0.5)
            
            # Anomalien plotten
            if np.any(anomalies):
                ax.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], c='red', label='Anomalien', alpha=0.7)
            
            # Grafik anpassen
            ax.set_title(f'Anomalieerkennung mit {self.method.capitalize()}')
            ax.set_xlabel('Hauptkomponente 1')
            ax.set_ylabel('Hauptkomponente 2')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung von Anomalien: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden")
            
        model_data = {
            "method": self.method,
            "contamination": self.contamination,
            "feature_dims": self.feature_dims,
            "threshold": self.threshold,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere das Model-Objekt
        if self.method in ['isolation_forest', 'dbscan']:
            joblib.dump(self.model, f"{path}_model.joblib")
        elif self.method == 'autoencoder':
            self.model.save(f"{path}_autoencoder")
            self.encoder.save(f"{path}_encoder")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        detector = cls(
            method=model_data['method'],
            contamination=model_data['contamination']
        )
        
        detector.feature_dims = model_data['feature_dims']
        detector.threshold = model_data['threshold']
        detector.is_fitted = model_data['is_fitted']
        
        # Lade das Model-Objekt
        if detector.method in ['isolation_forest', 'dbscan']:
            detector.model = joblib.load(f"{path}_model.joblib")
        elif detector.method == 'autoencoder':
            detector.model = keras.models.load_model(f"{path}_autoencoder")
            detector.encoder = keras.models.load_model(f"{path}_encoder")
        
        # Lade den Scaler
        detector.scaler = joblib.load(f"{path}_scaler.joblib")
        
        return detector


###########################################
# 2. Semantische Datenanalyse
###########################################

class SemanticAnalyzer:
    """Klasse für semantische Analyse von Text und anderen Daten mit Deep Learning"""
    
    def __init__(self, model_type: str = 'bert', model_name: str = 'bert-base-uncased'):
        """
        Initialisiert den semantischen Analysator.
        
        Args:
            model_type: Typ des zu verwendenden Modells ('bert', 'gpt2', 'custom')
            model_name: Name oder Pfad des vortrainierten Modells
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.max_length = 512  # Standard für BERT
        self.embeddings_cache = {}  # Cache für Texteinbettungen
        
        # Modell und Tokenizer laden
        self._load_model()
    
    def _load_model(self):
        """Lädt das Modell und den Tokenizer basierend auf dem ausgewählten Typ"""
        try:
            if self.model_type == 'bert':
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                if tf.test.is_gpu_available():
                    self.model = TFBertModel.from_pretrained(self.model_name)
                else:
                    self.model = BertModel.from_pretrained(self.model_name)
                self.max_length = 512
            
            elif self.model_type == 'gpt2':
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 hat kein Padding-Token
                if tf.test.is_gpu_available():
                    self.model = TFGPT2Model.from_pretrained(self.model_name)
                else:
                    self.model = GPT2Model.from_pretrained(self.model_name)
                self.max_length = 1024
            
            else:
                raise ValueError(f"Nicht unterstützter Modelltyp: {self.model_type}")
        
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells {self.model_name}: {str(e)}")
            # Fallback zu kleineren Modellen bei Speicher- oder Download-Problemen
            if self.model_type == 'bert':
                logger.info("Verwende ein kleineres BERT-Modell als Fallback")
                self.model_name = 'distilbert-base-uncased'
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertModel.from_pretrained(self.model_name)
            elif self.model_type == 'gpt2':
                logger.info("Verwende ein kleineres GPT-2-Modell als Fallback")
                self.model_name = 'distilgpt2'
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = GPT2Model.from_pretrained(self.model_name)
    
    def get_embeddings(self, texts: Union[str, List[str]], 
                      batch_size: int = 8, use_cache: bool = True) -> np.ndarray:
        """
        Erzeugt Einbettungen (Embeddings) für Texte.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            batch_size: Größe der Batches für die Verarbeitung
            use_cache: Ob bereits berechnete Einbettungen wiederverwendet werden sollen
            
        Returns:
            Array mit Einbettungen (shape: [n_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Initialisiere ein Array für die Ergebnisse
        all_embeddings = []
        texts_to_process = []
        texts_indices = []
        
        # Prüfe Cache für jede Anfrage
        for i, text in enumerate(texts):
            if use_cache and text in self.embeddings_cache:
                all_embeddings.append(self.embeddings_cache[text])
            else:
                texts_to_process.append(text)
                texts_indices.append(i)
        
        if texts_to_process:
            # Verarbeite Texte in Batches
            for i in range(0, len(texts_to_process), batch_size):
                batch_texts = texts_to_process[i:i+batch_size]
                batch_indices = texts_indices[i:i+batch_size]
                
                # Tokenisierung
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt" if isinstance(self.model, nn.Module) else "tf",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Modelloutput berechnen
                with torch.no_grad() if isinstance(self.model, nn.Module) else tf.device('/CPU:0'):
                    outputs = self.model(**inputs)
                
                # Embeddings aus dem letzten Hidden State extrahieren
                if self.model_type == 'bert':
                    # Verwende [CLS]-Token-Ausgabe als Satzrepräsentation (erstes Token)
                    if isinstance(self.model, nn.Module):
                        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                    else:
                        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                elif self.model_type == 'gpt2':
                    # Verwende den Durchschnitt aller Token-Repräsentationen
                    if isinstance(self.model, nn.Module):
                        embeddings = torch.mean(outputs.last_hidden_state, dim=1).numpy()
                    else:
                        embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
                
                # Füge die Embeddings an der richtigen Position ein
                for j, (idx, text, embedding) in enumerate(zip(batch_indices, batch_texts, embeddings)):
                    # Zum Cache hinzufügen
                    if use_cache:
                        self.embeddings_cache[text] = embedding
                    
                    # Aktualisiere Ergebnisarray an der richtigen Position
                    if idx >= len(all_embeddings):
                        all_embeddings.extend([None] * (idx - len(all_embeddings) + 1))
                    all_embeddings[idx] = embedding
        
        # Konvertiere zu NumPy-Array
        return np.vstack(all_embeddings)
    
    def analyze_sentiment(self, texts: Union[str, List[str]]) -> List[Dict]:
        """
        Führt eine Stimmungsanalyse für Texte durch.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            
        Returns:
            Liste mit Sentiment-Analysen für jeden Text (positive, negative, neutral)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Verwende NLTK für grundlegende Sentimentanalyse
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            
            analyzer = SentimentIntensityAnalyzer()
            results = []
            
            for text in texts:
                scores = analyzer.polarity_scores(text)
                
                # Bestimme die dominante Stimmung
                if scores['compound'] >= 0.05:
                    sentiment = 'positive'
                elif scores['compound'] <= -0.05:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': sentiment,
                    'scores': {
                        'positive': scores['pos'],
                        'negative': scores['neg'],
                        'neutral': scores['neu'],
                        'compound': scores['compound']
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Fehler bei der Sentimentanalyse: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback zu einem einfacheren Ansatz
            return [{'text': t[:100] + '...' if len(t) > 100 else t, 
                    'sentiment': 'unknown', 
                    'scores': {'positive': 0, 'negative': 0, 'neutral': 0, 'compound': 0}} 
                    for t in texts]
    
    def extract_topics(self, texts: Union[str, List[str]], num_topics: int = 5, 
                       words_per_topic: int = 5) -> List[Dict]:
        """
        Extrahiert Themen (Topics) aus Texten.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            num_topics: Anzahl der zu extrahierenden Themen
            words_per_topic: Anzahl der Wörter pro Thema
            
        Returns:
            Liste mit Themen und zugehörigen Top-Wörtern
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Tokenisiere und bereinige die Texte
            try:
                nltk.data.find('stopwords')
                nltk.data.find('punkt')
            except LookupError:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
            
            stop_words = set(stopwords.words('english'))
            
            # Texte vorverarbeiten
            processed_texts = []
            for text in texts:
                # Tokenisieren und Stopwords entfernen
                tokens = [w.lower() for w in word_tokenize(text) 
                         if w.isalpha() and w.lower() not in stop_words]
                processed_texts.append(' '.join(tokens))
            
            # Verwende Transformers für Themenmodellierung
            embeddings = self.get_embeddings(processed_texts)
            
            # Verwende K-Means-Clustering auf Embeddings
            kmeans = KMeans(n_clusters=min(num_topics, len(processed_texts)), random_state=42)
            kmeans.fit(embeddings)
            
            # Finde repräsentative Wörter für jedes Cluster
            topics = []
            
            # Alle Wörter aus allen Texten zusammenfassen
            all_words = []
            for text in processed_texts:
                all_words.extend(text.split())
            
            # Eindeutige Wörter
            unique_words = list(set(all_words))
            
            # Für jedes Wort ein Embedding berechnen
            if len(unique_words) > 0:
                word_embeddings = self.get_embeddings(unique_words)
                
                # Für jedes Cluster die nächsten Wörter bestimmen, die dem Clusterzentrum am nächsten sind
                for cluster_idx in range(kmeans.n_clusters):
                    center = kmeans.cluster_centers_[cluster_idx]
                    
                    # Berechne Distanzen zwischen Zentrum und Wort-Embeddings
                    distances = np.linalg.norm(word_embeddings - center, axis=1)
                    
                    # Finde die nächsten Wörter
                    closest_indices = np.argsort(distances)[:words_per_topic]
                    top_words = [unique_words[i] for i in closest_indices]
                    
                    # Beispieltexte für dieses Cluster finden
                    cluster_texts = [texts[i][:100] + "..." 
                                    for i, label in enumerate(kmeans.labels_) 
                                    if label == cluster_idx][:3]  # Maximal 3 Beispiele
                    
                    topic = {
                        "id": cluster_idx,
                        "words": top_words,
                        "examples": cluster_texts
                    }
                    topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Fehler bei der Themenextraktion: {str(e)}")
            logger.error(traceback.format_exc())
            return [{"id": 0, "words": ["error", "processing", "topics"], "examples": []}]
    
    def find_similar_texts(self, query: str, corpus: List[str], top_n: int = 5) -> List[Dict]:
        """
        Findet ähnliche Texte zu einer Anfrage in einem Korpus.
        
        Args:
            query: Anfrage-Text
            corpus: Liste von Texten, in denen gesucht werden soll
            top_n: Anzahl der zurückzugebenden ähnlichsten Texte
            
        Returns:
            Liste der ähnlichsten Texte mit Ähnlichkeitswerten
        """
        try:
            # Einbettungen für Anfrage und Korpus erzeugen
            query_embedding = self.get_embeddings(query).reshape(1, -1)
            corpus_embeddings = self.get_embeddings(corpus)
            
            # Kosinus-Ähnlichkeiten berechnen
            similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
            
            # Top-N ähnlichste Texte finden
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
            
        except Exception as e:
            logger.error(f"Fehler beim Finden ähnlicher Texte: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def generate_text_summary(self, text: str, max_length: int = 150) -> str:
        """
        Erzeugt eine Zusammenfassung eines längeren Textes.
        
        Args:
            text: Text, der zusammengefasst werden soll
            max_length: Maximale Länge der Zusammenfassung in Zeichen
            
        Returns:
            Zusammenfassung des Textes
        """
        try:
            # Vorverarbeitung des Textes
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) <= 1:
                return text[:max_length] + "..." if len(text) > max_length else text
            
            # Einbettungen für alle Sätze erzeugen
            sentence_embeddings = self.get_embeddings(sentences)
            
            # Durchschnittliche Einbettung berechnen (repräsentiert den Gesamttext)
            mean_embedding = np.mean(sentence_embeddings, axis=0).reshape(1, -1)
            
            # Ähnlichkeit jedes Satzes zum Durchschnitt berechnen
            similarities = np.dot(sentence_embeddings, mean_embedding.T).flatten()
            
            # Sätze nach Ähnlichkeit sortieren
            ranked_sentences = [(sentences[i], float(similarities[i])) for i in range(len(sentences))]
            ranked_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Top-Sätze auswählen, bis max_length erreicht ist
            summary = ""
            for sentence, _ in ranked_sentences:
                if len(summary) + len(sentence) <= max_length:
                    summary += sentence + " "
                else:
                    break
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Fehler bei der Textzusammenfassung: {str(e)}")
            logger.error(traceback.format_exc())
            return text[:max_length] + "..." if len(text) > max_length else text

###########################################
# 3. Prädiktive Modellierung
###########################################

class PredictiveModeler:
    """
    Klasse für die Entwicklung von prädiktiven Modellen, die verschiedene Datentypen
    verarbeiten und Vorhersagen treffen können.
    """
    
    def __init__(self, model_type: str = 'lstm', forecast_horizon: int = 7):
        """
        Initialisiert den Prädiktiven Modellierer.
        
        Args:
            model_type: Typ des zu verwendenden Modells ('lstm', 'transformer', 'gru', 'arima')
            forecast_horizon: Anzahl der Zeitschritte für Vorhersagen in die Zukunft
        """
        self.model_type = model_type.lower()
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.lookback = 10  # Standardwert für die Anzahl der zurückliegenden Zeitschritte
        self.feature_dims = None
        self.target_dims = None
        self.target_scaler = None
        self.history = None
        
    def _build_lstm_model(self, input_shape, output_dim):
        """Erstellt ein LSTM-Modell für Zeitreihenvorhersage"""
        model = keras.Sequential()
        model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(32))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(output_dim))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _build_transformer_model(self, input_shape, output_dim):
        """Erstellt ein Transformer-Modell für Zeitreihenvorhersage"""
        # Einfaches Transformer-Modell für Zeitreihen
        inputs = keras.Input(shape=input_shape)
        
        # Positional encoding layer
        class PositionalEncoding(layers.Layer):
            def __init__(self, position, d_model):
                super(PositionalEncoding, self).__init__()
                self.pos_encoding = self.positional_encoding(position, d_model)
                
            def get_angles(self, position, i, d_model):
                angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
                return position * angles
            
            def positional_encoding(self, position, d_model):
                angle_rads = self.get_angles(
                    position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                    i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                    d_model=d_model
                )
                
                # Apply sine to even indices
                sines = tf.math.sin(angle_rads[:, 0::2])
                # Apply cosine to odd indices
                cosines = tf.math.cos(angle_rads[:, 1::2])
                
                pos_encoding = tf.concat([sines, cosines], axis=-1)
                pos_encoding = pos_encoding[tf.newaxis, ...]
                
                return tf.cast(pos_encoding, tf.float32)
            
            def call(self, inputs):
                return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        
        x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
        
        # Multi-head attention layer
        x = layers.MultiHeadAttention(
            key_dim=input_shape[1], num_heads=4, dropout=0.1
        )(x, x, x, attention_mask=None, training=True)
        
        # Feed-forward network
        x = layers.Dropout(0.1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=1, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv1D(filters=input_shape[1], kernel_size=1)(x)
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Output layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(output_dim)(x)
        
        model = keras.Model(inputs, x)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def _build_gru_model(self, input_shape, output_dim):
        """Erstellt ein GRU-Modell für Zeitreihenvorhersage"""
        model = keras.Sequential()
        model.add(layers.GRU(64, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.GRU(32))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(output_dim))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _create_sequences(self, data, target=None, lookback=None):
        """
        Erstellt Sequenzen für die Zeitreihenmodellierung
        
        Args:
            data: Eingabedaten (numpy array)
            target: Zielvariablen (optional, numpy array)
            lookback: Anzahl der zurückliegenden Zeitschritte (optional)
            
        Returns:
            X: Sequenzen für die Eingabe
            y: Zielwerte (wenn target bereitgestellt wird)
        """
        if lookback is None:
            lookback = self.lookback
        
        X = []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
        X = np.array(X)
        
        if target is not None:
            y = target[lookback:]
            return X, y
        
        return X
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, 
            lookback: int = 10, epochs: int = 50, 
            validation_split: float = 0.2, batch_size: int = 32,
            verbose: int = 1):
        """
        Trainiert das prädiktive Modell mit den gegebenen Daten.
        
        Args:
            X: Eingabedaten (DataFrame)
            y: Zielvariablen (DataFrame, optional für Zeitreihen)
            lookback: Anzahl der zurückliegenden Zeitschritte für Zeitreihenmodelle
            epochs: Anzahl der Trainings-Epochen
            validation_split: Anteil der Daten für die Validierung
            batch_size: Batch-Größe für das Training
            verbose: Ausgabedetailstufe (0, 1, oder 2)
            
        Returns:
            self: Trainiertes Modell
        """
        try:
            self.lookback = lookback
            
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            # Skalierung der Eingabedaten
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_values)
            
            # Vorbereiten der Zielvariablen
            if y is not None:
                if isinstance(y, pd.DataFrame):
                    y_values = y.values
                else:
                    y_values = y
                    
                self.target_scaler = StandardScaler()
                y_scaled = self.target_scaler.fit_transform(y_values)
                self.target_dims = y_scaled.shape[1]
            else:
                # Wenn keine Zielvariablen bereitgestellt werden, nehmen wir an, dass X selbst eine Zeitreihe ist
                y_scaled = X_scaled
                self.target_dims = X_scaled.shape[1]
            
            self.feature_dims = X_scaled.shape[1]
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, lookback)
            
            # Modell basierend auf dem ausgewählten Typ erstellen
            input_shape = (lookback, self.feature_dims)
            output_dim = self.target_dims * self.forecast_horizon
            
            if self.model_type == 'lstm':
                self.model = self._build_lstm_model(input_shape, output_dim)
            elif self.model_type == 'transformer':
                self.model = self._build_transformer_model(input_shape, output_dim)
            elif self.model_type == 'gru':
                self.model = self._build_gru_model(input_shape, output_dim)
            else:
                raise ValueError(f"Nicht unterstützter Modelltyp: {self.model_type}")
            
            # Reshape y_sequences für das Forecast-Horizon
            y_prepared = y_sequences.reshape(y_sequences.shape[0], -1)
            
            # Modell trainieren
            self.history = self.model.fit(
                X_sequences, y_prepared,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des prädiktiven Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X: pd.DataFrame, return_sequences: bool = False) -> np.ndarray:
        """
        Macht Vorhersagen mit dem trainierten Modell.
        
        Args:
            X: Eingabedaten (DataFrame)
            return_sequences: Ob die Vorhersagesequenz zurückgegeben werden soll
            
        Returns:
            Vorhersagen für die Eingabedaten
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_values)
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences = self._create_sequences(X_scaled, lookback=self.lookback)
            
            # Vorhersagen machen
            predictions_scaled = self.model.predict(X_sequences)
            
            # Reshape für die Ausgabe
            predictions_scaled = predictions_scaled.reshape(
                predictions_scaled.shape[0], 
                self.forecast_horizon, 
                self.target_dims
            )
            
            # Rücktransformation
            predictions = np.zeros_like(predictions_scaled)
            for i in range(self.forecast_horizon):
                step_predictions = predictions_scaled[:, i, :]
                # Rücktransformation nur für jeden Zeitschritt
                predictions[:, i, :] = self.target_scaler.inverse_transform(step_predictions)
            
            if return_sequences:
                return predictions
            else:
                # Nur den ersten Vorhersageschritt zurückgeben
                return predictions[:, 0, :]
            
        except Exception as e:
            logger.error(f"Fehler bei der Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def forecast(self, X: pd.DataFrame, steps: int = None) -> np.ndarray:
        """
        Erstellt eine Vorhersage für mehrere Zeitschritte in die Zukunft.
        
        Args:
            X: Letzte bekannte Datenpunkte (mindestens lookback viele)
            steps: Anzahl der vorherzusagenden Schritte (Standard: forecast_horizon)
            
        Returns:
            Vorhersagesequenz für die nächsten 'steps' Zeitschritte
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        if steps is None:
            steps = self.forecast_horizon
            
        try:
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
            
            if len(X_values) < self.lookback:
                raise ValueError(f"Eingabedaten müssen mindestens {self.lookback} Zeitschritte enthalten")
            
            # Verwende nur die letzten 'lookback' Zeitschritte
            X_recent = X_values[-self.lookback:]
            
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_recent)
            X_sequence = X_scaled.reshape(1, self.lookback, self.feature_dims)
            
            # Erstelle die multi-step-Vorhersage
            forecast_values = []
            
            current_sequence = X_sequence.copy()
            
            for _ in range(steps):
                # Mache eine Vorhersage für den nächsten Schritt
                next_step_scaled = self.model.predict(current_sequence)[0]
                next_step_scaled = next_step_scaled.reshape(1, self.target_dims)
                
                # Rücktransformation
                next_step = self.target_scaler.inverse_transform(next_step_scaled)
                forecast_values.append(next_step[0])
                
                # Aktualisiere die Eingabesequenz für den nächsten Schritt
                # Entferne den ersten Zeitschritt und füge den neu vorhergesagten hinzu
                new_sequence = np.zeros_like(current_sequence)
                new_sequence[0, :-1, :] = current_sequence[0, 1:, :]
                new_sequence[0, -1, :] = next_step_scaled
                current_sequence = new_sequence
            
            return np.array(forecast_values)
            
        except Exception as e:
            logger.error(f"Fehler bei der Mehrschritt-Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluiert das Modell mit Testdaten.
        
        Args:
            X_test: Test-Eingabedaten
            y_test: Test-Zielvariablen
            
        Returns:
            Dictionary mit Bewertungsmetriken
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Testdaten
            if isinstance(X_test, pd.DataFrame):
                X_values = X_test.values
            else:
                X_values = X_test
                
            if isinstance(y_test, pd.DataFrame):
                y_values = y_test.values
            else:
                y_values = y_test
            
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_values)
            y_scaled = self.target_scaler.transform(y_values)
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, self.lookback)
            
            # Reshape y_sequences für das Forecast-Horizon
            y_prepared = y_sequences.reshape(y_sequences.shape[0], -1)
            
            # Modell evaluieren
            evaluation = self.model.evaluate(X_sequences, y_prepared, verbose=0)
            
            # Vorhersagen machen für detailliertere Metriken
            predictions = self.predict(X_test)
            
            # Tatsächliche Werte (ohne die ersten lookback Zeitschritte)
            actuals = y_values[self.lookback:]
            
            # Berechne RMSE, MAE, MAPE
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Entsprechende Anzahl an Vorhersagen auswählen
            predictions = predictions[:len(actuals)]
            
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # MAPE (Mean Absolute Percentage Error)
            # Vermeide Division durch Null
            mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
            
            return {
                'loss': evaluation[0],
                'mae': evaluation[1],
                'rmse': rmse,
                'mean_absolute_error': mae,
                'mean_absolute_percentage_error': mape
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Modellbewertung: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def plot_forecast(self, X: pd.DataFrame, y_true: pd.DataFrame = None, 
                     steps: int = None, feature_idx: int = 0,
                     save_path: str = None) -> plt.Figure:
        """
        Visualisiert die Vorhersage des Modells.
        
        Args:
            X: Eingabedaten
            y_true: Tatsächliche zukünftige Werte (optional)
            steps: Anzahl der vorherzusagenden Schritte
            feature_idx: Index der darzustellenden Feature-Dimension
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Standard-Schritte
            if steps is None:
                steps = self.forecast_horizon
            
            # Vorhersage erstellen
            forecast_values = self.forecast(X, steps)
            
            # Historische Daten (letzte lookback Zeitschritte)
            if isinstance(X, pd.DataFrame):
                historical_values = X.values[-self.lookback:, feature_idx]
            else:
                historical_values = X[-self.lookback:, feature_idx]
            
            # Zeitachse erstellen
            time_hist = np.arange(-self.lookback, 0)
            time_future = np.arange(0, steps)
            
            # Visualisierung erstellen
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historische Daten plotten
            ax.plot(time_hist, historical_values, 'b-', label='Historische Daten')
            
            # Vorhersage plotten
            ax.plot(time_future, forecast_values[:, feature_idx], 'r-', label='Vorhersage')
            
            # Tatsächliche zukünftige Werte plotten, falls vorhanden
            if y_true is not None:
                if isinstance(y_true, pd.DataFrame):
                    true_future = y_true.values[:steps, feature_idx]
                else:
                    true_future = y_true[:steps, feature_idx]
                
                ax.plot(time_future[:len(true_future)], true_future, 'g-', label='Tatsächliche Werte')
            
            # Grafik anpassen
            ax.set_title(f'Zeitreihenvorhersage mit {self.model_type.upper()}')
            ax.set_xlabel('Zeitschritte')
            ax.set_ylabel(f'Feature {feature_idx}')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Trennlinie zwischen historischen und Vorhersagedaten
            ax.axvline(x=0, color='k', linestyle='--')
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung der Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        model_data = {
            "model_type": self.model_type,
            "forecast_horizon": self.forecast_horizon,
            "lookback": self.lookback,
            "feature_dims": self.feature_dims,
            "target_dims": self.target_dims,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere das Model-Objekt
        self.model.save(f"{path}_model")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        joblib.dump(self.target_scaler, f"{path}_target_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        predictor = cls(
            model_type=model_data['model_type'],
            forecast_horizon=model_data['forecast_horizon']
        )
        
        predictor.lookback = model_data['lookback']
        predictor.feature_dims = model_data['feature_dims']
        predictor.target_dims = model_data['target_dims']
        predictor.is_fitted = model_data['is_fitted']
        
        # Lade das Model-Objekt
        predictor.model = keras.models.load_model(f"{path}_model")
        
        # Lade die Scaler
        predictor.scaler = joblib.load(f"{path}_scaler.joblib")
        predictor.target_scaler = joblib.load(f"{path}_target_scaler.joblib")
        
        return predictor

###########################################
# 4. Datensynthese und GAN-basierte Modelle
###########################################

class DataSynthesizer:
    """
    Klasse zur Generierung synthetischer Daten basierend auf realen Beispielen.
    Verwendet GAN (Generative Adversarial Network) für realistische Datensynthese.
    """
    
    def __init__(self, categorical_threshold: int = 10, noise_dim: int = 100):
        """
        Initialisiert den Datensynthetisierer.
        
        Args:
            categorical_threshold: Anzahl eindeutiger Werte, ab der eine Spalte als kategorisch gilt
            noise_dim: Dimension des Rauschvektors für den Generator
        """
        self.categorical_threshold = categorical_threshold
        self.noise_dim = noise_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        
        self.column_types = {}  # Speichert, ob eine Spalte kategorisch oder kontinuierlich ist
        self.categorical_mappings = {}  # Speichert Mappings für kategorische Spalten
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # Für kontinuierliche Variablen
        
        self.is_fitted = False
        self.feature_dims = None
        self.training_data = None
    
    def _identify_column_types(self, data: pd.DataFrame):
        """Identifiziert, ob Spalten kategorisch oder kontinuierlich sind"""
        self.column_types = {}
        
        for col in data.columns:
            n_unique = data[col].nunique()
            
            # Wenn die Anzahl eindeutiger Werte kleiner als der Schwellenwert ist oder
            # der Datentyp ist nicht numerisch, behandle die Spalte als kategorisch
            if n_unique < self.categorical_threshold or not pd.api.types.is_numeric_dtype(data[col]):
                self.column_types[col] = 'categorical'
                
                # Erstelle Mapping von Kategorien zu Zahlen
                categories = data[col].unique()
                self.categorical_mappings[col] = {
                    cat: i for i, cat in enumerate(categories)
                }
                # Umgekehrtes Mapping für die Rücktransformation
                self.categorical_mappings[f"{col}_reverse"] = {
                    i: cat for i, cat in enumerate(categories)
                }
            else:
                self.column_types[col] = 'continuous'
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Vorverarbeitung der Daten für das GAN"""
        processed_data = pd.DataFrame()
        
        for col in data.columns:
            if self.column_types[col] == 'categorical':
                # One-Hot-Encoding für kategorische Spalten
                mapped_col = data[col].map(self.categorical_mappings[col])
                one_hot = pd.get_dummies(mapped_col, prefix=col)
                processed_data = pd.concat([processed_data, one_hot], axis=1)
            else:
                # Skalierung für kontinuierliche Spalten
                processed_data[col] = data[col]
        
        # Skaliere alle Spalten auf [-1, 1]
        return self.scaler.fit_transform(processed_data)
    
    def _postprocess_data(self, generated_data: np.ndarray) -> pd.DataFrame:
        """Nachverarbeitung der generierten Daten zurück in das ursprüngliche Format"""
        # Rücktransformation der Skalierung
        rescaled_data = self.scaler.inverse_transform(generated_data)
        
        # Erstelle einen DataFrame mit den ursprünglichen Spalten
        result = pd.DataFrame()
        
        col_idx = 0
        for col, col_type in self.column_types.items():
            if col_type == 'categorical':
                # Anzahl der eindeutigen Werte für diese kategorische Spalte
                n_categories = len(self.categorical_mappings[col])
                
                # Extrahiere die One-Hot-kodierten Werte
                cat_values = rescaled_data[:, col_idx:col_idx+n_categories]
                
                # Konvertiere von One-Hot zurück zu kategorischen Werten
                # Nehme die Kategorie mit dem höchsten Wert
                cat_indices = np.argmax(cat_values, axis=1)
                
                # Mappe zurück zu den ursprünglichen Kategorien
                result[col] = [self.categorical_mappings[f"{col}_reverse"][idx] for idx in cat_indices]
                
                col_idx += n_categories
            else:
                # Kontinuierliche Spalte einfach übernehmen
                result[col] = rescaled_data[:, col_idx]
                col_idx += 1
        
        return result
    
    def _build_generator(self, output_dim):
        """Erstellt den Generator für das GAN"""
        model = keras.Sequential([
            layers.Dense(256, input_dim=self.noise_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(output_dim, activation='tanh')  # tanh für Output im Bereich [-1, 1]
        ])
        return model
    
    def _build_discriminator(self, input_dim):
        """Erstellt den Diskriminator für das GAN"""
        model = keras.Sequential([
            layers.Dense(512, input_dim=input_dim, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy')
        return model
    
    def _build_gan(self, generator, discriminator):
        """Kombiniert Generator und Diskriminator zum GAN"""
        discriminator.trainable = False  # Diskriminator beim GAN-Training nicht aktualisieren
        
        model = keras.Sequential([
            generator,
            discriminator
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy')
        return model
    
    def fit(self, data: pd.DataFrame, epochs: int = 2000, batch_size: int = 32, 
           sample_interval: int = 100, verbose: int = 1):
        """
        Trainiert das GAN-Modell mit den gegebenen Daten.
        
        Args:
            data: Eingabedaten (DataFrame)
            epochs: Anzahl der Trainings-Epochen
            batch_size: Batch-Größe für das Training
            sample_interval: Intervall für Stichproben der generierten Daten
            verbose: Ausgabedetailstufe (0, 1, oder 2)
            
        Returns:
            self: Trainiertes Modell
        """
        try:
            # Identifiziere Spaltentypen
            self._identify_column_types(data)
            
            # Vorverarbeitung der Daten
            processed_data = self._preprocess_data(data)
            self.feature_dims = processed_data.shape[1]
            
            # Speichere trainierte Daten für spätere Validierung
            self.training_data = data.copy()
            
            # Baue das GAN-Modell
            self.generator = self._build_generator(self.feature_dims)
            self.discriminator = self._build_discriminator(self.feature_dims)
            self.gan = self._build_gan(self.generator, self.discriminator)
            
            # Label für echte und gefälschte Daten
            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            
            # Trainingsschleife
            for epoch in range(epochs):
                # ---------------------
                #  Trainiere Diskriminator
                # ---------------------
                
                # Wähle eine zufällige Batch aus echten Daten
                idx = np.random.randint(0, processed_data.shape[0], batch_size)
                real_data = processed_data[idx]
                
                # Generiere eine Batch aus gefälschten Daten
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                fake_data = self.generator.predict(noise)
                
                # Trainiere den Diskriminator
                d_loss_real = self.discriminator.train_on_batch(real_data, real)
                d_loss_fake = self.discriminator.train_on_batch(fake_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ---------------------
                #  Trainiere Generator
                # ---------------------
                
                # Generiere neue Batch aus Rauschen
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                
                # Trainiere den Generator
                g_loss = self.gan.train_on_batch(noise, real)
                
                # Ausgabe für Fortschrittsüberwachung
                if verbose > 0 and epoch % sample_interval == 0:
                    print(f"Epoch {epoch}/{epochs} [D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}]")
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des GAN-Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def generate(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generiert synthetische Daten.
        
        Args:
            n_samples: Anzahl der zu generierenden Datensätze
            
        Returns:
            DataFrame mit synthetischen Daten im Format der Trainingsdaten
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere Rauschen als Input für den Generator
            noise = np.random.normal(0, 1, (n_samples, self.noise_dim))
            
            # Generiere Daten
            generated_data = self.generator.predict(noise)
            
            # Nachverarbeitung der Daten
            synthetic_data = self._postprocess_data(generated_data)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Fehler bei der Datengenerierung: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate_quality(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Bewertet die Qualität der generierten Daten durch Vergleich mit den Trainingsdaten.
        
        Args:
            n_samples: Anzahl der zu generierenden und bewertenden Datensätze
            
        Returns:
            Dictionary mit Qualitätsmetriken
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere synthetische Daten
            synthetic_data = self.generate(n_samples)
            
            # Statistischer Vergleich zwischen echten und synthetischen Daten
            metrics = {}
            
            # Vergleiche Mittelwerte und Standardabweichungen für kontinuierliche Spalten
            for col, col_type in self.column_types.items():
                if col_type == 'continuous':
                    # Berechne Mittelwert und Standardabweichung für echte Daten
                    real_mean = self.training_data[col].mean()
                    real_std = self.training_data[col].std()
                    
                    # Berechne dieselben Statistiken für synthetische Daten
                    synth_mean = synthetic_data[col].mean()
                    synth_std = synthetic_data[col].std()
                    
                    # Berechne die relative Differenz
                    mean_diff = abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10)
                    std_diff = abs(real_std - synth_std) / (abs(real_std) + 1e-10)
                    
                    metrics[f"{col}_mean_diff"] = float(mean_diff)
                    metrics[f"{col}_std_diff"] = float(std_diff)
                else:
                    # Vergleiche die Verteilung kategorischer Werte
                    real_dist = self.training_data[col].value_counts(normalize=True)
                    synth_dist = synthetic_data[col].value_counts(normalize=True)
                    
                    # Berechne die Jensen-Shannon-Divergenz
                    # (symmetrische Version der KL-Divergenz)
                    js_divergence = 0.0
                    
                    # Stelle sicher, dass beide Verteilungen dieselben Kategorien haben
                    all_categories = set(real_dist.index) | set(synth_dist.index)
                    
                    for cat in all_categories:
                        p = real_dist.get(cat, 0)
                        q = synth_dist.get(cat, 0)
                        
                        # Vermeide Logarithmus von 0
                        if p > 0 and q > 0:
                            m = 0.5 * (p + q)
                            js_divergence += 0.5 * (p * np.log(p / m) + q * np.log(q / m))
                    
                    metrics[f"{col}_js_divergence"] = float(js_divergence)
            
            # Gesamtqualitätsmetrik
            # Durchschnitt der normalisierten Abweichungen (niedriger ist besser)
            continuous_diffs = [v for k, v in metrics.items() if k.endswith('_diff')]
            categorical_diffs = [v for k, v in metrics.items() if k.endswith('_js_divergence')]
            
            if continuous_diffs:
                metrics['continuous_avg_diff'] = float(np.mean(continuous_diffs))
            if categorical_diffs:
                metrics['categorical_avg_diff'] = float(np.mean(categorical_diffs))
            
            # Gesamtbewertung (0 bis 1, höher ist besser)
            overall_score = 1.0
            if continuous_diffs:
                overall_score -= 0.5 * min(1.0, np.mean(continuous_diffs))
            if categorical_diffs:
                overall_score -= 0.5 * min(1.0, np.mean(categorical_diffs))
            
            metrics['overall_quality_score'] = float(overall_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Fehler bei der Qualitätsbewertung: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def plot_comparison(self, n_samples: int = 1000, 
                       features: List[str] = None,
                       save_path: str = None) -> plt.Figure:
        """
        Visualisiert einen Vergleich zwischen echten und synthetischen Daten.
        
        Args:
            n_samples: Anzahl der zu generierenden Datensätze
            features: Liste der darzustellenden Features (Standard: alle)
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere synthetische Daten
            synthetic_data = self.generate(n_samples)
            
            # Wähle die darzustellenden Features aus
            if features is None:
                # Wähle bis zu 6 Features für die Visualisierung
                features = list(self.column_types.keys())[:min(6, len(self.column_types))]
            
            # Bestimme die Anzahl der Zeilen und Spalten für das Subplot-Raster
            n_features = len(features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            # Erstelle die Figur
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
            if n_rows * n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, feature in enumerate(features):
                ax = axes[i]
                
                if self.column_types[feature] == 'continuous':
                    # Histogramm für kontinuierliche Variablen
                    sns.histplot(self.training_data[feature], kde=True, ax=ax, color='blue', alpha=0.5, label='Echte Daten')
                    sns.histplot(synthetic_data[feature], kde=True, ax=ax, color='red', alpha=0.5, label='Synthetische Daten')
                else:
                    # Balkendiagramm für kategorische Variablen
                    real_counts = self.training_data[feature].value_counts(normalize=True)
                    synth_counts = synthetic_data[feature].value_counts(normalize=True)
                    
                    # Kombiniere beide, um alle Kategorien zu erfassen
                    all_cats = sorted(set(real_counts.index) | set(synth_counts.index))
                    
                    # Erstelle ein DataFrame für Seaborn
                    plot_data = []
                    for cat in all_cats:
                        plot_data.append({'Category': cat, 'Frequency': real_counts.get(cat, 0), 'Type': 'Real'})
                        plot_data.append({'Category': cat, 'Frequency': synth_counts.get(cat, 0), 'Type': 'Synthetic'})
                    
                    plot_df = pd.DataFrame(plot_data)
                    
                    # Balkendiagramm
                    sns.barplot(x='Category', y='Frequency', hue='Type', data=plot_df, ax=ax)
                
                ax.set_title(f'Verteilung von {feature}')
                ax.legend()
                
                # Achsen anpassen
                if self.column_types[feature] == 'categorical':
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Verstecke ungenutzte Subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung des Datenvergleichs: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        model_data = {
            "categorical_threshold": self.categorical_threshold,
            "noise_dim": self.noise_dim,
            "feature_dims": self.feature_dims,
            "column_types": self.column_types,
            "categorical_mappings": self.categorical_mappings,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere die Modelle
        self.generator.save(f"{path}_generator")
        self.discriminator.save(f"{path}_discriminator")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten und Mappings
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        synthesizer = cls(
            categorical_threshold=model_data['categorical_threshold'],
            noise_dim=model_data['noise_dim']
        )
        
        synthesizer.feature_dims = model_data['feature_dims']
        synthesizer.column_types = model_data['column_types']
        synthesizer.categorical_mappings = model_data['categorical_mappings']
        synthesizer.is_fitted = model_data['is_fitted']
        
        # Lade die Modelle
        synthesizer.generator = keras.models.load_model(f"{path}_generator")
        synthesizer.discriminator = keras.models.load_model(f"{path}_discriminator")
        
        # Lade den Scaler
        synthesizer.scaler = joblib.load(f"{path}_scaler.joblib")
        
        # Rekonstruiere das GAN
        synthesizer.gan = synthesizer._build_gan(synthesizer.generator, synthesizer.discriminator)
        
        return synthesizer

###########################################
# 5. Federated Learning und Compute-to-Data
###########################################

class ComputeToDataManager:
    """
    Klasse zur Verwaltung von Compute-to-Data-Operationen für den Datenschutz.
    Ermöglicht die Ausführung von Berechnungen auf sensiblen Daten ohne direkten Zugriff.
    """
    
    def __init__(self, encryption_key: bytes = None):
        """
        Initialisiert den Compute-to-Data-Manager.
        
        Args:
            encryption_key: Schlüssel für die Verschlüsselung (optional)
        """
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.allowed_operations = {
            'aggregate': self._aggregate,
            'count': self._count,
            'mean': self._mean,
            'sum': self._sum,
            'min': self._min,
            'max': self._max,
            'std': self._std,
            'correlation': self._correlation,
            'histogram': self._histogram,
            'custom_model': self._custom_model_inference
        }
        
        # Datenschutzkonfiguration
        self.privacy_config = {
            'min_group_size': 5,  # Minimale Gruppengröße für Aggregationen
            'noise_level': 0.01,  # Standardrauschen für Differentialprivatsphäre
            'outlier_removal': True  # Entfernung von Ausreißern für sensible Berechnungen
        }
    
    def set_privacy_config(self, min_group_size: int = None, noise_level: float = None, 
                          outlier_removal: bool = None):
        """Aktualisiert die Datenschutzkonfiguration"""
        if min_group_size is not None:
            self.privacy_config['min_group_size'] = min_group_size
        if noise_level is not None:
            self.privacy_config['noise_level'] = noise_level
        if outlier_removal is not None:
            self.privacy_config['outlier_removal'] = outlier_removal
    
    def _encrypt_data(self, data: pd.DataFrame) -> bytes:
        """Verschlüsselt Daten für die sichere Speicherung"""
        serialized = data.to_json().encode()
        return self.cipher_suite.encrypt(serialized)
    
    def _decrypt_data(self, encrypted_data: bytes) -> pd.DataFrame:
        """Entschlüsselt Daten für die Verarbeitung"""
        decrypted = self.cipher_suite.decrypt(encrypted_data)
        return pd.read_json(decrypted.decode())
    
    def _add_differential_privacy_noise(self, result, scale: float = None):
        """Fügt Rauschen für Differential Privacy hinzu"""
        scale = scale or self.privacy_config['noise_level']
        
        if isinstance(result, (int, float)):
            return result + np.random.normal(0, scale * abs(result))
        elif isinstance(result, np.ndarray):
            return result + np.random.normal(0, scale * np.abs(result).mean(), result.shape)
        elif isinstance(result, pd.Series):
            noisy_values = result.values + np.random.normal(0, scale * np.abs(result.values).mean(), result.shape)
            return pd.Series(noisy_values, index=result.index)
        elif isinstance(result, pd.DataFrame):
            noisy_values = result.values + np.random.normal(0, scale * np.abs(result.values).mean(), result.shape)
            return pd.DataFrame(noisy_values, index=result.index, columns=result.columns)
        else:
            return result
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Entfernt Ausreißer aus den Daten"""
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            return data
        
        # Z-Score-basierte Ausreißererkennung
        z_scores = ((numeric_data - numeric_data.mean()) / numeric_data.std()).abs()
        mask = (z_scores < 3).all(axis=1)
        
        return data[mask]
    
    def _validate_group_size(self, data: pd.DataFrame, group_column: str = None, 
                            group_value: Any = None) -> bool:
        """
        Prüft, ob eine Gruppe die Mindestgröße für Datenschutz erfüllt
        
        Args:
            data: Daten
            group_column: Spalte für die Gruppierung
            group_value: Wert für die Filterung
            
        Returns:
            True, wenn die Gruppe groß genug ist
        """
        min_size = self.privacy_config['min_group_size']
        
        if group_column is not None and group_value is not None:
            group_data = data[data[group_column] == group_value]
            return len(group_data) >= min_size
        
        return len(data) >= min_size
    
    def _prepare_data_for_computation(self, data: pd.DataFrame, columns: List[str] = None,
                                    group_column: str = None, group_value: Any = None) -> pd.DataFrame:
        """
        Bereitet Daten für die Berechnung vor, einschließlich Filterung und Datenschutzprüfungen
        
        Args:
            data: Eingabedaten
            columns: Zu verwendende Spalten
            group_column: Spalte für die Gruppierung
            group_value: Wert für die Filterung
            
        Returns:
            Vorbereitete Daten
        """
        # Filtere Daten nach Gruppe, falls angegeben
        if group_column is not None and group_value is not None:
            prepared_data = data[data[group_column] == group_value]
        else:
            prepared_data = data.copy()
        
        # Prüfe Mindestgruppengröße
        if not self._validate_group_size(prepared_data):
            raise ValueError(f"Gruppe zu klein für Datenschutz (min: {self.privacy_config['min_group_size']})")
        
        # Entferne Ausreißer, falls konfiguriert
        if self.privacy_config['outlier_removal']:
            prepared_data = self._remove_outliers(prepared_data)
        
        # Wähle Spalten aus, falls angegeben
        if columns is not None:
            prepared_data = prepared_data[columns]
        
        return prepared_data
    
    # Implementierte Operationen für C2D
    
    def _aggregate(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Aggregiert Daten nach Spalten und Gruppen"""
        columns = params.get('columns')
        group_by = params.get('group_by')
        aggregations = params.get('aggregations', ['count', 'mean'])
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        if group_by:
            grouped = prepared_data.groupby(group_by)
            
            # Prüfe Mindestgröße für jede Gruppe
            group_sizes = grouped.size()
            valid_groups = group_sizes[group_sizes >= self.privacy_config['min_group_size']].index
            
            # Filtere ungültige Gruppen
            results = {}
            for agg in aggregations:
                if agg == 'count':
                    agg_result = grouped.size()
                else:
                    agg_func = getattr(grouped, agg, None)
                    if agg_func is None:
                        continue
                    agg_result = agg_func()
                
                # Behalte nur valide Gruppen
                agg_result = agg_result.loc[valid_groups]
                
                # Füge Rauschen hinzu
                results[agg] = self._add_differential_privacy_noise(agg_result)
        else:
            results = {}
            for agg in aggregations:
                if agg == 'count':
                    agg_result = len(prepared_data)
                else:
                    agg_func = getattr(prepared_data, agg, None)
                    if agg_func is None:
                        continue
                    agg_result = agg_func()
                
                # Füge Rauschen hinzu
                results[agg] = self._add_differential_privacy_noise(agg_result)
        
        # Konvertiere zu serialisierbarem Format
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
                serializable_results[k] = v.to_dict()
            else:
                serializable_results[k] = v
        
        return serializable_results
    
    def _count(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Zählt Datensätze nach Filterkriterien"""
        filter_column = params.get('filter_column')
        filter_value = params.get('filter_value')
        
        if filter_column and filter_value is not None:
            count = len(data[data[filter_column] == filter_value])
        else:
            count = len(data)
        
        # Füge Rauschen hinzu und runde auf ganze Zahlen
        noisy_count = round(self._add_differential_privacy_noise(count))
        
        # Stelle sicher, dass der Wert nicht negativ ist
        return {'count': max(0, noisy_count)}
    
    def _mean(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet den Mittelwert für numerische Spalten"""
        columns = params.get('columns')
        group_column = params.get('group_column')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        if group_column:
            grouped = prepared_data.groupby(group_column)
            
            # Prüfe Mindestgröße für jede Gruppe
            group_sizes = grouped.size()
            valid_groups = group_sizes[group_sizes >= self.privacy_config['min

          'data_uniqueness': {
                    'score': 0.0,
                    'weight': 0.2,
                    'explanation': ''
                },
                'data_applicability': {
                    'score': 0.0,
                    'weight': 0.25,
                    'explanation': ''
                },
                'time_relevance': {
                    'score': 0.0,
                    'weight': 0.1,
                    'explanation': ''
                },
                'enrichment_potential': {
                    'score': 0.0,
                    'weight': 0.1,
                    'explanation': ''
                }
            }
            
            # 1. Datenvolumen
            # Bewerte die Größe des Datensatzes (Zeilen und Spalten)
            rows_score = min(1.0, len(data) / 10000)  # Skaliert bis 10.000 Zeilen
            cols_score = min(1.0, len(data.columns) / 50)  # Skaliert bis 50 Spalten
            volume_score = (rows_score * 0.7) + (cols_score * 0.3)  # Gewichtung: Zeilen wichtiger als Spalten
            
            value_metrics['data_volume']['score'] = volume_score
            value_metrics['data_volume']['explanation'] = f"Dataset contains {len(data)} records and {len(data.columns)} fields. "
            
            if volume_score < 0.3:
                value_metrics['data_volume']['explanation'] += "This is a relatively small dataset, which limits its potential value."
            elif volume_score < 0.7:
                value_metrics['data_volume']['explanation'] += "This is a medium-sized dataset with solid commercial value potential."
            else:
                value_metrics['data_volume']['explanation'] += "This is a large dataset with significant commercial value potential."
            
            # 2. Datenqualität
            # Bewerte die Qualität anhand von fehlenden Werten, Varianz und Konsistenz
            
            # Berechne fehlende Werte
            missing_ratio = data.isna().mean().mean()
            completeness = 1 - missing_ratio
            
            # Berechne Varianz für numerische Spalten
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                normalized_variance = 0
                for col in numeric_cols:
                    # Normalisierte Varianz (0-1)
                    if data[col].std() > 0:
                        normalized_variance += min(1.0, data[col].std() / data[col].mean() if data[col].mean() != 0 else 0)
                
                avg_normalized_variance = normalized_variance / len(numeric_cols)
                variance_score = avg_normalized_variance
            else:
                variance_score = 0.5  # Mittlerer Wert, wenn keine numerischen Spalten vorhanden sind
            
            # Berechne Eindeutigkeit für kategorische Spalten
            categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
            if categorical_cols:
                uniqueness_ratio = 0
                for col in categorical_cols:
                    uniqueness_ratio += data[col].nunique() / len(data)
                
                avg_uniqueness_ratio = uniqueness_ratio / len(categorical_cols)
                uniqueness_score = min(1.0, avg_uniqueness_ratio * 3)  # Skaliert, um niedrigere Werte aufzuwerten
            else:
                uniqueness_score = 0.5  # Mittlerer Wert, wenn keine kategorischen Spalten vorhanden sind
            
            # Kombiniere die Qualitätsmetriken
            quality_score = (completeness * 0.5) + (variance_score * 0.25) + (uniqueness_score * 0.25)
            
            value_metrics['data_quality']['score'] = quality_score
            value_metrics['data_quality']['explanation'] = f"Data completeness: {completeness:.1%}. "
            
            if completeness < 0.8:
                value_metrics['data_quality']['explanation'] += "Missing data impacts overall quality. "
            
            value_metrics['data_quality']['explanation'] += f"Data variability: {variance_score:.1%}. "
            if variance_score < 0.3:
                value_metrics['data_quality']['explanation'] += "Low variability may indicate limited information content. "
            elif variance_score > 0.7:
                value_metrics['data_quality']['explanation'] += "High variability indicates rich information content. "
            
            # 3. Dateneinzigartigkeit
            # Bewerte die Einzigartigkeit basierend auf Spaltentypen und Datenbereichen
            
            # Analysiere Spaltentypen
            column_types = {
                'datetime': len(data.select_dtypes(include=['datetime']).columns),
                'numeric': len(data.select_dtypes(include=['number']).columns),
                'categorical': len(data.select_dtypes(include=['category', 'object']).columns),
                'boolean': len(data.select_dtypes(include=['bool']).columns),
                'text': sum(1 for col in data.select_dtypes(include=['object']).columns 
                          if data[col].astype(str).str.len().mean() > 20)
            }
            
            # Diversitätspunkte für Spaltentypen-Mix
            diversity_score = min(1.0, sum(1 for count in column_types.values() if count > 0) / 5)
            
            # Zusätzliche Punkte für seltene oder wertvolle Datentypen
            value_column_score = 0
            if column_types['datetime'] > 0:
                value_column_score += 0.2  # Zeitreihen sind wertvoll
            if column_types['text'] > 0:
                value_column_score += 0.3  # Textdaten sind wertvoll für NLP
            if column_types['boolean'] > 0 and column_types['numeric'] > 0:
                value_column_score += 0.1  # Gute Kombination für ML
            
            # Quelltyp-spezifischer Wert (wenn vorhanden)
            source_type = metadata.get('source_type', '')
            source_type_scores = {
                'health_data': 0.9,
                'health_insurance': 0.9,
                'smartwatch': 0.8,
                'social_media': 0.7,
                'browser': 0.6,
                'chat': 0.7,
                'streaming': 0.7,
                'calendar': 0.5,
                'iot_security': 0.6,
                'smart_home': 0.5
            }
            source_score = source_type_scores.get(source_type, 0.5)
            
            # Kombiniere Einzigartigkeit und Wert
            uniqueness_score = (diversity_score * 0.3) + (value_column_score * 0.3) + (source_score * 0.4)
            
            value_metrics['data_uniqueness']['score'] = uniqueness_score
            value_metrics['data_uniqueness']['explanation'] = f"Data contains {sum(column_types.values())} columns across {sum(1 for count in column_types.values() if count > 0)} different data types. "
            
            if source_type:
                value_metrics['data_uniqueness']['explanation'] += f"Source type '{source_type}' "
                if source_score > 0.7:
                    value_metrics['data_uniqueness']['explanation'] += "is particularly valuable in the data marketplace. "
                elif source_score > 0.5:
                    value_metrics['data_uniqueness']['explanation'] += "has good value in the data marketplace. "
                else:
                    value_metrics['data_uniqueness']['explanation'] += "has standard value in the data marketplace. "
            
            if column_types['text'] > 0:
                value_metrics['data_uniqueness']['explanation'] += f"Contains {column_types['text']} text columns which add significant value for NLP applications. "
            
            if column_types['datetime'] > 0:
                value_metrics['data_uniqueness']['explanation'] += f"Contains {column_types['datetime']} datetime columns enabling time-series analysis. "
            
            # 4. Anwendbarkeit der Daten
            # Bewerte potenzielle Anwendungsfälle basierend auf Datentyp und Marktanforderungen
            
            # Identifiziere potenzielle Anwendungsfälle basierend auf verfügbaren Daten
            application_scenarios = []
            
            # Prüfe auf typische Anwendungsfälle basierend auf Spaltenkombinationen
            has_location_data = any('location' in col.lower() or 'address' in col.lower() or 'geo' in col.lower() for col in data.columns)
            has_user_data = any('user' in col.lower() or 'customer' in col.lower() or 'id' in col.lower() for col in data.columns)
            has_time_data = column_types['datetime'] > 0
            has_numeric_features = column_types['numeric'] > 2  # Mindestens 3 numerische Spalten
            has_categorical_features = column_types['categorical'] > 0
            has_text_features = column_types['text'] > 0
            
            # Bewerte Anwendbarkeit für Machine Learning
            if has_numeric_features and (has_categorical_features or has_time_data):
                application_scenarios.append({
                    'name': 'Predictive Modeling',
                    'score': 0.8,
                    'description': 'Suitable for building predictive models using machine learning'
                })
            
            # Bewerte Anwendbarkeit für Zeitreihenanalyse
            if has_time_data and has_numeric_features:
                application_scenarios.append({
                    'name': 'Time Series Analysis',
                    'score': 0.9,
                    'description': 'Ideal for trend analysis and forecasting over time'
                })
            
            # Bewerte Anwendbarkeit für Personalisierung
            if has_user_data and (has_categorical_features or has_text_features):
                application_scenarios.append({
                    'name': 'Personalization & Recommendations',
                    'score': 0.85,
                    'description': 'Can be used for user preference modeling and recommendations'
                })
            
            # Bewerte Anwendbarkeit für geografische Analysen
            if has_location_data:
                application_scenarios.append({
                    'name': 'Geospatial Analysis',
                    'score': 0.75,
                    'description': 'Enables location-based insights and targeting'
                })
            
            # Bewerte Anwendbarkeit für NLP
            if has_text_features:
                application_scenarios.append({
                    'name': 'Natural Language Processing',
                    'score': 0.85,
                    'description': 'Text data can be used for sentiment analysis, topic modeling, etc.'
                })
            
            # Bewerte Anwendbarkeit für Clusteranalyse
            if has_numeric_features >= 3:
                application_scenarios.append({
                    'name': 'Clustering & Segmentation',
                    'score': 0.7,
                    'description': 'Can be used to identify natural groupings and segments'
                })
            
            # Berechne den Anwendbarkeitswert basierend auf den identifizierten Szenarien
            if application_scenarios:
                # Durchschnittliche Bewertung über alle Szenarien
                applicability_score = sum(scenario['score'] for scenario in application_scenarios) / len(application_scenarios)
                
                # Bonus für die Anzahl der Szenarien (mehr Szenarien = vielseitigere Daten)
                scenario_count_bonus = min(0.2, (len(application_scenarios) - 1) * 0.05)
                applicability_score += scenario_count_bonus
                
                # Begrenze auf Maximum von 1.0
                applicability_score = min(1.0, applicability_score)
            else:
                # Standard-Score, wenn keine spezifischen Szenarien identifiziert wurden
                applicability_score = 0.3
            
            value_metrics['data_applicability']['score'] = applicability_score
            value_metrics['data_applicability']['explanation'] = f"Identified {len(application_scenarios)} potential application scenarios. "
            
            if application_scenarios:
                top_scenarios = sorted(application_scenarios, key=lambda x: x['score'], reverse=True)[:3]
                value_metrics['data_applicability']['explanation'] += "Top applications: " + ", ".join(s['name'] for s in top_scenarios) + ". "
            
            if applicability_score > 0.7:
                value_metrics['data_applicability']['explanation'] += "Data has high versatility across multiple use cases. "
            elif applicability_score > 0.4:
                value_metrics['data_applicability']['explanation'] += "Data has moderate versatility for specific applications. "
            else:
                value_metrics['data_applicability']['explanation'] += "Data has limited application potential without additional enrichment. "
            
            # 5. Zeitliche Relevanz
            # Bewerte die Aktualität und zeitliche Spanne der Daten
            
            time_relevance_score = 0.5  # Mittlerer Standardwert
            
            # Prüfe auf zeitbezogene Metadaten
            time_range = metadata.get('time_range', {})
            if time_range:
                # Aktualität: Wie nah ist das neueste Datum an heute?
                end_date = time_range.get('end')
                if end_date:
                    try:
                        # Konvertiere ISO-String zu Datum
                        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        days_since_end = (datetime.now() - end_date).days
                        
                        # Berechne Aktualitätspunkte (höher für neuere Daten)
                        recency_score = max(0, 1 - (days_since_end / 365))  # Skaliert über ein Jahr
                    except:
                        recency_score = 0.5  # Standardwert bei Parsing-Fehler
                else:
                    recency_score = 0.5
                
                # Zeitspanne: Längere Zeitreihen sind wertvoller
                duration_days = time_range.get('duration_days', 0)
                if duration_days:
                    # Berechne Punktzahl für Zeitspanne (höher für längere Zeiträume)
                    span_score = min(1.0, duration_days / 365)  # Skaliert bis zu einem Jahr
                else:
                    span_score = 0.5
                
                # Kombiniere Aktualität und Zeitspanne
                time_relevance_score = (recency_score * 0.6) + (span_score * 0.4)  # Aktualität wichtiger als Spanne
            
            value_metrics['time_relevance']['score'] = time_relevance_score
            value_metrics['time_relevance']['explanation'] = ""
            
            if time_range:
                value_metrics['time_relevance']['explanation'] += f"Data spans {time_range.get('duration_days', '?')} days. "
                
                if time_relevance_score > 0.7:
                    value_metrics['time_relevance']['explanation'] += "Data is recent and covers a substantial time period. "
                elif time_relevance_score > 0.4:
                    value_metrics['time_relevance']['explanation'] += "Data has moderate temporal relevance. "
                else:
                    value_metrics['time_relevance']['explanation'] += "Data may be outdated or covers a limited time period. "
            else:
                value_metrics['time_relevance']['explanation'] += "No temporal information available to assess time relevance. "
            
            # 6. Anreicherungspotenzial
            # Bewerte, wie gut sich die Daten mit anderen Quellen kombinieren lassen
            
            # Identifiziere potenzielle Verknüpfungsfelder
            linkage_fields = []
            
            # Typische ID-Felder
            id_fields = [col for col in data.columns if 'id' in col.lower() or 'key' in col.lower()]
            if id_fields:
                linkage_fields.extend(id_fields)
            
            # Standortfelder für geografische Verknüpfung
            location_fields = [col for col in data.columns 
                              if 'location' in col.lower() or 'address' in col.lower() 
                              or 'city' in col.lower() or 'country' in col.lower()]
            if location_fields:
                linkage_fields.extend(location_fields)
            
            # Zeitfelder für zeitliche Verknüpfung
            time_fields = list(data.select_dtypes(include=['datetime']).columns)
            if time_fields:
                linkage_fields.extend(time_fields)
            
            # Kategorienwerte für kontextuelle Verknüpfung
            category_fields = [col for col in data.columns 
                              if data[col].dtype == 'object' and data[col].nunique() < 50]
            if category_fields:
                linkage_fields.extend(category_fields[:3])  # Beschränke auf die ersten drei
            
            # Einzigartige Linkage-Felder
            unique_linkage_fields = list(set(linkage_fields))
            
            # Berechne den Anreicherungswert basierend auf Linkage-Feldern
            if unique_linkage_fields:
                # Mehr Verknüpfungsfelder = höheres Anreicherungspotenzial
                field_count_score = min(1.0, len(unique_linkage_fields) / 5)  # Skaliert bis zu 5 Felder
                
                # Zusätzliche Punkte für besonders wertvolle Verknüpfungstypen
                id_bonus = 0.3 if any('id' in field.lower() for field in unique_linkage_fields) else 0
                location_bonus = 0.2 if any(field in location_fields for field in unique_linkage_fields) else 0
                time_bonus = 0.2 if any(field in time_fields for field in unique_linkage_fields) else 0
                
                # Berechne Gesamtwert mit Begrenzung auf 1.0
                enrichment_score = min(1.0, field_count_score + id_bonus + location_bonus + time_bonus)
            else:
                # Geringer Wert für Daten ohne offensichtliche Verknüpfungsmöglichkeiten
                enrichment_score = 0.2
            
            value_metrics['enrichment_potential']['score'] = enrichment_score
            value_metrics['enrichment_potential']['explanation'] = f"Identified {len(unique_linkage_fields)} potential linkage fields. "
            
            if unique_linkage_fields:
                field_types = []
                if any('id' in field.lower() for field in unique_linkage_fields):
                    field_types.append("IDs")
                if any(field in location_fields for field in unique_linkage_fields):
                    field_types.append("location data")
                if any(field in time_fields for field in unique_linkage_fields):
                    field_types.append("temporal markers")
                if any(field in category_fields for field in unique_linkage_fields):
                    field_types.append("categorical attributes")
                
                value_metrics['enrichment_potential']['explanation'] += "Linkage opportunities include: " + ", ".join(field_types) + ". "
            
            if enrichment_score > 0.7:
                value_metrics['enrichment_potential']['explanation'] += "Data has excellent potential for enrichment with other datasets. "
            elif enrichment_score > 0.4:
                value_metrics['enrichment_potential']['explanation'] += "Data has moderate potential for combination with complementary sources. "
            else:
                value_metrics['enrichment_potential']['explanation'] += "Data has limited enrichment potential without additional identifiers. "
            
            # Berechne den Gesamtwert als gewichtete Summe aller Faktoren
            total_weighted_score = sum(metric['score'] * metric['weight'] for metric in value_metrics.values())
            
            # Konvertiere den normalisierten Score in einen monetären Wert
            # Basierend auf einer konfigurierbaren Wertskala, z.B. 0-10 OCEAN Tokens
            min_value = self.config.get('min_token_value', 1.0)
            max_value = self.config.get('max_token_value', 10.0)
            estimated_token_value = min_value + (total_weighted_score * (max_value - min_value))
            
            # Erstelle die Wertzusammenfassung
            value_summary = {
                'normalized_score': float(total_weighted_score),
                'estimated_token_value': float(estimated_token_value),
                'metrics': value_metrics,
                'summary': f"This dataset has an estimated value of {estimated_token_value:.2f} OCEAN tokens, "
                          f"with its primary strengths being {', '.join(k for k, v in sorted([(k, v['score']) for k, v in value_metrics.items()], key=lambda x: x[1], reverse=True)[:2])}."
            }
            
            return value_summary
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenwertschätzung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'normalized_score': 0.3,  # Standardwert im Fehlerfall
                'estimated_token_value': self.config.get('min_token_value', 1.0),
                'metrics': {k: {'score': 0.3, 'weight': v['weight'], 'explanation': 'Error in calculation'} 
                           for k, v in value_metrics.items()}
            }

    def prepare_for_ocean_tokenization(self, data_asset: Dict) -> Dict:
        """
        Bereitet einen Daten-Asset für die Tokenisierung mit Ocean Protocol vor.
        
        Args:
            data_asset: Der vorzubereitende Daten-Asset mit Metadaten und C2D-Asset
            
        Returns:
            Für Ocean Protocol formatierte Asset-Informationen
        """
        try:
            self.logger.info(f"Bereite Asset für Ocean Protocol Tokenisierung vor")
            
            if 'error' in data_asset:
                raise ValueError(f"Fehlerhafter Daten-Asset: {data_asset['error']}")
            
            if 'c2d_asset' not in data_asset or 'metadata' not in data_asset:
                raise ValueError("Asset enthält nicht die erforderlichen Informationen (c2d_asset oder metadata)")
            
            # Extrahiere relevante Informationen
            asset_id = data_asset['c2d_asset']['asset_id']
            metadata = data_asset['metadata']
            
            # Erstelle ein DDO (Decentralized Data Object) im Ocean-Format
            name = metadata.get('name', f"Dataset {asset_id[:8]}")
            if 'source_type' in metadata:
                name = f"{metadata['source_type'].replace('_', ' ').title()} Dataset {asset_id[:8]}"
            
            description = metadata.get('description', '')
            if not description and 'potential_use_cases' in metadata and metadata['potential_use_cases']:
                # Verwende die Beschreibung des ersten Anwendungsfalls
                use_case = metadata['potential_use_cases'][0]
                description = f"{use_case.get('title', '')}: {use_case.get('description', '')}"
            
            # Kategorien basierend auf dem Quelltyp
            categories = []
            if 'source_type' in metadata:
                source_type = metadata['source_type']
                
                # Mapping von Quelltypen zu Ocean-Kategorien
                category_mapping = {
                    'browser': ['Web', 'Consumer Behavior'],
                    'calendar': ['Productivity', 'Time Management'],
                    'chat': ['Communication', 'Social'],
                    'social_media': ['Social', 'Marketing'],
                    'streaming': ['Entertainment', 'Media'],
                    'health_insurance': ['Healthcare', 'Insurance'],
                    'health_data': ['Healthcare', 'Wellness'],
                    'smartwatch': ['IoT', 'Wearables', 'Fitness'],
                    'iot_vacuum': ['IoT', 'Smart Home'],
                    'iot_thermostat': ['IoT', 'Energy'],
                    'iot_lighting': ['IoT', 'Smart Home'],
                    'iot_security': ['IoT', 'Security'],
                    'smart_home': ['IoT', 'Smart Home']
                }
                categories = category_mapping.get(source_type, ['Other'])
            
            # Erstelle Tags basierend auf Metadaten und Spalteninformationen
            tags = []
            
            # Tags basierend auf der Datenquelle
            if 'source_type' in metadata:
                tags.append(metadata['source_type'])
            
            # Tags basierend auf Spaltenstatistiken
            if 'column_statistics' in metadata:
                column_types = [stats.get('type') for stats in metadata['column_statistics'].values()]
                if 'datetime' in column_types or 'timestamp' in column_types:
                    tags.append('time-series')
                if 'text' in column_types:
                    tags.append('text-data')
                if 'categorical' in column_types:
                    tags.append('categorical-data')
                if any(t == 'numeric' for t in column_types):
                    tags.append('numeric-data')
                if any('location' in col.lower() for col in metadata.get('column_statistics', {}).keys()):
                    tags.append('geospatial')
            
            # Datenschutztags basierend auf dem Privacy-Level
            if 'privacy_level' in metadata:
                tags.append(f"privacy-{metadata['privacy_level']}")
            
            # Füge eine zeitliche Dimension hinzu, falls vorhanden
            if 'time_range' in metadata and metadata['time_range'] and 'duration_days' in metadata['time_range']:
                days = metadata['time_range']['duration_days']
                if days <= 7:
                    tags.append('short-term')
                elif days <= 90:
                    tags.append('medium-term')
                else:
                    tags.append('long-term')
            
            # Erstelle das DDO
            ddo = {
                'id': asset_id,
                'created': datetime.now().isoformat(),
                'updated': datetime.now().isoformat(),
                'type': 'dataset',
                'name': name,
                'description': description,
                'tags': list(set(tags)),  # Entferne Duplikate
                'categories': categories,
                'author': metadata.get('author', 'OceanData User'),
                'license': metadata.get('license', 'No License Specified'),
                'price': metadata.get('estimated_value', 5.0),  # Geschätzter Wert in OCEAN
                'files': [
                    {
                        'type': 'compute-to-data',
                        'method': 'c2d',
                        'schema': 'c2d-schema'
                    }
                ],
                'additionalInformation': {
                    'privacy': metadata.get('privacy_level', 'medium'),
                    'record_count': metadata.get('record_count', 0),
                    'field_count': metadata.get('field_count', 0),
                    'potential_use_cases': metadata.get('potential_use_cases', []),
                    'value_factors': metadata.get('value_factors', {})
                }
            }
            
            # Compute-to-Data-spezifische Informationen
            c2d_config = {
                'asset_id': asset_id,
                'allowed_operations': [
                    'aggregate', 'count', 'mean', 'sum', 'min', 'max', 'std', 
                    'correlation', 'histogram', 'custom_model'
                ],
                'timeout': 3600,  # 1 Stunde Standard-Timeout
                'privacy_config': {
                    'min_group_size': 5,
                    'noise_level': 0.01,
                    'outlier_removal': True
                }
            }
            
            # Preisgestaltung und Token-Informationen
            pricing = {
                'type': 'fixed',
                'baseToken': {
                    'address': '0x967da4048cD07aB37855c090aAF366e4ce1b9F48',  # Ocean Token Adresse
                    'name': 'Ocean Token',
                    'symbol': 'OCEAN'
                },
                'baseTokenAmount': metadata.get('estimated_value', 5.0),
                'datatoken': {
                    'name': f"DT-{name.replace(' ', '-')}",
                    'symbol': f"DT{asset_id[:4]}".upper()
                }
            }
            
            return {
                'ddo': ddo,
                'c2d_config': c2d_config,
                'pricing': pricing,
                'asset_id': asset_id
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Vorbereitung für Ocean Tokenisierung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'asset_id': data_asset.get('c2d_asset', {}).get('asset_id', 'unknown')
            }
    
    def tokenize_with_ocean(self, ocean_asset: Dict) -> Dict:
        """
        Tokenisiert einen Asset mit Ocean Protocol.
        Diese Funktion ist ein Platzhalter, da die tatsächliche Implementierung 
        von der Integration mit dem Ocean Protocol Smart Contract abhängt.
        
        Args:
            ocean_asset: Vorbereitete Asset-Informationen für Ocean Protocol
            
        Returns:
            Tokenisierungsergebnisse mit Transaktionsinformationen
        """
        # Dies ist eine Platzhalterimplementierung, die in einer echten Umgebung
        # einen Aufruf an die Ocean Protocol Bibliothek und Smart Contracts machen würde
        
        self.logger.info(f"Simuliere Tokenisierung mit Ocean Protocol für Asset {ocean_asset.get('asset_id', 'unknown')}")
        
        # Simulierte Tokenisierungsergebnisse
        tx_hash = f"0x{uuid.uuid4().hex                    if col in anonymized_data.columns and anonymized_data[col].dtype == 'object':
                        # Behalte nur den ersten Teil der Adresse/Standorts (z.B. Stadt)
                        anonymized_data[col] = anonymized_data[col].astype(str).apply(
                            lambda x: x.split(',')[0] if ',' in x else x
                        )
                
                # Vergröbere Zeitstempel (auf Stundenebene)
                for col in datetime_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = pd.to_datetime(anonymized_data[col]).dt.floor('H')
                
            elif privacy_level == 'low':
                # Minimale Anonymisierung, nur offensichtlich persönliche Kennung anonymisieren
                for col in id_columns:
                    if col in anonymized_data.columns and 'user' in col.lower():
                        salt = datetime.now().strftime("%Y%m%d")
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest() if pd.notna(x) else x
                        )
            
            # 2. Berechnung des Datenwerts und Erstellung von Metadaten
            
            # a) Berechne Basiskennzahlen
            metadata = {
                'source_type': source_type,
                'privacy_level': privacy_level,
                'record_count': len(anonymized_data),
                'field_count': len(anonymized_data.columns),
                'time_range': None,
                'created_at': datetime.now().isoformat(),
                'data_schema': {
                    col: str(anonymized_data[col].dtype) for col in anonymized_data.columns
                },
                'estimated_value': 0.0,
                'value_factors': {},
                'column_statistics': {},
                'potential_use_cases': []
            }
            
            # b) Berechne Zeitspanne, falls Zeitstempelspalten existieren
            if datetime_columns:
                # Wähle die erste Zeitstempelspalte
                time_col = datetime_columns[0]
                min_date = anonymized_data[time_col].min()
                max_date = anonymized_data[time_col].max()
                
                metadata['time_range'] = {
                    'start': min_date.isoformat() if pd.notna(min_date) else None,
                    'end': max_date.isoformat() if pd.notna(max_date) else None,
                    'duration_days': (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None
                }
            
            # c) Berechne Spaltenstatistiken für numerische und kategorische Spalten
            for col in anonymized_data.columns:
                col_stats = {}
                
                if pd.api.types.is_numeric_dtype(anonymized_data[col]):
                    # Statistiken für numerische Spalten
                    col_stats = {
                        'type': 'numeric',
                        'min': float(anonymized_data[col].min()) if not pd.isna(anonymized_data[col].min()) else None,
                        'max': float(anonymized_data[col].max()) if not pd.isna(anonymized_data[col].max()) else None,
                        'mean': float(anonymized_data[col].mean()) if not pd.isna(anonymized_data[col].mean()) else None,
                        'median': float(anonymized_data[col].median()) if not pd.isna(anonymized_data[col].median()) else None,
                        'std': float(anonymized_data[col].std()) if not pd.isna(anonymized_data[col].std()) else None,
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100)
                    }
                elif pd.api.types.is_string_dtype(anonymized_data[col]) or anonymized_data[col].dtype == 'object':
                    # Statistiken für kategorische/Text-Spalten
                    col_stats = {
                        'type': 'categorical' if anonymized_data[col].nunique() < 30 else 'text',
                        'unique_values': int(anonymized_data[col].nunique()),
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100),
                        'most_common': anonymized_data[col].value_counts().head(5).to_dict() if anonymized_data[col].nunique() < 30 else None
                    }
                    
                    # Wenn es sich um Text handelt, zusätzliche Textstatistiken
                    if col_stats['type'] == 'text':
                        # Durchschnittliche Textlänge
                        col_stats['avg_length'] = float(anonymized_data[col].astype(str).str.len().mean())
                        
                        # Optional: Sentimentanalyse auf Stichprobe
                        if self.config.get('analyze_text_sentiment', True):
                            sample = anonymized_data[col].dropna().sample(min(100, anonymized_data[col].dropna().shape[0])).tolist()
                            if sample:
                                sentiment_results = self.semantic_analyzer.analyze_sentiment(sample)
                                sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
                                for result in sentiment_results:
                                    sentiment_counts[result['sentiment']] += 1
                                
                                col_stats['sentiment'] = {
                                    k: v / len(sentiment_results) for k, v in sentiment_counts.items()
                                }
                else:
                    # Andere Datentypen (Datum, Boolean, etc.)
                    col_stats = {
                        'type': str(anonymized_data[col].dtype),
                        'unique_values': int(anonymized_data[col].nunique()),
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100)
                    }
                
                metadata['column_statistics'][col] = col_stats
            
            # d) Berechnung des geschätzten Datenwerts basierend auf verschiedenen Faktoren
            value_factors = {}
            
            # Faktor: Datenmenge
            data_size_factor = min(1.0, len(anonymized_data) / 10000)  # Skaliert bis 10.000 Datenpunkte
            value_factors['data_size'] = float(data_size_factor)
            
            # Faktor: Datenvollständigkeit
            completeness = 1.0 - anonymized_data.isna().mean().mean()  # Prozentsatz der nicht-fehlenden Werte
            value_factors['completeness'] = float(completeness)
            
            # Faktor: Zeitliche Spanne (falls verfügbar)
            time_factor = 0.5  # Standardwert
            if metadata['time_range'] and metadata['time_range']['duration_days']:
                # Höherer Wert für längere Zeitreihen (skaliert bis zu 1 Jahr)
                days = metadata['time_range']['duration_days']
                time_factor = min(1.0, days / 365)
            value_factors['time_span'] = float(time_factor)
            
            # Faktor: Datenquellentyp (unterschiedliche Gewichtung je nach Quelle)
            source_weights = {
                'browser': 0.7,
                'calendar': 0.6,
                'chat': 0.8,
                'social_media': 0.9,
                'streaming': 0.8,
                'health_insurance': 1.0,
                'health_data': 1.0,
                'smartwatch': 0.9,
                'iot_vacuum': 0.5,
                'iot_thermostat': 0.6,
                'iot_lighting': 0.5,
                'iot_security': 0.8,
                'smart_home': 0.7
            }
            source_factor = source_weights.get(source_type, 0.5)
            value_factors['source_type'] = float(source_factor)
            
            # Faktor: Datenschutzniveau (höherer Datenschutz = geringerer Wert)
            privacy_factors = {
                'low': 1.0,
                'medium': 0.8,
                'high': 0.6
            }
            privacy_factor = privacy_factors.get(privacy_level, 0.7)
            value_factors['privacy_level'] = float(privacy_factor)
            
            # Schätze den Gesamtwert als gewichteten Durchschnitt der Faktoren
            weights = {
                'data_size': 0.3,
                'completeness': 0.2,
                'time_span': 0.15,
                'source_type': 0.25,
                'privacy_level': 0.1
            }
            
            estimated_value = sum(value_factors[factor] * weights[factor] for factor in weights)
            
            # Konvertiere den normierten Wert in einen Tokenwert (angenommene Basis: 1-10 OCEAN)
            base_token_value = self.config.get('base_token_value', 5.0)  # 5 OCEAN als Basiswert
            token_value = base_token_value * estimated_value
            
            metadata['estimated_value'] = float(token_value)
            metadata['value_factors'] = value_factors
            
            # e) Identifiziere potenzielle Anwendungsfälle basierend auf den Daten
            use_cases = []
            
            # Browser-Daten: Surfgewohnheiten, Werbezielgruppen
            if source_type == 'browser':
                use_cases.append({
                    'title': 'Consumer Behavior Analysis',
                    'description': 'Analyze browsing patterns to understand consumer preferences and interests',
                    'value_proposition': 'Optimize marketing strategies and product recommendations'
                })
                use_cases.append({
                    'title': 'Ad Targeting Optimization',
                    'description': 'Use browsing history to create more relevant ad targeting profiles',
                    'value_proposition': 'Improve ad conversion rates and reduce marketing waste'
                })
            
            # Kalender-Daten: Zeitplanung, Produktivitätsanalyse
            elif source_type == 'calendar':
                use_cases.append({
                    'title': 'Productivity Pattern Analysis',
                    'description': 'Identify optimal meeting times and productivity patterns',
                    'value_proposition': 'Improve organizational efficiency and meeting scheduling'
                })
                use_cases.append({
                    'title': 'Work-Life Balance Insights',
                    'description': 'Analyze time allocation between work and personal activities',
                    'value_proposition': 'Develop better wellness and productivity programs'
                })
            
            # Gesundheitsdaten: Medizinische Forschung, Versicherungsanalyse
            elif source_type in ['health_data', 'health_insurance', 'smartwatch']:
                use_cases.append({
                    'title': 'Health Trend Analysis',
                    'description': 'Identify health patterns and correlations in anonymized health data',
                    'value_proposition': 'Support medical research and health intervention programs'
                })
                use_cases.append({
                    'title': 'Wellness Program Optimization',
                    'description': 'Use activity and health data to design better wellness initiatives',
                    'value_proposition': 'Improve health outcomes and reduce healthcare costs'
                })
            
            # IoT-Daten: Smart Home, Energieoptimierung
            elif source_type in ['iot_vacuum', 'iot_thermostat', 'iot_lighting', 'smart_home']:
                use_cases.append({
                    'title': 'Energy Usage Optimization',
                    'description': 'Analyze home energy consumption patterns',
                    'value_proposition': 'Develop more efficient energy management solutions'
                })
                use_cases.append({
                    'title': 'Smart Home Product Development',
                    'description': 'Understand usage patterns of smart home devices',
                    'value_proposition': 'Inform the design of new IoT products and services'
                })
            
            # Social Media und Chat: Stimmungsanalyse, Trendforschung
            elif source_type in ['social_media', 'chat']:
                use_cases.append({
                    'title': 'Sentiment Analysis and Trend Detection',
                    'description': 'Track public opinion and emerging topics in social conversations',
                    'value_proposition': 'Stay ahead of market trends and consumer sentiment shifts'
                })
                use_cases.append({
                    'title': 'Content Strategy Optimization',
                    'description': 'Analyze engagement patterns to optimize content strategy',
                    'value_proposition': 'Improve audience engagement and content performance'
                })
            
            # Streaming: Medienkonsum, Empfehlungen
            elif source_type == 'streaming':
                use_cases.append({
                    'title': 'Content Preference Analysis',
                    'description': 'Understand viewer preferences and consumption patterns',
                    'value_proposition': 'Optimize content creation and acquisition strategies'
                })
                use_cases.append({
                    'title': 'Recommendation Engine Improvement',
                    'description': 'Enhance content recommendation algorithms with viewing pattern data',
                    'value_proposition': 'Increase viewer satisfaction and platform engagement'
                })
            
            # Generische Anwendungsfälle für andere Quellen
            else:
                use_cases.append({
                    'title': 'Pattern Recognition',
                    'description': 'Identify patterns and trends in the data',
                    'value_proposition': 'Gain insights for business strategy and decision making'
                })
                use_cases.append({
                    'title': 'Predictive Analytics',
                    'description': 'Use historical data to predict future trends',
                    'value_proposition': 'Improve forecasting and strategic planning'
                })
            
            metadata['potential_use_cases'] = use_cases
            
            # Erstelle einen Asset für die Compute-to-Data-Funktionalität
            c2d_asset = self.c2d_manager.create_data_asset(
                anonymized_data,
                asset_metadata=metadata
            )
            
            return {
                'anonymized_data': anonymized_data,
                'metadata': metadata,
                'c2d_asset': c2d_asset
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenaufbereitung für Monetarisierung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'anonymized_data': None,
                'metadata': {
                    'source_type': source_type,
                    'privacy_level': privacy_level,
                    'error': str(e)
                }
            }
    
    def combine_data_sources(self, sources: List[Dict], combination_type: str = 'merge') -> Dict:
        """
        Kombiniert mehrere Datenquellen zu einem wertvolleren Datensatz.
        
        Args:
            sources: Liste von Datenquellen mit 'data' und 'metadata' Schlüsseln
            combination_type: Art der Kombination ('merge', 'enrich', 'correlate')
            
        Returns:
            Kombinierte Daten mit verbesserten Metadaten
        """
        try:
            self.logger.info(f"Kombiniere {len(sources)} Datenquellen mit Methode {combination_type}")
            
            # Prüfe, ob gültige Quellen vorhanden sind
            valid_sources = [s for s in sources if 'anonymized_data' in s and s['anonymized_data'] is not None]
            
            if len(valid_sources) < 2:
                raise ValueError("Mindestens zwei gültige Datenquellen werden für die Kombination benötigt")
            
            combined_data = None
            combined_metadata = {
                'combination_type': combination_type,
                'source_count': len(valid_sources),
                'source_types': [s['metadata']['source_type'] for s in valid_sources],
                'created_at': datetime.now().isoformat(),
                'record_count': 0,
                'field_count': 0,
                'estimated_value': 0.0,
                'combination_description': '',
                'source_metadata': [s['metadata'] for s in valid_sources]
            }
            
            # Implementierung je nach Kombinationstyp
            if combination_type == 'merge':
                # Horizontale Zusammenführung (Zeilen anfügen)
                # Geeignet für gleichartige Datensätze
                
                all_data = [s['anonymized_data'] for s in valid_sources]
                
                # Prüfe Spaltenkompatibilität
                common_columns = set.intersection(*[set(df.columns) for df in all_data])
                
                if not common_columns:
                    raise ValueError("Keine gemeinsamen Spalten für die Zusammenführung gefunden")
                
                # Zusammenführung der Daten mit gemeinsamen Spalten
                combined_data = pd.concat([df[list(common_columns)] for df in all_data], ignore_index=True)
                
                combined_metadata['combination_description'] = f"Horizontal merge of {len(valid_sources)} similar datasets, keeping {len(common_columns)} common columns"
                combined_metadata['record_count'] = len(combined_data)
                combined_metadata['field_count'] = len(combined_data.columns)
                
                # Wertschätzung: Summe der Einzelwerte mit einem Multiplikator für die Kombination
                base_value = sum(s['metadata'].get('estimated_value', 0) for s in valid_sources)
                combined_metadata['estimated_value'] = base_value * 1.2  # 20% Mehrwert durch Kombination
                
            elif combination_type == 'enrich':
                # Vertikale Anreicherung (Spalten anfügen)
                # Geeignet für komplementäre Datensätze mit einem gemeinsamen Identifikator
                
                # Erster Datensatz als Basis
                base_data = valid_sources[0]['anonymized_data']
                join_key = None
                
                # Suche nach einem gemeinsamen Schlüssel für die Verknüpfung
                for col in base_data.columns:
                    if all(col in s['anonymized_data'].columns for s in valid_sources[1:]):
                        # Prüfe, ob der Wert als Schlüssel dienen kann (eindeutige Werte)
                        if all(s['anonymized_data'][col].nunique() / len(s['anonymized_data']) > 0.7 for s in valid_sources):
                            join_key = col
                            break
                
                if not join_key:
                    # Falls kein expliziter Schlüssel existiert, erstelle einen künstlichen Schlüssel
                    self.logger.warning("Kein natürlicher Schlüssel gefunden, erstelle künstlichen Index")
                    
                    # Erstelle einen eindeutigen Index für jeden Datensatz
                    for i, source in enumerate(valid_sources):
                        source['anonymized_data']['temp_join_key'] = range(len(source['anonymized_data']))
                    
                    join_key = 'temp_join_key'
                
                # Führe die Daten zusammen
                combined_data = base_data.copy()
                
                for source in valid_sources[1:]:
                    # Finde neue Spalten, die noch nicht existieren
                    new_columns = [col for col in source['anonymized_data'].columns if col not in combined_data.columns]
                    
                    if new_columns:
                        # Verknüpfe über den Schlüssel mit linkem Join (behält alle Zeilen von combined_data)
                        combined_data = pd.merge(
                            combined_data,
                            source['anonymized_data'][new_columns + [join_key]],
                            on=join_key,
                            how='left'
                        )
                
                # Entferne den temporären Schlüssel, falls erstellt
                if join_key == 'temp_join_key' and join_key in combined_data.columns:
                    combined_data = combined_data.drop(columns=[join_key])
                
                combined_metadata['combination_description'] = f"Vertical enrichment using key '{join_key}', adding {len(combined_data.columns) - len(base_data.columns)} new features"
                combined_metadata['record_count'] = len(combined_data)
                combined_metadata['field_count'] = len(combined_data.columns)
                
                # Wertschätzung: Wert der Basis plus zusätzlicher Wert für neue Spalten
                base_value = valid_sources[0]['metadata'].get('estimated_value', 0)
                additional_value = sum(s['metadata'].get('estimated_value', 0) * 0.6 for s in valid_sources[1:])  # 60% des Werts der zusätzlichen Quellen
                combined_metadata['estimated_value'] = base_value + additional_value
                
            elif combination_type == 'correlate':
                # Korrelationsanalyse zwischen verschiedenen Datenquellen
                # Erstellt einen neuen Datensatz mit Korrelationsmaßen und abgeleiteten Features
                
                # Erstelle eine Liste für Korrelationsergebnisse
                correlation_results = []
                feature_pairs = []
                
                # Suche nach numerischen Spalten in jedem Datensatz
                for i, source1 in enumerate(valid_sources):
                    for j in range(i+1, len(valid_sources)):
                        source2 = valid_sources[j]
                        
                        # Extrahiere numerische Spalten
                        num_cols1 = source1['anonymized_data'].select_dtypes(include=['number']).columns
                        num_cols2 = source2['anonymized_data'].select_dtypes(include=['number']).columns
                        
                        # Prüfe, ob genügend numerische Spalten vorhanden sind
                        if len(num_cols1) == 0 or len(num_cols2) == 0:
                            continue
                        
                        # Wähle die Top-5-Spalten pro Quelle für die Analyse
                        select_cols1 = list(num_cols1)[:5]
                        select_cols2 = list(num_cols2)[:5]
                        
                        # Erstelle ein DataFrame mit Korrelationswerten
                        source_type1 = source1['metadata']['source_type']
                        source_type2 = source2['metadata']['source_type']
                        
                        for col1 in select_cols1:
                            for col2 in select_cols2:
                                # Erstelle 100 zufällige Datenpunkte für jede Spalte
                                # (für eine Stichprobenkorrelation ohne Verknüpfung der tatsächlichen Daten)
                                sample1 = source1['anonymized_data'][col1].sample(min(100, len(source1['anonymized_data']))).reset_index(drop=True)
                                sample2 = source2['anonymized_data'][col2].sample(min(100, len(source2['anonymized_data']))).reset_index(drop=True)
                                
                                # Berechne die Korrelation
                                try:
                                    corr = sample1.corr(sample2)
                                    
                                    # Speichere nur signifikante Korrelationen
                                    if abs(corr) > 0.3:
                                        corr_entry = {
                                            'source1': source_type1,
                                            'source2': source_type2,
                                            'feature1': col1,
                                            'feature2': col2,
                                            'correlation': float(corr),
                                            'abs_correlation': float(abs(corr))
                                        }
                                        correlation_results.append(corr_entry)
                                        feature_pairs.append((source1, col1, source2, col2))
                                except:
                                    # Ignoriere Fehler bei der Korrelationsberechnung
                                    pass
                
                # Erstelle ein neues DataFrame mit den Korrelationsergebnissen
                if correlation_results:
                    combined_data = pd.DataFrame(correlation_results)
                    
                    # Füge abgeleitete Features für die Top-Korrelationen hinzu
                    if feature_pairs:
                        # Sortiere nach absoluter Korrelationsstärke
                        sorted_pairs = sorted(list(zip(correlation_results, feature_pairs)), 
                                             key=lambda x: x[0]['abs_correlation'], 
                                             reverse=True)
                        
                        # Wähle die Top-10-Paare (oder weniger, falls nicht genügend vorhanden)
                        top_pairs = sorted_pairs[:min(10, len(sorted_pairs))]
                        
                        # Erstelle für jedes Top-Paar ein kombiniertes Feature
                        for i, (corr_info, (source1, col1, source2, col2)) in enumerate(top_pairs):
                            # Normalisiere die Werte
                            samples1 = source1['anonymized_data'][col1].sample(min(100, len(source1['anonymized_data']))).reset_index(drop=True)
                            samples2 = source2['anonymized_data'][col2].sample(min(100, len(source2['anonymized_data']))).reset_index(drop=True)
                            
                            # Z-Score-Normalisierung
                            normalized1 = (samples1 - samples1.mean()) / samples1.std()
                            normalized2 = (samples2 - samples2.mean()) / samples2.std()
                            
                            # Erstelle ein kombiniertes Feature basierend auf der Korrelationsrichtung
                            if corr_info['correlation'] >= 0:
                                combined_feature = normalized1 + normalized2
                            else:
                                combined_feature = normalized1 - normalized2
                            
                            # Füge das kombinierte Feature dem Ergebnis hinzu
                            feature_name = f"combined_{source1['metadata']['source_type']}_{col1}_{source2['metadata']['source_type']}_{col2}"
                            combined_data[feature_name] = combined_feature
                    
                    combined_metadata['combination_description'] = f"Correlation analysis between {len(valid_sources)} datasets, identifying {len(correlation_results)} significant cross-source relationships"
                    combined_metadata['record_count'] = len(combined_data)
                    combined_metadata['field_count'] = len(combined_data.columns)
                    
                    # Wertschätzung: Basierend auf der Anzahl und Stärke der gefundenen Korrelationen
                    avg_correlation = np.mean([r['abs_correlation'] for r in correlation_results]) if correlation_results else 0
                    correlation_value = len(correlation_results) * avg_correlation * 0.5  # Grundwert pro signifikante Korrelation
                    
                    # Füge Werte der Originalquellen hinzu
                    base_value = sum(s['metadata'].get('estimated_value', 0) * 0.3 for s in valid_sources)  # 30% der Originalwerte
                    combined_metadata['estimated_value'] = base_value + correlation_value
                else:
                    raise ValueError("Keine signifikanten Korrelationen zwischen den Datenquellen gefunden")
            
            else:
                raise ValueError(f"Nicht unterstützter Kombinationstyp: {combination_type}")
            
            # Erstelle einen neuen C2D-Asset für den kombinierten Datensatz
            c2d_asset = self.c2d_manager.create_data_asset(
                combined_data,
                asset_metadata=combined_metadata
            )
            
            return {
                'anonymized_data': combined_data,
                'metadata': combined_metadata,
                'c2d_asset': c2d_asset
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Kombination von Datenquellen: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'anonymized_data': None,
                'metadata': {
                    'combination_type': combination_type,
                    'source_count': len(sources),
                    'error': str(e)
                }
            }
    
    def estimate_data_value(self, data: pd.DataFrame, metadata: Dict = None) -> Dict:
        """
        Berechnet einen detaillierten Wert für einen Datensatz basierend auf mehreren Faktoren.
        
        Args:
            data: Der zu bewertende Datensatz
            metadata: Optionale Metadaten über den Datensatz
            
        Returns:
            Detaillierte Wertschätzung mit Begründungen
        """
        try:
            self.logger.info(f"Berechne detaillierten Datenwert für Datensatz mit {len(data)} Zeilen und {len(data.columns)} Spalten")
            
            if metadata is None:
                metadata = {}
            
            # Basisfaktoren für die Wertschätzung
            value_metrics = {
                'data_volume': {
                    'score': 0.0,
                    'weight': 0.15,
                    'explanation': ''
                },
                'data_quality': {
                    'score': 0.0,
                    'weight': 0.2,
                    'explanation': ''
                },
                'data_uniqueness': {
                    'score': 0._group_size']].index
            
            means = grouped.mean().loc[valid_groups]
            
            # Füge Rauschen hinzu
            noisy_means = self._add_differential_privacy_noise(means)
            
            return {'group_means': noisy_means.to_dict()}
        else:
            means = prepared_data.mean()
            
            # Füge Rauschen hinzu
            noisy_means = self._add_differential_privacy_noise(means)
            
            return {'means': noisy_means.to_dict()}
    
    def _sum(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Summe für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        sums = prepared_data.sum()
        
        # Füge Rauschen hinzu
        noisy_sums = self._add_differential_privacy_noise(sums)
        
        return {'sums': noisy_sums.to_dict()}
    
    def _min(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet das Minimum für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        minimums = prepared_data.min()
        
        # Hier kein Rauschen, da Minimum keine sensiblen Informationen preisgibt
        return {'minimums': minimums.to_dict()}
    
    def _max(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet das Maximum für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        maximums = prepared_data.max()
        
        # Hier kein Rauschen, da Maximum keine sensiblen Informationen preisgibt
        return {'maximums': maximums.to_dict()}
    
    def _std(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Standardabweichung für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        std_devs = prepared_data.std()
        
        # Füge Rauschen hinzu
        noisy_std_devs = self._add_differential_privacy_noise(std_devs)
        
        return {'standard_deviations': noisy_std_devs.to_dict()}
    
    def _correlation(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Korrelation zwischen numerischen Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        # Verwende nur numerische Spalten
        numeric_data = prepared_data.select_dtypes(include=['number'])
        
        if numeric_data.empty or numeric_data.shape[1] < 2:
            return {'error': 'Nicht genügend numerische Spalten für Korrelation'}
        
        corr_matrix = numeric_data.corr()
        
        # Füge Rauschen hinzu
        noisy_corr = self._add_differential_privacy_noise(corr_matrix)
        
        # Stelle sicher, dass Werte im gültigen Bereich liegen
        noisy_corr = noisy_corr.clip(-1, 1)
        
        return {'correlation_matrix': noisy_corr.to_dict()}
    
    def _histogram(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Erstellt ein Histogramm für eine Spalte"""
        column = params.get('column')
        bins = params.get('bins', 10)
        
        if column not in data.columns:
            return {'error': f'Spalte {column} nicht gefunden'}
        
        prepared_data = self._prepare_data_for_computation(data, [column])
        
        if pd.api.types.is_numeric_dtype(prepared_data[column]):
            # Numerisches Histogramm
            hist, bin_edges = np.histogram(prepared_data[column], bins=bins)
            
            # Füge Rauschen hinzu
            noisy_hist = self._add_differential_privacy_noise(hist).astype(int)
            
            # Stelle sicher, dass Werte nicht negativ sind
            noisy_hist = np.maximum(0, noisy_hist)
            
            return {
                'histogram': noisy_hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        else:
            # Kategorisches Histogramm
            value_counts = prepared_data[column].value_counts()
            
            # Prüfe auf zu kleine Gruppen
            value_counts = value_counts[value_counts >= self.privacy_config['min_group_size']]
            
            if value_counts.empty:
                return {'error': 'Keine Kategorien mit ausreichender Größe'}
            
            # Füge Rauschen hinzu
            noisy_counts = self._add_differential_privacy_noise(value_counts).astype(int)
            
            # Stelle sicher, dass Werte nicht negativ sind
            noisy_counts = noisy_counts.clip(0)
            
            return {
                'categories': value_counts.index.tolist(),
                'counts': noisy_counts.tolist()
            }
    
    def _custom_model_inference(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Führt Inferenz mit einem benutzerdefinierten Modell durch, ohne die Daten preiszugeben
        
        Args:
            data: Eingabedaten für die Inferenz
            params: Parameter für die Inferenz, einschließlich des Modelltyps und der Modellparameter
            
        Returns:
            Ergebnisse der Inferenz
        """
        model_type = params.get('model_type')
        model_config = params.get('model_config', {})
        columns = params.get('columns')
        
        if not model_type:
            return {'error': 'Kein Modelltyp angegeben'}
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        try:
            results = None
            
            if model_type == 'anomaly_detection':
                # Anomalieerkennung mit Isolation Forest
                detector = AnomalyDetector(method='isolation_forest', contamination=model_config.get('contamination', 0.05))
                detector.fit(prepared_data)
                predictions = detector.predict(prepared_data)
                
                # Aggregiere Ergebnisse für den Datenschutz
                anomaly_count = np.sum(predictions == -1)
                total_count = len(predictions)
                
                # Füge Rauschen hinzu
                noisy_anomaly_count = self._add_differential_privacy_noise(anomaly_count).astype(int)
                
                results = {
                    'anomaly_percentage': round(100 * noisy_anomaly_count / total_count, 2),
                    'total_records': total_count
                }
            
            elif model_type == 'clustering':
                # K-Means-Clustering
                n_clusters = model_config.get('n_clusters', 3)
                
                # Verarbeite nur numerische Spalten
                numeric_data = prepared_data.select_dtypes(include=['number'])
                
                if numeric_data.empty:
                    return {'error': 'Keine numerischen Spalten für Clustering gefunden'}
                
                # Standardisiere Daten
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                # Cluster-Analyse
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Cluster-Statistiken (datenschutzfreundlich)
                cluster_sizes = {}
                for i in range(n_clusters):
                    size = np.sum(clusters == i)
                    
                    # Prüfe Mindestgruppengröße
                    if size >= self.privacy_config['min_group_size']:
                        # Füge Rauschen hinzu
                        noisy_size = self._add_differential_privacy_noise(size).astype(int)
                        cluster_sizes[f'cluster_{i}'] = max(0, noisy_size)
                
                # Cluster-Zentren (keine persönlichen Daten)
                cluster_centers = kmeans.cluster_centers_
                
                results = {
                    'cluster_sizes': cluster_sizes,
                    'cluster_centers': cluster_centers.tolist()
                }
            
            elif model_type == 'sentiment_analysis':
                # Sentimentanalyse für Textdaten
                text_column = model_config.get('text_column')
                
                if text_column not in prepared_data.columns:
                    return {'error': f'Textspalte {text_column} nicht gefunden'}
                
                # Einfache Sentimentanalyse mit NLTK
                analyzer = SemanticAnalyzer()
                sentiment_results = analyzer.analyze_sentiment(prepared_data[text_column].tolist())
                
                # Aggregierte Ergebnisse für den Datenschutz
                sentiment_counts = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
                
                for result in sentiment_results:
                    sentiment_counts[result['sentiment']] += 1
                
                # Füge Rauschen hinzu
                for sentiment in sentiment_counts:
                    noisy_count = self._add_differential_privacy_noise(sentiment_counts[sentiment]).astype(int)
                    sentiment_counts[sentiment] = max(0, noisy_count)
                
                results = {
                    'sentiment_distribution': sentiment_counts,
                    'total_records': len(sentiment_results)
                }
            
            else:
                return {'error': f'Nicht unterstützter Modelltyp: {model_type}'}
            
            return results
            
        except Exception as e:
            logger.error(f"Fehler bei der Modell-Inferenz: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Modell-Inferenz: {str(e)}'}
    
    def execute_operation(self, encrypted_data: bytes, operation: str, params: Dict) -> Dict:
        """
        Führt eine Operation auf verschlüsselten Daten aus, ohne sie zu entschlüsseln
        
        Args:
            encrypted_data: Verschlüsselte Daten
            operation: Name der auszuführenden Operation
            params: Parameter für die Operation
            
        Returns:
            Ergebnisse der Operation
        """
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
            logger.error(f"Fehler bei der Ausführung der Operation {operation}: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Ausführung: {str(e)}'}
    
    def create_data_asset(self, data: pd.DataFrame, asset_metadata: Dict = None) -> Dict:
        """
        Erstellt einen verschlüsselten Daten-Asset mit Metadaten für den Marktplatz
        
        Args:
            data: Zu verschlüsselnde Daten
            asset_metadata: Metadaten für den Asset
            
        Returns:
            Asset-Details mit ID und Metadaten
        """
        try:
            # Erstelle eine eindeutige ID für den Asset
            asset_id = uuid.uuid4().hex
            
            # Verschlüssele die Daten
            encrypted_data = self._encrypt_data(data)
            
            # Erstelle Metadaten mit datenschutzfreundlichen Statistiken
            stats = {}
            
            # Anzahl der Datensätze
            stats['record_count'] = len(data)
            
            # Spalteninformationen
            stats['columns'] = []
            for col in data.columns:
                col_info = {
                    'name': col,
                    'type': str(data[col].dtype)
                }
                
                # Füge datenschutzfreundliche Statistiken hinzu
                if pd.api.types.is_numeric_dtype(data[col]):
                    # Füge Rauschen hinzu, um Datenschutz zu gewährleisten
                    col_info['min'] = float(self._add_differential_privacy_noise(data[col].min()))
                    col_info['max'] = float(self._add_differential_privacy_noise(data[col].max()))
                    col_info['mean'] = float(self._add_differential_privacy_noise(data[col].mean()))
                    col_info['std'] = float(self._add_differential_privacy_noise(data[col].std()))
                else:
                    # Für kategorische Daten nur die Anzahl eindeutiger Werte
                    col_info['unique_values'] = int(self._add_differential_privacy_noise(data[col].nunique()))
                
                stats['columns'].append(col_info)
            
            # Zusammenführen mit benutzerdefinierten Metadaten
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
            logger.error(f"Fehler bei der Erstellung des Daten-Assets: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Asset-Erstellung: {str(e)}'}
    
    def generate_access_token(self, asset_id: str, allowed_operations: List[str], 
                            expiration_time: int = 3600) -> Dict:
        """
        Generiert ein temporäres Zugriffstoken für einen Daten-Asset
        
        Args:
            asset_id: ID des Daten-Assets
            allowed_operations: Liste der erlaubten Operationen
            expiration_time: Ablaufzeit in Sekunden
            
        Returns:
            Token-Details mit Token und Ablaufzeit
        """
        # Prüfe, ob alle angeforderten Operationen unterstützt werden
        unsupported_ops = [op for op in allowed_operations if op not in self.allowed_operations]
        if unsupported_ops:
            return {'error': f'Nicht unterstützte Operationen: {", ".join(unsupported_ops)}'}
        
        # Erstelle Token-Daten
        token_data = {
            'asset_id': asset_id,
            'allowed_operations': allowed_operations,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=expiration_time)).isoformat(),
            'token_id': uuid.uuid4().hex
        }
        
        # Verschlüssele Token-Daten
        token_json = json.dumps(token_data).encode()
        encrypted_token = self.cipher_suite.encrypt(token_json)
        
        return {
            'token': encrypted_token.decode(),
            'token_id': token_data['token_id'],
            'expires_at': token_data['expires_at']
        }
    
    def validate_access_token(self, token: str, operation: str) -> bool:
        """
        Validiert ein Zugriffstoken für eine bestimmte Operation
        
        Args:
            token: Zugriffstoken
            operation: Angeforderte Operation
            
        Returns:
            True, wenn das Token gültig ist und die Operation erlaubt ist
        """
        try:
            # Entschlüssele das Token
            decrypted_token = self.cipher_suite.decrypt(token.encode())
            token_data = json.loads(decrypted_token.decode())
            
            # Prüfe, ob das Token abgelaufen ist
            expiry_time = datetime.fromisoformat(token_data['expires_at'])
            if datetime.now() > expiry_time:
                logger.warning(f"Zugriffstoken abgelaufen: {token_data['token_id']}")
                return False
            
            # Prüfe, ob die Operation erlaubt ist
            if operation not in token_data['allowed_operations']:
                logger.warning(f"Operation {operation} nicht erlaubt für Token {token_data['token_id']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Token-Validierung: {str(e)}")
            return False

###########################################
# 6. Integration und OceanData-Kernmodule
###########################################

class OceanDataAI:
    """
    Hauptklasse, die alle KI-Module für OceanData integriert und koordiniert.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialisiert den OceanData-KI-Manager.
        
        Args:
            config: Konfigurationsparameter für die KI-Module
        """
        self.config = config or {}
        self.logger = logging.getLogger("OceanData.AI")
        
        # Instanziiere die einzelnen Module
        self.anomaly_detector = AnomalyDetector(
            method=self.config.get('anomaly_detection_method', 'isolation_forest'),
            contamination=self.config.get('anomaly_contamination', 0.05)
        )
        
        self.semantic_analyzer = SemanticAnalyzer(
            model_type=self.config.get('semantic_model', 'bert'),
            model_name=self.config.get('semantic_model_name', 'bert-base-uncased')
        )
        
        self.predictive_modeler = PredictiveModeler(
            model_type=self.config.get('predictive_model', 'lstm'),
            forecast_horizon=self.config.get('forecast_horizon', 7)
        )
        
        self.data_synthesizer = DataSynthesizer(
            categorical_threshold=self.config.get('categorical_threshold', 10),
            noise_dim=self.config.get('gan_noise_dim', 100)
        )
        
        self.c2d_manager = ComputeToDataManager()
        
        # Aktiviere GPU-Beschleunigung, falls verfügbar
        self._setup_hardware_acceleration()
    
    def _setup_hardware_acceleration(self):
        """Konfiguriert Hardware-Beschleunigung für die KI-Module"""
        try:
            # Prüfe, ob GPUs für TensorFlow verfügbar sind
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.logger.info(f"GPU-Beschleunigung aktiviert mit {len(gpus)} GPUs")
                
                # Konfiguriere TensorFlow für bessere GPU-Nutzung
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                self.logger.info("Keine GPUs gefunden, verwende CPU-Computing")
        
        except Exception as e:
            self.logger.warning(f"Fehler bei der GPU-Konfiguration: {str(e)}")
            self.logger.info("Verwende Standard-CPU-Computing")
    
    def analyze_data_source(self, data: pd.DataFrame, source_type: str) -> Dict:
        """
        Führt eine umfassende Analyse einer Datenquelle durch.
        
        Args:
            data: Die zu analysierenden Daten
            source_type: Typ der Datenquelle (browser, calendar, iot, etc.)
            
        Returns:
            Umfassende Analyseergebnisse
        """
        results = {
            'source_type': source_type,
            'timestamp': datetime.now().isoformat(),
            'record_count': len(data),
            'column_count': len(data.columns),
            'analyses': {}
        }
        
        try:
            # 1. Ausreißererkennung
            self.logger.info(f"Führe Ausreißererkennung für {source_type}-Daten durch")
            self.anomaly_detector.fit(data)
            anomaly_predictions = self.anomaly_detector.predict(data)
            anomaly_insights = self.anomaly_detector.get_anomaly_insights(
                data, anomaly_predictions, top_n_features=5
            )
            
            results['analyses']['anomalies'] = {
                'count': int(np.sum(anomaly_predictions == -1) if self.anomaly_detector.method == 'isolation_forest' else np.sum(anomaly_predictions)),
                'percentage': float(np.mean(anomaly_predictions == -1) * 100 if self.anomaly_detector.method == 'isolation_forest' else np.mean(anomaly_predictions) * 100),
                'insights': anomaly_insights[:5]  # Begrenze auf die Top-5-Insights
            }
            
            # 2. Textanalyse für relevante Textspalten
            text_columns = [col for col in data.columns 
                           if data[col].dtype == 'object' 
                           and data[col].str.len().mean() > 20]
            
            if text_columns:
                self.logger.info(f"Führe Textanalyse für {len(text_columns)} Spalten durch")
                text_analyses = {}
                
                for col in text_columns[:3]:  # Begrenze auf die ersten 3 Textspalten
                    # Wähle Texte aus, die nicht leer sind
                    texts = data[col].dropna().astype(str).tolist()
                    if not texts:
                        continue
                        
                    # Stichprobe für Performance
                    sample_size = min(100, len(texts))
                    text_sample = random.sample(texts, sample_size)
                    
                    # Sentimentanalyse
                    sentiments = self.semantic_analyzer.analyze_sentiment(text_sample)
                    
                    # Themenextraktion
                    topics = self.semantic_analyzer.extract_topics(text_sample, num_topics=3)
                    
                    text_analyses[col] = {
                        'sentiment': {
                            'positive': len([s for s in sentiments if s['sentiment'] == 'positive']) / len(sentiments),
                            'neutral': len([s for s in sentiments if s['sentiment'] == 'neutral']) / len(sentiments),
                            'negative': len([s for s in sentiments if s['sentiment'] == 'negative']) / len(sentiments)
                        },
                        'topics': topics
                    }
                
                results['analyses']['text'] = text_analyses
            
            # 3. Zeitreihenanalyse für Zeitstempel-basierte Daten
            datetime_columns = [col for col in data.columns 
                              if pd.api.types.is_datetime64_any_dtype(data[col])]
            
            if datetime_columns and len(data) > 10:
                self.logger.info("Führe Zeitreihenanalyse durch")
                
                # Wähle die erste Zeitstempelspalte für die Analyse
                time_col = datetime_columns[0]
                
                # Sortiere Daten nach Zeit
                sorted_data = data.sort_values(by=time_col)
                
                # Wähle numerische Spalten für die Vorhersage
                numeric_cols = sorted_data.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    # Wähle die ersten 3 numerischen Spalten für die Demonstration
                    target_cols = numeric_cols[:3]
                    
                    # Erstelle Zeitreihen-Features
                    sorted_data['hour'] = sorted_data[time_col].dt.hour
                    sorted_data['day_of_week'] = sorted_data[time_col].dt.dayofweek
                    sorted_data['day_of_month'] = sorted_data[time_col].dt.day
                    sorted_data['month'] = sorted_data[time_col].dt.month
                    
                    # Bilde Feature- und Zielspalten
                    X = sorted_data[['hour', 'day_of_week', 'day_of_month', 'month']]
                    y = sorted_data[target_cols]
                    
                    if len(X) > 20:  # Mindestanzahl für sinnvolle Zeitreihenanalyse
                        # Trainiere ein einfaches Modell
                        self.predictive_modeler.fit(X, y, lookback=5, epochs=20, verbose=0)
                        
                        # Forecasting für die nächsten 7 Tage
                        forecast = self.predictive_modeler.forecast(X, steps=7)
                        
                        results['analyses']['time_series'] = {
                            'forecast_horizon': 7,
                            'forecasted_features': target_cols,
                            'forecasts': {
                                col: forecast[:, i].tolist() 
                                for i, col in enumerate(target_cols)
                            }
                        }
            
            # 4. Datensyntheseanalyse (optional, ressourcenintensiv)
            if self.config.get('enable_data_synthesis', False) and len(data) > 50:
                self.logger.info("Führe Datensynthese-Analyse durch")
                
                # Wähle eine Stichprobe und maximal 10 Spalten für die Demonstration
                sample_size = min(1000, len(data))
                data_sample = data.sample(sample_size)
                
                if len(data.columns) > 10:
                    # Wähle die wichtigsten Spalten aus
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()[:5]
                    cat_cols = data.select_dtypes(exclude=['number']).columns.tolist()[:5]
                    selected_cols = numeric_cols + cat_cols
                    data_sample = data_sample[selected_cols]
                
                # Trainiere das GAN mit einer kleinen Anzahl von Epochen
                self.data_synthesizer.fit(data_sample, epochs=100, verbose=0)
                
                # Generiere einige synthetische Beispiele
                synthetic_examples = self.data_synthesizer.generate(10)
                
                # Bewerte die Qualität
                quality_metrics = self.data_synthesizer.evaluate_quality(100)
                
                results['analyses']['data_synthesis'] = {
                    'quality_score': quality_metrics.get('overall_quality_score', 0),
                    'sample_examples': synthetic_examples.head(3).to_dict()
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenanalyse: {str(e)}")
            self.logger.error(traceback.format_exc())
            results['error'] = str(e)
            return results
    
    def prepare_data_for_monetization(self, data: pd.DataFrame, source_type: str, 
                                    privacy_level: str = 'high') -> Dict:
        """
        Bereitet Daten für die Monetarisierung vor, einschließlich Anonymisierung und Metadatengenerierung.
        
        Args:
            data: Die zu monetarisierenden Daten
            source_type: Typ der Datenquelle
            privacy_level: Datenschutzniveau ('low', 'medium', 'high')
            
        Returns:
            Aufbereitete Daten mit Metadaten für die Monetarisierung
        """
        try:
            self.logger.info(f"Bereite {source_type}-Daten für Monetarisierung vor (Privacy: {privacy_level})")
            
            # 1. Datenschutzmaßnahmen basierend auf dem gewählten Niveau
            anonymized_data = data.copy()
            
            # Identifiziere sensible Spalten
            id_columns = [col for col in data.columns 
                         if 'id' in col.lower() or 'user' in col.lower() or 'email' in col.lower()]
            location_columns = [col for col in data.columns 
                              if 'location' in col.lower() or 'address' in col.lower() or 'gps' in col.lower() or 'ip' in col.lower()]
            datetime_columns = [col for col in data.columns 
                              if pd.api.types.is_datetime64_any_dtype(data[col])]
            
            # Anonymisierung basierend auf Datenschutzniveau
            if privacy_level == 'high':
                # Vollständige Anonymisierung und Entfernung sensibler Daten
                anonymized_data = anonymized_data.drop(columns=id_columns, errors='ignore')
                
                # Anonymisiere Standortdaten
                for col in location_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: f"location_{hashlib.md5(str(x).encode()).hexdigest()[:8]}" if pd.notna(x) else x
                        )
                
                # Vergröbere Zeitstempel (nur auf Tagesebene)
                for col in datetime_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = pd.to_datetime(anonymized_data[col]).dt.floor('D')
                
            elif privacy_level == 'medium':
                # Moderate Anonymisierung
                for col in id_columns:
                    if col in anonymized_data.columns:
                        salt = datetime.now().strftime("%Y%m%d")
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest() if pd.notna(x) else x
                        )
                
                # Vergröbere Standortdaten
                for col in location_columns:
                    if col in anonymized_data.columns and anonymized_data[col].dtype == 'object':
                """
OceanData - Fortgeschrittene KI-Module für Datenanalyse und -monetarisierung

Diese Module erweitern den Kernalgorithmus von OceanData um fortschrittliche KI-Funktionen:
1. Anomalieerkennung
2. Semantische Datenanalyse 
3. Prädiktive Modellierung
4. Datensynthese und Erweiterung
5. Multimodale Analyse
6. Federated Learning und Compute-to-Data
7. Kontinuierliches Lernen und Modellanpassung
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertTokenizer, BertModel, TFBertModel
from transformers import GPT2Tokenizer, GPT2Model, TFGPT2Model
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import json
from typing import Dict, List, Union, Any, Tuple, Optional
import logging
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import h5py
import uuid
import traceback
import warnings

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Ignoriere TensorFlow und PyTorch Warnungen
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger("OceanData.AI")

###########################################
# 1. Anomalieerkennung & Outlier Detection
###########################################

class AnomalyDetector:
    """Klasse für die Erkennung von Anomalien in verschiedenen Datentypen"""
    
    def __init__(self, method: str = 'autoencoder', contamination: float = 0.05):
        """
        Initialisiert einen Anomaliedetektor.
        
        Args:
            method: Methode für die Anomalieerkennung ('autoencoder', 'isolation_forest', 'dbscan')
            contamination: Erwarteter Anteil an Anomalien in den Daten (für Isolation Forest)
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dims = None
        self.threshold = None
        
    def _build_autoencoder(self, input_dim):
        """Erstellt ein Autoencoder-Modell für die Anomalieerkennung"""
        encoding_dim = max(1, input_dim // 2)
        hidden_dim = max(1, encoding_dim // 2)
        
        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(inputs)
        encoded = layers.Dense(hidden_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Encoder model for extracting features
        encoder = keras.Model(inputs, encoded)
        
        return autoencoder, encoder
    
    def fit(self, X: pd.DataFrame, categorical_cols: List[str] = None):
        """
        Trainiert den Anomaliedetektor mit Daten.
        
        Args:
            X: Eingabedaten (DataFrame oder numpy array)
            categorical_cols: Liste der kategorischen Spalten, die One-Hot-kodiert werden müssen
        """
        try:
            # Vorbereiten der Daten
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten durch One-Hot-Encoding
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Konvertieren zu float und Behandlung von NaN-Werten
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            
            if X_prep.shape[1] == 0:
                raise ValueError("Nach der Vorverarbeitung sind keine numerischen Merkmale übrig")
                
            X_scaled = self.scaler.fit_transform(X_prep)
            self.feature_dims = X_scaled.shape[1]
            
            if self.method == 'autoencoder':
                # Autoencoder-Ansatz für Anomalieerkennung
                self.model, self.encoder = self._build_autoencoder(self.feature_dims)
                
                # Trainiere den Autoencoder
                self.model.fit(
                    X_scaled, X_scaled,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    verbose=0,
                    validation_split=0.1
                )
                
                # Berechne Rekonstruktionsfehler für alle Trainingsdaten
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Setze Threshold basierend auf den Rekonstruktionsfehlern
                self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
                
            elif self.method == 'isolation_forest':
                # Isolation Forest für Outlier-Erkennung
                self.model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
                self.model.fit(X_scaled)
                
            elif self.method == 'dbscan':
                # DBSCAN für Clustering-basierte Outlier-Erkennung
                self.model = DBSCAN(
                    eps=0.5,  # Maximum distance between samples
                    min_samples=5,  # Minimum samples in a cluster
                    n_jobs=-1
                )
                self.model.fit(X_scaled)
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des Anomaliedetektors: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X: pd.DataFrame, categorical_cols: List[str] = None) -> np.ndarray:
        """
        Identifiziert Anomalien in den Daten.
        
        Args:
            X: Zu prüfende Daten
            categorical_cols: Liste der kategorischen Spalten
            
        Returns:
            Array mit Anomalie-Scores für jede Zeile (-1 für Anomalien, 1 für normale Daten bei Isolation Forest;
            True für Anomalien, False für normale Daten bei Autoencoder)
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Daten
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Konvertieren und Behandlung von NaN-Werten
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            
            # Anpassen der Features, falls nötig
            missing_cols = set(range(self.feature_dims)) - set(range(X_prep.shape[1]))
            if missing_cols:
                for col in missing_cols:
                    X_prep[f'missing_{col}'] = 0
            
            X_prep = X_prep.iloc[:, :self.feature_dims]
            X_scaled = self.scaler.transform(X_prep)
            
            if self.method == 'autoencoder':
                # Berechne Rekonstruktionsfehler
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Anomalie, wenn Fehler über dem Threshold
                return reconstruction_errors > self.threshold
                
            elif self.method == 'isolation_forest':
                # -1 für Anomalien, 1 für normale Daten
                return self.model.predict(X_scaled)
                
            elif self.method == 'dbscan':
                # Ordne neue Punkte den nächsten Clustern zu
                labels = self.model.fit_predict(X_scaled)
                # -1 sind Noise-Punkte (Anomalien)
                return labels == -1
                
        except Exception as e:
            logger.error(f"Fehler bei der Anomalievorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_anomaly_insights(self, X: pd.DataFrame, predictions: np.ndarray, 
                           categorical_cols: List[str] = None, top_n_features: int = 3) -> List[Dict]:
        """
        Liefert Erklärungen für die erkannten Anomalien.
        
        Args:
            X: Originaldaten
            predictions: Ergebnisse der predict()-Methode
            categorical_cols: Liste der kategorischen Spalten
            top_n_features: Anzahl der wichtigsten Features, die zurückgegeben werden sollen
            
        Returns:
            Liste mit Erklärungen für jede Anomalie
        """
        insights = []
        
        try:
            anomaly_indices = np.where(predictions == -1)[0] if self.method == 'isolation_forest' else np.where(predictions)[0]
            
            if len(anomaly_indices) == 0:
                return []
            
            X_prep = X.copy()
            X_orig = X.copy()  # Original-Daten für Berichte
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            X_scaled = self.scaler.transform(X_prep)
            
            # Allgemeine Statistiken berechnen
            mean_values = np.mean(X_scaled, axis=0)
            std_values = np.std(X_scaled, axis=0)
            
            for idx in anomaly_indices:
                if idx >= len(X_scaled):
                    continue
                    
                sample = X_scaled[idx]
                orig_sample = X_orig.iloc[idx]
                
                if self.method == 'autoencoder':
                    # Rekonstruktionsfehler pro Feature berechnen
                    reconstruction = self.model.predict(sample.reshape(1, -1))[0]
                    feature_errors = np.square(sample - reconstruction)
                    
                    # Features mit den größten Fehlern identifizieren
                    top_features_idx = np.argsort(feature_errors)[-top_n_features:]
                    top_features = [(X_prep.columns[i], feature_errors[i], sample[i], reconstruction[i]) for i in top_features_idx]
                    
                    insight = {
                        "index": int(idx),
                        "anomaly_score": float(np.mean(feature_errors)),
                        "important_features": [
                            {
                                "feature": feat,
                                "error": float(err),
                                "actual_value": float(val),
                                "expected_value": float(rec)
                            } for feat, err, val, rec in top_features
                        ],
                        "original_data": orig_sample.to_dict()
                    }
                    
                else:  # Isolation Forest oder DBSCAN
                    # Z-Scores berechnen, um ungewöhnliche Werte zu identifizieren
                    z_scores = (sample - mean_values) / std_values
                    
                    # Features mit den höchsten Z-Scores (positiv oder negativ)
                    abs_z_scores = np.abs(z_scores)
                    top_features_idx = np.argsort(abs_z_scores)[-top_n_features:]
                    top_features = [(X_prep.columns[i], z_scores[i], sample[i], mean_values[i]) for i in top_features_idx]
                    
                    insight = {
                        "index": int(idx),
                        "anomaly_score": float(np.max(abs_z_scores)),
                        "important_features": [
                            {
                                "feature": feat,
                                "z_score": float(z),
                                "actual_value": float(val),
                                "average_value": float(avg)
                            } for feat, z, val, avg in top_features
                        ],
                        "original_data": orig_sample.to_dict()
                    }
                
                insights.append(insight)
                
            return insights
            
        except Exception as e:
            logger.error(f"Fehler bei der Generierung von Anomalie-Insights: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def visualize_anomalies(self, X: pd.DataFrame, predictions: np.ndarray, 
                            categorical_cols: List[str] = None, 
                            save_path: str = None) -> plt.Figure:
        """
        Visualisiert erkannte Anomalien in den Daten.
        
        Args:
            X: Daten
            predictions: Anomalieerkennungsergebnisse
            categorical_cols: Kategorische Spalten
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        try:
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            X_scaled = self.scaler.transform(X_prep)
            
            # PCA für Dimensionsreduktion
            if X_scaled.shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
            else:
                X_pca = X_scaled
            
            # Anomalien bestimmen
            if self.method == 'isolation_forest':
                anomalies = (predictions == -1)
            else:
                anomalies = predictions
            
            # Visualisierung erstellen
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Normale Datenpunkte plotten
            ax.scatter(X_pca[~anomalies, 0], X_pca[~anomalies, 1], c='blue', label='Normal', alpha=0.5)
            
            # Anomalien plotten
            if np.any(anomalies):
                ax.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], c='red', label='Anomalien', alpha=0.7)
            
            # Grafik anpassen
            ax.set_title(f'Anomalieerkennung mit {self.method.capitalize()}')
            ax.set_xlabel('Hauptkomponente 1')
            ax.set_ylabel('Hauptkomponente 2')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung von Anomalien: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden")
            
        model_data = {
            "method": self.method,
            "contamination": self.contamination,
            "feature_dims": self.feature_dims,
            "threshold": self.threshold,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere das Model-Objekt
        if self.method in ['isolation_forest', 'dbscan']:
            joblib.dump(self.model, f"{path}_model.joblib")
        elif self.method == 'autoencoder':
            self.model.save(f"{path}_autoencoder")
            self.encoder.save(f"{path}_encoder")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        detector = cls(
            method=model_data['method'],
            contamination=model_data['contamination']
        )
        
        detector.feature_dims = model_data['feature_dims']
        detector.threshold = model_data['threshold']
        detector.is_fitted = model_data['is_fitted']
        
        # Lade das Model-Objekt
        if detector.method in ['isolation_forest', 'dbscan']:
            detector.model = joblib.load(f"{path}_model.joblib")
        elif detector.method == 'autoencoder':
            detector.model = keras.models.load_model(f"{path}_autoencoder")
            detector.encoder = keras.models.load_model(f"{path}_encoder")
        
        # Lade den Scaler
        detector.scaler = joblib.load(f"{path}_scaler.joblib")
        
        return detector


###########################################
# 2. Semantische Datenanalyse
###########################################

class SemanticAnalyzer:
    """Klasse für semantische Analyse von Text und anderen Daten mit Deep Learning"""
    
    def __init__(self, model_type: str = 'bert', model_name: str = 'bert-base-uncased'):
        """
        Initialisiert den semantischen Analysator.
        
        Args:
            model_type: Typ des zu verwendenden Modells ('bert', 'gpt2', 'custom')
            model_name: Name oder Pfad des vortrainierten Modells
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.max_length = 512  # Standard für BERT
        self.embeddings_cache = {}  # Cache für Texteinbettungen
        
        # Modell und Tokenizer laden
        self._load_model()
    
    def _load_model(self):
        """Lädt das Modell und den Tokenizer basierend auf dem ausgewählten Typ"""
        try:
            if self.model_type == 'bert':
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                if tf.test.is_gpu_available():
                    self.model = TFBertModel.from_pretrained(self.model_name)
                else:
                    self.model = BertModel.from_pretrained(self.model_name)
                self.max_length = 512
            
            elif self.model_type == 'gpt2':
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 hat kein Padding-Token
                if tf.test.is_gpu_available():
                    self.model = TFGPT2Model.from_pretrained(self.model_name)
                else:
                    self.model = GPT2Model.from_pretrained(self.model_name)
                self.max_length = 1024
            
            else:
                raise ValueError(f"Nicht unterstützter Modelltyp: {self.model_type}")
        
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells {self.model_name}: {str(e)}")
            # Fallback zu kleineren Modellen bei Speicher- oder Download-Problemen
            if self.model_type == 'bert':
                logger.info("Verwende ein kleineres BERT-Modell als Fallback")
                self.model_name = 'distilbert-base-uncased'
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertModel.from_pretrained(self.model_name)
            elif self.model_type == 'gpt2':
                logger.info("Verwende ein kleineres GPT-2-Modell als Fallback")
                self.model_name = 'distilgpt2'
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = GPT2Model.from_pretrained(self.model_name)
    
    def get_embeddings(self, texts: Union[str, List[str]], 
                      batch_size: int = 8, use_cache: bool = True) -> np.ndarray:
        """
        Erzeugt Einbettungen (Embeddings) für Texte.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            batch_size: Größe der Batches für die Verarbeitung
            use_cache: Ob bereits berechnete Einbettungen wiederverwendet werden sollen
            
        Returns:
            Array mit Einbettungen (shape: [n_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Initialisiere ein Array für die Ergebnisse
        all_embeddings = []
        texts_to_process = []
        texts_indices = []
        
        # Prüfe Cache für jede Anfrage
        for i, text in enumerate(texts):
            if use_cache and text in self.embeddings_cache:
                all_embeddings.append(self.embeddings_cache[text])
            else:
                texts_to_process.append(text)
                texts_indices.append(i)
        
        if texts_to_process:
            # Verarbeite Texte in Batches
            for i in range(0, len(texts_to_process), batch_size):
                batch_texts = texts_to_process[i:i+batch_size]
                batch_indices = texts_indices[i:i+batch_size]
                
                # Tokenisierung
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt" if isinstance(self.model, nn.Module) else "tf",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Modelloutput berechnen
                with torch.no_grad() if isinstance(self.model, nn.Module) else tf.device('/CPU:0'):
                    outputs = self.model(**inputs)
                
                # Embeddings aus dem letzten Hidden State extrahieren
                if self.model_type == 'bert':
                    # Verwende [CLS]-Token-Ausgabe als Satzrepräsentation (erstes Token)
                    if isinstance(self.model, nn.Module):
                        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                    else:
                        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                elif self.model_type == 'gpt2':
                    # Verwende den Durchschnitt aller Token-Repräsentationen
                    if isinstance(self.model, nn.Module):
                        embeddings = torch.mean(outputs.last_hidden_state, dim=1).numpy()
                    else:
                        embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
                
                # Füge die Embeddings an der richtigen Position ein
                for j, (idx, text, embedding) in enumerate(zip(batch_indices, batch_texts, embeddings)):
                    # Zum Cache hinzufügen
                    if use_cache:
                        self.embeddings_cache[text] = embedding
                    
                    # Aktualisiere Ergebnisarray an der richtigen Position
                    if idx >= len(all_embeddings):
                        all_embeddings.extend([None] * (idx - len(all_embeddings) + 1))
                    all_embeddings[idx] = embedding
        
        # Konvertiere zu NumPy-Array
        return np.vstack(all_embeddings)
    
    def analyze_sentiment(self, texts: Union[str, List[str]]) -> List[Dict]:
        """
        Führt eine Stimmungsanalyse für Texte durch.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            
        Returns:
            Liste mit Sentiment-Analysen für jeden Text (positive, negative, neutral)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Verwende NLTK für grundlegende Sentimentanalyse
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            
            analyzer = SentimentIntensityAnalyzer()
            results = []
            
            for text in texts:
                scores = analyzer.polarity_scores(text)
                
                # Bestimme die dominante Stimmung
                if scores['compound'] >= 0.05:
                    sentiment = 'positive'
                elif scores['compound'] <= -0.05:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': sentiment,
                    'scores': {
                        'positive': scores['pos'],
                        'negative': scores['neg'],
                        'neutral': scores['neu'],
                        'compound': scores['compound']
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Fehler bei der Sentimentanalyse: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback zu einem einfacheren Ansatz
            return [{'text': t[:100] + '...' if len(t) > 100 else t, 
                    'sentiment': 'unknown', 
                    'scores': {'positive': 0, 'negative': 0, 'neutral': 0, 'compound': 0}} 
                    for t in texts]
    
    def extract_topics(self, texts: Union[str, List[str]], num_topics: int = 5, 
                       words_per_topic: int = 5) -> List[Dict]:
        """
        Extrahiert Themen (Topics) aus Texten.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            num_topics: Anzahl der zu extrahierenden Themen
            words_per_topic: Anzahl der Wörter pro Thema
            
        Returns:
            Liste mit Themen und zugehörigen Top-Wörtern
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Tokenisiere und bereinige die Texte
            try:
                nltk.data.find('stopwords')
                nltk.data.find('punkt')
            except LookupError:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
            
            stop_words = set(stopwords.words('english'))
            
            # Texte vorverarbeiten
            processed_texts = []
            for text in texts:
                # Tokenisieren und Stopwords entfernen
                tokens = [w.lower() for w in word_tokenize(text) 
                         if w.isalpha() and w.lower() not in stop_words]
                processed_texts.append(' '.join(tokens))
            
            # Verwende Transformers für Themenmodellierung
            embeddings = self.get_embeddings(processed_texts)
            
            # Verwende K-Means-Clustering auf Embeddings
            kmeans = KMeans(n_clusters=min(num_topics, len(processed_texts)), random_state=42)
            kmeans.fit(embeddings)
            
            # Finde repräsentative Wörter für jedes Cluster
            topics = []
            
            # Alle Wörter aus allen Texten zusammenfassen
            all_words = []
            for text in processed_texts:
                all_words.extend(text.split())
            
            # Eindeutige Wörter
            unique_words = list(set(all_words))
            
            # Für jedes Wort ein Embedding berechnen
            if len(unique_words) > 0:
                word_embeddings = self.get_embeddings(unique_words)
                
                # Für jedes Cluster die nächsten Wörter bestimmen, die dem Clusterzentrum am nächsten sind
                for cluster_idx in range(kmeans.n_clusters):
                    center = kmeans.cluster_centers_[cluster_idx]
                    
                    # Berechne Distanzen zwischen Zentrum und Wort-Embeddings
                    distances = np.linalg.norm(word_embeddings - center, axis=1)
                    
                    # Finde die nächsten Wörter
                    closest_indices = np.argsort(distances)[:words_per_topic]
                    top_words = [unique_words[i] for i in closest_indices]
                    
                    # Beispieltexte für dieses Cluster finden
                    cluster_texts = [texts[i][:100] + "..." 
                                    for i, label in enumerate(kmeans.labels_) 
                                    if label == cluster_idx][:3]  # Maximal 3 Beispiele
                    
                    topic = {
                        "id": cluster_idx,
                        "words": top_words,
                        "examples": cluster_texts
                    }
                    topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Fehler bei der Themenextraktion: {str(e)}")
            logger.error(traceback.format_exc())
            return [{"id": 0, "words": ["error", "processing", "topics"], "examples": []}]
    
    def find_similar_texts(self, query: str, corpus: List[str], top_n: int = 5) -> List[Dict]:
        """
        Findet ähnliche Texte zu einer Anfrage in einem Korpus.
        
        Args:
            query: Anfrage-Text
            corpus: Liste von Texten, in denen gesucht werden soll
            top_n: Anzahl der zurückzugebenden ähnlichsten Texte
            
        Returns:
            Liste der ähnlichsten Texte mit Ähnlichkeitswerten
        """
        try:
            # Einbettungen für Anfrage und Korpus erzeugen
            query_embedding = self.get_embeddings(query).reshape(1, -1)
            corpus_embeddings = self.get_embeddings(corpus)
            
            # Kosinus-Ähnlichkeiten berechnen
            similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
            
            # Top-N ähnlichste Texte finden
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
            
        except Exception as e:
            logger.error(f"Fehler beim Finden ähnlicher Texte: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def generate_text_summary(self, text: str, max_length: int = 150) -> str:
        """
        Erzeugt eine Zusammenfassung eines längeren Textes.
        
        Args:
            text: Text, der zusammengefasst werden soll
            max_length: Maximale Länge der Zusammenfassung in Zeichen
            
        Returns:
            Zusammenfassung des Textes
        """
        try:
            # Vorverarbeitung des Textes
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) <= 1:
                return text[:max_length] + "..." if len(text) > max_length else text
            
            # Einbettungen für alle Sätze erzeugen
            sentence_embeddings = self.get_embeddings(sentences)
            
            # Durchschnittliche Einbettung berechnen (repräsentiert den Gesamttext)
            mean_embedding = np.mean(sentence_embeddings, axis=0).reshape(1, -1)
            
            # Ähnlichkeit jedes Satzes zum Durchschnitt berechnen
            similarities = np.dot(sentence_embeddings, mean_embedding.T).flatten()
            
            # Sätze nach Ähnlichkeit sortieren
            ranked_sentences = [(sentences[i], float(similarities[i])) for i in range(len(sentences))]
            ranked_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Top-Sätze auswählen, bis max_length erreicht ist
            summary = ""
            for sentence, _ in ranked_sentences:
                if len(summary) + len(sentence) <= max_length:
                    summary += sentence + " "
                else:
                    break
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Fehler bei der Textzusammenfassung: {str(e)}")
            logger.error(traceback.format_exc())
            return text[:max_length] + "..." if len(text) > max_length else text

###########################################
# 3. Prädiktive Modellierung
###########################################

class PredictiveModeler:
    """
    Klasse für die Entwicklung von prädiktiven Modellen, die verschiedene Datentypen
    verarbeiten und Vorhersagen treffen können.
    """
    
    def __init__(self, model_type: str = 'lstm', forecast_horizon: int = 7):
        """
        Initialisiert den Prädiktiven Modellierer.
        
        Args:
            model_type: Typ des zu verwendenden Modells ('lstm', 'transformer', 'gru', 'arima')
            forecast_horizon: Anzahl der Zeitschritte für Vorhersagen in die Zukunft
        """
        self.model_type = model_type.lower()
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.lookback = 10  # Standardwert für die Anzahl der zurückliegenden Zeitschritte
        self.feature_dims = None
        self.target_dims = None
        self.target_scaler = None
        self.history = None
        
    def _build_lstm_model(self, input_shape, output_dim):
        """Erstellt ein LSTM-Modell für Zeitreihenvorhersage"""
        model = keras.Sequential()
        model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(32))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(output_dim))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _build_transformer_model(self, input_shape, output_dim):
        """Erstellt ein Transformer-Modell für Zeitreihenvorhersage"""
        # Einfaches Transformer-Modell für Zeitreihen
        inputs = keras.Input(shape=input_shape)
        
        # Positional encoding layer
        class PositionalEncoding(layers.Layer):
            def __init__(self, position, d_model):
                super(PositionalEncoding, self).__init__()
                self.pos_encoding = self.positional_encoding(position, d_model)
                
            def get_angles(self, position, i, d_model):
                angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
                return position * angles
            
            def positional_encoding(self, position, d_model):
                angle_rads = self.get_angles(
                    position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                    i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                    d_model=d_model
                )
                
                # Apply sine to even indices
                sines = tf.math.sin(angle_rads[:, 0::2])
                # Apply cosine to odd indices
                cosines = tf.math.cos(angle_rads[:, 1::2])
                
                pos_encoding = tf.concat([sines, cosines], axis=-1)
                pos_encoding = pos_encoding[tf.newaxis, ...]
                
                return tf.cast(pos_encoding, tf.float32)
            
            def call(self, inputs):
                return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        
        x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
        
        # Multi-head attention layer
        x = layers.MultiHeadAttention(
            key_dim=input_shape[1], num_heads=4, dropout=0.1
        )(x, x, x, attention_mask=None, training=True)
        
        # Feed-forward network
        x = layers.Dropout(0.1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=1, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv1D(filters=input_shape[1], kernel_size=1)(x)
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Output layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(output_dim)(x)
        
        model = keras.Model(inputs, x)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def _build_gru_model(self, input_shape, output_dim):
        """Erstellt ein GRU-Modell für Zeitreihenvorhersage"""
        model = keras.Sequential()
        model.add(layers.GRU(64, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.GRU(32))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(output_dim))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _create_sequences(self, data, target=None, lookback=None):
        """
        Erstellt Sequenzen für die Zeitreihenmodellierung
        
        Args:
            data: Eingabedaten (numpy array)
            target: Zielvariablen (optional, numpy array)
            lookback: Anzahl der zurückliegenden Zeitschritte (optional)
            
        Returns:
            X: Sequenzen für die Eingabe
            y: Zielwerte (wenn target bereitgestellt wird)
        """
        if lookback is None:
            lookback = self.lookback
        
        X = []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
        X = np.array(X)
        
        if target is not None:
            y = target[lookback:]
            return X, y
        
        return X
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, 
            lookback: int = 10, epochs: int = 50, 
            validation_split: float = 0.2, batch_size: int = 32,
            verbose: int = 1):
        """
        Trainiert das prädiktive Modell mit den gegebenen Daten.
        
        Args:
            X: Eingabedaten (DataFrame)
            y: Zielvariablen (DataFrame, optional für Zeitreihen)
            lookback: Anzahl der zurückliegenden Zeitschritte für Zeitreihenmodelle
            epochs: Anzahl der Trainings-Epochen
            validation_split: Anteil der Daten für die Validierung
            batch_size: Batch-Größe für das Training
            verbose: Ausgabedetailstufe (0, 1, oder 2)
            
        Returns:
            self: Trainiertes Modell
        """
        try:
            self.lookback = lookback
            
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            # Skalierung der Eingabedaten
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_values)
            
            # Vorbereiten der Zielvariablen
            if y is not None:
                if isinstance(y, pd.DataFrame):
                    y_values = y.values
                else:
                    y_values = y
                    
                self.target_scaler = StandardScaler()
                y_scaled = self.target_scaler.fit_transform(y_values)
                self.target_dims = y_scaled.shape[1]
            else:
                # Wenn keine Zielvariablen bereitgestellt werden, nehmen wir an, dass X selbst eine Zeitreihe ist
                y_scaled = X_scaled
                self.target_dims = X_scaled.shape[1]
            
            self.feature_dims = X_scaled.shape[1]
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, lookback)
            
            # Modell basierend auf dem ausgewählten Typ erstellen
            input_shape = (lookback, self.feature_dims)
            output_dim = self.target_dims * self.forecast_horizon
            
            if self.model_type == 'lstm':
                self.model = self._build_lstm_model(input_shape, output_dim)
            elif self.model_type == 'transformer':
                self.model = self._build_transformer_model(input_shape, output_dim)
            elif self.model_type == 'gru':
                self.model = self._build_gru_model(input_shape, output_dim)
            else:
                raise ValueError(f"Nicht unterstützter Modelltyp: {self.model_type}")
            
            # Reshape y_sequences für das Forecast-Horizon
            y_prepared = y_sequences.reshape(y_sequences.shape[0], -1)
            
            # Modell trainieren
            self.history = self.model.fit(
                X_sequences, y_prepared,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des prädiktiven Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X: pd.DataFrame, return_sequences: bool = False) -> np.ndarray:
        """
        Macht Vorhersagen mit dem trainierten Modell.
        
        Args:
            X: Eingabedaten (DataFrame)
            return_sequences: Ob die Vorhersagesequenz zurückgegeben werden soll
            
        Returns:
            Vorhersagen für die Eingabedaten
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_values)
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences = self._create_sequences(X_scaled, lookback=self.lookback)
            
            # Vorhersagen machen
            predictions_scaled = self.model.predict(X_sequences)
            
            # Reshape für die Ausgabe
            predictions_scaled = predictions_scaled.reshape(
                predictions_scaled.shape[0], 
                self.forecast_horizon, 
                self.target_dims
            )
            
            # Rücktransformation
            predictions = np.zeros_like(predictions_scaled)
            for i in range(self.forecast_horizon):
                step_predictions = predictions_scaled[:, i, :]
                # Rücktransformation nur für jeden Zeitschritt
                predictions[:, i, :] = self.target_scaler.inverse_transform(step_predictions)
            
            if return_sequences:
                return predictions
            else:
                # Nur den ersten Vorhersageschritt zurückgeben
                return predictions[:, 0, :]
            
        except Exception as e:
            logger.error(f"Fehler bei der Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def forecast(self, X: pd.DataFrame, steps: int = None) -> np.ndarray:
        """
        Erstellt eine Vorhersage für mehrere Zeitschritte in die Zukunft.
        
        Args:
            X: Letzte bekannte Datenpunkte (mindestens lookback viele)
            steps: Anzahl der vorherzusagenden Schritte (Standard: forecast_horizon)
            
        Returns:
            Vorhersagesequenz für die nächsten 'steps' Zeitschritte
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        if steps is None:
            steps = self.forecast_horizon
            
        try:
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
            
            if len(X_values) < self.lookback:
                raise ValueError(f"Eingabedaten müssen mindestens {self.lookback} Zeitschritte enthalten")
            
            # Verwende nur die letzten 'lookback' Zeitschritte
            X_recent = X_values[-self.lookback:]
            
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_recent)
            X_sequence = X_scaled.reshape(1, self.lookback, self.feature_dims)
            
            # Erstelle die multi-step-Vorhersage
            forecast_values = []
            
            current_sequence = X_sequence.copy()
            
            for _ in range(steps):
                # Mache eine Vorhersage für den nächsten Schritt
                next_step_scaled = self.model.predict(current_sequence)[0]
                next_step_scaled = next_step_scaled.reshape(1, self.target_dims)
                
                # Rücktransformation
                next_step = self.target_scaler.inverse_transform(next_step_scaled)
                forecast_values.append(next_step[0])
                
                # Aktualisiere die Eingabesequenz für den nächsten Schritt
                # Entferne den ersten Zeitschritt und füge den neu vorhergesagten hinzu
                new_sequence = np.zeros_like(current_sequence)
                new_sequence[0, :-1, :] = current_sequence[0, 1:, :]
                new_sequence[0, -1, :] = next_step_scaled
                current_sequence = new_sequence
            
            return np.array(forecast_values)
            
        except Exception as e:
            logger.error(f"Fehler bei der Mehrschritt-Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluiert das Modell mit Testdaten.
        
        Args:
            X_test: Test-Eingabedaten
            y_test: Test-Zielvariablen
            
        Returns:
            Dictionary mit Bewertungsmetriken
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Testdaten
            if isinstance(X_test, pd.DataFrame):
                X_values = X_test.values
            else:
                X_values = X_test
                
            if isinstance(y_test, pd.DataFrame):
                y_values = y_test.values
            else:
                y_values = y_test
            
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_values)
            y_scaled = self.target_scaler.transform(y_values)
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, self.lookback)
            
            # Reshape y_sequences für das Forecast-Horizon
            y_prepared = y_sequences.reshape(y_sequences.shape[0], -1)
            
            # Modell evaluieren
            evaluation = self.model.evaluate(X_sequences, y_prepared, verbose=0)
            
            # Vorhersagen machen für detailliertere Metriken
            predictions = self.predict(X_test)
            
            # Tatsächliche Werte (ohne die ersten lookback Zeitschritte)
            actuals = y_values[self.lookback:]
            
            # Berechne RMSE, MAE, MAPE
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Entsprechende Anzahl an Vorhersagen auswählen
            predictions = predictions[:len(actuals)]
            
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # MAPE (Mean Absolute Percentage Error)
            # Vermeide Division durch Null
            mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
            
            return {
                'loss': evaluation[0],
                'mae': evaluation[1],
                'rmse': rmse,
                'mean_absolute_error': mae,
                'mean_absolute_percentage_error': mape
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Modellbewertung: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def plot_forecast(self, X: pd.DataFrame, y_true: pd.DataFrame = None, 
                     steps: int = None, feature_idx: int = 0,
                     save_path: str = None) -> plt.Figure:
        """
        Visualisiert die Vorhersage des Modells.
        
        Args:
            X: Eingabedaten
            y_true: Tatsächliche zukünftige Werte (optional)
            steps: Anzahl der vorherzusagenden Schritte
            feature_idx: Index der darzustellenden Feature-Dimension
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Standard-Schritte
            if steps is None:
                steps = self.forecast_horizon
            
            # Vorhersage erstellen
            forecast_values = self.forecast(X, steps)
            
            # Historische Daten (letzte lookback Zeitschritte)
            if isinstance(X, pd.DataFrame):
                historical_values = X.values[-self.lookback:, feature_idx]
            else:
                historical_values = X[-self.lookback:, feature_idx]
            
            # Zeitachse erstellen
            time_hist = np.arange(-self.lookback, 0)
            time_future = np.arange(0, steps)
            
            # Visualisierung erstellen
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historische Daten plotten
            ax.plot(time_hist, historical_values, 'b-', label='Historische Daten')
            
            # Vorhersage plotten
            ax.plot(time_future, forecast_values[:, feature_idx], 'r-', label='Vorhersage')
            
            # Tatsächliche zukünftige Werte plotten, falls vorhanden
            if y_true is not None:
                if isinstance(y_true, pd.DataFrame):
                    true_future = y_true.values[:steps, feature_idx]
                else:
                    true_future = y_true[:steps, feature_idx]
                
                ax.plot(time_future[:len(true_future)], true_future, 'g-', label='Tatsächliche Werte')
            
            # Grafik anpassen
            ax.set_title(f'Zeitreihenvorhersage mit {self.model_type.upper()}')
            ax.set_xlabel('Zeitschritte')
            ax.set_ylabel(f'Feature {feature_idx}')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Trennlinie zwischen historischen und Vorhersagedaten
            ax.axvline(x=0, color='k', linestyle='--')
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung der Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        model_data = {
            "model_type": self.model_type,
            "forecast_horizon": self.forecast_horizon,
            "lookback": self.lookback,
            "feature_dims": self.feature_dims,
            "target_dims": self.target_dims,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere das Model-Objekt
        self.model.save(f"{path}_model")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        joblib.dump(self.target_scaler, f"{path}_target_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        predictor = cls(
            model_type=model_data['model_type'],
            forecast_horizon=model_data['forecast_horizon']
        )
        
        predictor.lookback = model_data['lookback']
        predictor.feature_dims = model_data['feature_dims']
        predictor.target_dims = model_data['target_dims']
        predictor.is_fitted = model_data['is_fitted']
        
        # Lade das Model-Objekt
        predictor.model = keras.models.load_model(f"{path}_model")
        
        # Lade die Scaler
        predictor.scaler = joblib.load(f"{path}_scaler.joblib")
        predictor.target_scaler = joblib.load(f"{path}_target_scaler.joblib")
        
        return predictor

###########################################
# 4. Datensynthese und GAN-basierte Modelle
###########################################

class DataSynthesizer:
    """
    Klasse zur Generierung synthetischer Daten basierend auf realen Beispielen.
    Verwendet GAN (Generative Adversarial Network) für realistische Datensynthese.
    """
    
    def __init__(self, categorical_threshold: int = 10, noise_dim: int = 100):
        """
        Initialisiert den Datensynthetisierer.
        
        Args:
            categorical_threshold: Anzahl eindeutiger Werte, ab der eine Spalte als kategorisch gilt
            noise_dim: Dimension des Rauschvektors für den Generator
        """
        self.categorical_threshold = categorical_threshold
        self.noise_dim = noise_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        
        self.column_types = {}  # Speichert, ob eine Spalte kategorisch oder kontinuierlich ist
        self.categorical_mappings = {}  # Speichert Mappings für kategorische Spalten
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # Für kontinuierliche Variablen
        
        self.is_fitted = False
        self.feature_dims = None
        self.training_data = None
    
    def _identify_column_types(self, data: pd.DataFrame):
        """Identifiziert, ob Spalten kategorisch oder kontinuierlich sind"""
        self.column_types = {}
        
        for col in data.columns:
            n_unique = data[col].nunique()
            
            # Wenn die Anzahl eindeutiger Werte kleiner als der Schwellenwert ist oder
            # der Datentyp ist nicht numerisch, behandle die Spalte als kategorisch
            if n_unique < self.categorical_threshold or not pd.api.types.is_numeric_dtype(data[col]):
                self.column_types[col] = 'categorical'
                
                # Erstelle Mapping von Kategorien zu Zahlen
                categories = data[col].unique()
                self.categorical_mappings[col] = {
                    cat: i for i, cat in enumerate(categories)
                }
                # Umgekehrtes Mapping für die Rücktransformation
                self.categorical_mappings[f"{col}_reverse"] = {
                    i: cat for i, cat in enumerate(categories)
                }
            else:
                self.column_types[col] = 'continuous'
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Vorverarbeitung der Daten für das GAN"""
        processed_data = pd.DataFrame()
        
        for col in data.columns:
            if self.column_types[col] == 'categorical':
                # One-Hot-Encoding für kategorische Spalten
                mapped_col = data[col].map(self.categorical_mappings[col])
                one_hot = pd.get_dummies(mapped_col, prefix=col)
                processed_data = pd.concat([processed_data, one_hot], axis=1)
            else:
                # Skalierung für kontinuierliche Spalten
                processed_data[col] = data[col]
        
        # Skaliere alle Spalten auf [-1, 1]
        return self.scaler.fit_transform(processed_data)
    
    def _postprocess_data(self, generated_data: np.ndarray) -> pd.DataFrame:
        """Nachverarbeitung der generierten Daten zurück in das ursprüngliche Format"""
        # Rücktransformation der Skalierung
        rescaled_data = self.scaler.inverse_transform(generated_data)
        
        # Erstelle einen DataFrame mit den ursprünglichen Spalten
        result = pd.DataFrame()
        
        col_idx = 0
        for col, col_type in self.column_types.items():
            if col_type == 'categorical':
                # Anzahl der eindeutigen Werte für diese kategorische Spalte
                n_categories = len(self.categorical_mappings[col])
                
                # Extrahiere die One-Hot-kodierten Werte
                cat_values = rescaled_data[:, col_idx:col_idx+n_categories]
                
                # Konvertiere von One-Hot zurück zu kategorischen Werten
                # Nehme die Kategorie mit dem höchsten Wert
                cat_indices = np.argmax(cat_values, axis=1)
                
                # Mappe zurück zu den ursprünglichen Kategorien
                result[col] = [self.categorical_mappings[f"{col}_reverse"][idx] for idx in cat_indices]
                
                col_idx += n_categories
            else:
                # Kontinuierliche Spalte einfach übernehmen
                result[col] = rescaled_data[:, col_idx]
                col_idx += 1
        
        return result
    
    def _build_generator(self, output_dim):
        """Erstellt den Generator für das GAN"""
        model = keras.Sequential([
            layers.Dense(256, input_dim=self.noise_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(output_dim, activation='tanh')  # tanh für Output im Bereich [-1, 1]
        ])
        return model
    
    def _build_discriminator(self, input_dim):
        """Erstellt den Diskriminator für das GAN"""
        model = keras.Sequential([
            layers.Dense(512, input_dim=input_dim, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy')
        return model
    
    def _build_gan(self, generator, discriminator):
        """Kombiniert Generator und Diskriminator zum GAN"""
        discriminator.trainable = False  # Diskriminator beim GAN-Training nicht aktualisieren
        
        model = keras.Sequential([
            generator,
            discriminator
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy')
        return model
    
    def fit(self, data: pd.DataFrame, epochs: int = 2000, batch_size: int = 32, 
           sample_interval: int = 100, verbose: int = 1):
        """
        Trainiert das GAN-Modell mit den gegebenen Daten.
        
        Args:
            data: Eingabedaten (DataFrame)
            epochs: Anzahl der Trainings-Epochen
            batch_size: Batch-Größe für das Training
            sample_interval: Intervall für Stichproben der generierten Daten
            verbose: Ausgabedetailstufe (0, 1, oder 2)
            
        Returns:
            self: Trainiertes Modell
        """
        try:
            # Identifiziere Spaltentypen
            self._identify_column_types(data)
            
            # Vorverarbeitung der Daten
            processed_data = self._preprocess_data(data)
            self.feature_dims = processed_data.shape[1]
            
            # Speichere trainierte Daten für spätere Validierung
            self.training_data = data.copy()
            
            # Baue das GAN-Modell
            self.generator = self._build_generator(self.feature_dims)
            self.discriminator = self._build_discriminator(self.feature_dims)
            self.gan = self._build_gan(self.generator, self.discriminator)
            
            # Label für echte und gefälschte Daten
            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            
            # Trainingsschleife
            for epoch in range(epochs):
                # ---------------------
                #  Trainiere Diskriminator
                # ---------------------
                
                # Wähle eine zufällige Batch aus echten Daten
                idx = np.random.randint(0, processed_data.shape[0], batch_size)
                real_data = processed_data[idx]
                
                # Generiere eine Batch aus gefälschten Daten
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                fake_data = self.generator.predict(noise)
                
                # Trainiere den Diskriminator
                d_loss_real = self.discriminator.train_on_batch(real_data, real)
                d_loss_fake = self.discriminator.train_on_batch(fake_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ---------------------
                #  Trainiere Generator
                # ---------------------
                
                # Generiere neue Batch aus Rauschen
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                
                # Trainiere den Generator
                g_loss = self.gan.train_on_batch(noise, real)
                
                # Ausgabe für Fortschrittsüberwachung
                if verbose > 0 and epoch % sample_interval == 0:
                    print(f"Epoch {epoch}/{epochs} [D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}]")
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des GAN-Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def generate(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generiert synthetische Daten.
        
        Args:
            n_samples: Anzahl der zu generierenden Datensätze
            
        Returns:
            DataFrame mit synthetischen Daten im Format der Trainingsdaten
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere Rauschen als Input für den Generator
            noise = np.random.normal(0, 1, (n_samples, self.noise_dim))
            
            # Generiere Daten
            generated_data = self.generator.predict(noise)
            
            # Nachverarbeitung der Daten
            synthetic_data = self._postprocess_data(generated_data)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Fehler bei der Datengenerierung: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate_quality(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Bewertet die Qualität der generierten Daten durch Vergleich mit den Trainingsdaten.
        
        Args:
            n_samples: Anzahl der zu generierenden und bewertenden Datensätze
            
        Returns:
            Dictionary mit Qualitätsmetriken
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere synthetische Daten
            synthetic_data = self.generate(n_samples)
            
            # Statistischer Vergleich zwischen echten und synthetischen Daten
            metrics = {}
            
            # Vergleiche Mittelwerte und Standardabweichungen für kontinuierliche Spalten
            for col, col_type in self.column_types.items():
                if col_type == 'continuous':
                    # Berechne Mittelwert und Standardabweichung für echte Daten
                    real_mean = self.training_data[col].mean()
                    real_std = self.training_data[col].std()
                    
                    # Berechne dieselben Statistiken für synthetische Daten
                    synth_mean = synthetic_data[col].mean()
                    synth_std = synthetic_data[col].std()
                    
                    # Berechne die relative Differenz
                    mean_diff = abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10)
                    std_diff = abs(real_std - synth_std) / (abs(real_std) + 1e-10)
                    
                    metrics[f"{col}_mean_diff"] = float(mean_diff)
                    metrics[f"{col}_std_diff"] = float(std_diff)
                else:
                    # Vergleiche die Verteilung kategorischer Werte
                    real_dist = self.training_data[col].value_counts(normalize=True)
                    synth_dist = synthetic_data[col].value_counts(normalize=True)
                    
                    # Berechne die Jensen-Shannon-Divergenz
                    # (symmetrische Version der KL-Divergenz)
                    js_divergence = 0.0
                    
                    # Stelle sicher, dass beide Verteilungen dieselben Kategorien haben
                    all_categories = set(real_dist.index) | set(synth_dist.index)
                    
                    for cat in all_categories:
                        p = real_dist.get(cat, 0)
                        q = synth_dist.get(cat, 0)
                        
                        # Vermeide Logarithmus von 0
                        if p > 0 and q > 0:
                            m = 0.5 * (p + q)
                            js_divergence += 0.5 * (p * np.log(p / m) + q * np.log(q / m))
                    
                    metrics[f"{col}_js_divergence"] = float(js_divergence)
            
            # Gesamtqualitätsmetrik
            # Durchschnitt der normalisierten Abweichungen (niedriger ist besser)
            continuous_diffs = [v for k, v in metrics.items() if k.endswith('_diff')]
            categorical_diffs = [v for k, v in metrics.items() if k.endswith('_js_divergence')]
            
            if continuous_diffs:
                metrics['continuous_avg_diff'] = float(np.mean(continuous_diffs))
            if categorical_diffs:
                metrics['categorical_avg_diff'] = float(np.mean(categorical_diffs))
            
            # Gesamtbewertung (0 bis 1, höher ist besser)
            overall_score = 1.0
            if continuous_diffs:
                overall_score -= 0.5 * min(1.0, np.mean(continuous_diffs))
            if categorical_diffs:
                overall_score -= 0.5 * min(1.0, np.mean(categorical_diffs))
            
            metrics['overall_quality_score'] = float(overall_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Fehler bei der Qualitätsbewertung: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def plot_comparison(self, n_samples: int = 1000, 
                       features: List[str] = None,
                       save_path: str = None) -> plt.Figure:
        """
        Visualisiert einen Vergleich zwischen echten und synthetischen Daten.
        
        Args:
            n_samples: Anzahl der zu generierenden Datensätze
            features: Liste der darzustellenden Features (Standard: alle)
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere synthetische Daten
            synthetic_data = self.generate(n_samples)
            
            # Wähle die darzustellenden Features aus
            if features is None:
                # Wähle bis zu 6 Features für die Visualisierung
                features = list(self.column_types.keys())[:min(6, len(self.column_types))]
            
            # Bestimme die Anzahl der Zeilen und Spalten für das Subplot-Raster
            n_features = len(features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            # Erstelle die Figur
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
            if n_rows * n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, feature in enumerate(features):
                ax = axes[i]
                
                if self.column_types[feature] == 'continuous':
                    # Histogramm für kontinuierliche Variablen
                    sns.histplot(self.training_data[feature], kde=True, ax=ax, color='blue', alpha=0.5, label='Echte Daten')
                    sns.histplot(synthetic_data[feature], kde=True, ax=ax, color='red', alpha=0.5, label='Synthetische Daten')
                else:
                    # Balkendiagramm für kategorische Variablen
                    real_counts = self.training_data[feature].value_counts(normalize=True)
                    synth_counts = synthetic_data[feature].value_counts(normalize=True)
                    
                    # Kombiniere beide, um alle Kategorien zu erfassen
                    all_cats = sorted(set(real_counts.index) | set(synth_counts.index))
                    
                    # Erstelle ein DataFrame für Seaborn
                    plot_data = []
                    for cat in all_cats:
                        plot_data.append({'Category': cat, 'Frequency': real_counts.get(cat, 0), 'Type': 'Real'})
                        plot_data.append({'Category': cat, 'Frequency': synth_counts.get(cat, 0), 'Type': 'Synthetic'})
                    
                    plot_df = pd.DataFrame(plot_data)
                    
                    # Balkendiagramm
                    sns.barplot(x='Category', y='Frequency', hue='Type', data=plot_df, ax=ax)
                
                ax.set_title(f'Verteilung von {feature}')
                ax.legend()
                
                # Achsen anpassen
                if self.column_types[feature] == 'categorical':
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Verstecke ungenutzte Subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung des Datenvergleichs: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        model_data = {
            "categorical_threshold": self.categorical_threshold,
            "noise_dim": self.noise_dim,
            "feature_dims": self.feature_dims,
            "column_types": self.column_types,
            "categorical_mappings": self.categorical_mappings,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere die Modelle
        self.generator.save(f"{path}_generator")
        self.discriminator.save(f"{path}_discriminator")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten und Mappings
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        synthesizer = cls(
            categorical_threshold=model_data['categorical_threshold'],
            noise_dim=model_data['noise_dim']
        )
        
        synthesizer.feature_dims = model_data['feature_dims']
        synthesizer.column_types = model_data['column_types']
        synthesizer.categorical_mappings = model_data['categorical_mappings']
        synthesizer.is_fitted = model_data['is_fitted']
        
        # Lade die Modelle
        synthesizer.generator = keras.models.load_model(f"{path}_generator")
        synthesizer.discriminator = keras.models.load_model(f"{path}_discriminator")
        
        # Lade den Scaler
        synthesizer.scaler = joblib.load(f"{path}_scaler.joblib")
        
        # Rekonstruiere das GAN
        synthesizer.gan = synthesizer._build_gan(synthesizer.generator, synthesizer.discriminator)
        
        return synthesizer

###########################################
# 5. Federated Learning und Compute-to-Data
###########################################

class ComputeToDataManager:
    """
    Klasse zur Verwaltung von Compute-to-Data-Operationen für den Datenschutz.
    Ermöglicht die Ausführung von Berechnungen auf sensiblen Daten ohne direkten Zugriff.
    """
    
    def __init__(self, encryption_key: bytes = None):
        """
        Initialisiert den Compute-to-Data-Manager.
        
        Args:
            encryption_key: Schlüssel für die Verschlüsselung (optional)
        """
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.allowed_operations = {
            'aggregate': self._aggregate,
            'count': self._count,
            'mean': self._mean,
            'sum': self._sum,
            'min': self._min,
            'max': self._max,
            'std': self._std,
            'correlation': self._correlation,
            'histogram': self._histogram,
            'custom_model': self._custom_model_inference
        }
        
        # Datenschutzkonfiguration
        self.privacy_config = {
            'min_group_size': 5,  # Minimale Gruppengröße für Aggregationen
            'noise_level': 0.01,  # Standardrauschen für Differentialprivatsphäre
            'outlier_removal': True  # Entfernung von Ausreißern für sensible Berechnungen
        }
    
    def set_privacy_config(self, min_group_size: int = None, noise_level: float = None, 
                          outlier_removal: bool = None):
        """Aktualisiert die Datenschutzkonfiguration"""
        if min_group_size is not None:
            self.privacy_config['min_group_size'] = min_group_size
        if noise_level is not None:
            self.privacy_config['noise_level'] = noise_level
        if outlier_removal is not None:
            self.privacy_config['outlier_removal'] = outlier_removal
    
    def _encrypt_data(self, data: pd.DataFrame) -> bytes:
        """Verschlüsselt Daten für die sichere Speicherung"""
        serialized = data.to_json().encode()
        return self.cipher_suite.encrypt(serialized)
    
    def _decrypt_data(self, encrypted_data: bytes) -> pd.DataFrame:
        """Entschlüsselt Daten für die Verarbeitung"""
        decrypted = self.cipher_suite.decrypt(encrypted_data)
        return pd.read_json(decrypted.decode())
    
    def _add_differential_privacy_noise(self, result, scale: float = None):
        """Fügt Rauschen für Differential Privacy hinzu"""
        scale = scale or self.privacy_config['noise_level']
        
        if isinstance(result, (int, float)):
            return result + np.random.normal(0, scale * abs(result))
        elif isinstance(result, np.ndarray):
            return result + np.random.normal(0, scale * np.abs(result).mean(), result.shape)
        elif isinstance(result, pd.Series):
            noisy_values = result.values + np.random.normal(0, scale * np.abs(result.values).mean(), result.shape)
            return pd.Series(noisy_values, index=result.index)
        elif isinstance(result, pd.DataFrame):
            noisy_values = result.values + np.random.normal(0, scale * np.abs(result.values).mean(), result.shape)
            return pd.DataFrame(noisy_values, index=result.index, columns=result.columns)
        else:
            return result
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Entfernt Ausreißer aus den Daten"""
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            return data
        
        # Z-Score-basierte Ausreißererkennung
        z_scores = ((numeric_data - numeric_data.mean()) / numeric_data.std()).abs()
        mask = (z_scores < 3).all(axis=1)
        
        return data[mask]
    
    def _validate_group_size(self, data: pd.DataFrame, group_column: str = None, 
                            group_value: Any = None) -> bool:
        """
        Prüft, ob eine Gruppe die Mindestgröße für Datenschutz erfüllt
        
        Args:
            data: Daten
            group_column: Spalte für die Gruppierung
            group_value: Wert für die Filterung
            
        Returns:
            True, wenn die Gruppe groß genug ist
        """
        min_size = self.privacy_config['min_group_size']
        
        if group_column is not None and group_value is not None:
            group_data = data[data[group_column] == group_value]
            return len(group_data) >= min_size
        
        return len(data) >= min_size
    
    def _prepare_data_for_computation(self, data: pd.DataFrame, columns: List[str] = None,
                                    group_column: str = None, group_value: Any = None) -> pd.DataFrame:
        """
        Bereitet Daten für die Berechnung vor, einschließlich Filterung und Datenschutzprüfungen
        
        Args:
            data: Eingabedaten
            columns: Zu verwendende Spalten
            group_column: Spalte für die Gruppierung
            group_value: Wert für die Filterung
            
        Returns:
            Vorbereitete Daten
        """
        # Filtere Daten nach Gruppe, falls angegeben
        if group_column is not None and group_value is not None:
            prepared_data = data[data[group_column] == group_value]
        else:
            prepared_data = data.copy()
        
        # Prüfe Mindestgruppengröße
        if not self._validate_group_size(prepared_data):
            raise ValueError(f"Gruppe zu klein für Datenschutz (min: {self.privacy_config['min_group_size']})")
        
        # Entferne Ausreißer, falls konfiguriert
        if self.privacy_config['outlier_removal']:
            prepared_data = self._remove_outliers(prepared_data)
        
        # Wähle Spalten aus, falls angegeben
        if columns is not None:
            prepared_data = prepared_data[columns]
        
        return prepared_data
    
    # Implementierte Operationen für C2D
    
    def _aggregate(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Aggregiert Daten nach Spalten und Gruppen"""
        columns = params.get('columns')
        group_by = params.get('group_by')
        aggregations = params.get('aggregations', ['count', 'mean'])
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        if group_by:
            grouped = prepared_data.groupby(group_by)
            
            # Prüfe Mindestgröße für jede Gruppe
            group_sizes = grouped.size()
            valid_groups = group_sizes[group_sizes >= self.privacy_config['min_group_size']].index
            
            # Filtere ungültige Gruppen
            results = {}
            for agg in aggregations:
                if agg == 'count':
                    agg_result = grouped.size()
                else:
                    agg_func = getattr(grouped, agg, None)
                    if agg_func is None:
                        continue
                    agg_result = agg_func()
                
                # Behalte nur valide Gruppen
                agg_result = agg_result.loc[valid_groups]
                
                # Füge Rauschen hinzu
                results[agg] = self._add_differential_privacy_noise(agg_result)
        else:
            results = {}
            for agg in aggregations:
                if agg == 'count':
                    agg_result = len(prepared_data)
                else:
                    agg_func = getattr(prepared_data, agg, None)
                    if agg_func is None:
                        continue
                    agg_result = agg_func()
                
                # Füge Rauschen hinzu
                results[agg] = self._add_differential_privacy_noise(agg_result)
        
        # Konvertiere zu serialisierbarem Format
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
                serializable_results[k] = v.to_dict()
            else:
                serializable_results[k] = v
        
        return serializable_results
    
    def _count(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Zählt Datensätze nach Filterkriterien"""
        filter_column = params.get('filter_column')
        filter_value = params.get('filter_value')
        
        if filter_column and filter_value is not None:
            count = len(data[data[filter_column] == filter_value])
        else:
            count = len(data)
        
        # Füge Rauschen hinzu und runde auf ganze Zahlen
        noisy_count = round(self._add_differential_privacy_noise(count))
        
        # Stelle sicher, dass der Wert nicht negativ ist
        return {'count': max(0, noisy_count)}
    
    def _mean(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet den Mittelwert für numerische Spalten"""
        columns = params.get('columns')
        group_column = params.get('group_column')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        if group_column:
            grouped = prepared_data.groupby(group_column)
            
            # Prüfe Mindestgröße für jede Gruppe
            group_sizes = grouped.size()
            valid_groups = group_sizes[group_sizes >= self.privacy_config['min

      tx_hash = f"0x{uuid.uuid4().hex}"
        token_address = f"0x{uuid.uuid4().hex}"
        
        # In einer echten Implementierung würde hier der Aufruf der Ocean Protocol API 
        # oder Smart Contracts erfolgen, um die tatsächliche Tokenisierung durchzuführen
        
        return {
            'success': True,
            'asset_id': ocean_asset.get('asset_id', 'unknown'),
            'token_address': token_address,
            'token_symbol': ocean_asset.get('pricing', {}).get('datatoken', {}).get('symbol', 'UNKNOWN'),
            'token_name': ocean_asset.get('pricing', {}).get('datatoken', {}).get('name', 'Unknown Token'),
            'token_price': ocean_asset.get('pricing', {}).get('baseTokenAmount', 0),
            'transaction_hash': tx_hash,
            'timestamp': datetime.now().isoformat(),
            'marketplace_url': f"https://market.oceanprotocol.com/asset/{ocean_asset.get('asset_id', 'unknown')}",
            'message': "Asset successfully tokenized and published to Ocean marketplace"
        }
    
    def run_ocean_compute_job(self, asset_id: str, algorithm_id: str, 
                             algorithm_params: Dict = None) -> Dict:
        """
        Führt einen Compute-to-Data-Job auf einem tokenisierten Dataset aus.
        Diese Funktion ist ein Platzhalter für die tatsächliche Ocean Protocol Integration.
        
        Args:
            asset_id: ID des zu berechnenden Assets
            algorithm_id: ID des auszuführenden Algorithmus
            algorithm_params: Parameter für den Algorithmus
            
        Returns:
            Job-Informationen und Ergebnisstatus
        """
        # Dies ist eine Platzhalterimplementierung für die Ocean Protocol C2D-Integration
        
        self.logger.info(f"Simuliere Ocean Protocol C2D-Job für Asset {asset_id} mit Algorithmus {algorithm_id}")
        
        # Simuliere Jobausführung
        job_id = f"job-{uuid.uuid4().hex[:16]}"
        
        # In einer echten Implementierung würde hier der Aufruf der Ocean Protocol C2D-API erfolgen
        
        return {
            'success': True,
            'job_id': job_id,
            'asset_id': asset_id,
            'algorithm_id': algorithm_id,
            'status': 'completed',
            'started_at': datetime.now().isoformat(),
            'completed_at': (datetime.now() + timedelta(minutes=2)).isoformat(),
            'result_url': f"https://storage.oceanprotocol.com/results/{job_id}",
            'message': "Compute job completed successfully"
        }

# Beispiel für die Verwendung des Algorithmus
def demo_ocean_data_pipeline():
    """Demonstriert die vollständige Daten-Pipeline von der Erfassung bis zur Monetarisierung"""
    
    # Konfiguration für die OceanData-KI
    config = {
        'anomaly_detection_method': 'isolation_forest',
        'semantic_model': 'bert',
        'predictive_model': 'lstm',
        'forecast_horizon': 7,
        'analyze_text_sentiment': True,
        'enable_data_synthesis': True,
        'base_token_value': 5.0,
        'min_token_value': 1.0,
        'max_token_value': 10.0
    }
    
    # Initialisiere die OceanData-KI
    ocean_ai = OceanDataAI(config)
    
    # 1. Simulierte Datenerfassung von verschiedenen Quellen
    # In einer echten Implementierung würden diese Daten von den DataConnector-Klassen kommen
    
    # Browser-Daten
    browser_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'url': [f"example{i % 10}.com/page{i % 5}" for i in range(100)],
        'duration': np.random.randint(10, 300, 100),
        'user_id': ['user123'] * 100,
        'device_type': np.random.choice(['desktop', 'mobile', 'tablet'], 100),
        'browser_type': np.random.choice(['chrome', 'firefox', 'safari'], 100)
    })
    
    # Smartwatch-Daten
    smartwatch_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'heart_rate': np.random.randint(60, 100, 100),
        'steps': np.random.randint(0, 1000, 100),
        'user_id': ['user123'] * 100,
        'sleep_quality': np.random.choice(['good', 'medium', 'poor'], 100),
        'calories_burned': np.random.randint(50, 300, 100)
    })
    
    # 2. Analyse der Datenquellen
    browser_analysis = ocean_ai.analyze_data_source(browser_data, 'browser')
    smartwatch_analysis = ocean_ai.analyze_data_source(smartwatch_data, 'smartwatch')
    
    print(f"Analyse der Browser-Daten abgeschlossen: {len(browser_analysis.get('analyses', {}))} Analysemodule ausgeführt")
    print(f"Analyse der Smartwatch-Daten abgeschlossen: {len(smartwatch_analysis.get('analyses', {}))} Analysemodule ausgeführt")
    
    # 3. Vorbereitung der Daten für die Monetarisierung
    browser_asset = ocean_ai.prepare_data_for_monetization(browser_data, 'browser', 'medium')
    smartwatch_asset = ocean_ai.prepare_data_for_monetization(smartwatch_data, 'smartwatch', 'high')
    
    print(f"Browser-Daten für Monetarisierung vorbereitet: Geschätzter Wert = {browser_asset['metadata'].get('estimated_value', 0):.2f} OCEAN")
    print(f"Smartwatch-Daten für Monetarisierung vorbereitet: Geschätzter Wert = {smartwatch_asset['metadata'].get('estimated_value', 0):.2f} OCEAN")
    
    # 4. Kombination der Datenquellen für höheren Wert
    combined_asset = ocean_ai.combine_data_sources([browser_asset, smartwatch_asset], 'correlate')
    
    print(f"Kombinierte Daten erstellt: Geschätzter Wert = {combined_asset['metadata'].get('estimated_value', 0):.2f} OCEAN")
    print(f"Wertsteigerung durch Kombination: {combined_asset['metadata'].get('estimated_value', 0) - browser_asset['metadata'].get('estimated_value', 0) - smartwatch_asset['metadata'].get('estimated_value', 0):.2f} OCEAN")
    
    # 5. Detaillierte Wertschätzung für den kombinierten Datensatz
    value_assessment = ocean_ai.estimate_data_value(combined_asset['anonymized_data'], combined_asset['metadata'])
    
    print(f"Detaillierte Wertschätzung: {value_assessment['estimated_token_value']:.2f} OCEAN")
    print(f"Wertzusammenfassung: {value_assessment['summary']}")
    
    # 6. Vorbereitung für Ocean Protocol Tokenisierung
    ocean_asset = ocean_ai.prepare_for_ocean_tokenization(combined_asset)
    
    print(f"Asset für Ocean Protocol vorbereitet: {ocean_asset['ddo']['name']}")
    
    # 7. Tokenisierung mit Ocean Protocol (simuliert)
    tokenization_result = ocean_ai.tokenize_with_ocean(ocean_asset)
    
    print(f"Asset erfolgreich tokenisiert: Token = {tokenization_result['token_symbol']} ({tokenization_result['token_address']})")
    print(f"Asset im Ocean Marketplace verfügbar unter: {tokenization_result['marketplace_url']}")
    
    # 8. Ausführung eines Compute-to-Data-Jobs auf dem tokenisierten Asset (simuliert)
    algorithm_params = {
        'operation': 'clustering',
        'n_clusters': 3,
        'features': ['heart_rate', 'steps', 'duration']
    }
    
    compute_job = ocean_ai.run_ocean_compute_job(
        ocean_asset['asset_id'], 
        'clustering-algorithm', 
        algorithm_params
    )
    
    print(f"Compute-to-Data-Job abgeschlossen: {compute_job['status']}")
    print(f"Ergebnisse verfügbar unter: {compute_job['result_url']}")
    
    return {
        'browser_asset': browser_asset,
        'smartwatch_asset': smartwatch_asset,
        'combined_asset': combined_asset,
        'value_assessment': value_assessment,
        'ocean_asset': ocean_asset,
        'tokenization_result': tokenization_result,
        'compute_job': compute_job
    }

if __name__ == "__main__":
    # Führe die Demo aus
    demo_results = demo_ocean_data_pipeline()
                'data_uniqueness': {
                    'score': 0.0,
                    'weight': 0.2,
                    'explanation': ''
                },
                'data_applicability': {
                    'score': 0.0,
                    'weight': 0.25,
                    'explanation': ''
                },
                'time_relevance': {
                    'score': 0.0,
                    'weight': 0.1,
                    'explanation': ''
                },
                'enrichment_potential': {
                    'score': 0.0,
                    'weight': 0.1,
                    'explanation': ''
                }
            }
            
            # 1. Datenvolumen
            # Bewerte die Größe des Datensatzes (Zeilen und Spalten)
            rows_score = min(1.0, len(data) / 10000)  # Skaliert bis 10.000 Zeilen
            cols_score = min(1.0, len(data.columns) / 50)  # Skaliert bis 50 Spalten
            volume_score = (rows_score * 0.7) + (cols_score * 0.3)  # Gewichtung: Zeilen wichtiger als Spalten
            
            value_metrics['data_volume']['score'] = volume_score
            value_metrics['data_volume']['explanation'] = f"Dataset contains {len(data)} records and {len(data.columns)} fields. "
            
            if volume_score < 0.3:
                value_metrics['data_volume']['explanation'] += "This is a relatively small dataset, which limits its potential value."
            elif volume_score < 0.7:
                value_metrics['data_volume']['explanation'] += "This is a medium-sized dataset with solid commercial value potential."
            else:
                value_metrics['data_volume']['explanation'] += "This is a large dataset with significant commercial value potential."
            
            # 2. Datenqualität
            # Bewerte die Qualität anhand von fehlenden Werten, Varianz und Konsistenz
            
            # Berechne fehlende Werte
            missing_ratio = data.isna().mean().mean()
            completeness = 1 - missing_ratio
            
            # Berechne Varianz für numerische Spalten
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                normalized_variance = 0
                for col in numeric_cols:
                    # Normalisierte Varianz (0-1)
                    if data[col].std() > 0:
                        normalized_variance += min(1.0, data[col].std() / data[col].mean() if data[col].mean() != 0 else 0)
                
                avg_normalized_variance = normalized_variance / len(numeric_cols)
                variance_score = avg_normalized_variance
            else:
                variance_score = 0.5  # Mittlerer Wert, wenn keine numerischen Spalten vorhanden sind
            
            # Berechne Eindeutigkeit für kategorische Spalten
            categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
            if categorical_cols:
                uniqueness_ratio = 0
                for col in categorical_cols:
                    uniqueness_ratio += data[col].nunique() / len(data)
                
                avg_uniqueness_ratio = uniqueness_ratio / len(categorical_cols)
                uniqueness_score = min(1.0, avg_uniqueness_ratio * 3)  # Skaliert, um niedrigere Werte aufzuwerten
            else:
                uniqueness_score = 0.5  # Mittlerer Wert, wenn keine kategorischen Spalten vorhanden sind
            
            # Kombiniere die Qualitätsmetriken
            quality_score = (completeness * 0.5) + (variance_score * 0.25) + (uniqueness_score * 0.25)
            
            value_metrics['data_quality']['score'] = quality_score
            value_metrics['data_quality']['explanation'] = f"Data completeness: {completeness:.1%}. "
            
            if completeness < 0.8:
                value_metrics['data_quality']['explanation'] += "Missing data impacts overall quality. "
            
            value_metrics['data_quality']['explanation'] += f"Data variability: {variance_score:.1%}. "
            if variance_score < 0.3:
                value_metrics['data_quality']['explanation'] += "Low variability may indicate limited information content. "
            elif variance_score > 0.7:
                value_metrics['data_quality']['explanation'] += "High variability indicates rich information content. "
            
            # 3. Dateneinzigartigkeit
            # Bewerte die Einzigartigkeit basierend auf Spaltentypen und Datenbereichen
            
            # Analysiere Spaltentypen
            column_types = {
                'datetime': len(data.select_dtypes(include=['datetime']).columns),
                'numeric': len(data.select_dtypes(include=['number']).columns),
                'categorical': len(data.select_dtypes(include=['category', 'object']).columns),
                'boolean': len(data.select_dtypes(include=['bool']).columns),
                'text': sum(1 for col in data.select_dtypes(include=['object']).columns 
                          if data[col].astype(str).str.len().mean() > 20)
            }
            
            # Diversitätspunkte für Spaltentypen-Mix
            diversity_score = min(1.0, sum(1 for count in column_types.values() if count > 0) / 5)
            
            # Zusätzliche Punkte für seltene oder wertvolle Datentypen
            value_column_score = 0
            if column_types['datetime'] > 0:
                value_column_score += 0.2  # Zeitreihen sind wertvoll
            if column_types['text'] > 0:
                value_column_score += 0.3  # Textdaten sind wertvoll für NLP
            if column_types['boolean'] > 0 and column_types['numeric'] > 0:
                value_column_score += 0.1  # Gute Kombination für ML
            
            # Quelltyp-spezifischer Wert (wenn vorhanden)
            source_type = metadata.get('source_type', '')
            source_type_scores = {
                'health_data': 0.9,
                'health_insurance': 0.9,
                'smartwatch': 0.8,
                'social_media': 0.7,
                'browser': 0.6,
                'chat': 0.7,
                'streaming': 0.7,
                'calendar': 0.5,
                'iot_security': 0.6,
                'smart_home': 0.5
            }
            source_score = source_type_scores.get(source_type, 0.5)
            
            # Kombiniere Einzigartigkeit und Wert
            uniqueness_score = (diversity_score * 0.3) + (value_column_score * 0.3) + (source_score * 0.4)
            
            value_metrics['data_uniqueness']['score'] = uniqueness_score
            value_metrics['data_uniqueness']['explanation'] = f"Data contains {sum(column_types.values())} columns across {sum(1 for count in column_types.values() if count > 0)} different data types. "
            
            if source_type:
                value_metrics['data_uniqueness']['explanation'] += f"Source type '{source_type}' "
                if source_score > 0.7:
                    value_metrics['data_uniqueness']['explanation'] += "is particularly valuable in the data marketplace. "
                elif source_score > 0.5:
                    value_metrics['data_uniqueness']['explanation'] += "has good value in the data marketplace. "
                else:
                    value_metrics['data_uniqueness']['explanation'] += "has standard value in the data marketplace. "
            
            if column_types['text'] > 0:
                value_metrics['data_uniqueness']['explanation'] += f"Contains {column_types['text']} text columns which add significant value for NLP applications. "
            
            if column_types['datetime'] > 0:
                value_metrics['data_uniqueness']['explanation'] += f"Contains {column_types['datetime']} datetime columns enabling time-series analysis. "
            
            # 4. Anwendbarkeit der Daten
            # Bewerte potenzielle Anwendungsfälle basierend auf Datentyp und Marktanforderungen
            
            # Identifiziere potenzielle Anwendungsfälle basierend auf verfügbaren Daten
            application_scenarios = []
            
            # Prüfe auf typische Anwendungsfälle basierend auf Spaltenkombinationen
            has_location_data = any('location' in col.lower() or 'address' in col.lower() or 'geo' in col.lower() for col in data.columns)
            has_user_data = any('user' in col.lower() or 'customer' in col.lower() or 'id' in col.lower() for col in data.columns)
            has_time_data = column_types['datetime'] > 0
            has_numeric_features = column_types['numeric'] > 2  # Mindestens 3 numerische Spalten
            has_categorical_features = column_types['categorical'] > 0
            has_text_features = column_types['text'] > 0
            
            # Bewerte Anwendbarkeit für Machine Learning
            if has_numeric_features and (has_categorical_features or has_time_data):
                application_scenarios.append({
                    'name': 'Predictive Modeling',
                    'score': 0.8,
                    'description': 'Suitable for building predictive models using machine learning'
                })
            
            # Bewerte Anwendbarkeit für Zeitreihenanalyse
            if has_time_data and has_numeric_features:
                application_scenarios.append({
                    'name': 'Time Series Analysis',
                    'score': 0.9,
                    'description': 'Ideal for trend analysis and forecasting over time'
                })
            
            # Bewerte Anwendbarkeit für Personalisierung
            if has_user_data and (has_categorical_features or has_text_features):
                application_scenarios.append({
                    'name': 'Personalization & Recommendations',
                    'score': 0.85,
                    'description': 'Can be used for user preference modeling and recommendations'
                })
            
            # Bewerte Anwendbarkeit für geografische Analysen
            if has_location_data:
                application_scenarios.append({
                    'name': 'Geospatial Analysis',
                    'score': 0.75,
                    'description': 'Enables location-based insights and targeting'
                })
            
            # Bewerte Anwendbarkeit für NLP
            if has_text_features:
                application_scenarios.append({
                    'name': 'Natural Language Processing',
                    'score': 0.85,
                    'description': 'Text data can be used for sentiment analysis, topic modeling, etc.'
                })
            
            # Bewerte Anwendbarkeit für Clusteranalyse
            if has_numeric_features >= 3:
                application_scenarios.append({
                    'name': 'Clustering & Segmentation',
                    'score': 0.7,
                    'description': 'Can be used to identify natural groupings and segments'
                })
            
            # Berechne den Anwendbarkeitswert basierend auf den identifizierten Szenarien
            if application_scenarios:
                # Durchschnittliche Bewertung über alle Szenarien
                applicability_score = sum(scenario['score'] for scenario in application_scenarios) / len(application_scenarios)
                
                # Bonus für die Anzahl der Szenarien (mehr Szenarien = vielseitigere Daten)
                scenario_count_bonus = min(0.2, (len(application_scenarios) - 1) * 0.05)
                applicability_score += scenario_count_bonus
                
                # Begrenze auf Maximum von 1.0
                applicability_score = min(1.0, applicability_score)
            else:
                # Standard-Score, wenn keine spezifischen Szenarien identifiziert wurden
                applicability_score = 0.3
            
            value_metrics['data_applicability']['score'] = applicability_score
            value_metrics['data_applicability']['explanation'] = f"Identified {len(application_scenarios)} potential application scenarios. "
            
            if application_scenarios:
                top_scenarios = sorted(application_scenarios, key=lambda x: x['score'], reverse=True)[:3]
                value_metrics['data_applicability']['explanation'] += "Top applications: " + ", ".join(s['name'] for s in top_scenarios) + ". "
            
            if applicability_score > 0.7:
                value_metrics['data_applicability']['explanation'] += "Data has high versatility across multiple use cases. "
            elif applicability_score > 0.4:
                value_metrics['data_applicability']['explanation'] += "Data has moderate versatility for specific applications. "
            else:
                value_metrics['data_applicability']['explanation'] += "Data has limited application potential without additional enrichment. "
            
            # 5. Zeitliche Relevanz
            # Bewerte die Aktualität und zeitliche Spanne der Daten
            
            time_relevance_score = 0.5  # Mittlerer Standardwert
            
            # Prüfe auf zeitbezogene Metadaten
            time_range = metadata.get('time_range', {})
            if time_range:
                # Aktualität: Wie nah ist das neueste Datum an heute?
                end_date = time_range.get('end')
                if end_date:
                    try:
                        # Konvertiere ISO-String zu Datum
                        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        days_since_end = (datetime.now() - end_date).days
                        
                        # Berechne Aktualitätspunkte (höher für neuere Daten)
                        recency_score = max(0, 1 - (days_since_end / 365))  # Skaliert über ein Jahr
                    except:
                        recency_score = 0.5  # Standardwert bei Parsing-Fehler
                else:
                    recency_score = 0.5
                
                # Zeitspanne: Längere Zeitreihen sind wertvoller
                duration_days = time_range.get('duration_days', 0)
                if duration_days:
                    # Berechne Punktzahl für Zeitspanne (höher für längere Zeiträume)
                    span_score = min(1.0, duration_days / 365)  # Skaliert bis zu einem Jahr
                else:
                    span_score = 0.5
                
                # Kombiniere Aktualität und Zeitspanne
                time_relevance_score = (recency_score * 0.6) + (span_score * 0.4)  # Aktualität wichtiger als Spanne
            
            value_metrics['time_relevance']['score'] = time_relevance_score
            value_metrics['time_relevance']['explanation'] = ""
            
            if time_range:
                value_metrics['time_relevance']['explanation'] += f"Data spans {time_range.get('duration_days', '?')} days. "
                
                if time_relevance_score > 0.7:
                    value_metrics['time_relevance']['explanation'] += "Data is recent and covers a substantial time period. "
                elif time_relevance_score > 0.4:
                    value_metrics['time_relevance']['explanation'] += "Data has moderate temporal relevance. "
                else:
                    value_metrics['time_relevance']['explanation'] += "Data may be outdated or covers a limited time period. "
            else:
                value_metrics['time_relevance']['explanation'] += "No temporal information available to assess time relevance. "
            
            # 6. Anreicherungspotenzial
            # Bewerte, wie gut sich die Daten mit anderen Quellen kombinieren lassen
            
            # Identifiziere potenzielle Verknüpfungsfelder
            linkage_fields = []
            
            # Typische ID-Felder
            id_fields = [col for col in data.columns if 'id' in col.lower() or 'key' in col.lower()]
            if id_fields:
                linkage_fields.extend(id_fields)
            
            # Standortfelder für geografische Verknüpfung
            location_fields = [col for col in data.columns 
                              if 'location' in col.lower() or 'address' in col.lower() 
                              or 'city' in col.lower() or 'country' in col.lower()]
            if location_fields:
                linkage_fields.extend(location_fields)
            
            # Zeitfelder für zeitliche Verknüpfung
            time_fields = list(data.select_dtypes(include=['datetime']).columns)
            if time_fields:
                linkage_fields.extend(time_fields)
            
            # Kategorienwerte für kontextuelle Verknüpfung
            category_fields = [col for col in data.columns 
                              if data[col].dtype == 'object' and data[col].nunique() < 50]
            if category_fields:
                linkage_fields.extend(category_fields[:3])  # Beschränke auf die ersten drei
            
            # Einzigartige Linkage-Felder
            unique_linkage_fields = list(set(linkage_fields))
            
            # Berechne den Anreicherungswert basierend auf Linkage-Feldern
            if unique_linkage_fields:
                # Mehr Verknüpfungsfelder = höheres Anreicherungspotenzial
                field_count_score = min(1.0, len(unique_linkage_fields) / 5)  # Skaliert bis zu 5 Felder
                
                # Zusätzliche Punkte für besonders wertvolle Verknüpfungstypen
                id_bonus = 0.3 if any('id' in field.lower() for field in unique_linkage_fields) else 0
                location_bonus = 0.2 if any(field in location_fields for field in unique_linkage_fields) else 0
                time_bonus = 0.2 if any(field in time_fields for field in unique_linkage_fields) else 0
                
                # Berechne Gesamtwert mit Begrenzung auf 1.0
                enrichment_score = min(1.0, field_count_score + id_bonus + location_bonus + time_bonus)
            else:
                # Geringer Wert für Daten ohne offensichtliche Verknüpfungsmöglichkeiten
                enrichment_score = 0.2
            
            value_metrics['enrichment_potential']['score'] = enrichment_score
            value_metrics['enrichment_potential']['explanation'] = f"Identified {len(unique_linkage_fields)} potential linkage fields. "
            
            if unique_linkage_fields:
                field_types = []
                if any('id' in field.lower() for field in unique_linkage_fields):
                    field_types.append("IDs")
                if any(field in location_fields for field in unique_linkage_fields):
                    field_types.append("location data")
                if any(field in time_fields for field in unique_linkage_fields):
                    field_types.append("temporal markers")
                if any(field in category_fields for field in unique_linkage_fields):
                    field_types.append("categorical attributes")
                
                value_metrics['enrichment_potential']['explanation'] += "Linkage opportunities include: " + ", ".join(field_types) + ". "
            
            if enrichment_score > 0.7:
                value_metrics['enrichment_potential']['explanation'] += "Data has excellent potential for enrichment with other datasets. "
            elif enrichment_score > 0.4:
                value_metrics['enrichment_potential']['explanation'] += "Data has moderate potential for combination with complementary sources. "
            else:
                value_metrics['enrichment_potential']['explanation'] += "Data has limited enrichment potential without additional identifiers. "
            
            # Berechne den Gesamtwert als gewichtete Summe aller Faktoren
            total_weighted_score = sum(metric['score'] * metric['weight'] for metric in value_metrics.values())
            
            # Konvertiere den normalisierten Score in einen monetären Wert
            # Basierend auf einer konfigurierbaren Wertskala, z.B. 0-10 OCEAN Tokens
            min_value = self.config.get('min_token_value', 1.0)
            max_value = self.config.get('max_token_value', 10.0)
            estimated_token_value = min_value + (total_weighted_score * (max_value - min_value))
            
            # Erstelle die Wertzusammenfassung
            value_summary = {
                'normalized_score': float(total_weighted_score),
                'estimated_token_value': float(estimated_token_value),
                'metrics': value_metrics,
                'summary': f"This dataset has an estimated value of {estimated_token_value:.2f} OCEAN tokens, "
                          f"with its primary strengths being {', '.join(k for k, v in sorted([(k, v['score']) for k, v in value_metrics.items()], key=lambda x: x[1], reverse=True)[:2])}."
            }
            
            return value_summary
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenwertschätzung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'normalized_score': 0.3,  # Standardwert im Fehlerfall
                'estimated_token_value': self.config.get('min_token_value', 1.0),
                'metrics': {k: {'score': 0.3, 'weight': v['weight'], 'explanation': 'Error in calculation'} 
                           for k, v in value_metrics.items()}
            }

    def prepare_for_ocean_tokenization(self, data_asset: Dict) -> Dict:
        """
        Bereitet einen Daten-Asset für die Tokenisierung mit Ocean Protocol vor.
        
        Args:
            data_asset: Der vorzubereitende Daten-Asset mit Metadaten und C2D-Asset
            
        Returns:
            Für Ocean Protocol formatierte Asset-Informationen
        """
        try:
            self.logger.info(f"Bereite Asset für Ocean Protocol Tokenisierung vor")
            
            if 'error' in data_asset:
                raise ValueError(f"Fehlerhafter Daten-Asset: {data_asset['error']}")
            
            if 'c2d_asset' not in data_asset or 'metadata' not in data_asset:
                raise ValueError("Asset enthält nicht die erforderlichen Informationen (c2d_asset oder metadata)")
            
            # Extrahiere relevante Informationen
            asset_id = data_asset['c2d_asset']['asset_id']
            metadata = data_asset['metadata']
            
            # Erstelle ein DDO (Decentralized Data Object) im Ocean-Format
            name = metadata.get('name', f"Dataset {asset_id[:8]}")
            if 'source_type' in metadata:
                name = f"{metadata['source_type'].replace('_', ' ').title()} Dataset {asset_id[:8]}"
            
            description = metadata.get('description', '')
            if not description and 'potential_use_cases' in metadata and metadata['potential_use_cases']:
                # Verwende die Beschreibung des ersten Anwendungsfalls
                use_case = metadata['potential_use_cases'][0]
                description = f"{use_case.get('title', '')}: {use_case.get('description', '')}"
            
            # Kategorien basierend auf dem Quelltyp
            categories = []
            if 'source_type' in metadata:
                source_type = metadata['source_type']
                
                # Mapping von Quelltypen zu Ocean-Kategorien
                category_mapping = {
                    'browser': ['Web', 'Consumer Behavior'],
                    'calendar': ['Productivity', 'Time Management'],
                    'chat': ['Communication', 'Social'],
                    'social_media': ['Social', 'Marketing'],
                    'streaming': ['Entertainment', 'Media'],
                    'health_insurance': ['Healthcare', 'Insurance'],
                    'health_data': ['Healthcare', 'Wellness'],
                    'smartwatch': ['IoT', 'Wearables', 'Fitness'],
                    'iot_vacuum': ['IoT', 'Smart Home'],
                    'iot_thermostat': ['IoT', 'Energy'],
                    'iot_lighting': ['IoT', 'Smart Home'],
                    'iot_security': ['IoT', 'Security'],
                    'smart_home': ['IoT', 'Smart Home']
                }
                categories = category_mapping.get(source_type, ['Other'])
            
            # Erstelle Tags basierend auf Metadaten und Spalteninformationen
            tags = []
            
            # Tags basierend auf der Datenquelle
            if 'source_type' in metadata:
                tags.append(metadata['source_type'])
            
            # Tags basierend auf Spaltenstatistiken
            if 'column_statistics' in metadata:
                column_types = [stats.get('type') for stats in metadata['column_statistics'].values()]
                if 'datetime' in column_types or 'timestamp' in column_types:
                    tags.append('time-series')
                if 'text' in column_types:
                    tags.append('text-data')
                if 'categorical' in column_types:
                    tags.append('categorical-data')
                if any(t == 'numeric' for t in column_types):
                    tags.append('numeric-data')
                if any('location' in col.lower() for col in metadata.get('column_statistics', {}).keys()):
                    tags.append('geospatial')
            
            # Datenschutztags basierend auf dem Privacy-Level
            if 'privacy_level' in metadata:
                tags.append(f"privacy-{metadata['privacy_level']}")
            
            # Füge eine zeitliche Dimension hinzu, falls vorhanden
            if 'time_range' in metadata and metadata['time_range'] and 'duration_days' in metadata['time_range']:
                days = metadata['time_range']['duration_days']
                if days <= 7:
                    tags.append('short-term')
                elif days <= 90:
                    tags.append('medium-term')
                else:
                    tags.append('long-term')
            
            # Erstelle das DDO
            ddo = {
                'id': asset_id,
                'created': datetime.now().isoformat(),
                'updated': datetime.now().isoformat(),
                'type': 'dataset',
                'name': name,
                'description': description,
                'tags': list(set(tags)),  # Entferne Duplikate
                'categories': categories,
                'author': metadata.get('author', 'OceanData User'),
                'license': metadata.get('license', 'No License Specified'),
                'price': metadata.get('estimated_value', 5.0),  # Geschätzter Wert in OCEAN
                'files': [
                    {
                        'type': 'compute-to-data',
                        'method': 'c2d',
                        'schema': 'c2d-schema'
                    }
                ],
                'additionalInformation': {
                    'privacy': metadata.get('privacy_level', 'medium'),
                    'record_count': metadata.get('record_count', 0),
                    'field_count': metadata.get('field_count', 0),
                    'potential_use_cases': metadata.get('potential_use_cases', []),
                    'value_factors': metadata.get('value_factors', {})
                }
            }
            
            # Compute-to-Data-spezifische Informationen
            c2d_config = {
                'asset_id': asset_id,
                'allowed_operations': [
                    'aggregate', 'count', 'mean', 'sum', 'min', 'max', 'std', 
                    'correlation', 'histogram', 'custom_model'
                ],
                'timeout': 3600,  # 1 Stunde Standard-Timeout
                'privacy_config': {
                    'min_group_size': 5,
                    'noise_level': 0.01,
                    'outlier_removal': True
                }
            }
            
            # Preisgestaltung und Token-Informationen
            pricing = {
                'type': 'fixed',
                'baseToken': {
                    'address': '0x967da4048cD07aB37855c090aAF366e4ce1b9F48',  # Ocean Token Adresse
                    'name': 'Ocean Token',
                    'symbol': 'OCEAN'
                },
                'baseTokenAmount': metadata.get('estimated_value', 5.0),
                'datatoken': {
                    'name': f"DT-{name.replace(' ', '-')}",
                    'symbol': f"DT{asset_id[:4]}".upper()
                }
            }
            
            return {
                'ddo': ddo,
                'c2d_config': c2d_config,
                'pricing': pricing,
                'asset_id': asset_id
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Vorbereitung für Ocean Tokenisierung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'asset_id': data_asset.get('c2d_asset', {}).get('asset_id', 'unknown')
            }
    
    def tokenize_with_ocean(self, ocean_asset: Dict) -> Dict:
        """
        Tokenisiert einen Asset mit Ocean Protocol.
        Diese Funktion ist ein Platzhalter, da die tatsächliche Implementierung 
        von der Integration mit dem Ocean Protocol Smart Contract abhängt.
        
        Args:
            ocean_asset: Vorbereitete Asset-Informationen für Ocean Protocol
            
        Returns:
            Tokenisierungsergebnisse mit Transaktionsinformationen
        """
        # Dies ist eine Platzhalterimplementierung, die in einer echten Umgebung
        # einen Aufruf an die Ocean Protocol Bibliothek und Smart Contracts machen würde
        
        self.logger.info(f"Simuliere Tokenisierung mit Ocean Protocol für Asset {ocean_asset.get('asset_id', 'unknown')}")
        
        # Simulierte Tokenisierungsergebnisse
        tx_hash = f"0x{uuid.uuid4().hex                    if col in anonymized_data.columns and anonymized_data[col].dtype == 'object':
                        # Behalte nur den ersten Teil der Adresse/Standorts (z.B. Stadt)
                        anonymized_data[col] = anonymized_data[col].astype(str).apply(
                            lambda x: x.split(',')[0] if ',' in x else x
                        )
                
                # Vergröbere Zeitstempel (auf Stundenebene)
                for col in datetime_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = pd.to_datetime(anonymized_data[col]).dt.floor('H')
                
            elif privacy_level == 'low':
                # Minimale Anonymisierung, nur offensichtlich persönliche Kennung anonymisieren
                for col in id_columns:
                    if col in anonymized_data.columns and 'user' in col.lower():
                        salt = datetime.now().strftime("%Y%m%d")
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest() if pd.notna(x) else x
                        )
            
            # 2. Berechnung des Datenwerts und Erstellung von Metadaten
            
            # a) Berechne Basiskennzahlen
            metadata = {
                'source_type': source_type,
                'privacy_level': privacy_level,
                'record_count': len(anonymized_data),
                'field_count': len(anonymized_data.columns),
                'time_range': None,
                'created_at': datetime.now().isoformat(),
                'data_schema': {
                    col: str(anonymized_data[col].dtype) for col in anonymized_data.columns
                },
                'estimated_value': 0.0,
                'value_factors': {},
                'column_statistics': {},
                'potential_use_cases': []
            }
            
            # b) Berechne Zeitspanne, falls Zeitstempelspalten existieren
            if datetime_columns:
                # Wähle die erste Zeitstempelspalte
                time_col = datetime_columns[0]
                min_date = anonymized_data[time_col].min()
                max_date = anonymized_data[time_col].max()
                
                metadata['time_range'] = {
                    'start': min_date.isoformat() if pd.notna(min_date) else None,
                    'end': max_date.isoformat() if pd.notna(max_date) else None,
                    'duration_days': (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None
                }
            
            # c) Berechne Spaltenstatistiken für numerische und kategorische Spalten
            for col in anonymized_data.columns:
                col_stats = {}
                
                if pd.api.types.is_numeric_dtype(anonymized_data[col]):
                    # Statistiken für numerische Spalten
                    col_stats = {
                        'type': 'numeric',
                        'min': float(anonymized_data[col].min()) if not pd.isna(anonymized_data[col].min()) else None,
                        'max': float(anonymized_data[col].max()) if not pd.isna(anonymized_data[col].max()) else None,
                        'mean': float(anonymized_data[col].mean()) if not pd.isna(anonymized_data[col].mean()) else None,
                        'median': float(anonymized_data[col].median()) if not pd.isna(anonymized_data[col].median()) else None,
                        'std': float(anonymized_data[col].std()) if not pd.isna(anonymized_data[col].std()) else None,
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100)
                    }
                elif pd.api.types.is_string_dtype(anonymized_data[col]) or anonymized_data[col].dtype == 'object':
                    # Statistiken für kategorische/Text-Spalten
                    col_stats = {
                        'type': 'categorical' if anonymized_data[col].nunique() < 30 else 'text',
                        'unique_values': int(anonymized_data[col].nunique()),
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100),
                        'most_common': anonymized_data[col].value_counts().head(5).to_dict() if anonymized_data[col].nunique() < 30 else None
                    }
                    
                    # Wenn es sich um Text handelt, zusätzliche Textstatistiken
                    if col_stats['type'] == 'text':
                        # Durchschnittliche Textlänge
                        col_stats['avg_length'] = float(anonymized_data[col].astype(str).str.len().mean())
                        
                        # Optional: Sentimentanalyse auf Stichprobe
                        if self.config.get('analyze_text_sentiment', True):
                            sample = anonymized_data[col].dropna().sample(min(100, anonymized_data[col].dropna().shape[0])).tolist()
                            if sample:
                                sentiment_results = self.semantic_analyzer.analyze_sentiment(sample)
                                sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
                                for result in sentiment_results:
                                    sentiment_counts[result['sentiment']] += 1
                                
                                col_stats['sentiment'] = {
                                    k: v / len(sentiment_results) for k, v in sentiment_counts.items()
                                }
                else:
                    # Andere Datentypen (Datum, Boolean, etc.)
                    col_stats = {
                        'type': str(anonymized_data[col].dtype),
                        'unique_values': int(anonymized_data[col].nunique()),
                        'missing_percentage': float(anonymized_data[col].isna().mean() * 100)
                    }
                
                metadata['column_statistics'][col] = col_stats
            
            # d) Berechnung des geschätzten Datenwerts basierend auf verschiedenen Faktoren
            value_factors = {}
            
            # Faktor: Datenmenge
            data_size_factor = min(1.0, len(anonymized_data) / 10000)  # Skaliert bis 10.000 Datenpunkte
            value_factors['data_size'] = float(data_size_factor)
            
            # Faktor: Datenvollständigkeit
            completeness = 1.0 - anonymized_data.isna().mean().mean()  # Prozentsatz der nicht-fehlenden Werte
            value_factors['completeness'] = float(completeness)
            
            # Faktor: Zeitliche Spanne (falls verfügbar)
            time_factor = 0.5  # Standardwert
            if metadata['time_range'] and metadata['time_range']['duration_days']:
                # Höherer Wert für längere Zeitreihen (skaliert bis zu 1 Jahr)
                days = metadata['time_range']['duration_days']
                time_factor = min(1.0, days / 365)
            value_factors['time_span'] = float(time_factor)
            
            # Faktor: Datenquellentyp (unterschiedliche Gewichtung je nach Quelle)
            source_weights = {
                'browser': 0.7,
                'calendar': 0.6,
                'chat': 0.8,
                'social_media': 0.9,
                'streaming': 0.8,
                'health_insurance': 1.0,
                'health_data': 1.0,
                'smartwatch': 0.9,
                'iot_vacuum': 0.5,
                'iot_thermostat': 0.6,
                'iot_lighting': 0.5,
                'iot_security': 0.8,
                'smart_home': 0.7
            }
            source_factor = source_weights.get(source_type, 0.5)
            value_factors['source_type'] = float(source_factor)
            
            # Faktor: Datenschutzniveau (höherer Datenschutz = geringerer Wert)
            privacy_factors = {
                'low': 1.0,
                'medium': 0.8,
                'high': 0.6
            }
            privacy_factor = privacy_factors.get(privacy_level, 0.7)
            value_factors['privacy_level'] = float(privacy_factor)
            
            # Schätze den Gesamtwert als gewichteten Durchschnitt der Faktoren
            weights = {
                'data_size': 0.3,
                'completeness': 0.2,
                'time_span': 0.15,
                'source_type': 0.25,
                'privacy_level': 0.1
            }
            
            estimated_value = sum(value_factors[factor] * weights[factor] for factor in weights)
            
            # Konvertiere den normierten Wert in einen Tokenwert (angenommene Basis: 1-10 OCEAN)
            base_token_value = self.config.get('base_token_value', 5.0)  # 5 OCEAN als Basiswert
            token_value = base_token_value * estimated_value
            
            metadata['estimated_value'] = float(token_value)
            metadata['value_factors'] = value_factors
            
            # e) Identifiziere potenzielle Anwendungsfälle basierend auf den Daten
            use_cases = []
            
            # Browser-Daten: Surfgewohnheiten, Werbezielgruppen
            if source_type == 'browser':
                use_cases.append({
                    'title': 'Consumer Behavior Analysis',
                    'description': 'Analyze browsing patterns to understand consumer preferences and interests',
                    'value_proposition': 'Optimize marketing strategies and product recommendations'
                })
                use_cases.append({
                    'title': 'Ad Targeting Optimization',
                    'description': 'Use browsing history to create more relevant ad targeting profiles',
                    'value_proposition': 'Improve ad conversion rates and reduce marketing waste'
                })
            
            # Kalender-Daten: Zeitplanung, Produktivitätsanalyse
            elif source_type == 'calendar':
                use_cases.append({
                    'title': 'Productivity Pattern Analysis',
                    'description': 'Identify optimal meeting times and productivity patterns',
                    'value_proposition': 'Improve organizational efficiency and meeting scheduling'
                })
                use_cases.append({
                    'title': 'Work-Life Balance Insights',
                    'description': 'Analyze time allocation between work and personal activities',
                    'value_proposition': 'Develop better wellness and productivity programs'
                })
            
            # Gesundheitsdaten: Medizinische Forschung, Versicherungsanalyse
            elif source_type in ['health_data', 'health_insurance', 'smartwatch']:
                use_cases.append({
                    'title': 'Health Trend Analysis',
                    'description': 'Identify health patterns and correlations in anonymized health data',
                    'value_proposition': 'Support medical research and health intervention programs'
                })
                use_cases.append({
                    'title': 'Wellness Program Optimization',
                    'description': 'Use activity and health data to design better wellness initiatives',
                    'value_proposition': 'Improve health outcomes and reduce healthcare costs'
                })
            
            # IoT-Daten: Smart Home, Energieoptimierung
            elif source_type in ['iot_vacuum', 'iot_thermostat', 'iot_lighting', 'smart_home']:
                use_cases.append({
                    'title': 'Energy Usage Optimization',
                    'description': 'Analyze home energy consumption patterns',
                    'value_proposition': 'Develop more efficient energy management solutions'
                })
                use_cases.append({
                    'title': 'Smart Home Product Development',
                    'description': 'Understand usage patterns of smart home devices',
                    'value_proposition': 'Inform the design of new IoT products and services'
                })
            
            # Social Media und Chat: Stimmungsanalyse, Trendforschung
            elif source_type in ['social_media', 'chat']:
                use_cases.append({
                    'title': 'Sentiment Analysis and Trend Detection',
                    'description': 'Track public opinion and emerging topics in social conversations',
                    'value_proposition': 'Stay ahead of market trends and consumer sentiment shifts'
                })
                use_cases.append({
                    'title': 'Content Strategy Optimization',
                    'description': 'Analyze engagement patterns to optimize content strategy',
                    'value_proposition': 'Improve audience engagement and content performance'
                })
            
            # Streaming: Medienkonsum, Empfehlungen
            elif source_type == 'streaming':
                use_cases.append({
                    'title': 'Content Preference Analysis',
                    'description': 'Understand viewer preferences and consumption patterns',
                    'value_proposition': 'Optimize content creation and acquisition strategies'
                })
                use_cases.append({
                    'title': 'Recommendation Engine Improvement',
                    'description': 'Enhance content recommendation algorithms with viewing pattern data',
                    'value_proposition': 'Increase viewer satisfaction and platform engagement'
                })
            
            # Generische Anwendungsfälle für andere Quellen
            else:
                use_cases.append({
                    'title': 'Pattern Recognition',
                    'description': 'Identify patterns and trends in the data',
                    'value_proposition': 'Gain insights for business strategy and decision making'
                })
                use_cases.append({
                    'title': 'Predictive Analytics',
                    'description': 'Use historical data to predict future trends',
                    'value_proposition': 'Improve forecasting and strategic planning'
                })
            
            metadata['potential_use_cases'] = use_cases
            
            # Erstelle einen Asset für die Compute-to-Data-Funktionalität
            c2d_asset = self.c2d_manager.create_data_asset(
                anonymized_data,
                asset_metadata=metadata
            )
            
            return {
                'anonymized_data': anonymized_data,
                'metadata': metadata,
                'c2d_asset': c2d_asset
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenaufbereitung für Monetarisierung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'anonymized_data': None,
                'metadata': {
                    'source_type': source_type,
                    'privacy_level': privacy_level,
                    'error': str(e)
                }
            }
    
    def combine_data_sources(self, sources: List[Dict], combination_type: str = 'merge') -> Dict:
        """
        Kombiniert mehrere Datenquellen zu einem wertvolleren Datensatz.
        
        Args:
            sources: Liste von Datenquellen mit 'data' und 'metadata' Schlüsseln
            combination_type: Art der Kombination ('merge', 'enrich', 'correlate')
            
        Returns:
            Kombinierte Daten mit verbesserten Metadaten
        """
        try:
            self.logger.info(f"Kombiniere {len(sources)} Datenquellen mit Methode {combination_type}")
            
            # Prüfe, ob gültige Quellen vorhanden sind
            valid_sources = [s for s in sources if 'anonymized_data' in s and s['anonymized_data'] is not None]
            
            if len(valid_sources) < 2:
                raise ValueError("Mindestens zwei gültige Datenquellen werden für die Kombination benötigt")
            
            combined_data = None
            combined_metadata = {
                'combination_type': combination_type,
                'source_count': len(valid_sources),
                'source_types': [s['metadata']['source_type'] for s in valid_sources],
                'created_at': datetime.now().isoformat(),
                'record_count': 0,
                'field_count': 0,
                'estimated_value': 0.0,
                'combination_description': '',
                'source_metadata': [s['metadata'] for s in valid_sources]
            }
            
            # Implementierung je nach Kombinationstyp
            if combination_type == 'merge':
                # Horizontale Zusammenführung (Zeilen anfügen)
                # Geeignet für gleichartige Datensätze
                
                all_data = [s['anonymized_data'] for s in valid_sources]
                
                # Prüfe Spaltenkompatibilität
                common_columns = set.intersection(*[set(df.columns) for df in all_data])
                
                if not common_columns:
                    raise ValueError("Keine gemeinsamen Spalten für die Zusammenführung gefunden")
                
                # Zusammenführung der Daten mit gemeinsamen Spalten
                combined_data = pd.concat([df[list(common_columns)] for df in all_data], ignore_index=True)
                
                combined_metadata['combination_description'] = f"Horizontal merge of {len(valid_sources)} similar datasets, keeping {len(common_columns)} common columns"
                combined_metadata['record_count'] = len(combined_data)
                combined_metadata['field_count'] = len(combined_data.columns)
                
                # Wertschätzung: Summe der Einzelwerte mit einem Multiplikator für die Kombination
                base_value = sum(s['metadata'].get('estimated_value', 0) for s in valid_sources)
                combined_metadata['estimated_value'] = base_value * 1.2  # 20% Mehrwert durch Kombination
                
            elif combination_type == 'enrich':
                # Vertikale Anreicherung (Spalten anfügen)
                # Geeignet für komplementäre Datensätze mit einem gemeinsamen Identifikator
                
                # Erster Datensatz als Basis
                base_data = valid_sources[0]['anonymized_data']
                join_key = None
                
                # Suche nach einem gemeinsamen Schlüssel für die Verknüpfung
                for col in base_data.columns:
                    if all(col in s['anonymized_data'].columns for s in valid_sources[1:]):
                        # Prüfe, ob der Wert als Schlüssel dienen kann (eindeutige Werte)
                        if all(s['anonymized_data'][col].nunique() / len(s['anonymized_data']) > 0.7 for s in valid_sources):
                            join_key = col
                            break
                
                if not join_key:
                    # Falls kein expliziter Schlüssel existiert, erstelle einen künstlichen Schlüssel
                    self.logger.warning("Kein natürlicher Schlüssel gefunden, erstelle künstlichen Index")
                    
                    # Erstelle einen eindeutigen Index für jeden Datensatz
                    for i, source in enumerate(valid_sources):
                        source['anonymized_data']['temp_join_key'] = range(len(source['anonymized_data']))
                    
                    join_key = 'temp_join_key'
                
                # Führe die Daten zusammen
                combined_data = base_data.copy()
                
                for source in valid_sources[1:]:
                    # Finde neue Spalten, die noch nicht existieren
                    new_columns = [col for col in source['anonymized_data'].columns if col not in combined_data.columns]
                    
                    if new_columns:
                        # Verknüpfe über den Schlüssel mit linkem Join (behält alle Zeilen von combined_data)
                        combined_data = pd.merge(
                            combined_data,
                            source['anonymized_data'][new_columns + [join_key]],
                            on=join_key,
                            how='left'
                        )
                
                # Entferne den temporären Schlüssel, falls erstellt
                if join_key == 'temp_join_key' and join_key in combined_data.columns:
                    combined_data = combined_data.drop(columns=[join_key])
                
                combined_metadata['combination_description'] = f"Vertical enrichment using key '{join_key}', adding {len(combined_data.columns) - len(base_data.columns)} new features"
                combined_metadata['record_count'] = len(combined_data)
                combined_metadata['field_count'] = len(combined_data.columns)
                
                # Wertschätzung: Wert der Basis plus zusätzlicher Wert für neue Spalten
                base_value = valid_sources[0]['metadata'].get('estimated_value', 0)
                additional_value = sum(s['metadata'].get('estimated_value', 0) * 0.6 for s in valid_sources[1:])  # 60% des Werts der zusätzlichen Quellen
                combined_metadata['estimated_value'] = base_value + additional_value
                
            elif combination_type == 'correlate':
                # Korrelationsanalyse zwischen verschiedenen Datenquellen
                # Erstellt einen neuen Datensatz mit Korrelationsmaßen und abgeleiteten Features
                
                # Erstelle eine Liste für Korrelationsergebnisse
                correlation_results = []
                feature_pairs = []
                
                # Suche nach numerischen Spalten in jedem Datensatz
                for i, source1 in enumerate(valid_sources):
                    for j in range(i+1, len(valid_sources)):
                        source2 = valid_sources[j]
                        
                        # Extrahiere numerische Spalten
                        num_cols1 = source1['anonymized_data'].select_dtypes(include=['number']).columns
                        num_cols2 = source2['anonymized_data'].select_dtypes(include=['number']).columns
                        
                        # Prüfe, ob genügend numerische Spalten vorhanden sind
                        if len(num_cols1) == 0 or len(num_cols2) == 0:
                            continue
                        
                        # Wähle die Top-5-Spalten pro Quelle für die Analyse
                        select_cols1 = list(num_cols1)[:5]
                        select_cols2 = list(num_cols2)[:5]
                        
                        # Erstelle ein DataFrame mit Korrelationswerten
                        source_type1 = source1['metadata']['source_type']
                        source_type2 = source2['metadata']['source_type']
                        
                        for col1 in select_cols1:
                            for col2 in select_cols2:
                                # Erstelle 100 zufällige Datenpunkte für jede Spalte
                                # (für eine Stichprobenkorrelation ohne Verknüpfung der tatsächlichen Daten)
                                sample1 = source1['anonymized_data'][col1].sample(min(100, len(source1['anonymized_data']))).reset_index(drop=True)
                                sample2 = source2['anonymized_data'][col2].sample(min(100, len(source2['anonymized_data']))).reset_index(drop=True)
                                
                                # Berechne die Korrelation
                                try:
                                    corr = sample1.corr(sample2)
                                    
                                    # Speichere nur signifikante Korrelationen
                                    if abs(corr) > 0.3:
                                        corr_entry = {
                                            'source1': source_type1,
                                            'source2': source_type2,
                                            'feature1': col1,
                                            'feature2': col2,
                                            'correlation': float(corr),
                                            'abs_correlation': float(abs(corr))
                                        }
                                        correlation_results.append(corr_entry)
                                        feature_pairs.append((source1, col1, source2, col2))
                                except:
                                    # Ignoriere Fehler bei der Korrelationsberechnung
                                    pass
                
                # Erstelle ein neues DataFrame mit den Korrelationsergebnissen
                if correlation_results:
                    combined_data = pd.DataFrame(correlation_results)
                    
                    # Füge abgeleitete Features für die Top-Korrelationen hinzu
                    if feature_pairs:
                        # Sortiere nach absoluter Korrelationsstärke
                        sorted_pairs = sorted(list(zip(correlation_results, feature_pairs)), 
                                             key=lambda x: x[0]['abs_correlation'], 
                                             reverse=True)
                        
                        # Wähle die Top-10-Paare (oder weniger, falls nicht genügend vorhanden)
                        top_pairs = sorted_pairs[:min(10, len(sorted_pairs))]
                        
                        # Erstelle für jedes Top-Paar ein kombiniertes Feature
                        for i, (corr_info, (source1, col1, source2, col2)) in enumerate(top_pairs):
                            # Normalisiere die Werte
                            samples1 = source1['anonymized_data'][col1].sample(min(100, len(source1['anonymized_data']))).reset_index(drop=True)
                            samples2 = source2['anonymized_data'][col2].sample(min(100, len(source2['anonymized_data']))).reset_index(drop=True)
                            
                            # Z-Score-Normalisierung
                            normalized1 = (samples1 - samples1.mean()) / samples1.std()
                            normalized2 = (samples2 - samples2.mean()) / samples2.std()
                            
                            # Erstelle ein kombiniertes Feature basierend auf der Korrelationsrichtung
                            if corr_info['correlation'] >= 0:
                                combined_feature = normalized1 + normalized2
                            else:
                                combined_feature = normalized1 - normalized2
                            
                            # Füge das kombinierte Feature dem Ergebnis hinzu
                            feature_name = f"combined_{source1['metadata']['source_type']}_{col1}_{source2['metadata']['source_type']}_{col2}"
                            combined_data[feature_name] = combined_feature
                    
                    combined_metadata['combination_description'] = f"Correlation analysis between {len(valid_sources)} datasets, identifying {len(correlation_results)} significant cross-source relationships"
                    combined_metadata['record_count'] = len(combined_data)
                    combined_metadata['field_count'] = len(combined_data.columns)
                    
                    # Wertschätzung: Basierend auf der Anzahl und Stärke der gefundenen Korrelationen
                    avg_correlation = np.mean([r['abs_correlation'] for r in correlation_results]) if correlation_results else 0
                    correlation_value = len(correlation_results) * avg_correlation * 0.5  # Grundwert pro signifikante Korrelation
                    
                    # Füge Werte der Originalquellen hinzu
                    base_value = sum(s['metadata'].get('estimated_value', 0) * 0.3 for s in valid_sources)  # 30% der Originalwerte
                    combined_metadata['estimated_value'] = base_value + correlation_value
                else:
                    raise ValueError("Keine signifikanten Korrelationen zwischen den Datenquellen gefunden")
            
            else:
                raise ValueError(f"Nicht unterstützter Kombinationstyp: {combination_type}")
            
            # Erstelle einen neuen C2D-Asset für den kombinierten Datensatz
            c2d_asset = self.c2d_manager.create_data_asset(
                combined_data,
                asset_metadata=combined_metadata
            )
            
            return {
                'anonymized_data': combined_data,
                'metadata': combined_metadata,
                'c2d_asset': c2d_asset
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Kombination von Datenquellen: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'anonymized_data': None,
                'metadata': {
                    'combination_type': combination_type,
                    'source_count': len(sources),
                    'error': str(e)
                }
            }
    
    def estimate_data_value(self, data: pd.DataFrame, metadata: Dict = None) -> Dict:
        """
        Berechnet einen detaillierten Wert für einen Datensatz basierend auf mehreren Faktoren.
        
        Args:
            data: Der zu bewertende Datensatz
            metadata: Optionale Metadaten über den Datensatz
            
        Returns:
            Detaillierte Wertschätzung mit Begründungen
        """
        try:
            self.logger.info(f"Berechne detaillierten Datenwert für Datensatz mit {len(data)} Zeilen und {len(data.columns)} Spalten")
            
            if metadata is None:
                metadata = {}
            
            # Basisfaktoren für die Wertschätzung
            value_metrics = {
                'data_volume': {
                    'score': 0.0,
                    'weight': 0.15,
                    'explanation': ''
                },
                'data_quality': {
                    'score': 0.0,
                    'weight': 0.2,
                    'explanation': ''
                },
                'data_uniqueness': {
                    'score': 0._group_size']].index
            
            means = grouped.mean().loc[valid_groups]
            
            # Füge Rauschen hinzu
            noisy_means = self._add_differential_privacy_noise(means)
            
            return {'group_means': noisy_means.to_dict()}
        else:
            means = prepared_data.mean()
            
            # Füge Rauschen hinzu
            noisy_means = self._add_differential_privacy_noise(means)
            
            return {'means': noisy_means.to_dict()}
    
    def _sum(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Summe für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        sums = prepared_data.sum()
        
        # Füge Rauschen hinzu
        noisy_sums = self._add_differential_privacy_noise(sums)
        
        return {'sums': noisy_sums.to_dict()}
    
    def _min(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet das Minimum für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        minimums = prepared_data.min()
        
        # Hier kein Rauschen, da Minimum keine sensiblen Informationen preisgibt
        return {'minimums': minimums.to_dict()}
    
    def _max(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet das Maximum für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        maximums = prepared_data.max()
        
        # Hier kein Rauschen, da Maximum keine sensiblen Informationen preisgibt
        return {'maximums': maximums.to_dict()}
    
    def _std(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Standardabweichung für numerische Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        std_devs = prepared_data.std()
        
        # Füge Rauschen hinzu
        noisy_std_devs = self._add_differential_privacy_noise(std_devs)
        
        return {'standard_deviations': noisy_std_devs.to_dict()}
    
    def _correlation(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet die Korrelation zwischen numerischen Spalten"""
        columns = params.get('columns')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        # Verwende nur numerische Spalten
        numeric_data = prepared_data.select_dtypes(include=['number'])
        
        if numeric_data.empty or numeric_data.shape[1] < 2:
            return {'error': 'Nicht genügend numerische Spalten für Korrelation'}
        
        corr_matrix = numeric_data.corr()
        
        # Füge Rauschen hinzu
        noisy_corr = self._add_differential_privacy_noise(corr_matrix)
        
        # Stelle sicher, dass Werte im gültigen Bereich liegen
        noisy_corr = noisy_corr.clip(-1, 1)
        
        return {'correlation_matrix': noisy_corr.to_dict()}
    
    def _histogram(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Erstellt ein Histogramm für eine Spalte"""
        column = params.get('column')
        bins = params.get('bins', 10)
        
        if column not in data.columns:
            return {'error': f'Spalte {column} nicht gefunden'}
        
        prepared_data = self._prepare_data_for_computation(data, [column])
        
        if pd.api.types.is_numeric_dtype(prepared_data[column]):
            # Numerisches Histogramm
            hist, bin_edges = np.histogram(prepared_data[column], bins=bins)
            
            # Füge Rauschen hinzu
            noisy_hist = self._add_differential_privacy_noise(hist).astype(int)
            
            # Stelle sicher, dass Werte nicht negativ sind
            noisy_hist = np.maximum(0, noisy_hist)
            
            return {
                'histogram': noisy_hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        else:
            # Kategorisches Histogramm
            value_counts = prepared_data[column].value_counts()
            
            # Prüfe auf zu kleine Gruppen
            value_counts = value_counts[value_counts >= self.privacy_config['min_group_size']]
            
            if value_counts.empty:
                return {'error': 'Keine Kategorien mit ausreichender Größe'}
            
            # Füge Rauschen hinzu
            noisy_counts = self._add_differential_privacy_noise(value_counts).astype(int)
            
            # Stelle sicher, dass Werte nicht negativ sind
            noisy_counts = noisy_counts.clip(0)
            
            return {
                'categories': value_counts.index.tolist(),
                'counts': noisy_counts.tolist()
            }
    
    def _custom_model_inference(self, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Führt Inferenz mit einem benutzerdefinierten Modell durch, ohne die Daten preiszugeben
        
        Args:
            data: Eingabedaten für die Inferenz
            params: Parameter für die Inferenz, einschließlich des Modelltyps und der Modellparameter
            
        Returns:
            Ergebnisse der Inferenz
        """
        model_type = params.get('model_type')
        model_config = params.get('model_config', {})
        columns = params.get('columns')
        
        if not model_type:
            return {'error': 'Kein Modelltyp angegeben'}
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        try:
            results = None
            
            if model_type == 'anomaly_detection':
                # Anomalieerkennung mit Isolation Forest
                detector = AnomalyDetector(method='isolation_forest', contamination=model_config.get('contamination', 0.05))
                detector.fit(prepared_data)
                predictions = detector.predict(prepared_data)
                
                # Aggregiere Ergebnisse für den Datenschutz
                anomaly_count = np.sum(predictions == -1)
                total_count = len(predictions)
                
                # Füge Rauschen hinzu
                noisy_anomaly_count = self._add_differential_privacy_noise(anomaly_count).astype(int)
                
                results = {
                    'anomaly_percentage': round(100 * noisy_anomaly_count / total_count, 2),
                    'total_records': total_count
                }
            
            elif model_type == 'clustering':
                # K-Means-Clustering
                n_clusters = model_config.get('n_clusters', 3)
                
                # Verarbeite nur numerische Spalten
                numeric_data = prepared_data.select_dtypes(include=['number'])
                
                if numeric_data.empty:
                    return {'error': 'Keine numerischen Spalten für Clustering gefunden'}
                
                # Standardisiere Daten
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                # Cluster-Analyse
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Cluster-Statistiken (datenschutzfreundlich)
                cluster_sizes = {}
                for i in range(n_clusters):
                    size = np.sum(clusters == i)
                    
                    # Prüfe Mindestgruppengröße
                    if size >= self.privacy_config['min_group_size']:
                        # Füge Rauschen hinzu
                        noisy_size = self._add_differential_privacy_noise(size).astype(int)
                        cluster_sizes[f'cluster_{i}'] = max(0, noisy_size)
                
                # Cluster-Zentren (keine persönlichen Daten)
                cluster_centers = kmeans.cluster_centers_
                
                results = {
                    'cluster_sizes': cluster_sizes,
                    'cluster_centers': cluster_centers.tolist()
                }
            
            elif model_type == 'sentiment_analysis':
                # Sentimentanalyse für Textdaten
                text_column = model_config.get('text_column')
                
                if text_column not in prepared_data.columns:
                    return {'error': f'Textspalte {text_column} nicht gefunden'}
                
                # Einfache Sentimentanalyse mit NLTK
                analyzer = SemanticAnalyzer()
                sentiment_results = analyzer.analyze_sentiment(prepared_data[text_column].tolist())
                
                # Aggregierte Ergebnisse für den Datenschutz
                sentiment_counts = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
                
                for result in sentiment_results:
                    sentiment_counts[result['sentiment']] += 1
                
                # Füge Rauschen hinzu
                for sentiment in sentiment_counts:
                    noisy_count = self._add_differential_privacy_noise(sentiment_counts[sentiment]).astype(int)
                    sentiment_counts[sentiment] = max(0, noisy_count)
                
                results = {
                    'sentiment_distribution': sentiment_counts,
                    'total_records': len(sentiment_results)
                }
            
            else:
                return {'error': f'Nicht unterstützter Modelltyp: {model_type}'}
            
            return results
            
        except Exception as e:
            logger.error(f"Fehler bei der Modell-Inferenz: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Modell-Inferenz: {str(e)}'}
    
    def execute_operation(self, encrypted_data: bytes, operation: str, params: Dict) -> Dict:
        """
        Führt eine Operation auf verschlüsselten Daten aus, ohne sie zu entschlüsseln
        
        Args:
            encrypted_data: Verschlüsselte Daten
            operation: Name der auszuführenden Operation
            params: Parameter für die Operation
            
        Returns:
            Ergebnisse der Operation
        """
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
            logger.error(f"Fehler bei der Ausführung der Operation {operation}: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Ausführung: {str(e)}'}
    
    def create_data_asset(self, data: pd.DataFrame, asset_metadata: Dict = None) -> Dict:
        """
        Erstellt einen verschlüsselten Daten-Asset mit Metadaten für den Marktplatz
        
        Args:
            data: Zu verschlüsselnde Daten
            asset_metadata: Metadaten für den Asset
            
        Returns:
            Asset-Details mit ID und Metadaten
        """
        try:
            # Erstelle eine eindeutige ID für den Asset
            asset_id = uuid.uuid4().hex
            
            # Verschlüssele die Daten
            encrypted_data = self._encrypt_data(data)
            
            # Erstelle Metadaten mit datenschutzfreundlichen Statistiken
            stats = {}
            
            # Anzahl der Datensätze
            stats['record_count'] = len(data)
            
            # Spalteninformationen
            stats['columns'] = []
            for col in data.columns:
                col_info = {
                    'name': col,
                    'type': str(data[col].dtype)
                }
                
                # Füge datenschutzfreundliche Statistiken hinzu
                if pd.api.types.is_numeric_dtype(data[col]):
                    # Füge Rauschen hinzu, um Datenschutz zu gewährleisten
                    col_info['min'] = float(self._add_differential_privacy_noise(data[col].min()))
                    col_info['max'] = float(self._add_differential_privacy_noise(data[col].max()))
                    col_info['mean'] = float(self._add_differential_privacy_noise(data[col].mean()))
                    col_info['std'] = float(self._add_differential_privacy_noise(data[col].std()))
                else:
                    # Für kategorische Daten nur die Anzahl eindeutiger Werte
                    col_info['unique_values'] = int(self._add_differential_privacy_noise(data[col].nunique()))
                
                stats['columns'].append(col_info)
            
            # Zusammenführen mit benutzerdefinierten Metadaten
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
            logger.error(f"Fehler bei der Erstellung des Daten-Assets: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': f'Fehler bei der Asset-Erstellung: {str(e)}'}
    
    def generate_access_token(self, asset_id: str, allowed_operations: List[str], 
                            expiration_time: int = 3600) -> Dict:
        """
        Generiert ein temporäres Zugriffstoken für einen Daten-Asset
        
        Args:
            asset_id: ID des Daten-Assets
            allowed_operations: Liste der erlaubten Operationen
            expiration_time: Ablaufzeit in Sekunden
            
        Returns:
            Token-Details mit Token und Ablaufzeit
        """
        # Prüfe, ob alle angeforderten Operationen unterstützt werden
        unsupported_ops = [op for op in allowed_operations if op not in self.allowed_operations]
        if unsupported_ops:
            return {'error': f'Nicht unterstützte Operationen: {", ".join(unsupported_ops)}'}
        
        # Erstelle Token-Daten
        token_data = {
            'asset_id': asset_id,
            'allowed_operations': allowed_operations,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=expiration_time)).isoformat(),
            'token_id': uuid.uuid4().hex
        }
        
        # Verschlüssele Token-Daten
        token_json = json.dumps(token_data).encode()
        encrypted_token = self.cipher_suite.encrypt(token_json)
        
        return {
            'token': encrypted_token.decode(),
            'token_id': token_data['token_id'],
            'expires_at': token_data['expires_at']
        }
    
    def validate_access_token(self, token: str, operation: str) -> bool:
        """
        Validiert ein Zugriffstoken für eine bestimmte Operation
        
        Args:
            token: Zugriffstoken
            operation: Angeforderte Operation
            
        Returns:
            True, wenn das Token gültig ist und die Operation erlaubt ist
        """
        try:
            # Entschlüssele das Token
            decrypted_token = self.cipher_suite.decrypt(token.encode())
            token_data = json.loads(decrypted_token.decode())
            
            # Prüfe, ob das Token abgelaufen ist
            expiry_time = datetime.fromisoformat(token_data['expires_at'])
            if datetime.now() > expiry_time:
                logger.warning(f"Zugriffstoken abgelaufen: {token_data['token_id']}")
                return False
            
            # Prüfe, ob die Operation erlaubt ist
            if operation not in token_data['allowed_operations']:
                logger.warning(f"Operation {operation} nicht erlaubt für Token {token_data['token_id']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Token-Validierung: {str(e)}")
            return False

###########################################
# 6. Integration und OceanData-Kernmodule
###########################################

class OceanDataAI:
    """
    Hauptklasse, die alle KI-Module für OceanData integriert und koordiniert.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialisiert den OceanData-KI-Manager.
        
        Args:
            config: Konfigurationsparameter für die KI-Module
        """
        self.config = config or {}
        self.logger = logging.getLogger("OceanData.AI")
        
        # Instanziiere die einzelnen Module
        self.anomaly_detector = AnomalyDetector(
            method=self.config.get('anomaly_detection_method', 'isolation_forest'),
            contamination=self.config.get('anomaly_contamination', 0.05)
        )
        
        self.semantic_analyzer = SemanticAnalyzer(
            model_type=self.config.get('semantic_model', 'bert'),
            model_name=self.config.get('semantic_model_name', 'bert-base-uncased')
        )
        
        self.predictive_modeler = PredictiveModeler(
            model_type=self.config.get('predictive_model', 'lstm'),
            forecast_horizon=self.config.get('forecast_horizon', 7)
        )
        
        self.data_synthesizer = DataSynthesizer(
            categorical_threshold=self.config.get('categorical_threshold', 10),
            noise_dim=self.config.get('gan_noise_dim', 100)
        )
        
        self.c2d_manager = ComputeToDataManager()
        
        # Aktiviere GPU-Beschleunigung, falls verfügbar
        self._setup_hardware_acceleration()
    
    def _setup_hardware_acceleration(self):
        """Konfiguriert Hardware-Beschleunigung für die KI-Module"""
        try:
            # Prüfe, ob GPUs für TensorFlow verfügbar sind
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.logger.info(f"GPU-Beschleunigung aktiviert mit {len(gpus)} GPUs")
                
                # Konfiguriere TensorFlow für bessere GPU-Nutzung
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                self.logger.info("Keine GPUs gefunden, verwende CPU-Computing")
        
        except Exception as e:
            self.logger.warning(f"Fehler bei der GPU-Konfiguration: {str(e)}")
            self.logger.info("Verwende Standard-CPU-Computing")
    
    def analyze_data_source(self, data: pd.DataFrame, source_type: str) -> Dict:
        """
        Führt eine umfassende Analyse einer Datenquelle durch.
        
        Args:
            data: Die zu analysierenden Daten
            source_type: Typ der Datenquelle (browser, calendar, iot, etc.)
            
        Returns:
            Umfassende Analyseergebnisse
        """
        results = {
            'source_type': source_type,
            'timestamp': datetime.now().isoformat(),
            'record_count': len(data),
            'column_count': len(data.columns),
            'analyses': {}
        }
        
        try:
            # 1. Ausreißererkennung
            self.logger.info(f"Führe Ausreißererkennung für {source_type}-Daten durch")
            self.anomaly_detector.fit(data)
            anomaly_predictions = self.anomaly_detector.predict(data)
            anomaly_insights = self.anomaly_detector.get_anomaly_insights(
                data, anomaly_predictions, top_n_features=5
            )
            
            results['analyses']['anomalies'] = {
                'count': int(np.sum(anomaly_predictions == -1) if self.anomaly_detector.method == 'isolation_forest' else np.sum(anomaly_predictions)),
                'percentage': float(np.mean(anomaly_predictions == -1) * 100 if self.anomaly_detector.method == 'isolation_forest' else np.mean(anomaly_predictions) * 100),
                'insights': anomaly_insights[:5]  # Begrenze auf die Top-5-Insights
            }
            
            # 2. Textanalyse für relevante Textspalten
            text_columns = [col for col in data.columns 
                           if data[col].dtype == 'object' 
                           and data[col].str.len().mean() > 20]
            
            if text_columns:
                self.logger.info(f"Führe Textanalyse für {len(text_columns)} Spalten durch")
                text_analyses = {}
                
                for col in text_columns[:3]:  # Begrenze auf die ersten 3 Textspalten
                    # Wähle Texte aus, die nicht leer sind
                    texts = data[col].dropna().astype(str).tolist()
                    if not texts:
                        continue
                        
                    # Stichprobe für Performance
                    sample_size = min(100, len(texts))
                    text_sample = random.sample(texts, sample_size)
                    
                    # Sentimentanalyse
                    sentiments = self.semantic_analyzer.analyze_sentiment(text_sample)
                    
                    # Themenextraktion
                    topics = self.semantic_analyzer.extract_topics(text_sample, num_topics=3)
                    
                    text_analyses[col] = {
                        'sentiment': {
                            'positive': len([s for s in sentiments if s['sentiment'] == 'positive']) / len(sentiments),
                            'neutral': len([s for s in sentiments if s['sentiment'] == 'neutral']) / len(sentiments),
                            'negative': len([s for s in sentiments if s['sentiment'] == 'negative']) / len(sentiments)
                        },
                        'topics': topics
                    }
                
                results['analyses']['text'] = text_analyses
            
            # 3. Zeitreihenanalyse für Zeitstempel-basierte Daten
            datetime_columns = [col for col in data.columns 
                              if pd.api.types.is_datetime64_any_dtype(data[col])]
            
            if datetime_columns and len(data) > 10:
                self.logger.info("Führe Zeitreihenanalyse durch")
                
                # Wähle die erste Zeitstempelspalte für die Analyse
                time_col = datetime_columns[0]
                
                # Sortiere Daten nach Zeit
                sorted_data = data.sort_values(by=time_col)
                
                # Wähle numerische Spalten für die Vorhersage
                numeric_cols = sorted_data.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    # Wähle die ersten 3 numerischen Spalten für die Demonstration
                    target_cols = numeric_cols[:3]
                    
                    # Erstelle Zeitreihen-Features
                    sorted_data['hour'] = sorted_data[time_col].dt.hour
                    sorted_data['day_of_week'] = sorted_data[time_col].dt.dayofweek
                    sorted_data['day_of_month'] = sorted_data[time_col].dt.day
                    sorted_data['month'] = sorted_data[time_col].dt.month
                    
                    # Bilde Feature- und Zielspalten
                    X = sorted_data[['hour', 'day_of_week', 'day_of_month', 'month']]
                    y = sorted_data[target_cols]
                    
                    if len(X) > 20:  # Mindestanzahl für sinnvolle Zeitreihenanalyse
                        # Trainiere ein einfaches Modell
                        self.predictive_modeler.fit(X, y, lookback=5, epochs=20, verbose=0)
                        
                        # Forecasting für die nächsten 7 Tage
                        forecast = self.predictive_modeler.forecast(X, steps=7)
                        
                        results['analyses']['time_series'] = {
                            'forecast_horizon': 7,
                            'forecasted_features': target_cols,
                            'forecasts': {
                                col: forecast[:, i].tolist() 
                                for i, col in enumerate(target_cols)
                            }
                        }
            
            # 4. Datensyntheseanalyse (optional, ressourcenintensiv)
            if self.config.get('enable_data_synthesis', False) and len(data) > 50:
                self.logger.info("Führe Datensynthese-Analyse durch")
                
                # Wähle eine Stichprobe und maximal 10 Spalten für die Demonstration
                sample_size = min(1000, len(data))
                data_sample = data.sample(sample_size)
                
                if len(data.columns) > 10:
                    # Wähle die wichtigsten Spalten aus
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()[:5]
                    cat_cols = data.select_dtypes(exclude=['number']).columns.tolist()[:5]
                    selected_cols = numeric_cols + cat_cols
                    data_sample = data_sample[selected_cols]
                
                # Trainiere das GAN mit einer kleinen Anzahl von Epochen
                self.data_synthesizer.fit(data_sample, epochs=100, verbose=0)
                
                # Generiere einige synthetische Beispiele
                synthetic_examples = self.data_synthesizer.generate(10)
                
                # Bewerte die Qualität
                quality_metrics = self.data_synthesizer.evaluate_quality(100)
                
                results['analyses']['data_synthesis'] = {
                    'quality_score': quality_metrics.get('overall_quality_score', 0),
                    'sample_examples': synthetic_examples.head(3).to_dict()
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenanalyse: {str(e)}")
            self.logger.error(traceback.format_exc())
            results['error'] = str(e)
            return results
    
    def prepare_data_for_monetization(self, data: pd.DataFrame, source_type: str, 
                                    privacy_level: str = 'high') -> Dict:
        """
        Bereitet Daten für die Monetarisierung vor, einschließlich Anonymisierung und Metadatengenerierung.
        
        Args:
            data: Die zu monetarisierenden Daten
            source_type: Typ der Datenquelle
            privacy_level: Datenschutzniveau ('low', 'medium', 'high')
            
        Returns:
            Aufbereitete Daten mit Metadaten für die Monetarisierung
        """
        try:
            self.logger.info(f"Bereite {source_type}-Daten für Monetarisierung vor (Privacy: {privacy_level})")
            
            # 1. Datenschutzmaßnahmen basierend auf dem gewählten Niveau
            anonymized_data = data.copy()
            
            # Identifiziere sensible Spalten
            id_columns = [col for col in data.columns 
                         if 'id' in col.lower() or 'user' in col.lower() or 'email' in col.lower()]
            location_columns = [col for col in data.columns 
                              if 'location' in col.lower() or 'address' in col.lower() or 'gps' in col.lower() or 'ip' in col.lower()]
            datetime_columns = [col for col in data.columns 
                              if pd.api.types.is_datetime64_any_dtype(data[col])]
            
            # Anonymisierung basierend auf Datenschutzniveau
            if privacy_level == 'high':
                # Vollständige Anonymisierung und Entfernung sensibler Daten
                anonymized_data = anonymized_data.drop(columns=id_columns, errors='ignore')
                
                # Anonymisiere Standortdaten
                for col in location_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: f"location_{hashlib.md5(str(x).encode()).hexdigest()[:8]}" if pd.notna(x) else x
                        )
                
                # Vergröbere Zeitstempel (nur auf Tagesebene)
                for col in datetime_columns:
                    if col in anonymized_data.columns:
                        anonymized_data[col] = pd.to_datetime(anonymized_data[col]).dt.floor('D')
                
            elif privacy_level == 'medium':
                # Moderate Anonymisierung
                for col in id_columns:
                    if col in anonymized_data.columns:
                        salt = datetime.now().strftime("%Y%m%d")
                        anonymized_data[col] = anonymized_data[col].apply(
                            lambda x: hashlib.sha256((str(x) + salt).encode()).hexdigest() if pd.notna(x) else x
                        )
                
                # Vergröbere Standortdaten
                for col in location_columns:
                    if col in anonymized_data.columns and anonymized_data[col].dtype == 'object':
                """
OceanData - Fortgeschrittene KI-Module für Datenanalyse und -monetarisierung

Diese Module erweitern den Kernalgorithmus von OceanData um fortschrittliche KI-Funktionen:
1. Anomalieerkennung
2. Semantische Datenanalyse 
3. Prädiktive Modellierung
4. Datensynthese und Erweiterung
5. Multimodale Analyse
6. Federated Learning und Compute-to-Data
7. Kontinuierliches Lernen und Modellanpassung
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertTokenizer, BertModel, TFBertModel
from transformers import GPT2Tokenizer, GPT2Model, TFGPT2Model
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import json
from typing import Dict, List, Union, Any, Tuple, Optional
import logging
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import h5py
import uuid
import traceback
import warnings

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Ignoriere TensorFlow und PyTorch Warnungen
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger("OceanData.AI")

###########################################
# 1. Anomalieerkennung & Outlier Detection
###########################################

class AnomalyDetector:
    """Klasse für die Erkennung von Anomalien in verschiedenen Datentypen"""
    
    def __init__(self, method: str = 'autoencoder', contamination: float = 0.05):
        """
        Initialisiert einen Anomaliedetektor.
        
        Args:
            method: Methode für die Anomalieerkennung ('autoencoder', 'isolation_forest', 'dbscan')
            contamination: Erwarteter Anteil an Anomalien in den Daten (für Isolation Forest)
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dims = None
        self.threshold = None
        
    def _build_autoencoder(self, input_dim):
        """Erstellt ein Autoencoder-Modell für die Anomalieerkennung"""
        encoding_dim = max(1, input_dim // 2)
        hidden_dim = max(1, encoding_dim // 2)
        
        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(inputs)
        encoded = layers.Dense(hidden_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Encoder model for extracting features
        encoder = keras.Model(inputs, encoded)
        
        return autoencoder, encoder
    
    def fit(self, X: pd.DataFrame, categorical_cols: List[str] = None):
        """
        Trainiert den Anomaliedetektor mit Daten.
        
        Args:
            X: Eingabedaten (DataFrame oder numpy array)
            categorical_cols: Liste der kategorischen Spalten, die One-Hot-kodiert werden müssen
        """
        try:
            # Vorbereiten der Daten
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten durch One-Hot-Encoding
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Konvertieren zu float und Behandlung von NaN-Werten
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            
            if X_prep.shape[1] == 0:
                raise ValueError("Nach der Vorverarbeitung sind keine numerischen Merkmale übrig")
                
            X_scaled = self.scaler.fit_transform(X_prep)
            self.feature_dims = X_scaled.shape[1]
            
            if self.method == 'autoencoder':
                # Autoencoder-Ansatz für Anomalieerkennung
                self.model, self.encoder = self._build_autoencoder(self.feature_dims)
                
                # Trainiere den Autoencoder
                self.model.fit(
                    X_scaled, X_scaled,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    verbose=0,
                    validation_split=0.1
                )
                
                # Berechne Rekonstruktionsfehler für alle Trainingsdaten
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Setze Threshold basierend auf den Rekonstruktionsfehlern
                self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)
                
            elif self.method == 'isolation_forest':
                # Isolation Forest für Outlier-Erkennung
                self.model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
                self.model.fit(X_scaled)
                
            elif self.method == 'dbscan':
                # DBSCAN für Clustering-basierte Outlier-Erkennung
                self.model = DBSCAN(
                    eps=0.5,  # Maximum distance between samples
                    min_samples=5,  # Minimum samples in a cluster
                    n_jobs=-1
                )
                self.model.fit(X_scaled)
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des Anomaliedetektors: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X: pd.DataFrame, categorical_cols: List[str] = None) -> np.ndarray:
        """
        Identifiziert Anomalien in den Daten.
        
        Args:
            X: Zu prüfende Daten
            categorical_cols: Liste der kategorischen Spalten
            
        Returns:
            Array mit Anomalie-Scores für jede Zeile (-1 für Anomalien, 1 für normale Daten bei Isolation Forest;
            True für Anomalien, False für normale Daten bei Autoencoder)
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Daten
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            # Konvertieren und Behandlung von NaN-Werten
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            
            # Anpassen der Features, falls nötig
            missing_cols = set(range(self.feature_dims)) - set(range(X_prep.shape[1]))
            if missing_cols:
                for col in missing_cols:
                    X_prep[f'missing_{col}'] = 0
            
            X_prep = X_prep.iloc[:, :self.feature_dims]
            X_scaled = self.scaler.transform(X_prep)
            
            if self.method == 'autoencoder':
                # Berechne Rekonstruktionsfehler
                reconstructions = self.model.predict(X_scaled)
                reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
                
                # Anomalie, wenn Fehler über dem Threshold
                return reconstruction_errors > self.threshold
                
            elif self.method == 'isolation_forest':
                # -1 für Anomalien, 1 für normale Daten
                return self.model.predict(X_scaled)
                
            elif self.method == 'dbscan':
                # Ordne neue Punkte den nächsten Clustern zu
                labels = self.model.fit_predict(X_scaled)
                # -1 sind Noise-Punkte (Anomalien)
                return labels == -1
                
        except Exception as e:
            logger.error(f"Fehler bei der Anomalievorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_anomaly_insights(self, X: pd.DataFrame, predictions: np.ndarray, 
                           categorical_cols: List[str] = None, top_n_features: int = 3) -> List[Dict]:
        """
        Liefert Erklärungen für die erkannten Anomalien.
        
        Args:
            X: Originaldaten
            predictions: Ergebnisse der predict()-Methode
            categorical_cols: Liste der kategorischen Spalten
            top_n_features: Anzahl der wichtigsten Features, die zurückgegeben werden sollen
            
        Returns:
            Liste mit Erklärungen für jede Anomalie
        """
        insights = []
        
        try:
            anomaly_indices = np.where(predictions == -1)[0] if self.method == 'isolation_forest' else np.where(predictions)[0]
            
            if len(anomaly_indices) == 0:
                return []
            
            X_prep = X.copy()
            X_orig = X.copy()  # Original-Daten für Berichte
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            X_scaled = self.scaler.transform(X_prep)
            
            # Allgemeine Statistiken berechnen
            mean_values = np.mean(X_scaled, axis=0)
            std_values = np.std(X_scaled, axis=0)
            
            for idx in anomaly_indices:
                if idx >= len(X_scaled):
                    continue
                    
                sample = X_scaled[idx]
                orig_sample = X_orig.iloc[idx]
                
                if self.method == 'autoencoder':
                    # Rekonstruktionsfehler pro Feature berechnen
                    reconstruction = self.model.predict(sample.reshape(1, -1))[0]
                    feature_errors = np.square(sample - reconstruction)
                    
                    # Features mit den größten Fehlern identifizieren
                    top_features_idx = np.argsort(feature_errors)[-top_n_features:]
                    top_features = [(X_prep.columns[i], feature_errors[i], sample[i], reconstruction[i]) for i in top_features_idx]
                    
                    insight = {
                        "index": int(idx),
                        "anomaly_score": float(np.mean(feature_errors)),
                        "important_features": [
                            {
                                "feature": feat,
                                "error": float(err),
                                "actual_value": float(val),
                                "expected_value": float(rec)
                            } for feat, err, val, rec in top_features
                        ],
                        "original_data": orig_sample.to_dict()
                    }
                    
                else:  # Isolation Forest oder DBSCAN
                    # Z-Scores berechnen, um ungewöhnliche Werte zu identifizieren
                    z_scores = (sample - mean_values) / std_values
                    
                    # Features mit den höchsten Z-Scores (positiv oder negativ)
                    abs_z_scores = np.abs(z_scores)
                    top_features_idx = np.argsort(abs_z_scores)[-top_n_features:]
                    top_features = [(X_prep.columns[i], z_scores[i], sample[i], mean_values[i]) for i in top_features_idx]
                    
                    insight = {
                        "index": int(idx),
                        "anomaly_score": float(np.max(abs_z_scores)),
                        "important_features": [
                            {
                                "feature": feat,
                                "z_score": float(z),
                                "actual_value": float(val),
                                "average_value": float(avg)
                            } for feat, z, val, avg in top_features
                        ],
                        "original_data": orig_sample.to_dict()
                    }
                
                insights.append(insight)
                
            return insights
            
        except Exception as e:
            logger.error(f"Fehler bei der Generierung von Anomalie-Insights: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def visualize_anomalies(self, X: pd.DataFrame, predictions: np.ndarray, 
                            categorical_cols: List[str] = None, 
                            save_path: str = None) -> plt.Figure:
        """
        Visualisiert erkannte Anomalien in den Daten.
        
        Args:
            X: Daten
            predictions: Anomalieerkennungsergebnisse
            categorical_cols: Kategorische Spalten
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        try:
            X_prep = X.copy()
            
            # Behandlung kategorischer Spalten
            if categorical_cols:
                X_prep = pd.get_dummies(X_prep, columns=categorical_cols)
            
            X_prep = X_prep.select_dtypes(include=['number']).fillna(0)
            X_scaled = self.scaler.transform(X_prep)
            
            # PCA für Dimensionsreduktion
            if X_scaled.shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
            else:
                X_pca = X_scaled
            
            # Anomalien bestimmen
            if self.method == 'isolation_forest':
                anomalies = (predictions == -1)
            else:
                anomalies = predictions
            
            # Visualisierung erstellen
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Normale Datenpunkte plotten
            ax.scatter(X_pca[~anomalies, 0], X_pca[~anomalies, 1], c='blue', label='Normal', alpha=0.5)
            
            # Anomalien plotten
            if np.any(anomalies):
                ax.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], c='red', label='Anomalien', alpha=0.7)
            
            # Grafik anpassen
            ax.set_title(f'Anomalieerkennung mit {self.method.capitalize()}')
            ax.set_xlabel('Hauptkomponente 1')
            ax.set_ylabel('Hauptkomponente 2')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung von Anomalien: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden")
            
        model_data = {
            "method": self.method,
            "contamination": self.contamination,
            "feature_dims": self.feature_dims,
            "threshold": self.threshold,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere das Model-Objekt
        if self.method in ['isolation_forest', 'dbscan']:
            joblib.dump(self.model, f"{path}_model.joblib")
        elif self.method == 'autoencoder':
            self.model.save(f"{path}_autoencoder")
            self.encoder.save(f"{path}_encoder")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        detector = cls(
            method=model_data['method'],
            contamination=model_data['contamination']
        )
        
        detector.feature_dims = model_data['feature_dims']
        detector.threshold = model_data['threshold']
        detector.is_fitted = model_data['is_fitted']
        
        # Lade das Model-Objekt
        if detector.method in ['isolation_forest', 'dbscan']:
            detector.model = joblib.load(f"{path}_model.joblib")
        elif detector.method == 'autoencoder':
            detector.model = keras.models.load_model(f"{path}_autoencoder")
            detector.encoder = keras.models.load_model(f"{path}_encoder")
        
        # Lade den Scaler
        detector.scaler = joblib.load(f"{path}_scaler.joblib")
        
        return detector


###########################################
# 2. Semantische Datenanalyse
###########################################

class SemanticAnalyzer:
    """Klasse für semantische Analyse von Text und anderen Daten mit Deep Learning"""
    
    def __init__(self, model_type: str = 'bert', model_name: str = 'bert-base-uncased'):
        """
        Initialisiert den semantischen Analysator.
        
        Args:
            model_type: Typ des zu verwendenden Modells ('bert', 'gpt2', 'custom')
            model_name: Name oder Pfad des vortrainierten Modells
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.max_length = 512  # Standard für BERT
        self.embeddings_cache = {}  # Cache für Texteinbettungen
        
        # Modell und Tokenizer laden
        self._load_model()
    
    def _load_model(self):
        """Lädt das Modell und den Tokenizer basierend auf dem ausgewählten Typ"""
        try:
            if self.model_type == 'bert':
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                if tf.test.is_gpu_available():
                    self.model = TFBertModel.from_pretrained(self.model_name)
                else:
                    self.model = BertModel.from_pretrained(self.model_name)
                self.max_length = 512
            
            elif self.model_type == 'gpt2':
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 hat kein Padding-Token
                if tf.test.is_gpu_available():
                    self.model = TFGPT2Model.from_pretrained(self.model_name)
                else:
                    self.model = GPT2Model.from_pretrained(self.model_name)
                self.max_length = 1024
            
            else:
                raise ValueError(f"Nicht unterstützter Modelltyp: {self.model_type}")
        
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells {self.model_name}: {str(e)}")
            # Fallback zu kleineren Modellen bei Speicher- oder Download-Problemen
            if self.model_type == 'bert':
                logger.info("Verwende ein kleineres BERT-Modell als Fallback")
                self.model_name = 'distilbert-base-uncased'
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertModel.from_pretrained(self.model_name)
            elif self.model_type == 'gpt2':
                logger.info("Verwende ein kleineres GPT-2-Modell als Fallback")
                self.model_name = 'distilgpt2'
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = GPT2Model.from_pretrained(self.model_name)
    
    def get_embeddings(self, texts: Union[str, List[str]], 
                      batch_size: int = 8, use_cache: bool = True) -> np.ndarray:
        """
        Erzeugt Einbettungen (Embeddings) für Texte.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            batch_size: Größe der Batches für die Verarbeitung
            use_cache: Ob bereits berechnete Einbettungen wiederverwendet werden sollen
            
        Returns:
            Array mit Einbettungen (shape: [n_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Initialisiere ein Array für die Ergebnisse
        all_embeddings = []
        texts_to_process = []
        texts_indices = []
        
        # Prüfe Cache für jede Anfrage
        for i, text in enumerate(texts):
            if use_cache and text in self.embeddings_cache:
                all_embeddings.append(self.embeddings_cache[text])
            else:
                texts_to_process.append(text)
                texts_indices.append(i)
        
        if texts_to_process:
            # Verarbeite Texte in Batches
            for i in range(0, len(texts_to_process), batch_size):
                batch_texts = texts_to_process[i:i+batch_size]
                batch_indices = texts_indices[i:i+batch_size]
                
                # Tokenisierung
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt" if isinstance(self.model, nn.Module) else "tf",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Modelloutput berechnen
                with torch.no_grad() if isinstance(self.model, nn.Module) else tf.device('/CPU:0'):
                    outputs = self.model(**inputs)
                
                # Embeddings aus dem letzten Hidden State extrahieren
                if self.model_type == 'bert':
                    # Verwende [CLS]-Token-Ausgabe als Satzrepräsentation (erstes Token)
                    if isinstance(self.model, nn.Module):
                        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                    else:
                        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                elif self.model_type == 'gpt2':
                    # Verwende den Durchschnitt aller Token-Repräsentationen
                    if isinstance(self.model, nn.Module):
                        embeddings = torch.mean(outputs.last_hidden_state, dim=1).numpy()
                    else:
                        embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
                
                # Füge die Embeddings an der richtigen Position ein
                for j, (idx, text, embedding) in enumerate(zip(batch_indices, batch_texts, embeddings)):
                    # Zum Cache hinzufügen
                    if use_cache:
                        self.embeddings_cache[text] = embedding
                    
                    # Aktualisiere Ergebnisarray an der richtigen Position
                    if idx >= len(all_embeddings):
                        all_embeddings.extend([None] * (idx - len(all_embeddings) + 1))
                    all_embeddings[idx] = embedding
        
        # Konvertiere zu NumPy-Array
        return np.vstack(all_embeddings)
    
    def analyze_sentiment(self, texts: Union[str, List[str]]) -> List[Dict]:
        """
        Führt eine Stimmungsanalyse für Texte durch.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            
        Returns:
            Liste mit Sentiment-Analysen für jeden Text (positive, negative, neutral)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Verwende NLTK für grundlegende Sentimentanalyse
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            
            analyzer = SentimentIntensityAnalyzer()
            results = []
            
            for text in texts:
                scores = analyzer.polarity_scores(text)
                
                # Bestimme die dominante Stimmung
                if scores['compound'] >= 0.05:
                    sentiment = 'positive'
                elif scores['compound'] <= -0.05:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': sentiment,
                    'scores': {
                        'positive': scores['pos'],
                        'negative': scores['neg'],
                        'neutral': scores['neu'],
                        'compound': scores['compound']
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Fehler bei der Sentimentanalyse: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback zu einem einfacheren Ansatz
            return [{'text': t[:100] + '...' if len(t) > 100 else t, 
                    'sentiment': 'unknown', 
                    'scores': {'positive': 0, 'negative': 0, 'neutral': 0, 'compound': 0}} 
                    for t in texts]
    
    def extract_topics(self, texts: Union[str, List[str]], num_topics: int = 5, 
                       words_per_topic: int = 5) -> List[Dict]:
        """
        Extrahiert Themen (Topics) aus Texten.
        
        Args:
            texts: Ein Text oder eine Liste von Texten
            num_topics: Anzahl der zu extrahierenden Themen
            words_per_topic: Anzahl der Wörter pro Thema
            
        Returns:
            Liste mit Themen und zugehörigen Top-Wörtern
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Tokenisiere und bereinige die Texte
            try:
                nltk.data.find('stopwords')
                nltk.data.find('punkt')
            except LookupError:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
            
            stop_words = set(stopwords.words('english'))
            
            # Texte vorverarbeiten
            processed_texts = []
            for text in texts:
                # Tokenisieren und Stopwords entfernen
                tokens = [w.lower() for w in word_tokenize(text) 
                         if w.isalpha() and w.lower() not in stop_words]
                processed_texts.append(' '.join(tokens))
            
            # Verwende Transformers für Themenmodellierung
            embeddings = self.get_embeddings(processed_texts)
            
            # Verwende K-Means-Clustering auf Embeddings
            kmeans = KMeans(n_clusters=min(num_topics, len(processed_texts)), random_state=42)
            kmeans.fit(embeddings)
            
            # Finde repräsentative Wörter für jedes Cluster
            topics = []
            
            # Alle Wörter aus allen Texten zusammenfassen
            all_words = []
            for text in processed_texts:
                all_words.extend(text.split())
            
            # Eindeutige Wörter
            unique_words = list(set(all_words))
            
            # Für jedes Wort ein Embedding berechnen
            if len(unique_words) > 0:
                word_embeddings = self.get_embeddings(unique_words)
                
                # Für jedes Cluster die nächsten Wörter bestimmen, die dem Clusterzentrum am nächsten sind
                for cluster_idx in range(kmeans.n_clusters):
                    center = kmeans.cluster_centers_[cluster_idx]
                    
                    # Berechne Distanzen zwischen Zentrum und Wort-Embeddings
                    distances = np.linalg.norm(word_embeddings - center, axis=1)
                    
                    # Finde die nächsten Wörter
                    closest_indices = np.argsort(distances)[:words_per_topic]
                    top_words = [unique_words[i] for i in closest_indices]
                    
                    # Beispieltexte für dieses Cluster finden
                    cluster_texts = [texts[i][:100] + "..." 
                                    for i, label in enumerate(kmeans.labels_) 
                                    if label == cluster_idx][:3]  # Maximal 3 Beispiele
                    
                    topic = {
                        "id": cluster_idx,
                        "words": top_words,
                        "examples": cluster_texts
                    }
                    topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Fehler bei der Themenextraktion: {str(e)}")
            logger.error(traceback.format_exc())
            return [{"id": 0, "words": ["error", "processing", "topics"], "examples": []}]
    
    def find_similar_texts(self, query: str, corpus: List[str], top_n: int = 5) -> List[Dict]:
        """
        Findet ähnliche Texte zu einer Anfrage in einem Korpus.
        
        Args:
            query: Anfrage-Text
            corpus: Liste von Texten, in denen gesucht werden soll
            top_n: Anzahl der zurückzugebenden ähnlichsten Texte
            
        Returns:
            Liste der ähnlichsten Texte mit Ähnlichkeitswerten
        """
        try:
            # Einbettungen für Anfrage und Korpus erzeugen
            query_embedding = self.get_embeddings(query).reshape(1, -1)
            corpus_embeddings = self.get_embeddings(corpus)
            
            # Kosinus-Ähnlichkeiten berechnen
            similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
            
            # Top-N ähnlichste Texte finden
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
            
        except Exception as e:
            logger.error(f"Fehler beim Finden ähnlicher Texte: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def generate_text_summary(self, text: str, max_length: int = 150) -> str:
        """
        Erzeugt eine Zusammenfassung eines längeren Textes.
        
        Args:
            text: Text, der zusammengefasst werden soll
            max_length: Maximale Länge der Zusammenfassung in Zeichen
            
        Returns:
            Zusammenfassung des Textes
        """
        try:
            # Vorverarbeitung des Textes
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) <= 1:
                return text[:max_length] + "..." if len(text) > max_length else text
            
            # Einbettungen für alle Sätze erzeugen
            sentence_embeddings = self.get_embeddings(sentences)
            
            # Durchschnittliche Einbettung berechnen (repräsentiert den Gesamttext)
            mean_embedding = np.mean(sentence_embeddings, axis=0).reshape(1, -1)
            
            # Ähnlichkeit jedes Satzes zum Durchschnitt berechnen
            similarities = np.dot(sentence_embeddings, mean_embedding.T).flatten()
            
            # Sätze nach Ähnlichkeit sortieren
            ranked_sentences = [(sentences[i], float(similarities[i])) for i in range(len(sentences))]
            ranked_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Top-Sätze auswählen, bis max_length erreicht ist
            summary = ""
            for sentence, _ in ranked_sentences:
                if len(summary) + len(sentence) <= max_length:
                    summary += sentence + " "
                else:
                    break
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Fehler bei der Textzusammenfassung: {str(e)}")
            logger.error(traceback.format_exc())
            return text[:max_length] + "..." if len(text) > max_length else text

###########################################
# 3. Prädiktive Modellierung
###########################################

class PredictiveModeler:
    """
    Klasse für die Entwicklung von prädiktiven Modellen, die verschiedene Datentypen
    verarbeiten und Vorhersagen treffen können.
    """
    
    def __init__(self, model_type: str = 'lstm', forecast_horizon: int = 7):
        """
        Initialisiert den Prädiktiven Modellierer.
        
        Args:
            model_type: Typ des zu verwendenden Modells ('lstm', 'transformer', 'gru', 'arima')
            forecast_horizon: Anzahl der Zeitschritte für Vorhersagen in die Zukunft
        """
        self.model_type = model_type.lower()
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.lookback = 10  # Standardwert für die Anzahl der zurückliegenden Zeitschritte
        self.feature_dims = None
        self.target_dims = None
        self.target_scaler = None
        self.history = None
        
    def _build_lstm_model(self, input_shape, output_dim):
        """Erstellt ein LSTM-Modell für Zeitreihenvorhersage"""
        model = keras.Sequential()
        model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(32))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(output_dim))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _build_transformer_model(self, input_shape, output_dim):
        """Erstellt ein Transformer-Modell für Zeitreihenvorhersage"""
        # Einfaches Transformer-Modell für Zeitreihen
        inputs = keras.Input(shape=input_shape)
        
        # Positional encoding layer
        class PositionalEncoding(layers.Layer):
            def __init__(self, position, d_model):
                super(PositionalEncoding, self).__init__()
                self.pos_encoding = self.positional_encoding(position, d_model)
                
            def get_angles(self, position, i, d_model):
                angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
                return position * angles
            
            def positional_encoding(self, position, d_model):
                angle_rads = self.get_angles(
                    position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                    i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                    d_model=d_model
                )
                
                # Apply sine to even indices
                sines = tf.math.sin(angle_rads[:, 0::2])
                # Apply cosine to odd indices
                cosines = tf.math.cos(angle_rads[:, 1::2])
                
                pos_encoding = tf.concat([sines, cosines], axis=-1)
                pos_encoding = pos_encoding[tf.newaxis, ...]
                
                return tf.cast(pos_encoding, tf.float32)
            
            def call(self, inputs):
                return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        
        x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
        
        # Multi-head attention layer
        x = layers.MultiHeadAttention(
            key_dim=input_shape[1], num_heads=4, dropout=0.1
        )(x, x, x, attention_mask=None, training=True)
        
        # Feed-forward network
        x = layers.Dropout(0.1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=1, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv1D(filters=input_shape[1], kernel_size=1)(x)
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Output layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(output_dim)(x)
        
        model = keras.Model(inputs, x)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def _build_gru_model(self, input_shape, output_dim):
        """Erstellt ein GRU-Modell für Zeitreihenvorhersage"""
        model = keras.Sequential()
        model.add(layers.GRU(64, return_sequences=True, input_shape=input_shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.GRU(32))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(output_dim))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _create_sequences(self, data, target=None, lookback=None):
        """
        Erstellt Sequenzen für die Zeitreihenmodellierung
        
        Args:
            data: Eingabedaten (numpy array)
            target: Zielvariablen (optional, numpy array)
            lookback: Anzahl der zurückliegenden Zeitschritte (optional)
            
        Returns:
            X: Sequenzen für die Eingabe
            y: Zielwerte (wenn target bereitgestellt wird)
        """
        if lookback is None:
            lookback = self.lookback
        
        X = []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
        X = np.array(X)
        
        if target is not None:
            y = target[lookback:]
            return X, y
        
        return X
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, 
            lookback: int = 10, epochs: int = 50, 
            validation_split: float = 0.2, batch_size: int = 32,
            verbose: int = 1):
        """
        Trainiert das prädiktive Modell mit den gegebenen Daten.
        
        Args:
            X: Eingabedaten (DataFrame)
            y: Zielvariablen (DataFrame, optional für Zeitreihen)
            lookback: Anzahl der zurückliegenden Zeitschritte für Zeitreihenmodelle
            epochs: Anzahl der Trainings-Epochen
            validation_split: Anteil der Daten für die Validierung
            batch_size: Batch-Größe für das Training
            verbose: Ausgabedetailstufe (0, 1, oder 2)
            
        Returns:
            self: Trainiertes Modell
        """
        try:
            self.lookback = lookback
            
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            # Skalierung der Eingabedaten
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_values)
            
            # Vorbereiten der Zielvariablen
            if y is not None:
                if isinstance(y, pd.DataFrame):
                    y_values = y.values
                else:
                    y_values = y
                    
                self.target_scaler = StandardScaler()
                y_scaled = self.target_scaler.fit_transform(y_values)
                self.target_dims = y_scaled.shape[1]
            else:
                # Wenn keine Zielvariablen bereitgestellt werden, nehmen wir an, dass X selbst eine Zeitreihe ist
                y_scaled = X_scaled
                self.target_dims = X_scaled.shape[1]
            
            self.feature_dims = X_scaled.shape[1]
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, lookback)
            
            # Modell basierend auf dem ausgewählten Typ erstellen
            input_shape = (lookback, self.feature_dims)
            output_dim = self.target_dims * self.forecast_horizon
            
            if self.model_type == 'lstm':
                self.model = self._build_lstm_model(input_shape, output_dim)
            elif self.model_type == 'transformer':
                self.model = self._build_transformer_model(input_shape, output_dim)
            elif self.model_type == 'gru':
                self.model = self._build_gru_model(input_shape, output_dim)
            else:
                raise ValueError(f"Nicht unterstützter Modelltyp: {self.model_type}")
            
            # Reshape y_sequences für das Forecast-Horizon
            y_prepared = y_sequences.reshape(y_sequences.shape[0], -1)
            
            # Modell trainieren
            self.history = self.model.fit(
                X_sequences, y_prepared,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des prädiktiven Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X: pd.DataFrame, return_sequences: bool = False) -> np.ndarray:
        """
        Macht Vorhersagen mit dem trainierten Modell.
        
        Args:
            X: Eingabedaten (DataFrame)
            return_sequences: Ob die Vorhersagesequenz zurückgegeben werden soll
            
        Returns:
            Vorhersagen für die Eingabedaten
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
                
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_values)
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences = self._create_sequences(X_scaled, lookback=self.lookback)
            
            # Vorhersagen machen
            predictions_scaled = self.model.predict(X_sequences)
            
            # Reshape für die Ausgabe
            predictions_scaled = predictions_scaled.reshape(
                predictions_scaled.shape[0], 
                self.forecast_horizon, 
                self.target_dims
            )
            
            # Rücktransformation
            predictions = np.zeros_like(predictions_scaled)
            for i in range(self.forecast_horizon):
                step_predictions = predictions_scaled[:, i, :]
                # Rücktransformation nur für jeden Zeitschritt
                predictions[:, i, :] = self.target_scaler.inverse_transform(step_predictions)
            
            if return_sequences:
                return predictions
            else:
                # Nur den ersten Vorhersageschritt zurückgeben
                return predictions[:, 0, :]
            
        except Exception as e:
            logger.error(f"Fehler bei der Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def forecast(self, X: pd.DataFrame, steps: int = None) -> np.ndarray:
        """
        Erstellt eine Vorhersage für mehrere Zeitschritte in die Zukunft.
        
        Args:
            X: Letzte bekannte Datenpunkte (mindestens lookback viele)
            steps: Anzahl der vorherzusagenden Schritte (Standard: forecast_horizon)
            
        Returns:
            Vorhersagesequenz für die nächsten 'steps' Zeitschritte
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        if steps is None:
            steps = self.forecast_horizon
            
        try:
            # Vorbereiten der Eingabedaten
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
            
            if len(X_values) < self.lookback:
                raise ValueError(f"Eingabedaten müssen mindestens {self.lookback} Zeitschritte enthalten")
            
            # Verwende nur die letzten 'lookback' Zeitschritte
            X_recent = X_values[-self.lookback:]
            
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_recent)
            X_sequence = X_scaled.reshape(1, self.lookback, self.feature_dims)
            
            # Erstelle die multi-step-Vorhersage
            forecast_values = []
            
            current_sequence = X_sequence.copy()
            
            for _ in range(steps):
                # Mache eine Vorhersage für den nächsten Schritt
                next_step_scaled = self.model.predict(current_sequence)[0]
                next_step_scaled = next_step_scaled.reshape(1, self.target_dims)
                
                # Rücktransformation
                next_step = self.target_scaler.inverse_transform(next_step_scaled)
                forecast_values.append(next_step[0])
                
                # Aktualisiere die Eingabesequenz für den nächsten Schritt
                # Entferne den ersten Zeitschritt und füge den neu vorhergesagten hinzu
                new_sequence = np.zeros_like(current_sequence)
                new_sequence[0, :-1, :] = current_sequence[0, 1:, :]
                new_sequence[0, -1, :] = next_step_scaled
                current_sequence = new_sequence
            
            return np.array(forecast_values)
            
        except Exception as e:
            logger.error(f"Fehler bei der Mehrschritt-Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluiert das Modell mit Testdaten.
        
        Args:
            X_test: Test-Eingabedaten
            y_test: Test-Zielvariablen
            
        Returns:
            Dictionary mit Bewertungsmetriken
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Vorbereiten der Testdaten
            if isinstance(X_test, pd.DataFrame):
                X_values = X_test.values
            else:
                X_values = X_test
                
            if isinstance(y_test, pd.DataFrame):
                y_values = y_test.values
            else:
                y_values = y_test
            
            # Skalierung der Eingabedaten
            X_scaled = self.scaler.transform(X_values)
            y_scaled = self.target_scaler.transform(y_values)
            
            # Daten in Sequenzen umwandeln für Zeitreihenmodelle
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, self.lookback)
            
            # Reshape y_sequences für das Forecast-Horizon
            y_prepared = y_sequences.reshape(y_sequences.shape[0], -1)
            
            # Modell evaluieren
            evaluation = self.model.evaluate(X_sequences, y_prepared, verbose=0)
            
            # Vorhersagen machen für detailliertere Metriken
            predictions = self.predict(X_test)
            
            # Tatsächliche Werte (ohne die ersten lookback Zeitschritte)
            actuals = y_values[self.lookback:]
            
            # Berechne RMSE, MAE, MAPE
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Entsprechende Anzahl an Vorhersagen auswählen
            predictions = predictions[:len(actuals)]
            
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # MAPE (Mean Absolute Percentage Error)
            # Vermeide Division durch Null
            mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
            
            return {
                'loss': evaluation[0],
                'mae': evaluation[1],
                'rmse': rmse,
                'mean_absolute_error': mae,
                'mean_absolute_percentage_error': mape
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Modellbewertung: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def plot_forecast(self, X: pd.DataFrame, y_true: pd.DataFrame = None, 
                     steps: int = None, feature_idx: int = 0,
                     save_path: str = None) -> plt.Figure:
        """
        Visualisiert die Vorhersage des Modells.
        
        Args:
            X: Eingabedaten
            y_true: Tatsächliche zukünftige Werte (optional)
            steps: Anzahl der vorherzusagenden Schritte
            feature_idx: Index der darzustellenden Feature-Dimension
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Standard-Schritte
            if steps is None:
                steps = self.forecast_horizon
            
            # Vorhersage erstellen
            forecast_values = self.forecast(X, steps)
            
            # Historische Daten (letzte lookback Zeitschritte)
            if isinstance(X, pd.DataFrame):
                historical_values = X.values[-self.lookback:, feature_idx]
            else:
                historical_values = X[-self.lookback:, feature_idx]
            
            # Zeitachse erstellen
            time_hist = np.arange(-self.lookback, 0)
            time_future = np.arange(0, steps)
            
            # Visualisierung erstellen
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historische Daten plotten
            ax.plot(time_hist, historical_values, 'b-', label='Historische Daten')
            
            # Vorhersage plotten
            ax.plot(time_future, forecast_values[:, feature_idx], 'r-', label='Vorhersage')
            
            # Tatsächliche zukünftige Werte plotten, falls vorhanden
            if y_true is not None:
                if isinstance(y_true, pd.DataFrame):
                    true_future = y_true.values[:steps, feature_idx]
                else:
                    true_future = y_true[:steps, feature_idx]
                
                ax.plot(time_future[:len(true_future)], true_future, 'g-', label='Tatsächliche Werte')
            
            # Grafik anpassen
            ax.set_title(f'Zeitreihenvorhersage mit {self.model_type.upper()}')
            ax.set_xlabel('Zeitschritte')
            ax.set_ylabel(f'Feature {feature_idx}')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Trennlinie zwischen historischen und Vorhersagedaten
            ax.axvline(x=0, color='k', linestyle='--')
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung der Vorhersage: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        model_data = {
            "model_type": self.model_type,
            "forecast_horizon": self.forecast_horizon,
            "lookback": self.lookback,
            "feature_dims": self.feature_dims,
            "target_dims": self.target_dims,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere das Model-Objekt
        self.model.save(f"{path}_model")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        joblib.dump(self.target_scaler, f"{path}_target_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        predictor = cls(
            model_type=model_data['model_type'],
            forecast_horizon=model_data['forecast_horizon']
        )
        
        predictor.lookback = model_data['lookback']
        predictor.feature_dims = model_data['feature_dims']
        predictor.target_dims = model_data['target_dims']
        predictor.is_fitted = model_data['is_fitted']
        
        # Lade das Model-Objekt
        predictor.model = keras.models.load_model(f"{path}_model")
        
        # Lade die Scaler
        predictor.scaler = joblib.load(f"{path}_scaler.joblib")
        predictor.target_scaler = joblib.load(f"{path}_target_scaler.joblib")
        
        return predictor

###########################################
# 4. Datensynthese und GAN-basierte Modelle
###########################################

class DataSynthesizer:
    """
    Klasse zur Generierung synthetischer Daten basierend auf realen Beispielen.
    Verwendet GAN (Generative Adversarial Network) für realistische Datensynthese.
    """
    
    def __init__(self, categorical_threshold: int = 10, noise_dim: int = 100):
        """
        Initialisiert den Datensynthetisierer.
        
        Args:
            categorical_threshold: Anzahl eindeutiger Werte, ab der eine Spalte als kategorisch gilt
            noise_dim: Dimension des Rauschvektors für den Generator
        """
        self.categorical_threshold = categorical_threshold
        self.noise_dim = noise_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        
        self.column_types = {}  # Speichert, ob eine Spalte kategorisch oder kontinuierlich ist
        self.categorical_mappings = {}  # Speichert Mappings für kategorische Spalten
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # Für kontinuierliche Variablen
        
        self.is_fitted = False
        self.feature_dims = None
        self.training_data = None
    
    def _identify_column_types(self, data: pd.DataFrame):
        """Identifiziert, ob Spalten kategorisch oder kontinuierlich sind"""
        self.column_types = {}
        
        for col in data.columns:
            n_unique = data[col].nunique()
            
            # Wenn die Anzahl eindeutiger Werte kleiner als der Schwellenwert ist oder
            # der Datentyp ist nicht numerisch, behandle die Spalte als kategorisch
            if n_unique < self.categorical_threshold or not pd.api.types.is_numeric_dtype(data[col]):
                self.column_types[col] = 'categorical'
                
                # Erstelle Mapping von Kategorien zu Zahlen
                categories = data[col].unique()
                self.categorical_mappings[col] = {
                    cat: i for i, cat in enumerate(categories)
                }
                # Umgekehrtes Mapping für die Rücktransformation
                self.categorical_mappings[f"{col}_reverse"] = {
                    i: cat for i, cat in enumerate(categories)
                }
            else:
                self.column_types[col] = 'continuous'
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Vorverarbeitung der Daten für das GAN"""
        processed_data = pd.DataFrame()
        
        for col in data.columns:
            if self.column_types[col] == 'categorical':
                # One-Hot-Encoding für kategorische Spalten
                mapped_col = data[col].map(self.categorical_mappings[col])
                one_hot = pd.get_dummies(mapped_col, prefix=col)
                processed_data = pd.concat([processed_data, one_hot], axis=1)
            else:
                # Skalierung für kontinuierliche Spalten
                processed_data[col] = data[col]
        
        # Skaliere alle Spalten auf [-1, 1]
        return self.scaler.fit_transform(processed_data)
    
    def _postprocess_data(self, generated_data: np.ndarray) -> pd.DataFrame:
        """Nachverarbeitung der generierten Daten zurück in das ursprüngliche Format"""
        # Rücktransformation der Skalierung
        rescaled_data = self.scaler.inverse_transform(generated_data)
        
        # Erstelle einen DataFrame mit den ursprünglichen Spalten
        result = pd.DataFrame()
        
        col_idx = 0
        for col, col_type in self.column_types.items():
            if col_type == 'categorical':
                # Anzahl der eindeutigen Werte für diese kategorische Spalte
                n_categories = len(self.categorical_mappings[col])
                
                # Extrahiere die One-Hot-kodierten Werte
                cat_values = rescaled_data[:, col_idx:col_idx+n_categories]
                
                # Konvertiere von One-Hot zurück zu kategorischen Werten
                # Nehme die Kategorie mit dem höchsten Wert
                cat_indices = np.argmax(cat_values, axis=1)
                
                # Mappe zurück zu den ursprünglichen Kategorien
                result[col] = [self.categorical_mappings[f"{col}_reverse"][idx] for idx in cat_indices]
                
                col_idx += n_categories
            else:
                # Kontinuierliche Spalte einfach übernehmen
                result[col] = rescaled_data[:, col_idx]
                col_idx += 1
        
        return result
    
    def _build_generator(self, output_dim):
        """Erstellt den Generator für das GAN"""
        model = keras.Sequential([
            layers.Dense(256, input_dim=self.noise_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(output_dim, activation='tanh')  # tanh für Output im Bereich [-1, 1]
        ])
        return model
    
    def _build_discriminator(self, input_dim):
        """Erstellt den Diskriminator für das GAN"""
        model = keras.Sequential([
            layers.Dense(512, input_dim=input_dim, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy')
        return model
    
    def _build_gan(self, generator, discriminator):
        """Kombiniert Generator und Diskriminator zum GAN"""
        discriminator.trainable = False  # Diskriminator beim GAN-Training nicht aktualisieren
        
        model = keras.Sequential([
            generator,
            discriminator
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy')
        return model
    
    def fit(self, data: pd.DataFrame, epochs: int = 2000, batch_size: int = 32, 
           sample_interval: int = 100, verbose: int = 1):
        """
        Trainiert das GAN-Modell mit den gegebenen Daten.
        
        Args:
            data: Eingabedaten (DataFrame)
            epochs: Anzahl der Trainings-Epochen
            batch_size: Batch-Größe für das Training
            sample_interval: Intervall für Stichproben der generierten Daten
            verbose: Ausgabedetailstufe (0, 1, oder 2)
            
        Returns:
            self: Trainiertes Modell
        """
        try:
            # Identifiziere Spaltentypen
            self._identify_column_types(data)
            
            # Vorverarbeitung der Daten
            processed_data = self._preprocess_data(data)
            self.feature_dims = processed_data.shape[1]
            
            # Speichere trainierte Daten für spätere Validierung
            self.training_data = data.copy()
            
            # Baue das GAN-Modell
            self.generator = self._build_generator(self.feature_dims)
            self.discriminator = self._build_discriminator(self.feature_dims)
            self.gan = self._build_gan(self.generator, self.discriminator)
            
            # Label für echte und gefälschte Daten
            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            
            # Trainingsschleife
            for epoch in range(epochs):
                # ---------------------
                #  Trainiere Diskriminator
                # ---------------------
                
                # Wähle eine zufällige Batch aus echten Daten
                idx = np.random.randint(0, processed_data.shape[0], batch_size)
                real_data = processed_data[idx]
                
                # Generiere eine Batch aus gefälschten Daten
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                fake_data = self.generator.predict(noise)
                
                # Trainiere den Diskriminator
                d_loss_real = self.discriminator.train_on_batch(real_data, real)
                d_loss_fake = self.discriminator.train_on_batch(fake_data, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ---------------------
                #  Trainiere Generator
                # ---------------------
                
                # Generiere neue Batch aus Rauschen
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                
                # Trainiere den Generator
                g_loss = self.gan.train_on_batch(noise, real)
                
                # Ausgabe für Fortschrittsüberwachung
                if verbose > 0 and epoch % sample_interval == 0:
                    print(f"Epoch {epoch}/{epochs} [D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}]")
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Fehler beim Training des GAN-Modells: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def generate(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generiert synthetische Daten.
        
        Args:
            n_samples: Anzahl der zu generierenden Datensätze
            
        Returns:
            DataFrame mit synthetischen Daten im Format der Trainingsdaten
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere Rauschen als Input für den Generator
            noise = np.random.normal(0, 1, (n_samples, self.noise_dim))
            
            # Generiere Daten
            generated_data = self.generator.predict(noise)
            
            # Nachverarbeitung der Daten
            synthetic_data = self._postprocess_data(generated_data)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Fehler bei der Datengenerierung: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def evaluate_quality(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Bewertet die Qualität der generierten Daten durch Vergleich mit den Trainingsdaten.
        
        Args:
            n_samples: Anzahl der zu generierenden und bewertenden Datensätze
            
        Returns:
            Dictionary mit Qualitätsmetriken
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere synthetische Daten
            synthetic_data = self.generate(n_samples)
            
            # Statistischer Vergleich zwischen echten und synthetischen Daten
            metrics = {}
            
            # Vergleiche Mittelwerte und Standardabweichungen für kontinuierliche Spalten
            for col, col_type in self.column_types.items():
                if col_type == 'continuous':
                    # Berechne Mittelwert und Standardabweichung für echte Daten
                    real_mean = self.training_data[col].mean()
                    real_std = self.training_data[col].std()
                    
                    # Berechne dieselben Statistiken für synthetische Daten
                    synth_mean = synthetic_data[col].mean()
                    synth_std = synthetic_data[col].std()
                    
                    # Berechne die relative Differenz
                    mean_diff = abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10)
                    std_diff = abs(real_std - synth_std) / (abs(real_std) + 1e-10)
                    
                    metrics[f"{col}_mean_diff"] = float(mean_diff)
                    metrics[f"{col}_std_diff"] = float(std_diff)
                else:
                    # Vergleiche die Verteilung kategorischer Werte
                    real_dist = self.training_data[col].value_counts(normalize=True)
                    synth_dist = synthetic_data[col].value_counts(normalize=True)
                    
                    # Berechne die Jensen-Shannon-Divergenz
                    # (symmetrische Version der KL-Divergenz)
                    js_divergence = 0.0
                    
                    # Stelle sicher, dass beide Verteilungen dieselben Kategorien haben
                    all_categories = set(real_dist.index) | set(synth_dist.index)
                    
                    for cat in all_categories:
                        p = real_dist.get(cat, 0)
                        q = synth_dist.get(cat, 0)
                        
                        # Vermeide Logarithmus von 0
                        if p > 0 and q > 0:
                            m = 0.5 * (p + q)
                            js_divergence += 0.5 * (p * np.log(p / m) + q * np.log(q / m))
                    
                    metrics[f"{col}_js_divergence"] = float(js_divergence)
            
            # Gesamtqualitätsmetrik
            # Durchschnitt der normalisierten Abweichungen (niedriger ist besser)
            continuous_diffs = [v for k, v in metrics.items() if k.endswith('_diff')]
            categorical_diffs = [v for k, v in metrics.items() if k.endswith('_js_divergence')]
            
            if continuous_diffs:
                metrics['continuous_avg_diff'] = float(np.mean(continuous_diffs))
            if categorical_diffs:
                metrics['categorical_avg_diff'] = float(np.mean(categorical_diffs))
            
            # Gesamtbewertung (0 bis 1, höher ist besser)
            overall_score = 1.0
            if continuous_diffs:
                overall_score -= 0.5 * min(1.0, np.mean(continuous_diffs))
            if categorical_diffs:
                overall_score -= 0.5 * min(1.0, np.mean(categorical_diffs))
            
            metrics['overall_quality_score'] = float(overall_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Fehler bei der Qualitätsbewertung: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def plot_comparison(self, n_samples: int = 1000, 
                       features: List[str] = None,
                       save_path: str = None) -> plt.Figure:
        """
        Visualisiert einen Vergleich zwischen echten und synthetischen Daten.
        
        Args:
            n_samples: Anzahl der zu generierenden Datensätze
            features: Liste der darzustellenden Features (Standard: alle)
            save_path: Pfad zum Speichern der Visualisierung
            
        Returns:
            Matplotlib-Figur mit der Visualisierung
        """
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        try:
            # Generiere synthetische Daten
            synthetic_data = self.generate(n_samples)
            
            # Wähle die darzustellenden Features aus
            if features is None:
                # Wähle bis zu 6 Features für die Visualisierung
                features = list(self.column_types.keys())[:min(6, len(self.column_types))]
            
            # Bestimme die Anzahl der Zeilen und Spalten für das Subplot-Raster
            n_features = len(features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            # Erstelle die Figur
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
            if n_rows * n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for i, feature in enumerate(features):
                ax = axes[i]
                
                if self.column_types[feature] == 'continuous':
                    # Histogramm für kontinuierliche Variablen
                    sns.histplot(self.training_data[feature], kde=True, ax=ax, color='blue', alpha=0.5, label='Echte Daten')
                    sns.histplot(synthetic_data[feature], kde=True, ax=ax, color='red', alpha=0.5, label='Synthetische Daten')
                else:
                    # Balkendiagramm für kategorische Variablen
                    real_counts = self.training_data[feature].value_counts(normalize=True)
                    synth_counts = synthetic_data[feature].value_counts(normalize=True)
                    
                    # Kombiniere beide, um alle Kategorien zu erfassen
                    all_cats = sorted(set(real_counts.index) | set(synth_counts.index))
                    
                    # Erstelle ein DataFrame für Seaborn
                    plot_data = []
                    for cat in all_cats:
                        plot_data.append({'Category': cat, 'Frequency': real_counts.get(cat, 0), 'Type': 'Real'})
                        plot_data.append({'Category': cat, 'Frequency': synth_counts.get(cat, 0), 'Type': 'Synthetic'})
                    
                    plot_df = pd.DataFrame(plot_data)
                    
                    # Balkendiagramm
                    sns.barplot(x='Category', y='Frequency', hue='Type', data=plot_df, ax=ax)
                
                ax.set_title(f'Verteilung von {feature}')
                ax.legend()
                
                # Achsen anpassen
                if self.column_types[feature] == 'categorical':
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Verstecke ungenutzte Subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Optional: Speichern
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung des Datenvergleichs: {str(e)}")
            logger.error(traceback.format_exc())
            # Leere Figur zurückgeben im Fehlerfall
            return plt.figure()
    
    def save(self, path: str):
        """Speichert das trainierte Modell"""
        if not self.is_fitted:
            raise ValueError("Das Modell muss zuerst trainiert werden mit fit()")
            
        model_data = {
            "categorical_threshold": self.categorical_threshold,
            "noise_dim": self.noise_dim,
            "feature_dims": self.feature_dims,
            "column_types": self.column_types,
            "categorical_mappings": self.categorical_mappings,
            "is_fitted": self.is_fitted
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere die Modelle
        self.generator.save(f"{path}_generator")
        self.discriminator.save(f"{path}_discriminator")
        
        # Speichere den Scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Speichere allgemeine Modellmetadaten und Mappings
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell"""
        with open(f"{path}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        synthesizer = cls(
            categorical_threshold=model_data['categorical_threshold'],
            noise_dim=model_data['noise_dim']
        )
        
        synthesizer.feature_dims = model_data['feature_dims']
        synthesizer.column_types = model_data['column_types']
        synthesizer.categorical_mappings = model_data['categorical_mappings']
        synthesizer.is_fitted = model_data['is_fitted']
        
        # Lade die Modelle
        synthesizer.generator = keras.models.load_model(f"{path}_generator")
        synthesizer.discriminator = keras.models.load_model(f"{path}_discriminator")
        
        # Lade den Scaler
        synthesizer.scaler = joblib.load(f"{path}_scaler.joblib")
        
        # Rekonstruiere das GAN
        synthesizer.gan = synthesizer._build_gan(synthesizer.generator, synthesizer.discriminator)
        
        return synthesizer

###########################################
# 5. Federated Learning und Compute-to-Data
###########################################

class ComputeToDataManager:
    """
    Klasse zur Verwaltung von Compute-to-Data-Operationen für den Datenschutz.
    Ermöglicht die Ausführung von Berechnungen auf sensiblen Daten ohne direkten Zugriff.
    """
    
    def __init__(self, encryption_key: bytes = None):
        """
        Initialisiert den Compute-to-Data-Manager.
        
        Args:
            encryption_key: Schlüssel für die Verschlüsselung (optional)
        """
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.allowed_operations = {
            'aggregate': self._aggregate,
            'count': self._count,
            'mean': self._mean,
            'sum': self._sum,
            'min': self._min,
            'max': self._max,
            'std': self._std,
            'correlation': self._correlation,
            'histogram': self._histogram,
            'custom_model': self._custom_model_inference
        }
        
        # Datenschutzkonfiguration
        self.privacy_config = {
            'min_group_size': 5,  # Minimale Gruppengröße für Aggregationen
            'noise_level': 0.01,  # Standardrauschen für Differentialprivatsphäre
            'outlier_removal': True  # Entfernung von Ausreißern für sensible Berechnungen
        }
    
    def set_privacy_config(self, min_group_size: int = None, noise_level: float = None, 
                          outlier_removal: bool = None):
        """Aktualisiert die Datenschutzkonfiguration"""
        if min_group_size is not None:
            self.privacy_config['min_group_size'] = min_group_size
        if noise_level is not None:
            self.privacy_config['noise_level'] = noise_level
        if outlier_removal is not None:
            self.privacy_config['outlier_removal'] = outlier_removal
    
    def _encrypt_data(self, data: pd.DataFrame) -> bytes:
        """Verschlüsselt Daten für die sichere Speicherung"""
        serialized = data.to_json().encode()
        return self.cipher_suite.encrypt(serialized)
    
    def _decrypt_data(self, encrypted_data: bytes) -> pd.DataFrame:
        """Entschlüsselt Daten für die Verarbeitung"""
        decrypted = self.cipher_suite.decrypt(encrypted_data)
        return pd.read_json(decrypted.decode())
    
    def _add_differential_privacy_noise(self, result, scale: float = None):
        """Fügt Rauschen für Differential Privacy hinzu"""
        scale = scale or self.privacy_config['noise_level']
        
        if isinstance(result, (int, float)):
            return result + np.random.normal(0, scale * abs(result))
        elif isinstance(result, np.ndarray):
            return result + np.random.normal(0, scale * np.abs(result).mean(), result.shape)
        elif isinstance(result, pd.Series):
            noisy_values = result.values + np.random.normal(0, scale * np.abs(result.values).mean(), result.shape)
            return pd.Series(noisy_values, index=result.index)
        elif isinstance(result, pd.DataFrame):
            noisy_values = result.values + np.random.normal(0, scale * np.abs(result.values).mean(), result.shape)
            return pd.DataFrame(noisy_values, index=result.index, columns=result.columns)
        else:
            return result
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Entfernt Ausreißer aus den Daten"""
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            return data
        
        # Z-Score-basierte Ausreißererkennung
        z_scores = ((numeric_data - numeric_data.mean()) / numeric_data.std()).abs()
        mask = (z_scores < 3).all(axis=1)
        
        return data[mask]
    
    def _validate_group_size(self, data: pd.DataFrame, group_column: str = None, 
                            group_value: Any = None) -> bool:
        """
        Prüft, ob eine Gruppe die Mindestgröße für Datenschutz erfüllt
        
        Args:
            data: Daten
            group_column: Spalte für die Gruppierung
            group_value: Wert für die Filterung
            
        Returns:
            True, wenn die Gruppe groß genug ist
        """
        min_size = self.privacy_config['min_group_size']
        
        if group_column is not None and group_value is not None:
            group_data = data[data[group_column] == group_value]
            return len(group_data) >= min_size
        
        return len(data) >= min_size
    
    def _prepare_data_for_computation(self, data: pd.DataFrame, columns: List[str] = None,
                                    group_column: str = None, group_value: Any = None) -> pd.DataFrame:
        """
        Bereitet Daten für die Berechnung vor, einschließlich Filterung und Datenschutzprüfungen
        
        Args:
            data: Eingabedaten
            columns: Zu verwendende Spalten
            group_column: Spalte für die Gruppierung
            group_value: Wert für die Filterung
            
        Returns:
            Vorbereitete Daten
        """
        # Filtere Daten nach Gruppe, falls angegeben
        if group_column is not None and group_value is not None:
            prepared_data = data[data[group_column] == group_value]
        else:
            prepared_data = data.copy()
        
        # Prüfe Mindestgruppengröße
        if not self._validate_group_size(prepared_data):
            raise ValueError(f"Gruppe zu klein für Datenschutz (min: {self.privacy_config['min_group_size']})")
        
        # Entferne Ausreißer, falls konfiguriert
        if self.privacy_config['outlier_removal']:
            prepared_data = self._remove_outliers(prepared_data)
        
        # Wähle Spalten aus, falls angegeben
        if columns is not None:
            prepared_data = prepared_data[columns]
        
        return prepared_data
    
    # Implementierte Operationen für C2D
    
    def _aggregate(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Aggregiert Daten nach Spalten und Gruppen"""
        columns = params.get('columns')
        group_by = params.get('group_by')
        aggregations = params.get('aggregations', ['count', 'mean'])
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        if group_by:
            grouped = prepared_data.groupby(group_by)
            
            # Prüfe Mindestgröße für jede Gruppe
            group_sizes = grouped.size()
            valid_groups = group_sizes[group_sizes >= self.privacy_config['min_group_size']].index
            
            # Filtere ungültige Gruppen
            results = {}
            for agg in aggregations:
                if agg == 'count':
                    agg_result = grouped.size()
                else:
                    agg_func = getattr(grouped, agg, None)
                    if agg_func is None:
                        continue
                    agg_result = agg_func()
                
                # Behalte nur valide Gruppen
                agg_result = agg_result.loc[valid_groups]
                
                # Füge Rauschen hinzu
                results[agg] = self._add_differential_privacy_noise(agg_result)
        else:
            results = {}
            for agg in aggregations:
                if agg == 'count':
                    agg_result = len(prepared_data)
                else:
                    agg_func = getattr(prepared_data, agg, None)
                    if agg_func is None:
                        continue
                    agg_result = agg_func()
                
                # Füge Rauschen hinzu
                results[agg] = self._add_differential_privacy_noise(agg_result)
        
        # Konvertiere zu serialisierbarem Format
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
                serializable_results[k] = v.to_dict()
            else:
                serializable_results[k] = v
        
        return serializable_results
    
    def _count(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Zählt Datensätze nach Filterkriterien"""
        filter_column = params.get('filter_column')
        filter_value = params.get('filter_value')
        
        if filter_column and filter_value is not None:
            count = len(data[data[filter_column] == filter_value])
        else:
            count = len(data)
        
        # Füge Rauschen hinzu und runde auf ganze Zahlen
        noisy_count = round(self._add_differential_privacy_noise(count))
        
        # Stelle sicher, dass der Wert nicht negativ ist
        return {'count': max(0, noisy_count)}
    
    def _mean(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Berechnet den Mittelwert für numerische Spalten"""
        columns = params.get('columns')
        group_column = params.get('group_column')
        
        prepared_data = self._prepare_data_for_computation(data, columns)
        
        if group_column:
            grouped = prepared_data.groupby(group_column)
            
            # Prüfe Mindestgröße für jede Gruppe
            group_sizes = grouped.size()
            valid_groups = group_sizes[group_sizes >= self.privacy_config['min
