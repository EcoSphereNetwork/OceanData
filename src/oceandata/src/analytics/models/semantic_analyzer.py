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
