"""
OceanData - Social Media Connector

Dieser Konnektor ermöglicht die Integration von Social-Media-Daten in OceanData.
Unterstützte Plattformen: Twitter, Facebook, Instagram, LinkedIn, TikTok
"""

import pandas as pd
import numpy as np
import json
import logging
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import hashlib
import time
import re

# Importiere Basisklassen
from ..base import DataConnector, DataCategory, PrivacyLevel

# Konfiguriere Logger
logger = logging.getLogger("OceanData.Connectors.SocialMedia")

class SocialMediaConnector(DataConnector):
    """Konnektor für Social-Media-Daten."""

    SUPPORTED_PLATFORMS = ['twitter', 'facebook', 'instagram', 'linkedin', 'tiktok']

    def __init__(self, user_id: str, platform: str, api_credentials: Dict[str, str] = None):
        """
        Initialisiert einen Social-Media-Datenkonnektor.

        Args:
            user_id: ID des Benutzers
            platform: Social-Media-Plattform (twitter, facebook, instagram, linkedin, tiktok)
            api_credentials: API-Zugangsdaten für die Plattform (optional)
        """
        if platform.lower() not in self.SUPPORTED_PLATFORMS:
            raise ValueError(f"Nicht unterstützte Plattform: {platform}. "
                           f"Unterstützte Plattformen: {', '.join(self.SUPPORTED_PLATFORMS)}")

        super().__init__(f'socialmedia_{platform.lower()}', user_id, DataCategory.SOCIAL)
        
        self.platform = platform.lower()
        self.api_credentials = api_credentials or {}
        self.api_client = None
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        
        # Plattformspezifische Metadaten
        self.metadata.update({
            "platform": self.platform,
            "source_details": {
                "name": f"{platform.capitalize()} Social Media Data",
                "version": "1.0",
                "description": f"User activity and content data from {platform.capitalize()}"
            }
        })

    def connect(self) -> bool:
        """
        Verbindung zur Social-Media-API herstellen.

        Returns:
            bool: True, wenn die Verbindung erfolgreich hergestellt wurde
        """
        try:
            # In einer echten Implementierung würden wir hier die Verbindung zur API herstellen
            # Für das MVP simulieren wir eine erfolgreiche Verbindung
            
            # Prüfe, ob API-Zugangsdaten vorhanden sind
            if not self.api_credentials and self.platform != 'demo':
                logger.warning(f"Keine API-Zugangsdaten für {self.platform} vorhanden. "
                              f"Verwende Demo-Modus.")
                self.platform = 'demo'
            
            logger.info(f"Verbindung zu {self.platform}-API für Benutzer {self.user_id} hergestellt")
            
            # Simuliere API-Client
            self.api_client = {
                "platform": self.platform,
                "connected": True,
                "user_id": self.user_id,
                "connection_time": datetime.now().isoformat()
            }
            
            # Simuliere Rate-Limit-Informationen
            self.rate_limit_remaining = 1000
            self.rate_limit_reset = datetime.now() + timedelta(hours=1)
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Verbinden mit {self.platform}-API: {str(e)}")
            return False

    def fetch_data(self) -> pd.DataFrame:
        """
        Social-Media-Daten abrufen.

        Returns:
            pd.DataFrame: Die abgerufenen Social-Media-Daten
        """
        # Prüfe, ob eine Verbindung besteht
        if not self.api_client:
            logger.error(f"Keine Verbindung zur {self.platform}-API. Rufe connect() zuerst auf.")
            return pd.DataFrame()
        
        try:
            # In einer echten Implementierung würden wir hier die tatsächlichen Daten abrufen
            # Für das MVP generieren wir realistische Beispieldaten basierend auf der Plattform
            
            # Generiere zufällige Zeitstempel der letzten 90 Tage
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            # Plattformspezifische Daten generieren
            if self.platform == 'twitter':
                return self._generate_twitter_data(start_date, end_date)
            elif self.platform == 'facebook':
                return self._generate_facebook_data(start_date, end_date)
            elif self.platform == 'instagram':
                return self._generate_instagram_data(start_date, end_date)
            elif self.platform == 'linkedin':
                return self._generate_linkedin_data(start_date, end_date)
            elif self.platform == 'tiktok':
                return self._generate_tiktok_data(start_date, end_date)
            else:  # Demo-Modus oder unbekannte Plattform
                return self._generate_generic_social_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Fehler beim Abrufen von {self.platform}-Daten: {str(e)}")
            return pd.DataFrame()

    def _generate_twitter_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generiert Beispiel-Twitter-Daten"""
        # Anzahl der zu generierenden Datensätze
        num_tweets = np.random.randint(50, 200)
        num_likes = np.random.randint(100, 500)
        num_follows = np.random.randint(20, 100)
        
        # Generiere Zeitstempel
        tweet_dates = pd.date_range(start=start_date, end=end_date, periods=num_tweets)
        like_dates = pd.date_range(start=start_date, end=end_date, periods=num_likes)
        follow_dates = pd.date_range(start=start_date, end=end_date, periods=num_follows)
        
        # Beispiel-Hashtags und Themen
        hashtags = ['#tech', '#AI', '#datascience', '#privacy', '#blockchain', 
                   '#crypto', '#web3', '#python', '#machinelearning', '#oceanprotocol']
        topics = ['Technology', 'Science', 'Business', 'Politics', 'Entertainment', 
                 'Sports', 'Health', 'Education', 'Environment', 'Cryptocurrency']
        
        # Generiere Tweet-Daten
        tweets_data = []
        for i, date in enumerate(tweet_dates):
            # Zufällige Hashtags auswählen (0-3)
            num_hashtags = np.random.randint(0, 4)
            selected_hashtags = np.random.choice(hashtags, num_hashtags, replace=False).tolist()
            
            # Zufälliges Thema auswählen
            topic = np.random.choice(topics)
            
            # Zufällige Interaktionszahlen
            likes = np.random.randint(0, 50)
            retweets = np.random.randint(0, 10)
            replies = np.random.randint(0, 5)
            
            tweets_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'tweet',
                'content_length': np.random.randint(10, 280),
                'hashtags': selected_hashtags,
                'topic': topic,
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'has_media': np.random.choice([True, False], p=[0.3, 0.7]),
                'is_reply': np.random.choice([True, False], p=[0.2, 0.8]),
                'device': np.random.choice(['Android', 'iPhone', 'Web', 'iPad'], 
                                         p=[0.4, 0.4, 0.15, 0.05])
            })
        
        # Generiere Like-Daten
        likes_data = []
        for i, date in enumerate(like_dates):
            # Zufälliges Thema auswählen
            topic = np.random.choice(topics)
            
            likes_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'like',
                'content_type': np.random.choice(['tweet', 'reply'], p=[0.8, 0.2]),
                'topic': topic,
                'author_follower_count': np.random.randint(100, 100000),
                'has_media': np.random.choice([True, False], p=[0.3, 0.7]),
                'device': np.random.choice(['Android', 'iPhone', 'Web', 'iPad'], 
                                         p=[0.4, 0.4, 0.15, 0.05])
            })
        
        # Generiere Follow-Daten
        follows_data = []
        for i, date in enumerate(follow_dates):
            follows_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'follow',
                'followed_user_verified': np.random.choice([True, False], p=[0.2, 0.8]),
                'followed_user_followers': np.random.randint(100, 1000000),
                'followed_user_category': np.random.choice(topics),
                'device': np.random.choice(['Android', 'iPhone', 'Web', 'iPad'], 
                                         p=[0.4, 0.4, 0.15, 0.05])
            })
        
        # Kombiniere alle Daten
        all_data = tweets_data + likes_data + follows_data
        
        # Konvertiere zu DataFrame
        df = pd.DataFrame(all_data)
        
        # Sortiere nach Zeitstempel
        df = df.sort_values('timestamp')
        
        logger.info(f"Twitter-Daten generiert: {len(df)} Datensätze")
        return df

    def _generate_facebook_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generiert Beispiel-Facebook-Daten"""
        # Anzahl der zu generierenden Datensätze
        num_posts = np.random.randint(30, 100)
        num_likes = np.random.randint(100, 300)
        num_comments = np.random.randint(20, 80)
        num_friends = np.random.randint(10, 50)
        
        # Generiere Zeitstempel
        post_dates = pd.date_range(start=start_date, end=end_date, periods=num_posts)
        like_dates = pd.date_range(start=start_date, end=end_date, periods=num_likes)
        comment_dates = pd.date_range(start=start_date, end=end_date, periods=num_comments)
        friend_dates = pd.date_range(start=start_date, end=end_date, periods=num_friends)
        
        # Beispiel-Kategorien
        categories = ['Family', 'Friends', 'News', 'Entertainment', 'Sports', 
                     'Politics', 'Technology', 'Food', 'Travel', 'Health']
        
        # Generiere Post-Daten
        posts_data = []
        for i, date in enumerate(post_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            # Zufällige Interaktionszahlen
            likes = np.random.randint(0, 50)
            comments = np.random.randint(0, 15)
            shares = np.random.randint(0, 5)
            
            posts_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'post',
                'content_length': np.random.randint(10, 500),
                'category': category,
                'privacy_setting': np.random.choice(['Public', 'Friends', 'Friends of Friends', 'Only Me'], 
                                                  p=[0.2, 0.6, 0.15, 0.05]),
                'likes': likes,
                'comments': comments,
                'shares': shares,
                'has_media': np.random.choice([True, False], p=[0.4, 0.6]),
                'media_type': np.random.choice(['None', 'Photo', 'Video', 'Link'], 
                                             p=[0.6, 0.25, 0.1, 0.05]),
                'device': np.random.choice(['Android', 'iPhone', 'Web', 'iPad'], 
                                         p=[0.4, 0.4, 0.15, 0.05])
            })
        
        # Generiere Like-Daten
        likes_data = []
        for i, date in enumerate(like_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            likes_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'like',
                'content_type': np.random.choice(['post', 'photo', 'video', 'comment'], 
                                               p=[0.5, 0.3, 0.1, 0.1]),
                'category': category,
                'from_friend': np.random.choice([True, False], p=[0.7, 0.3]),
                'from_page': np.random.choice([True, False], p=[0.3, 0.7]),
                'device': np.random.choice(['Android', 'iPhone', 'Web', 'iPad'], 
                                         p=[0.4, 0.4, 0.15, 0.05])
            })
        
        # Generiere Kommentar-Daten
        comments_data = []
        for i, date in enumerate(comment_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            comments_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'comment',
                'content_length': np.random.randint(5, 200),
                'category': category,
                'on_friend_content': np.random.choice([True, False], p=[0.7, 0.3]),
                'on_page_content': np.random.choice([True, False], p=[0.3, 0.7]),
                'has_media': np.random.choice([True, False], p=[0.1, 0.9]),
                'device': np.random.choice(['Android', 'iPhone', 'Web', 'iPad'], 
                                         p=[0.4, 0.4, 0.15, 0.05])
            })
        
        # Generiere Freundschafts-Daten
        friends_data = []
        for i, date in enumerate(friend_dates):
            friends_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'friendship',
                'action': np.random.choice(['add', 'accept'], p=[0.5, 0.5]),
                'mutual_friends': np.random.randint(0, 50),
                'friend_category': np.random.choice(['School', 'Work', 'Family', 'Other'], 
                                                  p=[0.3, 0.3, 0.2, 0.2]),
                'device': np.random.choice(['Android', 'iPhone', 'Web', 'iPad'], 
                                         p=[0.4, 0.4, 0.15, 0.05])
            })
        
        # Kombiniere alle Daten
        all_data = posts_data + likes_data + comments_data + friends_data
        
        # Konvertiere zu DataFrame
        df = pd.DataFrame(all_data)
        
        # Sortiere nach Zeitstempel
        df = df.sort_values('timestamp')
        
        logger.info(f"Facebook-Daten generiert: {len(df)} Datensätze")
        return df

    def _generate_instagram_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generiert Beispiel-Instagram-Daten"""
        # Anzahl der zu generierenden Datensätze
        num_posts = np.random.randint(20, 80)
        num_likes = np.random.randint(100, 400)
        num_comments = np.random.randint(10, 50)
        num_follows = np.random.randint(20, 100)
        
        # Generiere Zeitstempel
        post_dates = pd.date_range(start=start_date, end=end_date, periods=num_posts)
        like_dates = pd.date_range(start=start_date, end=end_date, periods=num_likes)
        comment_dates = pd.date_range(start=start_date, end=end_date, periods=num_comments)
        follow_dates = pd.date_range(start=start_date, end=end_date, periods=num_follows)
        
        # Beispiel-Hashtags und Kategorien
        hashtags = ['#travel', '#food', '#fashion', '#fitness', '#photography', 
                   '#art', '#nature', '#beauty', '#lifestyle', '#love']
        categories = ['Travel', 'Food', 'Fashion', 'Fitness', 'Photography', 
                     'Art', 'Nature', 'Beauty', 'Lifestyle', 'Personal']
        
        # Generiere Post-Daten
        posts_data = []
        for i, date in enumerate(post_dates):
            # Zufällige Hashtags auswählen (0-5)
            num_hashtags = np.random.randint(0, 6)
            selected_hashtags = np.random.choice(hashtags, num_hashtags, replace=False).tolist()
            
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            # Zufällige Interaktionszahlen
            likes = np.random.randint(0, 100)
            comments = np.random.randint(0, 20)
            
            posts_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'post',
                'content_type': np.random.choice(['photo', 'video', 'carousel'], p=[0.7, 0.2, 0.1]),
                'caption_length': np.random.randint(0, 300),
                'hashtags': selected_hashtags,
                'category': category,
                'likes': likes,
                'comments': comments,
                'location_tagged': np.random.choice([True, False], p=[0.3, 0.7]),
                'people_tagged': np.random.randint(0, 5),
                'filter_used': np.random.choice([True, False], p=[0.6, 0.4]),
                'device': np.random.choice(['Android', 'iPhone', 'iPad'], p=[0.3, 0.65, 0.05])
            })
        
        # Generiere Like-Daten
        likes_data = []
        for i, date in enumerate(like_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            likes_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'like',
                'content_type': np.random.choice(['photo', 'video', 'carousel'], p=[0.7, 0.2, 0.1]),
                'category': category,
                'from_following': np.random.choice([True, False], p=[0.8, 0.2]),
                'from_explore': np.random.choice([True, False], p=[0.2, 0.8]),
                'creator_followers': np.random.randint(100, 1000000),
                'device': np.random.choice(['Android', 'iPhone', 'iPad'], p=[0.3, 0.65, 0.05])
            })
        
        # Generiere Kommentar-Daten
        comments_data = []
        for i, date in enumerate(comment_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            comments_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'comment',
                'content_length': np.random.randint(1, 100),
                'category': category,
                'on_following_content': np.random.choice([True, False], p=[0.8, 0.2]),
                'has_emoji': np.random.choice([True, False], p=[0.7, 0.3]),
                'has_mention': np.random.choice([True, False], p=[0.3, 0.7]),
                'device': np.random.choice(['Android', 'iPhone', 'iPad'], p=[0.3, 0.65, 0.05])
            })
        
        # Generiere Follow-Daten
        follows_data = []
        for i, date in enumerate(follow_dates):
            follows_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'follow',
                'followed_user_verified': np.random.choice([True, False], p=[0.2, 0.8]),
                'followed_user_followers': np.random.randint(100, 1000000),
                'followed_user_category': np.random.choice(categories),
                'from_explore': np.random.choice([True, False], p=[0.4, 0.6]),
                'from_suggestion': np.random.choice([True, False], p=[0.3, 0.7]),
                'device': np.random.choice(['Android', 'iPhone', 'iPad'], p=[0.3, 0.65, 0.05])
            })
        
        # Kombiniere alle Daten
        all_data = posts_data + likes_data + comments_data + follows_data
        
        # Konvertiere zu DataFrame
        df = pd.DataFrame(all_data)
        
        # Sortiere nach Zeitstempel
        df = df.sort_values('timestamp')
        
        logger.info(f"Instagram-Daten generiert: {len(df)} Datensätze")
        return df

    def _generate_linkedin_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generiert Beispiel-LinkedIn-Daten"""
        # Anzahl der zu generierenden Datensätze
        num_posts = np.random.randint(10, 50)
        num_likes = np.random.randint(50, 200)
        num_comments = np.random.randint(5, 30)
        num_connections = np.random.randint(10, 50)
        num_job_views = np.random.randint(20, 100)
        
        # Generiere Zeitstempel
        post_dates = pd.date_range(start=start_date, end=end_date, periods=num_posts)
        like_dates = pd.date_range(start=start_date, end=end_date, periods=num_likes)
        comment_dates = pd.date_range(start=start_date, end=end_date, periods=num_comments)
        connection_dates = pd.date_range(start=start_date, end=end_date, periods=num_connections)
        job_view_dates = pd.date_range(start=start_date, end=end_date, periods=num_job_views)
        
        # Beispiel-Kategorien und Branchen
        categories = ['Professional', 'Industry News', 'Career', 'Education', 'Technology', 
                     'Business', 'Marketing', 'Finance', 'HR', 'Leadership']
        industries = ['Technology', 'Finance', 'Healthcare', 'Education', 'Manufacturing', 
                     'Retail', 'Media', 'Consulting', 'Government', 'Non-profit']
        job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'Marketing Manager', 
                     'Financial Analyst', 'HR Manager', 'CEO', 'CTO', 'Director', 'Consultant']
        
        # Generiere Post-Daten
        posts_data = []
        for i, date in enumerate(post_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            # Zufällige Interaktionszahlen
            likes = np.random.randint(0, 50)
            comments = np.random.randint(0, 10)
            shares = np.random.randint(0, 5)
            
            posts_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'post',
                'content_length': np.random.randint(50, 1000),
                'category': category,
                'likes': likes,
                'comments': comments,
                'shares': shares,
                'has_media': np.random.choice([True, False], p=[0.3, 0.7]),
                'media_type': np.random.choice(['None', 'Image', 'Document', 'Video', 'Link'], 
                                             p=[0.7, 0.1, 0.1, 0.05, 0.05]),
                'mentions_company': np.random.choice([True, False], p=[0.2, 0.8]),
                'device': np.random.choice(['Mobile', 'Desktop'], p=[0.4, 0.6])
            })
        
        # Generiere Like-Daten
        likes_data = []
        for i, date in enumerate(like_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            likes_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'like',
                'content_type': np.random.choice(['post', 'article', 'comment'], p=[0.7, 0.2, 0.1]),
                'category': category,
                'from_connection': np.random.choice([True, False], p=[0.7, 0.3]),
                'from_company': np.random.choice([True, False], p=[0.3, 0.7]),
                'creator_industry': np.random.choice(industries),
                'creator_title': np.random.choice(job_titles),
                'device': np.random.choice(['Mobile', 'Desktop'], p=[0.4, 0.6])
            })
        
        # Generiere Kommentar-Daten
        comments_data = []
        for i, date in enumerate(comment_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            comments_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'comment',
                'content_length': np.random.randint(10, 300),
                'category': category,
                'on_connection_content': np.random.choice([True, False], p=[0.7, 0.3]),
                'on_company_content': np.random.choice([True, False], p=[0.3, 0.7]),
                'has_mention': np.random.choice([True, False], p=[0.2, 0.8]),
                'device': np.random.choice(['Mobile', 'Desktop'], p=[0.4, 0.6])
            })
        
        # Generiere Verbindungs-Daten
        connections_data = []
        for i, date in enumerate(connection_dates):
            connections_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'connection',
                'action': np.random.choice(['sent', 'received', 'accepted'], p=[0.4, 0.3, 0.3]),
                'connection_industry': np.random.choice(industries),
                'connection_title': np.random.choice(job_titles),
                'mutual_connections': np.random.randint(0, 30),
                'device': np.random.choice(['Mobile', 'Desktop'], p=[0.4, 0.6])
            })
        
        # Generiere Job-Ansichts-Daten
        job_views_data = []
        for i, date in enumerate(job_view_dates):
            job_views_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'job_view',
                'job_title': np.random.choice(job_titles),
                'job_industry': np.random.choice(industries),
                'job_location': np.random.choice(['Remote', 'Hybrid', 'On-site'], p=[0.3, 0.3, 0.4]),
                'company_size': np.random.choice(['Small', 'Medium', 'Large', 'Enterprise'], 
                                               p=[0.2, 0.3, 0.3, 0.2]),
                'saved_job': np.random.choice([True, False], p=[0.2, 0.8]),
                'applied': np.random.choice([True, False], p=[0.1, 0.9]),
                'device': np.random.choice(['Mobile', 'Desktop'], p=[0.4, 0.6])
            })
        
        # Kombiniere alle Daten
        all_data = posts_data + likes_data + comments_data + connections_data + job_views_data
        
        # Konvertiere zu DataFrame
        df = pd.DataFrame(all_data)
        
        # Sortiere nach Zeitstempel
        df = df.sort_values('timestamp')
        
        logger.info(f"LinkedIn-Daten generiert: {len(df)} Datensätze")
        return df

    def _generate_tiktok_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generiert Beispiel-TikTok-Daten"""
        # Anzahl der zu generierenden Datensätze
        num_videos = np.random.randint(10, 50)
        num_likes = np.random.randint(100, 500)
        num_comments = np.random.randint(10, 50)
        num_follows = np.random.randint(20, 100)
        
        # Generiere Zeitstempel
        video_dates = pd.date_range(start=start_date, end=end_date, periods=num_videos)
        like_dates = pd.date_range(start=start_date, end=end_date, periods=num_likes)
        comment_dates = pd.date_range(start=start_date, end=end_date, periods=num_comments)
        follow_dates = pd.date_range(start=start_date, end=end_date, periods=num_follows)
        
        # Beispiel-Hashtags und Kategorien
        hashtags = ['#fyp', '#foryou', '#viral', '#trending', '#dance', 
                   '#comedy', '#music', '#food', '#fashion', '#fitness']
        categories = ['Entertainment', 'Dance', 'Comedy', 'Education', 'Food', 
                     'Fashion', 'Fitness', 'DIY', 'Technology', 'Travel']
        
        # Generiere Video-Daten
        videos_data = []
        for i, date in enumerate(video_dates):
            # Zufällige Hashtags auswählen (1-5)
            num_hashtags = np.random.randint(1, 6)
            selected_hashtags = np.random.choice(hashtags, num_hashtags, replace=False).tolist()
            
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            # Zufällige Interaktionszahlen
            views = np.random.randint(100, 10000)
            likes = np.random.randint(10, views // 10)
            comments = np.random.randint(0, likes // 5)
            shares = np.random.randint(0, likes // 10)
            
            videos_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'video',
                'duration_seconds': np.random.randint(5, 60),
                'caption_length': np.random.randint(0, 150),
                'hashtags': selected_hashtags,
                'category': category,
                'views': views,
                'likes': likes,
                'comments': comments,
                'shares': shares,
                'sound_original': np.random.choice([True, False], p=[0.3, 0.7]),
                'effect_used': np.random.choice([True, False], p=[0.6, 0.4]),
                'duet': np.random.choice([True, False], p=[0.1, 0.9]),
                'device': np.random.choice(['Android', 'iPhone'], p=[0.3, 0.7])
            })
        
        # Generiere Like-Daten
        likes_data = []
        for i, date in enumerate(like_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            likes_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'like',
                'video_duration': np.random.randint(5, 60),
                'category': category,
                'from_following': np.random.choice([True, False], p=[0.3, 0.7]),
                'from_foryou': np.random.choice([True, False], p=[0.7, 0.3]),
                'creator_followers': np.random.randint(100, 1000000),
                'watched_full': np.random.choice([True, False], p=[0.6, 0.4]),
                'device': np.random.choice(['Android', 'iPhone'], p=[0.3, 0.7])
            })
        
        # Generiere Kommentar-Daten
        comments_data = []
        for i, date in enumerate(comment_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            comments_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'comment',
                'content_length': np.random.randint(1, 150),
                'category': category,
                'on_following_content': np.random.choice([True, False], p=[0.3, 0.7]),
                'has_emoji': np.random.choice([True, False], p=[0.7, 0.3]),
                'has_mention': np.random.choice([True, False], p=[0.2, 0.8]),
                'device': np.random.choice(['Android', 'iPhone'], p=[0.3, 0.7])
            })
        
        # Generiere Follow-Daten
        follows_data = []
        for i, date in enumerate(follow_dates):
            follows_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'follow',
                'followed_user_verified': np.random.choice([True, False], p=[0.2, 0.8]),
                'followed_user_followers': np.random.randint(100, 10000000),
                'followed_user_category': np.random.choice(categories),
                'from_foryou': np.random.choice([True, False], p=[0.6, 0.4]),
                'from_suggested': np.random.choice([True, False], p=[0.3, 0.7]),
                'device': np.random.choice(['Android', 'iPhone'], p=[0.3, 0.7])
            })
        
        # Kombiniere alle Daten
        all_data = videos_data + likes_data + comments_data + follows_data
        
        # Konvertiere zu DataFrame
        df = pd.DataFrame(all_data)
        
        # Sortiere nach Zeitstempel
        df = df.sort_values('timestamp')
        
        logger.info(f"TikTok-Daten generiert: {len(df)} Datensätze")
        return df

    def _generate_generic_social_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generiert generische Social-Media-Daten für den Demo-Modus"""
        # Anzahl der zu generierenden Datensätze
        num_posts = np.random.randint(30, 100)
        num_likes = np.random.randint(50, 200)
        num_comments = np.random.randint(10, 50)
        
        # Generiere Zeitstempel
        post_dates = pd.date_range(start=start_date, end=end_date, periods=num_posts)
        like_dates = pd.date_range(start=start_date, end=end_date, periods=num_likes)
        comment_dates = pd.date_range(start=start_date, end=end_date, periods=num_comments)
        
        # Beispiel-Kategorien
        categories = ['Entertainment', 'News', 'Personal', 'Professional', 'Education', 
                     'Lifestyle', 'Technology', 'Travel', 'Food', 'Health']
        
        # Generiere Post-Daten
        posts_data = []
        for i, date in enumerate(post_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            # Zufällige Interaktionszahlen
            likes = np.random.randint(0, 50)
            comments = np.random.randint(0, 10)
            shares = np.random.randint(0, 5)
            
            posts_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'post',
                'content_length': np.random.randint(10, 500),
                'category': category,
                'likes': likes,
                'comments': comments,
                'shares': shares,
                'has_media': np.random.choice([True, False], p=[0.4, 0.6]),
                'platform': np.random.choice(['Generic Social', 'Demo Platform']),
                'device': np.random.choice(['Mobile', 'Desktop'], p=[0.7, 0.3])
            })
        
        # Generiere Like-Daten
        likes_data = []
        for i, date in enumerate(like_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            likes_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'like',
                'content_type': np.random.choice(['post', 'photo', 'video', 'article'], 
                                               p=[0.4, 0.3, 0.2, 0.1]),
                'category': category,
                'platform': np.random.choice(['Generic Social', 'Demo Platform']),
                'device': np.random.choice(['Mobile', 'Desktop'], p=[0.7, 0.3])
            })
        
        # Generiere Kommentar-Daten
        comments_data = []
        for i, date in enumerate(comment_dates):
            # Zufällige Kategorie auswählen
            category = np.random.choice(categories)
            
            comments_data.append({
                'user_id': self.user_id,
                'timestamp': date,
                'activity_type': 'comment',
                'content_length': np.random.randint(5, 200),
                'category': category,
                'has_media': np.random.choice([True, False], p=[0.1, 0.9]),
                'platform': np.random.choice(['Generic Social', 'Demo Platform']),
                'device': np.random.choice(['Mobile', 'Desktop'], p=[0.7, 0.3])
            })
        
        # Kombiniere alle Daten
        all_data = posts_data + likes_data + comments_data
        
        # Konvertiere zu DataFrame
        df = pd.DataFrame(all_data)
        
        # Sortiere nach Zeitstempel
        df = df.sort_values('timestamp')
        
        logger.info(f"Generische Social-Media-Daten generiert: {len(df)} Datensätze")
        return df

    def get_platform_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken zur Social-Media-Plattform zurück.

        Returns:
            Dict: Statistiken zur Plattform
        """
        if not self.data is None and isinstance(self.data, pd.DataFrame) and not self.data.empty:
            stats = {
                'platform': self.platform,
                'total_records': len(self.data),
                'date_range': {
                    'start': self.data['timestamp'].min().isoformat() if 'timestamp' in self.data.columns else None,
                    'end': self.data['timestamp'].max().isoformat() if 'timestamp' in self.data.columns else None
                },
                'activity_types': {}
            }
            
            # Aktivitätstypen zählen
            if 'activity_type' in self.data.columns:
                activity_counts = self.data['activity_type'].value_counts().to_dict()
                stats['activity_types'] = activity_counts
            
            # Kategorien zählen
            if 'category' in self.data.columns:
                category_counts = self.data['category'].value_counts().to_dict()
                stats['categories'] = category_counts
            
            # Geräte zählen
            if 'device' in self.data.columns:
                device_counts = self.data['device'].value_counts().to_dict()
                stats['devices'] = device_counts
            
            return stats
        else:
            return {
                'platform': self.platform,
                'total_records': 0,
                'error': 'Keine Daten verfügbar. Rufe get_data() zuerst auf.'
            }