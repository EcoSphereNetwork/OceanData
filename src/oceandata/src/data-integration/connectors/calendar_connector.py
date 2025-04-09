"""
OceanData - Calendar Connector

Dieser Konnektor ermöglicht die Integration von Kalenderdaten in OceanData.
Unterstützte Kalendertypen: Google Calendar, Microsoft Outlook, Apple iCloud, lokale ICS-Dateien
"""

import pandas as pd
import numpy as np
import json
import logging
import os
import requests
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Union
import hashlib
import uuid
import re
import pytz

# Importiere Basisklassen
from ..base import DataConnector, DataCategory, PrivacyLevel

# Konfiguriere Logger
logger = logging.getLogger("OceanData.Connectors.Calendar")

class CalendarConnector(DataConnector):
    """Konnektor für Kalenderdaten."""

    SUPPORTED_CALENDARS = ['google', 'outlook', 'icloud', 'ics', 'demo']

    def __init__(self, user_id: str, calendar_type: str, api_credentials: Dict[str, str] = None):
        """
        Initialisiert einen Kalender-Datenkonnektor.

        Args:
            user_id: ID des Benutzers
            calendar_type: Kalendertyp (google, outlook, icloud, ics)
            api_credentials: API-Zugangsdaten für den Kalender (optional)
        """
        if calendar_type.lower() not in self.SUPPORTED_CALENDARS:
            raise ValueError(f"Nicht unterstützter Kalendertyp: {calendar_type}. "
                           f"Unterstützte Kalendertypen: {', '.join(self.SUPPORTED_CALENDARS)}")

        super().__init__(f'calendar_{calendar_type.lower()}', user_id, DataCategory.CUSTOM)
        
        self.calendar_type = calendar_type.lower()
        self.api_credentials = api_credentials or {}
        self.api_client = None
        self.calendars = []  # Liste der verfügbaren Kalender
        
        # Kalender-spezifische Metadaten
        self.metadata.update({
            "calendar_type": self.calendar_type,
            "source_details": {
                "name": f"{calendar_type.capitalize()} Calendar Data",
                "version": "1.0",
                "description": f"Calendar events and scheduling data from {calendar_type.capitalize()}"
            }
        })

    def connect(self) -> bool:
        """
        Verbindung zum Kalender-API herstellen.

        Returns:
            bool: True, wenn die Verbindung erfolgreich hergestellt wurde
        """
        try:
            # In einer echten Implementierung würden wir hier die Verbindung zur API herstellen
            # Für das MVP simulieren wir eine erfolgreiche Verbindung
            
            # Prüfe, ob API-Zugangsdaten vorhanden sind
            if not self.api_credentials and self.calendar_type != 'demo':
                logger.warning(f"Keine API-Zugangsdaten für {self.calendar_type} vorhanden. "
                              f"Verwende Demo-Modus.")
                self.calendar_type = 'demo'
            
            logger.info(f"Verbindung zu {self.calendar_type}-Kalender für Benutzer {self.user_id} hergestellt")
            
            # Simuliere API-Client
            self.api_client = {
                "calendar_type": self.calendar_type,
                "connected": True,
                "user_id": self.user_id,
                "connection_time": datetime.now().isoformat()
            }
            
            # Simuliere verfügbare Kalender
            if self.calendar_type == 'google':
                self.calendars = [
                    {"id": "primary", "name": "Hauptkalender", "color": "#4285F4", "selected": True},
                    {"id": "work", "name": "Arbeit", "color": "#0B8043", "selected": True},
                    {"id": "personal", "name": "Persönlich", "color": "#8E24AA", "selected": True},
                    {"id": "family", "name": "Familie", "color": "#D50000", "selected": True}
                ]
            elif self.calendar_type == 'outlook':
                self.calendars = [
                    {"id": "primary", "name": "Kalender", "color": "#0078D7", "selected": True},
                    {"id": "work", "name": "Arbeit", "color": "#107C10", "selected": True},
                    {"id": "personal", "name": "Persönlich", "color": "#5C2D91", "selected": True}
                ]
            elif self.calendar_type == 'icloud':
                self.calendars = [
                    {"id": "primary", "name": "iCloud", "color": "#FF9500", "selected": True},
                    {"id": "work", "name": "Arbeit", "color": "#007AFF", "selected": True},
                    {"id": "personal", "name": "Persönlich", "color": "#FF2D55", "selected": True}
                ]
            else:  # Demo oder ICS
                self.calendars = [
                    {"id": "primary", "name": "Hauptkalender", "color": "#4285F4", "selected": True},
                    {"id": "work", "name": "Arbeit", "color": "#0B8043", "selected": True}
                ]
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Verbinden mit {self.calendar_type}-Kalender: {str(e)}")
            return False

    def fetch_data(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Kalenderdaten abrufen.

        Args:
            start_date: Startdatum für die Abfrage (optional)
            end_date: Enddatum für die Abfrage (optional)

        Returns:
            pd.DataFrame: Die abgerufenen Kalenderdaten
        """
        # Prüfe, ob eine Verbindung besteht
        if not self.api_client:
            logger.error(f"Keine Verbindung zum {self.calendar_type}-Kalender. Rufe connect() zuerst auf.")
            return pd.DataFrame()
        
        try:
            # Standardzeitraum: 1 Jahr zurück bis 1 Jahr in die Zukunft
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365)
            if end_date is None:
                end_date = datetime.now() + timedelta(days=365)
            
            # In einer echten Implementierung würden wir hier die tatsächlichen Daten abrufen
            # Für das MVP generieren wir realistische Beispieldaten
            
            # Generiere Kalenderdaten
            return self._generate_calendar_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Fehler beim Abrufen von {self.calendar_type}-Kalenderdaten: {str(e)}")
            return pd.DataFrame()

    def _generate_calendar_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generiert Beispiel-Kalenderdaten"""
        # Anzahl der zu generierenden Ereignisse pro Kalender
        events_per_calendar = {
            "primary": np.random.randint(100, 300),
            "work": np.random.randint(50, 150),
            "personal": np.random.randint(30, 100),
            "family": np.random.randint(20, 80)
        }
        
        # Beispiel-Kategorien für Ereignisse
        categories = {
            "primary": ['Meeting', 'Termin', 'Erinnerung', 'Geburtstag', 'Feiertag', 'Sonstiges'],
            "work": ['Meeting', 'Konferenz', 'Deadline', 'Präsentation', 'Schulung', 'Geschäftsreise'],
            "personal": ['Sport', 'Arzttermin', 'Hobby', 'Einkaufen', 'Freizeit', 'Bildung'],
            "family": ['Familientreffen', 'Ausflug', 'Urlaub', 'Geburtstag', 'Jubiläum', 'Schule']
        }
        
        # Beispiel-Orte für Ereignisse
        locations = {
            "primary": ['', 'Zuhause', 'Büro', 'Stadt', 'Online'],
            "work": ['Büro', 'Konferenzraum A', 'Konferenzraum B', 'Meetingraum', 'Kunde', 'Online'],
            "personal": ['Fitnessstudio', 'Arztpraxis', 'Café', 'Park', 'Kino', 'Restaurant'],
            "family": ['Zuhause', 'Schule', 'Park', 'Restaurant', 'Freizeitpark', 'Verwandte']
        }
        
        # Beispiel-Teilnehmer für Ereignisse
        attendees_pool = {
            "work": ['Kollege A', 'Kollege B', 'Chef', 'Kunde A', 'Kunde B', 'Partner', 'Team'],
            "personal": ['Freund A', 'Freund B', 'Partner', 'Trainer', 'Arzt', 'Lehrer'],
            "family": ['Partner', 'Kind A', 'Kind B', 'Eltern', 'Geschwister', 'Verwandte']
        }
        
        # Alle Ereignisse
        all_events = []
        
        # Generiere Ereignisse für jeden Kalender
        for calendar in self.calendars:
            calendar_id = calendar["id"]
            calendar_name = calendar["name"]
            calendar_color = calendar["color"]
            
            # Anzahl der Ereignisse für diesen Kalender
            num_events = events_per_calendar.get(calendar_id, np.random.randint(30, 100))
            
            # Kategorien für diesen Kalender
            cal_categories = categories.get(calendar_id, categories["primary"])
            cal_locations = locations.get(calendar_id, locations["primary"])
            cal_attendees = attendees_pool.get(calendar_id, [])
            
            # Generiere Ereignisse
            for i in range(num_events):
                # Zufälliges Startdatum zwischen start_date und end_date
                event_date = start_date + (end_date - start_date) * np.random.random()
                
                # Zufällige Dauer (15 Minuten bis 3 Stunden)
                duration_minutes = np.random.choice([15, 30, 45, 60, 90, 120, 180])
                
                # Zufällige Kategorie
                category = np.random.choice(cal_categories)
                
                # Zufälliger Ort
                location = np.random.choice(cal_locations)
                
                # Zufällige Teilnehmer (0-3)
                num_attendees = np.random.randint(0, 4) if cal_attendees else 0
                attendees = np.random.choice(cal_attendees, num_attendees, replace=False).tolist() if num_attendees > 0 else []
                
                # Zufällige Eigenschaften
                is_all_day = np.random.random() < 0.1  # 10% ganztägige Ereignisse
                has_reminder = np.random.random() < 0.7  # 70% haben Erinnerungen
                is_recurring = np.random.random() < 0.2  # 20% sind wiederkehrend
                is_online = "Online" in location or np.random.random() < 0.3  # 30% sind online (zusätzlich zu explizit "Online")
                
                # Für wiederkehrende Ereignisse
                recurrence_type = None
                recurrence_interval = None
                recurrence_count = None
                
                if is_recurring:
                    recurrence_type = np.random.choice(['daily', 'weekly', 'monthly', 'yearly'], p=[0.1, 0.6, 0.2, 0.1])
                    recurrence_interval = np.random.choice([1, 2, 3, 4])
                    recurrence_count = np.random.randint(2, 12)
                
                # Erstelle Ereignis
                event = {
                    'user_id': self.user_id,
                    'calendar_id': calendar_id,
                    'calendar_name': calendar_name,
                    'calendar_color': calendar_color,
                    'event_id': str(uuid.uuid4()),
                    'title': f"{category} {i+1}",
                    'start_time': event_date if not is_all_day else event_date.replace(hour=0, minute=0, second=0, microsecond=0),
                    'end_time': event_date + timedelta(minutes=duration_minutes) if not is_all_day else event_date.replace(hour=23, minute=59, second=59),
                    'is_all_day': is_all_day,
                    'category': category,
                    'location': location,
                    'attendees': attendees,
                    'attendee_count': len(attendees),
                    'has_reminder': has_reminder,
                    'reminder_minutes_before': np.random.choice([5, 10, 15, 30, 60]) if has_reminder else None,
                    'is_recurring': is_recurring,
                    'recurrence_type': recurrence_type,
                    'recurrence_interval': recurrence_interval,
                    'recurrence_count': recurrence_count,
                    'is_online': is_online,
                    'online_meeting_url': f"https://meet.example.com/{uuid.uuid4().hex[:8]}" if is_online else None,
                    'created_at': event_date - timedelta(days=np.random.randint(1, 30)),
                    'updated_at': event_date - timedelta(days=np.random.randint(0, 5)),
                    'status': np.random.choice(['confirmed', 'tentative', 'cancelled'], p=[0.8, 0.15, 0.05])
                }
                
                all_events.append(event)
        
        # Konvertiere zu DataFrame
        df = pd.DataFrame(all_events)
        
        # Sortiere nach Startzeit
        df = df.sort_values('start_time')
        
        logger.info(f"{self.calendar_type}-Kalenderdaten generiert: {len(df)} Ereignisse")
        return df

    def get_available_calendars(self) -> List[Dict[str, Any]]:
        """
        Gibt eine Liste der verfügbaren Kalender zurück.

        Returns:
            List[Dict]: Liste der verfügbaren Kalender
        """
        if not self.api_client:
            logger.error(f"Keine Verbindung zum {self.calendar_type}-Kalender. Rufe connect() zuerst auf.")
            return []
        
        return self.calendars

    def set_calendar_selection(self, calendar_ids: List[str]) -> bool:
        """
        Wählt Kalender für die Datenabfrage aus.

        Args:
            calendar_ids: Liste der Kalender-IDs, die ausgewählt werden sollen

        Returns:
            bool: True, wenn die Auswahl erfolgreich war
        """
        if not self.api_client:
            logger.error(f"Keine Verbindung zum {self.calendar_type}-Kalender. Rufe connect() zuerst auf.")
            return False
        
        try:
            # Setze alle Kalender auf nicht ausgewählt
            for calendar in self.calendars:
                calendar["selected"] = False
            
            # Wähle die angegebenen Kalender aus
            selected_count = 0
            for calendar in self.calendars:
                if calendar["id"] in calendar_ids:
                    calendar["selected"] = True
                    selected_count += 1
            
            logger.info(f"{selected_count} Kalender ausgewählt")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Auswählen der Kalender: {str(e)}")
            return False

    def create_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Erstellt ein neues Ereignis im Kalender.

        Args:
            event_data: Daten für das neue Ereignis

        Returns:
            Dict: Erstelltes Ereignis oder Fehlermeldung
        """
        if not self.api_client:
            logger.error(f"Keine Verbindung zum {self.calendar_type}-Kalender. Rufe connect() zuerst auf.")
            return {"status": "error", "message": "Keine Verbindung zum Kalender"}
        
        try:
            # In einer echten Implementierung würden wir hier das Ereignis erstellen
            # Für das MVP simulieren wir eine erfolgreiche Erstellung
            
            # Prüfe, ob die erforderlichen Felder vorhanden sind
            required_fields = ['title', 'start_time', 'end_time', 'calendar_id']
            for field in required_fields:
                if field not in event_data:
                    return {"status": "error", "message": f"Feld '{field}' fehlt"}
            
            # Erstelle eine Ereignis-ID
            event_id = str(uuid.uuid4())
            
            # Füge Metadaten hinzu
            event_data.update({
                'event_id': event_id,
                'user_id': self.user_id,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'status': 'confirmed'
            })
            
            logger.info(f"Ereignis '{event_data['title']}' erstellt (simuliert)")
            
            return {
                "status": "success",
                "message": "Ereignis erfolgreich erstellt",
                "event": event_data
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Ereignisses: {str(e)}")
            return {"status": "error", "message": f"Fehler beim Erstellen des Ereignisses: {str(e)}"}

    def get_calendar_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken zu den Kalenderdaten zurück.

        Returns:
            Dict: Statistiken zu den Kalenderdaten
        """
        if not self.data is None and isinstance(self.data, pd.DataFrame) and not self.data.empty:
            stats = {
                'calendar_type': self.calendar_type,
                'total_events': len(self.data),
                'date_range': {
                    'start': self.data['start_time'].min().isoformat() if 'start_time' in self.data.columns else None,
                    'end': self.data['end_time'].max().isoformat() if 'end_time' in self.data.columns else None
                },
                'calendars': {}
            }
            
            # Ereignisse pro Kalender zählen
            if 'calendar_id' in self.data.columns:
                calendar_counts = self.data['calendar_id'].value_counts().to_dict()
                
                for calendar in self.calendars:
                    calendar_id = calendar["id"]
                    if calendar_id in calendar_counts:
                        stats['calendars'][calendar_id] = {
                            'name': calendar["name"],
                            'event_count': calendar_counts[calendar_id]
                        }
            
            # Kategorien zählen
            if 'category' in self.data.columns:
                category_counts = self.data['category'].value_counts().to_dict()
                stats['categories'] = category_counts
            
            # Ganztägige vs. zeitgebundene Ereignisse
            if 'is_all_day' in self.data.columns:
                all_day_count = self.data['is_all_day'].sum()
                stats['all_day_events'] = int(all_day_count)
                stats['timed_events'] = len(self.data) - int(all_day_count)
            
            # Wiederkehrende Ereignisse
            if 'is_recurring' in self.data.columns:
                recurring_count = self.data['is_recurring'].sum()
                stats['recurring_events'] = int(recurring_count)
                stats['single_events'] = len(self.data) - int(recurring_count)
            
            # Online-Ereignisse
            if 'is_online' in self.data.columns:
                online_count = self.data['is_online'].sum()
                stats['online_events'] = int(online_count)
                stats['in_person_events'] = len(self.data) - int(online_count)
            
            # Ereignisse pro Wochentag
            if 'start_time' in self.data.columns:
                self.data['weekday'] = self.data['start_time'].dt.dayofweek
                weekday_counts = self.data['weekday'].value_counts().sort_index().to_dict()
                weekday_names = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
                stats['events_by_weekday'] = {weekday_names[day]: count for day, count in weekday_counts.items()}
            
            # Ereignisse pro Stunde
            if 'start_time' in self.data.columns and 'is_all_day' in self.data.columns:
                timed_events = self.data[~self.data['is_all_day']]
                if not timed_events.empty:
                    timed_events['hour'] = timed_events['start_time'].dt.hour
                    hour_counts = timed_events['hour'].value_counts().sort_index().to_dict()
                    stats['events_by_hour'] = {f"{hour:02d}:00": count for hour, count in hour_counts.items()}
            
            return stats
        else:
            return {
                'calendar_type': self.calendar_type,
                'total_events': 0,
                'error': 'Keine Daten verfügbar. Rufe get_data() zuerst auf.'
            }