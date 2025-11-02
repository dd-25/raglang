"""
Google Calendar Client
"""

import logging
import re
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from beetu_v2.agents.linkagent.constants import (
    CALENDAR_SETTINGS,
    GOOGLE_CALENDAR_SETTINGS
)
from beetu_v2.agents.linkagent.dto import CalendarEvent
from beetu_v2.agents.linkagent.utils.datetime_utils import (
    parse_google_calendar_datetime,
    ist_datetime_to_utc_iso,
    format_ist_datetime
)

# Set up logging
logger = logging.getLogger(__name__)


class GoogleCalendarClient:
    """
    Google Calendar implementation of the CalendarClient interface.
    
    This class provides Google Calendar-specific implementations of the
    calendar client interface methods using the Google Calendar API.
    """
    
    def __init__(self, credentials_file: str = GOOGLE_CALENDAR_SETTINGS.CREDENTIALS_FILE):
        """
        Initialize the Google Calendar client.
        
        Args:
            credentials_file: Path to the Google Calendar API credentials file
        """
        self.credentials_file = credentials_file
        self.service = None
        self._initialize_service()
    
    def _initialize_service(self):
        """
        Initialize the Google Calendar API service.
        
        Authenticates with the Google Calendar API using OAuth 2.0 and creates
        a service object for making API calls.
        """
        try:
            creds = None
            token_file = GOOGLE_CALENDAR_SETTINGS.TOKEN_FILE
            scopes = GOOGLE_CALENDAR_SETTINGS.SCOPES
            
            # Check if token file exists and load credentials
            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(token_file, scopes)
            
            # If no credentials or they're invalid, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, scopes)
                    # Use a fixed port (8006) for the local OAuth redirect instead
                    # of a random available port. This keeps the redirect URI
                    # consistent, so you only need to whitelist:
                    #     http://localhost:8006/
                    # in the Google Cloud Console.
                    #
                    # NOTE:
                    # If port 8006 is already in use you will get an OSError.
                    # In that case, free the port (or pick another constant
                    # port and update both the code and the OAuth redirect URI).
                    creds = flow.run_local_server(port=8006)
                # Save the credentials for the next run
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
            
            # Build the service object
            self.service = build('calendar', GOOGLE_CALENDAR_SETTINGS.API_VERSION, credentials=creds)
            
            # Test the service by getting calendar list
            try:
                calendars = self.service.calendarList().list().execute()
                calendar_count = len(calendars.get('items', []))
                if calendar_count == 0:
                    logger.warning("No calendars accessible")
                elif calendar_count > 0:
                    primary_found = any(cal.get('primary', False) for cal in calendars['items'])
                    if not primary_found:
                        logger.warning("No primary calendar found!")
            except Exception as test_e:
                logger.error(f"Cannot access calendars: {test_e}")
            
        except FileNotFoundError:
            logger.error(f"Credentials file not found: {self.credentials_file}")
            logger.warning("Using placeholder data as fallback due to missing credentials")
            self.service = None
        except Exception as e:
            logger.error(f"Failed to initialize Google Calendar API service: {str(e)}")
            logger.warning("Using placeholder data as fallback due to authentication failure")
            self.service = None
    
    def get_events(
        self, 
        start_date: datetime,
        end_date: datetime,
        calendar_id: str = CALENDAR_SETTINGS.PRIMARY_CALENDAR_ID
    ) -> List[CalendarEvent]:
        """
        Retrieve Google Calendar events between specified dates.
        
        Args:
            start_date: Start date/time in IST timezone
            end_date: End date/time in IST timezone
            calendar_id: ID of the calendar to query
            
        Returns:
            List of CalendarEvent objects representing events in the date range
        """
        try:
            # Always enforce primary calendar usage
            calendar_id = CALENDAR_SETTINGS.PRIMARY_CALENDAR_ID
            
            # Convert to UTC ISO format for API call
            time_min_utc = ist_datetime_to_utc_iso(start_date)
            time_max_utc = ist_datetime_to_utc_iso(end_date)
            
            # If service is available, call the Google Calendar API
            if self.service:
                events_result = self.service.events().list(
                    calendarId=calendar_id,
                    timeMin=time_min_utc,
                    timeMax=time_max_utc,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()
                
                events = events_result.get('items', [])
            else:
                # Fallback to placeholder data if service is not available
                logger.warning("Using placeholder data for events (no Google Calendar service)")
                events = self._get_placeholder_events(10, start_date, end_date)
            
            # Convert Google Calendar events to CalendarEvent DTOs
            return [self._convert_to_calendar_event(event) for event in events]
            
        except Exception as e:
            logger.error(f"Error fetching events: {str(e)}")
            # Fallback to placeholder data on error
            events = self._get_placeholder_events(10, start_date, end_date)
            return [self._convert_to_calendar_event(event) for event in events]
    
    def _convert_to_calendar_event(self, google_event: Dict[str, Any]) -> CalendarEvent:
        """
        Convert a Google Calendar event to a CalendarEvent DTO.
        
        Args:
            google_event: Google Calendar event data
            
        Returns:
            CalendarEvent object with data from the Google Calendar event
        """
        # Extract start and end times using the centralized parsing function
        start_time = parse_google_calendar_datetime(google_event.get('start', {}))
        end_time = parse_google_calendar_datetime(google_event.get('end', {}))
        
        # Extract attendees
        attendees = []
        for attendee in google_event.get('attendees', []):
            email = attendee.get('email')
            if email:
                attendees.append(email)
        
        # LLM will handle all link extraction - no need to extract here!
        
        # Create CalendarEvent object
        return CalendarEvent(
            id=google_event.get('id', ''),
            title=google_event.get('summary', 'Untitled Event'),
            description=google_event.get('description', ''),
            start_time=start_time,
            end_time=end_time,
            location=google_event.get('location', ''),
            organizer=google_event.get('organizer', {}).get('email', ''),
            attendees=attendees,
            is_recurring=bool(google_event.get('recurrence', [])),
            status=google_event.get('status', 'confirmed')
        )
    
    def _get_placeholder_events(
        self, 
        count: int, 
        time_min, 
        time_max
    ) -> List[Dict[str, Any]]:
        """
        Generate placeholder events for testing.
        
        Args:
            count: Number of events to generate
            time_min: Minimum time for events (in IST)
            time_max: Maximum time for events (in IST)
            
        Returns:
            List of placeholder event dictionaries
        """
        events = []
        
        # Event templates with different link types
        templates = [
            {
                'summary': 'Team Meeting',
                'description': 'Weekly team sync with Zoom link: https://zoom.us/j/{id}',
                'duration': 1  # hours
            },
            {
                'summary': 'Yoga Session',
                'description': 'Join our yoga session on YouTube: https://youtube.com/watch?v={id}',
                'duration': 1.5
            },
            {
                'summary': 'Project Planning',
                'description': 'Planning session on Google Meet: https://meet.google.com/{id}',
                'duration': 2
            },
            {
                'summary': 'Training Workshop',
                'description': 'Workshop materials: https://docs.google.com/document/d/{id} and recording: https://youtu.be/{id}',
                'duration': 3
            },
            {
                'summary': 'Client Call',
                'description': 'Microsoft Teams meeting: https://teams.microsoft.com/l/meetup-join/{id}',
                'duration': 1
            }
        ]
        
        # Calculate total time range
        total_duration = (time_max - time_min).total_seconds()
        
        # Generate events
        for i in range(count):
            template = templates[i % len(templates)]
            
            # Calculate event time - distribute events evenly across the time range
            start_offset = timedelta(seconds=total_duration * (i / count))
            start_time = time_min + start_offset
            
            # Calculate end time
            end_time = start_time + timedelta(hours=template['duration'])
            
            # Generate random ID
            event_id = f"event{i}_{hash(start_time) % 10000}"
            
            # Create event with ISO format strings
            event = {
                'id': event_id,
                'summary': template['summary'],
                'description': template['description'].format(id=event_id),
                'start': {'dateTime': start_time.isoformat()},
                'end': {'dateTime': end_time.isoformat()},
                'status': 'confirmed',
                'htmlLink': f'https://calendar.google.com/calendar/event?eid={event_id}'
            }
            
            events.append(event)
        
        return events


# Global calendar client instance
calendar_client = GoogleCalendarClient()
