"""
Ultra-Minimal Calendar Agent Tools Module

Single tool that provides raw calendar data to LLM with standardized date handling.
LLM handles ALL processing:
- Date range parsing
- Link extraction and categorization  
- Event filtering
- Response formatting

Date Format Standards:
- Accepts: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ (UTC)
- Converts internally to IST for calendar operations
- Eliminates date parsing errors through standardization
"""

import json
import logging
from datetime import datetime, timedelta

from langchain_core.tools import tool

from beetu_v2.agents.linkagent.calendar_client import calendar_client
from beetu_v2.agents.linkagent.utils.datetime_utils import (
    parse_date_string,
    format_ist_datetime
)
from beetu_v2.agents.linkagent.constants import IST_TIMEZONE

# Set up logging
logger = logging.getLogger(__name__)


@tool
def get_calendar_events(start_date: str, end_date: str) -> str:
    """
    Fetch raw calendar events between specified dates. 
    LLM will extract links, categorize, filter and format the response.
    
    Args:
        start_date: Start date in UTC ISO format (YYYY-MM-DDTHH:MM:SSZ) or simple date (YYYY-MM-DD)
        end_date: End date in UTC ISO format (YYYY-MM-DDTHH:MM:SSZ) or simple date (YYYY-MM-DD)
        
    Returns:
        JSON string with raw event data - LLM handles all processing
    """
    # Input validation
    if not start_date or not end_date:
        return json.dumps({"error": "Both start_date and end_date are required"})
    
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        return json.dumps({"error": "Dates must be strings"})
    
    try:
        # Parse dates - handle both ISO datetime and simple date formats
        start_datetime = parse_date_string(start_date)
        end_datetime = parse_date_string(end_date)
        
        if not start_datetime or not end_datetime:
            return json.dumps({
                "error": "Invalid date format. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ (UTC)."
            })
        
        # Enforce 3-day limits for performance and relevance
        today = datetime.now(IST_TIMEZONE).replace(hour=0, minute=0, second=0, microsecond=0)
        max_past = today - timedelta(days=3)
        max_future = today + timedelta(days=3)
        
        # Validate date ranges (convert to IST for comparison)
        start_ist = start_datetime.astimezone(IST_TIMEZONE).replace(hour=0, minute=0, second=0, microsecond=0)
        end_ist = end_datetime.astimezone(IST_TIMEZONE).replace(hour=0, minute=0, second=0, microsecond=0)
        
        if start_ist < max_past:
            return json.dumps({
                "error": f"Date range too far back. Maximum is 3 days ago ({max_past.strftime('%Y-%m-%d')}). Requested start: {start_ist.strftime('%Y-%m-%d')}"
            })
        
        if end_ist > max_future:
            return json.dumps({
                "error": f"Date range too far ahead. Maximum is 3 days from today ({max_future.strftime('%Y-%m-%d')}). Requested end: {end_ist.strftime('%Y-%m-%d')}"
            })
        
        # Get raw events from calendar (function handles IST conversion internally)
        events = calendar_client.get_events(start_datetime, end_datetime)
        
        # Minimal processing - just convert to basic dict
        events_data = []
        for event in events:
            events_data.append({
                "id": event.id,
                "title": event.title,
                "description": event.description or "",
                "start_time": format_ist_datetime(event.start_time),
                "end_time": format_ist_datetime(event.end_time) if event.end_time else "",
                "location": event.location or "",
                "organizer": event.organizer or "",
                "attendees": event.attendees or [],
                "is_recurring": event.is_recurring or False,
                "status": event.status or "confirmed"
            })
        
        return json.dumps({
            "events": events_data,
            "count": len(events_data),
            "date_range": f"{start_date} to {end_date}",
            "timezone": "IST"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch events: {str(e)}"})


@tool
def get_current_datetime() -> str:
    """
    Get current date and time in standardized format to help with date calculations.
    
    Returns:
        JSON string with current datetime information in multiple formats
    """
    try:
        now = datetime.now(IST_TIMEZONE)
        
        return json.dumps({
            "current_date_simple": now.strftime("%Y-%m-%d"),
            "current_datetime_utc": now.astimezone(datetime.now().astimezone().utcoffset()).isoformat(),
            "current_datetime_ist": format_ist_datetime(now),
            "week_start": (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d"),
            "week_end": (now + timedelta(days=6-now.weekday())).strftime("%Y-%m-%d"),
            "month_start": now.replace(day=1).strftime("%Y-%m-%d"),
            "next_week_start": (now + timedelta(days=7-now.weekday())).strftime("%Y-%m-%d"),
            "timezone": "IST",
            "helpful_formats": {
                "simple_date": "YYYY-MM-DD",
                "utc_iso": "YYYY-MM-DDTHH:MM:SSZ"
            }
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to get current datetime: {str(e)}"})
