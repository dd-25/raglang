"""
Calendar Link Agent Datetime Utilities

This module provides datetime parsing and formatting utilities:
- parse_date_string: Parse a date string into a datetime object
- format_ist_datetime: Format a datetime object in IST timezone
- parse_google_calendar_datetime: Parse Google Calendar datetime format
- ist_datetime_to_utc_iso: Convert IST datetime to UTC ISO format
"""

from datetime import datetime
import pytz

from beetu_v2.agents.linkagent.constants import IST_TIMEZONE

def parse_date_string(date_str: str) -> datetime:
    """
    Parse a date string into a datetime object.
    Supports both simple dates and UTC ISO datetime formats.
    
    Args:
        date_str: Date string in YYYY-MM-DD format or UTC ISO format (YYYY-MM-DDTHH:MM:SSZ)
        
    Returns:
        datetime object in IST timezone
    """
    if not date_str:
        return None
        
    try:
        # Try UTC ISO format first (YYYY-MM-DDTHH:MM:SSZ)
        if 'T' in date_str and ('Z' in date_str or '+' in date_str or '-' in date_str[-6:]):
            # Handle UTC datetime string
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.astimezone(IST_TIMEZONE)
        
        # Try simple date format (YYYY-MM-DD)
        elif len(date_str) == 10 and date_str.count('-') == 2:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            # Set to midnight in IST timezone
            return IST_TIMEZONE.localize(
                date.replace(hour=0, minute=0, second=0, microsecond=0)
            )
        
        # Try datetime without timezone (YYYY-MM-DD HH:MM:SS)
        else:
            # Assume IST timezone if no timezone info
            date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            return IST_TIMEZONE.localize(date)
            
    except (ValueError, TypeError):
        return None

def format_ist_datetime(dt: datetime) -> str:
    """
    Format a datetime object in IST timezone.
    
    Args:
        dt: datetime object
        
    Returns:
        Formatted datetime string in IST timezone
    """
    if not dt:
        return ""
        
    # Convert to IST if not already in IST
    if dt.tzinfo != IST_TIMEZONE:
        dt = dt.astimezone(IST_TIMEZONE)
    
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def parse_google_calendar_datetime(date_dict: dict) -> datetime:
    """
    Parse Google Calendar datetime format to IST datetime.
    
    Args:
        date_dict: Google Calendar date/datetime dict
        
    Returns:
        datetime object in IST timezone
    """
    try:
        if 'dateTime' in date_dict:
            # Parse datetime with timezone
            dt_str = date_dict['dateTime']
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            return dt.astimezone(IST_TIMEZONE)
        elif 'date' in date_dict:
            # Parse date only (all-day event)
            date_str = date_dict['date']
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return IST_TIMEZONE.localize(dt)
        else:
            # Fallback to current time
            return datetime.now(IST_TIMEZONE)
    except Exception:
        return datetime.now(IST_TIMEZONE)

def ist_datetime_to_utc_iso(dt: datetime) -> str:
    """
    Convert IST datetime to UTC ISO format for Google Calendar API.
    
    Args:
        dt: datetime object in IST timezone
        
    Returns:
        UTC ISO format string
    """
    try:
        # Handle timezone-aware vs naive datetime properly
        if dt.tzinfo is None:
            # Naive datetime - localize to IST
            dt = IST_TIMEZONE.localize(dt)
        elif dt.tzinfo != IST_TIMEZONE:
            # Different timezone - convert to IST first
            dt = dt.astimezone(IST_TIMEZONE)
        # else: Already in IST timezone, use as-is
        
        # Convert to UTC
        utc_dt = dt.astimezone(pytz.UTC)
        
        # Return ISO format
        return utc_dt.isoformat()
    except Exception:
        return datetime.now(pytz.UTC).isoformat()