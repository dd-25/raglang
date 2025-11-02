"""
Ultra-Minimal Calendar Data Models

LLM handles all link extraction and categorization from raw event data.
No complex link models needed!
"""

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class CalendarEvent(BaseModel):
    """Simple calendar event data - LLM processes everything else."""
    id: str
    title: str
    description: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    organizer: Optional[str] = None
    attendees: Optional[List[str]] = None
    is_recurring: Optional[bool] = False
    status: Optional[str] = "confirmed"
