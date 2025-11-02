"""
Minimal Calendar Link Agent Constants
"""

from dataclasses import dataclass
import os
import pytz


@dataclass
class LinkAgentSettings:
    """LLM configuration for calendar agent"""
    MODEL: str = os.getenv("LINK_AGENT_MODEL", "gpt-4o-mini")
    API_KEY: str = os.getenv("OPENAI_API_KEY", "")


class CALENDAR_SETTINGS:
    """Calendar settings"""
    DEFAULT_CALENDAR_SERVICE = "google"
    PRIMARY_CALENDAR_ID = "primary"


class GOOGLE_CALENDAR_SETTINGS:
    """Google Calendar settings"""
    SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
    API_VERSION = "v3"
    CREDENTIALS_FILE = "google_calendar_credentials.json"
    TOKEN_FILE = "google_calendar_token.json"


# LLM handles all link extraction and categorization - no regex patterns needed!


# IST Timezone
IST_TIMEZONE = pytz.timezone('Asia/Kolkata')

# Global settings
LINK_AGENT_SETTINGS = LinkAgentSettings()
