"""
Minimal Calendar Link Agent - LLM-Driven React Implementation

Simple React agent where LLM makes ALL decisions:
- Date range determination
- Event filtering and parsing  
- Response formatting
- No hardcoded logic
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from beetu_v2.agents.linkagent.constants import LINK_AGENT_SETTINGS, IST_TIMEZONE
from beetu_v2.agents.linkagent.tools import get_calendar_events, get_current_datetime

logger = logging.getLogger(__name__)


def create_calendar_react_agent():
    """Create minimal React agent where LLM makes all decisions."""
    
    # Current date info for LLM context
    today = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
    
    system_prompt = f"""You are a calendar assistant. Today is {today} (IST timezone).

Your task:
1. Understand what calendar events the user wants (upcoming, past, specific dates, etc.)
2. Determine the appropriate date range (start_date and end_date in YYYY-MM-DD format)
3. Use get_calendar_events tool to fetch the data
4. Parse and format a helpful response based on the user's specific request

STRICT DATE RANGE LIMITS (IMPORTANT):
- For PAST events: Maximum 3 days back from today (from {(datetime.now(IST_TIMEZONE) - timedelta(days=3)).strftime("%Y-%m-%d")} to {today})
- For FUTURE events: Maximum 3 days forward from today (from {today} to {(datetime.now(IST_TIMEZONE) + timedelta(days=3)).strftime("%Y-%m-%d")})
- NEVER fetch data beyond these 3-day windows in either direction
- If user asks for dates outside this range, politely explain the 3-day limit

Date reasoning guidelines (within 3-day limits):
- "upcoming/future events" → start: today, end: max 3 days from today
- "past events" → start: max 3 days ago, end: today
- "today's events" → start: today, end: today
- "tomorrow" → start: tomorrow, end: tomorrow
- "yesterday" → start: yesterday, end: yesterday
- Always respect the 3-day boundary in both directions

IMPORTANT - Date Format Rules:
- ALWAYS use YYYY-MM-DD format for dates (e.g., "2025-09-18")
- You can also use UTC ISO format: YYYY-MM-DDTHH:MM:SSZ (e.g., "2025-09-18T00:00:00Z")
- Simple dates (YYYY-MM-DD) are automatically converted to IST timezone
- ISO format dates are converted from UTC to IST internally
- This standardized format prevents parsing errors

Response guidelines:
- Focus on what the user specifically asked for
- Include relevant links (Zoom, Meet, YouTube, docs) when available
- Mention event details like time, location, attendees if relevant
- Be concise but helpful
- If user asks for dates outside 3-day window, politely explain: "I can only access events within 3 days (past or future) from today for performance reasons."

You have access to:
- get_calendar_events(start_date, end_date): Fetches events with 3-day limit validation
- get_current_datetime(): Get current date/time in standardized formats for calculations
"""

    # Create LLM
    llm = init_chat_model(
        model=LINK_AGENT_SETTINGS.MODEL,
        api_key=LINK_AGENT_SETTINGS.API_KEY,
        temperature=0.1  # Lower temperature for more consistent date parsing
    )
    
    # Create React agent with calendar tools
    agent = create_react_agent(
        model=llm,
        tools=[get_calendar_events, get_current_datetime],
        prompt=system_prompt
    )
    
    return agent

# Main function called by supervisor
def link_agent_react(
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    user_details: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Process calendar query using minimal React agent.
    LLM makes all decisions about dates, filtering, and formatting.
    
    Args:
        query: User's calendar question
        conversation_history: Optional conversation context
        user_details: Optional user context
        
    Returns:
        Formatted response string
    """
    try:
        # Create the React agent
        agent = create_calendar_react_agent()
        
        # Build context message if needed
        context_parts = [query]
        if user_details:
            context_parts.insert(0, f"User context: {user_details}")
        if conversation_history:
            context_parts.insert(-1, f"Recent conversation: {conversation_history[-2:]}")
        
        full_query = " | ".join(context_parts)
        
        # Run the agent - it will decide everything
        result = agent.invoke({"messages": [{"role": "user", "content": full_query}]})
        
        # Extract response from the result
        if "messages" in result and result["messages"]:
            return result["messages"][-1].content
        else:
            return "I couldn't process your calendar request. Please try again."
            
    except Exception as e:
        logger.error(f"Calendar agent error: {str(e)}")
        return f"I encountered an error while checking your calendar: {str(e)}"
