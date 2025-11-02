"""
Supervisor Data Transfer Objects (DTOs)

This module contains all data models used by the supervisor system:
- Configuration settings
- Workflow states and results
- Tool execution results
- Agent routing information
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import os


@dataclass
class SupervisorSettings:
    """Configuration settings for the supervisor system"""
    MODEL: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    TIMEOUT: float = float(os.getenv("TIMEOUT", "30.0"))


@dataclass
class SupervisorResult:
    """Result from supervisor query processing."""
    success: bool
    response: str
    processing_time: float
    query_count: int
    agent_used: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentToolResult:
    """Result from individual agent tool execution."""
    tool_name: str
    result: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class WorkflowState:
    """State object for supervisor workflow."""
    query: str
    response: Optional[str] = None
    agent_used: Optional[str] = None
    query_count: int = 0
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentCapabilities:
    """Agent capabilities and specializations."""
    name: str
    description: str
    specializations: List[str]
    supported_queries: List[str]
    fallback_available: bool = True
