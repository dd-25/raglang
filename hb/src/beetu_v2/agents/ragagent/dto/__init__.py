"""
RAG Agent Data Transfer Objects (DTOs)

This module contains all data models used by the RAG agent components:
- Ingestion results and metadata
- Retrieval chunks and results  
- Response generation results
- User context structures
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from beetu_v2.agents.ragagent.constants import RAG_SETTINGS


@dataclass
class IngestionResult:
    """Ingestion result with essential information only."""
    success: bool
    chunks_created: int
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class RetrievedChunk:
    """Retrieved chunk with essential metadata."""
    text: str
    source: str
    score: float
    chunk_id: str
    tokens: int


@dataclass 
class RetrievalResult:
    """Retrieval result with chunks and metadata."""
    chunks: List[RetrievedChunk]
    total_found: int
    retrieval_time: float
    
    def get_context_text(self, max_tokens: int = RAG_SETTINGS.MAX_CONTEXT_TOKENS) -> str:
        """Get concatenated context text within token limit."""
        context_parts = []
        current_tokens = 0
        
        for chunk in self.chunks:
            if current_tokens + chunk.tokens <= max_tokens:
                context_parts.append(f"Source: {chunk.source}\n{chunk.text}")
                current_tokens += chunk.tokens
            else:
                break
        
        return "\n\n".join(context_parts)
    
    def get_sources(self) -> List[str]:
        """Get unique sources from chunks."""
        return list(set(chunk.source for chunk in self.chunks))


@dataclass
class ResponseResult:
    """Response generation result."""
    success: bool
    response: str
    sources: List[str]
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class UserDetails:
    """User details for personalized responses."""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    subscription_status: Optional[str] = None
    preferences: Optional[List[str]] = None
    experience_level: Optional[str] = None
    goals: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            k: v for k, v in {
                'name': self.name,
                'age': self.age,
                'gender': self.gender,
                'subscription_status': self.subscription_status,
                'preferences': self.preferences,
                'experience_level': self.experience_level,
                'goals': self.goals
            }.items() if v is not None
        }
    
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'UserDetails':
        """Create from dictionary for backward compatibility."""
        if not data:
            return cls()
        
        return cls(
            name=data.get('name'),
            age=data.get('age'),
            gender=data.get('gender'),
            subscription_status=data.get('subscription_status'),
            preferences=data.get('preferences'),
            experience_level=data.get('experience_level'),
            goals=data.get('goals')
        )
