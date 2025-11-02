"""
Production-optimized RAG Agent Configuration
"""
from beetu_v2.config import settings


class RAG_SETTINGS:
    """Scalable RAG agent settings for production use"""
    
    # LLM Configuration
    API_KEY: str = settings.RAG_LLM_API_KEY
    PROVIDER: str = settings.RAG_LLM_PROVIDER or "openai"
    MODEL: str = settings.RAG_LLM_MODEL
    TEMPERATURE: float = 0.1
    
    # Retrieval Configuration  
    DEFAULT_TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.3
    
    # Context Management
    MAX_CONTEXT_TOKENS: int = 3000
    MAX_CONVERSATION_HISTORY: int = 6
