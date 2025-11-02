"""
Production-Optimized RAG Agent - Complete Integration Module

This module provides the main interface for the optimized RAG agent system:
- Intelligent document processing
- Context-aware retrieval 
- User-personalized response generation
- Minimal overhead operations
"""

from typing import Dict, Any, Optional, List
from beetu_v2.agents.ragagent.services.data_ingestion import ingest_file_to_vectordb
from beetu_v2.agents.ragagent.services.retrieval import retrieve_chunks
from beetu_v2.agents.ragagent.services.respond import process_query_end_to_end
from beetu_v2.agents.ragagent.dto import IngestionResult, RetrievalResult, ResponseResult


class OptimizedRAGAgent:
    """
    Production-ready RAG agent with user context awareness.
    
    Features:
    - Smart document ingestion with minimal overhead
    - Context-aware retrieval with dynamic parameters
    - Personalized response generation
    - Optimized for production performance
    """
    
    def __init__(self):
        """Initialize optimized RAG agent."""
        pass
    
    def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        namespace: str = "default"
    ) -> IngestionResult:
        """
        Ingest document into vector database.
        
        Args:
            file_bytes: Document content as bytes
            filename: Original filename with extension
            namespace: Vector database namespace
            
        Returns:
            IngestionResult with processing status and metrics
        """
        return ingest_file_to_vectordb(file_bytes, filename, namespace)
    
    def retrieve_context(
        self,
        query: str,
        user_details: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
        use_reranking: bool = True
    ) -> RetrievalResult:
        """
        Retrieve relevant context for query with user details.
        
        Args:
            query: Search query
            user_details: User details for personalization
            namespace: Vector database namespace
            use_reranking: Enable Cohere reranking
            
        Returns:
            RetrievalResult with relevant chunks and metadata
        """
        return retrieve_chunks(query, user_details, namespace, use_reranking)
    
    def generate_response(
        self,
        query: str,
        user_details: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[str]] = None,
        namespace: str = "default"
    ) -> ResponseResult:
        """
        Generate complete response with retrieval and user details.
        
        Args:
            query: User's question
            user_details: User details information
            conversation_history: Recent conversation messages
            namespace: Vector database namespace
            
        Returns:
            ResponseResult with generated response and metadata
        """
        return process_query_end_to_end(
            query=query,
            user_details=user_details,
            conversation_history=conversation_history,
            namespace=namespace
        )
    
    def process_query(
        self,
        query: str,
        user_details: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[str]] = None,
        namespace: str = "default"
    ) -> ResponseResult:
        """
        Main query processing method for backward compatibility.
        
        This method maintains the interface expected by the supervisor system
        while using the optimized implementation.
        """
        return self.generate_response(
            query=query,
            user_details=user_details,
            conversation_history=conversation_history,
            namespace=namespace
        )


# Global optimized RAG agent instance
optimized_rag_agent = OptimizedRAGAgent()

# Alternative name for easier imports
rag_agent_instance = optimized_rag_agent


# Main functions for external usage (backward compatibility)
def ingest_file_to_vectordb_optimized(file_bytes: bytes, filename: str, namespace: str = "default") -> IngestionResult:
    """Optimized document ingestion function."""
    return optimized_rag_agent.ingest_document(file_bytes, filename, namespace)


def retrieve_chunks_optimized(
    query: str,
    user_details: Optional[Dict[str, Any]] = None,
    namespace: str = "default",
    use_reranking: bool = True
) -> RetrievalResult:
    """Optimized retrieval function."""
    return optimized_rag_agent.retrieve_context(query, user_details, namespace, use_reranking)


def process_query_optimized(
    query: str,
    user_details: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[str]] = None,
    namespace: str = "default"
) -> ResponseResult:
    """Optimized end-to-end query processing."""
    return optimized_rag_agent.process_query(query, user_details, conversation_history, namespace)
