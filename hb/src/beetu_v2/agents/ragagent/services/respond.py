"""
Production-Optimized Response Generation

Context-aware response generation with:
- User profile integration
- Conversation history management  
- Optimized prompt engineering
- Minimal overhead processing
"""

import logging
import time
from typing import List, Optional, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model

from beetu_v2.agents.ragagent.services.retrieval import retrieve_chunks
from beetu_v2.agents.ragagent.constants import RAG_SETTINGS
from beetu_v2.agents.ragagent.dto import RetrievedChunk, ResponseResult

logger = logging.getLogger(__name__)


class ContextAwareGenerator:
    """
    Production-optimized response generator with user context awareness.
    Adapts responses based on user profile and conversation history.
    """
    
    def __init__(self):
        self.model = init_chat_model(
            model=RAG_SETTINGS.MODEL,
            model_provider=RAG_SETTINGS.PROVIDER,
            temperature=RAG_SETTINGS.TEMPERATURE,
            api_key=RAG_SETTINGS.API_KEY
        )
    
    def generate_response(
        self,
        query: str,
        user_details: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[str]] = None,
        namespace: str = "default"
    ) -> ResponseResult:
        """
        Generate response with scalable user details and conversation context.
        
        Args:
            query: User's question
            user_details: User information (name, age, gender, subscription_status, etc.)
            conversation_history: Recent conversation messages
            namespace: Vector database namespace
        """
        start_time = time.time()
        
        try:
            # Retrieve relevant chunks
            retrieval_result = retrieve_chunks(
                query=query,
                user_details=user_details,
                namespace=namespace,
                use_reranking=True
            )
            
            if not retrieval_result.chunks:
                return self._create_fallback_response(query, start_time)
            
            # Generate response with context
            response_content = self._generate_with_context(
                query, retrieval_result.chunks, user_details, conversation_history
            )
            
            return ResponseResult(
                success=True,
                response=response_content,
                sources=retrieval_result.get_sources(),
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ResponseResult(
                success=False,
                response="I encountered an error while processing your question.",
                sources=[],
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _generate_with_context(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        user_details: Optional[Dict[str, Any]],
        conversation_history: Optional[List[str]]
    ) -> str:
        """Generate response using retrieved context and user details."""
        
        # Build system prompt with context and user info
        system_prompt = self._build_system_prompt(chunks, user_details)
        
                # Build user message with strict instruction
        user_message = f"""Question: {query}

IMPORTANT: Base your answer ONLY on the context information provided above. Do not add any information from general knowledge. If the context doesn't contain enough information to answer the question, clearly state that you don't have that information in the knowledge base."""
        
        # Add conversation history if provided  
        messages = [SystemMessage(content=system_prompt)]
        if conversation_history:
            for msg in conversation_history[-3:]:  # Last 3 messages for context
                messages.append(HumanMessage(content=msg))
        messages.append(HumanMessage(content=user_message))
        
        # Generate response
        response = self.model.invoke(messages)
        return getattr(response, "content", str(response))
    
    def _build_system_prompt(
        self,
        chunks: List[RetrievedChunk],
        user_details: Optional[Dict[str, Any]]
    ) -> str:
        """Build system prompt with context and user details."""
        
        # Format context from chunks
        context_text = "\n\n".join([
            f"Source: {chunk.source}\n{chunk.text}" 
            for chunk in chunks
        ])
        
        # Build user details string if provided
        user_info = ""
        if user_details:
            details = []
            for key, value in user_details.items():
                if value:
                    details.append(f"{key}: {value}")
            if details:
                user_info = f"\n\nUser Information: {', '.join(details)}"
        
        # Build complete system prompt
        prompt = f"""You are a knowledge base assistant that ONLY provides information from the provided context documents. You do NOT use any general knowledge or information outside of what is provided.

CONTEXT INFORMATION:
{context_text}
{user_info}

STRICT RESPONSE RULES:
- ONLY use information explicitly provided in the context above
- Do NOT add any information from your general training or knowledge
- Do NOT make assumptions or provide general advice not found in the context
- If the context doesn't contain sufficient information, clearly state this
- Always cite the specific source documents when providing information
- Be direct and factual based solely on the provided context
- If asked about something not covered in the context, say you don't have that information in your knowledge base

EXAMPLE RESPONSES:
- Good: "Based on the provided documents, [specific information from context with source]"
- Bad: "Generally speaking, yoga is beneficial..." (using general knowledge)
- Good: "I don't have information about that specific topic in my knowledge base"
- Bad: "While I don't have specific details, typically..." (providing general knowledge)"""
        
        return prompt
    
    def _create_fallback_response(self, query: str, start_time: float) -> ResponseResult:
        """Create fallback response when no context is found."""
        
        fallback_text = (
            "I don't have information about this topic in my knowledge base. "
            "I can only provide answers based on the documents that have been uploaded and indexed. "
            "Please try a different question that might be covered in the available health and wellness content, "
            "or ensure that relevant documents have been added to the knowledge base."
        )
        
        return ResponseResult(
            success=False,  # Changed to False to indicate no knowledge base match
            response=fallback_text,
            sources=[],
            processing_time=time.time() - start_time
        )
    
    def create_fallback_response(self, query: str, start_time: float) -> ResponseResult:
        """Public method to create fallback response."""
        return self._create_fallback_response(query, start_time)
    
    def generate_with_context(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        user_details: Optional[Dict[str, Any]],
        conversation_history: Optional[List[str]]
    ) -> str:
        """Public method to generate response with context."""
        return self._generate_with_context(query, chunks, user_details, conversation_history)


# Global generator instance  
context_generator = ContextAwareGenerator()


def generate_response_from_chunks(
    query: str,
    chunks: List[RetrievedChunk],
    user_details: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[str]] = None
) -> ResponseResult:
    """
    Generate response from pre-retrieved chunks.
    Production-ready function with user details support.
    """
    start_time = time.time()
    
    try:
        if not chunks:
            return context_generator.create_fallback_response(query, start_time)
        
        response_content = context_generator.generate_with_context(
            query, chunks, user_details, conversation_history
        )
        
        sources = list(set(chunk.source for chunk in chunks))
        
        return ResponseResult(
            success=True,
            response=response_content,
            sources=sources,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        return ResponseResult(
            success=False,
            response="I encountered an error while generating a response.",
            sources=[],
            processing_time=time.time() - start_time,
            error_message=str(e)
        )


def process_query_end_to_end(
    query: str,
    user_details: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[str]] = None,
    namespace: str = "default",
    **kwargs  # For backward compatibility
) -> ResponseResult:
    """
    Complete RAG pipeline: Query → Retrieve → Generate Response
    Production-ready with extensible user details support.
    """
    _ = kwargs  # Mark as used for linter (backward compatibility)
    return context_generator.generate_response(
        query=query,
        user_details=user_details,
        conversation_history=conversation_history,
        namespace=namespace
    )
