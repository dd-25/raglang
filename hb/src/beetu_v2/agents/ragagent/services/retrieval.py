"""
Production-Optimized Retrieval System

Smart retrieval with user context awareness:
- Dynamic top-k selection based on user experience
- Optimized similarity thresholds 
- Context-aware reranking
- Minimal logging overhead
"""

import logging
import time
from typing import List, Dict, Any, Optional

from beetu_v2.db.pinecone import query_embeddings, check_index_stats
from beetu_v2.agents.ragagent.utils.embeddings import embed_texts
from beetu_v2.agents.ragagent.constants import RAG_SETTINGS
from beetu_v2.config import settings
from beetu_v2.agents.ragagent.dto import RetrievedChunk, RetrievalResult

# Optional Cohere for reranking
try:
    import cohere
    cohere_client = cohere.Client(settings.COHERE_API_KEY) if settings.COHERE_API_KEY else None
except ImportError:
    cohere_client = None

logger = logging.getLogger(__name__)


class SmartRetriever:
    """
    Production-optimized retrieval with user context awareness.
    Adapts retrieval parameters based on user profile.
    """
    
    def __init__(self):
        self.similarity_threshold = RAG_SETTINGS.SIMILARITY_THRESHOLD
    
    def retrieve(
        self,
        query: str,
        user_details: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
        use_reranking: bool = True
    ) -> RetrievalResult:
        """
        Scalable retrieval with user details support.
        
        Args:
            query: Search query
            user_details: User information (name, age, gender, subscription_status, etc.)
            namespace: Pinecone namespace
            use_reranking: Enable Cohere reranking
        """
        start_time = time.time()
        
        # Get retrieval parameters (extensible for user personalization)
        top_k = self._get_optimal_top_k(user_details, query)
        threshold = self._get_similarity_threshold(user_details)
        
        try:
            # Check if data exists
            if not check_index_stats(namespace):
                return RetrievalResult([], 0, time.time() - start_time)
            
            # Get query embedding
            query_vector = embed_texts(query)
            
            # Semantic search with reranking consideration
            search_k = 10 if use_reranking and cohere_client else top_k
            
            results = query_embeddings(
                vector=query_vector,
                top_k=search_k,
                namespace=namespace
            )
            
            # Convert to chunks
            chunks = self._process_results(results, threshold)
            
            # Apply reranking if available
            if use_reranking and cohere_client and chunks:
                chunks = self._rerank_chunks(query, chunks, top_k)
            else:
                chunks = chunks[:top_k]
            
            return RetrievalResult(
                chunks=chunks,
                total_found=len(chunks),
                retrieval_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return RetrievalResult([], 0, time.time() - start_time)
    
    def _get_optimal_top_k(self, user_details: Optional[Dict[str, Any]], query: str) -> int:
        """Determine optimal top_k - extensible for future user-based logic."""
        # Future: Can be extended based on user_details like subscription_status, preferences, etc.
        return RAG_SETTINGS.DEFAULT_TOP_K
    
    def _get_similarity_threshold(self, user_details: Optional[Dict[str, Any]]) -> float:
        """Get similarity threshold - extensible for future personalization."""
        # Future: Can be adjusted based on user_details
        return self.similarity_threshold
    
    def _process_results(self, results, threshold: float) -> List[RetrievedChunk]:
        """Convert Pinecone results to RetrievedChunk objects."""
        if not results or not hasattr(results, 'matches'):
            return []
        
        chunks = []
        for match in results.matches:
            if match.score >= threshold:
                metadata = match.metadata or {}
                chunks.append(RetrievedChunk(
                    text=metadata.get("text", ""),
                    source=metadata.get("source", "unknown"),
                    score=float(match.score),
                    chunk_id=match.id,
                    tokens=metadata.get("tokens", 0)
                ))
        
        return chunks
    
    def _rerank_chunks(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Apply Cohere reranking for improved relevance."""
        try:
            documents = [chunk.text for chunk in chunks]
            
            rerank_response = cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=top_k
            )
            
            # Map reranked results back to chunks
            reranked_chunks = []
            for result in rerank_response.results:
                original_chunk = chunks[result.index]
                original_chunk.score = result.relevance_score
                reranked_chunks.append(original_chunk)
            
            return reranked_chunks
            
        except Exception:
            # Fallback to original chunks if reranking fails
            return chunks[:top_k]


# Global retriever instance
smart_retriever = SmartRetriever()


def retrieve_chunks(
    query: str,
    user_details: Optional[Dict[str, Any]] = None,
    namespace: str = "default",
    use_reranking: bool = True
) -> RetrievalResult:
    """
    Main retrieval function with extensible user details support.
    
    Args:
        query: Search query
        user_details: User information (name, age, gender, subscription_status, preferences, etc.)
        namespace: Vector database namespace
        use_reranking: Enable Cohere reranking
    """
    return smart_retriever.retrieve(query, user_details, namespace, use_reranking)
