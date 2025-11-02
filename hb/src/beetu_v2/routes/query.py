from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta

from beetu_v2.supervisor.workflow import SupervisorSystem
from beetu_v2.routes.dto import QueryRequest, QueryResponse

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Simple rate limiting (in production, use Redis or proper rate limiter)
rate_limit_store = defaultdict(list)
RATE_LIMIT_REQUESTS = 30  # requests per window
RATE_LIMIT_WINDOW = 60   # seconds

def check_rate_limit(client_ip: str) -> bool:
    """Simple rate limiting - 30 requests per minute"""
    now = datetime.now()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    
    # Clean old entries
    rate_limit_store[client_ip] = [
        timestamp for timestamp in rate_limit_store[client_ip] 
        if timestamp > window_start
    ]
    
    # Check if limit exceeded
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    rate_limit_store[client_ip].append(now)
    return True


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process user query through supervisor system for intelligent agent routing.
    
    Flow:
    1. Take user query as input
    2. Route through supervisor to appropriate agent (yoga, math, general knowledge)
    3. Return structured response from supervisor with timing metrics
    """
    # Rate limiting (in production, get real client IP)
    client_ip = "default"  # In production: request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Maximum 30 requests per minute."
        )
    
    route_start_time = time.time()
    
    try:
        # Initialize supervisor system
        supervisor = SupervisorSystem()
        
        # Process query through supervisor for intelligent routing (now async)
        supervisor_result = await supervisor.process_query(request.query)
        
        # Calculate total route processing time
        total_processing_time = time.time() - route_start_time
        
        # Check if supervisor processing was successful
        if not supervisor_result["success"]:
            logger.error("Supervisor processing failed: %s", supervisor_result.get("error", "Unknown error"))
            return QueryResponse(
                success=False,
                response="I'm sorry, I encountered an error while processing your question.",
                sources=[],
                metadata={
                    "supervisor_used": True,
                    "supervisor_processing_time": supervisor_result["processing_time"],
                    "total_route_processing_time": total_processing_time,
                    "query_count": supervisor_result["query_count"]
                },
                error_message=supervisor_result.get("error", supervisor_result["response"])
            )
        
        # Return successful response with supervisor output and detailed timing
        return QueryResponse(
            success=True,
            response=supervisor_result["response"],
            sources=[],  # Supervisor handles routing, sources not directly available
            metadata={
                "supervisor_used": True,
                "supervisor_processing_time": supervisor_result["processing_time"],
                "total_route_processing_time": total_processing_time,
                "route_overhead_time": total_processing_time - supervisor_result["processing_time"],
                "query_count": supervisor_result["query_count"],
                "model": supervisor.model_name,
                "max_iterations": supervisor.max_iterations
            }
        )
        
    except Exception as e:
        total_processing_time = time.time() - route_start_time
        logger.error("Supervisor query processing failed in %.3f seconds: %s", total_processing_time, str(e))
        return QueryResponse(
            success=False,
            response="I'm sorry, I encountered an error while processing your question.",
            sources=[],
            metadata={
                "supervisor_used": True,
                "total_route_processing_time": total_processing_time,
                "error_occurred_at": "route_level"
            },
            error_message=str(e)
        )


@router.get("/health")
def health_check():
    """Health check endpoint for the query service with dependency validation."""
    health_status = {"status": "healthy", "service": "query", "components": {}}
    
    # Test supervisor system
    try:
        supervisor = SupervisorSystem()
        health_status["components"]["supervisor_system"] = "ready"
    except Exception as e:
        health_status["components"]["supervisor_system"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Test Pinecone connection
    try:
        from beetu_v2.db.pinecone import check_pinecone_connection
        if check_pinecone_connection():
            health_status["components"]["pinecone"] = "ready"
        else:
            health_status["components"]["pinecone"] = "error: connection failed"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["pinecone"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Test API key availability (without exposing values)
    try:
        from beetu_v2.config import settings
        health_status["components"]["api_keys"] = {
            "openai": "configured" if settings.OPENAI_API_KEY else "missing",
            "pinecone": "configured" if settings.PINECONE_API_KEY else "missing"
        }
    except Exception as e:
        health_status["components"]["api_keys"] = f"error: {str(e)}"
    
    health_status["supported_features"] = {
        "intelligent_routing": True,
        "multi_agent_system": True,
        "yoga_wellness_queries": True,
        "mathematical_calculations": True,
        "general_knowledge_queries": True,
        "rate_limiting": True
    }
    
    return health_status