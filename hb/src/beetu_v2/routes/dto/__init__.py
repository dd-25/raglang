"""
Routes Data Transfer Objects (DTOs)

This module contains all Pydantic models used by API routes:
- Request and response models for upload endpoint
- Request and response models for query endpoint
- Health check response models
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class UploadResponse(BaseModel):
    """Response model for file upload endpoints."""
    status: str
    message: str
    filename: str
    namespace: str
    chunks_generated: int
    processing_stats: Dict[str, Any]
    error_details: Optional[str] = None


class UploadErrorResponse(BaseModel):
    """Error response model for upload failures."""
    status: str
    error: str
    filename: str
    namespace: str
    details: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    top_k: Optional[int] = 5
    namespace: Optional[str] = "default"
    use_reranking: Optional[bool] = True
    conversation_history: Optional[List[str]] = None
    user_details: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    success: bool
    response: str
    sources: List[str] = []
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    service: Optional[str] = None
    components: Optional[Dict[str, str]] = None
    supported_features: Optional[Dict[str, bool]] = None
    supported_formats: Optional[List[str]] = None
    max_file_size_mb: Optional[int] = None
    memory_threshold_mb: Optional[int] = None


class FormatInfo(BaseModel):
    """File format information model."""
    description: str
    extensions: List[str]
    notes: str


class SupportedFormatsResponse(BaseModel):
    """Supported formats response model."""
    supported_formats: Dict[str, FormatInfo]
    limits: Dict[str, Any]
