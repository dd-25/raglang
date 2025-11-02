from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging
import tempfile
from beetu_v2.agents.ragagent.services.data_ingestion import ingest_file_to_vectordb
from beetu_v2.constants import UPLOAD_SETTINGS
from beetu_v2.routes.dto import UploadResponse, UploadErrorResponse

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    namespace: str = Query(default="default", description="Namespace for document storage")
):
    """
    Upload and process a document through the simplified data ingestion pipeline.
    
    Supports file formats: PDF, DOCX, TXT, JSON, CSV, TSV
    Core functionality: parse → extract → chunk → embed → upload to Pinecone
    
    Args:
        file: The uploaded file
        namespace: Pinecone namespace for storage (default: "default")
        
    Returns:
        Processing results with statistics and chunk information
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )
    
    MAX_FILE_SIZE = UPLOAD_SETTINGS.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
    
    try:
        # Use SpooledTemporaryFile for memory-efficient file handling
        with tempfile.SpooledTemporaryFile(
            max_size=UPLOAD_SETTINGS.DISK_THRESHOLD_MB * 1024 * 1024,  # 10MB threshold for memory vs disk
            mode='w+b'
        ) as spooled_file:
            
            # Stream file content into SpooledTemporaryFile
            file_size = 0
            chunk_size = UPLOAD_SETTINGS.STREAMING_CHUNK_SIZE_KB * 1024  # Convert KB to bytes
            
            # Reset file pointer to beginning
            await file.seek(0)
            
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                    
                file_size += len(chunk)
                
                # Check file size during streaming to prevent memory issues
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size ({file_size} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)"
                    )
                
                spooled_file.write(chunk)
            
            # Validate file size
            if file_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Empty file is not allowed"
                )
            
            # Validate file extension
            file_extension = '.' + file.filename.split('.')[-1].lower()
            
            if file_extension not in UPLOAD_SETTINGS.ALLOWED_FILE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format. Allowed formats: {', '.join(UPLOAD_SETTINGS.ALLOWED_FILE_TYPES)}"
                )
            
            # Reset spooled file pointer to beginning for reading
            spooled_file.seek(0)
            
            # Read file content efficiently
            file_bytes = spooled_file.read()
            
            # Process through simplified RAG ingestion pipeline
            result = ingest_file_to_vectordb(file_bytes, file.filename, namespace)
            
            if result.success:
                # Successful processing
                file_extension = file.filename.split('.')[-1].lower()
                response_data = UploadResponse(
                    status="success",
                    message=f"File '{file.filename}' processed successfully",
                    filename=file.filename,
                    namespace=namespace,
                    chunks_generated=result.chunks_created,
                    processing_stats={
                        "file_type": file_extension,
                        "processing_time": result.processing_time,
                        "chunks_created": result.chunks_created,
                        "file_size_bytes": file_size
                    }
                )
                return JSONResponse(
                    status_code=200,
                    content=response_data.dict()
                )
            
            else:
                # Processing failed
                error_message = result.error_message or 'Unknown processing error'
                logger.error("Processing failed for %s: %s", file.filename, error_message)
                
                error_response = UploadErrorResponse(
                    status="error",
                    error=error_message,
                    filename=file.filename,
                    namespace=namespace,
                    details=f"Processing time: {result.processing_time:.2f}s"
                )
                
                return JSONResponse(
                    status_code=422,
                    content=error_response.dict()
                )
        
    except (ValueError, TypeError, RuntimeError) as e:
        # Handle expected errors
        logger.exception("Error during file upload for %s:", file.filename)
        
        error_response = UploadErrorResponse(
            status="error",
            error="Processing error during file upload",
            filename=file.filename or "unknown",
            namespace=namespace,
            details=str(e)
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.get("/health")
async def upload_health():
    """Health check endpoint for upload service."""
    return {
        "status": "healthy",
        "service": "file_upload",
        "supported_formats": list(UPLOAD_SETTINGS.ALLOWED_FILE_TYPES),
        "max_file_size_mb": UPLOAD_SETTINGS.MAX_FILE_SIZE_MB,
        "memory_threshold_mb": UPLOAD_SETTINGS.DISK_THRESHOLD_MB
    }


@router.get("/formats")
async def supported_formats():
    """Get information about supported file formats."""
    return {
        "supported_formats": {
            "pdf": {
                "description": "Portable Document Format",
                "extensions": [".pdf"],
                "notes": "Supports text extraction from PDF documents"
            },
            "docx": {
                "description": "Microsoft Word Document",
                "extensions": [".docx"],
                "notes": "Supports modern Word document format"
            },
            "text": {
                "description": "Plain Text",
                "extensions": [".txt"],
                "notes": "UTF-8 encoded text files"
            },
            "json": {
                "description": "JavaScript Object Notation",
                "extensions": [".json"],
                "notes": "Structured data with intelligent key-value extraction"
            },
            "csv": {
                "description": "Comma-Separated Values",
                "extensions": [".csv", ".tsv"],
                "notes": "Tabular data converted to readable format"
            }
        },
        "limits": {
            "max_file_size_mb": UPLOAD_SETTINGS.MAX_FILE_SIZE_MB,
            "memory_threshold_mb": UPLOAD_SETTINGS.DISK_THRESHOLD_MB,
            "supported_encodings": ["UTF-8"],
            "streaming_chunk_size_kb": UPLOAD_SETTINGS.STREAMING_CHUNK_SIZE_KB,
        }
    }