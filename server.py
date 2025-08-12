"""
FastAPI server for the LangGraph RAG application.
Provides REST API endpoints for file upload and query processing.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
from pathlib import Path
from src.modules.graphBuilder import BuildGraph
from src.modules.dataIngestion import DataIngestion

project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

app = FastAPI(
    title="LangGraph RAG API",
    description="RAG application with file upload and query capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    result: str
    query: str
    success: bool

class UploadResponse(BaseModel):
    filename: str
    message: str
    success: bool

# Global instances
graph = None
data_ingestion = None

@app.on_event("startup")
async def startup_event():
    """Initialize the graph and data ingestion on startup."""
    global graph, data_ingestion
    try:
        # Initialize BuildGraph and DataIngestion
        builder = BuildGraph()
        graph = builder.buildRAG()
        data_ingestion = DataIngestion()
        print("✅ LangGraph RAG system and Data Ingestion initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "LangGraph RAG API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Process uploaded file directly from memory without storing on server.
    
    Args:
        file: The uploaded file
        
    Returns:
        UploadResponse with success status and message
    """
    try:
        if not data_ingestion:
            raise HTTPException(status_code=503, detail="Data ingestion system not initialized")
        
        # Validate file type
        allowed_extensions = {'.txt', '.pdf', '.docx', '.md'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}"
            )
        
        # Read file content directly into memory
        file_content = await file.read()
        print("File Read successfully")
        
        if not file_content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        try:
            # Process the document directly from memory
            chunks = data_ingestion.uploadDoc(file_content, file.filename)
            
            return UploadResponse(
                filename=file.filename,
                message=f"File '{file.filename}' processed successfully from memory. Created {len(chunks)} chunks.",
                success=True
            )
            
        except Exception as processing_error:
            return UploadResponse(
                filename=file.filename,
                message=f"File '{file.filename}' processing failed: {str(processing_error)}",
                success=False
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query using the RAG system.
    
    Args:
        request: QueryRequest containing the user's query
        
    Returns:
        QueryResponse with the RAG system's response
    """
    try:
        if not graph:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process the query through BuildGraph.buildRAG().invoke()
        input_state = {"query": request.query}
        final_state = graph.invoke(input_state)
        
        result = final_state.get('result', 'No response generated')
        
        return QueryResponse(
            result=result,
            query=request.query,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "graph_initialized": graph is not None,
        "data_ingestion_initialized": data_ingestion is not None,
        "storage_mode": "memory_only",  # Indicates files are processed in memory
        "timestamp": Path(__file__).stat().st_mtime
    }

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )