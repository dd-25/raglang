"""
Unified RAG System - Complete document processing, retrieval, and response generation.

This module provides a comprehensive RAG (Retrieval-Augmented Generation) system that:
1. Ingests documents independently for vector database upload
2. Retrieves relevant chunks using semantic search and reranking
3. Generates intelligent responses using retrieved context

The system is designed to be used directly in FastAPI routes with clear separation of concerns.
"""

from __future__ import annotations
import logging
import time
import io
import json
import uuid
from typing import List, Dict, Any, Optional, Union, Generator
from dataclasses import dataclass
from enum import Enum

# OpenAI and LlamaIndex imports
from openai import OpenAI
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PDFReader, DocxReader

# Internal imports
from src.beetu_v2.db.pinecone import upsert_embeddings, query_embeddings, check_index_stats
from src.beetu_v2.config import settings
from src.beetu_v2.constants import (
    EMBEDDING_SETTINGS, 
    CHUNKING_SETTINGS, 
    UPLOAD_SETTINGS, 
    RETRIEVER_SETTINGS,
    PINECONE_SETTINGS
)
from src.beetu_v2.agents.ragagent.utils.token import count_tokens
from src.beetu_v2.agents.ragagent.utils.helper import split_sentences, split_json_chunk
from src.beetu_v2.agents.ragagent.utils.embeddings import embed_texts

# Optional imports
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import cohere
except ImportError:
    cohere = None

# Set up logging
logger = logging.getLogger(__name__)


# ================================
# DATA MODELS AND ENUMS
# ================================

class RetrievalStrategy(Enum):
    """Different retrieval strategies for various use cases."""
    SEMANTIC = "semantic"          # Pure semantic similarity
    HYBRID = "hybrid"             # Semantic + keyword matching
    MULTI_QUERY = "multi_query"   # Multiple query variations
    CONTEXTUAL = "contextual"     # Context-aware retrieval
    RERANK = "rerank"             # Retrieval with re-ranking


@dataclass
class IngestionResult:
    """Result of document ingestion process."""
    filename: str
    chunks_created: int
    total_tokens: int
    file_type: str
    success: bool
    processing_time: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "chunks_created": self.chunks_created,
            "total_tokens": self.total_tokens,
            "file_type": self.file_type,
            "success": self.success,
            "processing_time": self.processing_time,
            "error_message": self.error_message
        }


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata and scores."""
    text: str
    source: str
    score: float
    chunk_id: str
    file_type: str
    chunk_length: int
    namespace: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "file_type": self.file_type,
            "chunk_length": self.chunk_length,
            "namespace": self.namespace,
            "metadata": self.metadata
        }


@dataclass
class RetrievalResult:
    """Complete retrieval result with chunks and metadata."""
    chunks: List[RetrievedChunk]
    query: str
    strategy: RetrievalStrategy
    total_found: int
    retrieval_time: float
    context_length: int
    
    def get_context_text(self, max_tokens: Optional[int] = None) -> str:
        """Get concatenated context text from all chunks."""
        context_parts = []
        current_tokens = 0
        
        for chunk in self.chunks:
            chunk_tokens = chunk.chunk_length
            if max_tokens and current_tokens + chunk_tokens > max_tokens:
                break
            context_parts.append(f"Source: {chunk.source}\n{chunk.text}")
            current_tokens += chunk_tokens
            
        return "\n\n---\n\n".join(context_parts)
    
    def get_sources(self) -> List[str]:
        """Get unique sources from retrieved chunks."""
        return list(set(chunk.source for chunk in self.chunks))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "query": self.query,
            "strategy": self.strategy.value,
            "total_found": self.total_found,
            "retrieval_time": self.retrieval_time,
            "context_length": self.context_length
        }


@dataclass
class ResponseResult:
    """Structured result for response generation."""
    success: bool
    response: str
    query: str
    chunks_used: List[RetrievedChunk]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "response": self.response,
            "query": self.query,
            "chunks_used": [chunk.to_dict() for chunk in self.chunks_used],
            "metadata": self.metadata,
            "error_message": self.error_message
        }


# ================================
# DOCUMENT INGESTION SYSTEM
# ================================

class DocumentIngestionPipeline:
    """
    Independent document ingestion pipeline for uploading files to vector database.
    Designed to be used directly in upload routes.
    """
    
    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_SETTINGS.EMBED_MODEL,
        chunk_size: int = CHUNKING_SETTINGS.MAX_TOKENS,
        chunk_overlap: int = CHUNKING_SETTINGS.OVERLAP_TOKENS
    ):
        """Initialize the ingestion pipeline with configurable parameters."""
        self.embedding_model = OpenAIEmbedding(
            model=embedding_model_name,
            api_key=settings.OPENAI_API_KEY
        )
        
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[.!?]+",
            include_metadata=True,
            include_prev_next_rel=False
        )
        
        self.supported_extensions = {
            'pdf', 'docx', 'txt', 'json', 'csv', 'tsv'
        }
    
    def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        namespace: str = "default",
    ) -> IngestionResult:
        """
        Main ingestion method - processes file and uploads to vector database.
        Returns success response for route usage.
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting ingestion for file: {filename} ({len(file_bytes)} bytes)")
            
            # Extract text and create document
            document = self._extract_text_and_create_document(file_bytes, filename)
            logger.info(f"Document extracted: {len(document.text)} characters")
            
            # Chunk the document
            text_nodes = self._chunk_document(document)
            logger.info(f"Document chunked into {len(text_nodes)} nodes")
            
            # Convert TextNodes to strings and validate
            valid_chunks = self._validate_chunks(text_nodes)
            logger.info(f"Valid chunks for upload: {len(valid_chunks)}/{len(text_nodes)}")
            
            # Embed and upload to Pinecone
            self._embed_and_upload(valid_chunks, namespace=namespace, source=filename)
            
            total_tokens = sum(count_tokens(chunk) for chunk in valid_chunks)
            processing_time = time.time() - start_time
            
            logger.info(f"Successfully ingested {filename}: {len(valid_chunks)} chunks uploaded in {processing_time:.2f}s")
            
            return IngestionResult(
                filename=filename,
                chunks_created=len(valid_chunks),
                total_tokens=total_tokens,
                file_type=document.metadata.get("file_type", "unknown"),
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Processing failed for {filename}: {str(e)}")
            return IngestionResult(
                filename=filename,
                chunks_created=0,
                total_tokens=0,
                file_type="unknown",
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _extract_text_and_create_document(self, file_bytes: bytes, filename: str) -> Document:
        """Extract text from different file types using LlamaIndex readers."""
        extension = filename.split('.')[-1].lower()
        
        if extension not in self.supported_extensions:
            # Try to decode as text for unknown extensions
            try:
                combined_text = file_bytes.decode("utf-8")
                if combined_text.strip().startswith(('{', '[')):
                    extension = "json"
            except UnicodeDecodeError:
                raise ValueError(f"Unsupported file type: {extension}")
        else:
            combined_text = self._extract_by_type(file_bytes, filename, extension)
        
        # Validate that we extracted some content
        if not combined_text or not combined_text.strip():
            raise ValueError(f"No text content extracted from file: {filename}")
        
        # Create LlamaIndex Document
        document = Document(
            text=combined_text,
            metadata={
                "filename": filename, 
                "file_type": extension,
                "original_size": len(file_bytes),
                "extracted_length": len(combined_text)
            }
        )
        
        logger.info(f"Successfully processed {filename}: extracted {len(combined_text)} characters")
        return document
    
    def _extract_by_type(self, file_bytes: bytes, filename: str, extension: str) -> str:
        """Extract text based on file type."""
        if extension == "pdf":
            return self._extract_pdf(file_bytes, filename)
        elif extension == "docx":
            return self._extract_docx(file_bytes, filename)
        elif extension == "json":
            return self._extract_json(file_bytes, filename)
        elif extension == "txt":
            return file_bytes.decode("utf-8")
        elif extension in ["csv", "tsv"]:
            return self._extract_csv_tsv(file_bytes, filename, extension)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def _extract_pdf(self, file_bytes: bytes, filename: str) -> str:
        """Extract text from PDF files with fallback support."""
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = filename
        
        try:
            pdf_reader = PDFReader()
            documents = pdf_reader.load_data(file_obj)
            if not documents:
                raise ValueError(f"No content extracted from PDF: {filename}")
            
            combined_text = "\n\n".join([doc.text for doc in documents if doc.text.strip()])
            
            if not combined_text.strip():
                raise ValueError(f"PDF appears to be empty or contains only images: {filename}")
                
            return combined_text
            
        except Exception as pdf_error:
            logger.warning(f"LlamaIndex PDF reader failed for {filename}: {str(pdf_error)}")
            
            # Try fallback method using PyPDF2
            if PyPDF2:
                try:
                    file_obj.seek(0)
                    pdf_reader_fallback = PyPDF2.PdfReader(file_obj)
                    
                    text_parts = []
                    for page in pdf_reader_fallback.pages:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(page_text.strip())
                    
                    combined_text = "\n\n".join(text_parts)
                    
                    if not combined_text.strip():
                        raise ValueError(f"PDF appears to be empty or contains only images: {filename}")
                        
                    logger.info(f"Successfully extracted PDF using PyPDF2 fallback: {filename}")
                    return combined_text
                    
                except Exception:
                    logger.error(f"Both LlamaIndex and PyPDF2 failed for {filename}")
                    raise ValueError(f"PDF processing failed - file may be corrupted, password-protected, or contain only images: {filename}") from pdf_error
            else:
                logger.error("PyPDF2 not available for fallback. Install with: pip install PyPDF2")
                raise ValueError(f"PDF processing failed and no fallback available: {filename}") from pdf_error
    
    def _extract_docx(self, file_bytes: bytes, filename: str) -> str:
        """Extract text from DOCX files."""
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = filename
        
        try:
            docx_reader = DocxReader()
            documents = docx_reader.load_data(file_obj)
            if not documents:
                raise ValueError(f"No content extracted from DOCX: {filename}")
            
            combined_text = "\n\n".join([doc.text for doc in documents if doc.text.strip()])
            
            if not combined_text.strip():
                raise ValueError(f"DOCX appears to be empty: {filename}")
                
            return combined_text
            
        except Exception as docx_error:
            raise ValueError(f"Error reading DOCX {filename}: {str(docx_error)}") from docx_error
    
    def _extract_json(self, file_bytes: bytes, filename: str) -> str:
        """Extract and structure text from JSON files."""
        try:
            json_str = file_bytes.decode("utf-8")
            json_data = json.loads(json_str)
            
            json_chunks = self._parse_json_content(json_data)
            if not json_chunks:
                raise ValueError(f"No meaningful content extracted from JSON: {filename}")
            
            return "\n\n".join(json_chunks)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {filename}: {str(e)}")
    
    def _extract_csv_tsv(self, file_bytes: bytes, filename: str, extension: str) -> str:
        """Extract text from CSV/TSV files."""
        try:
            text_content = file_bytes.decode("utf-8")
            lines = text_content.split('\n')
            formatted_lines = []
            
            delimiter = ',' if extension == 'csv' else '\t'
            
            for i, line in enumerate(lines[:100]):  # Limit to first 100 rows for safety
                if line.strip():
                    cells = line.split(delimiter)
                    formatted_line = f"Row {i+1}: " + " | ".join(cells)
                    formatted_lines.append(formatted_line)
            
            return "\n".join(formatted_lines)
            
        except UnicodeDecodeError:
            raise ValueError(f"Unable to decode {extension.upper()} file: {filename}")
    
    def _parse_json_content(self, json_data: Dict[str, Any], parent_key: str = "") -> List[str]:
        """Parse JSON content recursively and extract meaningful text chunks."""
        chunks = []
        
        def extract_from_value(value: Any, key_path: str = "") -> List[str]:
            extracted = []
            
            if isinstance(value, dict):
                for k, v in value.items():
                    new_path = f"{key_path}.{k}" if key_path else k
                    extracted.extend(extract_from_value(v, new_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    new_path = f"{key_path}[{i}]" if key_path else f"[{i}]"
                    extracted.extend(extract_from_value(item, new_path))
            else:
                if key_path:
                    extracted.append(f"{key_path}: {str(value)}")
                else:
                    extracted.append(str(value))
            
            return extracted
        
        all_extracts = extract_from_value(json_data, parent_key)
        
        # Group related items into chunks
        current_chunk = ""
        for extract in all_extracts:
            test_chunk = current_chunk + "\n" + extract if current_chunk else extract
            
            if count_tokens(test_chunk) <= CHUNKING_SETTINGS.MAX_TOKENS:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = extract
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_document(self, document: Document) -> List[TextNode]:
        """Chunk the document into smaller pieces using appropriate strategy."""
        file_type = document.metadata.get("file_type", "")
        
        if file_type == "json":
            return self._chunk_json_document(document)
        else:
            return self._chunk_standard_document(document)
    
    def _chunk_json_document(self, document: Document) -> List[TextNode]:
        """Handle JSON document chunking (already pre-chunked)."""
        json_chunks = document.text.split("\n\n")
        valid_nodes = []
        
        for chunk_text in json_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
                
            if count_tokens(chunk_text) <= CHUNKING_SETTINGS.MAX_TOKENS:
                chunk_node = TextNode(
                    text=chunk_text,
                    metadata=document.metadata.copy()
                )
                valid_nodes.append(chunk_node)
            else:
                json_sub_chunks = split_json_chunk(chunk_text)
                for sub_chunk in json_sub_chunks:
                    if sub_chunk.strip():
                        chunk_node = TextNode(
                            text=sub_chunk.strip(),
                            metadata=document.metadata.copy()
                        )
                        valid_nodes.append(chunk_node)
        
        return valid_nodes
    
    def _chunk_standard_document(self, document: Document) -> List[TextNode]:
        """Handle standard document chunking for PDF, DOCX, TXT, etc."""
        nodes = self.node_parser.get_nodes_from_documents([document])
        valid_nodes = []
        
        for node in nodes:
            if not node.text.strip() or len(node.text.strip()) < 10:
                continue
            
            if count_tokens(node.text) > CHUNKING_SETTINGS.MAX_TOKENS:
                # Split further by sentences
                sentences = split_sentences(node.text)
                current_chunk = ""
                
                for sentence in sentences:
                    test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    
                    if count_tokens(test_chunk) <= CHUNKING_SETTINGS.MAX_TOKENS:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunk_node = TextNode(
                                text=current_chunk.strip(),
                                metadata=node.metadata
                            )
                            valid_nodes.append(chunk_node)
                        current_chunk = sentence
                
                if current_chunk.strip():
                    chunk_node = TextNode(
                        text=current_chunk.strip(),
                        metadata=node.metadata
                    )
                    valid_nodes.append(chunk_node)
            else:
                valid_nodes.append(node)
        
        return valid_nodes
    
    def _validate_chunks(self, text_nodes: List[TextNode]) -> List[str]:
        """Validate chunks and convert to strings."""
        valid_chunks = []
        
        for i, node in enumerate(text_nodes):
            chunk_tokens = count_tokens(node.text)
            chunk_bytes = len(node.text.encode('utf-8'))
            
            # Log chunk details for debugging (first 5 chunks)
            if i < 5:
                logger.info(f"Chunk {i+1}: {chunk_tokens} tokens, {chunk_bytes} bytes")
            
            if chunk_tokens > CHUNKING_SETTINGS.MAX_TOKENS:
                logger.error(f"Chunk {i+1} too large: {chunk_tokens} tokens")
                continue
                
            if chunk_bytes > 3_000_000:  # 3MB safety limit
                logger.error(f"Chunk {i+1} too large: {chunk_bytes} bytes")
                continue
                
            valid_chunks.append(node.text)
        
        return valid_chunks
    
    def _embed_and_upload(
        self,
        chunks: List[str],
        namespace: str = "default",
        source: str = "unknown",
        id_prefix: str = "chunk",
    ):
        """Create embeddings for chunks and upload to Pinecone in batches."""
        if not chunks:
            return
        
        all_vectors = []
        
        # Create vectors for all chunks
        for chunk in chunks:
            # Validate chunk size
            if count_tokens(chunk) > CHUNKING_SETTINGS.MAX_TOKENS:
                logger.error(f"Chunk too large: {count_tokens(chunk)} tokens")
                raise ValueError(f"Chunk too large ({count_tokens(chunk)} tokens). Fix chunking logic.")
            
            chunk_bytes = len(chunk.encode('utf-8'))
            if chunk_bytes > UPLOAD_SETTINGS.MAX_CHUNK_SIZE_BYTES:
                logger.error(f"Chunk too large in bytes: {chunk_bytes} bytes")
                raise ValueError(f"Chunk too large ({chunk_bytes} bytes). Reduce chunk size.")
            
            logger.info(f"Processing chunk: {count_tokens(chunk)} tokens, {chunk_bytes} bytes")
            
            vec = self.embedding_model.get_text_embedding(chunk)
            all_vectors.append({
                "id": f"{id_prefix}-{str(uuid.uuid4())[:8]}",
                "values": vec,
                "metadata": {
                    "text": chunk,
                    "source": source,
                    "namespace": namespace,
                    "chunk_length": count_tokens(chunk),
                    "chunk_bytes": chunk_bytes,
                    "file_type": source.split('.')[-1].lower() if '.' in source else "unknown"
                },
            })
        
        # Upload vectors in batches
        self._upload_vectors_in_batches(all_vectors, namespace)
        logger.info(f"Completed uploading all {len(all_vectors)} vectors for {source}")
    
    def _upload_vectors_in_batches(self, all_vectors: List[Dict], namespace: str):
        """Upload vectors to Pinecone in optimized batches."""
        logger.info(f"Uploading {len(all_vectors)} vectors in batches...")
        
        current_batch = []
        current_batch_size = 0
        
        for vector in all_vectors:
            # Estimate vector size: embeddings (1536 floats * 4 bytes) + metadata
            vector_size = len(vector["values"]) * 4 + vector["metadata"]["chunk_bytes"] + 200
            
            # Check if adding this vector would exceed limits
            if (current_batch_size + vector_size > UPLOAD_SETTINGS.MAX_VECTORS_BATCH_SIZE_BYTES or 
                len(current_batch) >= UPLOAD_SETTINGS.MAX_VECTORS_PER_BATCH):
                
                # Upload current batch
                if current_batch:
                    logger.info(f"Uploading batch of {len(current_batch)} vectors ({current_batch_size:,} bytes)")
                    try:
                        upsert_embeddings(vectors=current_batch, namespace=namespace)
                        logger.info(f"Successfully uploaded batch of {len(current_batch)} vectors")
                    except Exception as e:
                        logger.error(f"Failed to upload batch: {e}")
                        raise e
                    
                    current_batch = []
                    current_batch_size = 0
            
            # Add vector to current batch
            current_batch.append(vector)
            current_batch_size += vector_size
        
        # Upload final batch
        if current_batch:
            logger.info(f"Uploading final batch of {len(current_batch)} vectors ({current_batch_size:,} bytes)")
            try:
                upsert_embeddings(vectors=current_batch, namespace=namespace)
                logger.info(f"Successfully uploaded final batch of {len(current_batch)} vectors")
            except Exception as e:
                logger.error(f"Failed to upload final batch: {e}")
                raise e


# ================================
# RETRIEVAL SYSTEM
# ================================

class DocumentRetriever:
    """
    Comprehensive document retriever with multiple strategies and optimizations.
    Handles query embedding, vector search, and reranking.
    """
    
    def __init__(
        self,
        default_top_k: int = PINECONE_SETTINGS.TOP_K,
        default_namespace: str = "default",
        similarity_threshold: float = RETRIEVER_SETTINGS.SIMILARITY_THRESHOLD,
        max_context_tokens: int = RETRIEVER_SETTINGS.MAX_TOKENS
    ):
        self.default_top_k = default_top_k
        self.default_namespace = default_namespace
        self.similarity_threshold = similarity_threshold
        self.max_context_tokens = max_context_tokens
        
        # Initialize Cohere client if available
        self.cohere_client = None
        if cohere and settings.COHERE_API_KEY:
            try:
                self.cohere_client = cohere.Client(settings.COHERE_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere client: {e}")
        
        # Query variation templates for multi-query retrieval
        self.query_templates = [
            "explain {query}",
            "describe {query}",
            "what is {query}",
            "how to {query}",
            "why {query}"
        ]
        
        # Synonym mapping for query expansion
        self.synonym_map = {
            "how": "what way",
            "what": "which",
            "why": "reason for",
            "when": "time of"
        }
    
    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Main retrieval method that embeds query, finds related chunks, and applies reranking.
        """
        start_time = time.time()
        
        top_k = top_k or self.default_top_k
        namespace = namespace or self.default_namespace
        
        try:
            logger.info(f"Starting retrieval for query: '{query[:50]}...' using {strategy.value} strategy")
            
            # Route to appropriate strategy
            chunks = self._execute_strategy(strategy, query, top_k, namespace, filters, **kwargs)
            
            # Filter by similarity threshold
            filtered_chunks = [
                chunk for chunk in chunks 
                if chunk.score >= self.similarity_threshold
            ]
            
            retrieval_time = time.time() - start_time
            context_length = sum(chunk.chunk_length for chunk in filtered_chunks)
            
            logger.info(f"Retrieved {len(filtered_chunks)} chunks in {retrieval_time:.2f}s")
            
            return RetrievalResult(
                chunks=filtered_chunks,
                query=query,
                strategy=strategy,
                total_found=len(filtered_chunks),
                retrieval_time=retrieval_time,
                context_length=context_length
            )
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {str(e)}")
            raise e
    
    def _execute_strategy(
        self,
        strategy: RetrievalStrategy,
        query: str,
        top_k: int,
        namespace: str,
        filters: Optional[Dict[str, Any]],
        **kwargs
    ) -> List[RetrievedChunk]:
        """Execute the specified retrieval strategy."""
        if strategy == RetrievalStrategy.SEMANTIC:
            return self._semantic_retrieval(query, top_k, namespace, filters)
        elif strategy == RetrievalStrategy.HYBRID:
            return self._hybrid_retrieval(query, top_k, namespace, filters, **kwargs)
        elif strategy == RetrievalStrategy.MULTI_QUERY:
            return self._multi_query_retrieval(query, top_k, namespace, filters, **kwargs)
        elif strategy == RetrievalStrategy.CONTEXTUAL:
            return self._contextual_retrieval(query, top_k, namespace, filters, **kwargs)
        elif strategy == RetrievalStrategy.RERANK:
            return self._rerank_retrieval(query, top_k, namespace, filters, **kwargs)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
    
    def _semantic_retrieval(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filters: Optional[Dict[str, Any]]
    ) -> List[RetrievedChunk]:
        """Basic semantic similarity retrieval using embedding and vector search."""
        # Check index stats before querying
        logger.info(f"Checking index stats for namespace '{namespace}'...")
        has_vectors = check_index_stats(namespace)
        
        if not has_vectors:
            logger.warning(f"No vectors found in namespace '{namespace}' - returning empty results")
            return []
        
        # Create query embedding
        logger.info(f"Creating embedding for query: '{query[:50]}...'")
        query_vector = embed_texts(query)
        logger.info(f"Query embedding created with dimension: {len(query_vector)}")
        
        # Query Pinecone
        logger.info(f"Querying Pinecone with top_k={top_k}, namespace='{namespace}'")
        results = query_embeddings(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace
        )
        
        logger.info(f"Pinecone returned {len(results.matches) if hasattr(results, 'matches') else 0} matches")
        return self._process_pinecone_results(results)
    
    def _hybrid_retrieval(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filters: Optional[Dict[str, Any]],
        keyword_boost: float = RETRIEVER_SETTINGS.KEYWORD_BOOST,
        **kwargs
    ) -> List[RetrievedChunk]:
        """Hybrid retrieval combining semantic and keyword matching."""
        # Get semantic results
        semantic_chunks = self._semantic_retrieval(query, top_k * 2, namespace, filters)
        
        # Apply keyword boosting
        query_keywords = set(query.lower().split())
        
        for chunk in semantic_chunks:
            text_words = set(chunk.text.lower().split())
            keyword_overlap = len(query_keywords.intersection(text_words))
            keyword_ratio = keyword_overlap / len(query_keywords) if query_keywords else 0
            
            # Apply keyword boost
            chunk.score += keyword_ratio * keyword_boost
        
        # Re-sort by updated scores and return top_k
        semantic_chunks.sort(key=lambda x: x.score, reverse=True)
        return semantic_chunks[:top_k]
    
    def _multi_query_retrieval(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filters: Optional[Dict[str, Any]],
        query_variations: Optional[List[str]] = None,
        **kwargs
    ) -> List[RetrievedChunk]:
        """Multi-query retrieval using query variations."""
        if query_variations is None:
            query_variations = self._generate_query_variations(query)
        
        all_chunks = []
        seen_chunk_ids = set()
        
        # Retrieve for each query variation
        for variation in [query] + query_variations:
            chunks = self._semantic_retrieval(variation, top_k // 2, namespace, filters)
            
            for chunk in chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)
        
        # Sort by score and return top_k
        all_chunks.sort(key=lambda x: x.score, reverse=True)
        return all_chunks[:top_k]
    
    def _contextual_retrieval(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filters: Optional[Dict[str, Any]],
        conversation_history: Optional[List[str]] = None,
        **kwargs
    ) -> List[RetrievedChunk]:
        """Context-aware retrieval considering conversation history."""
        # If we have conversation history, create a contextual query
        if conversation_history:
            # Combine recent history with current query
            context = " ".join(conversation_history[-3:])  # Last 3 messages
            contextual_query = f"Context: {context}\nQuestion: {query}"
        else:
            contextual_query = query
        
        return self._semantic_retrieval(contextual_query, top_k, namespace, filters)
    
    def _rerank_retrieval(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filters: Optional[Dict[str, Any]],
        rerank_factor: int = RETRIEVER_SETTINGS.RERANK_FACTOR,
        **kwargs
    ) -> List[RetrievedChunk]:
        """Retrieval with Cohere reranking for improved relevance."""
        # Retrieve more candidates for reranking
        initial_top_k = min(top_k * rerank_factor, RETRIEVER_SETTINGS.COHERE_TOP_N)
        candidates = self._semantic_retrieval(query, initial_top_k, namespace, filters)
        
        if not candidates:
            return []
        
        # If Cohere is not available, return top candidates by original score
        if not self.cohere_client:
            logger.warning("Cohere reranker not available, falling back to semantic scores")
            return candidates[:top_k]
        
        try:
            # Prepare documents for Cohere reranking
            documents = [chunk.text for chunk in candidates]
            
            # Use Cohere reranker
            rerank_response = self.cohere_client.rerank(
                model=RETRIEVER_SETTINGS.COHERE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_k,
                return_documents=False
            )
            
            # Map reranked results back to our chunks
            reranked_chunks = []
            for result in rerank_response.results:
                original_chunk = candidates[result.index]
                # Update the score with Cohere's relevance score
                original_chunk.score = result.relevance_score
                reranked_chunks.append(original_chunk)
            
            logger.info(f"Cohere reranking completed: {len(reranked_chunks)} chunks reranked")
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            # Fall back to original candidates
            return candidates[:top_k]
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations for multi-query retrieval."""
        variations = []
        
        # Question variations
        if not query.endswith('?'):
            variations.append(f"{query}?")
        
        # Template-based variations
        for template in self.query_templates:
            if template.split()[0].lower() not in query.lower():
                variations.append(template.format(query=query))
        
        # Synonym-based variations
        for original, synonym in self.synonym_map.items():
            if original in query.lower():
                variations.append(query.lower().replace(original, synonym))
        
        return variations[:3]  # Limit to 3 variations
    
    def _process_pinecone_results(self, results) -> List[RetrievedChunk]:
        """Convert Pinecone results to RetrievedChunk objects."""
        chunks = []
        
        if not results:
            logger.warning("No results returned from Pinecone")
            return chunks
            
        if not hasattr(results, 'matches'):
            logger.warning("Results object has no 'matches' attribute")
            return chunks
        
        logger.info(f"Processing {len(results.matches)} Pinecone matches")
        
        for i, match in enumerate(results.matches):
            metadata = match.metadata or {}
            logger.info(f"Match {i+1}: ID={match.id}, Score={match.score:.4f}, Source={metadata.get('source', 'unknown')}")
            
            chunk = RetrievedChunk(
                text=metadata.get("text", ""),
                source=metadata.get("source", "unknown"),
                score=float(match.score),
                chunk_id=match.id,
                file_type=metadata.get("file_type", "unknown"),
                chunk_length=metadata.get("chunk_length", 0),
                namespace=metadata.get("namespace", "default"),
                metadata=metadata
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} RetrievedChunk objects")
        return chunks


# ================================
# RESPONSE GENERATION SYSTEM
# ================================

class ExpertResponseGenerator:
    """
    Expert response generator that creates intelligent responses from retrieved chunks.
    Uses OpenAI models to generate contextual responses based on retrieved information.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        Initialize the expert response generator.
        
        Args:
            model: OpenAI model to use
            max_tokens: Maximum tokens for response
            temperature: Response creativity (0.0-1.0)
        """
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate_response(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        conversation_history: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None,
        **kwargs
    ) -> ResponseResult:
        """
        Generate an expert response based on query and retrieved chunks.
        """
        start_time = time.time()
        
        try:
            logger.info("Generating expert response for query: '%s'", query[:50] + "...")
            
            # Format context from chunks
            context = self._format_context_from_chunks(chunks)
            
            # Build conversation messages
            messages = self._build_conversation_messages(
                query=query,
                context=context,
                conversation_history=conversation_history,
                custom_system_prompt=custom_system_prompt
            )
            
            # Prepare API parameters
            api_params = self._prepare_api_parameters(messages, **kwargs)
            
            # Generate response
            response = self.client.chat.completions.create(**api_params)
            
            # Extract response content
            response_content = response.choices[0].message.content
            generation_time = time.time() - start_time
            
            # Build metadata
            metadata = self._build_response_metadata(
                response=response,
                api_params=api_params,
                chunks=chunks,
                context=context,
                generation_time=generation_time
            )
            
            logger.info("Response generated successfully in %.2f seconds", generation_time)
            
            return ResponseResult(
                success=True,
                response=response_content,
                query=query,
                chunks_used=chunks,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Response generation failed: {str(e)}"
            logger.error(error_msg)
            
            return ResponseResult(
                success=False,
                response=f"I apologize, but I encountered an error while generating a response: {str(e)}",
                query=query,
                chunks_used=chunks or [],
                metadata={
                    "error": str(e),
                    "generation_time": time.time() - start_time,
                    "chunks_used": len(chunks) if chunks else 0
                },
                error_message=error_msg
            )
    
    def generate_streaming_response(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        conversation_history: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response for real-time output.
        """
        try:
            logger.info("Starting streaming expert response for query: '%s'", query[:50] + "...")
            
            # Format context from chunks
            context = self._format_context_from_chunks(chunks)
            
            # Build conversation messages
            messages = self._build_conversation_messages(
                query=query,
                context=context,
                conversation_history=conversation_history,
                custom_system_prompt=custom_system_prompt
            )
            
            # Prepare API parameters for streaming
            api_params = self._prepare_api_parameters(messages, stream=True, **kwargs)
            
            # Generate streaming response
            stream = self.client.chat.completions.create(**api_params)
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error("Streaming response failed: %s", str(e))
            yield f"Error: {str(e)}"
    
    def _format_context_from_chunks(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into a coherent context string."""
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Build source information
            source_info = f"Source {i}: {chunk.source}"
            
            # Add additional metadata if available
            if hasattr(chunk, 'metadata') and chunk.metadata:
                if 'page' in chunk.metadata:
                    source_info += f", Page {chunk.metadata['page']}"
                if 'section' in chunk.metadata:
                    source_info += f", Section {chunk.metadata['section']}"
            
            # Format chunk with source information
            chunk_text = f"[{source_info}]\n{chunk.text}\n"
            context_parts.append(chunk_text)
        
        return "\n".join(context_parts)
    
    def _build_conversation_messages(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build conversation messages for the OpenAI API."""
        # Get system prompt
        if custom_system_prompt:
            system_prompt = custom_system_prompt.format(context=context)
        else:
            system_prompt = self._get_expert_system_prompt().format(context=context)
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history (limit to last 5 messages for context management)
        if conversation_history:
            for i, msg in enumerate(conversation_history[-5:]):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": msg})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def _prepare_api_parameters(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare parameters for OpenAI API call."""
        return {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": stream,
            **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "temperature"]}
        }
    
    def _build_response_metadata(
        self,
        response: Any,
        api_params: Dict[str, Any],
        chunks: List[RetrievedChunk],
        context: str,
        generation_time: float
    ) -> Dict[str, Any]:
        """Build comprehensive metadata for the response."""
        return {
            "model_used": api_params["model"],
            "generation_time": generation_time,
            "chunks_used": len(chunks),
            "context_length": len(context),
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "temperature": api_params["temperature"],
            "sources": list(set(chunk.source for chunk in chunks)),
            "unique_file_types": list(set(chunk.file_type for chunk in chunks)),
            "average_chunk_score": sum(chunk.score for chunk in chunks) / len(chunks) if chunks else 0,
            "timestamp": time.time()
        }
    
    def _get_expert_system_prompt(self) -> str:
        """Get the expert system prompt template."""
        return """You are an Expert AI Assistant with deep knowledge across multiple domains. You provide accurate, comprehensive, and helpful responses based on the provided context. Your expertise allows you to understand complex topics and explain them clearly to users.

Guidelines:
1. Analyze the provided context thoroughly and extract relevant information
2. Provide accurate, well-structured responses based on the context
3. If the context doesn't contain sufficient information, clearly state what's missing
4. Cite sources when possible using the source information provided
5. Use your expertise to provide insights and explanations that help users understand the topic
6. Be concise yet comprehensive - balance detail with clarity
7. If you're uncertain about something, state your confidence level
8. Organize complex information using bullet points, numbered lists, or structured formats when helpful
9. Maintain a professional yet approachable tone
10. Draw connections between different pieces of information when relevant

Context:
{context}

Based on your expertise and the provided context, please answer the user's question accurately and helpfully."""


# ================================
# UNIFIED RAG SYSTEM
# ================================

class UnifiedRAGSystem:
    """
    Complete RAG system that integrates ingestion, retrieval, and response generation.
    Provides high-level interface for all RAG operations.
    """
    
    def __init__(self):
        """Initialize the unified RAG system with all components."""
        self.ingestion_pipeline = DocumentIngestionPipeline()
        self.retriever = DocumentRetriever()
        self.response_generator = ExpertResponseGenerator()
        
        logger.info("Unified RAG System initialized successfully")
    
    # Document Ingestion Interface
    def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        namespace: str = "default"
    ) -> IngestionResult:
        """
        Ingest a document into the system.
        Independent operation for upload routes.
        """
        return self.ingestion_pipeline.ingest_file(file_bytes, filename, namespace)
    
    # Query Processing Interface  
    def process_query(
        self,
        query: str,
        strategy: str = "rerank",
        top_k: int = 5,
        namespace: str = "default",
        conversation_history: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ResponseResult, Generator[str, None, None]]:
        """
        Complete query processing: retrieval + response generation.
        
        Args:
            query: User's question
            strategy: Retrieval strategy (semantic, hybrid, rerank, etc.)
            top_k: Number of chunks to retrieve
            namespace: Vector database namespace
            conversation_history: Previous conversation context
            stream: Whether to return streaming response
            **kwargs: Additional parameters
            
        Returns:
            ResponseResult or streaming generator
        """
        try:
            # Step 1: Retrieve relevant chunks
            retrieval_strategy = RetrievalStrategy(strategy)
            retrieval_result = self.retriever.retrieve(
                query=query,
                strategy=retrieval_strategy,
                top_k=top_k,
                namespace=namespace,
                conversation_history=conversation_history,
                **kwargs
            )
            
            # Step 2: Generate response
            if stream:
                return self.response_generator.generate_streaming_response(
                    query=query,
                    chunks=retrieval_result.chunks,
                    conversation_history=conversation_history,
                    **kwargs
                )
            else:
                response_result = self.response_generator.generate_response(
                    query=query,
                    chunks=retrieval_result.chunks,
                    conversation_history=conversation_history,
                    **kwargs
                )
                
                # Enhance metadata with retrieval info
                response_result.metadata["retrieval"] = {
                    "strategy": strategy,
                    "chunks_retrieved": len(retrieval_result.chunks),
                    "retrieval_time": retrieval_result.retrieval_time,
                    "context_length": retrieval_result.context_length
                }
                
                return response_result
                
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return ResponseResult(
                success=False,
                response="I'm sorry, I encountered an error while processing your question.",
                query=query,
                chunks_used=[],
                metadata={"error": str(e)},
                error_message=str(e)
            )
    
    # Specialized retrieval methods
    def search_by_source(
        self,
        source_filename: str,
        query: str,
        top_k: int = 5,
        namespace: str = "default"
    ) -> RetrievalResult:
        """Search within a specific source document."""
        return self.retriever.retrieve(
            query=query,
            strategy=RetrievalStrategy.SEMANTIC,
            top_k=top_k,
            namespace=namespace,
            filters={"source": source_filename}
        )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get health status of all RAG components."""
        return {
            "status": "healthy",
            "components": {
                "ingestion": "ready",
                "retrieval": "ready", 
                "response_generation": "ready"
            },
            "supported_strategies": [strategy.value for strategy in RetrievalStrategy],
            "supported_file_types": list(self.ingestion_pipeline.supported_extensions),
            "cohere_available": self.retriever.cohere_client is not None,
            "timestamp": time.time()
        }


# ================================
# GLOBAL INSTANCES AND CONVENIENCE FUNCTIONS
# ================================

# Global RAG system instance
rag_system = UnifiedRAGSystem()

# Convenience functions for direct usage in routes
def ingest_file_to_vectordb(
    file_bytes: bytes,
    filename: str,
    namespace: str = "default"
) -> IngestionResult:
    """
    Ingest file to vector database - for upload routes.
    Independent operation that returns success response.
    """
    return rag_system.ingest_document(file_bytes, filename, namespace)


def query_rag_system(
    query: str,
    strategy: str = "rerank",
    top_k: int = 5,
    namespace: str = "default",
    conversation_history: Optional[List[str]] = None,
    stream: bool = False,
    **kwargs
) -> Union[ResponseResult, Generator[str, None, None]]:
    """
    Query RAG system - for query routes.
    Handles retrieval + response generation.
    """
    return rag_system.process_query(
        query=query,
        strategy=strategy,
        top_k=top_k,
        namespace=namespace,
        conversation_history=conversation_history,
        stream=stream,
        **kwargs
    )


def get_rag_health() -> Dict[str, Any]:
    """Get RAG system health status."""
    return rag_system.get_system_health()


# Backward compatibility functions
def ask_expert(
    query: str,
    chunks: List[RetrievedChunk],
    conversation_history: Optional[List[str]] = None,
    stream: bool = False,
    **kwargs
) -> Union[ResponseResult, Generator[str, None, None]]:
    """
    Backward compatibility function for direct chunk-based queries.
    """
    if stream:
        return rag_system.response_generator.generate_streaming_response(
            query=query,
            chunks=chunks,
            conversation_history=conversation_history,
            **kwargs
        )
    else:
        return rag_system.response_generator.generate_response(
            query=query,
            chunks=chunks,
            conversation_history=conversation_history,
            **kwargs
        )
