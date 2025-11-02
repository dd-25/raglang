"""
Production-Optimized Document Ingestion Pipeline

Streamlined document processing with minimal overhead:
- Multi-format support (PDF, DOCX, TXT, JSON, CSV, TSV)
- Optimized chunking and embedding
- Batch processing for efficiency
- Minimal logging and error handling
"""

import io
import json
import uuid
import logging
import time
from typing import List, Dict, Any, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PDFReader, DocxReader

from beetu_v2.db.pinecone import upsert_embeddings
from beetu_v2.config import settings
from beetu_v2.agents.ragagent.utils.token import count_tokens
from beetu_v2.constants import EMBEDDING_SETTINGS, CHUNKING_SETTINGS, UPLOAD_SETTINGS
from beetu_v2.agents.ragagent.dto import IngestionResult

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Production-optimized document processor.
    Focuses on core functionality with minimal overhead.
    """
    
    def __init__(self):
        self.embedding_model = OpenAIEmbedding(
            model=EMBEDDING_SETTINGS.EMBED_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        
        self.chunker = SentenceSplitter(
            chunk_size=CHUNKING_SETTINGS.MAX_TOKENS,
            chunk_overlap=CHUNKING_SETTINGS.OVERLAP_TOKENS
        )
    
    def process_file(self, file_bytes: bytes, filename: str, namespace: str = "default") -> IngestionResult:
        """Main processing function - optimized for production."""
        start_time = time.time()
        
        try:
            # Extract text
            text = self._extract_text(file_bytes, filename)
            if not text.strip():
                return IngestionResult(False, 0, time.time() - start_time, "No text extracted")
            
            # Create chunks
            chunks = self._create_chunks(text)
            if not chunks:
                return IngestionResult(False, 0, time.time() - start_time, "No valid chunks created")
            
            # Create embeddings and upload
            self._embed_and_upload(chunks, namespace, filename)
            
            return IngestionResult(
                success=True,
                chunks_created=len(chunks),
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return IngestionResult(
                success=False,
                chunks_created=0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _extract_text(self, file_bytes: bytes, filename: str) -> str:
        """Extract text based on file extension."""
        ext = filename.split('.')[-1].lower()
        
        if ext == "pdf":
            return self._extract_pdf(file_bytes, filename)
        elif ext == "docx":
            return self._extract_docx(file_bytes, filename)
        elif ext == "json":
            return self._extract_json(file_bytes)
        elif ext in ["txt", "md"]:
            return file_bytes.decode("utf-8")
        elif ext in ["csv", "tsv"]:
            return self._extract_csv(file_bytes, ext)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _extract_pdf(self, file_bytes: bytes, filename: str) -> str:
        """Extract PDF text using LlamaIndex."""
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = filename
        
        pdf_reader = PDFReader()
        documents = pdf_reader.load_data(file_obj)
        
        return "\n\n".join([doc.text for doc in documents if doc.text.strip()])
    
    def _extract_docx(self, file_bytes: bytes, filename: str) -> str:
        """Extract DOCX text using LlamaIndex."""
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = filename
        
        docx_reader = DocxReader()
        documents = docx_reader.load_data(file_obj)
        
        return "\n\n".join([doc.text for doc in documents if doc.text.strip()])
    
    def _extract_json(self, file_bytes: bytes) -> str:
        """Extract structured JSON content."""
        data = json.loads(file_bytes.decode("utf-8"))
        return self._flatten_json(data)
    
    def _extract_csv(self, file_bytes: bytes, ext: str) -> str:
        """Extract CSV/TSV content."""
        text = file_bytes.decode("utf-8")
        lines = text.split('\\n')[:100]  # Limit rows for efficiency
        
        delimiter = ',' if ext == "csv" else '\\t'
        formatted = []
        
        for i, line in enumerate(lines):
            if line.strip():
                fields = line.split(delimiter)
                prefix = "Headers" if i == 0 else f"Row {i}"
                formatted.append(f"{prefix}: {', '.join(fields)}")
        
        return "\\n".join(formatted)
    
    def _flatten_json(self, data: Any, prefix: str = "") -> str:
        """Flatten JSON into readable text chunks."""
        lines = []
        
        def extract_values(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    if isinstance(v, (str, int, float, bool)):
                        lines.append(f"{new_path}: {v}")
                    else:
                        extract_values(v, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:50]):  # Limit list items
                    new_path = f"{path}[{i}]"
                    if isinstance(item, (str, int, float, bool)):
                        lines.append(f"{new_path}: {item}")
                    else:
                        extract_values(item, new_path)
        
        extract_values(data, prefix)
        return "\\n".join(lines)
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create optimized text chunks."""
        document = Document(text=text)
        nodes = self.chunker.get_nodes_from_documents([document])
        
        chunks = []
        for node in nodes:
            chunk_text = node.text.strip()
            if chunk_text and len(chunk_text) > 10 and count_tokens(chunk_text) <= CHUNKING_SETTINGS.MAX_TOKENS:
                chunks.append(chunk_text)
        
        return chunks
    
    def _embed_and_upload(self, chunks: List[str], namespace: str, source: str):
        """Create embeddings and upload to Pinecone."""
        vectors = []
        
        for chunk in chunks:
            embedding = self.embedding_model.get_text_embedding(chunk)
            vectors.append({
                "id": f"chunk-{uuid.uuid4().hex[:8]}",
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source": source,
                    "namespace": namespace,
                    "tokens": count_tokens(chunk)
                }
            })
        
        # Upload in batches for efficiency
        batch_size = UPLOAD_SETTINGS.MAX_VECTORS_PER_BATCH
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            upsert_embeddings(vectors=batch, namespace=namespace)


# Global processor instance
document_processor = DocumentProcessor()


def ingest_file_to_vectordb(file_bytes: bytes, filename: str, namespace: str = "default") -> IngestionResult:
    """
    Main ingestion function for route usage.
    Optimized for production with minimal overhead.
    """
    return document_processor.process_file(file_bytes, filename, namespace)
