import os
from dotenv import load_dotenv
import requests
import pdfplumber
from io import BytesIO
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.schema import Document
from src.config.config import Config
from src.db.pinecone import PineconeDB

class DataIngestion:
    def __init__(self):
        self.config = Config()
        self.llm = self.config.getLLM()
        self.embed_model = OpenAIEmbedding()
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1,  # minimum overlap
            breakpoint_percentile_threshold=95,  # increase for more coarse chunks
            embed_model=self.embed_model  # Required parameter
        )
        self.pc = PineconeDB()
    
    def uploadFile(self, file):
        pass
    
    def parseDoc(self, file_content: bytes, filename: str = "document") -> str:
        """
        Parse document from file content bytes.
        
        Args:
            file_content: Raw file content as bytes
            filename: Optional filename for error messages
            
        Returns:
            Extracted text from the document
        """
        all_text = []
        
        try:
            # Create BytesIO from file content
            pdf_content = BytesIO(file_content)
            
            # Extract text and tables from PDF
            with pdfplumber.open(pdf_content) as pdf:
                for page in pdf.pages:
                    # Extract text
                    text = page.extract_text() or ""
                    if text.strip():
                        all_text.append(text.strip())

                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:  # Check if row is not None
                                row_text = " | ".join(cell.strip() if cell else "" for cell in row)
                                if row_text.strip():
                                    all_text.append(row_text)
            
            return "\n".join(all_text)
            
        except Exception as e:
            raise ValueError(f"Failed to parse document '{filename}': {str(e)}")

    def docChunker(self, text: str) -> list[str]:
        doc = Document(text=text)
        nodes = self.splitter.get_nodes_from_documents([doc])
        return [node.text for node in nodes]

    def uploadDoc(self, file_content: bytes, filename: str = "document"):
        """
        Process document from file content without storing to disk.
        
        Args:
            file_content: Raw file content as bytes
            filename: Optional filename for error messages
            
        Returns:
            List of text chunks
        """
        all_text = self.parseDoc(file_content, filename)
        chunks = self.docChunker(all_text)
        # Pass filename as source to Pinecone
        self.pc.storeChunks(chunks, source=filename)
        return chunks