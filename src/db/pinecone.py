import uuid
from pinecone import Pinecone
from src.config.config import Config
from llama_index.embeddings.openai import OpenAIEmbedding
import logging

logger = logging.getLogger(__name__)

class PineconeDB:
    def __init__(self):
        """Initialize without connecting to Pinecone immediately (lazy loading)."""
        self.config = Config()
        self._pc = None
        self._index = None
        self._embed_model = None
        
    def _get_pinecone_client(self):
        """Lazy initialize Pinecone client."""
        if self._pc is None:
            try:
                self._pc = Pinecone(api_key=self.config.getString("PINECONE_API_KEY"))
                logger.info("Pinecone client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone client: {e}")
                raise
        return self._pc
    
    def _get_index(self):
        """Lazy initialize Pinecone index."""
        if self._index is None:
            try:
                pc = self._get_pinecone_client()
                index_name = self.config.getString("PINECONE_INDEX_NAME")
                self._index = pc.Index(index_name)
                logger.info(f"Pinecone index '{index_name}' connected")
            except Exception as e:
                logger.error(f"Failed to connect to Pinecone index: {e}")
                raise
        return self._index
    
    def _get_embed_model(self):
        """Lazy initialize embedding model."""
        if self._embed_model is None:
            try:
                self._embed_model = OpenAIEmbedding()
                logger.info("OpenAI embedding model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                raise
        return self._embed_model
        
    def upsert(self, vectors: list):
        """Upsert vectors into the Pinecone index."""
        try:
            index = self._get_index()
            index.upsert(vectors=vectors)
            logger.info(f"Upserted {len(vectors)} vectors")
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise
        
    def query(self, vector: list, top_k: int = 10):
        """Query the Pinecone index for similar vectors."""
        try:
            index = self._get_index()
            return index.query(vector=vector, top_k=top_k)
        except Exception as e:
            logger.error(f"Failed to query vectors: {e}")
            raise
    
    def storeChunks(self, chunks, source: str = "document"):
        """Store text chunks as vectors in Pinecone."""
        try:
            embed_model = self._get_embed_model()
            embeddedVectors = []
            
            # Process chunks one by one to avoid batch API issues
            for i, chunk in enumerate(chunks):
                # Handle both string chunks and chunk objects
                if hasattr(chunk, 'text'):
                    text = chunk.text
                else:
                    text = str(chunk)
                
                # Get single embedding (avoid batch method that may have API issues)
                embedding = embed_model.get_text_embedding(text)
                uniqueId = str(uuid.uuid5(uuid.NAMESPACE_DNS, text))
                
                embeddedVectors.append({
                    "id": uniqueId,
                    "values": embedding,
                    "metadata": {
                        "text": text,
                        "source": source,
                        "chunk_id": i
                    }
                })
                    
            if embeddedVectors:
                self.upsert(vectors=embeddedVectors)
                logger.info(f"Stored {len(embeddedVectors)} chunks in Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise
            
    def semanticSearch(self, query: str, top_k: int = 5):
        """Perform semantic search and return relevant context."""
        try:
            embed_model = self._get_embed_model()
            index = self._get_index()
            
            # Get query embedding
            query_embedding = embed_model.get_text_embedding(query)
            
            # Search Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract context from results
            context_docs = []
            for match in results.get('matches', []):
                if hasattr(match, 'metadata') and 'text' in match.metadata:
                    context_docs.append({
                        "text": match.metadata['text'],
                        "source": match.metadata.get('source', 'unknown'),
                        "chunk_id": match.metadata.get('chunk_id', 0),
                        "relevance_score": round(match.get('score', 0) * 10, 1),
                        "metadata": match.metadata
                    })
            
            logger.info(f"Found {len(context_docs)} relevant chunks for query")
            return context_docs
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []