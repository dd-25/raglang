from pinecone import Pinecone
from src.config.config import Config

class PineconeDB:
    def __init__(self):
        configService = Config()
        self.pc = Pinecone(api_key=configService.getString("PINECONE_API_KEY"))
        self.index = self.pc.Index(configService.getString("PINECONE_INDEX_NAME"))

    def upsert(self, vectors: list):
        """Upsert vectors into the Pinecone index."""
        self.index.upsert(vectors=vectors)

    def query(self, vector: list, top_k: int = 10):
        """Query the Pinecone index for similar vectors."""
        return self.index.query(vector=vector, top_k=top_k)