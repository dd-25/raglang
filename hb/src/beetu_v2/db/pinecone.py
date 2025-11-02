from typing import List, Any
import logging
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
from beetu_v2.config import settings
from beetu_v2.constants import EMBEDDING_SETTINGS, PINECONE_SETTINGS

# Set up logging
logger = logging.getLogger(__name__)

# Initialize Pinecone client safely
try:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    INDEX_NAME = settings.PINECONE_INDEX_NAME
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {e}")
    raise


def check_pinecone_connection():
    """
    Diagnostic function to check Pinecone connection and index status.
    """
    try:
        # List all indexes
        indexes = pc.list_indexes()
        
        # Check if our index exists
        if INDEX_NAME not in [idx.name for idx in indexes]:
            logger.error(f"Index '{INDEX_NAME}' not found!")
            return False
        
        # Try to describe the index
        idx_config = pc.describe_index(INDEX_NAME)
        
        # Try to connect to the index
        idx = pc.Index(host=idx_config.host)
        stats = idx.describe_index_stats()
        
        return True
        
    except Exception as e:
        logger.error(f"Pinecone connection check failed: {str(e)}")
        return False


# def create_index_if_not_exists(dim: int = PINECONE_SETTINGS.DIMENSION, metric: str = "cosine"):
#     """
#     Create the Pinecone index with the specified dimension if it does not exist.
#     """
#     indexes = [idx.name for idx in pc.list_indexes()]
#     if INDEX_NAME not in indexes:
#         spec = ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1)
#         return pc.create_index(
#             name=INDEX_NAME,
#             dimension=dim,
#             metric=metric,
#             spec=spec
#         )
# create_index_if_not_exists()


def get_index():
    """Get the Pinecone Index client for the configured index (assumes it already exists)."""
    try:
        idx_config = pc.describe_index(INDEX_NAME)
        return pc.Index(host=idx_config.host)
    except Exception as e:
        logger.error(f"Failed to get Pinecone index '{INDEX_NAME}': {str(e)}")
        raise


def check_index_stats(namespace: str = "default"):
    """
    Check what's actually in the Pinecone index.
    """
    try:
        idx = get_index()
        stats = idx.describe_index_stats()
        
        # Check specific namespace
        if namespace in (stats.namespaces or {}):
            ns_count = stats.namespaces[namespace].vector_count
            return ns_count > 0
        else:
            logger.warning(f"Namespace '{namespace}' not found!")
            return False
            
    except Exception as e:
        logger.error(f"Failed to check index stats: {str(e)}")
        return False

def upsert_embeddings(vectors: List[Any], namespace: str = "default"):
    """
    Upsert embeddings into Pinecone. Vectors should be a list of tuples: (id, values, metadata)
    """
    idx = get_index()
    return idx.upsert(vectors=vectors, namespace=namespace)

def query_embeddings(vector: List[float], top_k: int = PINECONE_SETTINGS.TOP_K, namespace: str = "default"):
    """
    Query top-K similar vectors from a namespace.
    """
    idx = get_index()
    return idx.query(vector=vector, top_k=top_k, namespace=namespace, include_metadata=True)

def delete_namespace(namespace: str):
    """
    Delete all vectors in a specific namespace.
    """
    idx = get_index()
    idx.delete(delete_all=True, namespace=namespace)
