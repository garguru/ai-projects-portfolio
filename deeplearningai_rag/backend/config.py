import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the advanced RAG system"""

    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800  # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100  # Characters to overlap between chunks
    MAX_RESULTS: int = 5  # Maximum search results to return
    MAX_HISTORY: int = 2  # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    # Hybrid Search Settings
    USE_HYBRID_SEARCH: bool = True  # Enable hybrid search by default
    SEMANTIC_WEIGHT: float = 0.6  # Weight for semantic search in hybrid mode
    KEYWORD_WEIGHT: float = 0.4  # Weight for BM25/keyword search in hybrid mode
    FUSION_METHOD: str = "rrf"  # "rrf", "weighted", or "diversity"
    RRF_K: int = 60  # RRF parameter for rank fusion

    # Query Enhancement Settings
    USE_QUERY_ENHANCEMENT: bool = True  # Enable query enhancement features
    USE_QUERY_REWRITING: bool = True  # Enable query rewriting
    USE_HYDE: bool = True  # Enable Hypothetical Document Embeddings
    USE_QUERY_DECOMPOSITION: bool = False  # Enable query decomposition
    USE_QUERY_EXPANSION: bool = False  # Enable query expansion
    MAX_QUERY_VARIATIONS: int = 3  # Maximum query variations to generate

    # RAG-Fusion Settings
    USE_RAG_FUSION: bool = False  # Enable RAG-Fusion (set to False by default)
    RAG_FUSION_QUERIES: int = 3  # Number of query variations for RAG-Fusion
    RAG_FUSION_METHOD: str = "rrf"  # Fusion method for RAG-Fusion

    # Reranking Settings
    USE_RERANKING: bool = True  # Enable result reranking
    RERANK_STRATEGY: str = "cross_encoder"  # "cross_encoder", "mmr", or "combined"
    RERANK_TOP_K: int = 20  # Number of initial results to rerank
    MMR_LAMBDA: float = 0.7  # MMR balance parameter (relevance vs diversity)

    # Performance Settings
    USE_CACHING: bool = True  # Enable query result caching
    CACHE_TTL: int = 3600  # Cache time-to-live in seconds (1 hour)
    ENABLE_PARALLEL_SEARCH: bool = True  # Enable parallel query processing

    # Evaluation Settings
    ENABLE_METRICS: bool = True  # Enable performance metrics collection
    LOG_QUERIES: bool = True  # Log queries for analysis
    FEEDBACK_COLLECTION: bool = False  # Enable user feedback collection


config = Config()
