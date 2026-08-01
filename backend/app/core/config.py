import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Dynamically resolve absolute path to .env file in the backend folder
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent.parent
env_file_path = backend_dir / ".env"

class Settings(BaseSettings):
    # Pinecone configurations
    PINECONE_API_KEY: str = Field(..., description="Pinecone service API key")
    PINECONE_INDEX_NAME: str = Field(default="agentic-rag-index", description="Name of Pinecone Index")

    # Ollama Model Configurations
    OLLAMA_ROUTER_MODEL: str = Field(default="llama3", description="Ollama model for supervisor agent routing decisions")
    OLLAMA_GRADER_MODEL: str = Field(default="llama3", description="Ollama model for CRAG document relevance and hallucination grading")
    OLLAMA_GENERATION_MODEL: str = Field(default="llama3", description="Ollama model for final response generation")
    OLLAMA_METADATA_MODEL: str = Field(default="llama3", description="Ollama model for ingestion chunk summary/entity extraction")

    # Default Embedding Model
    OLLAMA_EMBEDDING_MODEL: str = Field(default="embeddinggemma:300m-qat-q4_0", description="Ollama embedding model name")

    # Semantic Reranker Configs
    CROSS_ENCODER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="CrossEncoder reranking model name")

    # Ingestion Configurations
    CHUNK_SIZE: int = Field(default=800, description="Max token limit per chunk")
    CHUNK_OVERLAP: int = Field(default=160, description="Token overlap for text splitting chunks")

    # Retrieval configurations
    RAG_RETRIEVE_TOP_K: int = Field(default=5, description="Total dense + sparse vectors retrieved before RRF")
    RAG_RERANK_TOP_N: int = Field(default=3, description="Top final documents returned after Semantic Reranking")

    # Agent workflow constraints
    MAX_WEB_SEARCH_RETRIES: int = Field(default=2, description="Maximum loop count for web search fallbacks")

    # Configuration for Settings loading
    model_config = SettingsConfigDict(
        env_file=str(env_file_path),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings
settings = Settings()
