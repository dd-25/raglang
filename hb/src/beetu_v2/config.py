from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_ENV: str = "development"
    APP_PORT: int = 8005

    # API Keys
    OPENAI_API_KEY: str
    COHERE_API_KEY: str = ""  # Optional for reranking
    LLAMA_CLOUD_API_KEY: str = ""  # Optional for LlamaParse
    
    # Vector Database
    PINECONE_API_KEY: str
    PINECONE_ENV: str
    PINECONE_INDEX_NAME: str

    # LLM Configuration (can be changed easily)
    RAG_LLM_API_KEY: str
    RAG_LLM_PROVIDER: str = "openai"  # openai, anthropic, google, etc.
    RAG_LLM_MODEL: str = "gpt-4o-mini"  # Model name

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

# Validate required API keys on startup
def validate_required_keys():
    """Validate that all required API keys are present"""
    required_keys = [
        ("OPENAI_API_KEY", settings.OPENAI_API_KEY),
        ("PINECONE_API_KEY", settings.PINECONE_API_KEY),
        ("RAG_LLM_API_KEY", settings.RAG_LLM_API_KEY),
    ]
    
    missing_keys = []
    for key_name, key_value in required_keys:
        if not key_value or key_value.strip() == "":
            missing_keys.append(key_name)
    
    if missing_keys:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_keys)}. "
            f"Please check your .env file."
        )

# Validate on import (will catch issues early)
validate_required_keys()
