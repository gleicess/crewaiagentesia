"""
Configurações principais do sistema de agentes CSV
"""
import os
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

class Settings(BaseSettings):
    """Configurações do aplicativo"""

    # API Keys
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")

    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_anon_key: str = Field(..., env="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")

    # CrewAI
    crewai_storage_dir: str = Field("./crewai_storage", env="CREWAI_STORAGE_DIR")
    crewai_log_level: str = Field("INFO", env="CREWAI_LOG_LEVEL")

    # Streamlit
    streamlit_host: str = Field("localhost", env="STREAMLIT_HOST")
    streamlit_port: int = Field(8501, env="STREAMLIT_PORT")
    streamlit_theme_base: str = Field("light", env="STREAMLIT_THEME_BASE")

    # Memory
    memory_namespace: str = Field("csv_agent_memories", env="MEMORY_NAMESPACE")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(1536, env="EMBEDDING_DIMENSIONS")

    # Analysis
    max_csv_size_mb: int = Field(50, env="MAX_CSV_SIZE_MB")
    max_analysis_iterations: int = Field(5, env="MAX_ANALYSIS_ITERATIONS")
    default_plot_width: int = Field(800, env="DEFAULT_PLOT_WIDTH")
    default_plot_height: int = Field(600, env="DEFAULT_PLOT_HEIGHT")

    # Database
    database_url: Optional[str] = Field(None, env="DATABASE_URL")

    class Config:
        env_file = ".env"
        case_sensitive = False

# Instância global das configurações
settings = Settings()

# Modelos LLM disponíveis
LLM_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "local": "ollama"
}

# Configurações de visualização
PLOT_THEMES = {
    "plotly": "plotly",
    "plotly_white": "plotly_white",
    "plotly_dark": "plotly_dark",
    "ggplot2": "ggplot2",
    "seaborn": "seaborn",
    "simple_white": "simple_white"
}

# Tipos de arquivo suportados
SUPPORTED_FILE_TYPES = {
    "csv": ["csv"],
    "excel": ["xlsx", "xls"],
    "json": ["json"],
    "parquet": ["parquet"]
}

# Configurações de memória
MEMORY_CONFIG = {
    "short_term_memory": True,
    "long_term_memory": True,
    "entity_memory": True,
    "contextual_memory": True,
    "max_memories": 1000,
    "similarity_threshold": 0.7
}
