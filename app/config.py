from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。

    说明：
    - 默认使用环境变量配置；也可通过 COVERAGE_PLATFORM_CONFIG 指定 yaml 配置文件路径。
    - embedding_dim 默认 1536，可通过配置文件或环境变量覆盖。
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # 服务
    api_prefix: str = "/api/v1"

    # 数据库
    # 环境变量：DATABASE_URL
    database_url: str = "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/coverage_platform"

    # 向量配置
    # 环境变量：EMBEDDING_DIM
    embedding_dim: int = 1536

    # OpenAI
    # 环境变量：OPENAI_API_KEY / OPENAI_BASE_URL
    openai_api_key: Optional[str] = "sk-cjbdj1lb6rb4pe1gioqafqcnntl3fi0rzzdmbfsj2t7jvzc4"
    openai_base_url: Optional[str] = "https://api.xiaomimimo.com/v1"
    openai_model_verifier: str = "MiMo-V2-Flash"
    openai_model_quality: str = "MiMo-V2-Flash"

    # Confluence Data Center
    # 环境变量：CONFLUENCE_BASE_URL / CONFLUENCE_TOKEN
    confluence_base_url: str = ""
    confluence_token: Optional[str] = None

    # 默认开关（每次评审 run 仍可覆盖）
    enable_llm_verifier_default: bool = False
    enable_llm_quality_default: bool = False

    # LLM 调用限额（成本保护；每次评审 run 可进一步约束）
    llm_max_verify_per_scenario: int = 5
    llm_max_total_verify_per_run: int = 500

    # RAG-Anything 配置
    # ---
    rag_working_dir: str = "./rag_storage"
    rag_enable_graph: bool = True
    rag_query_mode: str = "hybrid"  # hybrid, local, global, naive
    
    # RAG 存储后端（使用 PostgreSQL）
    rag_use_postgres: bool = True
    
    # 附件处理配置
    rag_enable_attachment_processing: bool = True
    rag_attachment_output_dir: str = "./attachment_output"
    rag_supported_attachment_extensions: str = ".pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.png,.jpg,.jpeg"


@lru_cache
def get_settings() -> Settings:
    return Settings()
