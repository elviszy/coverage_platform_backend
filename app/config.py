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
    database_url: str = "postgresql+psycopg://postgres:postgres@10.6.5.201:5432/coverage_platform"

    # 向量配置
    # 环境变量：EMBEDDING_DIM
    embedding_dim: int = 1536

    # OpenAI
    # 环境变量：OPENAI_API_KEY / OPENAI_BASE_URL
    openai_api_key: Optional[str] = ""  # 必须通过环境变量配置
    openai_base_url: Optional[str] = ""  # 必须通过环境变量配置
    openai_model_verifier: str = "gpt-4o"  # LLM 验证模型
    openai_model_quality: str = "gpt-4o"  # 质量评审模型
    
    # Embedding API（可独立配置，不配置则使用 OpenAI 配置）
    # 环境变量：EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL
    embedding_api_key: Optional[str] = ""  # 不设置则使用 openai_api_key
    embedding_base_url: Optional[str] = ""  # 不设置则使用 openai_base_url
    embedding_model: Optional[str] = "text-embedding-3-small"  # Embedding 模型
    
    # 功能点提取 LLM 配置
    # 环境变量：FEATURE_EXTRACTOR_API_KEY / FEATURE_EXTRACTOR_BASE_URL / FEATURE_EXTRACTOR_MODEL
    feature_extractor_api_key: str = ""  # 不设置则使用 openai_api_key
    feature_extractor_base_url: str = ""  # 不设置则使用 openai_base_url
    feature_extractor_model: str = "gpt-4o"  # 功能点提取模型

    # Confluence Data Center
    # 环境变量：CONFLUENCE_BASE_URL / CONFLUENCE_TOKEN
    confluence_base_url: str = ""
    confluence_token: Optional[str] = ""

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
