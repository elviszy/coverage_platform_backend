"""RAG-Anything 服务封装层

提供 RAG-Anything 核心功能的封装，包括：
- 知识图谱构建与查询
- 多模态附件处理
- 统一的 Embedding 服务
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_settings

settings = get_settings()


class RAGService:
    """RAG-Anything 服务单例
    
    封装 RAG-Anything 的核心功能，提供：
    - 知识图谱的构建和查询
    - 多模态文档处理
    - 与现有 pgvector 的协同工作
    """

    _instance: Optional["RAGService"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._rag = None
        self._lightrag = None

    async def initialize(self) -> bool:
        """初始化 RAG 实例
        
        Returns:
            bool: 初始化是否成功
        """
        if self._initialized:
            return True

        try:
            from app.raganything import RAGAnything, RAGAnythingConfig
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
            from lightrag.utils import EmbeddingFunc

            # 确保工作目录存在
            os.makedirs(settings.rag_working_dir, exist_ok=True)
            os.makedirs(settings.rag_attachment_output_dir, exist_ok=True)

            # 创建配置
            config = RAGAnythingConfig(
                working_dir=settings.rag_working_dir,
                parser="mineru",
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )

            # 构建 LLM 函数
            llm_func = self._create_llm_func()
            vision_func = self._create_vision_func()
            embedding_func = self._create_embedding_func()

            # 初始化 RAGAnything
            if settings.rag_use_postgres:
                # 使用 PostgreSQL 存储
                from lightrag import LightRAG
                
                # 解析数据库连接信息
                db_config = self._parse_database_url()
                
                self._lightrag = LightRAG(
                    working_dir=settings.rag_working_dir,
                    llm_model_func=llm_func,
                    embedding_func=embedding_func,
                    # PostgreSQL 存储配置
                    kv_storage="PGKVStorage",
                    vector_storage="PGVectorStorage",
                    graph_storage="PGGraphStorage",
                    doc_status_storage="PGDocStatusStorage",
                    # 数据库连接参数
                    **db_config,
                )
                
                # 初始化存储
                await self._lightrag.initialize_storages()
                
                # 使用已初始化的 LightRAG 创建 RAGAnything
                self._rag = RAGAnything(
                    lightrag=self._lightrag,
                    vision_model_func=vision_func,
                )
            else:
                # 使用默认文件存储
                self._rag = RAGAnything(
                    config=config,
                    llm_model_func=llm_func,
                    vision_model_func=vision_func,
                    embedding_func=embedding_func,
                )

            self._initialized = True
            return True

        except Exception as e:
            print(f"[RAGService] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _parse_database_url(self) -> Dict[str, Any]:
        """解析数据库 URL 为 LightRAG 配置参数"""
        # 格式: postgresql+psycopg://user:password@host:port/database
        url = settings.database_url
        
        # 移除驱动前缀
        if "://" in url:
            url = url.split("://", 1)[1]
        
        # 解析用户名密码
        if "@" in url:
            auth, hostpart = url.rsplit("@", 1)
            if ":" in auth:
                user, password = auth.split(":", 1)
            else:
                user, password = auth, ""
        else:
            user, password = "postgres", ""
            hostpart = url
        
        # 解析主机端口数据库
        if "/" in hostpart:
            hostport, database = hostpart.rsplit("/", 1)
        else:
            hostport, database = hostpart, "coverage_platform"
        
        if ":" in hostport:
            host, port = hostport.split(":", 1)
            port = int(port)
        else:
            host, port = hostport, 5432
        
        return {
            "pg_url": f"postgresql://{user}:{password}@{host}:{port}/{database}",
            "embedding_dim": settings.embedding_dim,
        }

    def _create_llm_func(self):
        """创建 LLM 函数"""
        from lightrag.llm.openai import openai_complete_if_cache

        def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                settings.openai_model_verifier,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                **kwargs,
            )

        return llm_func

    def _create_vision_func(self):
        """创建视觉模型函数（用于图像处理）"""
        from lightrag.llm.openai import openai_complete_if_cache

        def vision_func(
            prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
        ):
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    **kwargs,
                )
            elif image_data:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                                },
                            ],
                        },
                    ],
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    **kwargs,
                )
            else:
                return self._create_llm_func()(prompt, system_prompt, history_messages, **kwargs)

        return vision_func

    def _create_embedding_func(self):
        """创建 Embedding 函数"""
        from lightrag.llm.openai import openai_embed
        from lightrag.utils import EmbeddingFunc

        # 根据维度选择模型
        if settings.embedding_dim == 1536:
            model = "text-embedding-3-small"
        elif settings.embedding_dim == 3072:
            model = "text-embedding-3-large"
        else:
            model = "text-embedding-3-small"

        return EmbeddingFunc(
            embedding_dim=settings.embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model=model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            ),
        )

    # ========== 知识图谱操作 ==========

    async def insert_requirement(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """将需求文本插入知识图谱
        
        Args:
            text: 需求文本内容
            metadata: 元数据（criterion_id, path, table_title 等）
            
        Returns:
            bool: 插入是否成功
        """
        if not self._initialized:
            await self.initialize()

        try:
            lightrag = self._rag.lightrag if self._rag else self._lightrag
            if lightrag:
                await lightrag.ainsert(text)
                return True
            return False
        except Exception as e:
            print(f"[RAGService] 插入需求失败: {e}")
            return False

    async def insert_scenario(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """将测试场景插入知识图谱
        
        Args:
            text: 场景文本内容
            metadata: 元数据
            
        Returns:
            bool: 插入是否成功
        """
        if not self._initialized:
            await self.initialize()

        try:
            lightrag = self._rag.lightrag if self._rag else self._lightrag
            if lightrag:
                await lightrag.ainsert(text)
                return True
            return False
        except Exception as e:
            print(f"[RAGService] 插入场景失败: {e}")
            return False

    async def query(self, query_text: str, mode: Optional[str] = None) -> str:
        """查询知识图谱
        
        Args:
            query_text: 查询文本
            mode: 查询模式 (hybrid, local, global, naive)
            
        Returns:
            str: 查询结果
        """
        if not self._initialized:
            await self.initialize()

        mode = mode or settings.rag_query_mode

        try:
            if self._rag:
                return await self._rag.aquery(query_text, mode=mode)
            elif self._lightrag:
                return await self._lightrag.aquery(query_text, param={"mode": mode})
            return ""
        except Exception as e:
            print(f"[RAGService] 查询失败: {e}")
            return ""

    async def find_related_requirements(
        self, scenario_text: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """为测试场景查找相关需求（使用图谱检索）
        
        Args:
            scenario_text: 测试场景文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 相关需求列表
        """
        if not self._initialized:
            await self.initialize()

        try:
            query = f"找出与以下测试场景相关的需求和验收标准，返回最相关的{top_k}个：\n{scenario_text}"
            result = await self.query(query, mode=settings.rag_query_mode)
            return self._parse_requirements_from_result(result, top_k)
        except Exception as e:
            print(f"[RAGService] 查找相关需求失败: {e}")
            return []

    def _parse_requirements_from_result(self, result: str, top_k: int) -> List[Dict[str, Any]]:
        """从查询结果解析需求列表"""
        # 简单实现：返回原始结果作为单个条目
        if not result:
            return []
        return [{"text": result, "source": "knowledge_graph"}]

    # ========== 附件处理 ==========

    async def process_attachment(self, file_path: str) -> bool:
        """处理附件文件（PDF/Word/图片等）
        
        Args:
            file_path: 附件文件路径
            
        Returns:
            bool: 处理是否成功
        """
        if not settings.rag_enable_attachment_processing:
            return False

        if not self._initialized:
            await self.initialize()

        # 检查文件扩展名
        ext = Path(file_path).suffix.lower()
        supported_exts = settings.rag_supported_attachment_extensions.split(",")
        if ext not in supported_exts:
            print(f"[RAGService] 不支持的文件类型: {ext}")
            return False

        try:
            if self._rag:
                await self._rag.process_document_complete(
                    file_path=file_path,
                    output_dir=settings.rag_attachment_output_dir,
                )
                return True
            return False
        except Exception as e:
            print(f"[RAGService] 处理附件失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def process_attachments_batch(self, file_paths: List[str]) -> Dict[str, bool]:
        """批量处理附件文件
        
        Args:
            file_paths: 附件文件路径列表
            
        Returns:
            Dict[str, bool]: 每个文件的处理结果
        """
        results = {}
        for path in file_paths:
            results[path] = await self.process_attachment(path)
        return results

    # ========== Embedding 服务 ==========

    def get_embedding(self, text: str) -> List[float]:
        """获取文本的 Embedding 向量
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: Embedding 向量
        """
        try:
            embedding_func = self._create_embedding_func()
            result = asyncio.get_event_loop().run_until_complete(
                embedding_func.func([text])
            )
            return result[0] if result else []
        except Exception as e:
            print(f"[RAGService] 获取 Embedding 失败: {e}")
            # 回退到原有的伪向量实现
            from app.services.embedding import _hash_embedding
            return _hash_embedding(text, settings.embedding_dim)

    async def get_embedding_async(self, text: str) -> List[float]:
        """异步获取文本的 Embedding 向量"""
        try:
            embedding_func = self._create_embedding_func()
            result = await embedding_func.func([text])
            return result[0] if result else []
        except Exception as e:
            print(f"[RAGService] 获取 Embedding 失败: {e}")
            from app.services.embedding import _hash_embedding
            return _hash_embedding(text, settings.embedding_dim)

    # ========== 状态查询 ==========

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialized

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "initialized": self._initialized,
            "working_dir": settings.rag_working_dir,
            "use_postgres": settings.rag_use_postgres,
            "query_mode": settings.rag_query_mode,
            "attachment_processing": settings.rag_enable_attachment_processing,
        }


# ========== 服务获取函数 ==========

_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """获取 RAG 服务单例"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


async def initialize_rag_service() -> bool:
    """初始化 RAG 服务（应在应用启动时调用）"""
    service = get_rag_service()
    return await service.initialize()
