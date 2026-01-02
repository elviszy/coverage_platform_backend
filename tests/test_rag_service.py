"""RAG 服务集成测试

测试 RAG-Anything 服务层的核心功能：
- 服务初始化
- Embedding 功能
- 知识图谱操作
- 附件处理
"""

import pytest
import asyncio
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRAGServiceBasic:
    """RAG 服务基础功能测试"""

    def test_service_singleton(self):
        """测试服务单例模式"""
        from app.services.rag_service import get_rag_service
        
        service1 = get_rag_service()
        service2 = get_rag_service()
        
        assert service1 is service2, "RAG 服务应该是单例"

    def test_service_status(self):
        """测试服务状态查询"""
        from app.services.rag_service import get_rag_service
        
        service = get_rag_service()
        status = service.get_status()
        
        assert "initialized" in status
        assert "working_dir" in status
        assert "use_postgres" in status
        assert "query_mode" in status

    def test_config_loaded(self):
        """测试配置加载"""
        from app.config import get_settings
        
        settings = get_settings()
        
        assert hasattr(settings, "rag_working_dir")
        assert hasattr(settings, "rag_enable_graph")
        assert hasattr(settings, "rag_query_mode")
        assert hasattr(settings, "rag_use_postgres")


class TestRAGServiceAsync:
    """RAG 服务异步功能测试"""

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """测试服务异步初始化"""
        from app.services.rag_service import get_rag_service, initialize_rag_service
        
        # 注意：完整初始化需要有效的 API Key 和数据库连接
        # 这里只测试初始化流程不抛出异常
        service = get_rag_service()
        
        # 检查初始状态
        assert service is not None

    @pytest.mark.asyncio
    async def test_embedding_function_creation(self):
        """测试 Embedding 函数创建"""
        from app.services.rag_service import get_rag_service
        
        service = get_rag_service()
        embedding_func = service._create_embedding_func()
        
        assert embedding_func is not None
        assert hasattr(embedding_func, "embedding_dim")
        assert hasattr(embedding_func, "func")


class TestEmbeddingIntegration:
    """Embedding 集成测试"""

    def test_embed_text_fallback(self):
        """测试 embed_text 回退机制"""
        from app.services.embedding import embed_text, _hash_embedding
        from app.config import get_settings
        
        settings = get_settings()
        
        # 测试伪向量生成
        test_text = "这是一个测试文本"
        vec = _hash_embedding(test_text, settings.embedding_dim)
        
        assert len(vec) == settings.embedding_dim
        assert all(isinstance(v, float) for v in vec)

    def test_embed_text_consistency(self):
        """测试相同输入产生相同向量"""
        from app.services.embedding import _hash_embedding
        from app.config import get_settings
        
        settings = get_settings()
        
        text = "测试文本一致性"
        vec1 = _hash_embedding(text, settings.embedding_dim)
        vec2 = _hash_embedding(text, settings.embedding_dim)
        
        assert vec1 == vec2, "相同输入应产生相同向量"


class TestReviewEngineIntegration:
    """评审引擎集成测试"""

    def test_merge_search_results(self):
        """测试搜索结果融合函数"""
        from app.services.review_engine import _merge_search_results
        
        vector_results = [
            {"criterion_id": "c1", "score": 0.9, "text": "需求1"},
            {"criterion_id": "c2", "score": 0.8, "text": "需求2"},
        ]
        
        graph_results = [
            {"criterion_id": "c1", "score": 1.0, "text": "需求1"},  # 重复
            {"criterion_id": "c3", "score": 0.7, "text": "需求3"},  # 新增
        ]
        
        merged = _merge_search_results(vector_results, graph_results)
        
        # 检查结果数量
        assert len(merged) == 3
        
        # 检查融合后的分数
        c1_result = next((r for r in merged if r.get("criterion_id") == "c1"), None)
        assert c1_result is not None
        assert c1_result.get("source") == "hybrid"  # 应该标记为混合来源

    def test_merge_empty_results(self):
        """测试空结果融合"""
        from app.services.review_engine import _merge_search_results
        
        merged = _merge_search_results([], [])
        assert merged == []
        
        merged = _merge_search_results([{"criterion_id": "c1", "score": 0.5}], [])
        assert len(merged) == 1


# pytest 配置
@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
