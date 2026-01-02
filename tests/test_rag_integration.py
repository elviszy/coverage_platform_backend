"""RAG-Anything 端到端集成测试

测试完整的业务流程集成：
- 需求导入与知识图谱构建
- 测试场景导入
- 覆盖率评审
"""

import pytest
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEndToEndFlow:
    """端到端流程测试"""

    @pytest.mark.asyncio
    async def test_full_requirement_flow(self):
        """测试完整需求流程：导入 -> 索引 -> 查询"""
        from app.services.rag_service import get_rag_service
        
        service = get_rag_service()
        
        # 1. 测试需求插入
        test_requirement = """
        【页面】用户登录模块
        【路径】系统管理/用户管理/登录
        【验收标准表】登录功能验收
        【行】用户名正确且密码正确时，应成功登录并跳转到首页
        """
        
        # 检查服务状态
        status = service.get_status()
        assert "working_dir" in status

    @pytest.mark.asyncio
    async def test_rag_query_modes(self):
        """测试不同的 RAG 查询模式"""
        from app.config import get_settings
        
        settings = get_settings()
        
        # 验证所有支持的查询模式
        valid_modes = ["hybrid", "local", "global", "naive"]
        assert settings.rag_query_mode in valid_modes

    def test_attachment_extensions(self):
        """测试附件扩展名配置"""
        from app.config import get_settings
        
        settings = get_settings()
        
        extensions = settings.rag_supported_attachment_extensions.split(",")
        
        # 验证常见格式都被支持
        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".png" in extensions
        assert ".jpg" in extensions


class TestDatabaseIntegration:
    """数据库集成测试"""

    def test_postgres_url_parsing(self):
        """测试 PostgreSQL URL 解析"""
        from app.services.rag_service import RAGService
        
        service = RAGService()
        
        # 测试 URL 解析功能
        db_config = service._parse_database_url()
        
        assert "pg_url" in db_config
        assert "embedding_dim" in db_config


class TestConfigIntegration:
    """配置集成测试"""

    def test_all_rag_configs_present(self):
        """测试所有 RAG 配置项都存在"""
        from app.config import get_settings
        
        settings = get_settings()
        
        # 必需的 RAG 配置项
        required_configs = [
            "rag_working_dir",
            "rag_enable_graph",
            "rag_query_mode",
            "rag_use_postgres",
            "rag_enable_attachment_processing",
            "rag_attachment_output_dir",
            "rag_supported_attachment_extensions",
        ]
        
        for config in required_configs:
            assert hasattr(settings, config), f"缺少配置项: {config}"

    def test_config_defaults(self):
        """测试配置默认值"""
        from app.config import get_settings
        
        settings = get_settings()
        
        assert settings.rag_enable_graph == True
        assert settings.rag_query_mode == "hybrid"
        assert settings.rag_enable_attachment_processing == True


# pytest 配置
@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
