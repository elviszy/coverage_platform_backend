# Coverage Platform 测试框架

本目录包含 Coverage Platform 的测试用例。

## 测试类型

- `test_rag_service.py` - RAG 服务集成测试
- `test_rag_integration.py` - RAG-Anything 与业务流程集成测试

## 运行测试

```bash
cd coverage_platform_backend
pip install pytest pytest-asyncio
pytest tests/ -v
```

## 环境要求

- Python 3.10+
- PostgreSQL (可选，使用文件存储时不需要)
- 已配置的 OpenAI API Key
