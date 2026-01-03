from __future__ import annotations

import hashlib
import math
from typing import List

from app.config import get_settings


settings = get_settings()


def _hash_embedding(text: str, dim: int) -> List[float]:
    """生成确定性的伪向量。

    说明：
    - 便于在未配置 OpenAI Embedding 时跑通端到端流程。
    - 上线环境建议使用真实 embedding（例如 OpenAI text-embedding-3-large）。
    """

    seed = text.encode("utf-8")
    buf = b""
    counter = 0
    while len(buf) < dim * 2:
        counter += 1
        buf += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()

    vec: List[float] = []
    for i in range(dim):
        x = int.from_bytes(buf[i * 2 : i * 2 + 2], "big")
        vec.append((x / 65535.0) * 2.0 - 1.0)

    return vec


def embed_text(text: str) -> List[float]:
    """对文本生成 embedding。

    策略：
    - 优先使用 RAG-Anything 服务的统一 Embedding。
    - 若 RAG 服务不可用，则调用 OpenAI embedding（支持独立配置 embedding API）。
    - 否则退化为伪向量。
    - 对于超长文本（>7500字符），使用 LLM 摘要法处理。
    """
    
    # 空值检查
    if not text or not text.strip():
        print("[Embedding] 警告：输入文本为空，返回零向量")
        return [0.0] * settings.embedding_dim
    
    # 文本长度限制（DashScope API 限制 8192 字符）
    MAX_TEXT_LENGTH = 7500  # 留一些余量
    
    # 如果文本过长，使用 LLM 摘要法
    if len(text) > MAX_TEXT_LENGTH:
        print(f"[Embedding] 文本过长({len(text)}字符)，使用 LLM 摘要法处理")
        summary = _summarize_text_with_llm(text)
        if summary:
            print(f"[Embedding] 摘要生成成功，长度: {len(summary)}字符")
            return _embed_single_text(summary)
        else:
            # 摘要失败，回退到截断
            print("[Embedding] 摘要失败，回退到截断处理")
            text = text[:MAX_TEXT_LENGTH]
    
    return _embed_single_text(text)


def _summarize_text_with_llm(text: str, max_summary_length: int = 3000) -> str:
    """使用 LLM 对长文本生成摘要。"""
    try:
        import openai
        
        # 使用功能点提取专用配置（与 feature_extractor 相同）
        api_key = settings.feature_extractor_api_key
        base_url = settings.feature_extractor_base_url
        model = settings.feature_extractor_model
        
        if not api_key:
            print("[Embedding] 无可用的 API Key，无法生成摘要")
            return ""
        
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        # 只取前 30000 字符发送给 LLM（避免超出 LLM 上下文限制）
        text_for_summary = text[:30000] if len(text) > 30000 else text
        
        prompt = f"""请对以下文档内容生成一份简洁的摘要，提取核心要点和关键信息。
摘要应该：
1. 保留文档的主要主题和关键概念
2. 包含重要的业务规则、流程步骤或技术要求
3. 控制在 {max_summary_length} 字符以内

文档内容：
{text_for_summary}

请直接输出摘要内容，不要添加额外说明："""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3,
        )
        
        summary = response.choices[0].message.content.strip()
        return summary[:max_summary_length] if len(summary) > max_summary_length else summary
        
    except Exception as e:
        print(f"[Embedding] LLM 摘要生成失败: {e}")
        return ""


def _embed_single_text(text: str) -> List[float]:
    """对单个文本块生成 embedding（内部函数）。"""

    # 尝试使用 RAG 服务
    try:
        from app.services.rag_service import get_rag_service
        rag_service = get_rag_service()
        if rag_service.is_initialized:
            return rag_service.get_embedding(text)
    except Exception:
        pass  # RAG 服务不可用，继续使用原有方式

    # 获取 embedding API 配置（优先使用独立配置，否则回退到 OpenAI 配置）
    api_key = settings.embedding_api_key or settings.openai_api_key
    base_url = settings.embedding_base_url or settings.openai_base_url
    
    if api_key:
        # 延迟导入，避免未安装 openai 时影响本地开发（尽管 requirements 已包含）
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 确定模型名称
        if settings.embedding_model:
            model = settings.embedding_model
        elif settings.embedding_dim == 1536:
            model = "text-embedding-3-small"
        elif settings.embedding_dim == 3072:
            model = "text-embedding-3-large"
        else:
            raise ValueError(f"不支持的 embedding_dim：{settings.embedding_dim}（建议使用 1536 或 3072）")

        resp = client.embeddings.create(
            model=model,
            input=text,
            dimensions=settings.embedding_dim,  # 指定维度（DashScope 支持）
        )
        vec = resp.data[0].embedding

        # 维度校验：如果配置与实际不一致，优先抛错，避免后续入库失败/检索异常
        if len(vec) != settings.embedding_dim:
            raise ValueError(
                f"embedding 维度不一致：配置为 {settings.embedding_dim}，实际为 {len(vec)}"
            )
        return vec

    return _hash_embedding(text, settings.embedding_dim)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度（用于无 pgvector 或调试场景）。"""

    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def embed_texts_batch(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """批量生成 embedding（性能优化）。
    
    DashScope API 支持一次请求处理多条文本，大幅减少网络开销。
    
    Args:
        texts: 文本列表
        batch_size: 每批处理数量（DashScope 限制为 25）
        
    Returns:
        embedding 列表，与输入顺序一致
    """
    if not texts:
        return []
    
    api_key = settings.embedding_api_key or settings.openai_api_key
    base_url = settings.embedding_base_url or settings.openai_base_url
    
    if not api_key:
        # 无 API 配置，使用伪向量
        return [_hash_embedding(text, settings.embedding_dim) for text in texts]
    
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 确定模型名称
    if settings.embedding_model:
        model = settings.embedding_model
    elif settings.embedding_dim == 1536:
        model = "text-embedding-3-small"
    elif settings.embedding_dim == 3072:
        model = "text-embedding-3-large"
    else:
        raise ValueError(f"不支持的 embedding_dim：{settings.embedding_dim}")
    
    all_embeddings: List[List[float]] = []
    
    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"[Embedding] 批量处理 {i+1}-{min(i+batch_size, len(texts))}/{len(texts)}...")
        
        try:
            resp = client.embeddings.create(
                model=model,
                input=batch,
                dimensions=settings.embedding_dim,
            )
            
            # 按索引排序（API 返回可能乱序）
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            
            # 维度校验
            for vec in batch_embeddings:
                if len(vec) != settings.embedding_dim:
                    raise ValueError(
                        f"embedding 维度不一致：配置为 {settings.embedding_dim}，实际为 {len(vec)}"
                    )
            
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"[Embedding] 批量处理失败: {e}")
            # 回退到单条处理
            for text in batch:
                try:
                    vec = embed_text(text)
                    all_embeddings.append(vec)
                except Exception:
                    all_embeddings.append(_hash_embedding(text, settings.embedding_dim))
    
    return all_embeddings
