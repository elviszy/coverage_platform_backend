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
    - 若 RAG 服务不可用，则调用 OpenAI embedding。
    - 否则退化为伪向量。
    """

    # 尝试使用 RAG 服务
    try:
        from app.services.rag_service import get_rag_service
        rag_service = get_rag_service()
        if rag_service.is_initialized:
            return rag_service.get_embedding(text)
    except Exception:
        pass  # RAG 服务不可用，继续使用原有方式

    if settings.openai_api_key:
        # 延迟导入，避免未安装 openai 时影响本地开发（尽管 requirements 已包含）
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        if settings.embedding_dim == 1536:
            model = "text-embedding-3-small"
        elif settings.embedding_dim == 3072:
            model = "text-embedding-3-large"
        else:
            raise ValueError(f"不支持的 embedding_dim：{settings.embedding_dim}（建议使用 1536 或 3072）")

        resp = client.embeddings.create(
            model=model,
            input=text,
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
