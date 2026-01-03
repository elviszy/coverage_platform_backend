"""公共测试标准管理服务。

提供公共测试标准的导入、CRUD、关键词提取和 Embedding 索引功能。
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.models import PublicTestCriterion
from app.services.embedding import embed_text


logger = logging.getLogger(__name__)


# ==================== 关键词库 ====================

# 预定义的关键词库（按测试类型分类）- 作为 LLM 提取失败时的备选
KEYWORD_PATTERNS: dict[str, list[str]] = {
    "选择客户/商品": ["选择", "客户", "商品", "搜索", "选中", "带出", "返回"],
    "增删改": ["新增", "保存", "编辑", "修改", "删除", "刷新", "落表", "多次点击", "双开", "提交"],
    "审核": ["审核", "通过", "驳回", "批量审核", "工作流", "审核状态", "审批"],
    "查询": ["查询", "搜索", "翻页", "分页", "排序", "导出", "筛选", "列表", "刷新"],
    "校验": ["校验", "验证", "必填", "为空", "重复", "拦截", "提示", "格式"],
    "导入": ["导入", "上传", "模板", "批量", "格式", "excel"],
    "数值": ["金额", "单价", "数量", "小数", "精度", "四舍五入", "计算"],
    "其他通用": ["取消", "确认", "权限", "日志", "超时", "并发"],
}


def extract_keywords_with_llm(text: str, category: str) -> List[str]:
    """
    使用 LLM 从文本中提取关键词。
    
    Args:
        text: 待提取的文本
        category: 测试类型
        
    Returns:
        关键词列表
    """
    from app.config import get_settings
    settings = get_settings()
    
    if not settings.openai_api_key:
        logger.debug("未配置 openai_api_key，跳过 LLM 关键词提取")
        return []
    
    try:
        from openai import OpenAI
        
        print(f"[LLM] 开始提取关键词 - 类型: {category}, 文本: {text[:50]}...")
        print(f"[LLM] API Base URL: {settings.openai_base_url}")
        print(f"[LLM] 模型: {settings.openai_model_quality}")
        
        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        
        prompt = f"""请从以下测试用例描述中提取关键词，用于后续的测试覆盖度匹配。

测试类型：{category}
测试描述：{text}

要求：
1. 提取 5-10 个最能代表这个测试点的关键词
2. 关键词应该是具体的功能点、操作动作、业务术语
3. 避免提取过于通用的词（如"测试"、"检查"、"是否"）
4. 每个关键词用逗号分隔，直接输出关键词列表，不要解释

输出格式示例：新增,保存,响应时间,单据,3秒"""

        response = client.chat.completions.create(
            model=settings.openai_model_quality,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        
        result = response.choices[0].message.content.strip()
        print(f"[LLM] 原始响应: {result}")
        
        # 解析关键词
        keywords = [kw.strip() for kw in result.split(",") if kw.strip()]
        
        print(f"[LLM] 提取关键词成功: {keywords}")
        return keywords
        
    except Exception as e:
        print(f"[LLM] 提取关键词失败: {e}")
        return []


def extract_keywords_fallback(text: str, category: str) -> List[str]:
    """
    使用 jieba 分词从文本中提取关键词（备选方案）。
    """
    try:
        import jieba
        import jieba.analyse
    except ImportError:
        logger.warning("jieba 未安装，使用简单正则匹配")
        # 回退到简单正则
        words = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
        return list(set(words))[:10]
    
    keywords = set()
    
    # 使用 jieba 的 TF-IDF 提取关键词
    tfidf_keywords = jieba.analyse.extract_tags(text, topK=15, withWeight=False)
    keywords.update(tfidf_keywords)
    
    # 使用 TextRank 提取关键词（补充）
    textrank_keywords = jieba.analyse.textrank(text, topK=10, withWeight=False)
    keywords.update(textrank_keywords)
    
    # 从预定义库获取匹配的关键词
    for cat, kw_list in KEYWORD_PATTERNS.items():
        if cat in category or category in cat:
            for kw in kw_list:
                if kw in text:
                    keywords.add(kw)
    
    # 扩展的停用词列表
    stop_words = {
        "进行", "操作", "情况", "是否", "需要", "可以", "或者", "以及", "如果",
        "正常", "检查", "确认", "以后", "之后", "之前", "时候", "测试", "功能",
        "相关", "相应", "对应", "比如", "例如", "注意", "要求", "应该", "能够",
        "已经", "然后", "这个", "那个", "什么", "怎么", "如何", "所有", "其他",
    }
    
    # 过滤停用词和过短的词
    keywords = {kw for kw in keywords if kw not in stop_words and len(kw) >= 2}
    
    return list(keywords)


def extract_keywords(text: str, category: str, use_llm: bool = True) -> List[str]:
    """
    从文本中提取关键词。
    
    策略：
    1. 优先使用 LLM 提取（更准确）
    2. LLM 失败时回退到规则提取
    
    Args:
        text: 待提取的文本
        category: 测试类型
        use_llm: 是否使用 LLM（默认 True）
        
    Returns:
        关键词列表（去重）
    """
    keywords = []
    
    if use_llm:
        keywords = extract_keywords_with_llm(text, category)
    
    # 如果 LLM 提取失败或关键词太少，使用规则提取补充
    if len(keywords) < 3:
        fallback_keywords = extract_keywords_fallback(text, category)
        # 合并结果
        keywords = list(set(keywords + fallback_keywords))
    
    return keywords


def generate_criterion_id(category: str, test_point: str) -> str:
    """
    根据类型和测试点生成唯一 ID。
    
    格式：pc_{hash[:12]}
    """
    content = f"{category}|{test_point}"
    hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"pc_{hash_val}"


def normalize_text(category: str, test_point: str, test_content: Optional[str]) -> str:
    """
    生成规范化文本（用于 Embedding）。
    
    格式：{类型} | {测试点} | {测试内容}
    """
    parts = [category, test_point]
    if test_content:
        parts.append(test_content)
    return " | ".join(parts)


# ==================== Markdown 解析 ====================


def parse_markdown_table(content: str) -> List[dict]:
    """
    解析 Markdown 表格格式的公共测试标准。
    
    支持格式：
    类型	测试点	测试内容
    选择客户/商品	正常情况	选择客户落到主界面后，各字段值显示是否正确
    
    也支持标准 Markdown 表格：
    | 类型 | 测试点 | 测试内容 |
    |------|--------|----------|
    | 选择客户/商品 | 正常情况 | ... |
    
    Returns:
        解析后的记录列表 [{"category": ..., "test_point": ..., "test_content": ...}, ...]
    """
    results = []
    lines = content.strip().split('\n')
    
    # 跳过空行和标题行
    data_lines = []
    headers_found = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 跳过分隔行（如 |---|---|---|）
        if re.match(r'^[\|\-\s:]+$', line):
            continue
        
        # 检测是否为表头行
        if not headers_found and ('类型' in line or '测试点' in line):
            headers_found = True
            continue
        
        if headers_found:
            data_lines.append(line)
    
    # 解析数据行
    for line in data_lines:
        # 移除首尾的 |
        line = line.strip('|').strip()
        
        # 支持 tab 分隔或 | 分隔
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
        else:
            parts = [p.strip() for p in line.split('\t')]
        
        if len(parts) >= 2:
            record = {
                "category": parts[0],
                "test_point": parts[1],
                "test_content": parts[2] if len(parts) > 2 else None,
            }
            # 跳过空记录
            if record["category"] and record["test_point"]:
                results.append(record)
    
    return results


# ==================== CRUD 操作 ====================


def import_criteria(
    db: Session, 
    content: str, 
    replace_all: bool = False,
    use_llm: bool = True,
) -> dict:
    """
    导入公共测试标准。
    
    Args:
        db: 数据库会话
        content: Markdown 内容
        replace_all: 是否替换全部现有数据
        use_llm: 是否使用 LLM 提取关键词（默认 True）
        
    Returns:
        {imported: int, updated: int, skipped: int, errors: list}
    """
    result = {"imported": 0, "updated": 0, "skipped": 0, "errors": []}
    
    # 解析 Markdown
    records = parse_markdown_table(content)
    
    if not records:
        result["errors"].append("未能解析到有效的测试标准记录")
        return result
    
    # 如果需要替换全部，先删除现有数据
    if replace_all:
        db.execute(
            PublicTestCriterion.__table__.delete()
        )
        db.commit()
    
    # 导入记录
    for record in records:
        try:
            criterion_id = generate_criterion_id(record["category"], record["test_point"])
            normalized = normalize_text(record["category"], record["test_point"], record["test_content"])
            keywords = extract_keywords(normalized, record["category"], use_llm=use_llm)
            
            # 检查是否已存在
            existing = db.get(PublicTestCriterion, criterion_id)
            
            if existing:
                if replace_all:
                    # 更新记录
                    existing.category = record["category"]
                    existing.test_point = record["test_point"]
                    existing.test_content = record["test_content"]
                    existing.normalized_text = normalized
                    existing.keywords = keywords
                    existing.is_active = True
                    result["updated"] += 1
                else:
                    result["skipped"] += 1
            else:
                # 新增记录
                new_criterion = PublicTestCriterion(
                    criterion_id=criterion_id,
                    category=record["category"],
                    test_point=record["test_point"],
                    test_content=record["test_content"],
                    normalized_text=normalized,
                    keywords=keywords,
                    is_active=True,
                )
                db.add(new_criterion)
                result["imported"] += 1
                
        except Exception as e:
            result["errors"].append(f"处理记录 {record} 时出错: {str(e)}")
    
    db.commit()
    
    logger.info(f"公共测试标准导入完成: imported={result['imported']}, updated={result['updated']}, skipped={result['skipped']}")
    
    return result


def list_criteria(
    db: Session,
    category: Optional[str] = None,
    is_active: Optional[bool] = True,
    search: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> Tuple[List[PublicTestCriterion], int]:
    """
    列表查询公共测试标准。
    
    Returns:
        (记录列表, 总数)
    """
    query = select(PublicTestCriterion)
    
    if category:
        query = query.where(PublicTestCriterion.category == category)
    
    if is_active is not None:
        query = query.where(PublicTestCriterion.is_active == is_active)
    
    if search:
        search_pattern = f"%{search}%"
        query = query.where(
            (PublicTestCriterion.test_point.ilike(search_pattern)) |
            (PublicTestCriterion.test_content.ilike(search_pattern))
        )
    
    # 获取总数
    count_query = select(func.count()).select_from(query.subquery())
    total = db.execute(count_query).scalar() or 0
    
    # 分页
    query = query.order_by(PublicTestCriterion.category, PublicTestCriterion.test_point)
    query = query.limit(limit).offset(offset)
    
    items = list(db.execute(query).scalars().all())
    
    return items, total


def get_categories(db: Session) -> List[str]:
    """获取所有测试类型。"""
    query = select(PublicTestCriterion.category).distinct()
    query = query.where(PublicTestCriterion.is_active == True)
    query = query.order_by(PublicTestCriterion.category)
    
    return list(db.execute(query).scalars().all())


def get_criterion(db: Session, criterion_id: str) -> Optional[PublicTestCriterion]:
    """根据 ID 获取测试标准。"""
    return db.get(PublicTestCriterion, criterion_id)


def update_criterion(
    db: Session, 
    criterion_id: str, 
    data: dict
) -> Optional[PublicTestCriterion]:
    """
    更新公共测试标准。
    
    Args:
        db: 数据库会话
        criterion_id: 标准 ID
        data: 更新字段 {category?, test_point?, test_content?, is_active?}
        
    Returns:
        更新后的记录，如果不存在返回 None
    """
    criterion = db.get(PublicTestCriterion, criterion_id)
    if not criterion:
        return None
    
    # 更新字段
    if "category" in data and data["category"]:
        criterion.category = data["category"]
    
    if "test_point" in data and data["test_point"]:
        criterion.test_point = data["test_point"]
    
    if "test_content" in data:
        criterion.test_content = data["test_content"]
    
    if "is_active" in data:
        criterion.is_active = data["is_active"]
    
    # 重新生成规范化文本和关键词
    criterion.normalized_text = normalize_text(
        criterion.category, 
        criterion.test_point, 
        criterion.test_content
    )
    criterion.keywords = extract_keywords(criterion.normalized_text, criterion.category)
    
    # 清除 embedding（需要重新索引）
    criterion.embedding = None
    
    db.commit()
    db.refresh(criterion)
    
    return criterion


def delete_criterion(db: Session, criterion_id: str) -> bool:
    """
    删除公共测试标准（硬删除）。
    
    Returns:
        是否删除成功
    """
    criterion = db.get(PublicTestCriterion, criterion_id)
    if not criterion:
        return False
    
    db.delete(criterion)
    db.commit()
    
    return True


# ==================== Embedding 索引 ====================


def index_criteria(db: Session, force: bool = False) -> dict:
    """
    为公共测试标准生成 Embedding 索引（批量处理优化）。
    
    Args:
        db: 数据库会话
        force: 是否强制重建所有索引
        
    Returns:
        {indexed: int, failed: int}
    """
    from app.services.embedding import embed_texts_batch
    
    result = {"indexed": 0, "failed": 0}
    
    # 查询需要索引的记录
    query = select(PublicTestCriterion).where(PublicTestCriterion.is_active == True)
    
    if not force:
        # 仅索引没有 embedding 的记录
        query = query.where(PublicTestCriterion.embedding == None)
    
    criteria = list(db.execute(query).scalars().all())
    
    if not criteria:
        print("[索引] 没有需要索引的记录")
        return result
    
    print(f"[索引] 开始批量索引 {len(criteria)} 条公共测试标准...")
    
    # 收集所有文本
    texts = [c.normalized_text for c in criteria]
    
    # 批量生成 embedding
    try:
        embeddings = embed_texts_batch(texts, batch_size=20)
        
        # 更新记录
        for criterion, embedding in zip(criteria, embeddings):
            criterion.embedding = embedding
            result["indexed"] += 1
        
        db.commit()
        
    except Exception as e:
        print(f"[索引] 批量处理失败，回退到逐条处理: {e}")
        # 回退到逐条处理
        for criterion in criteria:
            try:
                from app.services.embedding import embed_text
                embedding = embed_text(criterion.normalized_text)
                criterion.embedding = embedding
                result["indexed"] += 1
            except Exception as e2:
                print(f"[索引] {criterion.criterion_id} 失败: {e2}")
                result["failed"] += 1
        
        db.commit()
    
    print(f"[索引] 完成: indexed={result['indexed']}, failed={result['failed']}")
    
    return result


def get_criteria_with_embeddings(
    db: Session,
    category: Optional[str] = None,
) -> List[PublicTestCriterion]:
    """
    获取所有已索引的公共测试标准（用于覆盖度分析）。
    
    Args:
        db: 数据库会话
        category: 限定类型（可选）
        
    Returns:
        已索引的测试标准列表
    """
    query = select(PublicTestCriterion).where(
        PublicTestCriterion.is_active == True,
        PublicTestCriterion.embedding != None,
    )
    
    if category:
        query = query.where(PublicTestCriterion.category == category)
    
    query = query.order_by(PublicTestCriterion.category, PublicTestCriterion.test_point)
    
    return list(db.execute(query).scalars().all())
