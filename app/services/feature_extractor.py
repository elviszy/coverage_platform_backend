"""功能点提取服务

使用 LLM 从需求文档中提取功能点，每个功能点包含标题、描述和原文片段。
"""
from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional

import openai

from app.config import get_settings


settings = get_settings()


def extract_feature_points(text: str) -> str:
    """从需求文本中提取需求点和功能点，返回 Markdown 格式文本。
    
    Args:
        text: 需求文档文本
        
    Returns:
        Markdown 格式的需求点和功能点列表
    """
    if not text or not text.strip():
        return ""
    
    # 文本太短则不提取
    if len(text.strip()) < 20:
        return ""
    
    try:
        return _extract_with_llm(text)
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"[FeatureExtractor] ❌ LLM 提取严重失败:\n{error_msg}")
        # 回退到简单规则提取
        print("[FeatureExtractor] ⚠️ 正在尝试回退到规则提取...")
        return _extract_with_rules(text)


def _extract_with_llm(text: str) -> str:
    """使用 LLM 提取功能点。"""
    # 优先使用 feature_extractor 配置，如果没有则回退到 openai 配置
    api_key = settings.feature_extractor_api_key or settings.openai_api_key
    base_url = settings.feature_extractor_base_url or settings.openai_base_url
    model = settings.feature_extractor_model or settings.openai_model_smart or "gpt-4"
    
    print(f"[FeatureExtractor] 准备调用 LLM。API配置: base_url={base_url}, model={model}, has_key={bool(api_key)}")
    
    if not api_key:
        print("[FeatureExtractor] ❌ 错误：无可用的 API Key，将被迫回退到规则提取！")
        return _extract_with_rules(text)
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    # 对于长文本，分段处理
    MAX_CHUNK_LEN = 15000
    if len(text) > MAX_CHUNK_LEN:
        print(f"[FeatureExtractor] 文本过长({len(text)}字符)，分段处理")
        return _extract_from_chunks(client, model, text, MAX_CHUNK_LEN)
    
    text_to_process = text
    
    prompt = f"""你是一位资深的软件测试专家。请仔细阅读以下需求文档，提取所有可测试的功能点并输出为 Markdown 格式。

## 提取规则
1. 每个功能点必须是一个独立的、可测试的功能单元
2. 标题用"动词+名词"形式概括（如：新增调价单、删除记录、查询列表）
3. 描述要用自己的话归纳总结，说明用户做什么、系统如何响应
4. 原文依据要引用需求文档中的关键语句

## 输出格式
请严格按以下 Markdown 格式输出，不要有任何其他内容：

### 1. [功能点标题]
**描述**：[用一句话描述该功能的具体行为]
**原文**：[引用需求文档中与该功能相关的关键语句]

### 2. [功能点标题]
**描述**：[用一句话描述该功能的具体行为]
**原文**：[引用需求文档中与该功能相关的关键语句]

## 示例输出

### 1. 新增调价单
**描述**：用户点击新增按钮，系统打开新页签进入调价单创建页面
**原文**：点击"新增"按钮时跳转至新开页签【新建采购员调价单】

### 2. 查询调价单
**描述**：用户输入筛选条件后点击查询按钮，系统根据条件过滤并展示列表数据
**原文**：输入筛选项后，点击"查询"按钮则可对该tab数据进行搜索

### 3. 删除调价单
**描述**：用户点击删除按钮，系统弹出确认弹窗，确认后移除该条数据
**原文**：点击时弹出二次确认弹窗"确认要删除该条采购员调价单？"，确认删除时该条数据需移除

---
## 需求文档
{text_to_process}"""

    print(f"[FeatureExtractor] 调用 LLM 提取功能点，文本长度: {len(text_to_process)}")
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.2,
    )
    
    content = response.choices[0].message.content.strip()
    print(f"[FeatureExtractor] LLM 响应长度: {len(content)} 字符")
    
    # 直接返回 Markdown 内容
    return content


def _parse_llm_response(content: str) -> List[Dict[str, Any]]:
    """解析 LLM 响应中的功能点（支持 Markdown 和 JSON 格式）。"""
    feature_points = []
    
    # 首先尝试解析 Markdown 格式
    # 格式：### N. 标题\n**描述**：xxx\n**原文**：xxx
    md_pattern = r'###\s*\d+\.\s*(.+?)\n\*\*描述\*\*[：:]\s*(.+?)\n\*\*原文\*\*[：:]\s*(.+?)(?=\n###|\n---|\Z)'
    matches = re.findall(md_pattern, content, re.DOTALL)
    
    if matches:
        for match in matches:
            title = match[0].strip()
            description = match[1].strip()
            source_excerpt = match[2].strip()
            feature_points.append({
                "title": title,
                "description": description,
                "source_excerpt": source_excerpt,
            })
        print(f"[FeatureExtractor] 成功解析 Markdown 格式，提取到 {len(feature_points)} 个功能点")
        return feature_points
    
    # 尝试解析 JSON 格式（兼容旧格式）
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "feature_points" in data:
            return data["feature_points"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 JSON 块
    json_match = re.search(r'\{[\s\S]*"feature_points"[\s\S]*\}', content)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("feature_points", [])
        except json.JSONDecodeError:
            pass
    
    print("[FeatureExtractor] 无法解析 LLM 响应，返回原始 Markdown")
    # 如果都解析失败，将整个内容作为一个功能点的描述
    if content.strip():
        return [{
            "title": "功能点提取结果",
            "description": "请查看原文描述",
            "source_excerpt": content[:500] if len(content) > 500 else content,
        }]
    return []


def _extract_with_rules(text: str) -> str:
    """使用简单规则提取功能点（兜底方案），返回 Markdown。"""
    lines = []
    
    lines.append("- **模块**: 基础功能")
    lines.append("  - **页面**: 默认页面")
    
    count = 0
    
    # 按句子分割
    sentences = re.split(r'[。！？\n]', text)
    
    # 功能关键词
    func_keywords = ['支持', '提供', '可以', '能够', '允许', '实现', '具备', '应', '需要', '必须']
    
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue
        
        # 检查是否包含功能关键词
        if any(kw in sent for kw in func_keywords):
            count += 1
            # 提取简短标题
            title = sent[:20] + "..." if len(sent) > 20 else sent
            
            lines.append(f"    - **功能点**: {title}")
            lines.append(f"      - **规则**: {sent[:50] if len(sent) > 50 else sent}")
            lines.append(f"      - **原文**: {sent[:80] if len(sent) > 80 else sent}")
            
            # 最多返回 10 个
            if count >= 10:
                break
    
    return "\n".join(lines)


def _extract_from_chunks(client, model: str, text: str, chunk_size: int) -> str:
    """对长文本分段处理，合并所有 Markdown 内容。"""
    all_markdown_parts = []
    
    # 按段落或换行符分割文本
    paragraphs = re.split(r'\n\n+', text)
    
    current_chunk = ""
    chunks = []
    
    for para in paragraphs:
        # 如果加上这段不会超过限制，就加上
        if len(current_chunk) + len(para) + 2 < chunk_size:
            current_chunk += para + "\n\n"
        else:
            # 当前块满了，保存并开始新块
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    # 添加最后一块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    print(f"[FeatureExtractor] 文本分为 {len(chunks)} 个块处理")
    
    # 逐块提取功能点
    for i, chunk in enumerate(chunks):
        print(f"[FeatureExtractor] 处理第 {i+1}/{len(chunks)} 块，长度: {len(chunk)}")
        try:
            prompt = _build_extraction_prompt(chunk)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.2,
            )
            content = response.choices[0].message.content.strip()
            all_markdown_parts.append(content)
        except Exception as e:
            print(f"[FeatureExtractor] 块 {i+1} 处理失败: {e}")
    
    result = "\n".join(all_markdown_parts)
    print(f"[FeatureExtractor] 总共提取 Markdown 长度: {len(result)} 字符")
    return result


def _build_extraction_prompt(text: str) -> str:
    """构建需求点/功能点提取提示词。"""
    return f"""你是一位资深的产品经理(PM)和需求分析师(BA)。请对以下需求文档进行**深度业务逻辑还原**，提取完整的需求点和功能点。

## ⚠️ 核心任务
你的任务是从这篇可能存在排版混乱（尤其是表格结构被破坏）的文档中，**重构业务逻辑链条**。
不要仅仅摘录文字，而是要像整理 XMind 脑图一样，识别出`模块 -> 页面 -> 功能点 -> 具体规则`的层级关系。

## 处理原则
1. **重构表格逻辑**：原文中出现的"序号"、"列名"等可能是表格行被 copy 出来的结果，请根据上下文将其还原为具体的业务规则。
2. **忽略无效信息**：忽略页眉、页脚、单纯的表头（如"序号"、"功能点"、"说明"等无意义词汇）。
3. **层级化输出**：按功能模块组织内容，同一个功能点下的多条规则请聚合在一起。

## 提取结构（XMind 风格嵌套列表）
使用 Markdown 缩进列表表示层级关系：

- **模块**: [业务模块名称]
  - **页面**: [页面名称]
    - **功能点**: [具体功能名称]
      - **规则**: [具体的业务规则/交互/数据要求]
      - **规则**: [另一条规则]
      - **原文**: [简要引用关键原文]

## 示例

- **模块**: 采购调价管理
  - **页面**: 采购调价单列表
    - **功能点**: 列表查询
      - **规则**: 支持按单据编号、单据日期、审核状态进行精确或模糊查询
      - **规则**: 点击"重置"按钮，清空筛选项并重新加载列表
      - **原文**: 输入筛选项后，点击“查询”按钮... 点击“重置”按钮...
    - **功能点**: 列表数据展示
      - **规则**: 默认按创建时间倒序排列
      - **规则**: 支持列表自定义列配置
      - **原文**: 默认根据采购调价单创建时间降序排序...

- **模块**: 基础数据
  - **页面**: 导入校验
    - **功能点**: 批量导入
      - **规则**: 必须校验导入商品的权限，无权限则报错
      - **规则**: 校验商品是否存在重复
      - **原文**: 批量导入调价明细校验：校验导入商品的权限... 商品重复校验...

- **模块**: 基础功能 (常用示例)
  - **页面**: 通用操作
    - **功能点**: 新增调价单
      - **规则**: 用户点击新增按钮，系统打开新页签进入调价单创建页面
      - **原文**: 点击"新增"按钮时跳转至新开页签【新建采购员调价单】
    - **功能点**: 查询调价单
      - **规则**: 用户输入筛选条件后点击查询按钮，系统根据条件过滤并展示列表数据
      - **原文**: 输入筛选项后，点击"查询"按钮则可对该tab数据进行搜索
    - **功能点**: 删除调价单
      - **规则**: 点击删除按钮时弹出二次确认弹窗"确认要删除？"
      - **规则**: 确认删除时该条数据需移除，若取消则不做任何更新
      - **原文**: 点击时弹出二次确认弹窗... 确认删除时该条数据需移除

---
## 需求文档（请开始深度还原）
{text}"""


