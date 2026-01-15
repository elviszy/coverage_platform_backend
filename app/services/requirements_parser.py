from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup


@dataclass
class CriterionRow:
    """解析得到的验收标准（表格级）。"""

    criterion_id: str
    path: str
    table_idx: int
    row_idx: int
    table_title: Optional[str]
    headers: List[str]
    row_data: Dict[str, Any]
    normalized_text: str


def _clean_text(x: str) -> str:
    """清理文本：合并空白字符。"""
    return " ".join((x or "").split()).strip()


def _extract_table_content(table_tag) -> tuple[list[str], list[dict], str]:
    """从 table 标签中提取表头、数据行和完整文本。
    
    Returns:
        headers: 表头列表
        row_dicts: 数据行字典列表
        full_text: 完整表格文本（用于 normalized_text）
    """
    rows = table_tag.find_all("tr")
    if not rows:
        return [], [], ""

    def cells_text(tr):
        cells = tr.find_all(["th", "td"])
        return [_clean_text(c.get_text(" ", strip=True)) for c in cells]

    header_cells = cells_text(rows[0])
    headers = [h if h else f"列{i+1}" for i, h in enumerate(header_cells)]

    data_rows: list[dict] = []
    all_rows_text: list[str] = []
    
    # 表头文本
    all_rows_text.append(" | ".join(headers))
    
    for tr in rows[1:]:
        vals = cells_text(tr)
        if not any(v.strip() for v in vals):
            continue
        row_dict: dict = {}
        for i, h in enumerate(headers):
            row_dict[h] = vals[i] if i < len(vals) else ""
        data_rows.append(row_dict)
        # 每行文本
        all_rows_text.append(" | ".join(vals))

    full_text = "\n".join(all_rows_text)
    return headers, data_rows, full_text


def parse_storage_to_criteria(
    *,
    page_id: str,
    page_url: str,
    title: str,
    page_version: int,
    storage_html: str,
    path: str = "",
) -> list[CriterionRow]:
    """从 Confluence body.storage 中抽取验收标准。

    解析策略（表格级切分）：
    - 每个表格作为 1 条记录存储
    - normalized_text 包含完整表格内容（所有行）
    - 保留语义完整性，后续由 LLM 精确定位需求点
    """
    doc_title = title

    soup = BeautifulSoup(storage_html or "", "lxml")

    heading_stack: list[tuple[int, str]] = []

    def current_path() -> str:
        parts = [t for _, t in heading_stack if t]
        full = ([path] if path else []) + parts
        return "/".join([p for p in full if p])

    results: list[CriterionRow] = []

    table_idx = -1

    # 顺序遍历所有重要元素
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "table"]):
        if el.name and el.name.startswith("h"):
            try:
                level = int(el.name[1:])
            except Exception:
                level = 6
            heading_title = _clean_text(el.get_text(" ", strip=True))
            if not heading_title:
                continue

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_title))
            continue

        if el.name == "table":
            table_idx += 1
            headers, row_dicts, full_text = _extract_table_content(el)
            if not headers or not row_dicts:
                continue

            cur_path = current_path()
            table_title = heading_stack[-1][1] if heading_stack else doc_title
            
            # 表格级切分：整个表格作为 1 条记录
            normalized_text = _clean_text(
                f"【页面】{doc_title}【路径】{cur_path}【章节】{table_title}【表格内容】\n{full_text}"
            )

            results.append(
                CriterionRow(
                    criterion_id=f"{page_id}:{page_version}:t:{table_idx}",
                    path=cur_path,
                    table_idx=table_idx,
                    row_idx=0,  # 表格级切分，row_idx 固定为 0
                    table_title=table_title,
                    headers=headers,
                    row_data={"_all_rows": row_dicts},  # 存储所有行数据
                    normalized_text=normalized_text,
                )
            )

    # 兜底：如果页面中没有表格，则按段落切分全文
    if not results:
        raw_text = soup.get_text("\n", strip=True)
        lines = [ln.strip() for ln in (raw_text or "").replace("\r", "").split("\n")]

        paras: list[str] = []
        buf: list[str] = []
        for ln in lines:
            if not ln:
                if buf:
                    paras.append(_clean_text(" ".join(buf)))
                    buf = []
                continue
            buf.append(ln)
        if buf:
            paras.append(_clean_text(" ".join(buf)))

        if not paras:
            paras = [""]

        for i, para in enumerate(paras):
            base_path = path
            normalized_text = _clean_text(
                f"【页面】{doc_title}【路径】{base_path}【段落】{para}"
            )
            results.append(
                CriterionRow(
                    criterion_id=f"{page_id}:{page_version}:p:{i}:{uuid.uuid4().hex[:8]}",
                    path=base_path,
                    table_idx=0,
                    row_idx=i,
                    table_title=doc_title,
                    headers=["text"],
                    row_data={"text": para},
                    normalized_text=normalized_text,
                )
            )

    return results

