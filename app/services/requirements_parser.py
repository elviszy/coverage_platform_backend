from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup


@dataclass
class CriterionRow:
    """解析得到的验收标准行。"""

    criterion_id: str
    path: str
    table_idx: int
    row_idx: int
    table_title: Optional[str]
    headers: List[str]
    row_data: Dict[str, Any]
    normalized_text: str


def _clean_text(x: str) -> str:
    return " ".join((x or "").split()).strip()


def _extract_table_rows(table_tag) -> tuple[list[str], list[dict]]:
    rows = table_tag.find_all("tr")
    if not rows:
        return [], []

    def cells_text(tr):
        # 兼容 th/td
        cells = tr.find_all(["th", "td"])
        return [_clean_text(c.get_text(" ", strip=True)) for c in cells]

    header_cells = cells_text(rows[0])
    headers = [h if h else f"列{i+1}" for i, h in enumerate(header_cells)]

    data_rows: list[dict] = []
    for tr in rows[1:]:
        vals = cells_text(tr)
        if not any(v.strip() for v in vals):
            continue
        row_dict: dict = {}
        for i, h in enumerate(headers):
            row_dict[h] = vals[i] if i < len(vals) else ""
        data_rows.append(row_dict)

    return headers, data_rows


def parse_storage_to_criteria(
    *,
    page_id: str,
    page_url: str,
    title: str,
    page_version: int,
    storage_html: str,
    path: str = "",
) -> list[CriterionRow]:
    """从 Confluence body.storage 中抽取表格行级验收标准。

    解析策略（MVP）：
    - 维护一个标题栈（h1~h6），形成 path。
    - 遍历 DOM，遇到 table 就按当前 path 抽取行级数据。
    - table_title：优先取当前最后一个标题。
    """

    soup = BeautifulSoup(storage_html or "", "lxml")

    heading_stack: list[tuple[int, str]] = []

    def current_path() -> str:
        parts = [t for _, t in heading_stack if t]
        full = ([path] if path else []) + parts
        return "/".join([p for p in full if p])

    results: list[CriterionRow] = []

    table_idx = -1

    # 用顺序遍历所有重要元素，保持上下文
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "table"]):
        if el.name and el.name.startswith("h"):
            try:
                level = int(el.name[1:])
            except Exception:
                level = 6
            title = _clean_text(el.get_text(" ", strip=True))
            if not title:
                continue

            # 栈：比当前 level 大的全部弹出
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        if el.name == "table":
            table_idx += 1
            headers, row_dicts = _extract_table_rows(el)
            if not headers or not row_dicts:
                continue

            path = current_path()
            table_title = heading_stack[-1][1] if heading_stack else title

            for row_idx, row_data in enumerate(row_dicts):
                # 统一文本模板：用于向量检索
                kv = "；".join([f"{k}:{_clean_text(str(v))}" for k, v in row_data.items()])
                normalized_text = _clean_text(
                    f"【页面】{title}【路径】{path}【验收标准表】{table_title}【行】{kv}"
                )

                results.append(
                    CriterionRow(
                        criterion_id=f"{page_id}:{page_version}:t:{table_idx}:{row_idx}",
                        path=path,
                        table_idx=table_idx,
                        row_idx=row_idx,
                        table_title=table_title,
                        headers=headers,
                        row_data=row_data,
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

        # 如果仍为空，给一个占位，避免索引任务完全没有输出
        if not paras:
            paras = [""]

        for i, para in enumerate(paras):
            base_path = path
            normalized_text = _clean_text(
                f"【页面】{title}【路径】{base_path}【段落】{para}"
            )
            results.append(
                CriterionRow(
                    criterion_id=f"{page_id}:{page_version}:p:{i}:{uuid.uuid4().hex[:8]}",
                    path=base_path,
                    table_idx=0,
                    row_idx=i,
                    table_title=title,
                    headers=["text"],
                    row_data={"text": para},
                    normalized_text=normalized_text,
                )
            )

    return results
