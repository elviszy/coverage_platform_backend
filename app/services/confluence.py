from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from app.config import get_settings


settings = get_settings()


_PAGE_ID_RE = re.compile(r"pageId=(\d+)")


@dataclass
class ConfluencePage:
    """Confluence 页面数据（MVP）。"""

    page_id: str
    page_url: str
    title: str
    version: int
    body_storage: str
    labels: List[str]
    ancestors: List[str]


@dataclass
class ConfluenceAttachment:
    attachment_id: str
    filename: str
    media_type: Optional[str]
    download_url: str


def extract_page_id(page_url: str) -> Optional[str]:
    """从 Confluence 页面链接中提取 pageId。"""

    m = _PAGE_ID_RE.search(page_url)
    if not m:
        return None
    return m.group(1)


def _build_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # 认证方式（MVP）：
    # - 若设置 CONFLUENCE_TOKEN，则使用 Bearer
    # - 如需 basic（用户名 + token），后续再加开关
    token = settings.confluence_token
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def _normalize_base_url(base_url: Optional[str] = None) -> str:
    base = (base_url or settings.confluence_base_url or "").strip()
    if not base:
        raise ValueError("未配置 CONFLUENCE_BASE_URL")

    if not base.startswith("http://") and not base.startswith("https://"):
        base = "https://" + base

    if not settings.confluence_token:
        raise ValueError("未配置 CONFLUENCE_TOKEN")

    return base


def fetch_page_by_id(page_id: str, base_url: Optional[str] = None) -> ConfluencePage:
    """通过 Confluence Data Center REST API 拉取页面。"""

    base = _normalize_base_url(base_url)

    url = base.rstrip("/") + f"/rest/api/content/{page_id}"
    params = {
        "expand": "body.storage,version,metadata.labels,ancestors",
    }

    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params, headers=_build_headers())
        resp.raise_for_status()
        data = resp.json()

    title = data.get("title") or ""
    version = int((data.get("version") or {}).get("number") or 0)
    body_storage = ((data.get("body") or {}).get("storage") or {}).get("value") or ""

    labels = []
    label_results = (((data.get("metadata") or {}).get("labels") or {}).get("results")) or []
    for it in label_results:
        name = it.get("name")
        if isinstance(name, str) and name:
            labels.append(name)

    ancestors_titles: List[str] = []
    for anc in data.get("ancestors") or []:
        t = anc.get("title")
        if isinstance(t, str) and t:
            ancestors_titles.append(t)

    page_url = base.rstrip("/") + f"/pages/viewpage.action?pageId={page_id}"

    return ConfluencePage(
        page_id=str(page_id),
        page_url=page_url,
        title=title,
        version=version,
        body_storage=body_storage,
        labels=labels,
        ancestors=ancestors_titles,
    )


def fetch_child_page_ids(
    parent_page_id: str,
    base_url: Optional[str] = None,
    limit: int = 50,
) -> List[str]:
    """获取某个页面的子页面 ID 列表（分页）。

    说明：
    - Confluence Data Center REST API：/rest/api/content/{id}/child/page
    - 返回 results[].id
    """

    base = _normalize_base_url(base_url)
    url = base.rstrip("/") + f"/rest/api/content/{parent_page_id}/child/page"

    start = 0
    ids: List[str] = []

    with httpx.Client(timeout=30) as client:
        while True:
            resp = client.get(
                url,
                params={"limit": limit, "start": start},
                headers=_build_headers(),
            )
            resp.raise_for_status()
            data = resp.json() or {}

            results = data.get("results") or []
            if not isinstance(results, list):
                results = []

            for it in results:
                if isinstance(it, dict) and it.get("id"):
                    ids.append(str(it["id"]))

            size = data.get("size")
            if isinstance(size, int) and size == 0:
                break

            # Data Center 通常返回 limit/size/start，也可能给 _links.next
            if len(results) < limit:
                break

            start += limit

    return ids


def fetch_attachments(
    page_id: str,
    base_url: Optional[str] = None,
    limit: int = 50,
) -> List[ConfluenceAttachment]:
    base = _normalize_base_url(base_url)
    url = base.rstrip("/") + f"/rest/api/content/{page_id}/child/attachment"

    start = 0
    items: List[ConfluenceAttachment] = []

    with httpx.Client(timeout=60) as client:
        while True:
            resp = client.get(
                url,
                params={
                    "limit": limit,
                    "start": start,
                    "expand": "metadata,extensions",
                },
                headers=_build_headers(),
            )
            resp.raise_for_status()
            data = resp.json() or {}

            results = data.get("results") or []
            if not isinstance(results, list):
                results = []

            for it in results:
                if not isinstance(it, dict):
                    continue
                aid = it.get("id")
                title = it.get("title")
                links = it.get("_links") or {}
                download = links.get("download") if isinstance(links, dict) else None
                if not aid or not title or not download:
                    continue
                download_url = base.rstrip("/") + str(download)
                media_type = None
                exts = it.get("extensions")
                if isinstance(exts, dict) and isinstance(exts.get("mediaType"), str):
                    media_type = exts.get("mediaType")

                items.append(
                    ConfluenceAttachment(
                        attachment_id=str(aid),
                        filename=str(title),
                        media_type=media_type,
                        download_url=download_url,
                    )
                )

            if len(results) < limit:
                break
            start += limit

    return items


def download_attachment(
    attachment: ConfluenceAttachment,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join([c if c not in "\\/:*?\"<>|" else "_" for c in attachment.filename])
    if not safe_name:
        safe_name = "attachment"
    target = out_dir / f"{attachment.attachment_id}_{safe_name}"

    with httpx.Client(timeout=120) as client:
        resp = client.get(attachment.download_url, headers=_build_headers())
        resp.raise_for_status()
        content = resp.content

    # 防止写空文件覆盖
    if not content:
        raise ValueError("附件下载为空")

    tmp = str(target) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(content)
    os.replace(tmp, target)

    return target
