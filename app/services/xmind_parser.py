from __future__ import annotations

import io
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class ScenarioNode:
    """从 XMind 提取的场景节点（用于入库）。"""

    node_id: str
    title: str
    path: str
    notes: Optional[str]
    context_text: str


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _extract_notes(obj: dict) -> Optional[str]:
    # XMind 不同版本 notes 结构不同，这里做宽松兼容
    notes = obj.get("notes")
    if isinstance(notes, str) and notes.strip():
        return notes.strip()
    if isinstance(notes, dict):
        plain = notes.get("plain")
        if isinstance(plain, str) and plain.strip():
            return plain.strip()
    return None


def _walk_topic(topic: dict, parent_path: str) -> Iterable[ScenarioNode]:
    title = topic.get("title")
    if not isinstance(title, str):
        title = ""

    path = "/".join([p for p in [parent_path, title] if p])
    notes = _extract_notes(topic)

    # children 兼容：
    # - JSON 结构：children.topics
    children = topic.get("children") or {}
    topics = []
    if isinstance(children, dict):
        t = children.get("topics")
        if isinstance(t, list):
            topics = t

    node_id = topic.get("id")
    if not isinstance(node_id, str) or not node_id:
        node_id = _sha1(path)

    if not topics:
        context_text = f"{title} | 路径:{parent_path}"
        if notes:
            context_text += f" | 备注:{notes}"

        yield ScenarioNode(
            node_id=node_id,
            title=title,
            path=parent_path,
            notes=notes,
            context_text=context_text,
        )
        return

    for child in topics:
        if isinstance(child, dict):
            yield from _walk_topic(child, path)


def _strip_ns(tag: str) -> str:
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_child(elem: ET.Element, name: str) -> Optional[ET.Element]:
    for ch in list(elem):
        if _strip_ns(ch.tag) == name:
            return ch
    return None


def _find_children(elem: ET.Element, name: str) -> List[ET.Element]:
    out: List[ET.Element] = []
    for ch in list(elem):
        if _strip_ns(ch.tag) == name:
            out.append(ch)
    return out


def _extract_notes_xml(topic_elem: ET.Element) -> Optional[str]:
    notes_elem = _find_child(topic_elem, "notes")
    if notes_elem is None:
        return None
    plain_elem = _find_child(notes_elem, "plain")
    if plain_elem is not None and isinstance(plain_elem.text, str) and plain_elem.text.strip():
        return plain_elem.text.strip()
    html_elem = _find_child(notes_elem, "html")
    if html_elem is not None and isinstance(html_elem.text, str) and html_elem.text.strip():
        return html_elem.text.strip()
    return None


def _walk_topic_xml(topic_elem: ET.Element, parent_path: str) -> Iterable[ScenarioNode]:
    title_elem = _find_child(topic_elem, "title")
    title = title_elem.text.strip() if title_elem is not None and isinstance(title_elem.text, str) else ""

    path = "/".join([p for p in [parent_path, title] if p])
    notes = _extract_notes_xml(topic_elem)

    node_id = topic_elem.attrib.get("id")
    if not isinstance(node_id, str) or not node_id:
        node_id = _sha1(path)

    child_topics: List[ET.Element] = []
    children_elem = _find_child(topic_elem, "children")
    if children_elem is not None:
        topics_elems = _find_children(children_elem, "topics")
        for te in topics_elems:
            for t in _find_children(te, "topic"):
                child_topics.append(t)

    if not child_topics:
        context_text = f"{title} | 路径:{parent_path}"
        if notes:
            context_text += f" | 备注:{notes}"

        yield ScenarioNode(
            node_id=node_id,
            title=title,
            path=parent_path,
            notes=notes,
            context_text=context_text,
        )
        return

    for ch in child_topics:
        yield from _walk_topic_xml(ch, path)


def parse_xmind(raw_bytes: bytes, parse_mode: str = "leaf_only") -> List[ScenarioNode]:
    """解析 .xmind 文件。

    说明：
    - .xmind 本质是 zip。
    - 优先解析 content.json（新格式）。
    - 兼容旧格式 content.xml。
    - parse_mode:
      - leaf_only：只返回叶子节点（默认）
      - all_nodes：暂未实现（后续可补）
    """

    nodes: List[ScenarioNode] = []

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        names = set(zf.namelist())
        if "content.json" in names:
            content = zf.read("content.json")
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, list):
                sheets = data
            elif isinstance(data, dict):
                sheets = [data]
            else:
                sheets = []

            for sheet in sheets:
                if not isinstance(sheet, dict):
                    continue
                root_topic = sheet.get("rootTopic")
                if not isinstance(root_topic, dict):
                    continue

                root_title = root_topic.get("title")
                if not isinstance(root_title, str):
                    root_title = ""

                for n in _walk_topic(root_topic, root_title):
                    nodes.append(n)
        elif "content.xml" in names:
            content = zf.read("content.xml")
            root = ET.fromstring(content)
            for sheet_elem in root.iter():
                if _strip_ns(sheet_elem.tag) != "sheet":
                    continue
                title_elem = _find_child(sheet_elem, "title")
                sheet_title = (
                    title_elem.text.strip()
                    if title_elem is not None and isinstance(title_elem.text, str)
                    else ""
                )

                topic_elem = _find_child(sheet_elem, "topic")
                if topic_elem is None:
                    continue

                for n in _walk_topic_xml(topic_elem, sheet_title):
                    nodes.append(n)
        else:
            return nodes

    return nodes
