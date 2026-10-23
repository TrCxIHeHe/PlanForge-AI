"""Turn loosely structured model output into validated content blocks."""
from __future__ import annotations

import re
from typing import Any

from .schemas import Block

_ALIASES = {
    "paragraph": "paragraph", "text": "paragraph", "p": "paragraph",
    "bullets": "bullets", "bullet": "bullets", "list": "bullets", "ul": "bullets",
    "numbered": "numbered", "ordered": "numbered", "steps": "numbered", "ol": "numbered",
    "table": "table",
}
_PLACEHOLDER = re.compile(r"lorem ipsum|\bTODO\b|\bTBD\b|\[insert|\[your |<placeholder", re.IGNORECASE)


class ContentError(ValueError):
    """The model output cannot be turned into a usable section."""


def clean_text(value: Any) -> str:
    """Strip markdown noise the Word renderer would otherwise print literally."""
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"```[a-z]*", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = text.replace("`", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [line for line in re.split(r"\n+", value)]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict):  # {"item": "..."} / {"text": "..."} shapes
            item = item.get("text") or item.get("item") or next(iter(item.values()), "")
        text = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", clean_text(item))
        if text:
            out.append(text)
    return out


def _parse_block(raw: Any) -> Block | None:
    if isinstance(raw, str):
        raw = {"type": "paragraph", "text": raw}
    if not isinstance(raw, dict):
        return None
    kind = _ALIASES.get(str(raw.get("type", "")).strip().lower())
    if kind is None:
        kind = "table" if raw.get("rows") else "bullets" if raw.get("items") else "paragraph"
    caption = clean_text(raw.get("caption"))[:160]
    if kind == "paragraph":
        text = clean_text(raw.get("text") or raw.get("content"))
        return Block(type="paragraph", text=text) if text else None
    if kind in ("bullets", "numbered"):
        items = _strings(raw.get("items") or raw.get("text"))
        return Block(type=kind, items=items[:12], caption=caption) if items else None
    headers = [clean_text(h) for h in raw.get("headers") or [] if clean_text(h)]
    rows = []
    for row in raw.get("rows") or []:
        if isinstance(row, dict):
            row = [row.get(h, "") for h in headers]
        if isinstance(row, list):
            cells = [clean_text(c) for c in row]
            if any(cells):
                rows.append(cells)
    if not headers or not rows:
        return None
    width = len(headers)
    rows = [(r + [""] * width)[:width] for r in rows[:15]]  # pad/truncate ragged rows
    return Block(type="table", headers=headers[:7], rows=[r[: len(headers[:7])] for r in rows], caption=caption)


def _from_plain_text(text: str) -> list[Block]:
    blocks: list[Block] = []
    bullets: list[str] = []
    for line in clean_text(text).splitlines():
        if re.match(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", line):
            bullets.append(line)
            continue
        if bullets:
            blocks.append(Block(type="bullets", items=_strings(bullets)))
            bullets = []
        if line.strip():
            blocks.append(Block(type="paragraph", text=line.strip()))
    if bullets:
        blocks.append(Block(type="bullets", items=_strings(bullets)))
    return blocks


def block_text(block: Block) -> str:
    return " ".join([block.text, *block.items, *block.headers, *(c for r in block.rows for c in r)]).strip()


def parse_section_payload(raw: dict[str, Any]) -> tuple[list[Block], list[str]]:
    """Return (blocks, key_points) or raise ContentError."""
    raw_blocks = raw.get("blocks")
    blocks: list[Block] = []
    if isinstance(raw_blocks, list):
        blocks = [b for b in (_parse_block(item) for item in raw_blocks) if b]
    elif isinstance(raw.get("content"), str):  # tolerate the older {"content": "..."} shape
        blocks = _from_plain_text(raw["content"])
    if sum(len(block_text(b)) for b in blocks) < 60:
        raise ContentError("section had no substantive content")
    key_points = _strings(raw.get("key_points"))[:5]
    if not key_points:
        first = next((b.text for b in blocks if b.type == "paragraph"), "")
        key_points = [s.strip() for s in re.split(r"(?<=[.!?])\s+", first)[:2] if s.strip()]
    return blocks, key_points


def placeholder_hits(blocks: list[Block]) -> list[str]:
    return sorted({m.group(0).lower() for b in blocks for m in [_PLACEHOLDER.search(block_text(b))] if m})
