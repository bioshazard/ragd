from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    text: str
    segment_index: int | None = None
    t_start: float | None = None
    t_end: float | None = None


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict


def _word_count(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_units(text: str, max_unit_tokens: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    units: list[str] = []
    for para in paragraphs:
        if _word_count(para) > max_unit_tokens:
            units.extend(_split_sentences(para))
        else:
            units.append(para)
    return units


def _tail_words(text: str, count: int) -> str:
    if count <= 0:
        return ""
    words = text.split()
    if len(words) <= count:
        return text
    return " ".join(words[-count:])


def build_units(content: str | list[dict]) -> list[Unit]:
    if isinstance(content, str):
        units = _split_units(content, 240)
        return [Unit(text=unit, segment_index=i) for i, unit in enumerate(units)]

    units: list[Unit] = []
    for idx, segment in enumerate(content):
        text = segment.get("text", "").strip()
        if not text:
            continue
        units.append(
            Unit(
                text=text,
                segment_index=idx,
                t_start=segment.get("t_start"),
                t_end=segment.get("t_end"),
            )
        )
    return units


def chunk_units(
    units: list[Unit],
    target_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_units: list[Unit] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _word_count(unit.text)
        if current_units and current_tokens + unit_tokens > target_tokens:
            chunk_text = " ".join(u.text for u in current_units).strip()
            chunks.append(_build_chunk(chunk_text, current_units))

            overlap_text = _tail_words(chunk_text, overlap_tokens)
            current_units = []
            current_tokens = 0
            if overlap_text:
                current_units.append(Unit(text=overlap_text))
                current_tokens = _word_count(overlap_text)

        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        chunk_text = " ".join(u.text for u in current_units).strip()
        chunks.append(_build_chunk(chunk_text, current_units))

    return chunks


def _build_chunk(text: str, units: list[Unit]) -> Chunk:
    meta_units = [u for u in units if u.segment_index is not None]
    metadata: dict = {}
    if meta_units:
        metadata["segment_start"] = meta_units[0].segment_index
        metadata["segment_end"] = meta_units[-1].segment_index
        t_starts = [u.t_start for u in meta_units if u.t_start is not None]
        t_ends = [u.t_end for u in meta_units if u.t_end is not None]
        if t_starts:
            metadata["t_start"] = min(t_starts)
        if t_ends:
            metadata["t_end"] = max(t_ends)

    return Chunk(text=text, metadata=metadata)
