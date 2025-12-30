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


def _tail_words(text: str, count: int, max_chars: int | None = None) -> str:
    if count <= 0:
        return ""
    words = text.split()
    if len(words) <= count:
        result = text
    else:
        result = " ".join(words[-count:])

    if max_chars is not None and len(result) > max_chars:
        trimmed_words = result.split()
        while trimmed_words and len(" ".join(trimmed_words)) > max_chars:
            trimmed_words = trimmed_words[1:]
        result = " ".join(trimmed_words)
    return result


def build_units(content: str | list[dict], max_unit_tokens: int = 240) -> list[Unit]:
    if isinstance(content, str):
        units = _split_units(content, max_unit_tokens)
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
    max_chars: int | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_units: list[Unit] = []
    current_tokens = 0
    current_chars = 0

    expanded_units: list[Unit] = []
    if max_chars is not None:
        for unit in units:
            if len(unit.text) > max_chars:
                expanded_units.extend(_split_long_unit(unit, max_chars))
            else:
                expanded_units.append(unit)
    else:
        expanded_units = units

    for unit in expanded_units:
        unit_tokens = _word_count(unit.text)
        unit_chars = len(unit.text)
        if current_units and (
            current_tokens + unit_tokens > target_tokens
            or (max_chars is not None and current_chars + unit_chars > max_chars)
        ):
            chunk_text = " ".join(u.text for u in current_units).strip()
            chunks.append(_build_chunk(chunk_text, current_units))

            overlap_text = _tail_words(chunk_text, overlap_tokens, max_chars=max_chars)
            current_units = []
            current_tokens = 0
            current_chars = 0
            if overlap_text:
                current_units.append(Unit(text=overlap_text))
                current_tokens = _word_count(overlap_text)
                current_chars = len(overlap_text)

        current_units.append(unit)
        current_tokens += unit_tokens
        current_chars += unit_chars

    if current_units:
        chunk_text = " ".join(u.text for u in current_units).strip()
        chunks.append(_build_chunk(chunk_text, current_units))

    return chunks


def _split_long_unit(unit: Unit, max_chars: int) -> list[Unit]:
    words = unit.text.split()
    if not words:
        return []
    parts: list[Unit] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        if current and current_len + 1 + word_len > max_chars:
            parts.append(_clone_unit(unit, " ".join(current)))
            current = [word]
            current_len = word_len
        else:
            if current:
                current_len += 1 + word_len
            else:
                current_len = word_len
            current.append(word)

    if current:
        parts.append(_clone_unit(unit, " ".join(current)))
    return parts


def _clone_unit(unit: Unit, text: str) -> Unit:
    return Unit(
        text=text,
        segment_index=unit.segment_index,
        t_start=unit.t_start,
        t_end=unit.t_end,
    )


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
