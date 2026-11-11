"""Sentence-aware chunking: whole sentences are packed into chunks of ~chunk_size words,
and consecutive chunks share trailing sentences worth up to `overlap` words."""
import re


def _units(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    units = _units(text)
    chunks, cur, cur_words = [], [], 0
    for u in units:
        w = len(u.split())
        if cur and cur_words + w > chunk_size:
            chunks.append("\n".join(cur))
            # carry over trailing sentences up to `overlap` words
            keep, kw = [], 0
            for s in reversed(cur):
                sw = len(s.split())
                if kw + sw > overlap:
                    break
                keep.insert(0, s)
                kw += sw
            cur, cur_words = keep, kw
        cur.append(u)
        cur_words += w
    if cur:
        last = "\n".join(cur)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return chunks
