from __future__ import annotations

import re

from models.schemas import Chunk, ResumeDocument

SECTION_HEADERS = {
    "experience": re.compile(r"^\s*(experience|work experience|professional experience)\s*$", re.IGNORECASE),
    "education": re.compile(r"^\s*(education|academic background)\s*$", re.IGNORECASE),
    "skills": re.compile(r"^\s*(skills|technical skills|core skills)\s*$", re.IGNORECASE),
}


def _detect_section(line: str, default_section: str) -> str:
    for section, pattern in SECTION_HEADERS.items():
        if pattern.match(line):
            return section
    return default_section


def chunk_resume(document: ResumeDocument, max_chunk_chars: int = 1200) -> list[Chunk]:
    lines = [line.rstrip() for line in document.text.splitlines() if line.strip()]
    if not lines:
        return []

    chunks: list[Chunk] = []
    current_section = "general"
    buffer: list[str] = []

    def flush(section: str, content: list[str], chunk_index: int) -> None:
        if not content:
            return
        text = "\n".join(content).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{document.source_id}::chunk::{chunk_index}",
                source_id=document.source_id,
                text=text,
                section_type=section,
                metadata={"section_type": section},
            )
        )

    chunk_index = 0
    for line in lines:
        maybe_section = _detect_section(line, current_section)
        if maybe_section != current_section and buffer:
            flush(current_section, buffer, chunk_index)
            chunk_index += 1
            buffer = []

        current_section = maybe_section

        if sum(len(item) for item in buffer) + len(line) > max_chunk_chars and buffer:
            flush(current_section, buffer, chunk_index)
            chunk_index += 1
            buffer = []

        buffer.append(line)

    flush(current_section, buffer, chunk_index)
    return chunks

