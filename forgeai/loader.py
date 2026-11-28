"""Document loading and character-based chunking utilities."""

import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class Doc(BaseModel):
    doc_id: str
    kind: Literal["pdf", "text", "csv"]
    text: str
    path: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page: int | None = None


def load_csv_df(path: str) -> pd.DataFrame:
    """Load a CSV file as a pandas DataFrame."""
    return pd.read_csv(path)


def load_folder(path: str) -> list[Doc]:
    """Load supported documents from a folder, ordered by filename."""
    docs: list[Doc] = []
    for file_path in sorted(Path(path).iterdir()):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()
        if extension == ".pdf":
            reader = PdfReader(str(file_path))
            pages = [(page.extract_text() or "") for page in reader.pages]
            if sum(len(page) for page in pages) < 50:
                logger.warning("PDF %s yielded almost no extractable text", file_path)
            text_parts: list[str] = []
            for page_number, page_text in enumerate(pages, start=1):
                text_parts.append(page_text)
                if page_number < len(pages):
                    text_parts.append(f"[[page {page_number}]]")
            text = "\n\n".join(text_parts)
            docs.append(
                Doc(
                    doc_id=file_path.name,
                    kind="pdf",
                    text=text,
                    path=str(file_path),
                )
            )
        elif extension in {".txt", ".md"}:
            docs.append(
                Doc(
                    doc_id=file_path.name,
                    kind="text",
                    text=file_path.read_text(encoding="utf-8"),
                    path=str(file_path),
                )
            )
        elif extension == ".csv":
            dataframe = load_csv_df(str(file_path))
            rows: list[str] = []
            for _, row in dataframe.iterrows():
                values: list[str] = []
                for column, value in row.items():
                    rendered = "" if pd.isna(value) else str(value)
                    rendered = rendered.replace("\r\n", "\n").replace("\r", "\n")
                    rendered = rendered.replace("\n", " ")
                    values.append(f"{column}: {rendered}")
                rows.append(" | ".join(values))
            docs.append(
                Doc(
                    doc_id=file_path.name,
                    kind="csv",
                    text="\n".join(rows),
                    path=str(file_path),
                )
            )
        else:
            logger.warning("Skipping unsupported file: %s", file_path)
    return docs


def _validate_chunk_options(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")


def _sentence_units(paragraph: str, size: int) -> list[tuple[str, str]]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    units: list[tuple[str, str]] = []
    for sentence_index, sentence in enumerate(sentences):
        separator = " " if sentence_index else ""
        if len(sentence) <= size:
            units.append((sentence, separator))
        else:
            for start in range(0, len(sentence), size):
                units.append(
                    (
                        sentence[start : start + size],
                        separator if start == 0 else "",
                    )
                )
    return units


def _text_units(text: str, size: int) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    first = True
    for paragraph in re.split(r"\n\n+", text):
        if not paragraph:
            continue
        if len(paragraph) > size:
            parts = _sentence_units(paragraph, size)
            for index, (part, separator) in enumerate(parts):
                if index == 0:
                    separator = "" if first else "\n\n"
                units.append((part, separator))
        else:
            units.append((paragraph, "" if first else "\n\n"))
        first = False
    return units


def _pack_text_units(
    units: list[tuple[str, str]], size: int, overlap: int, page: int | None
) -> list[tuple[str, int | None]]:
    chunks: list[tuple[str, int | None]] = []
    cursor = 0
    previous_text = ""

    while cursor < len(units):
        prefix = previous_text[-overlap:] if overlap else ""
        body = ""
        while cursor < len(units):
            unit, separator = units[cursor]
            joiner = separator if body or prefix else ""
            candidate = f"{body}{joiner}{unit}"
            if len(prefix) + len(candidate) <= size:
                body = candidate
                cursor += 1
                continue

            if body:
                break

            available = size - len(prefix) - len(joiner)
            if available <= 0:
                joiner = ""
                available = size - len(prefix)
            piece = unit[:available]
            remainder = unit[available:]
            body = f"{joiner}{piece}"
            if remainder:
                units[cursor] = (remainder, "")
            else:
                cursor += 1
            break

        chunk_text = f"{prefix}{body}"
        chunks.append((chunk_text, page))
        previous_text = chunk_text

    return chunks


def _pdf_pages(text: str) -> list[tuple[int | None, str]]:
    marker = re.compile(r"\n\n\[\[page (\d+)\]\]\n\n")
    pages: list[tuple[int | None, str]] = []
    start = 0
    page_number = 1
    for match in marker.finditer(text):
        pages.append((page_number, text[start : match.start()]))
        page_number = int(match.group(1)) + 1
        start = match.end()
    pages.append((page_number if marker.search(text) else None, text[start:]))
    return pages


def _chunk_csv_rows(rows: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cursor = 0
    previous_rows: list[str] = []

    while cursor < len(rows):
        carried: list[str] = []
        carried_length = 0
        if overlap:
            for row in reversed(previous_rows):
                added_length = len(row) + (1 if carried else 0)
                if carried_length >= overlap:
                    break
                carried.insert(0, row)
                carried_length += added_length

        chunk_rows = carried
        fresh_count = 0
        while cursor + fresh_count < len(rows):
            row = rows[cursor + fresh_count]
            candidate = chunk_rows + [row]
            candidate_length = len("\n".join(candidate))
            if chunk_rows and candidate_length > size and fresh_count > 0:
                break
            if chunk_rows and candidate_length > size and not carried:
                break
            if not chunk_rows and len(row) > size:
                chunk_rows.append(row)
                fresh_count += 1
                break
            if candidate_length > size and carried and fresh_count == 0:
                chunk_rows = []
                carried = []
                continue
            chunk_rows = candidate
            fresh_count += 1
            if candidate_length > size:
                break

        if fresh_count == 0:
            row = rows[cursor]
            chunk_rows = [row]
            fresh_count = 1

        chunks.append("\n".join(chunk_rows))
        cursor += fresh_count
        previous_rows = chunk_rows

    return chunks


def chunk(doc: Doc, size: int, overlap: int) -> list[Chunk]:
    """Split a document into chunks, preserving CSV rows and PDF page metadata."""
    _validate_chunk_options(size, overlap)

    if doc.kind == "csv":
        texts = _chunk_csv_rows(doc.text.splitlines(), size, overlap)
        return [
            Chunk(chunk_id=f"{doc.doc_id}::{index}", doc_id=doc.doc_id, text=text)
            for index, text in enumerate(texts)
        ]

    page_groups = _pdf_pages(doc.text) if doc.kind == "pdf" else [(None, doc.text)]
    texts_with_pages: list[tuple[str, int | None]] = []
    for page_number, page_text in page_groups:
        texts_with_pages.extend(
            _pack_text_units(_text_units(page_text, size), size, overlap, page_number)
        )

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::{index}",
            doc_id=doc.doc_id,
            text=text,
            page=page,
        )
        for index, (text, page) in enumerate(texts_with_pages)
    ]


def chunk_all(docs: list[Doc], size: int, overlap: int) -> list[Chunk]:
    """Chunk each document and return the combined chunks in document order."""
    _validate_chunk_options(size, overlap)
    return [document_chunk for doc in docs for document_chunk in chunk(doc, size, overlap)]
