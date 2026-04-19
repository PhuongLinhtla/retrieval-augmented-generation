from pathlib import Path
import re

from pypdf import PdfReader

from .schemas import TextSpan


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".log"}


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_text_spans(file_path: Path) -> list[TextSpan]:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {suffix}")

    if suffix == ".pdf":
        return _load_pdf(file_path)
    return _load_plain_text(file_path)


def chunk_spans(
    spans: list[TextSpan],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int, str]]:
    chunks: list[tuple[int, int, str]] = []
    chunk_index = 0

    for span in spans:
        units = _build_units(span.text, max(220, chunk_size // 2))
        if not units:
            continue

        buffer: list[str] = []
        buffer_length = 0

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue

            extra_length = len(unit) + (1 if buffer else 0)
            if buffer and buffer_length + extra_length > chunk_size:
                chunk_text = " ".join(buffer).strip()
                if chunk_text:
                    chunks.append((span.page_number, chunk_index, chunk_text))
                    chunk_index += 1

                overlap_text = _take_overlap(chunk_text, chunk_overlap)
                if overlap_text:
                    buffer = [overlap_text, unit]
                else:
                    buffer = [unit]
                buffer_length = sum(len(part) for part in buffer) + max(0, len(buffer) - 1)
            else:
                buffer.append(unit)
                buffer_length += extra_length

        if buffer:
            chunk_text = " ".join(buffer).strip()
            if chunk_text:
                chunks.append((span.page_number, chunk_index, chunk_text))
                chunk_index += 1

    return chunks


def _load_pdf(file_path: Path) -> list[TextSpan]:
    reader = PdfReader(str(file_path))
    spans: list[TextSpan] = []

    for index, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            spans.append(TextSpan(page_number=index, text=text))

    return spans


def _load_plain_text(file_path: Path) -> list[TextSpan]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(text)
    if not text:
        return []
    return [TextSpan(page_number=1, text=text)]


def _build_units(text: str, max_unit_chars: int) -> list[str]:
    paragraphs = [clean_text(par) for par in re.split(r"\n\n+", text)]
    paragraphs = [par for par in paragraphs if par]

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_unit_chars:
            units.append(paragraph)
            continue
        units.extend(_split_long_paragraph(paragraph, max_unit_chars))

    return units


def _split_long_paragraph(paragraph: str, max_unit_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not sentences:
        return _split_by_chars(paragraph, max_unit_chars)

    out: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue

        if len(current) + 1 + len(sentence) <= max_unit_chars:
            current = f"{current} {sentence}"
        else:
            out.append(current)
            current = sentence

    if current:
        out.append(current)

    final_units: list[str] = []
    for unit in out:
        if len(unit) <= max_unit_chars:
            final_units.append(unit)
        else:
            final_units.extend(_split_by_chars(unit, max_unit_chars))

    return final_units


def _split_by_chars(text: str, max_unit_chars: int) -> list[str]:
    slices: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(text_length, start + max_unit_chars)
        if end < text_length:
            split_pos = text.rfind(" ", start, end)
            if split_pos > start + max_unit_chars // 2:
                end = split_pos

        part = text[start:end].strip()
        if part:
            slices.append(part)

        if end == text_length:
            break
        start = end

    return slices


def _take_overlap(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return ""

    tail = text[-overlap_chars:].strip()
    first_space = tail.find(" ")
    if first_space > 0:
        tail = tail[first_space + 1 :]
    return tail.strip()
