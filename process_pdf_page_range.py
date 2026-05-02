#!/usr/bin/env python3
"""
Process only a page range from a PDF (e.g. pages 7-17).

Workflow:
1. Extract selected pages to a temporary PDF
2. Try text extraction from the subset PDF
3. If text is empty and ocrmypdf is available, OCR the subset PDF
4. Chunk the extracted text and write JSONL ready for LightRAG ingestion
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader, PdfWriter


def extract_page_range(input_pdf: Path, start_page: int, end_page: int, output_pdf: Path) -> None:
    reader = PdfReader(str(input_pdf))
    total_pages = len(reader.pages)

    if start_page < 1 or end_page < start_page:
        raise ValueError("Invalid page range")
    if end_page > total_pages:
        raise ValueError(f"end_page={end_page} exceeds total pages={total_pages}")

    writer = PdfWriter()
    for page_index in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_index])

    with open(output_pdf, "wb") as f:
        writer.write(f)


def extract_text_from_pdf(pdf_path: Path) -> str:
    text_parts: list[str] = []
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text)
    return "\n\n".join(text_parts).strip()


def run_ocr_if_available(pdf_path: Path, ocr_pdf_path: Path) -> bool:
    if shutil.which("ocrmypdf") is None:
        return False

    cmd = [
        "ocrmypdf",
        "-l",
        "vie+eng",
        "--skip-text",
        str(pdf_path),
        str(ocr_pdf_path),
    ]
    subprocess.run(cmd, check=True)
    return True


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(len(text), pos + chunk_size)
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        pos = max(end - overlap, pos + 1)
    return chunks


def write_jsonl(chunks: Iterable[str], output_path: Path, source_name: str) -> int:
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk, "source": source_name}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Process a PDF page range into JSONL chunks.")
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument("start_page", type=int, help="Start page (1-based, inclusive)")
    parser.add_argument("end_page", type=int, help="End page (1-based, inclusive)")
    parser.add_argument("--output-dir", type=Path, default=Path("math_docs_processed"), help="Output directory")
    parser.add_argument("--working-dir", type=Path, default=Path("internal_docs"), help="LightRAG working directory")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"❌ PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.working_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.pdf.stem
    subset_pdf = args.output_dir / f"{base_name}_p{args.start_page}_{args.end_page}.pdf"
    subset_ocr_pdf = args.output_dir / f"{base_name}_p{args.start_page}_{args.end_page}_ocr.pdf"
    subset_txt = args.output_dir / f"{base_name}_p{args.start_page}_{args.end_page}.txt"
    output_jsonl = args.output_dir / f"{base_name}_p{args.start_page}_{args.end_page}_processed.jsonl"

    print(f"📄 Input: {args.pdf}")
    print(f"📑 Pages: {args.start_page}-{args.end_page}")
    print("-" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pdf = Path(tmpdir) / subset_pdf.name
        extract_page_range(args.pdf, args.start_page, args.end_page, tmp_pdf)
        shutil.copy2(tmp_pdf, subset_pdf)
        print(f"✓ Extracted subset PDF: {subset_pdf.name}")

        text = extract_text_from_pdf(subset_pdf)
        if not text.strip():
            print("⚠️  No text layer found in subset PDF.")
            if run_ocr_if_available(subset_pdf, subset_ocr_pdf):
                print(f"✓ OCR completed: {subset_ocr_pdf.name}")
                text = extract_text_from_pdf(subset_ocr_pdf)
                if text.strip():
                    subset_txt.write_text(text, encoding="utf-8")
                    print(f"✓ Text saved: {subset_txt.name}")
            else:
                print("⚠️  ocrmypdf not available; keeping subset PDF only.")

    if not text.strip():
        print("❌ No extractable text found in pages {}-{}.".format(args.start_page, args.end_page))
        print("   If this is a scanned PDF, install ocrmypdf or use an OCR tool first.")
        return 2

    chunks = chunk_text(text)
    if not chunks:
        print("❌ Text extracted, but no chunks were produced.")
        return 3

    count = write_jsonl(chunks, output_jsonl, args.pdf.name)
    shutil.copy2(output_jsonl, args.working_dir / output_jsonl.name)

    print(f"✓ Wrote chunks: {output_jsonl.name} ({count} chunks)")
    print(f"✓ Copied to: {args.working_dir / output_jsonl.name}")
    print("\nReady for ingestion or WebUI upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
