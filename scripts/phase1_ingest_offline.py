from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag import RAGPipeline, load_settings
from rag.ollama_http import vision_to_text
from rag.text_processing import SUPPORTED_EXTENSIONS

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _collect_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            for item in path.rglob("*"):
                if item.is_file():
                    files.append(item)
    return files


def _extract_pdf_images(pdf_file: Path, output_dir: Path, max_images: int) -> list[Path]:
    reader = PdfReader(str(pdf_file))
    images: list[Path] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        page_images = getattr(page, "images", []) or []
        for image_idx, image_obj in enumerate(page_images, start=1):
            if len(images) >= max_images:
                return images

            raw_bytes = getattr(image_obj, "data", None)
            if not raw_bytes:
                continue

            image_name = str(getattr(image_obj, "name", "") or "")
            ext = Path(image_name).suffix.lower()
            if ext not in _IMAGE_EXTENSIONS:
                ext = ".png"

            target = output_dir / f"{pdf_file.stem}_p{page_idx}_img{image_idx}{ext}"
            target.write_bytes(raw_bytes)
            images.append(target)

    return images


def _run_external_parser(cmd_template: str, input_file: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = cmd_template.format(input=str(input_file), output_dir=str(output_dir))
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[WARN] External parser failed for {input_file.name}: {result.stderr.strip()}")
        return []

    produced = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".markdown"}
    ]
    return produced


def _vision_to_markdown(image_file: Path, extracted_text: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{image_file.stem}.vision.md"
    target.write_text(
        "\n".join(
            [
                f"# Vision extraction: {image_file.name}",
                "",
                extracted_text.strip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 (offline): extract and ingest documents without loading chat LLM.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files/folders to ingest. If omitted, use ./documents.",
    )
    parser.add_argument(
        "--disable-vision",
        action="store_true",
        help="Disable vision extraction for image files and PDF embedded images.",
    )
    parser.add_argument(
        "--vision-model",
        default="",
        help="Override OLLAMA_VISION_MODEL for this run.",
    )
    parser.add_argument(
        "--max-images-per-pdf",
        type=int,
        default=12,
        help="Maximum embedded images extracted per PDF for vision OCR.",
    )
    parser.add_argument(
        "--marker-cmd",
        default="",
        help="Optional command template for Marker parser. Use {input} and {output_dir} placeholders.",
    )
    parser.add_argument(
        "--nougat-cmd",
        default="",
        help="Optional command template for Nougat parser. Use {input} and {output_dir} placeholders.",
    )

    args = parser.parse_args()

    settings = load_settings(llm_provider="local")
    pipeline = RAGPipeline(settings)

    targets = args.paths or [str(settings.documents_dir)]
    all_files = _collect_files(targets)
    if not all_files:
        print("No files found for ingestion.")
        return

    ingest_candidates: list[Path] = []
    image_candidates: list[Path] = []

    intermediate_dir = settings.storage_dir / "phase1_intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    for file_path in all_files:
        suffix = file_path.suffix.lower()
        if suffix in SUPPORTED_EXTENSIONS:
            ingest_candidates.append(file_path)
        elif suffix in _IMAGE_EXTENSIONS:
            image_candidates.append(file_path)

    pdf_files = [path for path in ingest_candidates if path.suffix.lower() == ".pdf"]

    if args.marker_cmd:
        for pdf_file in pdf_files:
            produced = _run_external_parser(
                args.marker_cmd,
                pdf_file,
                intermediate_dir / "marker" / pdf_file.stem,
            )
            ingest_candidates.extend(produced)

    if args.nougat_cmd:
        for pdf_file in pdf_files:
            produced = _run_external_parser(
                args.nougat_cmd,
                pdf_file,
                intermediate_dir / "nougat" / pdf_file.stem,
            )
            ingest_candidates.extend(produced)

    if not args.disable_vision:
        for pdf_file in pdf_files:
            try:
                extracted = _extract_pdf_images(
                    pdf_file,
                    intermediate_dir / "pdf_images" / pdf_file.stem,
                    max(0, args.max_images_per_pdf),
                )
                image_candidates.extend(extracted)
            except Exception as exc:
                print(f"[WARN] Failed to extract images from {pdf_file.name}: {exc}")

        vision_model = args.vision_model.strip() or settings.ollama_vision_model
        if vision_model and image_candidates:
            generated_markdown: list[Path] = []
            for image_file in image_candidates:
                try:
                    extracted_text = vision_to_text(
                        host=settings.ollama_host,
                        model=vision_model,
                        image_bytes=image_file.read_bytes(),
                        timeout=settings.ollama_timeout,
                    )
                    generated_markdown.append(
                        _vision_to_markdown(
                            image_file,
                            extracted_text,
                            intermediate_dir / "vision_text",
                        )
                    )
                except Exception as exc:
                    print(f"[WARN] Vision OCR failed for {image_file.name}: {exc}")

            ingest_candidates.extend(generated_markdown)

    deduped_files = sorted({path.resolve() for path in ingest_candidates if path.exists()})
    if not deduped_files:
        print("No supported files found after preprocessing.")
        return

    print("Running Phase 1 ingestion...")
    print(f"- Files for embedding: {len(deduped_files)}")
    summary = pipeline.ingest_files([Path(path) for path in deduped_files])

    print("Ingestion summary")
    print("- Indexed files:", summary["indexed_files"])
    print("- Indexed child chunks:", summary["indexed_chunks"])
    print("- Indexed parent chunks:", summary.get("indexed_parent_chunks", 0))
    print("- Skipped files:", summary["skipped_files"])
    print("- Failed files:", summary["failed_files"])
    print("- Total chunks in DB:", summary.get("total_chunks_in_db", 0))

    errors = summary.get("errors", [])
    if errors:
        print("Errors:")
        for err in errors:
            print("  *", err)


if __name__ == "__main__":
    main()
