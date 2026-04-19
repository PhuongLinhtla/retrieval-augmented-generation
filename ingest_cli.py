from __future__ import annotations

import argparse
from pathlib import Path

from rag import RAGPipeline, load_settings
from rag.text_processing import SUPPORTED_EXTENSIONS


def _collect_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
            continue
        if path.is_dir():
            for item in path.rglob("*"):
                if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(item)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest files into RAG vector database")
    parser.add_argument(
        "paths",
        nargs="*",
        help="File or folder paths. If omitted, use ./documents folder.",
    )
    parser.add_argument(
        "--provider",
        default="local",
        choices=["local", "openai", "gemini"],
        help="LLM provider used when asking questions.",
    )

    args = parser.parse_args()

    settings = load_settings(llm_provider=args.provider)
    pipeline = RAGPipeline(settings)

    targets = args.paths or [str(settings.documents_dir)]
    files = _collect_files(targets)

    if not files:
        print("No supported files found. Supported extensions:", ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        return

    summary = pipeline.ingest_files(files)

    print("Ingestion summary")
    print("- Indexed files:", summary["indexed_files"])
    print("- Indexed chunks:", summary["indexed_chunks"])
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
