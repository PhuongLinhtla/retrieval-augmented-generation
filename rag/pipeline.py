from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .embedding_service import EmbeddingService
from .llm_clients import GroundedLLM
from .repository import MetadataRepository
from .retriever import AdvancedRetriever
from .schemas import AnswerResult
from .text_processing import SUPPORTED_EXTENSIONS, chunk_spans, load_text_spans
from .vector_store import HNSWVectorStore


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = MetadataRepository(settings.db_path)
        self.embedder = EmbeddingService(settings.embedding_model)
        self.vector_store = HNSWVectorStore(
            index_path=settings.index_path,
            meta_path=settings.index_meta_path,
        )
        self.retriever = AdvancedRetriever(
            repository=self.repository,
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=settings.top_k,
            ann_candidates=settings.ann_candidates,
            min_similarity=settings.min_similarity,
            mmr_lambda=settings.mmr_lambda,
        )
        self.llm = GroundedLLM(
            provider=settings.llm_provider,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
        )

        self._sync_index_from_database()

    def _sync_index_from_database(self) -> None:
        ids, vectors = self.repository.fetch_all_embeddings()
        if ids.size == 0:
            self.vector_store.clear()
            return

        should_rebuild = (
            not self.vector_store.ready
            or self.vector_store.count != int(ids.shape[0])
            or self.vector_store.dim != int(vectors.shape[1])
        )

        if should_rebuild:
            self.vector_store.rebuild(ids, vectors)

    def ingest_files(self, file_paths: list[Path]) -> dict[str, object]:
        summary: dict[str, object] = {
            "indexed_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "indexed_chunks": 0,
            "errors": [],
        }

        for path in file_paths:
            file_path = Path(path)
            if not file_path.exists() or not file_path.is_file():
                summary["failed_files"] = int(summary["failed_files"]) + 1
                summary["errors"].append(f"Missing file: {file_path}")
                continue

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                summary["skipped_files"] = int(summary["skipped_files"]) + 1
                continue

            try:
                spans = load_text_spans(file_path)
                if not spans:
                    summary["skipped_files"] = int(summary["skipped_files"]) + 1
                    continue

                document_id = self.repository.upsert_document(
                    source_path=str(file_path.resolve()),
                    source_name=file_path.name,
                    file_type=file_path.suffix.lower(),
                    total_pages=len(spans),
                )

                self.repository.delete_chunks_for_document(document_id)

                chunk_rows = chunk_spans(
                    spans=spans,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
                if not chunk_rows:
                    summary["skipped_files"] = int(summary["skipped_files"]) + 1
                    continue

                chunk_ids = self.repository.insert_chunks(document_id, chunk_rows)
                vectors = self.embedder.embed_texts([row[2] for row in chunk_rows])
                self.repository.insert_embeddings(chunk_ids, vectors)

                summary["indexed_files"] = int(summary["indexed_files"]) + 1
                summary["indexed_chunks"] = int(summary["indexed_chunks"]) + len(chunk_ids)

            except Exception as exc:
                summary["failed_files"] = int(summary["failed_files"]) + 1
                summary["errors"].append(f"{file_path.name}: {exc}")

        ids, vectors = self.repository.fetch_all_embeddings()
        self.vector_store.rebuild(ids, vectors)
        summary["total_chunks_in_db"] = self.repository.count_chunks()

        return summary

    def ingest_folder(self, folder: Path, recursive: bool = True) -> dict[str, object]:
        base = Path(folder)
        if not base.exists() or not base.is_dir():
            return {
                "indexed_files": 0,
                "skipped_files": 0,
                "failed_files": 1,
                "indexed_chunks": 0,
                "errors": [f"Folder does not exist: {base}"],
            }

        pattern = "**/*" if recursive else "*"
        files = [
            path
            for path in base.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return self.ingest_files(files)

    def ask(self, question: str) -> AnswerResult:
        contexts = self.retriever.retrieve(question, top_k=self.settings.top_k)
        answer, provider = self.llm.generate_answer(question, contexts)
        return AnswerResult(answer=answer, contexts=contexts, provider=provider)

    def list_documents(self) -> list[dict[str, object]]:
        return self.repository.list_documents()

    def vector_backend(self) -> str:
        return self.vector_store.backend_name

    def settings_dict(self) -> dict[str, object]:
        raw = asdict(self.settings)
        raw["openai_api_key"] = "***" if self.settings.openai_api_key else ""
        raw["gemini_api_key"] = "***" if self.settings.gemini_api_key else ""
        return raw
