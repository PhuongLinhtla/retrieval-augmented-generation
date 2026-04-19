from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .embedding_service import EmbeddingService
from .llm_clients import GroundedLLM
from .repository import MetadataRepository
from .retriever import AdvancedRetriever
from .schemas import AnswerResult
from .text_processing import (
    SUPPORTED_EXTENSIONS,
    build_parent_child_chunks_with_tokens,
    load_text_spans,
)
from .vector_store import HNSWVectorStore
from .vietnamese_nlp import build_contextual_header, extract_document_metadata


_SLIDE_HINTS = ("slide", "slides", "lecture", "bai_giang", "ppt")
_BOOK_HINTS = ("book", "textbook", "chapter", "sach", "chuong")


def _infer_document_type(source_name: str, file_type: str) -> str:
    lowered_name = source_name.strip().lower()
    lowered_type = file_type.strip().lower()

    if lowered_type in {".ppt", ".pptx"}:
        return "slide"
    if any(token in lowered_name for token in _SLIDE_HINTS):
        return "slide"
    if any(token in lowered_name for token in _BOOK_HINTS):
        return "book"
    if lowered_type == ".pdf":
        return "book"
    return "note"


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = MetadataRepository(settings.db_path)
        self.embedder = EmbeddingService(
            backend=settings.embedding_backend,
            model_name=settings.embedding_model,
            ollama_host=settings.ollama_host,
            ollama_embedding_model=settings.ollama_embedding_model,
            request_timeout=settings.ollama_timeout,
            use_reranker=settings.use_reranker,
            reranker_model=settings.reranker_model,
            reranker_batch_size=settings.reranker_batch_size,
        )
        self.vector_store = HNSWVectorStore(
            index_path=settings.index_path,
            meta_path=settings.index_meta_path,
        )
        self.retriever = AdvancedRetriever(
            repository=self.repository,
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=settings.top_k,
            top_k_dense=settings.top_k_dense,
            top_k_sparse=settings.top_k_sparse,
            fusion_top_k=settings.fusion_top_k,
            rerank_top_k=settings.rerank_top_k,
            ann_candidates=settings.ann_candidates,
            min_similarity=settings.min_similarity,
            mmr_lambda=settings.mmr_lambda,
            use_hybrid_retrieval=settings.use_hybrid_retrieval,
            use_reranker=settings.use_reranker,
        )
        self.llm = GroundedLLM(
            provider=settings.llm_provider,
            ollama_host=settings.ollama_host,
            ollama_llm_model=settings.ollama_llm_model,
            ollama_fallback_model=settings.ollama_fallback_model,
            ollama_num_ctx=settings.ollama_num_ctx,
            ollama_temperature=settings.ollama_temperature,
            ollama_timeout=settings.ollama_timeout,
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
            "indexed_parent_chunks": 0,
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

                inferred_metadata = extract_document_metadata(
                    source_name=file_path.name,
                    text=spans[0].text if spans else "",
                )

                document_id = self.repository.upsert_document(
                    source_path=str(file_path.resolve()),
                    source_name=file_path.name,
                    file_type=file_path.suffix.lower(),
                    doc_type=_infer_document_type(file_path.name, file_path.suffix.lower()),
                    total_pages=len(spans),
                    title=inferred_metadata.title,
                    subject=inferred_metadata.subject,
                    grade=inferred_metadata.grade,
                    chapter=inferred_metadata.chapter,
                    published_year=inferred_metadata.published_year,
                    source=inferred_metadata.source,
                )

                self.repository.delete_chunks_for_document(document_id)
                self.repository.delete_parent_chunks_for_document(document_id)

                parent_rows, child_rows = build_parent_child_chunks_with_tokens(
                    spans=spans,
                    child_chunk_target_tokens=self.settings.child_chunk_target_tokens,
                    child_chunk_min_tokens=self.settings.child_chunk_min_tokens,
                    child_chunk_max_tokens=self.settings.child_chunk_max_tokens,
                    child_chunk_overlap_sentences=self.settings.child_chunk_overlap_sentences,
                    parent_chunk_target_tokens=self.settings.parent_chunk_target_tokens,
                    parent_chunk_min_tokens=self.settings.parent_chunk_min_tokens,
                    parent_chunk_max_tokens=self.settings.parent_chunk_max_tokens,
                    parent_child_group_min=self.settings.parent_child_group_min,
                    parent_child_group_max=self.settings.parent_child_group_max,
                    chapter_hint=inferred_metadata.chapter,
                )
                if not child_rows:
                    summary["skipped_files"] = int(summary["skipped_files"]) + 1
                    continue

                parent_mapping = self.repository.insert_parent_chunks(document_id, parent_rows)

                chunk_rows = [
                    (page_number, chunk_index, content)
                    for page_number, chunk_index, _parent_index, content in child_rows
                ]

                chunk_ids = self.repository.insert_chunks(document_id, chunk_rows)

                child_parent_mappings: list[tuple[int, int]] = []
                for idx, (_page_number, _chunk_index, parent_index, _content) in enumerate(child_rows):
                    parent_chunk_id = parent_mapping.get(parent_index)
                    if parent_chunk_id is None:
                        continue
                    child_parent_mappings.append((chunk_ids[idx], parent_chunk_id))
                self.repository.insert_child_parent_map(child_parent_mappings)

                parent_lookup = {
                    int(parent_index): {
                        "title": str(title or ""),
                        "section_path": str(section_path or ""),
                    }
                    for _page, parent_index, title, section_path, _keywords, _content in parent_rows
                }

                embed_payloads: list[str] = []
                for page_number, _chunk_index, parent_index, content in child_rows:
                    parent_info = parent_lookup.get(int(parent_index), {})
                    header = build_contextual_header(
                        subject=inferred_metadata.subject,
                        grade=inferred_metadata.grade,
                        chapter=inferred_metadata.chapter,
                        section_title=str(
                            parent_info.get("section_path") or parent_info.get("title") or ""
                        ),
                        page_number=int(page_number),
                    )
                    embed_payloads.append(f"{header}\n{content}".strip() if header else str(content))

                vectors = self.embedder.embed_texts(embed_payloads)
                self.repository.insert_embeddings(chunk_ids, vectors)

                summary["indexed_files"] = int(summary["indexed_files"]) + 1
                summary["indexed_chunks"] = int(summary["indexed_chunks"]) + len(chunk_ids)
                summary["indexed_parent_chunks"] = (
                    int(summary["indexed_parent_chunks"]) + len(parent_rows)
                )

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
        intent = self.retriever.infer_intent(question)
        retrieved = self.retriever.retrieve(
            question,
            top_k=max(self.settings.rerank_top_k, self.settings.evidence_chunk_max),
        )
        confidence = self.retriever.estimate_confidence(retrieved)

        if self._needs_clarification(confidence, retrieved):
            clarification = self._build_clarification_prompt(intent=intent)
            return AnswerResult(
                answer=clarification,
                contexts=retrieved,
                provider="local",
                confidence=confidence,
                intent=intent,
                needs_clarification=True,
            )

        contexts = self._select_context_budget(retrieved)

        evidence_sentences = []
        if self.settings.enable_evidence_compression:
            evidence_sentences = self.retriever.compress_evidence_sentences(
                question=question,
                contexts=contexts,
                sentence_min=self.settings.evidence_sentence_min,
                sentence_max=self.settings.evidence_sentence_max,
            )

        answer, provider = self.llm.generate_answer(
            question,
            contexts,
            intent=intent,
            confidence=confidence,
            evidence_sentences=evidence_sentences,
        )
        return AnswerResult(
            answer=answer,
            contexts=contexts,
            provider=provider,
            confidence=confidence,
            intent=intent,
            needs_clarification=False,
        )

    def _needs_clarification(self, confidence: float, contexts) -> bool:
        if not contexts:
            return True

        ordered = sorted((float(item.final_score) for item in contexts), reverse=True)
        top_1 = ordered[0]
        top_2 = ordered[1] if len(ordered) >= 2 else 0.0
        top_5 = ordered[:5]
        median = top_5[len(top_5) // 2] if top_5 else 0.0
        margin = max(0.0, top_1 - median)
        top_gap = max(0.0, top_1 - top_2)
        top_rerank = max(float(item.rerank_score) for item in contexts)

        return (
            confidence < self.settings.low_confidence_threshold
            or margin < self.settings.low_confidence_margin
            or top_gap < self.settings.low_confidence_top_gap
            or (
                self.settings.use_reranker
                and top_rerank < self.settings.reranker_low_score_threshold
            )
        )

    def _select_context_budget(self, contexts):
        if not contexts:
            return []

        ordered = sorted(contexts, key=lambda item: float(item.final_score), reverse=True)

        min_chunks = max(1, self.settings.evidence_chunk_min)
        max_chunks = max(min_chunks, self.settings.evidence_chunk_max)
        min_tokens = max(100, self.settings.evidence_min_tokens)
        max_tokens = max(min_tokens, self.settings.evidence_max_tokens)

        selected = []
        total_tokens = 0

        for item in ordered:
            if len(selected) >= max_chunks:
                break

            payload = item.chunk.content or item.chunk.parent_content
            token_count = self.embedder.count_tokens(payload)

            if (
                selected
                and total_tokens + token_count > max_tokens
                and len(selected) >= min_chunks
                and total_tokens >= min_tokens
            ):
                break

            selected.append(item)
            total_tokens += token_count

        if len(selected) < min_chunks:
            for item in ordered[len(selected) :]:
                if len(selected) >= min_chunks or len(selected) >= max_chunks:
                    break
                selected.append(item)
                total_tokens += self.embedder.count_tokens(item.chunk.content or item.chunk.parent_content)

        if total_tokens < min_tokens:
            for item in ordered[len(selected) :]:
                if len(selected) >= max_chunks:
                    break
                candidate_tokens = self.embedder.count_tokens(item.chunk.content or item.chunk.parent_content)
                if selected and total_tokens + candidate_tokens > max_tokens:
                    break
                selected.append(item)
                total_tokens += candidate_tokens
                if total_tokens >= min_tokens:
                    break

        return selected or ordered[:max_chunks]

    def _build_clarification_prompt(self, intent: str) -> str:
        guidance = {
            "definition": "Ban dang can dinh nghia theo mon/lop/chuong nao?",
            "proof": "Ban can chung minh theo tai lieu nao (mon/lop/chuong)?",
            "procedure": "Ban muon quy trinh trong chuong nao hoac bo bai nao?",
            "comparison": "Ban can so sanh hai khai niem nao, thuoc tai lieu nao?",
            "example": "Ban can vi du thuoc mon/lop/chuong nao?",
            "general": "Ban co the cho biet mon, lop, chuong hoac ten tai lieu cu the duoc khong?",
        }
        return (
            "Do tin cay retrieval hien dang thap nen minh chua tra loi de tranh sai. "
            + guidance.get(intent, guidance["general"])
        )

    def list_documents(self) -> list[dict[str, object]]:
        return self.repository.list_documents()

    def vector_backend(self) -> str:
        return self.vector_store.backend_name

    def settings_dict(self) -> dict[str, object]:
        raw = asdict(self.settings)
        raw["openai_api_key"] = "***" if self.settings.openai_api_key else ""
        raw["gemini_api_key"] = "***" if self.settings.gemini_api_key else ""
        return raw
