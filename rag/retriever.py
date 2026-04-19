from __future__ import annotations

import re

import numpy as np

from .embedding_service import EmbeddingService
from .repository import MetadataRepository
from .schemas import RetrievedChunk
from .vector_store import HNSWVectorStore


_WORD_PATTERN = re.compile(r"\b[\w]{2,}\b", re.IGNORECASE)
_STOPWORDS = {
    "la",
    "va",
    "voi",
    "cua",
    "cho",
    "trong",
    "nhung",
    "the",
    "for",
    "and",
    "with",
    "from",
    "that",
    "this",
    "is",
    "are",
}


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_PATTERN.findall(text)}


def _lexical_overlap(query: set[str], text: str) -> float:
    if not query:
        return 0.0
    target = _tokenize(text)
    if not target:
        return 0.0
    intersection = query.intersection(target)
    union = query.union(target)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _build_query_variants(query: str) -> list[str]:
    base = query.strip()
    if not base:
        return []

    variants = [base]
    normalized = re.sub(r"[:;,.!?()\[\]{}]+", " ", base.lower())
    normalized = " ".join(normalized.split())
    if normalized and normalized not in variants:
        variants.append(normalized)

    focused_tokens = [token for token in _tokenize(base) if token not in _STOPWORDS]
    if len(focused_tokens) >= 3:
        focused_query = " ".join(focused_tokens)
        if focused_query not in variants:
            variants.append(focused_query)

    return variants[:3]


class AdvancedRetriever:
    def __init__(
        self,
        repository: MetadataRepository,
        embedder: EmbeddingService,
        vector_store: HNSWVectorStore,
        top_k: int,
        ann_candidates: int,
        min_similarity: float,
        mmr_lambda: float,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.ann_candidates = ann_candidates
        self.min_similarity = min_similarity
        self.mmr_lambda = mmr_lambda

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query = question.strip()
        if not query:
            return []

        wanted_k = top_k or self.top_k

        merged_hits: dict[int, float] = {}
        for query_variant in _build_query_variants(query):
            query_vector = self.embedder.embed_query(query_variant)
            ann_hits = self.vector_store.search(
                query_vector=query_vector,
                top_k=wanted_k,
                ann_candidates=self.ann_candidates,
            )
            for chunk_id, similarity in ann_hits:
                current = merged_hits.get(chunk_id)
                if current is None or similarity > current:
                    merged_hits[chunk_id] = similarity

        ann_hits = sorted(merged_hits.items(), key=lambda item: item[1], reverse=True)
        if not ann_hits:
            return []

        chunk_ids = [chunk_id for chunk_id, _ in ann_hits]
        chunk_records = self.repository.get_chunks_by_ids(chunk_ids)
        if not chunk_records:
            return []

        by_id = {record.chunk_id: record for record in chunk_records}
        query_tokens = _tokenize(query)

        candidates: list[dict[str, object]] = []
        texts_to_embed: list[str] = []

        for chunk_id, similarity in ann_hits:
            record = by_id.get(chunk_id)
            if record is None:
                continue

            lexical_score = _lexical_overlap(query_tokens, record.content)
            relevance = 0.82 * similarity + 0.18 * lexical_score
            if similarity < self.min_similarity and lexical_score < 0.08:
                continue

            candidates.append(
                {
                    "record": record,
                    "similarity": similarity,
                    "lexical_score": lexical_score,
                    "relevance": relevance,
                }
            )
            texts_to_embed.append(record.content)

        if not candidates:
            return []

        text_vectors = self.embedder.embed_texts(texts_to_embed)
        for idx, candidate in enumerate(candidates):
            candidate["vector"] = text_vectors[idx]

        selected = self._mmr_select(candidates, wanted_k)
        selected.sort(key=lambda item: float(item["relevance"]), reverse=True)

        return [
            RetrievedChunk(
                chunk=item["record"],
                similarity=float(item["similarity"]),
                lexical_score=float(item["lexical_score"]),
                final_score=float(item["relevance"]),
            )
            for item in selected
        ]

    def _mmr_select(
        self,
        candidates: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        remaining = [dict(candidate) for candidate in candidates]
        selected: list[dict[str, object]] = []

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = -1e9

            for idx, candidate in enumerate(remaining):
                relevance = float(candidate["relevance"])
                if not selected:
                    mmr_score = relevance
                else:
                    candidate_vector = candidate["vector"]
                    max_similarity = max(
                        float(np.dot(candidate_vector, picked["vector"])) for picked in selected
                    )
                    mmr_score = (
                        self.mmr_lambda * relevance
                        - (1.0 - self.mmr_lambda) * max_similarity
                    )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(remaining.pop(best_idx))

        return selected
