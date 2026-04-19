from __future__ import annotations

from collections import Counter
import math

from .embedding_service import EmbeddingService
from .repository import MetadataRepository
from .schemas import RetrievedChunk
from .vector_store import HNSWVectorStore
from .vietnamese_nlp import (
    classify_question_intent,
    expand_query_variants,
    normalize_text_for_matching,
    split_sentences_vietnamese,
    tokenize_vietnamese,
)


class AdvancedRetriever:
    def __init__(
        self,
        repository: MetadataRepository,
        embedder: EmbeddingService,
        vector_store: HNSWVectorStore,
        top_k: int,
        top_k_dense: int,
        top_k_sparse: int,
        fusion_top_k: int,
        rerank_top_k: int,
        ann_candidates: int,
        min_similarity: float,
        mmr_lambda: float,
        use_hybrid_retrieval: bool,
        use_reranker: bool,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.vector_store = vector_store

        self.top_k = max(1, int(top_k))
        self.top_k_dense = max(self.top_k, int(top_k_dense))
        self.top_k_sparse = max(self.top_k, int(top_k_sparse))
        self.fusion_top_k = max(self.top_k_dense, int(fusion_top_k))
        self.rerank_top_k = max(self.top_k, int(rerank_top_k))

        self.ann_candidates = max(4, int(ann_candidates))
        self.min_similarity = max(0.0, float(min_similarity))
        self.mmr_lambda = min(0.95, max(0.1, float(mmr_lambda)))
        self.use_hybrid_retrieval = bool(use_hybrid_retrieval)
        self.use_reranker = bool(use_reranker)

        self._sparse_cache_total = -1
        self._sparse_rows: list[dict[str, object]] = []
        self._sparse_tokens: dict[int, list[str]] = {}
        self._doc_freq: dict[str, int] = {}
        self._avg_doc_len = 0.0

    def infer_intent(self, question: str) -> str:
        return classify_question_intent(question)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query = question.strip()
        if not query:
            return []

        wanted_k = max(1, int(top_k or self.top_k))
        variants = expand_query_variants(query)
        if not variants:
            return []

        dense_hits = self._dense_search(variants)
        sparse_hits = self._sparse_search(variants) if self.use_hybrid_retrieval else {}
        fused = self._rrf_fuse(dense_hits, sparse_hits)
        if not fused:
            return []

        candidate_ids = [chunk_id for chunk_id, _ in fused[: self.fusion_top_k]]
        chunk_records = self.repository.get_chunks_by_ids(candidate_ids)
        if not chunk_records:
            return []

        by_id = {record.chunk_id: record for record in chunk_records}
        dense_max = max(dense_hits.values(), default=1e-6)
        sparse_max = max(sparse_hits.values(), default=1e-6)
        rrf_max = max((score for _, score in fused), default=1e-6)

        query_tokens = set(tokenize_vietnamese(query, remove_stopwords=True, keep_diacritics=False))
        candidates: list[dict[str, object]] = []
        for rank, (chunk_id, rrf_score) in enumerate(fused[: self.fusion_top_k], start=1):
            record = by_id.get(chunk_id)
            if record is None:
                continue

            dense_score = float(dense_hits.get(chunk_id, 0.0))
            sparse_score = float(sparse_hits.get(chunk_id, 0.0))
            lexical_score = self._lexical_overlap(query_tokens, record.content)

            if dense_score < self.min_similarity and sparse_score <= 0.0 and lexical_score < 0.03:
                continue

            candidates.append(
                {
                    "rank": rank,
                    "record": record,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "lexical_score": lexical_score,
                    "rrf_score": float(rrf_score),
                    "dense_norm": dense_score / max(dense_max, 1e-6),
                    "sparse_norm": sparse_score / max(sparse_max, 1e-6),
                    "rrf_norm": float(rrf_score) / max(rrf_max, 1e-6),
                    "parent_key": record.parent_chunk_id or record.chunk_id,
                }
            )

        if not candidates:
            return []

        self._apply_rerank(question, candidates)
        self._apply_final_score(candidates)

        ranked = sorted(candidates, key=lambda item: float(item["final_score"]), reverse=True)
        diverse = self._select_unique_parents(ranked, wanted_k)

        return [
            RetrievedChunk(
                chunk=item["record"],
                similarity=float(item["dense_score"]),
                lexical_score=float(item["lexical_score"]),
                final_score=float(item["final_score"]),
                dense_score=float(item["dense_score"]),
                sparse_score=float(item["sparse_score"]),
                rrf_score=float(item["rrf_score"]),
                rerank_score=float(item["rerank_score"]),
            )
            for item in diverse
        ]

    def estimate_confidence(self, contexts: list[RetrievedChunk]) -> float:
        if not contexts:
            return 0.0

        ordered = sorted(contexts, key=lambda item: item.final_score, reverse=True)
        top_1 = float(ordered[0].final_score)

        top_5 = ordered[:5]
        median = sorted(item.final_score for item in top_5)[len(top_5) // 2]
        gap_top_median = max(0.0, top_1 - float(median))

        if len(ordered) >= 2:
            gap_top_2 = max(0.0, top_1 - float(ordered[1].final_score))
        else:
            gap_top_2 = top_1

        confidence = 0.55 * top_1 + 0.25 * min(1.0, gap_top_median * 1.8) + 0.20 * min(
            1.0,
            gap_top_2 * 2.2,
        )
        return float(max(0.0, min(1.0, confidence)))

    def compress_evidence_sentences(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        sentence_min: int,
        sentence_max: int,
    ) -> list[str]:
        if not contexts:
            return []

        query_tokens = set(tokenize_vietnamese(question, remove_stopwords=True, keep_diacritics=False))
        sentence_rows: list[tuple[float, str]] = []

        for idx, item in enumerate(contexts, start=1):
            source_text = item.chunk.parent_content or item.chunk.content
            for sentence in split_sentences_vietnamese(source_text):
                sent = sentence.strip()
                if len(sent) < 20:
                    continue
                overlap = self._lexical_overlap(query_tokens, sent)
                score = 0.60 * item.final_score + 0.40 * overlap
                citation = (
                    f"{sent} [Nguon {idx}: {item.chunk.source_name}, "
                    f"trang {item.chunk.parent_page_number or item.chunk.page_number}]"
                )
                sentence_rows.append((score, citation))

        sentence_rows.sort(key=lambda row: row[0], reverse=True)
        desired = max(sentence_min, min(sentence_max, len(sentence_rows)))
        return [text for _score, text in sentence_rows[:desired]]

    def _dense_search(self, variants: list[str]) -> dict[int, float]:
        merged: dict[int, float] = {}

        for query_variant in variants:
            vector = self.embedder.embed_query(query_variant)
            hits = self.vector_store.search(
                query_vector=vector,
                top_k=self.top_k_dense,
                ann_candidates=max(self.ann_candidates, self.top_k_dense),
            )
            for chunk_id, similarity in hits:
                current = merged.get(chunk_id)
                if current is None or similarity > current:
                    merged[chunk_id] = float(similarity)

        return merged

    def _sparse_search(self, variants: list[str]) -> dict[int, float]:
        self._refresh_sparse_cache_if_needed()
        if not self._sparse_rows or self._avg_doc_len <= 0.0:
            return {}

        k1 = 1.4
        b = 0.75
        merged_scores: dict[int, float] = {}

        for variant in variants:
            terms = tokenize_vietnamese(variant, remove_stopwords=True, keep_diacritics=False)
            if not terms:
                continue

            qtf = Counter(terms)
            for row in self._sparse_rows:
                chunk_id = int(row["chunk_id"])
                doc_tokens = self._sparse_tokens.get(chunk_id, [])
                if not doc_tokens:
                    continue

                tf = Counter(doc_tokens)
                doc_len = len(doc_tokens)
                score = 0.0
                for term, term_qtf in qtf.items():
                    freq = float(tf.get(term, 0))
                    if freq <= 0.0:
                        continue
                    df = float(self._doc_freq.get(term, 0))
                    n_docs = float(len(self._sparse_rows))
                    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                    denom = freq + k1 * (1.0 - b + b * (doc_len / max(self._avg_doc_len, 1e-6)))
                    score += term_qtf * idf * ((freq * (k1 + 1.0)) / max(denom, 1e-6))

                if score <= 0.0:
                    continue

                current = merged_scores.get(chunk_id)
                if current is None or score > current:
                    merged_scores[chunk_id] = float(score)

        ranked = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)
        return {chunk_id: score for chunk_id, score in ranked[: self.top_k_sparse]}

    def _refresh_sparse_cache_if_needed(self) -> None:
        total_chunks = self.repository.count_chunks()
        if total_chunks == self._sparse_cache_total:
            return

        rows = self.repository.fetch_sparse_corpus()
        self._sparse_rows = rows
        self._sparse_cache_total = total_chunks

        sparse_tokens: dict[int, list[str]] = {}
        doc_freq: Counter[str] = Counter()
        total_len = 0

        for row in rows:
            chunk_id = int(row["chunk_id"])
            content = str(row.get("content") or "")
            heading = str(row.get("parent_title") or row.get("section_path") or "")
            subject = str(row.get("subject") or "")
            chapter = str(row.get("chapter") or "")

            blended = " ".join(part for part in [subject, chapter, heading, content] if part)
            tokens = tokenize_vietnamese(
                normalize_text_for_matching(blended, keep_diacritics=False),
                remove_stopwords=True,
                keep_diacritics=False,
            )
            sparse_tokens[chunk_id] = tokens
            total_len += len(tokens)
            for token in set(tokens):
                doc_freq[token] += 1

        self._sparse_tokens = sparse_tokens
        self._doc_freq = dict(doc_freq)
        self._avg_doc_len = (total_len / len(rows)) if rows else 0.0

    def _rrf_fuse(
        self,
        dense_scores: dict[int, float],
        sparse_scores: dict[int, float],
        rrf_k: int = 60,
    ) -> list[tuple[int, float]]:
        if not dense_scores and not sparse_scores:
            return []

        fused: dict[int, float] = {}
        dense_ranked = sorted(dense_scores.items(), key=lambda item: item[1], reverse=True)
        sparse_ranked = sorted(sparse_scores.items(), key=lambda item: item[1], reverse=True)

        for rank, (chunk_id, _score) in enumerate(dense_ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

        for rank, (chunk_id, _score) in enumerate(sparse_ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

        if dense_ranked and not sparse_ranked:
            # Fallback: keep dense-only ordering if sparse index is empty.
            return dense_ranked[: self.fusion_top_k]

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return ranked[: self.fusion_top_k]

    def _apply_rerank(self, question: str, candidates: list[dict[str, object]]) -> None:
        if not candidates:
            return

        if not (self.use_reranker and self.embedder.use_reranker):
            for item in candidates:
                item["rerank_score"] = float(
                    0.60 * item["dense_norm"] + 0.30 * item["sparse_norm"] + 0.10 * item["lexical_score"]
                )
            return

        pair_inputs: list[tuple[str, str]] = []
        for item in candidates:
            record = item["record"]
            parent = (record.parent_content or "").strip()
            section = (record.section_path or record.parent_title or "").strip()
            chunk_text = (record.content or "").strip()
            payload = "\n".join(part for part in [section, parent[:1800], chunk_text[:500]] if part)
            pair_inputs.append((question, payload))

        scores = self.embedder.cross_encode_pairs(pair_inputs)
        if len(scores) != len(candidates):
            for item in candidates:
                item["rerank_score"] = float(
                    0.60 * item["dense_norm"] + 0.30 * item["sparse_norm"] + 0.10 * item["lexical_score"]
                )
            return

        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)

    def _apply_final_score(self, candidates: list[dict[str, object]]) -> None:
        for item in candidates:
            dense = float(item["dense_norm"])
            sparse = float(item["sparse_norm"])
            lexical = float(item["lexical_score"])
            rrf = float(item["rrf_norm"])
            rerank = float(item.get("rerank_score", 0.0))

            final_score = (
                0.42 * rerank
                + 0.30 * rrf
                + 0.18 * dense
                + 0.07 * sparse
                + 0.03 * lexical
            )
            item["final_score"] = float(max(0.0, min(1.0, final_score)))

    def _select_unique_parents(
        self,
        ranked_candidates: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        selected: list[dict[str, object]] = []
        seen_parent_keys: set[int] = set()

        for item in ranked_candidates:
            parent_key = int(item["parent_key"])
            if parent_key in seen_parent_keys:
                continue
            selected.append(item)
            seen_parent_keys.add(parent_key)
            if len(selected) >= top_k:
                return selected

        for item in ranked_candidates:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break

        return selected

    def _lexical_overlap(self, query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0

        doc_tokens = set(
            tokenize_vietnamese(
                normalize_text_for_matching(text, keep_diacritics=False),
                remove_stopwords=True,
                keep_diacritics=False,
            )
        )
        if not doc_tokens:
            return 0.0

        overlap = query_tokens.intersection(doc_tokens)
        union = query_tokens.union(doc_tokens)
        if not union:
            return 0.0
        return len(overlap) / len(union)
