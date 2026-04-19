from __future__ import annotations

from pathlib import Path
import json

import numpy as np

try:
    import hnswlib
except Exception:  # pragma: no cover - runtime availability depends on environment.
    hnswlib = None


class _LSHANNIndex:
    def __init__(self, dim: int, n_planes: int = 18, seed: int = 42) -> None:
        self.dim = dim
        self.n_planes = max(8, min(24, n_planes))
        rng = np.random.default_rng(seed)

        planes = rng.standard_normal((self.n_planes, dim), dtype=np.float32)
        norms = np.linalg.norm(planes, axis=1, keepdims=True)
        self.planes = planes / np.maximum(norms, 1e-9)

        self.ids = np.empty((0,), dtype=np.int64)
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.signatures = np.empty((0,), dtype=np.uint64)
        self.buckets: dict[int, list[int]] = {}

    def build(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        self.ids = ids.astype(np.int64)
        self.vectors = vectors.astype(np.float32)
        self.signatures = self._compute_signatures(self.vectors)

        buckets: dict[int, list[int]] = {}
        for pos, signature in enumerate(self.signatures):
            key = int(signature)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(pos)
        self.buckets = buckets

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.ids.size == 0 or k <= 0:
            return []

        query = query_vector.astype(np.float32).reshape(-1)
        signature = self._signature_for_query(query)
        candidate_target = min(self.ids.shape[0], max(k * 6, 30))
        candidate_positions = self._candidate_positions(signature, candidate_target)

        if not candidate_positions:
            return []

        selected_positions = np.array(sorted(candidate_positions), dtype=np.int64)
        selected_vectors = self.vectors[selected_positions]
        similarities = selected_vectors @ query

        top_n = min(k, similarities.shape[0])
        if top_n <= 0:
            return []

        if top_n == similarities.shape[0]:
            order = np.argsort(similarities)[::-1]
        else:
            top_idx = np.argpartition(similarities, -top_n)[-top_n:]
            order = top_idx[np.argsort(similarities[top_idx])[::-1]]

        hits: list[tuple[int, float]] = []
        for rel_idx in order:
            pos = int(selected_positions[rel_idx])
            chunk_id = int(self.ids[pos])
            score = float(np.clip(similarities[rel_idx], -1.0, 1.0))
            hits.append((chunk_id, score))
        return hits

    def _compute_signatures(self, vectors: np.ndarray) -> np.ndarray:
        projections = vectors @ self.planes.T
        signs = projections >= 0

        signatures = np.zeros((vectors.shape[0],), dtype=np.uint64)
        for bit in range(signs.shape[1]):
            signatures |= signs[:, bit].astype(np.uint64) << np.uint64(bit)
        return signatures

    def _signature_for_query(self, query_vector: np.ndarray) -> int:
        projections = self.planes @ query_vector
        signature = 0
        for bit, value in enumerate(projections):
            if float(value) >= 0.0:
                signature |= 1 << bit
        return signature

    def _candidate_positions(self, query_signature: int, target_count: int) -> set[int]:
        selected: set[int] = set(self.buckets.get(query_signature, []))

        for bit in range(self.n_planes):
            if len(selected) >= target_count:
                break
            neighbor = query_signature ^ (1 << bit)
            selected.update(self.buckets.get(neighbor, []))

        if len(selected) < target_count:
            for bit1 in range(self.n_planes):
                if len(selected) >= target_count:
                    break
                for bit2 in range(bit1 + 1, self.n_planes):
                    if len(selected) >= target_count:
                        break
                    neighbor = query_signature ^ (1 << bit1) ^ (1 << bit2)
                    selected.update(self.buckets.get(neighbor, []))

        if len(selected) < target_count and self.signatures.size > 0:
            xor_values = self.signatures ^ np.uint64(query_signature)
            hamming = np.fromiter(
                (int(value).bit_count() for value in xor_values),
                dtype=np.int16,
                count=xor_values.shape[0],
            )
            fallback_count = min(xor_values.shape[0], max(target_count * 2, target_count))
            if fallback_count == xor_values.shape[0]:
                nearest = np.arange(xor_values.shape[0], dtype=np.int64)
            else:
                nearest = np.argpartition(hamming, fallback_count - 1)[:fallback_count]
            selected.update(int(pos) for pos in nearest.tolist())

        return selected


class HNSWVectorStore:
    def __init__(
        self,
        index_path: Path,
        meta_path: Path,
        ef_search: int = 120,
        ef_construction: int = 200,
        m: int = 32,
    ) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.m = m
        self.backend = "hnsw" if hnswlib is not None else "lsh"

        self.index: object | None = None
        self._lsh_index: _LSHANNIndex | None = None
        self.dim: int | None = None
        self.count = 0
        self.max_elements = 0

        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self.backend != "hnsw":
            return

        if not self.index_path.exists() or not self.meta_path.exists():
            return

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if str(meta.get("backend", "hnsw")) != "hnsw":
            return

        dim = int(meta["dim"])
        max_elements = int(meta["max_elements"])
        count = int(meta["count"])

        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(str(self.index_path), max_elements=max_elements)
        index.set_ef(max(self.ef_search, int(meta.get("ef_search", self.ef_search))))

        self.index = index
        self.dim = dim
        self.count = count
        self.max_elements = max_elements

    def _save_meta(self) -> None:
        if self.dim is None:
            return
        meta = {
            "backend": self.backend,
            "dim": self.dim,
            "count": self.count,
            "max_elements": self.max_elements,
            "ef_search": self.ef_search,
            "ef_construction": self.ef_construction,
            "m": self.m,
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def rebuild(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        if ids.size == 0 or vectors.size == 0:
            self.clear()
            return

        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array")
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError("ids and vectors size mismatch")

        dim = int(vectors.shape[1])
        self.dim = dim
        self.count = int(ids.shape[0])

        if self.backend == "hnsw":
            max_elements = max(1024, int(ids.shape[0] * 1.5) + 64)
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(
                max_elements=max_elements,
                ef_construction=self.ef_construction,
                M=self.m,
            )
            index.add_items(vectors.astype(np.float32), ids.astype(np.int64))
            index.set_ef(max(self.ef_search, min(400, ids.shape[0] * 2)))
            index.save_index(str(self.index_path))

            self.index = index
            self._lsh_index = None
            self.max_elements = max_elements
            self._save_meta()
            return

        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

        planes = 20 if dim >= 192 else 16
        lsh_index = _LSHANNIndex(dim=dim, n_planes=planes)
        lsh_index.build(ids=ids, vectors=vectors)

        self.index = None
        self._lsh_index = lsh_index
        self.max_elements = self.count

    def clear(self) -> None:
        self.index = None
        self._lsh_index = None
        self.dim = None
        self.count = 0
        self.max_elements = 0

        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        ann_candidates: int,
    ) -> list[tuple[int, float]]:
        if self.count == 0:
            return []

        k = min(max(top_k, ann_candidates), self.count)
        if k <= 0:
            return []

        if self.backend == "lsh":
            if self._lsh_index is None:
                return []
            return self._lsh_index.search(query_vector=query_vector, k=k)

        if self.index is None:
            return []

        self.index.set_ef(max(self.ef_search, ann_candidates * 4))

        labels, distances = self.index.knn_query(
            query_vector.astype(np.float32).reshape(1, -1),
            k=k,
        )

        hits: list[tuple[int, float]] = []
        for label, distance in zip(labels[0], distances[0]):
            if int(label) < 0:
                continue
            similarity = 1.0 - float(distance)
            hits.append((int(label), similarity))
        return hits

    @property
    def ready(self) -> bool:
        if self.backend == "lsh":
            return self._lsh_index is not None and self.count > 0
        return self.index is not None and self.count > 0

    @property
    def backend_name(self) -> str:
        return self.backend
