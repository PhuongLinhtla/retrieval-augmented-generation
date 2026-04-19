import numpy as np

from .runtime_env import configure_runtime_environment
from .ollama_http import embed_text
from .vietnamese_nlp import estimate_token_count

configure_runtime_environment()


class EmbeddingService:
    def __init__(
        self,
        *,
        backend: str,
        model_name: str,
        ollama_host: str,
        ollama_embedding_model: str,
        request_timeout: float,
        use_reranker: bool = True,
        reranker_model: str = "",
        reranker_batch_size: int = 8,
    ) -> None:
        selected_backend = (backend or "sentence_transformers").strip().lower()
        if selected_backend not in {"sentence_transformers", "ollama"}:
            selected_backend = "sentence_transformers"

        self.backend = selected_backend
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.ollama_embedding_model = ollama_embedding_model
        self.request_timeout = max(10.0, float(request_timeout))
        self.use_reranker = bool(use_reranker and reranker_model.strip())
        self.reranker_model = reranker_model.strip()
        self.reranker_batch_size = max(2, int(reranker_batch_size))
        self.model = None
        self._tokenizer = None
        self._cross_encoder = None

        if self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self._tokenizer = getattr(self.model, "tokenizer", None)
            if hasattr(self.model, "get_embedding_dimension"):
                self.dimension = int(self.model.get_embedding_dimension())
            else:
                self.dimension = int(self.model.get_sentence_embedding_dimension())
        else:
            probe = embed_text(
                host=self.ollama_host,
                model=self.ollama_embedding_model,
                text="xin chao",
                timeout=self.request_timeout,
            )
            self.dimension = len(probe)

    def _format_for_embedding(self, text: str, *, is_query: bool) -> str:
        payload = text.strip()
        if not payload:
            return ""

        model_name = self.model_name.casefold()
        if "multilingual-e5" in model_name or model_name.endswith("e5-large") or "e5-" in model_name:
            prefix = "query: " if is_query else "passage: "
            return f"{prefix}{payload}"
        return payload

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        prepared = [self._format_for_embedding(text, is_query=False) for text in texts]

        if self.backend == "sentence_transformers":
            vectors = self.model.encode(
                prepared,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=32,
            )
            return vectors.astype(np.float32)

        rows: list[list[float]] = []
        for text in prepared:
            vector = embed_text(
                host=self.ollama_host,
                model=self.ollama_embedding_model,
                text=text,
                timeout=self.request_timeout,
            )
            rows.append(vector)

        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-9)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        prepared = self._format_for_embedding(query, is_query=True)
        if self.backend == "sentence_transformers":
            vectors = self.model.encode(
                [prepared],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=1,
            )
            return vectors.astype(np.float32)[0]

        vector = embed_text(
            host=self.ollama_host,
            model=self.ollama_embedding_model,
            text=prepared,
            timeout=self.request_timeout,
        )
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0.0:
            arr = arr / norm
        return arr

    def count_tokens(self, text: str) -> int:
        payload = text.strip()
        if not payload:
            return 0

        if self._tokenizer is None:
            return estimate_token_count(payload)

        try:
            encoded = self._tokenizer.encode(payload, add_special_tokens=False)
            return int(len(encoded))
        except Exception:
            return estimate_token_count(payload)

    def _load_cross_encoder(self):
        if not self.use_reranker:
            return None
        if self._cross_encoder is not None:
            return self._cross_encoder

        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.reranker_model)
            return self._cross_encoder
        except Exception:
            self._cross_encoder = None
            self.use_reranker = False
            return None

    def cross_encode_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        model = self._load_cross_encoder()
        if model is None:
            return []

        scores = model.predict(
            pairs,
            batch_size=self.reranker_batch_size,
            show_progress_bar=False,
        )
        values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return []

        if float(values.min()) < 0.0 or float(values.max()) > 1.0:
            values = 1.0 / (1.0 + np.exp(-values))

        values = np.clip(values, 0.0, 1.0)
        return [float(item) for item in values]
