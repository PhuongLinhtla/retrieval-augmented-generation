import numpy as np

from .runtime_env import configure_runtime_environment

configure_runtime_environment()

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vectors = self.embed_texts([query])
        return vectors[0]
