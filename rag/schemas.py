from dataclasses import dataclass


@dataclass(slots=True)
class TextSpan:
    page_number: int
    text: str


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: int
    document_id: int
    source_name: str
    source_path: str
    page_number: int
    chunk_index: int
    content: str


@dataclass(slots=True)
class RetrievedChunk:
    chunk: ChunkRecord
    similarity: float
    lexical_score: float
    final_score: float


@dataclass(slots=True)
class AnswerResult:
    answer: str
    contexts: list[RetrievedChunk]
    provider: str
