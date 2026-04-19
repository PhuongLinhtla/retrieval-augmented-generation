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
    source: str
    page_number: int
    chunk_index: int
    content: str
    file_type: str = ""
    doc_type: str = ""
    title: str = ""
    subject: str = ""
    grade: str = ""
    chapter: str = ""
    published_year: str = ""
    parent_chunk_id: int | None = None
    parent_page_number: int | None = None
    parent_index: int | None = None
    parent_title: str = ""
    parent_content: str = ""
    section_path: str = ""
    keywords: str = ""
    context_header: str = ""


@dataclass(slots=True)
class RetrievedChunk:
    chunk: ChunkRecord
    similarity: float
    lexical_score: float
    final_score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


@dataclass(slots=True)
class AnswerResult:
    answer: str
    contexts: list[RetrievedChunk]
    provider: str
    confidence: float = 0.0
    intent: str = "general"
    needs_clarification: bool = False
