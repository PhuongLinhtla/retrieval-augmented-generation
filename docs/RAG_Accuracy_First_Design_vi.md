# Accuracy-First RAG Design (Vietnamese Learning Assistant)

Tai lieu nay map truc tiep plan accuracy-first vao codebase hien tai.

## 1. Data preparation (accuracy foundation)

- Bat buoc metadata cho moi document:
  - `doc_id`, `title`, `subject`, `grade`, `chapter`, `published_year`, `source`, `file_type`, `doc_type`.
- PDF ingestion:
  - Van giu `page_number` cho moi chunk.
  - Co bo loc header/footer lap lai giua nhieu trang de giam retrieval noise.
- OCR text hygiene:
  - Chuan hoa khoang trang.
  - Fix loi OCR pho bien (`l/1`, `O/0`, dau cau, ky tu noise).

Implemented in:
- `rag/repository.py`
- `rag/text_processing.py`
- `rag/vietnamese_nlp.py`

## 2. Text cleaning and normalization

- Giu nguyen dau tieng Viet trong noi dung.
- Co norm khong dau phuc vu retrieval lexical.
- Tach cau theo quy tac tieng Viet (co bao ve viet tat co dau cham).

Implemented in:
- `rag/vietnamese_nlp.py`

## 3. Two-tier chunking (child + parent)

- Child chunk:
  - Muc tieu token: 350-550 (default 450), min 200, max 700.
  - Overlap theo so cau (default 2 cau).
- Parent chunk:
  - Muc tieu token: 900-1600 (default target 1200).
  - Gom 2-4 child lien ke.
- Rule khong cat:
  - Tranh cat giua bullet list.
  - Tranh cat giua cap `Cau hoi` - `Loi giai`.
  - Han che cat giua formula blocks.
- Co `section_path`, `keywords` cho parent chunk.

Implemented in:
- `rag/text_processing.py`

## 4. Embeddings

- Ho tro local sentence-transformers va Ollama embeddings.
- Tuong thich E5 format (`query:` / `passage:`).
- Co contextual header khi embed child chunk:
  - `Mon · Lop · Chuong · Section · Trang` + content.

Implemented in:
- `rag/embedding_service.py`
- `rag/pipeline.py`

## 5. Hybrid index and retrieval

- Dense retrieval: ANN (HNSW/LSH fallback).
- Sparse retrieval: BM25 in-memory tu corpus chunk + metadata heading.
- Fusion: RRF (Reciprocal Rank Fusion).

Implemented in:
- `rag/retriever.py`
- `rag/repository.py`

## 6. Retrieval pipeline details

- Query normalize va expansion:
  - Ban co dau, khong dau, abbreviation expansion, focused tokens.
- Dense and sparse chay song song.
- RRF tron ket qua truoc khi rerank.

Implemented in:
- `rag/retriever.py`
- `rag/vietnamese_nlp.py`

## 7. Rerank

- Ho tro cross-encoder reranker (default: `BAAI/bge-reranker-v2-m3`).
- Neu reranker unavailable -> fallback weighted scoring.
- Chon ket qua da da dang parent de tranh mot muc ap dao.

Implemented in:
- `rag/embedding_service.py`
- `rag/retriever.py`

## 8. Evidence compression

- Optional sentence-level evidence extraction tu top contexts.
- Chon top sentence theo relevance + overlap query.
- Moi cau giu citation.

Implemented in:
- `rag/retriever.py`
- `rag/pipeline.py`

## 9. LLM answer constraints

- Two-pass generation:
  - Pass 1: rut facts + conflicts + missing.
  - Pass 2: tong hop cau tra loi cuoi cung chi tu facts.
- Bat buoc citation `[Nguon X]`.
- Neu conflict: neu conflict, khong tu chon bua.
- Neu thieu du lieu: noi ro thieu du lieu.

Implemented in:
- `rag/llm_clients.py`

## 10. Uncertainty awareness

- Confidence score duoc tinh tu quality cua top retrieval.
- Khi confidence thap: tra loi theo huong clarification thay vi doan.

Implemented in:
- `rag/retriever.py`
- `rag/pipeline.py`

## 11. Hardware tuning (RTX 3060 12GB)

Suggested defaults:
- Tang `TOP_K_DENSE`, `TOP_K_SPARSE`, `FUSION_TOP_K` neu uu tien accuracy.
- Giam cac gia tri tren neu uu tien latency.
- Reranker bat se tang chat luong, doi lai tang thoi gian.

Configured in:
- `.env.example`
- `rag/config.py`

## 12. Acceptance checklist

- [ ] Metadata day du trong `documents` table.
- [ ] Child-parent chunk ratio hop ly.
- [ ] Dense + sparse deu co ket qua tren query mau.
- [ ] Reranker score duoc hien thi tren UI.
- [ ] Cau tra loi co citation va tu choi khi thieu du lieu.
- [ ] Confidence gate kich hoat dung voi query ngoai kho.
