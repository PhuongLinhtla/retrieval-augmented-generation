# Tài liệu kỹ thuật — RAG Learning Assistant (Đề tài 9)

## 0) Mục tiêu hệ thống
Hệ thống này là một RAG (Retrieval-Augmented Generation) phục vụ trợ lý học tập:
- **Thu nạp tài liệu (ingestion)** từ PDF/TXT/MD, chia nhỏ thành các đoạn (chunk).
- **Nhúng (embedding)** các chunk thành vector bằng SentenceTransformers.
- **Lưu trữ** metadata + nội dung + vector vào SQLite.
- **Truy xuất (retrieval)** theo cosine similarity bằng ANN (ưu tiên HNSW), có **fallback LSH** khi hnswlib không dùng được.
- **Rerank** kết hợp semantic + lexical và chọn context đa dạng bằng **MMR**.
- **Sinh câu trả lời** dựa trên context (grounded) qua OpenAI/Gemini hoặc chế độ local fallback.

Các điểm nổi bật:
- Hiển thị **Nguồn tham khảo (Context)** dưới mỗi câu trả lời trong UI.
- Chạy được trên Windows dù thiếu C++ Build Tools nhờ cơ chế fallback.

---

## 1) Ngăn xếp công nghệ (tech stack)

### 1.1 Python
Toàn bộ hệ thống viết bằng Python, chia thành:
- UI: Streamlit.
- CLI: argparse.
- Core RAG: module `rag/`.

### 1.2 Streamlit (UI chatbot)
**Dùng để** tạo giao diện web nhanh (không cần backend web framework).
- Chat UI: `st.chat_message`, `st.chat_input`.
- Upload file: `st.sidebar.file_uploader`.
- Cache pipeline: `@st.cache_resource` để tránh nạp model nhiều lần.

### 1.3 SentenceTransformers (embedding)
**Dùng để** biến văn bản thành vector embedding.
- Model mặc định: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- `normalize_embeddings=True` để embedding có chuẩn $\ell_2 = 1$.

### 1.4 NumPy
**Dùng để**:
- Xử lý vector, dot product.
- Thao tác mảng và tính điểm tương đồng.

### 1.5 SQLite (lưu trữ)
**Dùng để** lưu:
- Bảng `documents`: thông tin file.
- Bảng `chunks`: các đoạn văn bản đã chia.
- Bảng `embeddings`: vector embedding lưu dạng BLOB.

SQLite phù hợp cho bài lab vì:
- Dễ setup, 1 file `.sqlite3`.
- Không cần server DB.

### 1.6 hnswlib (ANN HNSW) — tùy chọn
**Dùng để** ANN search rất nhanh trên dữ liệu vector.
- Ở Windows, hnswlib thường cần C++ build tools → vì vậy project đặt `hnswlib` ở chế độ *optional*.
- Khi không import được hnswlib, hệ thống tự chuyển sang LSH.

### 1.7 ANN-LSH (fallback nội bộ)
**Dùng để** truy xuất xấp xỉ khi HNSW không sẵn sàng.
- Cài đặt bằng random hyperplane LSH.
- Lưu vector trong RAM và bucket theo chữ ký (signature).

### 1.8 pypdf
**Dùng để** đọc PDF và trích text theo từng trang.

### 1.9 python-dotenv
**Dùng để** nạp biến môi trường từ file `.env`.
- Tuyệt đối không commit `.env`.
- Chỉ commit `.env.example`.

### 1.10 OpenAI SDK + Google Generative AI SDK
**Dùng để** gọi LLM bên ngoài khi có API key.
- OpenAI: `openai>=1.x` (client kiểu `OpenAI`).
- Gemini: `google-generativeai`.

---

## 2) Các kỹ thuật/thuật toán (techniques)

### 2.1 Chunking có overlap
**Mục tiêu**: tạo các đoạn văn bản có kích thước ổn định để embedding và retrieval.
- Chia theo paragraph → sentence → fallback chia theo ký tự.
- Khi chunk đầy, tạo chunk mới nhưng giữ lại **overlap** (đuôi chunk trước) để giảm mất ngữ cảnh.

**Vì sao cần overlap?**
- Câu trả lời thường cần thông tin ở ranh giới giữa hai đoạn.
- Overlap giảm hiện tượng “đứt mạch” thông tin.

### 2.2 Embedding + chuẩn hoá vector
Embedding được tạo với `normalize_embeddings=True`.
- Khi vector đã chuẩn hoá, **cosine similarity** tương đương với **dot product**:

$$\cos(\theta)=\frac{x\cdot y}{\|x\|\|y\|}\;\Rightarrow\;\|x\|=\|y\|=1\;\Rightarrow\;\cos(\theta)=x\cdot y$$

### 2.3 ANN (Approximate Nearest Neighbor)
**Bài toán**: tìm các vector gần nhất với vector truy vấn trong tập lớn.
- HNSW: nhanh và chất lượng cao.
- LSH: nhanh hơn brute-force, dễ triển khai, phù hợp fallback.

### 2.4 HNSW (Hierarchical Navigable Small World)
**Ý tưởng**: xây đồ thị nhiều tầng để duyệt gần đúng.
- Tham số quan trọng:
  - `M`: số liên kết của mỗi node.
  - `ef_construction`: độ “kỹ” khi build.
  - `ef_search`: độ “kỹ” khi search.

### 2.5 LSH bằng random hyperplane (SimHash-style)
**Ý tưởng**: chiếu vector lên nhiều siêu phẳng ngẫu nhiên; dấu của mỗi chiếu tạo ra 1 bit.
- Vector signature: một số nguyên chứa các bit dấu.
- Query signature: lấy bucket cùng signature + lân cận (flip 1–2 bit) + fallback theo Hamming.

### 2.6 Query rewriting nhẹ (query variants)
Hệ thống tạo tối đa 3 biến thể truy vấn:
- Câu gốc.
- Câu đã chuẩn hoá dấu câu.
- Câu “focus” bỏ stopwords nếu đủ dài.

**Mục tiêu**: tăng recall khi câu hỏi có nhiễu hoặc nhiều từ phụ.

### 2.7 Hybrid rerank: semantic + lexical overlap
Sau khi ANN trả về candidate, hệ thống rerank theo:
- `similarity` (semantic): từ embedding.
- `lexical_score` (từ vựng): Jaccard overlap giữa token của query và token của chunk.

Công thức:
- `relevance = 0.82 * similarity + 0.18 * lexical_score`

Có lọc ngưỡng:
- Nếu `similarity < MIN_SIMILARITY` **và** `lexical_score < 0.08` → loại.

### 2.8 MMR (Maximal Marginal Relevance)
**Mục tiêu**: chọn context vừa liên quan vừa đa dạng để giảm trùng lặp.
- Nếu chọn chunk A rồi, chunk B sẽ bị trừ điểm nếu quá giống A.

Công thức (phiên bản dùng dot product do vector đã chuẩn hoá):
$$\text{MMR}(d)=\lambda \cdot \text{relevance}(d) - (1-\lambda)\cdot \max_{d'\in S}\text{sim}(d,d')$$

Trong đó:
- $S$ là tập đã chọn.
- $\lambda$ là `MMR_LAMBDA`.

### 2.9 Grounding prompt + trích dẫn nguồn
Khi gọi LLM, prompt được xây theo dạng:
- Nêu rõ **chỉ được dùng thông tin trong context**.
- Nếu thiếu thông tin, phải nói thiếu dữ liệu.
- Gắn nhãn `[Nguon 1]`, `[Nguon 2]`... để trace.

### 2.10 Local synthesis fallback
Nếu không có API key (hoặc lỗi gọi API), hệ thống:
- Tóm tắt các chunk được truy xuất.
- Nhắc người dùng cấu hình API nếu muốn câu trả lời “tự nhiên” hơn.

### 2.11 Ổn định môi trường chạy (Windows)
Module runtime cấu hình:
- Tắt TensorFlow trong transformers để tránh xung đột (Keras 3).
- Loại bỏ user-site khỏi `sys.path` để tránh shadow package trong venv/conda.

---

## 3) Luồng xử lý end-to-end

### 3.1 Ingestion
1. Đọc file (PDF/TXT/MD).
2. Clean text.
3. Chunking + overlap.
4. Embedding từng chunk.
5. Ghi vào SQLite.
6. Rebuild index ANN từ toàn bộ embedding trong DB.

### 3.2 Retrieval
1. Tạo biến thể câu hỏi.
2. Embedding query.
3. ANN search lấy candidates.
4. Lấy chunk text từ SQLite.
5. Hybrid rerank + lọc ngưỡng.
6. MMR chọn top-k.

### 3.3 Generation
1. Xây prompt có context.
2. Gọi OpenAI hoặc Gemini (nếu chọn provider và có key).
3. Nếu lỗi/không có key → local synthesis.

---

## 4) Ánh xạ “công nghệ/kỹ thuật” ↔ vị trí trong code

- UI Streamlit: `app.py`
- CLI ingestion: `ingest_cli.py`
- Cấu hình hệ thống & `.env`: `rag/config.py`
- Đọc PDF + chunking: `rag/text_processing.py`
- Embedding: `rag/embedding_service.py`
- SQLite repository: `rag/repository.py`
- ANN HNSW + fallback LSH: `rag/vector_store.py`
- Retrieval + hybrid rerank + MMR: `rag/retriever.py`
- LLM clients + grounding: `rag/llm_clients.py`
- Orchestration end-to-end: `rag/pipeline.py`
- Runtime environment hardening (Windows): `rag/runtime_env.py`

Ghi chú: thư mục `app/` trong project hiện là skeleton (các file rỗng), chưa tham gia luồng chạy chính.

---

## 5) Cấu hình quan trọng
Trong `.env` (đề xuất):
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `ANN_CANDIDATES`, `TOP_K`
- `MIN_SIMILARITY`, `MMR_LAMBDA`
- `LLM_PROVIDER=local|openai|gemini`
- `OPENAI_API_KEY`, `GEMINI_API_KEY`

---

## 6) Hạn chế & hướng cải thiện
- SQLite + rebuild index mỗi lần ingest phù hợp quy mô nhỏ; dữ liệu lớn nên:
  - incremental update index,
  - tách pipeline ingest thành background worker.
- LSH fallback hiện chỉ ở RAM (không persist ra file).
- Lexical overlap dùng stopwords đơn giản; có thể thay bằng BM25 hoặc tokenizer tiếng Việt.

---

## Phụ lục A — Chạy và kiểm tra nhanh
- UI: `streamlit run app.py`
- CLI: `python ingest_cli.py documents`
