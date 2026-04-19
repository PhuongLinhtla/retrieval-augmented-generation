<<<<<<< HEAD
# RAG Learning Assistant (Đề tài 9)

Hệ thống RAG (Retrieval-Augmented Generation) cho trợ lý học tập:
- Nhúng tài liệu PDF/TXT/MD bằng SentenceTransformers.
- Lưu metadata + vector vào SQLite.
- Dùng ANN + cosine similarity để truy xuất ngữ cảnh (context).
  - Ưu tiên backend HNSW (hnswlib) khi sẵn sàng.
  - Tự động fallback sang ANN-LSH nếu hnswlib không khả dụng (thường gặp trên Windows khi thiếu C++ Build Tools).
- Lấy top-3 ngữ cảnh có ý nghĩa gần nhất.
- Có giao diện chat và khối Nguồn tham khảo (Context) bên dưới mỗi câu trả lời.
- Hỗ trợ OpenAI/Gemini để tạo câu trả lời bám sát dữ liệu nội bộ (grounded answer).

## 1) Kiến trúc

- Thu nạp dữ liệu (Ingestion):
  1. Đọc file PDF/TXT/MD.
  2. Chia đoạn (chunking) có overlap.
  3. Nhúng từng chunk bằng SentenceTransformers.
  4. Lưu chunk + vector vào SQLite.
  5. Rebuild ANN index (HNSW hoặc LSH fallback) từ vector trong DB.

- Truy xuất (Retrieval - RAG):
  1. Nhúng câu hỏi.
  2. Tìm kiếm ANN theo cosine trong HNSW hoặc LSH fallback.
  3. Hybrid rerank = semantic similarity + lexical overlap.
  4. MMR selection để chọn top-3 ngữ cảnh đa dạng, giảm trùng lặp.

- Sinh câu trả lời (Generation):
  1. Đưa top-3 ngữ cảnh vào prompt.
  2. Gọi OpenAI/Gemini (nếu có API key) hoặc local synthesis fallback.
  3. Ép trả lời dựa trên context; nếu thiếu dữ liệu thì nêu rõ.

## 2) Cấu trúc thư mục

- app.py: Giao diện chatbot Streamlit + truy vết nguồn.
- ingest_cli.py: CLI để ingest tài liệu.
- rag/config.py: Biến môi trường và tham số hệ thống.
- rag/text_processing.py: Parser + chunking.
- rag/repository.py: SQLite metadata + vector.
- rag/vector_store.py: ANN index (HNSW + LSH fallback).
- rag/retriever.py: Truy xuất + rerank + MMR.
- rag/llm_clients.py: OpenAI/Gemini/local grounded answer.
- rag/pipeline.py: Điều phối end-to-end.
- storage/: SQLite DB + HNSW index.
- documents/: Đặt tài liệu nội bộ để index.

## 3) Cài đặt

1. Tạo và kích hoạt virtual environment (khuyến nghị).
2. Cài package:

   pip install -r requirements.txt

   Trên Windows, hnswlib được đặt ở chế độ optional để tránh lỗi build C++.
   Hệ thống vẫn chạy ANN bằng backend LSH fallback.

3. Tạo file .env từ .env.example và điền API key nếu cần:

   - LLM_PROVIDER=local | openai | gemini
   - OPENAI_API_KEY=...
   - GEMINI_API_KEY=...

4. Bật secret guard trước khi push lên GitHub:

   - Sau khi `git init`, chạy:

     powershell -ExecutionPolicy Bypass -File scripts/setup_git_hooks.ps1

   - Hook `pre-commit` và `pre-push` sẽ quét secret trong file và chặn commit/push nếu phát hiện key/token.
   - Không bao giờ commit file `.env`. Chỉ commit `.env.example`.

## 4) Chạy hệ thống

- Chạy UI Streamlit:

  streamlit run app.py

- Trong sidebar:
  - Upload file PDF/TXT/MD và bấm Embed uploaded files.
  - Hoặc bấm Index documents folder để index toàn bộ thư mục documents.

- Chat:
  - Đặt câu hỏi trong khung chat.
  - Bên dưới mỗi câu trả lời của bot có khối Nguồn tham khảo (Context) để click mở.
  - Khối này hiển thị:
    - Nguồn file/trang
    - Điểm tương đồng (similarity score)
    - Đoạn văn bản gốc truy xuất từ DB

## 5) Chạy ingest bằng CLI

- Index file/thư mục cụ thể:

  python ingest_cli.py documents

- Hoặc:

  python ingest_cli.py path_to_file.pdf path_to_folder

## 6) Tùy chỉnh và tối ưu

Trong file .env:
- CHUNK_SIZE: Kích thước chunk.
- CHUNK_OVERLAP: Độ overlap giữa các chunk.
- ANN_CANDIDATES: Số ứng viên ANN ban đầu.
- TOP_K: Số context sau cùng (mặc định 3).
- MIN_SIMILARITY: Ngưỡng lọc context yếu.
- MMR_LAMBDA: Cân bằng relevance và diversity.

## 7) Cải tiến thuật toán đã áp dụng

- ANN HNSW cho tìm kiếm vector nhanh trên dữ liệu lớn.
- Cosine similarity trên embedding đã normalize.
- Hybrid rerank semantic + lexical để tăng độ chính xác.
- MMR để giảm trùng lặp context, tăng độ bao phủ thông tin.
- Prompt grounding bắt buộc dựa trên context, giảm hallucination.
- Fallback local synthesis khi API bên ngoài không sẵn sàng.

## 8) Lưu ý

- Dữ liệu tham khảo hiển thị trực tiếp từ chunk trong DB.
- LLM có thể fallback local nếu API key không hợp lệ.
- Nên dùng model embedding đa ngôn ngữ (multilingual) cho tài liệu tiếng Việt.
- Nếu muốn bật HNSW trên Windows, cài Microsoft C++ Build Tools rồi chạy thêm: pip install hnswlib

## 9) Khắc phục sự cố (Troubleshooting)

- Lỗi ModuleNotFoundError: No module named 'torchvision' khi chạy Streamlit:
  - Nguyên nhân: Streamlit file watcher quét lazy modules của transformers (không phải lỗi logic RAG).
  - Cấu hình đã được đặt sẵn trong [.streamlit/config.toml](.streamlit/config.toml) với `fileWatcherType = "none"`.
  - Nếu đang chạy app cũ, hãy dừng server và chạy lại để áp dụng cấu hình mới.
=======
rag_project/ ├── app/ │ ├── __init__.py │ ├── main.py │ # FastAPI entry point & WebSocket chat endpoint
## retrieval-augmented-generation
A small FastAPI-based RAG chatbot system with vector search integration.
 It provides a WebSocket-based chat interface, processes user queries by generating embeddings, retrieves the most relevant text chunks using vector similarity search, and returns responses with source context and similarity scores.

## Features
 - FastAPI WebSocket endpoint for real-time chatbot interaction
 - Embedding generation using SentenceTransformers
 - Vector similarity search to retrieve top-k relevant text chunks
 - RAG pipeline combining retrieval and response generation
 - Source tracing with similarity scores for each response
 - Lightweight demo frontend for chat interaction

## Quickstart (local)
 1.Create and activate a virtual environment:
    python3 -m venv .venv source .venv/bin/activate  

 2.Install dependencies:
     pip install -r requirements.txt  

 3.Provide environment variables (see section below) in a .env file or your shell.

 4.Start services with Docker Compose (optional for local development):
 docker-compose up -d
 This can be used to run supporting services such as a vector database or API backend if configured in docker-compose.yml.

 Run the FastAPI app:
 uvicorn app.main:app --reload --host 0.0.0.0 --port 8000  
 Connect to the WebSocket endpoint:
 ws://localhost:8000/ws  
 Use a simple frontend or WebSocket client to send queries and receive responses with context and similarity scores.

## Environment variables
Keep sensitive values out of source control (use `.env` and `.gitignore`).

Common variables used in this project:

- VECTOR_DB_API_KEY - API key for vector database (e.g., Pinecone)
- VECTOR_DB_ENV - Environment/region for vector database
- OPENAI_API_KEY - API key for LLM (optional)
- APP_ENV - Application environment (development | production)

Adjust configuration in your application files as needed.

## Project Layout

```text
rag_project/
├── app/                     # Application code (FastAPI + RAG pipeline)
│   ├── main.py              # FastAPI app & WebSocket chat endpoint
│   ├── rag_pipeline.py      # Core RAG logic (embedding + retrieval + response)
│   ├── embedding.py         # Text embedding using SentenceTransformers
│   ├── vector_db.py         # Vector search (e.g., Pinecone / Milvus)
│   ├── models/              # (Optional) Pydantic schemas
│   └── services/            # (Optional) data processing / helpers
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys, etc.)
└── README.md
```

## Development notes
- The core logic is implemented in the RAG pipeline (`app/rag_pipeline.py`), including embedding, vector retrieval, and response generation.
- Vector search is handled in `app/vector_db.py` (can be integrated with Pinecone or Milvus).
- Embedding is generated using SentenceTransformers (`app/embedding.py`).
- A simple WebSocket-based chat interface is provided via FastAPI.

## Testing
There are no automated tests included by default. Add tests under a tests/ folder and run with pytest.

## Git / .env Hygiene
- .env and virtual environment directories are intentionally excluded from version control. See .gitignore.

## License
This repository is for educational purposes. Add a LICENSE file if needed.
>>>>>>> d85ee869468c36146a66be379af217e0942f7569
