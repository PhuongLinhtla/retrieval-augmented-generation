# RAG Learning Assistant (Toi uu cho tieng Viet va 6GB VRAM)

He thong RAG nay duoc toi uu de chay on dinh tren may GPU nho (khoang 6GB VRAM), uu tien tai lieu tieng Viet va hoc tap noi bo.

## 0) Accuracy-first architecture (ban nang cap)

He thong da nang cap theo huong tang do chinh xac:

- Chunking 2 tang child-parent theo token budget.
- Hybrid retrieval (Dense + BM25) va RRF fusion.
- Reranker cross-encoder (co fallback neu khong kha dung).
- Evidence compression sentence-level (tuy chon).
- Confidence gate: khi do tin cay thap se hoi lam ro thay vi doan.
- Prompt 2-pass: facts truoc, tong hop sau, bat buoc citation.

Tai lieu thiet ke chi tiet:

- `docs/RAG_Accuracy_First_Design_vi.md`

## 1) Chien luoc model cho 6GB VRAM

- LLM uu tien #1 (chat): Ollama + Qwen 2.5 14B Instruct (GGUF Q4).
  - Goi y: `qwen2.5:14b-instruct-q4_K_M`
  - Muc tieu: tang chat luong tong hop/giai thich khi retrieval tot.
- Neu OOM hoac qua cham: fallback xuong Llama 3.1 8B Instruct.
  - Goi y: `llama3.1:8b-instruct-q5_K_M`
  - Muc tieu: chay muot hon, giu duoc context cao hon tren may 12GB.
- Vision model (doc anh/bieu do): Ollama + Moondream hoac Qwen2-VL 2B.
  - Mac dinh: `moondream`
- Embedding model:
  - Nhanh/nhẹ, da tot cho tieng Viet: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - Neu muon dong bo qua Ollama: `nomic-embed-text`

## 2) Engine van hanh

- Uu tien Ollama (llama.cpp) thay vi vLLM cho may nho.
- Ollama thuong quan ly bo nho on dinh hon tren may RAM/VRAM han che.

## 3) Pipeline tach 2 pha (rat quan trong)

### Pha 1 - Ingestion offline
Chi chay khi can nap tai lieu moi:

- Trich xuat text tu PDF/TXT/MD/PPTX.
- Co the doc anh/bieu do bang vision model (Moondream) va chuyen thanh text.
- Tao embedding va luu vao DB/index.

Script da co:

- `scripts/phase1_ingest_offline.py`

### Pha 2 - Chat realtime
Chi chay luc ban ngoi hoc hoi dap:

- Bat model chat Qwen qua Ollama.
- Retrieval doc chunk tu SQLite + ANN index (CPU/RAM).
- Sinh cau tra loi tieng Viet co kem nguon.

Script da co:

- `scripts/phase2_chat_serve.py`

## 4) Cau hinh .env

Tham khao file `.env.example`.

Gia tri mac dinh cho 6GB:

- `LLM_PROVIDER=ollama`
- `OLLAMA_LLM_MODEL=qwen2.5:14b-instruct-q4_K_M`
- `OLLAMA_FALLBACK_MODEL=llama3.1:8b-instruct-q5_K_M`
- `OLLAMA_VISION_MODEL=moondream`
- `EMBEDDING_BACKEND=sentence_transformers`
- `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `OLLAMA_EMBED_MODEL=nomic-embed-text`
- `OLLAMA_NUM_CTX=8192`
- `EVIDENCE_MIN_TOKENS=1500` va `EVIDENCE_MAX_TOKENS=3000`
- `EVIDENCE_CHUNK_MIN=6` va `EVIDENCE_CHUNK_MAX=10`
- `OLLAMA_TEMPERATURE=0.0-0.2`

## 5) Cai dat

1. Tao va kich hoat venv cua ban.
2. Cai package:

   `pip install -r requirements.txt`

3. Cai Ollama va pull model:

  `ollama pull qwen2.5:14b-instruct-q4_K_M`

  `ollama pull llama3.1:8b-instruct-q5_K_M`

   `ollama pull moondream`

   `ollama pull nomic-embed-text`

## 6) Chay theo 2 pha

### Pha 1 - Nap du lieu offline

Mac dinh ingest thu muc `documents`:

`python scripts/phase1_ingest_offline.py`

Ingest duong dan cu the:

`python scripts/phase1_ingest_offline.py documents my_slide.pptx my_file.pdf`

Tat vision neu muon nhanh hon:

`python scripts/phase1_ingest_offline.py --disable-vision`

Tich hop Marker/Nougat (tuy chon) qua command template:

`python scripts/phase1_ingest_offline.py --marker-cmd "marker_single {input} --output_dir {output_dir}"`

`python scripts/phase1_ingest_offline.py --nougat-cmd "nougat {input} -o {output_dir}"`

### Pha 2 - Chat realtime

`python scripts/phase2_chat_serve.py --port 8511`

Sau do mo trinh duyet:

`http://localhost:8511`

## 7) Chay truc tiep bang Streamlit

Neu khong dung script pha 2, co the chay truc tiep:

`python -m streamlit run app.py --server.port 8511`

## 8) Kha nang tieng Viet da ap dung

- Embedding da uu tien model multilingual cho tieng Viet.
- Retrieval lexical da them chuan hoa bo dau (khong dau/co dau) de match tot hon.
- Prompt generation uu tien cau tra loi tieng Viet va trace nguon.

## 9) Cau truc chinh

- `app.py`: giao dien Streamlit.
- `ingest_cli.py`: ingest co ban.
- `scripts/phase1_ingest_offline.py`: pha 1 offline.
- `scripts/phase2_chat_serve.py`: pha 2 chat.
- `rag/ollama_http.py`: client Ollama cho generate/embed/vision.
- `rag/embedding_service.py`: embedding backend `sentence_transformers` hoac `ollama`.
- `rag/llm_clients.py`: provider `ollama|openai|gemini|local`.
- `rag/text_processing.py`: parser PDF/TXT/MD/PPTX + chunking.
- `rag/retriever.py`: retrieval + rerank + MMR.
- `rag/pipeline.py`: orchestration end-to-end.

## 10) Luu y deploy

- He thong hien la local web app (localhost), khong tu dong public internet.
- Muon public cho nguoi ngoai truy cap, can deploy len cloud (Streamlit Community Cloud/Render/VPS).
- Khong commit `.env` va API key.
