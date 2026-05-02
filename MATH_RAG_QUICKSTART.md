# Math Learning RAG Setup - Quick Start

## Overview
Hệ thống RAG chuyên biệt cho học tập toán với khả năng nhận dạng công thức toán học (LaTeX).

**Giải quyết vấn đề:**
- ✅ OCR PDF toán có công thức (dùng **Nougat**)
- ✅ Chunking thông minh (giữ công thức với context)
- ✅ Embedding optimized cho toán học
- ✅ Q&A step-by-step với LaTeX formulas

---

## Quick Start (5 phút)

### 1. Setup Nougat OCR
```bash
cd ~/Documents/LightRAG
source .venv/bin/activate

# Cài Nougat (Meta's scientific OCR)
./setup_math_rag.sh
```

### 2. Copy PDF Files & Process
```bash
# Copy files to ocr_input/
cp ~/Downloads/*.pdf ./ocr_input/

# Batch OCR + chunking
./process_math_docs.sh
# Output: internal_docs/ (ready for ingestion)
```

### 3. Ingest & Query
**Option A: Via WebUI** (recommended)
```bash
# 1. Ensure server running
lightrag-server
# 2. Open http://localhost:8000
# 3. Upload processed files via WebUI
```

**Option B: Via Python**
```bash
# Run demo
python demo_math_rag.py
# Choose: 1 (demo queries) or 2 (interactive Q&A)
```

---

## What's Included

| File | Purpose |
|------|---------|
| `setup_math_rag.sh` | Install Nougat OCR |
| `process_math_docs.sh` | Batch OCR + intelligent chunking |
| `math_doc_processor.py` | Smart formula extraction & chunking |
| `demo_math_rag.py` | End-to-end demo (ingest + query) |
| `MATH_RAG_GUIDE.md` | Detailed documentation |

---

## Workflow

```
PDF (scan/digital with formulas)
  ↓
[Nougat OCR] → Markdown with LaTeX ($$...$$)
  ↓
[math_doc_processor] → Smart chunks + formula index
  ↓
[LightRAG ingestion] → Knowledge graph
  ↓
[Query + LLM] → Answer with step-by-step + formulas
```

---

## Example Queries

```
"Giải bài toán: Tìm x sao cho x^2 - 5x + 6 = 0"
→ Answer: Phương trình $x^2 - 5x + 6 = 0$ được giải như sau...

"Định nghĩa hàm số bậc hai"
→ Answer: Hàm số bậc hai có dạng $$f(x) = ax^2 + bx + c$$...

"Chứng minh: tổng 3 góc tam giác = 180°"
→ Answer: Chứng minh chi tiết step-by-step...
```

---

## Troubleshooting

### "Nougat installation failed"
```bash
# Ensure Python 3.8+
python3 --version

# Try manual install
pip install --upgrade nougat-ocr
```

### "No PDF files found in ocr_input/"
```bash
mkdir -p ocr_input
cp /path/to/your/*.pdf ocr_input/
./process_math_docs.sh
```

### "Server not responding"
```bash
# Ensure server running
source .venv/bin/activate
lightrag-server &
sleep 2
curl http://localhost:8000/health
```

---

## Configuration (.env)

Key settings for math RAG:
```env
# Retrieval
TOP_K_RETRIEVAL=10          # Fetch more context for math
RETRIEVAL_MODE=mix          # Hybrid search (BM25 + vector)

# LLM (mistral is good for math)
LLM_TEMPERATURE=0.0         # Deterministic (exact formulas)
LLM_MAX_TOKENS=1000         # More space for step-by-step

# Embedding
EMBEDDING_MODEL=nomic-embed-text  # Default; can upgrade to mxbai-embed-large
```

---

## Advanced: Custom Embedding Model

For better math performance, use **mixedbread-ai/mxbai-embed-large**:

```bash
# Edit .env
EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large

# Note: Requires Ollama to pull the model, or use API
```

---

## Support & Docs

- **Full guide**: `MATH_RAG_GUIDE.md`
- **Prompt templates**: See MATH_RAG_GUIDE.md → "User Prompts Templates"
- **Source code**: `math_doc_processor.py`, `demo_math_rag.py`

---

## Next Steps

1. ✅ Setup complete. Copy PDFs to `ocr_input/`
2. Run `./process_math_docs.sh`
3. Open WebUI: http://localhost:8000
4. Start asking math questions!

**Happy learning! 🎓**
