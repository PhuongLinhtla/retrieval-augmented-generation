# 🚀 LightRAG Setup Cho Tài Liệu Nội Bộ (Hoàn Toàn Local)

## 📋 Yêu Cầu

- ✅ **Docker** (để chạy Neo4j + Embedding)
- ✅ **Ollama** (để chạy LLM local)
- ✅ **Python 3.10+** (bạn đã có 3.12)
- ✅ **4GB RAM tối thiểu** (tốt hơn 8GB+)

## 🎯 Kiến Trúc

```
┌─────────────────────┐
│  Tài liệu nội bộ    │
│  (PDF, TXT, DOCX)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────┐
│   LightRAG (Python)         │
│  - Extract Entities         │
│  - Build Knowledge Graph    │
└────┬────────────────┬───────┘
     │                │
     ▼                ▼
┌──────────────┐  ┌──────────────────┐
│   Ollama     │  │  Neo4j Database  │
│ (LLM Local)  │  │  (Graph Storage) │
│ Port: 11434  │  │  Port: 7687      │
└──────────────┘  └──────────────────┘
     │
     ▼
┌─────────────────────────┐
│  WebUI (React)          │
│  http://localhost:8000  │
└─────────────────────────┘
```

## 🔧 Bước 1: Cài Đặt Ollama

### 1.1 Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 1.2 macOS
```bash
# Download từ https://ollama.ai/download/Ollama-darwin.zip
unzip Ollama-darwin.zip
sudo mv Ollama.app /Applications/
```

### 1.3 Windows
```
Download từ https://ollama.ai/download/OllamaSetup.exe
```

## 🌐 Bước 2: Chạy Ollama & Pull Models

```bash
# Terminal 1: Chạy Ollama server
ollama serve

# Terminal 2: Pull models
ollama pull mistral        # LLM chính (4.1GB, nhanh, tốt)
ollama pull nomic-embed-text   # Embedding model

# (Tùy chọn) Models khác:
ollama pull neural-chat    # Nhỏ hơn, nhanh hơn
ollama pull llama2          # Tương tự Mistral
```

**Dung lượng:**
- `mistral`: 4.1 GB
- `nomic-embed-text`: 274 MB
- **Tổng: ~5 GB**

## 🐳 Bước 3: Chạy Docker Services (Neo4j + Embedding)

```bash
cd /home/hainguyen/Documents/LightRAG

# Khởi động containers
docker-compose -f docker-compose.local.yml up -d

# Kiểm tra status
docker-compose -f docker-compose.local.yml ps

# Xem logs (nếu có vấn đề)
docker-compose -f docker-compose.local.yml logs -f
```

**Kiểm tra kết nối:**
```bash
# Neo4j Dashboard (mở browser)
http://localhost:7474
# Username: neo4j
# Password: lightrag_password_2024

# Embedding API (test)
curl http://localhost:8001/v1/models
```

## 📦 Bước 4: Cài Đặt LightRAG Python Package

```bash
cd /home/hainguyen/Documents/LightRAG

# Option A: Dùng uv (nhanh hơn)
uv sync

# Option B: Dùng pip
python3 -m pip install -e .
```

## ✅ Bước 5: Test Setup

```bash
# Kích hoạt virtual environment
source .venv/bin/activate

# Chạy demo
python3 demo_local.py
```

**Kết quả mong đợi:**
```
✅ Ollama is running
✅ Prerequisites checked
📄 Reading document: ./internal_docs/sample_doc.txt
🔄 Processing document...
✅ Document inserted into knowledge graph!
🔍 Querying: 'LightRAG là gì?'
📊 Query Result: ...
```

## 🌐 Bước 6: Chạy WebUI Server

```bash
# Kích hoạt virtual environment nếu chưa
source .venv/bin/activate

# Chạy API server
lightrag-server
```

**Truy cập:**
- WebUI: `http://localhost:8000`
- API: `http://localhost:8000/docs`

## 📚 Bước 7: Upload Tài Liệu Nội Bộ

### Via WebUI:
1. Mở `http://localhost:8000`
2. Click **"Upload"** tab
3. Kéo thả tệp PDF/TXT/DOCX
4. Click **"Process"** và đợi

### Via Python Script:
```python
import asyncio
from lightrag import LightRAG

rag = LightRAG(working_dir="./internal_docs")
await rag.initialize_storages()

with open("my_document.txt") as f:
    await rag.ainsert(f.read())

# Query
result = await rag.aquery("Câu hỏi về tài liệu")
print(result)
```

## 🔍 Sử Dụng LightRAG

### Query Modes:
```python
from lightrag import QueryParam

# 1. Local: Tìm entity cụ thể
result = await rag.aquery("...", param=QueryParam(mode="local"))

# 2. Global: Tìm kiếm rộng
result = await rag.aquery("...", param=QueryParam(mode="global"))

# 3. Hybrid: Kết hợp local + global (khuyến nghị)
result = await rag.aquery("...", param=QueryParam(mode="hybrid"))

# 4. Mix: Dùng vector + graph
result = await rag.aquery("...", param=QueryParam(mode="mix"))
```

## 🛠️ Troubleshooting

### ❌ "Ollama is not running"
```bash
# Terminal mới: Khởi động Ollama
ollama serve
```

### ❌ "Neo4j connection failed"
```bash
# Kiểm tra containers
docker-compose -f docker-compose.local.yml logs neo4j

# Restart
docker-compose -f docker-compose.local.yml restart neo4j
```

### ❌ "Embedding model not found"
```bash
# Kiểm tra image
docker images | grep vllm

# Kiểm tra port 8001
curl http://localhost:8001/v1/models
```

### ❌ "Memory/GPU issues"
```bash
# Giảm model size
# Thay trong docker-compose.local.yml:
# FROM: nomic-ai/nomic-embed-text-v1.5
# TO: sentence-transformers/all-MiniLM-L6-v2 (nhỏ hơn, nhanh hơn)
```

## 📊 Monitoring

### Xem Knowledge Graph:
```bash
# Neo4j Browser
http://localhost:7474

# Query Cypher:
MATCH (n) RETURN n LIMIT 50
```

### Logs:
```bash
# LightRAG logs
tail -f ./internal_docs/lightrag.log

# Docker logs
docker-compose -f docker-compose.local.yml logs -f
```

## 🎓 Tài Liệu Tham Khảo

- 📖 [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- 📋 [Ollama Models](https://ollama.ai/library)
- 🗄️ [Neo4j Documentation](https://neo4j.com/docs/)
- 🐳 [Docker Compose](https://docs.docker.com/compose/)

## ⚡ Optimization Tips

1. **Tăng tốc độ:**
   - Dùng `mistral` hoặc `neural-chat` (nhỏ, nhanh)
   - Tăng `top_k` trong query (mặc định 60)

2. **Tiết kiệm RAM:**
   - Dùng `all-MiniLM-L6-v2` thay vì `nomic-embed-text`
   - Giảm `chunk_size` (mặc định 1500)

3. **Cải thiện chất lượng:**
   - Dùng model lớn hơn (llama2, mistral-7b)
   - Điều chỉnh `similarity_threshold`
   - Enable reranking (nếu có resource)

## 💾 Backup & Recovery

```bash
# Backup knowledge graph
docker-compose -f docker-compose.local.yml exec neo4j \
  neo4j-admin database dump neo4j \
  --to-path=/backup

# Backup tài liệu
tar -czf internal_docs_backup.tar.gz ./internal_docs

# Restore
docker-compose -f docker-compose.local.yml exec neo4j \
  neo4j-admin database load neo4j \
  --from-path=/backup
```

---

**Chúc bạn thành công! 🚀**

Nếu gặp vấn đề, hãy kiểm tra logs hoặc tạo issue trên GitHub.
