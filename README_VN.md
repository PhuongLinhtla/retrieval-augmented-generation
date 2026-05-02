# 📚 LightRAG Cho Tài Liệu Nội Bộ

Hệ thống Retrieval-Augmented Generation (RAG) chạy hoàn toàn **local**, **miễn phí**, không cần API key!

## ✨ Tính Năng

- ✅ **Hoàn toàn local** - Dữ liệu không upload lên cloud
- ✅ **Miễn phí** - Ollama + Neo4j + Embedding (tất cả free)
- ✅ **Giao diện web** - Upload & query qua WebUI
- ✅ **Knowledge Graph** - Xây dựng đồ thị tri thức từ tài liệu
- ✅ **Hỗ trợ đa định dạng** - PDF, TXT, DOCX, v.v.
- ✅ **Truy vấn thông minh** - Local, Global, Hybrid modes

## 🚀 Bắt Đầu Nhanh (5 phút)

### Yêu Cầu
- Docker + Docker Compose
- Ollama
- Python 3.10+

### 1️⃣ Cài Ollama (nếu chưa có)

**Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:** Download từ https://ollama.ai/download

### 2️⃣ Khởi Động Ollama

```bash
# Terminal 1
ollama serve

# Terminal 2: Pull models (đợi ~5 phút)
ollama pull mistral
ollama pull nomic-embed-text
```

### 3️⃣ Chạy Startup Script

```bash
cd /home/hainguyen/Documents/LightRAG
chmod +x startup.sh
./startup.sh
```

**Script sẽ:**
- ✅ Kiểm tra Ollama, Docker
- ✅ Cài Python dependencies
- ✅ Khởi động Neo4j + Embedding service
- ✅ Tạo thư mục tài liệu

### 4️⃣ Chạy WebUI Server

```bash
source .venv/bin/activate
lightrag-server
```

Mở browser: **http://localhost:8000**

### 5️⃣ Upload Tài Liệu

1. Trên WebUI → Tab **"Documents"**
2. Click **"Upload"** → Chọn file PDF/TXT
3. Click **"Process"** → Chờ xây dựng knowledge graph
4. Tab **"Retrieval"** → Nhập câu hỏi → "Send"

## 📖 Cách Sử Dụng

### Via WebUI (Dễ nhất)
```
http://localhost:8000
├── Documents Tab
│   ├── Upload tài liệu
│   ├── Xem danh sách
│   └── Xóa tài liệu
├── Knowledge Graph Tab
│   ├── Xem entities
│   ├── Xem relationships
│   └── Tìm kiếm trong graph
└── Retrieval Tab
    ├── Nhập câu hỏi
    ├── Chọn query mode
    └── Xem kết quả
```

### Via Python Script

```python
import asyncio
from lightrag import LightRAG, QueryParam

async def main():
    # Khởi tạo RAG
    rag = LightRAG(working_dir="./internal_docs")
    await rag.initialize_storages()
    
    # Thêm tài liệu
    with open("company_policy.txt") as f:
        await rag.ainsert(f.read())
    
    # Truy vấn
    result = await rag.aquery(
        "Chính sách nghỉ phép của công ty?",
        param=QueryParam(mode="hybrid")  # hybrid hoặc local, global, mix
    )
    
    print(result)

asyncio.run(main())
```

### Via API (FastAPI)

```bash
# Bắt đầu server
lightrag-server

# Upload tài liệu
curl -X POST "http://localhost:8000/api/documents" \
  -F "file=@document.pdf"

# Truy vấn
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Câu hỏi của bạn",
    "mode": "hybrid"
  }'

# API docs
http://localhost:8000/docs
```

## 🎯 Query Modes

| Mode | Khi nào dùng | Ưu điểm | Nhược điểm |
|------|-------------|--------|-----------|
| **local** | Tìm entity cụ thể | Nhanh, chính xác | Cần entity phù hợp |
| **global** | Tìm kiếm rộng | Toàn diện | Chậm hơn |
| **hybrid** | Mặc định | Cân bằng | Tốc độ trung bình |
| **mix** | Kết hợp graph + vector | Chất lượng cao | Chậm nhất |
| **naive** | Fallback | Đơn giản | Kém chính xác |

## 📊 Giám Sát

### Neo4j Dashboard
```
http://localhost:7474
Username: neo4j
Password: lightrag_password_2024
```

Query Cypher để xem knowledge graph:
```cypher
MATCH (n) RETURN n LIMIT 100
```

### Logs
```bash
# LightRAG logs
tail -f ./internal_docs/*.log

# Docker logs
docker-compose -f docker-compose.local.yml logs -f
```

## 🛠️ Cấu Hình

File `.env` trong thư mục gốc:

```bash
# LLM Model
LLM_MODEL_FUNC="ollama_complete"
OLLAMA_MODEL_NAME="mistral"

# Embedding
EMBEDDING_FUNC="ollama_embed"
EMBEDDING_MODEL_NAME="nomic-embed-text"

# Storage
GRAPH_STORAGE_TYPE="neo4j"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="lightrag_password_2024"

# Server
LIGHTRAG_SERVER_PORT="8000"
LIGHTRAG_WORKING_DIR="./internal_docs"
```

## 💡 Tối Ưu Hiệu Năng

### Tăng Tốc
```bash
# Dùng model nhỏ hơn
ollama pull neural-chat  # Nhanh hơn mistral

# Giảm chunk size trong code
# Mặc định: 1500 tokens → Thay đổi thành: 800
```

### Tiết Kiệm RAM
```bash
# Embedding model nhẹ hơn
# Thay nomic-embed-text bằng:
# sentence-transformers/all-MiniLM-L6-v2

# Trong docker-compose.local.yml:
# command: --model sentence-transformers/all-MiniLM-L6-v2
```

### Cải Thiện Chất Lượng
```bash
# Dùng model lớn
ollama pull mistral:7b     # 26GB, chất lượng cao
ollama pull llama2:13b     # Tương tự mistral

# Enable reranking (trong config)
ENABLE_RERANKING=true
```

## 🚨 Troubleshooting

### ❌ "Ollama not running"
```bash
# Terminal mới: Khởi động Ollama
ollama serve
```

### ❌ "Neo4j connection refused"
```bash
# Restart Docker services
docker-compose -f docker-compose.local.yml restart neo4j
```

### ❌ "Out of memory"
```bash
# Giảm model size hoặc tăng swap
# Hoặc dùng CPU thay vì GPU

# Trong docker-compose.local.yml, comment GPU section
```

### ❌ "Slow queries"
```bash
# Kiểm tra CPU/RAM
htop  # hoặc Task Manager trên Windows

# Dùng model nhỏ hơn
ollama pull neural-chat  # Thay mistral
```

## 📁 Thư Mục Cấu Trúc

```
LightRAG/
├── .env                    # Cấu hình
├── startup.sh              # Script khởi động
├── docker-compose.local.yml # Docker compose
├── SETUP_LOCAL_GUIDE.md    # Hướng dẫn chi tiết
├── demo_local.py           # Script demo
├── lightrag/               # Main package
├── lightrag_webui/         # WebUI React app
├── internal_docs/          # Tài liệu của bạn
│   ├── chunks/
│   ├── entities/
│   └── *.txt (tài liệu upload)
└── .venv/                  # Python environment
```

## 🔒 Bảo Mật

### Local Only
- ✅ Dữ liệu lưu local (`./internal_docs`)
- ✅ Không kết nối internet (trừ nếu dùng remote LLM)
- ✅ Neo4j mã hóa password

### Recommendations
```bash
# 1. Backup định kỳ
tar -czf internal_docs_backup.tar.gz ./internal_docs

# 2. Thay đổi password Neo4j
docker-compose -f docker-compose.local.yml exec neo4j \
  cypher-shell -u neo4j -p lightrag_password_2024 \
  "ALTER USER neo4j SET PASSWORD 'NEW_PASSWORD'"

# 3. Không expose port public (hoặc add authentication)
```

## 📞 Hỗ Trợ

- 📖 **Docs**: https://github.com/HKUDS/LightRAG
- 🐛 **Issues**: https://github.com/HKUDS/LightRAG/issues
- 💬 **Discord**: https://discord.gg/yF2MmDJyGJ
- 📧 **GitHub Discussions**: https://github.com/HKUDS/LightRAG/discussions

## 📄 License

MIT License - xem [LICENSE](LICENSE)

---

**Happy RAG! 🚀**

Làm cho việc quản lý tài liệu nội bộ trở nên dễ dàng hơn! 💪
