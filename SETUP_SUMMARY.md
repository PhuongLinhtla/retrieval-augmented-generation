# 📦 Setup Summary - LightRAG Cho Tài Liệu Nội Bộ

## ✨ Những Gì Đã Được Chuẩn Bị

Tôi đã tạo một **setup hoàn chỉnh** để bạn chạy LightRAG **miễn phí, hoàn toàn local** cho tài liệu nội bộ.

### 🎯 Cấu Hình Chọn
```
├── LLM: Ollama Local (mistral)
├── Embedding: Local Docker (nomic-embed-text)
├── Storage: Neo4j Graph Database
├── Server: WebUI (http://localhost:8000)
└── Tất cả: Miễn phí, không cần API key
```

---

## 📂 Files Được Tạo

### 1. **Cấu Hình** (`.env`)
```
→ Tệp: .env
→ Chứa: LLM settings, database config, server port
→ Chỉnh sửa: Nếu muốn đổi LLM model hoặc port
```

### 2. **Docker Compose** (`docker-compose.local.yml`)
```
→ Tệp: docker-compose.local.yml
→ Services: Neo4j + vLLM Embedding model
→ Dùng: docker-compose -f docker-compose.local.yml up -d
```

### 3. **Startup Script** (`startup.sh`)
```
→ Tệp: startup.sh (executable)
→ Tác dụng: 
  ✓ Kiểm tra Ollama, Docker
  ✓ Cài Python dependencies
  ✓ Khởi động services
  ✓ Tạo thư mục dữ liệu
→ Chạy: ./startup.sh
```

### 4. **Demo Script** (`demo_local.py`)
```
→ Tệp: demo_local.py (executable)
→ Tác dụng: Test setup + demo functionality
→ Chạy: python3 demo_local.py
```

### 5. **Hướng Dẫn Chi Tiết** (`SETUP_LOCAL_GUIDE.md`)
```
→ Tệp: SETUP_LOCAL_GUIDE.md
→ Chứa: Hướng dẫn 7 bước chi tiết
→ Giải thích: Kiến trúc, troubleshooting, optimization
```

### 6. **Hướng Dẫn Tiếng Việt** (`README_VN.md`)
```
→ Tệp: README_VN.md
→ Chứa: Hướng dẫn đầy đủ tiếng Việt
→ Bao gồm: Cách sử dụng, query modes, monitoring
```

### 7. **Quick Start** (`QUICKSTART.md`)
```
→ Tệp: QUICKSTART.md
→ Chứa: 5 bước khởi động nhanh (15 phút)
→ Dùng: Nếu muốn chạy ngay mà không đọc docs đầy đủ
```

---

## 🚀 Bắt Đầu Trong 5 Phút

### Bước 1: Cài Ollama (nếu chưa có)
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Bước 2: Khởi Động Ollama + Pull Models (5-10 phút)
```bash
# Terminal 1
ollama serve

# Terminal 2
ollama pull mistral
ollama pull nomic-embed-text
```

### Bước 3: Chạy Startup Script
```bash
cd /home/hainguyen/Documents/LightRAG
./startup.sh
```

### Bước 4: Chạy WebUI Server
```bash
source .venv/bin/activate
lightrag-server
```

### Bước 5: Mở Browser
```
http://localhost:8000
```

**Xong! 🎉**

---

## 📚 Cách Sử Dụng

### Via WebUI (Dễ nhất)
1. Upload tài liệu PDF/TXT
2. Chờ xây dựng knowledge graph
3. Nhập câu hỏi → Click "Send"
4. Nhận kết quả

### Via Python Script
```python
import asyncio
from lightrag import LightRAG, QueryParam

async def main():
    rag = LightRAG(working_dir="./internal_docs")
    await rag.initialize_storages()
    
    # Thêm tài liệu
    with open("document.txt") as f:
        await rag.ainsert(f.read())
    
    # Truy vấn
    result = await rag.aquery("Câu hỏi?", param=QueryParam(mode="hybrid"))
    print(result)

asyncio.run(main())
```

### Via REST API
```bash
# Upload
curl -X POST "http://localhost:8000/api/documents" \
  -F "file=@document.pdf"

# Query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Câu hỏi?", "mode": "hybrid"}'
```

---

## 📊 Services & Ports

| Service | URL | Mục đích |
|---------|-----|---------|
| **WebUI** | http://localhost:8000 | Giao diện web |
| **API Docs** | http://localhost:8000/docs | Swagger docs |
| **Neo4j Dashboard** | http://localhost:7474 | Xem graph |
| **Ollama API** | http://localhost:11434 | LLM API |
| **Embedding API** | http://localhost:8001 | Embedding API |

---

## 🎯 Query Modes

```python
# 1. Local - Tìm entity cụ thể (nhanh)
await rag.aquery("...", param=QueryParam(mode="local"))

# 2. Global - Tìm kiếm rộng (chậm)
await rag.aquery("...", param=QueryParam(mode="global"))

# 3. Hybrid - Kết hợp (khuyến nghị)
await rag.aquery("...", param=QueryParam(mode="hybrid"))

# 4. Mix - KG + Vector (chất lượng cao)
await rag.aquery("...", param=QueryParam(mode="mix"))

# 5. Naive - Pure vector search (cơ bản)
await rag.aquery("...", param=QueryParam(mode="naive"))
```

---

## 💾 Lưu Trữ & Backup

```bash
# Backup tài liệu
tar -czf internal_docs_backup.tar.gz ./internal_docs

# Backup Neo4j
docker-compose -f docker-compose.local.yml exec neo4j \
  neo4j-admin database dump neo4j --to-path=/backup

# Restore
tar -xzf internal_docs_backup.tar.gz
```

---

## 🆘 Troubleshooting

### ❌ Ollama không chạy
```bash
✓ Chạy: ollama serve
```

### ❌ Neo4j không kết nối
```bash
✓ Restart: docker-compose -f docker-compose.local.yml restart neo4j
```

### ❌ Port bị chiếm
```bash
# Đổi port trong .env
LIGHTRAG_SERVER_PORT="8001"
```

### ❌ Hết bộ nhớ
```bash
✓ Dùng model nhỏ: ollama pull neural-chat
✓ Hoặc tăng swap: swapon
```

---

## ⚙️ Cấu Hình Nâng Cao

### Thay Đổi LLM Model
```bash
# File: .env
OLLAMA_MODEL_NAME="neural-chat"  # Thay từ "mistral"
```

Các model phổ biến:
- `mistral` - Cân bằng (4.1GB, đang dùng)
- `neural-chat` - Nhanh (4.1GB)
- `llama2` - Tương tự mistral
- `mistral:7b` - Lớn (26GB, chất lượng cao)

### Thay Đổi Port
```bash
# File: .env
LIGHTRAG_SERVER_PORT="8001"
```

### Thay Đổi LLM Provider
```bash
# File: .env
# Từ: LLM_MODEL_FUNC="ollama_complete"
# Sang: LLM_MODEL_FUNC="gpt_4o_complete"  # Nếu có OpenAI API
```

---

## 📖 Tài Liệu Thêm

- 📘 **Hướng dẫn chi tiết**: `SETUP_LOCAL_GUIDE.md`
- 📗 **README tiếng Việt**: `README_VN.md`  
- 📙 **Quick start**: `QUICKSTART.md`
- 🔗 **GitHub**: https://github.com/HKUDS/LightRAG
- 📚 **Docs**: https://github.com/HKUDS/LightRAG#readme

---

## ✅ Checklist Cuối Cùng

Trước khi chạy:
- [ ] Docker installed & running
- [ ] Ollama installed
- [ ] Python 3.10+ (bạn có 3.12 ✓)
- [ ] 4GB+ RAM free
- [ ] 10GB disk space free

Khi chạy `./startup.sh`:
- [ ] Script kiểm tra Ollama ✓
- [ ] Script kiểm tra Docker ✓
- [ ] Script cài dependencies ✓
- [ ] Docker containers khởi động ✓

Khi chạy WebUI:
- [ ] Neo4j ready (port 7687)
- [ ] Embedding service ready (port 8001)
- [ ] WebUI accessible (port 8000)

---

## 🎉 Lợi Ích

| Lợi ích | Giá trị |
|--------|--------|
| **Hoàn toàn local** | Dữ liệu an toàn, không upload cloud |
| **Miễn phí** | Không cần API key, không tốn tiền |
| **Offline** | Chạy mà không cần internet |
| **Tùy biến** | Có thể thay đổi LLM, model |
| **Knowledge Graph** | Xây dựng đồ thị tri thức tự động |
| **WebUI** | Giao diện thân thiện, dễ sử dụng |

---

## 🚀 Tiếp Theo

1. **Ngay bây giờ**: Xem `QUICKSTART.md` (5 phút)
2. **Nếu muốn chi tiết**: Xem `SETUP_LOCAL_GUIDE.md`
3. **Nếu gặp vấn đề**: Xem `README_VN.md` → Troubleshooting
4. **Khi production**: Xem docs GitHub

---

**Bạn đã sẵn sàng! Hãy bắt đầu! 🚀**

Mọi thắc mắc vui lòng tham khảo docs hoặc GitHub issues.
