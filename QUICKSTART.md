# 🚀 QUICKSTART - Chạy LightRAG Ngay

## 📋 Tóm Tắt Setup

Bạn đã chọn:
- ✅ LLM: **Ollama Local** (mistral)
- ✅ Embedding: **Local Docker** (nomic-embed-text)  
- ✅ Storage: **Neo4j** (Graph Database)
- ✅ Server: **WebUI** (http://localhost:8000)

---

## ⏰ Bước-Theo-Bước (15 phút)

### Bước 1: Cài Ollama (nếu chưa có)
```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download: https://ollama.ai/download
```

### Bước 2: Khởi Động Ollama + Pull Models
```bash
# Terminal 1 (giữ mở)
ollama serve

# Terminal 2 (nói chung)
ollama pull mistral
ollama pull nomic-embed-text
```
⏱️ **Đợi ~5-10 phút** (tùy tốc độ mạng)

### Bước 3: Chạy Startup Script
```bash
cd /home/hainguyen/Documents/LightRAG
chmod +x startup.sh
./startup.sh
```

Script sẽ:
- Kiểm tra Ollama, Docker
- Cài dependencies Python
- Khởi động Neo4j + Embedding
- Tạo thư mục

### Bước 4: Chạy WebUI Server
```bash
source .venv/bin/activate
lightrag-server
```

Lệnh này sẽ hiển thị:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Bước 5: Mở Browser
👉 **http://localhost:8000**

---

## 📚 Upload Tài Liệu & Test

### Via WebUI (Dễ nhất)
1. Mở http://localhost:8000
2. Tab **"Documents"**
3. Click **"Upload"** 
4. Kéo thả PDF hoặc TXT
5. Click **"Process"** (chờ ~2-5 phút)
6. Tab **"Retrieval"**
7. Nhập câu hỏi → **"Send"**

### Ví Dụ Tài Liệu Test
Tạo file `test_document.txt`:
```
Công ty ABC được thành lập năm 2020.
CEO là Nguyễn Văn A.
Chính sách nghỉ phép: 15 ngày/năm
Lương thưởng: Theo hiệu suất
```

Upload file này → Query: "CEO của công ty là ai?"

---

## 🔍 Kiểm Tra Services

### Neo4j Dashboard
```
http://localhost:7474
User: neo4j
Pass: lightrag_password_2024
```

### API Documentation
```
http://localhost:8000/docs
```

### Docker Status
```bash
docker-compose -f docker-compose.local.yml ps
```

---

## 🆘 Vấn Đề Thường Gặp

### ❌ "Ollama not found"
```bash
✓ Kiểm tra: ollama --version
✓ Cài: curl -fsSL https://ollama.ai/install.sh | sh
```

### ❌ "Cannot connect to Neo4j"
```bash
✓ Kiểm tra: docker ps | grep neo4j
✓ Restart: docker-compose -f docker-compose.local.yml restart neo4j
```

### ❌ "Port 8000 already in use"
```bash
# Đổi port trong .env
LIGHTRAG_SERVER_PORT="8001"
```

### ❌ "Out of memory"
```bash
✓ Dùng model nhỏ hơn:
  ollama pull neural-chat  (thay mistral)
✓ Giảm chunk_size trong lightrag/operate.py
```

---

## 📁 File Quan Trọng

| File | Mục đích |
|------|---------|
| `.env` | Cấu hình (LLM, storage, port) |
| `startup.sh` | Script khởi động tự động |
| `docker-compose.local.yml` | Docker services |
| `demo_local.py` | Test script |
| `SETUP_LOCAL_GUIDE.md` | Hướng dẫn chi tiết |
| `README_VN.md` | Hướng dẫn đầy đủ |
| `internal_docs/` | Nơi lưu tài liệu |

---

## 💡 Mẹo

1. **Tăng tốc độ**
   - Dùng `neural-chat` thay `mistral` (nhanh 2x)
   - Giảm `top_k` trong query

2. **Cải thiện chất lượng**
   - Dùng `mistral:7b` hoặc `llama2:13b` (chậm hơn nhưng tốt hơn)
   - Enable `mix` mode query

3. **Tiết kiệm không gian**
   - Xóa old embeddings: `rm -rf ./internal_docs/vec_db`
   - Neo4j data: Xóa `docker volume rm lightrag-neo4j`

---

## 🎓 Tiếp Theo

- ✓ **Bước 1-5 hoàn tất**: LightRAG ready!
- ✓ **Upload 1-2 tài liệu test**: Thấy kết quả
- ✓ **Fine-tune cấu hình**: Tùy từng tài liệu
- ✓ **Deploy production**: Xem docs

---

## 📞 Hỗ Trợ

- 📖 [GitHub Repo](https://github.com/HKUDS/LightRAG)
- 🐛 [Issues](https://github.com/HKUDS/LightRAG/issues)
- 💬 [Discord](https://discord.gg/yF2MmDJyGJ)

---

## ✅ Checklist

Trước khi bắt đầu:
- [ ] Docker installed
- [ ] Ollama installed
- [ ] Python 3.10+ available
- [ ] 4GB+ RAM free
- [ ] 10GB disk space free

Bắt đầu:
- [ ] Run `ollama serve` (Terminal 1)
- [ ] Run `ollama pull mistral` + `nomic-embed-text` (Terminal 2)
- [ ] Run `./startup.sh` (Terminal 3)
- [ ] Run `lightrag-server` (Terminal 4)
- [ ] Open http://localhost:8000

---

**Bạn đã sẵn sàng! 🚀**

Nếu có vấn đề, hãy kiểm tra logs hoặc xem SETUP_LOCAL_GUIDE.md
