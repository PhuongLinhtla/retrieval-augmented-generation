# Math Learning RAG - Prompt Templates & Configuration

## System Prompts

### For Math Tutoring (chung)
```
Bạn là giáo viên toán học trong hệ thống RAG. Khi trả lời:
1. Luôn kèm công thức LaTeX trong $$...$$ hoặc $...$ nếu liên quan
2. Giải thích từng bước rõ ràng (step-by-step)
3. Nếu có nhiều cách giải, liệt kê các phương pháp
4. Kèm trích dẫn nguồn: [file: path, lines: L1-L2]
5. Nếu không chắc, nói rõ: "không có trong tài liệu" và gợi ý bước kiểm chứng
```

---

## User Prompts Templates

### 1. Giải thích định nghĩa/khái niệm (Definition)
```
Prompt: "Giải thích khái niệm: [tên khái niệm]. Nêu:
(1) Định nghĩa toán học (kèm công thức nếu có)
(2) Ý nghĩa trong thực tế
(3) 1 ví dụ cụ thể
(4) Những lỗi thường mắc phải"

Ví dụ: "Giải thích khái niệm: Hàm số bậc hai"

Temperature: 0.1 (deterministic)
Top_k: 8
Max_tokens: 500
Retrieval mode: hybrid/mix
```

### 2. Giải bài toán (Problem Solving)
```
Prompt: "Giải bài toán sau:

[BÀI TOÁN]

Yêu cầu:
(1) Xác định loại bài / phương pháp giải
(2) Liệt kê các bước giải (step-by-step)
(3) Viết công thức chính tại mỗi bước (dùng $$...$$)
(4) Kiểm chứng kết quả (nếu có)
(5) Giải thích vì sao phương pháp này tối ưu"

Ví dụ: "Giải bài toán: Tìm x sao cho $x^2 - 5x + 6 = 0$"

Temperature: 0.0-0.1
Top_k: 10 (cần nhiều context)
Max_tokens: 800-1200 (bài phức tạp)
Retrieval mode: mix (combine BM25 + vector)
```

### 3. Chứng minh định lý (Proof)
```
Prompt: "Chứng minh: [định lý]

Yêu cầu:
(1) Nêu giả thiết và kết luận (dùng LaTeX)
(2) Chứng minh chi tiết từng bước
(3) Kèm lý do cho mỗi bước (vì sao có quyền làm như vậy?)
(4) Kết luận rõ ràng"

Ví dụ: "Chứng minh: Tổng ba góc trong tam giác bằng 180°"

Temperature: 0.0
Top_k: 12
Max_tokens: 1000
```

### 4. So sánh & Phân tích (Comparison)
```
Prompt: "So sánh:
(1) [Khái niệm A] vs [Khái niệm B]

Yêu cầu:
- Những điểm giống nhau
- Những điểm khác biệt chính
- Khi nào dùng A, khi nào dùng B
- Bảng so sánh (nếu phù hợp)"

Ví dụ: "So sánh: Hàm số bậc nhất vs Hàm số bậc hai"

Temperature: 0.1
Top_k: 8
Max_tokens: 600
```

### 5. Ứng dụng thực tế (Application)
```
Prompt: "Giải thích cách áp dụng [khái niệm/công thức] để giải quyết vấn đề:
[VẤN ĐỀ THỰC TẾ]

Yêu cầu:
(1) Mô phỏng toán học của vấn đề
(2) Công thức/phương pháp áp dụng
(3) Các bước tính toán
(4) Diễn giải kết quả"

Ví dụ: "Ứng dụng hàm bậc hai để tính quỹ đạo của quả bóng được ném..."

Temperature: 0.1-0.2
Top_k: 10
Max_tokens: 800
```

### 6. Ghi chú & Công thức quan trọng (Quick Reference)
```
Prompt: "Tóm tắt những công thức & định lý quan trọng về: [chủ đề]

Định dạng:
- Tên: [công thức/định lý]
  Công thức: $$...$$
  Điều kiện/Chú thích: ...
  Ứng dụng: ...
"

Ví dụ: "Tóm tắt công thức về hàm số bậc hai"

Temperature: 0.0
Top_k: 6
Max_tokens: 400
Retrieval mode: local (không cần hybrid)
```

---

## Configuration for LightRAG (.env)

```env
# Math-optimized settings

# LLM (mistral tốt cho toán)
LLM_BINDING=ollama
LLM_ENDPOINT=http://localhost:11434
LLM_MODEL=mistral
LLM_TEMPERATURE=0.0
LLM_TOP_K=10
LLM_MAX_TOKENS=1000

# Embedding
EMBEDDING_BINDING=ollama
EMBEDDING_ENDPOINT=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
# Hoặc dùng: mixedbread-ai/mxbai-embed-large (tốt hơn cho toán, nhưng chậm)

# Retrieval tuning
RETRIEVAL_MODE=mix  # hybrid search
TOP_K_RETRIEVAL=10  # cần nhiều context
SCORE_THRESHOLD=0.05

# Chunking (tối ưu cho công thức)
CHUNK_SIZE=400
CHUNK_OVERLAP=50
```

---

## Processing Workflow

### Step 1: OCR with Nougat
```bash
# Cài Nougat
pip install nougat-ocr

# Chạy OCR (output: Markdown with LaTeX)
nougat "file.pdf" -o ./output_math --pdf

# Output: output_math/[filename].mmd
```

### Step 2: Process with Math-aware Chunking
```bash
# Activate venv
source .venv/bin/activate

# Run processor
python math_doc_processor.py "./output_math/[filename].mmd"

# Output: ./math_docs_processed/[filename]_processed.jsonl
```

### Step 3: Ingest into LightRAG
```python
from lightrag import LightRAG
import json

rag = LightRAG(working_dir="./internal_docs")
rag.initialize_storages()

# Load processed chunks
with open("math_docs_processed/[filename]_processed.jsonl", "r") as f:
    for line in f:
        chunk = json.loads(line)
        await rag.ainsert(chunk['text'])

print("✅ Math document ingested into knowledge graph")
```

### Step 4: Query (in WebUI or Python)
```python
# Example query
response = await rag.aquery(
    "Giải bài toán: x^2 - 5x + 6 = 0",
    mode="local",
    top_k=10
)
print(response)

# Response will include:
# - Step-by-step solution with LaTeX
# - Source citations
# - Alternative methods (if applicable)
```

---

## Troubleshooting

### Problem: Công thức toán được OCR thành hình (không phải text)
**Solution**: Dùng Nougat thay vì tesseract/ocrmypdf

### Problem: RAG trả lời không kèm công thức
**Solution**: Điều chỉnh system prompt + set `LLM_TEMPERATURE=0.0`

### Problem: Search không tìm công thức cụ thể (vd: "phương trình bậc 2")
**Solution**: 
- Tăng `TOP_K_RETRIEVAL=15`
- Thêm annotation text vào formula_index (làm bởi `math_doc_processor.py`)
- Dùng `mode=mix` để kết hợp BM25 + vector

### Problem: LLM giải thích không rõ ràng
**Solution**: 
- Tăng `max_tokens` (ít nhất 800)
- Sử dụng prompt cụ thể: "step-by-step", "giải thích từng bước"
- Thêm "kèm công thức LaTeX" vào system prompt

---

## Performance Tips

1. **Caching**: Lưu kết quả query phổ biến để tái sử dụng
2. **Batch Processing**: Process nhiều file toán cùng lúc (parallel `nougat`)
3. **Formula Annotation**: Thêm description text cho công thức (giúp search tốt hơn)
4. **Prompt Engineering**: Tinh chỉnh prompts dựa trên feedback của người dùng

---

## Example Full Flow

```bash
# 1. OCR PDF with formulas
nougat "Toan_9_Chuong1.pdf" -o ./ocr_output --pdf

# 2. Process with math-aware chunking
python math_doc_processor.py "./ocr_output/Toan_9_Chuong1.mmd"

# 3. Copy processed file to internal_docs
cp ./math_docs_processed/Toan_9_Chuong1_processed.jsonl ./internal_docs/

# 4. Ingest into LightRAG (modify demo_local.py or use WebUI)
# Use WebUI: Upload → Choose "Math Mode" → Process

# 5. Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Giải bài toán: Tìm x sao cho x^2 - 5x + 6 = 0",
    "mode": "local",
    "top_k": 10
  }'

# Response:
{
  "answer": "Phương trình $x^2 - 5x + 6 = 0$ được giải như sau:\n\n1. Phân tích thành nhân tử: $(x-2)(x-3)=0$\n\n2. Suy ra: $x=2$ hoặc $x=3$\n\nNguồn: Toan_9_Chuong1.pdf, lines 45-50",
  "source": [{"file": "Toan_9_Chuong1.pdf", "lines": "L45-L50"}]
}
```
