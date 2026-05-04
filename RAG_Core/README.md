# RAG Core - Simplified & Modular RAG System

Một triển khai đơn giản, dễ dùng và có thể mở rộng của hệ thống Retrieval-Augmented Generation, chứa các kỹ thuật cốt lõi từ LightRAG nhưng được viết lại với code sạch hơn.

## 📋 Các Kỹ Thuật Cốt Lõi

### 1. **Text Processing** (`text_processor.py`)
- ✅ Clean text: loại bỏ ký tự không hợp lệ, normalize whitespace
- ✅ Chunking: chia text theo token size với overlap
- ✅ Hỗ trợ chia theo character markers

```python
from RAG_Core import TextProcessor
from RAG_Core.config import ChunkConfig

# Tạo processor với config
config = ChunkConfig(
    chunk_token_size=1200,
    chunk_overlap_token_size=100,
    split_by_character="\n\n"
)
processor = TextProcessor(config)

# Process text
chunks = processor.process(
    text="Your long text here...",
    file_path="document.txt",
    chunk_id_prefix="doc1"
)
```

### 2. **Entity Extraction** (`entity_extractor.py`)
- ✅ Trích xuất entities (tên, kiểu, mô tả)
- ✅ Trích xuất relations (source, target, keywords, description)
- ✅ Parse LLM output theo định dạng chuẩn
- ✅ Mock extraction cho testing

```python
from RAG_Core import EntityExtractor
from RAG_Core.config import ExtractionConfig

config = ExtractionConfig(
    entity_types=["PERSON", "ORGANIZATION", "LOCATION"]
)
extractor = EntityExtractor(config)

# Extract từ chunk
result = extractor.extract_from_chunk(chunk, use_mock=True)

# Extract batch
results = extractor.extract_batch(chunks, use_mock=True)
```

### 3. **Knowledge Graph Builder** (`graph_builder.py`)
- ✅ Build knowledge graph từ extraction results
- ✅ Merge entities & relations (2-phase approach)
- ✅ Xử lý entity types, keywords, source tracking
- ✅ In-memory storage

```python
from RAG_Core import GraphBuilder

builder = GraphBuilder()

# Add extraction results
for result in extraction_results:
    builder.add_extraction_result(result)

# Query graph
node = builder.get_node("John Smith")
edge = builder.get_edge("John Smith", "Google")
neighbors = builder.get_neighbors("Google")

# Statistics
print(f"Entities: {builder.get_entity_count()}")
print(f"Relations: {builder.get_relation_count()}")
```

### 4. **Query Engine** (`query_engine.py`)
- ✅ Keyword search trên entities
- ✅ Entity search với neighbor lookup
- ✅ Relation search
- ✅ Path finding (BFS)
- ✅ Entity similarity computation
- ✅ Graph statistics

```python
from RAG_Core.query_engine import QueryEngine

engine = QueryEngine(builder.nodes, builder.edges)

# Keyword search
results = engine.keyword_search("Google", top_k=10)

# Entity search
result = engine.entity_search("John Smith", max_depth=2)

# Find paths
paths = engine.find_paths("John Smith", "Mountain View", max_hops=3)

# Graph stats
stats = engine.get_graph_statistics()
```

## 🏗️ Kiến Trúc

```
RAG_Core/
├── __init__.py              # Package exports
├── types.py                 # Type definitions (Entity, Relation, etc)
├── config.py                # Configuration classes
├── utils.py                 # Helper functions
├── text_processor.py        # Clean & chunk text
├── entity_extractor.py      # Extract entities/relations
├── graph_builder.py         # Build knowledge graph
├── query_engine.py          # Query & retrieve
└── README.md                # This file
```

## 🌐 WebUI

Bạn có thể chạy giao diện web để thao tác toàn bộ pipeline theo kiểu:

- Paste hoặc upload văn bản
- Cấu hình chunking và query parameters
- Chạy extraction + build graph
- Xem entities, relations, graph statistics
- Chạy keyword search và entity neighborhood search

### Chạy WebUI

```bash
cd /home/hainguyen/Documents/RAG_Core
pip install -r requirements.txt
python webui.py
```

Mở trình duyệt tại `http://127.0.0.1:7860`.

### Các file WebUI

- `webui.py`: Flask backend, nối trực tiếp vào TextProcessor/EntityExtractor/GraphBuilder/QueryEngine
- `templates/index.html`: giao diện chính
- `static/styles.css`: style responsive cho desktop/mobile

## 🔧 Configuration

### ChunkConfig
```python
ChunkConfig(
    chunk_token_size=1200,           # Tokens per chunk
    chunk_overlap_token_size=100,    # Overlap tokens
    split_by_character="\n\n",       # Character marker for splitting
    split_by_character_only=False    # Only split by character?
)
```

### ExtractionConfig
```python
ExtractionConfig(
    llm_model="gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=2000,
    entity_types=["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"],
    max_entities_per_chunk=100,
    max_relations_per_chunk=100
)
```

### GraphConfig
```python
GraphConfig(
    max_entity_descriptions=5,
    max_relation_descriptions=5,
    summary_max_tokens=500,
    summary_context_size=2000,
    max_source_ids_per_entity=10,
    max_source_ids_per_relation=10
)
```

### QueryConfig
```python
QueryConfig(
    top_k=10,
    top_k_chunks=10,
    cosine_threshold=0.5,
    use_hybrid_search=True
)
```

## 📚 Ví Dụ Đầy Đủ

```python
from RAG_Core import TextProcessor, EntityExtractor, GraphBuilder
from RAG_Core.query_engine import QueryEngine
from RAG_Core.config import ChunkConfig, ExtractionConfig, GraphConfig, QueryConfig

# 1. Setup
chunk_config = ChunkConfig(chunk_token_size=512)
extraction_config = ExtractionConfig()
graph_config = GraphConfig()
query_config = QueryConfig()

processor = TextProcessor(chunk_config)
extractor = EntityExtractor(extraction_config)
builder = GraphBuilder(graph_config)

# 2. Process text
document = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.
    Steve Jobs later returned to Apple and led the company to great success.
    Apple is headquartered in Cupertino, California.
    Tim Cook is the current CEO of Apple.
"""

chunks = processor.process(document, file_path="apple.txt", chunk_id_prefix="apple")
print(f"Created {len(chunks)} chunks")

# 3. Extract entities and relations
results = extractor.extract_batch(chunks, use_mock=True)
print(f"Extracted from {len(results)} chunks")

# 4. Build knowledge graph
for result in results:
    builder.add_extraction_result(result)

print(f"Graph: {builder.get_entity_count()} entities, {builder.get_relation_count()} relations")

# 5. Query graph
engine = QueryEngine(builder.nodes, builder.edges, query_config)

# Search by keyword
keyword_results = engine.keyword_search("Apple", top_k=5)
print("\n=== Keyword Search Results ===")
for entity_id, score in keyword_results:
    print(f"  {entity_id}: {score:.3f}")

# Search entity with context
if builder.nodes:
    first_entity = list(builder.nodes.keys())[0]
    search_result = engine.entity_search(first_entity, max_depth=2)
    print(f"\n=== Entity Search ({first_entity}) ===")
    print(f"  Found: {len(search_result.nodes)} nodes, {len(search_result.edges)} edges")

# Graph statistics
stats = engine.get_graph_statistics()
print(f"\n=== Graph Statistics ===")
print(f"  Density: {stats['density']:.3f}")
print(f"  Avg Degree: {stats['avg_degree']:.2f}")
```

## 🎯 Main Differences from LightRAG

| Aspect | RAG Core | LightRAG |
|--------|----------|----------|
| **Complexity** | Simplified | Full-featured |
| **Code Style** | Clean & modular | Production-grade |
| **Type Hints** | ✅ Comprehensive | ✅ Yes |
| **Documentation** | ✅ Detailed | ✅ Extensive |
| **Storage Backends** | In-memory only | Multiple (Neo4j, etc) |
| **LLM Integration** | Mock/custom | OpenAI, Ollama, etc |
| **Vector DB** | None included | Multiple support |
| **Use Case** | Learning/prototyping | Production RAG |

## 📝 Type Definitions

### Entity
```python
@dataclass
class Entity:
    name: str
    type: str
    description: str
    source_id: str
    file_path: str = "unknown_source"
    timestamp: int = 0
```

### Relation
```python
@dataclass
class Relation:
    src_id: str
    tgt_id: str
    description: str
    keywords: str = ""
    weight: float = 1.0
    source_id: str = ""
    file_path: str = "unknown_source"
```

### GraphNode
```python
@dataclass
class GraphNode:
    entity_id: str
    entity_type: str
    description: str
    source_ids: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    created_at: int = 0
```

### GraphEdge
```python
@dataclass
class GraphEdge:
    src_id: str
    tgt_id: str
    description: str
    keywords: str = ""
    weight: float = 1.0
    source_ids: List[str] = field(default_factory=list)
```

## 🚀 Extending RAG Core

### Custom Tokenizer
```python
class CustomTokenizer:
    @staticmethod
    def encode(text: str) -> List[str]:
        # Your tokenization logic
        return tokens
    
    @staticmethod
    def decode(tokens: List[str]) -> str:
        return " ".join(tokens)
```

### Custom LLM Integration
```python
def my_llm_function(prompt: str) -> str:
    # Call your LLM
    response = ...
    return response

# Use in extractor
result = extractor.extract_from_chunk(chunk, llm_func=my_llm_function)
```

### Custom Graph Storage
```python
class Neo4jGraphStorage:
    def add_node(self, entity_id, node_data):
        # Add to Neo4j
        pass
```

## 📦 Dependencies

- `dataclasses` (Python 3.7+)
- `typing` (Python 3.5+)
- `enum` (Python 3.4+)
- `logging` (built-in)
- `re` (built-in)
- `hashlib` (built-in)

## 📄 License

This is a simplified educational implementation of RAG concepts.

## 🔗 References

- Based on [LightRAG](https://github.com/GAIR-NLP/LightRAG)
- Knowledge Graph Construction concepts
- Information Extraction techniques

## 💡 Tips

1. **For Testing**: Use `use_mock=True` in extraction to get quick results
2. **For Production**: Implement proper LLM integration
3. **For Scaling**: Replace in-memory storage with database backend
4. **For Quality**: Implement proper entity/relation post-processing

---

**Happy RAG Building! 🚀**
