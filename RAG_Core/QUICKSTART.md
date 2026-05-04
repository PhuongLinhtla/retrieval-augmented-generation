# Quick Start Guide

## Installation

```bash
cd /home/hainguyen/Documents/RAG_Core
python -m pip install -r requirements.txt
```

## Running the Example

```bash
cd /home/hainguyen/Documents/RAG_Core
python example.py
```

## Basic Usage (3 minutes)

### 1. Clean and Chunk Text

```python
from RAG_Core import TextProcessor
from RAG_Core.config import ChunkConfig

# Setup
config = ChunkConfig(chunk_token_size=1200)
processor = TextProcessor(config)

# Process
chunks = processor.process(
    text="Your long document here...",
    file_path="document.txt"
)

# Result: List of Chunk objects
print(f"Created {len(chunks)} chunks")
```

### 2. Extract Entities & Relations

```python
from RAG_Core import EntityExtractor

extractor = EntityExtractor()

# Extract from chunks (using mock for quick testing)
results = extractor.extract_batch(chunks, use_mock=True)

# Or with custom LLM
def my_llm(prompt):
    # Call your LLM API
    return llm_response

results = extractor.extract_batch(chunks, llm_func=my_llm)

print(f"Extracted {sum(len(r.entities) for r in results)} entity types")
print(f"Extracted {sum(len(r.relations) for r in results)} relation types")
```

### 3. Build Knowledge Graph

```python
from RAG_Core import GraphBuilder

builder = GraphBuilder()

# Add extraction results
for result in results:
    builder.add_extraction_result(result)

print(f"Entities: {builder.get_entity_count()}")
print(f"Relations: {builder.get_relation_count()}")
```

### 4. Query the Graph

```python
from RAG_Core.query_engine import QueryEngine

engine = QueryEngine(builder.nodes, builder.edges)

# Keyword search
results = engine.keyword_search("Apple", top_k=10)
for entity_id, score in results:
    print(f"{entity_id}: {score:.3f}")

# Entity search with context
result = engine.entity_search("John Smith", max_depth=2)
print(f"Found {len(result.nodes)} related entities")

# Graph stats
stats = engine.get_graph_statistics()
print(f"Graph density: {stats['density']:.3f}")
```

## Complete Pipeline

```python
from RAG_Core import TextProcessor, EntityExtractor, GraphBuilder
from RAG_Core.query_engine import QueryEngine

# 1. Text Processing
processor = TextProcessor()
chunks = processor.process(text, file_path="doc.txt")

# 2. Entity Extraction
extractor = EntityExtractor()
results = extractor.extract_batch(chunks, use_mock=True)

# 3. Graph Building
builder = GraphBuilder()
for result in results:
    builder.add_extraction_result(result)

# 4. Query
engine = QueryEngine(builder.nodes, builder.edges)
search_results = engine.keyword_search("keyword")
```

## Configuration

### Minimal (defaults)
```python
from RAG_Core import TextProcessor, EntityExtractor, GraphBuilder

processor = TextProcessor()
extractor = EntityExtractor()
builder = GraphBuilder()
```

### Custom Config
```python
from RAG_Core.config import ChunkConfig, ExtractionConfig, GraphConfig

chunk_config = ChunkConfig(
    chunk_token_size=1024,
    chunk_overlap_token_size=128
)
extraction_config = ExtractionConfig(
    entity_types=["PERSON", "ORG", "LOC"]
)
graph_config = GraphConfig(
    max_entity_descriptions=10
)

processor = TextProcessor(chunk_config)
extractor = EntityExtractor(extraction_config)
builder = GraphBuilder(graph_config)
```

## Key Classes

| Class | Purpose |
|-------|---------|
| `TextProcessor` | Clean & chunk text |
| `EntityExtractor` | Extract entities/relations |
| `GraphBuilder` | Build knowledge graph |
| `QueryEngine` | Query & retrieve |

## Data Types

| Type | Purpose |
|------|---------|
| `Chunk` | Text chunk with metadata |
| `Entity` | Extracted entity |
| `Relation` | Extracted relation |
| `ExtractionResult` | Extraction output |
| `GraphNode` | Graph entity |
| `GraphEdge` | Graph relation |

## Common Tasks

### Get all entities of a type
```python
org_entities = [
    (name, node) for name, node in builder.nodes.items()
    if node.entity_type == "ORGANIZATION"
]
```

### Find most connected entity
```python
from collections import Counter

degrees = {}
for node_id in builder.nodes:
    degree = sum(1 for (s, t) in builder.edges if s == node_id or t == node_id)
    degrees[node_id] = degree

most_connected = max(degrees, key=degrees.get)
```

### Export graph to dict
```python
graph_data = {
    "nodes": {name: asdict(node) for name, node in builder.nodes.items()},
    "edges": {str(key): asdict(edge) for key, edge in builder.edges.items()}
}
```

## Tips & Tricks

1. **Start with mock**: Use `use_mock=True` for fast prototyping
2. **Custom tokenizer**: Implement `SimpleTokenizer` interface for different languages
3. **Batch processing**: Process documents in batches for efficiency
4. **Debugging**: Enable `logging.DEBUG` to see detailed logs
5. **Saving graph**: Implement custom export functions for your database

## Troubleshooting

### No entities extracted
- Check if mock extractor is being used correctly
- Verify chunk content is not empty
- Try lowering extraction thresholds

### Graph is too sparse
- Extract from more documents
- Reduce entity/relation filtering
- Adjust entity type definitions

### Slow query performance
- Reduce graph size
- Use in-memory indexes
- Optimize neighbor lookup

## Next Steps

1. Read [README.md](README.md) for detailed documentation
2. Check [example.py](example.py) for complete example
3. Integrate with your LLM API
4. Add vector database support
5. Deploy with production storage backend

---

**Happy RAG! 🚀**
