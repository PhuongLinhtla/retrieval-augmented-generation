"""Configuration for RAG Core"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

class QueryMode(str, Enum):
    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"


@dataclass
class ChunkConfig:
    """Configuration for text chunking"""
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    split_by_character: Optional[str] = "\n"
    split_by_character_only: bool = False


@dataclass
class ExtractionConfig:
    """Configuration for entity/relation extraction"""
    language: str = "English"
    # LLM parameters
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.3
    max_tokens: int = 2000
    
    # Entity extraction
    entity_types: List[str] = None
    max_entities_per_chunk: int = 100
    
    # Relation extraction
    max_relations_per_chunk: int = 100
    
    # GLEANING
    max_gleaning: int = 1
    
    # Prompt templates
    entity_extract_template: str = ""
    relation_extract_template: str = ""
    keyword_extract_template: str = ""
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = [
                "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT",
                "PRODUCT", "EVENT", "TECHNOLOGY"
            ]


@dataclass
class GraphConfig:
    """Configuration for knowledge graph"""
    language: str = "English"
    # Merge settings
    max_entity_descriptions: int = 5
    max_relation_descriptions: int = 5
    force_llm_summary_on_merge: int = 3
    
    # Summary settings
    summary_max_tokens: int = 500
    summary_context_size: int = 2000
    summary_language: str = "English"
    
    # Limits
    max_source_ids_per_entity: int = 10
    max_source_ids_per_relation: int = 10
    max_file_paths_per_entity: int = 5
    
    # Storage
    storage_backend: str = "networkx"  # networkx, neo4j, etc
    vector_db_backend: str = "faiss"   # faiss, milvus, etc


@dataclass
class QueryConfig:
    """Configuration for graph querying"""
    language: str = "English"
    # Query parameters
    top_k: int = 30
    top_k_chunks: int = 20
    mode: QueryMode = QueryMode.HYBRID
    
    # Retrieval settings
    cosine_threshold: float = 0.3
    related_chunk_number: int = 15
    
    # Translation settings
    use_translation: bool = True
    source_language: str = "Vietnamese"
    target_language: str = "English"
    
    # Token limits for context
    max_entity_tokens: int = 2000
    max_relation_tokens: int = 2000
    max_total_tokens: int = 8000
    
    # Query modes
    use_entity_embedding: bool = True
    use_relation_embedding: bool = True
    use_hybrid_search: bool = True


@dataclass
class RAGConfig:
    """Complete RAG configuration"""
    chunk_config: ChunkConfig = None
    extraction_config: ExtractionConfig = None
    graph_config: GraphConfig = None
    query_config: QueryConfig = None
    
    # General settings
    workspace: str = "./rag_workspace"
    batch_size: int = 10
    max_async_tasks: int = 5
    
    def __post_init__(self):
        if self.chunk_config is None:
            self.chunk_config = ChunkConfig()
        if self.extraction_config is None:
            self.extraction_config = ExtractionConfig()
        if self.graph_config is None:
            self.graph_config = GraphConfig()
        if self.query_config is None:
            self.query_config = QueryConfig()
