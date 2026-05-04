"""Type definitions for RAG Core"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum


class EntityType(str, Enum):
    """Standard entity types"""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"
    UNKNOWN = "UNKNOWN"


@dataclass
class Entity:
    """Extracted entity representation"""
    name: str
    type: str
    description: str
    source_id: str  # chunk_id
    file_path: str = "unknown_source"
    timestamp: int = 0
    
    def __hash__(self):
        return hash((self.name, self.type))
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.type == other.type


@dataclass
class Relation:
    """Extracted relation/edge representation"""
    src_id: str
    tgt_id: str
    description: str
    keywords: str = ""
    weight: float = 1.0
    source_id: str = ""  # chunk_id
    file_path: str = "unknown_source"
    timestamp: int = 0
    
    def __hash__(self):
        # Make relation undirected by sorting
        src, tgt = (self.src_id, self.tgt_id) if self.src_id <= self.tgt_id else (self.tgt_id, self.src_id)
        return hash((src, tgt))
    
    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        src1, tgt1 = (self.src_id, self.tgt_id) if self.src_id <= self.tgt_id else (self.tgt_id, self.src_id)
        src2, tgt2 = (other.src_id, other.tgt_id) if other.src_id <= other.tgt_id else (other.tgt_id, other.src_id)
        return src1 == src2 and tgt1 == tgt2


@dataclass
class Chunk:
    """Text chunk representation"""
    chunk_id: str
    content: str
    tokens: int
    chunk_order_index: int
    file_path: str = "unknown_source"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result from entity/relation extraction"""
    entities: Dict[str, List[Entity]] = field(default_factory=dict)
    relations: Dict[Tuple[str, str], List[Relation]] = field(default_factory=dict)
    chunk_id: str = ""
    timestamp: int = 0


@dataclass
class GraphNode:
    """Node in knowledge graph"""
    entity_id: str
    entity_type: str
    description: str
    source_ids: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    created_at: int = 0
    embeddings: Optional[List[float]] = None


@dataclass
class GraphEdge:
    """Edge in knowledge graph"""
    src_id: str
    tgt_id: str
    description: str
    keywords: str = ""
    weight: float = 1.0
    source_ids: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    created_at: int = 0
    embeddings: Optional[List[float]] = None


@dataclass
class QueryResult:
    """Result from graph query"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    query_text: str = ""
    score: float = 0.0
    source_chunks: List[str] = field(default_factory=list)
