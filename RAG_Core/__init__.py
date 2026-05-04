"""RAG Core - Simplified and modular RAG system"""

__version__ = "0.1.0"

from .text_processor import TextProcessor
from .entity_extractor import EntityExtractor
from .graph_builder import GraphBuilder
from .query_engine import QueryEngine
from .markitdown_processor import MarkItDownProcessor

__all__ = [
    "TextProcessor",
    "EntityExtractor", 
    "GraphBuilder",
    "QueryEngine",
    "MarkItDownProcessor",
]
