"""Knowledge graph construction and management"""

import logging
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

from .types import (
    Entity, Relation, ExtractionResult, GraphNode, GraphEdge
)
from .config import GraphConfig
from .utils import merge_descriptions, format_timestamp
from .llm_client import get_llm_client
from .prompts import PROMPTS

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build and manage knowledge graph from extracted entities and relations"""
    
    def __init__(self, config: Optional[GraphConfig] = None):
        """Initialize graph builder"""
        self.config = config or GraphConfig()
        self.llm_client = get_llm_client()
        
        # In-memory graph storage
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str], GraphEdge] = {}
    
    def add_extraction_result(self, result: ExtractionResult) -> None:
        """Add extraction result to graph
        
        Args:
            result: ExtractionResult with entities and relations
        """
        logger.info(f"Adding extraction result from {result.chunk_id}")
        
        # Phase 1: Add/merge entities
        for entity_name, entities in result.entities.items():
            self._merge_entity(entity_name, entities, result.chunk_id)
        
        # Phase 2: Add/merge relations
        for (src_id, tgt_id), relations in result.relations.items():
            self._merge_relation(src_id, tgt_id, relations, result.chunk_id)
    
    def add_extraction_batch(self, results: List[ExtractionResult]) -> None:
        """Add multiple extraction results to graph
        
        Args:
            results: List of ExtractionResult objects
        """
        for i, result in enumerate(results, 1):
            logger.info(f"Processing extraction {i}/{len(results)}")
            self.add_extraction_result(result)
    
    def _merge_entity(
        self,
        entity_name: str,
        entities: List[Entity],
        chunk_id: str
    ) -> GraphNode:
        """Merge entity into graph
        
        Args:
            entity_name: Name of entity
            entities: List of Entity objects with same name
            chunk_id: Source chunk ID
            
        Returns:
            Merged GraphNode
        """
        if entity_name in self.nodes:
            # Entity already exists, merge with new data
            node = self.nodes[entity_name]
            
            # Merge descriptions
            existing_descriptions = node.description.split(" | ") if node.description else []
            new_descriptions = [e.description for e in entities]
            all_descriptions = existing_descriptions + new_descriptions
            
            # Merge source IDs
            existing_sources = set(node.source_ids)
            new_sources = {e.source_id for e in entities}
            all_sources = list(existing_sources | new_sources)
            
            # Merge file paths
            existing_paths = set(node.file_paths)
            new_paths = {e.file_path for e in entities}
            all_paths = list(existing_paths | new_paths)
            
            # Get most common entity type
            entity_type = max(
                [e.type for e in entities] + [node.entity_type],
                key=lambda x: [e.type for e in entities].count(x) if x else 0
            ) or "UNKNOWN"
            
            # Summarize descriptions if needed using LLM (LightRAG style)
            if len(all_descriptions) > 3:  # Threshold for summarization
                description = self._summarize_descriptions("Entity", entity_name, all_descriptions)
            else:
                description = " | ".join(all_descriptions)
            
            node.description = description
            node.entity_type = entity_type
            node.source_ids = all_sources
            node.file_paths = all_paths
            node.created_at = format_timestamp()
            
            logger.debug(f"Merged entity: {entity_name} ({entity_type})")
        else:
            # New entity
            entity = entities[0]  # Use first one as base
            
            descriptions = [e.description for e in entities]
            description = merge_descriptions(descriptions)
            source_ids = list({e.source_id for e in entities})
            file_paths = list({e.file_path for e in entities})
            
            node = GraphNode(
                entity_id=entity_name,
                entity_type=entity.type,
                description=description,
                source_ids=source_ids,
                file_paths=file_paths,
                created_at=format_timestamp()
            )
            self.nodes[entity_name] = node
            logger.debug(f"Created entity: {entity_name} ({entity.type})")
        
        return node
    
    def _merge_relation(
        self,
        src_id: str,
        tgt_id: str,
        relations: List[Relation],
        chunk_id: str
    ) -> GraphEdge:
        """Merge relation into graph
        
        Args:
            src_id: Source entity ID
            tgt_id: Target entity ID
            relations: List of Relation objects
            chunk_id: Source chunk ID
            
        Returns:
            Merged GraphEdge
        """
        # Normalize edge key (undirected)
        edge_key = tuple(sorted([src_id, tgt_id]))
        
        if edge_key in self.edges:
            # Edge already exists, merge
            edge = self.edges[edge_key]
            
            # Merge descriptions
            existing_descriptions = edge.description.split(" | ") if edge.description else []
            new_descriptions = [r.description for r in relations]
            all_descriptions = existing_descriptions + new_descriptions
            
            # Merge keywords
            existing_keywords = set(edge.keywords.split(",")) if edge.keywords else set()
            new_keywords = set()
            for r in relations:
                if r.keywords:
                    new_keywords.update(k.strip() for k in r.keywords.split(","))
            all_keywords = sorted(existing_keywords | new_keywords)
            
            # Merge source IDs
            existing_sources = set(edge.source_ids)
            new_sources = {r.source_id for r in relations}
            all_sources = list(existing_sources | new_sources)
            
            # Merge file paths
            existing_paths = set(edge.file_paths)
            new_paths = {r.file_path for r in relations}
            all_paths = list(existing_paths | new_paths)
            
            # Sum weights
            weight = sum(r.weight for r in relations) + edge.weight
            
            # Summarize descriptions if needed using LLM (LightRAG style)
            if len(all_descriptions) > 3:
                description = self._summarize_descriptions("Relationship", f"{src_id}-{tgt_id}", all_descriptions)
            else:
                description = " | ".join(all_descriptions)
            keywords = ",".join(all_keywords)
            
            edge.description = description
            edge.keywords = keywords
            edge.weight = weight
            edge.source_ids = all_sources
            edge.file_paths = all_paths
            edge.created_at = format_timestamp()
            
            logger.debug(f"Merged relation: {src_id} -> {tgt_id}")
        else:
            # New edge
            relation = relations[0]
            
            descriptions = [r.description for r in relations]
            description = merge_descriptions(descriptions)
            
            # Collect keywords
            all_keywords = set()
            for r in relations:
                if r.keywords:
                    all_keywords.update(k.strip() for k in r.keywords.split(","))
            keywords = ",".join(sorted(all_keywords))
            
            source_ids = list({r.source_id for r in relations})
            file_paths = list({r.file_path for r in relations})
            weight = sum(r.weight for r in relations)
            
            edge = GraphEdge(
                src_id=edge_key[0],
                tgt_id=edge_key[1],
                description=description,
                keywords=keywords,
                weight=weight,
                source_ids=source_ids,
                file_paths=file_paths,
                created_at=format_timestamp()
            )
            self.edges[edge_key] = edge
            logger.debug(f"Created relation: {edge_key[0]} -> {edge_key[1]}")
        
        # Ensure both nodes exist
        for entity_id in [src_id, tgt_id]:
            if entity_id not in self.nodes:
                # Create minimal node
                node = GraphNode(
                    entity_id=entity_id,
                    entity_type="UNKNOWN",
                    description=f"Referenced in relation",
                    source_ids=[chunk_id],
                    file_paths=["unknown_source"],
                    created_at=format_timestamp()
                )
                self.nodes[entity_id] = node
                logger.debug(f"Created implicit entity: {entity_id}")
        
        return edge
    
    def get_node(self, entity_id: str) -> Optional[GraphNode]:
        """Get node by entity ID
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            GraphNode or None
        """
        return self.nodes.get(entity_id)
    
    def get_edge(self, src_id: str, tgt_id: str) -> Optional[GraphEdge]:
        """Get edge between two entities
        
        Args:
            src_id: Source entity ID
            tgt_id: Target entity ID
            
        Returns:
            GraphEdge or None
        """
        edge_key = tuple(sorted([src_id, tgt_id]))
        return self.edges.get(edge_key)
    
    def get_neighbors(self, entity_id: str) -> List[str]:
        """Get neighbor entities of a node
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            List of neighbor entity IDs
        """
        neighbors = set()
        for (src, tgt) in self.edges.keys():
            if src == entity_id:
                neighbors.add(tgt)
            elif tgt == entity_id:
                neighbors.add(src)
        return list(neighbors)
    
    def get_entity_count(self) -> int:
        """Get total number of entities"""
        return len(self.nodes)
    
    def get_relation_count(self) -> int:
        """Get total number of relations"""
        return len(self.edges)
    
    def get_entities(self) -> Dict[str, GraphNode]:
        """Get all entities
        
        Returns:
            Dictionary mapping entity_id to GraphNode
        """
        return dict(self.nodes)
    
    def get_relations(self) -> Dict[Tuple[str, str], GraphEdge]:
        """Get all relations
        
        Returns:
            Dictionary mapping (src, tgt) to GraphEdge
        """
        return dict(self.edges)
    
    def get_subgraph(self, entity_ids: Set[str], depth: int = 1) -> Tuple[Dict, Dict]:
        """Get subgraph of entities and their relations
        
        Args:
            entity_ids: Set of entity IDs to include
            depth: Depth of neighbors to include
            
        Returns:
            Tuple of (nodes_dict, edges_dict)
        """
        nodes = {}
        edges = {}
        
        # Add initial entities
        visited = set()
        to_visit = set(entity_ids)
        current_depth = 0
        
        while to_visit and current_depth < depth:
            next_visit = set()
            
            for entity_id in to_visit:
                if entity_id in visited:
                    continue
                
                visited.add(entity_id)
                
                # Add node
                if entity_id in self.nodes:
                    nodes[entity_id] = self.nodes[entity_id]
                
                # Add neighbors
                neighbors = self.get_neighbors(entity_id)
                next_visit.update(neighbors)
            
            to_visit = next_visit
            current_depth += 1
        
        # Add edges between visited nodes
        for (src, tgt), edge in self.edges.items():
            if src in visited and tgt in visited:
                edges[(src, tgt)] = edge
        return nodes, edges

    def _summarize_descriptions(self, type_name: str, name: str, descriptions: List[str]) -> str:
        """Use LLM to summarize multiple descriptions into one cohesive summary"""
        logger.info(f"Summarizing {len(descriptions)} descriptions for {type_name} '{name}'")
        
        desc_list_str = "\n".join([f"- {d}" for d in descriptions])
        prompt = PROMPTS["summarize_entity_descriptions"].format(
            description_type=type_name,
            description_name=name,
            description_list=desc_list_str,
            summary_length=1000,
            language=getattr(self.config, "language", "English")
        )
        
        try:
            summary = self.llm_client.generate(prompt)
            return summary.strip()
        except Exception as e:
            logger.error(f"Failed to summarize descriptions: {e}")
            return " | ".join(descriptions[:3])


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from .text_processor import TextProcessor
    from .entity_extractor import EntityExtractor
    
    # Create components
    processor = TextProcessor()
    extractor = EntityExtractor()
    builder = GraphBuilder()
    
    # Sample text
    sample_text = """
    John Smith is a software engineer at Google.
    Google is headquartered in Mountain View, California.
    Mary Johnson also works at Google in the AI research division.
    Google was founded by Larry Page and Sergey Brin.
    """
    
    # Process
    chunks = processor.process(sample_text)
    results = extractor.extract_batch(chunks, use_mock=True)
    
    # Build graph
    for result in results:
        builder.add_extraction_result(result)
    
    # Print results
    print(f"\nGraph Statistics:")
    print(f"Entities: {builder.get_entity_count()}")
    print(f"Relations: {builder.get_relation_count()}")
    
    print(f"\nEntities:")
    for entity_id, node in builder.get_entities().items():
        print(f"  {entity_id}: {node.entity_type}")
