"""Knowledge graph query and retrieval engine"""

import logging
import json
from typing import Dict, List, Optional, Set, Tuple
import math

from .types import GraphNode, GraphEdge, QueryResult, Chunk
from .config import QueryConfig, QueryMode
from .utils import SimpleTokenizer
from .llm_client import get_llm_client
from .prompts import PROMPTS
from .entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


class QueryEngine:
    """Query and retrieve information from knowledge graph"""
    
    def __init__(
        self,
        nodes: Dict[str, GraphNode],
        edges: Dict[Tuple[str, str], GraphEdge],
        config: Optional[QueryConfig] = None,
        chunks: Optional[Dict[str, Chunk]] = None
    ):
        """Initialize query engine
        
        Args:
            nodes: Dictionary of GraphNode objects
            edges: Dictionary of GraphEdge objects
            config: QueryConfig instance
            chunks: Dictionary of Chunk objects
        """
        self.nodes = nodes
        self.edges = edges
        self.config = config or QueryConfig()
        self.chunks = chunks or {}
        self.tokenizer = SimpleTokenizer()
        self.llm_client = get_llm_client()
        self.translation_client = get_llm_client(purpose="translation")
        self.extractor = EntityExtractor()
    
    def keyword_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Search entities by keywords
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (entity_id, score) tuples sorted by score
        """
        top_k = top_k or self.config.top_k
        logger.info(f"Keyword search: '{query}'")
        
        query_tokens = set(self.tokenizer.encode(query.lower()))
        results = []
        
        # Score entities based on keyword match
        for entity_id, node in self.nodes.items():
            # Tokenize entity description
            description_tokens = set(self.tokenizer.encode(node.description.lower()))
            entity_name_tokens = set(self.tokenizer.encode(entity_id.lower()))
            
            # Calculate score as Jaccard similarity
            all_tokens = description_tokens | entity_name_tokens
            if all_tokens:
                intersection = query_tokens & all_tokens
                score = len(intersection) / len(all_tokens)
                
                if score > 0:
                    results.append((entity_id, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(results)} entities, returning top {top_k}")
        return results[:top_k]
    
    def entity_search(
        self,
        entity_id: str,
        max_depth: int = 2
    ) -> QueryResult:
        """Search for entity and its neighbors
        
        Args:
            entity_id: Entity to search for
            max_depth: Maximum depth of neighbors to include
            
        Returns:
            QueryResult containing entity and related graph
        """
        logger.info(f"Entity search: '{entity_id}' (depth={max_depth})")
        
        if entity_id not in self.nodes:
            logger.warning(f"Entity not found: {entity_id}")
            return QueryResult(
                nodes=[],
                edges=[],
                query_text=entity_id,
                score=0.0
            )
        
        # Get subgraph
        nodes = {}
        edges = {}
        visited = set()
        
        # BFS to find neighbors
        to_visit = [(entity_id, 0)]
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Add node
            if current_id in self.nodes:
                nodes[current_id] = self.nodes[current_id]
            
            # Find neighbors at this depth
            if depth < max_depth:
                for (src, tgt), edge in self.edges.items():
                    if src == current_id and tgt not in visited:
                        to_visit.append((tgt, depth + 1))
                        edges[(src, tgt)] = edge
                    elif tgt == current_id and src not in visited:
                        to_visit.append((src, depth + 1))
                        edges[(src, tgt)] = edge
        
        # Convert to lists
        node_list = list(nodes.values())
        edge_list = list(edges.values())
        
        result = QueryResult(
            nodes=node_list,
            edges=edge_list,
            query_text=entity_id,
            score=1.0 if entity_id in nodes else 0.0
        )
        
        logger.info(f"Found {len(node_list)} nodes and {len(edge_list)} edges")
        return result
    
    def relation_search(
        self,
        src_id: str,
        tgt_id: str
    ) -> Optional[GraphEdge]:
        """Find relation between two entities
        
        Args:
            src_id: Source entity ID
            tgt_id: Target entity ID
            
        Returns:
            GraphEdge or None
        """
        logger.info(f"Relation search: '{src_id}' -> '{tgt_id}'")
        
        edge_key = tuple(sorted([src_id, tgt_id]))
        edge = self.edges.get(edge_key)
        
        if edge:
            logger.info(f"Found relation with weight {edge.weight}")
        else:
            logger.info("Relation not found")
        
        return edge
    
    def find_paths(
        self,
        start_id: str,
        end_id: str,
        max_hops: int = 3
    ) -> List[List[str]]:
        """Find all paths between two entities
        
        Args:
            start_id: Start entity ID
            end_id: End entity ID
            max_hops: Maximum number of hops
            
        Returns:
            List of paths (each path is a list of entity IDs)
        """
        logger.info(f"Finding paths from '{start_id}' to '{end_id}' (max_hops={max_hops})")
        
        paths = []
        
        def dfs(current: str, target: str, visited: Set[str], path: List[str], hops: int):
            if hops > max_hops:
                return
            
            if current == target:
                paths.append(path[:])
                return
            
            visited.add(current)
            
            # Find neighbors
            for (src, tgt) in self.edges.keys():
                neighbor = None
                if src == current and tgt not in visited:
                    neighbor = tgt
                elif tgt == current and src not in visited:
                    neighbor = src
                
                if neighbor:
                    path.append(neighbor)
                    dfs(neighbor, target, visited, path, hops + 1)
                    path.pop()
            
            visited.remove(current)
        
        if start_id not in self.nodes or end_id not in self.nodes:
            logger.warning("Start or end entity not found")
            return []
        
        dfs(start_id, end_id, set(), [start_id], 0)
        
        logger.info(f"Found {len(paths)} paths")
        return paths
    
    def compute_similarity(
        self,
        entity1_id: str,
        entity2_id: str
    ) -> float:
        """Compute similarity between two entities
        
        Uses simple Jaccard similarity on neighbors
        
        Args:
            entity1_id: First entity ID
            entity2_id: Second entity ID
            
        Returns:
            Similarity score (0-1)
        """
        neighbors1 = set()
        neighbors2 = set()
        
        # Find neighbors
        for (src, tgt) in self.edges.keys():
            if src == entity1_id:
                neighbors1.add(tgt)
            elif tgt == entity1_id:
                neighbors1.add(src)
            
            if src == entity2_id:
                neighbors2.add(tgt)
            elif tgt == entity2_id:
                neighbors2.add(src)
        
        # Add the entities themselves
        neighbors1.add(entity1_id)
        neighbors2.add(entity2_id)
        
        # Jaccard similarity
        if not neighbors1 and not neighbors2:
            return 0.0
        
        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_entity_context(
        self,
        entity_id: str,
        context_size: int = 3
    ) -> Dict:
        """Get context for an entity
        
        Args:
            entity_id: Entity ID
            context_size: Number of related entities to include
            
        Returns:
            Dictionary with entity and context information
        """
        if entity_id not in self.nodes:
            return {}
        
        node = self.nodes[entity_id]
        
        # Find related entities
        related = []
        for (src, tgt), edge in self.edges.items():
            if src == entity_id:
                related.append((tgt, edge.weight))
            elif tgt == entity_id:
                related.append((src, edge.weight))
        
        # Sort by weight
        related.sort(key=lambda x: x[1], reverse=True)
        related = related[:context_size]
        
        context = {
            "entity_id": entity_id,
            "entity_type": node.entity_type,
            "description": node.description,
            "source_ids": node.source_ids,
            "file_paths": node.file_paths,
            "related_entities": [
                {
                    "entity_id": eid,
                    "weight": weight,
                    "type": self.nodes.get(eid, {}).entity_type if eid in self.nodes else "UNKNOWN"
                }
                for eid, weight in related
            ]
        }
        
        return context
    
    def get_graph_statistics(self) -> Dict:
        """Get graph statistics
        
        Returns:
            Dictionary with graph metrics
        """
        # Count entity types
        entity_types = {}
        for node in self.nodes.values():
            entity_types[node.entity_type] = entity_types.get(node.entity_type, 0) + 1
        
        # Calculate density
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 0
        density = num_edges / max_edges if max_edges > 0 else 0
        
        # Find most connected entities
        node_degrees = {}
        for node_id in self.nodes:
            degree = 0
            for (src, tgt) in self.edges:
                if src == node_id or tgt == node_id:
                    degree += 1
            node_degrees[node_id] = degree
        
        most_connected = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        
        stats = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density,
            "entity_types": entity_types,
            "avg_degree": (2 * num_edges / num_nodes) if num_nodes > 0 else 0,
            "most_connected": [{"entity_id": eid, "degree": deg} for eid, deg in most_connected]
        }
        
        return stats

    def answer(self, question: str) -> str:
        """Answer a natural language question using the knowledge graph
        
        Args:
            question: User question
            
        Returns:
            Natural language answer
        """
        original_question = question
        
        # 1. Translate Vietnamese to English if needed
        if self.config.use_translation:
            logger.info(f"Translating question to English: {question}")
            question = self._translate(question, "translate_to_english")
            logger.info(f"Translated question: {question}")
            
        mode = self.config.mode
        logger.info(f"Answering question in mode: {mode}")
        
        if mode == QueryMode.NAIVE:
            answer = self._naive_answer(question)
        elif mode == QueryMode.LOCAL:
            answer = self._local_answer(question)
        elif mode == QueryMode.GLOBAL:
            answer = self._global_answer(question)
        elif mode == QueryMode.HYBRID:
            answer = self._hybrid_answer(question)
        else:
            answer = self._hybrid_answer(question)
            
        # 2. Translate English back to Vietnamese if needed
        if self.config.use_translation:
            logger.info("Translating answer back to Vietnamese")
            answer = self._translate(answer, "translate_to_vietnamese")
            
        return answer

    def _translate(self, text: str, prompt_key: str) -> str:
        """Helper to translate text using a specialized translation model"""
        prompt = PROMPTS[prompt_key].format(text=text)
        try:
            # Use the dedicated translation client
            return self.translation_client.generate(prompt).strip()
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text

    def _naive_answer(self, question: str) -> str:
        """Basic keyword search + LLM"""
        # (Current implementation of answer() was essentially naive)
        results = self.keyword_search(question, top_k=self.config.top_k)
        if not results:
            return PROMPTS["fail_response"]
            
        context_data = self._build_context_from_results(results)
        return self._generate_answer(question, context_data)

    def _local_answer(self, question: str) -> str:
        """Local search: specific entities and their 1st-degree neighbors"""
        keywords = self.extractor.extract_keywords(question, self.llm_client.generate)
        ll_keywords = keywords.get("low_level_keywords", [])
        
        all_relevant_nodes = set()
        for kw in ll_keywords:
            results = self.keyword_search(kw, top_k=5)
            for node_id, _ in results:
                all_relevant_nodes.add(node_id)
                # Add neighbors
                for (src, tgt) in self.edges:
                    if src == node_id: all_relevant_nodes.add(tgt)
                    if tgt == node_id: all_relevant_nodes.add(src)
        
        if not all_relevant_nodes:
            return self._naive_answer(question)
            
        context_data = self._build_context_from_node_list(list(all_relevant_nodes))
        return self._generate_answer(question, context_data)

    def _global_answer(self, question: str) -> str:
        """Global search: high-level themes and summaries"""
        keywords = self.extractor.extract_keywords(question, self.llm_client.generate)
        hl_keywords = keywords.get("high_level_keywords", [])
        
        # In a real Global Search, we'd look at community summaries.
        # For this simplified RAG_Core, we look at nodes matching high-level keywords.
        all_relevant_nodes = set()
        for kw in hl_keywords:
            results = self.keyword_search(kw, top_k=10)
            for node_id, _ in results:
                all_relevant_nodes.add(node_id)
        
        if not all_relevant_nodes:
            return self._naive_answer(question)
            
        context_data = self._build_context_from_node_list(list(all_relevant_nodes))
        return self._generate_answer(question, context_data)

    def _hybrid_answer(self, question: str) -> str:
        """Hybrid search: Combine local and global"""
        keywords = self.extractor.extract_keywords(question, self.llm_client.generate)
        hl_keywords = keywords.get("high_level_keywords", [])
        ll_keywords = keywords.get("low_level_keywords", [])
        
        all_relevant_nodes = set()
        # Local part
        for kw in ll_keywords:
            results = self.keyword_search(kw, top_k=5)
            for node_id, _ in results:
                all_relevant_nodes.add(node_id)
        
        # Global part
        for kw in hl_keywords:
            results = self.keyword_search(kw, top_k=5)
            for node_id, _ in results:
                all_relevant_nodes.add(node_id)
        
        if not all_relevant_nodes:
            return self._naive_answer(question)
            
        context_data = self._build_context_from_node_list(list(all_relevant_nodes))
        return self._generate_answer(question, context_data)

    def _build_context_from_results(self, results: List[Tuple[str, float]]) -> str:
        node_ids = [r[0] for r in results]
        return self._build_context_from_node_list(node_ids)

    def _build_context_from_node_list(self, node_ids: List[str]) -> str:
        """Build context string from a list of node IDs using LightRAG patterns"""
        entities_data = []
        relations_data = []
        relevant_chunk_ids = set()
        
        # 1. Collect Entities
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                entities_data.append({
                    "entity_name": node_id,
                    "entity_type": node.entity_type,
                    "description": node.description
                })
                relevant_chunk_ids.update(node.source_ids)
        
        # 2. Collect Relationships between these entities
        for (u, v), edge in self.edges.items():
            if u in node_ids and v in node_ids:
                relations_data.append({
                    "src_id": u,
                    "tgt_id": v,
                    "description": edge.description,
                    "keywords": edge.keywords,
                    "weight": edge.weight
                })
                relevant_chunk_ids.update(edge.source_ids)
        
        # 3. Collect and format Document Chunks
        text_chunks_data = []
        reference_list = []
        
        # Limit the number of chunks to avoid context overflow
        chunk_list = list(relevant_chunk_ids)[:self.config.related_chunk_number]
        
        for i, chunk_id in enumerate(chunk_list):
            content = self._get_chunk_content(chunk_id)
            if content:
                ref_id = i + 1
                text_chunks_data.append({
                    "content": content,
                    "source_id": chunk_id,
                    "reference_id": ref_id
                })
                title = f"Document Chunk {chunk_id[:8]}"
                reference_list.append(f"[{ref_id}] {title}")
        
        # 4. Format using templates from prompts.py
        try:
            context_str = PROMPTS["kg_query_context"].format(
                entities_str=json.dumps(entities_data, ensure_ascii=False, indent=2),
                relations_str=json.dumps(relations_data, ensure_ascii=False, indent=2),
                text_chunks_str=json.dumps(text_chunks_data, ensure_ascii=False, indent=2),
                reference_list_str="\n".join(reference_list)
            )
        except KeyError:
            # Fallback if the template is not exactly matching
            context_str = f"Entities:\n{json.dumps(entities_data)}\n\nRelations:\n{json.dumps(relations_data)}\n\nChunks:\n{json.dumps(text_chunks_data)}"
        
        return context_str

    def _get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """Helper to retrieve chunk content from shared storage"""
        if hasattr(self, 'chunks') and chunk_id in self.chunks:
            return self.chunks[chunk_id].content
        return None

    def _generate_answer(self, question: str, context_data: str) -> str:
        """Generate final answer using LLM and context"""
        mode = self.config.mode
        prompt_key = "rag_response" if mode != QueryMode.NAIVE else "naive_rag_response"
        
        try:
            prompt = PROMPTS[prompt_key].format(
                context_data=context_data,
                content_data=context_data, # For naive_rag_response
                response_type="multiple paragraphs",
                user_prompt=f"User Query: {question}",
                query=question # In case it's still needed
            )
        except KeyError as e:
            logger.warning(f"Missing placeholder in prompt: {e}. Falling back to basic format.")
            prompt = f"Context:\n{context_data}\n\nQuestion: {question}"

        return self.llm_client.generate(prompt)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from .graph_builder import GraphBuilder
    from .text_processor import TextProcessor
    from .entity_extractor import EntityExtractor
    
    # Create and build graph
    processor = TextProcessor()
    extractor = EntityExtractor()
    builder = GraphBuilder()
    
    sample_text = """
    John Smith is a software engineer at Google.
    Google is headquartered in Mountain View.
    Mary Johnson also works at Google.
    """
    
    chunks = processor.process(sample_text)
    results = extractor.extract_batch(chunks, use_mock=True)
    
    for result in results:
        builder.add_extraction_result(result)
    
    # Create query engine and test
    engine = QueryEngine(builder.nodes, builder.edges)
    
    # Keyword search
    print("\n=== Keyword Search ===")
    results = engine.keyword_search("Google", top_k=5)
    for entity_id, score in results:
        print(f"{entity_id}: {score:.3f}")
    
    # Entity search
    print("\n=== Entity Search ===")
    if builder.nodes:
        first_entity = list(builder.nodes.keys())[0]
        result = engine.entity_search(first_entity)
        print(f"Found {len(result.nodes)} nodes, {len(result.edges)} edges")
    
    # Graph statistics
    print("\n=== Graph Statistics ===")
    stats = engine.get_graph_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
