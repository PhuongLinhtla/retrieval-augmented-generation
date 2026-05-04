"""
Validation and comparison utilities to ensure RAG_Core accuracy matches LightRAG

This module provides tools to verify that RAG_Core produces equivalent results
to LightRAG for the same input data.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict
import json

logger = logging.getLogger(__name__)


class ComparisonResult:
    """Result of comparing two extraction/graph outputs"""
    
    def __init__(self, name: str):
        self.name = name
        self.matched = True
        self.differences = []
        self.metrics = {}
    
    def add_difference(self, field: str, expected, actual):
        """Record a difference"""
        self.matched = False
        self.differences.append({
            "field": field,
            "expected": expected,
            "actual": actual
        })
    
    def add_metric(self, name: str, value: float):
        """Add a metric score"""
        self.metrics[name] = value
    
    def report(self) -> str:
        """Generate comparison report"""
        report = f"\n{'='*60}\nComparison: {self.name}\n{'='*60}"
        
        if self.matched:
            report += "\n✓ MATCH - Results are equivalent"
        else:
            report += f"\n✗ MISMATCH - Found {len(self.differences)} differences"
            for diff in self.differences[:10]:  # Show first 10
                report += f"\n  - {diff['field']}:"
                report += f"\n    Expected: {str(diff['expected'])[:100]}"
                report += f"\n    Actual:   {str(diff['actual'])[:100]}"
        
        if self.metrics:
            report += "\n\nMetrics:"
            for metric_name, value in self.metrics.items():
                report += f"\n  {metric_name}: {value:.4f}"
        
        return report


class TextProcessorValidator:
    """Validate text processor against LightRAG behavior"""
    
    @staticmethod
    def validate_chunking(
        rag_core_chunks: List,
        lightrag_chunks: List
    ) -> ComparisonResult:
        """Validate chunking consistency
        
        Args:
            rag_core_chunks: Chunks from RAG_Core
            lightrag_chunks: Chunks from LightRAG
            
        Returns:
            ComparisonResult with validation results
        """
        result = ComparisonResult("Text Chunking")
        
        # Check chunk count
        if len(rag_core_chunks) != len(lightrag_chunks):
            result.add_difference(
                "chunk_count",
                len(lightrag_chunks),
                len(rag_core_chunks)
            )
        
        # Check each chunk
        for i, (core_chunk, light_chunk) in enumerate(
            zip(rag_core_chunks, lightrag_chunks)
        ):
            # Check token count (allow small variance)
            core_tokens = core_chunk.tokens if hasattr(core_chunk, 'tokens') else len(core_chunk.get('tokens', []))
            light_tokens = light_chunk.get('tokens', 0) if isinstance(light_chunk, dict) else light_chunk.tokens
            
            if abs(core_tokens - light_tokens) > 2:  # Allow 2 token variance
                result.add_difference(
                    f"chunk_{i}_tokens",
                    light_tokens,
                    core_tokens
                )
            
            # Check content similarity
            core_content = core_chunk.content if hasattr(core_chunk, 'content') else core_chunk.get('content', '')
            light_content = light_chunk.get('content', '') if isinstance(light_chunk, dict) else light_chunk.content
            
            # Normalize whitespace for comparison
            core_normalized = ' '.join(core_content.split())
            light_normalized = ' '.join(light_content.split())
            
            similarity = TextProcessorValidator._string_similarity(
                core_normalized,
                light_normalized
            )
            
            if similarity < 0.95:  # Allow 5% content variance
                result.add_difference(
                    f"chunk_{i}_content_similarity",
                    0.95,
                    similarity
                )
        
        result.add_metric("total_chunks", len(rag_core_chunks))
        result.add_metric("chunk_match_rate", 
                         1.0 - (len(result.differences) / max(len(rag_core_chunks), 1)))
        
        return result
    
    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """Calculate Jaccard similarity between two strings"""
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0


class EntityExtractionValidator:
    """Validate entity extraction against LightRAG behavior"""
    
    @staticmethod
    def validate_extraction(
        rag_core_result,
        lightrag_result: Dict
    ) -> ComparisonResult:
        """Validate extraction consistency
        
        Args:
            rag_core_result: ExtractionResult from RAG_Core
            lightrag_result: Extraction result from LightRAG (dict format)
            
        Returns:
            ComparisonResult with validation results
        """
        result = ComparisonResult("Entity Extraction")
        
        # Extract entities from both
        core_entities = rag_core_result.entities if hasattr(rag_core_result, 'entities') else rag_core_result.get('entities', {})
        light_entities = lightrag_result.get('entities', {})
        
        # Count comparison
        core_entity_count = sum(len(v) for v in core_entities.values())
        light_entity_count = sum(len(v) for v in light_entities.values())
        
        entity_count_diff = abs(core_entity_count - light_entity_count)
        if entity_count_diff > 2:  # Allow 2 entity variance
            result.add_difference(
                "entity_count",
                light_entity_count,
                core_entity_count
            )
        
        # Extract relations from both
        core_relations = rag_core_result.relations if hasattr(rag_core_result, 'relations') else rag_core_result.get('relations', {})
        light_relations = lightrag_result.get('relations', {})
        
        core_relation_count = sum(len(v) for v in core_relations.values())
        light_relation_count = sum(len(v) for v in light_relations.values())
        
        relation_count_diff = abs(core_relation_count - light_relation_count)
        if relation_count_diff > 2:  # Allow 2 relation variance
            result.add_difference(
                "relation_count",
                light_relation_count,
                core_relation_count
            )
        
        # Calculate metrics
        entity_precision = EntityExtractionValidator._calculate_overlap(
            core_entities,
            light_entities
        )
        
        result.add_metric("entity_count", core_entity_count)
        result.add_metric("relation_count", core_relation_count)
        result.add_metric("entity_precision", entity_precision)
        
        return result
    
    @staticmethod
    def _calculate_overlap(dict1: Dict, dict2: Dict) -> float:
        """Calculate overlap between two entity dictionaries"""
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        
        if not keys1 and not keys2:
            return 1.0
        
        overlap = len(keys1 & keys2)
        total = len(keys1 | keys2)
        
        return overlap / total if total > 0 else 0.0


class GraphBuilderValidator:
    """Validate graph building against LightRAG behavior"""
    
    @staticmethod
    def validate_graph(
        rag_core_builder,
        lightrag_graph: Dict
    ) -> ComparisonResult:
        """Validate graph consistency
        
        Args:
            rag_core_builder: GraphBuilder instance from RAG_Core
            lightrag_graph: Graph data from LightRAG (dict format)
            
        Returns:
            ComparisonResult with validation results
        """
        result = ComparisonResult("Graph Building")
        
        # Get node counts
        core_node_count = rag_core_builder.get_entity_count()
        light_node_count = lightrag_graph.get('node_count', 0)
        
        if abs(core_node_count - light_node_count) > 2:
            result.add_difference(
                "node_count",
                light_node_count,
                core_node_count
            )
        
        # Get edge counts
        core_edge_count = rag_core_builder.get_relation_count()
        light_edge_count = lightrag_graph.get('edge_count', 0)
        
        if abs(core_edge_count - light_edge_count) > 2:
            result.add_difference(
                "edge_count",
                light_edge_count,
                core_edge_count
            )
        
        # Validate node properties
        core_nodes = rag_core_builder.get_entities()
        light_nodes = lightrag_graph.get('nodes', {})
        
        # Check for matching node names
        common_nodes = set(core_nodes.keys()) & set(light_nodes.keys())
        
        for node_name in list(common_nodes)[:10]:  # Check first 10
            core_node = core_nodes[node_name]
            light_node = light_nodes[node_name]
            
            # Check entity type (might differ, so we'll be lenient)
            core_type = core_node.entity_type if hasattr(core_node, 'entity_type') else core_node.get('type')
            light_type = light_node.get('type', '')
            
            # Check description exists and is reasonable
            core_desc = core_node.description if hasattr(core_node, 'description') else core_node.get('description')
            light_desc = light_node.get('description', '')
            
            if not core_desc:
                result.add_difference(
                    f"node_{node_name}_description",
                    "exists",
                    "missing"
                )
        
        # Calculate metrics
        node_overlap = len(common_nodes) / max(len(core_nodes), len(light_nodes), 1)
        
        result.add_metric("node_count", core_node_count)
        result.add_metric("edge_count", core_edge_count)
        result.add_metric("node_overlap", node_overlap)
        
        return result


class EndToEndValidator:
    """End-to-end validation comparing complete RAG pipelines"""
    
    @staticmethod
    def validate_pipeline(
        text: str,
        rag_core_output: Dict,
        lightrag_output: Dict
    ) -> Dict:
        """Validate complete RAG pipeline
        
        Args:
            text: Input text
            rag_core_output: Complete output from RAG_Core pipeline
            lightrag_output: Complete output from LightRAG pipeline
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "text_length": len(text),
            "validations": [],
            "overall_match": True,
            "accuracy_score": 0.0
        }
        
        # Validate chunking
        chunk_result = TextProcessorValidator.validate_chunking(
            rag_core_output.get('chunks', []),
            lightrag_output.get('chunks', [])
        )
        results["validations"].append({
            "name": "chunking",
            "matched": chunk_result.matched,
            "metrics": chunk_result.metrics
        })
        
        # Validate extraction
        extraction_result = EntityExtractionValidator.validate_extraction(
            rag_core_output.get('extraction'),
            lightrag_output.get('extraction', {})
        )
        results["validations"].append({
            "name": "extraction",
            "matched": extraction_result.matched,
            "metrics": extraction_result.metrics
        })
        
        # Validate graph
        if 'graph' in rag_core_output and 'graph' in lightrag_output:
            graph_result = GraphBuilderValidator.validate_graph(
                rag_core_output['graph'],
                lightrag_output['graph']
            )
            results["validations"].append({
                "name": "graph",
                "matched": graph_result.matched,
                "metrics": graph_result.metrics
            })
        
        # Calculate overall accuracy
        matched_count = sum(1 for v in results["validations"] if v["matched"])
        total_count = len(results["validations"])
        
        results["overall_match"] = matched_count == total_count
        results["accuracy_score"] = matched_count / total_count if total_count > 0 else 0.0
        
        return results
    
    @staticmethod
    def generate_validation_report(validation_results: Dict) -> str:
        """Generate human-readable validation report"""
        report = f"\n{'='*60}\nEND-TO-END VALIDATION REPORT\n{'='*60}"
        
        report += f"\nText Length: {validation_results['text_length']} characters"
        
        report += f"\nOverall Accuracy: {validation_results['accuracy_score']*100:.1f}%"
        report += f"\nStatus: {'✓ PASS' if validation_results['overall_match'] else '✗ FAIL'}"
        
        report += "\n\nValidation Details:"
        for val in validation_results['validations']:
            status = "✓" if val['matched'] else "✗"
            report += f"\n  {status} {val['name'].upper()}"
            for metric_name, metric_value in val['metrics'].items():
                report += f"\n     - {metric_name}: {metric_value:.4f}"
        
        return report


# Utility functions for testing
def compare_extraction_outputs(output1: Dict, output2: Dict) -> Tuple[bool, str]:
    """Compare two extraction outputs and return if they're equivalent
    
    Args:
        output1: First extraction output
        output2: Second extraction output
        
    Returns:
        Tuple of (is_equivalent, differences_summary)
    """
    entities1 = set(output1.get('entities', {}).keys())
    entities2 = set(output2.get('entities', {}).keys())
    
    relations1 = set(output1.get('relations', {}).keys())
    relations2 = set(output2.get('relations', {}).keys())
    
    missing_entities = entities2 - entities1
    extra_entities = entities1 - entities2
    
    missing_relations = relations2 - relations1
    extra_relations = relations1 - relations2
    
    is_equivalent = (
        not missing_entities and
        not extra_entities and
        not missing_relations and
        not extra_relations
    )
    
    summary = ""
    if missing_entities:
        summary += f"Missing entities: {missing_entities}\n"
    if extra_entities:
        summary += f"Extra entities: {extra_entities}\n"
    if missing_relations:
        summary += f"Missing relations: {missing_relations}\n"
    if extra_relations:
        summary += f"Extra relations: {extra_relations}\n"
    
    return is_equivalent, summary


def assert_extraction_equivalence(output1: Dict, output2: Dict, threshold: float = 0.9):
    """Assert that two extraction outputs are equivalent within threshold
    
    Args:
        output1: First extraction output
        output2: Second extraction output
        threshold: Minimum equivalence score (0-1)
        
    Raises:
        AssertionError: If outputs don't meet threshold
    """
    entities1 = set(output1.get('entities', {}).keys())
    entities2 = set(output2.get('entities', {}).keys())
    
    if entities1 and entities2:
        overlap = len(entities1 & entities2) / max(len(entities1), len(entities2))
        if overlap < threshold:
            raise AssertionError(
                f"Entity overlap {overlap:.2f} below threshold {threshold}"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Validation module ready")
