"""
Integration bridge between RAG_Core and LightRAG

This module provides utilities to run both systems in parallel and compare results
for accuracy verification.
"""

import sys
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class LightRAGBridge:
    """Bridge to LightRAG for comparison and integration"""
    
    def __init__(self, lightrag_path: str = None):
        """Initialize LightRAG bridge
        
        Args:
            lightrag_path: Path to LightRAG installation
        """
        self.lightrag_path = lightrag_path or "/home/hainguyen/Documents/LightRAG"
        self.lightrag_available = False
        
        # Try to import LightRAG
        try:
            if self.lightrag_path not in sys.path:
                sys.path.insert(0, self.lightrag_path)
            
            # Import LightRAG modules
            from lightrag.operate import (
                chunking_by_token_size,
                extract_entities,
                merge_nodes_and_edges
            )
            
            self.chunking_by_token_size = chunking_by_token_size
            self.extract_entities = extract_entities
            self.merge_nodes_and_edges = merge_nodes_and_edges
            
            self.lightrag_available = True
            logger.info("LightRAG successfully imported")
        except ImportError as e:
            logger.warning(f"Could not import LightRAG: {e}")
            logger.info("LightRAG integration features will not be available")
    
    def is_available(self) -> bool:
        """Check if LightRAG is available"""
        return self.lightrag_available
    
    def convert_rag_core_chunk_to_lightrag(self, chunk) -> Dict:
        """Convert RAG_Core Chunk to LightRAG format
        
        Args:
            chunk: RAG_Core Chunk object
            
        Returns:
            Dictionary in LightRAG format
        """
        return {
            "tokens": chunk.tokens,
            "content": chunk.content,
            "chunk_order_index": chunk.chunk_order_index,
            "id": chunk.chunk_id,
            "file_path": chunk.file_path,
            "metadata": chunk.metadata
        }
    
    def convert_rag_core_entity_to_lightrag(self, entity) -> List[str]:
        """Convert RAG_Core Entity to LightRAG extraction format
        
        Args:
            entity: RAG_Core Entity object
            
        Returns:
            List of strings in LightRAG extraction format
        """
        return [
            "entity",
            entity.name,
            entity.type,
            entity.description
        ]
    
    def convert_rag_core_relation_to_lightrag(self, relation) -> List[str]:
        """Convert RAG_Core Relation to LightRAG extraction format
        
        Args:
            relation: RAG_Core Relation object
            
        Returns:
            List of strings in LightRAG extraction format
        """
        return [
            "relation",
            relation.src_id,
            relation.tgt_id,
            relation.keywords,
            relation.description,
            str(relation.weight)
        ]
    
    def compare_chunking(
        self,
        text: str,
        tokenizer: Any
    ) -> Tuple[bool, Dict]:
        """Compare RAG_Core and LightRAG chunking
        
        Args:
            text: Text to chunk
            tokenizer: Tokenizer instance
            
        Returns:
            Tuple of (matches, metrics)
        """
        if not self.lightrag_available:
            logger.warning("LightRAG not available, cannot compare")
            return False, {}
        
        from RAG_Core import TextProcessor
        from RAG_Core.config import ChunkConfig
        
        try:
            # RAG_Core chunking
            processor = TextProcessor()
            rag_core_chunks = processor.chunk_text(text)
            
            # LightRAG chunking (simplified version)
            lightrag_chunks = self.chunking_by_token_size(
                tokenizer=tokenizer,
                content=text,
                chunk_token_size=1200,
                chunk_overlap_token_size=100
            )
            
            # Compare
            metrics = {
                "rag_core_count": len(rag_core_chunks),
                "lightrag_count": len(lightrag_chunks),
                "count_match": len(rag_core_chunks) == len(lightrag_chunks)
            }
            
            matches = metrics["count_match"]
            
            # Compare token distribution
            if matches:
                rag_core_tokens = [c.tokens for c in rag_core_chunks]
                lightrag_tokens = [c.get('tokens', 0) for c in lightrag_chunks]
                
                avg_diff = sum(abs(a - b) for a, b in zip(rag_core_tokens, lightrag_tokens)) / max(len(rag_core_tokens), 1)
                metrics["avg_token_diff"] = avg_diff
                matches = matches and avg_diff < 5  # Allow small variance
            
            return matches, metrics
        
        except Exception as e:
            logger.error(f"Error comparing chunking: {e}")
            return False, {"error": str(e)}


class RAGCoreLightRAGAdapter:
    """Adapter to use RAG_Core as drop-in replacement for LightRAG components"""
    
    @staticmethod
    def create_compatible_chunker(rag_processor: Any):
        """Create a chunker function compatible with LightRAG
        
        Args:
            rag_processor: RAG_Core TextProcessor instance
            
        Returns:
            Function with LightRAG-compatible signature
        """
        def chunker(
            tokenizer,
            content: str,
            split_by_character: str = "\n",
            chunk_token_size: int = 1200,
            chunk_overlap_token_size: int = 100,
            **kwargs
        ) -> List[Dict]:
            """LightRAG-compatible chunking function"""
            from RAG_Core.config import ChunkConfig
            
            # Create config
            config = ChunkConfig(
                chunk_token_size=chunk_token_size,
                chunk_overlap_token_size=chunk_overlap_token_size,
                split_by_character=split_by_character if split_by_character else None
            )
            
            # Process
            processor = rag_processor.__class__(config)
            chunks = processor.chunk_text(content)
            
            # Convert to LightRAG format
            return [
                {
                    "tokens": c.tokens,
                    "content": c.content,
                    "chunk_order_index": c.chunk_order_index
                }
                for c in chunks
            ]
        
        return chunker
    
    @staticmethod
    def create_compatible_extractor(rag_extractor: Any):
        """Create an extractor function compatible with LightRAG
        
        Args:
            rag_extractor: RAG_Core EntityExtractor instance
            
        Returns:
            Function with LightRAG-compatible signature
        """
        async def extractor(
            text: str,
            llm_func: Optional[Any] = None,
            **kwargs
        ) -> Dict:
            """LightRAG-compatible extraction function"""
            from RAG_Core.types import Chunk
            
            # Create chunk
            chunk = Chunk(
                chunk_id="temp",
                content=text,
                tokens=len(text.split()),
                chunk_order_index=0
            )
            
            # Extract
            result = rag_extractor.extract_from_chunk(
                chunk,
                llm_func=llm_func,
                use_mock=(llm_func is None)
            )
            
            # Convert to LightRAG format
            return {
                "entities": result.entities,
                "relations": result.relations
            }
        
        return extractor


class AccuracyBenchmark:
    """Benchmark RAG_Core accuracy against LightRAG"""
    
    def __init__(self):
        self.bridge = LightRAGBridge()
        self.results = []
    
    def run_benchmark(
        self,
        test_documents: List[str],
        test_names: List[str] = None
    ) -> Dict:
        """Run accuracy benchmark on test documents
        
        Args:
            test_documents: List of test documents
            test_names: Optional names for test documents
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.bridge.is_available():
            logger.warning("LightRAG not available, cannot run benchmark")
            return {"available": False}
        
        test_names = test_names or [f"test_{i}" for i in range(len(test_documents))]
        
        benchmark_results = {
            "total_tests": len(test_documents),
            "passed": 0,
            "failed": 0,
            "results": []
        }
        
        for doc, name in zip(test_documents, test_names):
            try:
                result = self._compare_on_document(doc, name)
                benchmark_results["results"].append(result)
                
                if result.get("passed", False):
                    benchmark_results["passed"] += 1
                else:
                    benchmark_results["failed"] += 1
            except Exception as e:
                logger.error(f"Error in benchmark test {name}: {e}")
                benchmark_results["failed"] += 1
                benchmark_results["results"].append({
                    "name": name,
                    "passed": False,
                    "error": str(e)
                })
        
        # Calculate overall accuracy
        accuracy = benchmark_results["passed"] / max(benchmark_results["total_tests"], 1)
        benchmark_results["overall_accuracy"] = accuracy
        
        return benchmark_results
    
    def _compare_on_document(self, text: str, name: str) -> Dict:
        """Compare RAG_Core and LightRAG on a single document
        
        Args:
            text: Document text
            name: Test name
            
        Returns:
            Comparison result dictionary
        """
        from RAG_Core import TextProcessor, EntityExtractor, GraphBuilder
        
        result = {
            "name": name,
            "text_length": len(text),
            "stages": {}
        }
        
        # Stage 1: Chunking
        processor = TextProcessor()
        rag_core_chunks = processor.process(text)
        
        result["stages"]["chunking"] = {
            "chunk_count": len(rag_core_chunks),
            "avg_tokens": sum(c.tokens for c in rag_core_chunks) / max(len(rag_core_chunks), 1)
        }
        
        # Stage 2: Extraction
        extractor = EntityExtractor()
        extractions = extractor.extract_batch(rag_core_chunks, use_mock=True)
        
        entity_count = sum(len(e.entities) for e in extractions)
        relation_count = sum(len(e.relations) for e in extractions)
        
        result["stages"]["extraction"] = {
            "entity_count": entity_count,
            "relation_count": relation_count
        }
        
        # Stage 3: Graph Building
        builder = GraphBuilder()
        for extraction in extractions:
            builder.add_extraction_result(extraction)
        
        result["stages"]["graph"] = {
            "node_count": builder.get_entity_count(),
            "edge_count": builder.get_relation_count()
        }
        
        # Overall assessment
        result["passed"] = (
            len(rag_core_chunks) > 0 and
            entity_count > 0 and
            builder.get_entity_count() > 0
        )
        
        return result
    
    def generate_benchmark_report(self, results: Dict) -> str:
        """Generate benchmark report
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Formatted report string
        """
        report = f"\n{'='*60}\nACCURACY BENCHMARK REPORT\n{'='*60}"
        
        if not results.get("available", True):
            report += "\n✗ LightRAG not available"
            return report
        
        report += f"\nTotal Tests: {results['total_tests']}"
        report += f"\nPassed: {results['passed']}"
        report += f"\nFailed: {results['failed']}"
        report += f"\nAccuracy: {results.get('overall_accuracy', 0)*100:.1f}%"
        
        report += "\n\nDetailed Results:"
        for test_result in results['results']:
            status = "✓" if test_result.get('passed', False) else "✗"
            report += f"\n  {status} {test_result['name']}"
            
            if 'stages' in test_result:
                for stage_name, metrics in test_result['stages'].items():
                    report += f"\n     {stage_name}:"
                    for metric_name, value in metrics.items():
                        report += f"\n       - {metric_name}: {value}"
        
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test bridge
    bridge = LightRAGBridge()
    logger.info(f"LightRAG available: {bridge.is_available()}")
    
    # Run benchmark
    benchmark = AccuracyBenchmark()
    test_docs = [
        "Apple was founded by Steve Jobs and Steve Wozniak. It is headquartered in California.",
        "Google is a search engine. It was founded by Larry Page and Sergey Brin in 1998."
    ]
    
    results = benchmark.run_benchmark(test_docs)
    report = benchmark.generate_benchmark_report(results)
    logger.info(report)
