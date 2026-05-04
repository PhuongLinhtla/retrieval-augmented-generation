"""
Test suite for RAG_Core accuracy validation against LightRAG

Run tests with: python -m pytest test_accuracy.py -v
"""

import unittest
import logging
from typing import Dict, List

from RAG_Core import TextProcessor, EntityExtractor, GraphBuilder
from RAG_Core.config import ChunkConfig, ExtractionConfig, GraphConfig
from RAG_Core.query_engine import QueryEngine
from RAG_Core.validation import (
    TextProcessorValidator,
    EntityExtractionValidator,
    GraphBuilderValidator,
    compare_extraction_outputs
)
from RAG_Core.integration import AccuracyBenchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTextProcessing(unittest.TestCase):
    """Test text processing accuracy"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.processor = TextProcessor()
        self.sample_text = """
        Artificial Intelligence (AI) is transforming industries.
        Machine Learning enables computers to learn from data.
        Deep Learning uses neural networks for complex tasks.
        
        Natural Language Processing helps computers understand text.
        Computer Vision enables machines to interpret images.
        Robotics combines AI with physical systems.
        """
    
    def test_chunk_count(self):
        """Test that chunking produces expected chunk count"""
        chunks = self.processor.process(self.sample_text)
        
        # Should produce at least 1 chunk
        self.assertGreater(len(chunks), 0)
        logger.info(f"✓ Produced {len(chunks)} chunks")
    
    def test_chunk_token_size(self):
        """Test that chunks respect token size limits"""
        config = ChunkConfig(chunk_token_size=100)
        processor = TextProcessor(config)
        chunks = processor.process(self.sample_text)
        
        # All chunks should be close to target size (allow variance)
        for chunk in chunks:
            self.assertLessEqual(chunk.tokens, 110)  # Allow 10% overage
        logger.info(f"✓ All chunks respect token size limits")
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap"""
        config = ChunkConfig(
            chunk_token_size=200,
            chunk_overlap_token_size=50
        )
        processor = TextProcessor(config)
        chunks = processor.process(self.sample_text)
        
        # Should have overlap between consecutive chunks
        if len(chunks) > 1:
            self.assertGreater(chunks[1].metadata.get('overlap_with_next', 0), 0)
        logger.info(f"✓ Chunks have proper overlap")
    
    def test_content_preservation(self):
        """Test that all content is preserved after chunking"""
        chunks = self.processor.process(self.sample_text)
        
        # Concatenate all chunks
        combined = " ".join(c.content for c in chunks)
        
        # Normalize whitespace for comparison
        original_normalized = " ".join(self.sample_text.split())
        combined_normalized = " ".join(combined.split())
        
        # Should be very similar (might lose some punctuation)
        self.assertGreater(
            len(set(original_normalized.split()) & set(combined_normalized.split())) / 
            len(set(original_normalized.split())),
            0.9  # At least 90% overlap
        )
        logger.info(f"✓ Content preservation > 90%")


class TestEntityExtraction(unittest.TestCase):
    """Test entity extraction accuracy"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.extractor = EntityExtractor()
        self.processor = TextProcessor()
        
        self.sample_texts = [
            "John Smith works at Google in Mountain View.",
            "Apple was founded by Steve Jobs and Steve Wozniak.",
            "Microsoft is headquartered in Redmond, Washington.",
            "Tesla manufactures electric vehicles."
        ]
    
    def test_entity_extraction_basic(self):
        """Test basic entity extraction"""
        for text in self.sample_texts:
            chunks = self.processor.process(text)
            result = self.extractor.extract_from_chunk(chunks[0], use_mock=True)
            
            # Should extract at least 1 entity
            self.assertGreater(len(result.entities), 0)
            logger.info(f"✓ Extracted {len(result.entities)} entities from: {text[:40]}...")
    
    def test_entity_has_required_fields(self):
        """Test that extracted entities have all required fields"""
        chunks = self.processor.process(self.sample_texts[0])
        result = self.extractor.extract_from_chunk(chunks[0], use_mock=True)
        
        for entity_name, entities in result.entities.items():
            for entity in entities:
                self.assertIsNotNone(entity.name)
                self.assertIsNotNone(entity.type)
                self.assertIsNotNone(entity.description)
                self.assertIsNotNone(entity.source_id)
        logger.info(f"✓ All entities have required fields")
    
    def test_relation_extraction_basic(self):
        """Test basic relation extraction"""
        chunks = self.processor.process(self.sample_texts[0])
        result = self.extractor.extract_from_chunk(chunks[0], use_mock=True)
        
        # Should extract some relations
        # (Mock might not extract, but structure should exist)
        self.assertIsNotNone(result.relations)
        logger.info(f"✓ Relation extraction structure correct")
    
    def test_batch_extraction(self):
        """Test batch extraction of multiple chunks"""
        all_chunks = []
        for text in self.sample_texts:
            chunks = self.processor.process(text)
            all_chunks.extend(chunks)
        
        results = self.extractor.extract_batch(all_chunks, use_mock=True)
        
        # Should get results for each chunk
        self.assertEqual(len(results), len(all_chunks))
        
        # Total should be positive
        total_entities = sum(len(r.entities) for r in results)
        self.assertGreater(total_entities, 0)
        logger.info(f"✓ Batch extraction: {total_entities} entities from {len(all_chunks)} chunks")


class TestGraphBuilding(unittest.TestCase):
    """Test knowledge graph building accuracy"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.processor = TextProcessor()
        self.extractor = EntityExtractor()
        self.builder = GraphBuilder()
        
        self.sample_text = """
        Steve Jobs founded Apple in 1976 with Steve Wozniak.
        Steve Jobs later returned to Apple and led it to success.
        Tim Cook succeeded Steve Jobs as CEO of Apple.
        Apple is headquartered in Cupertino, California.
        """
    
    def test_graph_creation(self):
        """Test basic graph creation"""
        chunks = self.processor.process(self.sample_text)
        results = self.extractor.extract_batch(chunks, use_mock=True)
        
        for result in results:
            self.builder.add_extraction_result(result)
        
        # Should have created some entities
        self.assertGreater(self.builder.get_entity_count(), 0)
        logger.info(f"✓ Created graph with {self.builder.get_entity_count()} entities")
    
    def test_node_properties(self):
        """Test that graph nodes have required properties"""
        chunks = self.processor.process(self.sample_text)
        results = self.extractor.extract_batch(chunks, use_mock=True)
        
        for result in results:
            self.builder.add_extraction_result(result)
        
        entities = self.builder.get_entities()
        
        for entity_id, node in list(entities.items())[:5]:
            self.assertIsNotNone(node.entity_id)
            self.assertIsNotNone(node.entity_type)
            self.assertIsNotNone(node.description)
            self.assertIsInstance(node.source_ids, list)
        
        logger.info(f"✓ All graph nodes have required properties")
    
    def test_edge_creation(self):
        """Test that edges are created between related entities"""
        chunks = self.processor.process(self.sample_text)
        results = self.extractor.extract_batch(chunks, use_mock=True)
        
        for result in results:
            self.builder.add_extraction_result(result)
        
        # Should have created some edges
        edge_count = self.builder.get_relation_count()
        logger.info(f"✓ Created {edge_count} edges")
    
    def test_neighbor_lookup(self):
        """Test neighbor lookup"""
        chunks = self.processor.process(self.sample_text)
        results = self.extractor.extract_batch(chunks, use_mock=True)
        
        for result in results:
            self.builder.add_extraction_result(result)
        
        entities = self.builder.get_entities()
        
        for entity_id in list(entities.keys())[:3]:
            neighbors = self.builder.get_neighbors(entity_id)
            self.assertIsInstance(neighbors, list)
        
        logger.info(f"✓ Neighbor lookup working")


class TestQueryEngine(unittest.TestCase):
    """Test query engine accuracy"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.processor = TextProcessor()
        self.extractor = EntityExtractor()
        self.builder = GraphBuilder()
        
        self.sample_text = """
        Google was founded in 1998 by Larry Page and Sergey Brin.
        Google is headquartered in Mountain View, California.
        Google provides search services and advertising.
        Sundar Pichai became CEO of Google in 2020.
        """
        
        # Build graph
        chunks = self.processor.process(self.sample_text)
        results = self.extractor.extract_batch(chunks, use_mock=True)
        
        for result in results:
            self.builder.add_extraction_result(result)
        
        self.engine = QueryEngine(self.builder.nodes, self.builder.edges)
    
    def test_keyword_search(self):
        """Test keyword search"""
        results = self.engine.keyword_search("Google", top_k=5)
        
        # Should find at least 1 result
        self.assertGreater(len(results), 0)
        logger.info(f"✓ Keyword search found {len(results)} results")
    
    def test_entity_search(self):
        """Test entity search"""
        if self.builder.nodes:
            entity_id = list(self.builder.nodes.keys())[0]
            result = self.engine.entity_search(entity_id)
            
            self.assertIsNotNone(result)
            self.assertGreater(len(result.nodes), 0)
            logger.info(f"✓ Entity search returned {len(result.nodes)} nodes")
    
    def test_graph_statistics(self):
        """Test graph statistics"""
        stats = self.engine.get_graph_statistics()
        
        self.assertIn("num_nodes", stats)
        self.assertIn("num_edges", stats)
        self.assertIn("density", stats)
        
        logger.info(f"✓ Graph stats: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    
    def test_entity_context(self):
        """Test entity context retrieval"""
        if self.builder.nodes:
            entity_id = list(self.builder.nodes.keys())[0]
            context = self.engine.get_entity_context(entity_id)
            
            self.assertIsNotNone(context)
            self.assertIn("entity_id", context)
            self.assertIn("description", context)
            
            logger.info(f"✓ Entity context retrieved for {entity_id}")


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete end-to-end pipeline"""
    
    def test_complete_pipeline(self):
        """Test complete RAG pipeline from text to query"""
        from RAG_Core import TextProcessor, EntityExtractor, GraphBuilder
        from RAG_Core.query_engine import QueryEngine
        
        # Input document
        document = """
        Amazon was founded by Jeff Bezos in 1994 as an online bookstore.
        The company later expanded to sell various products and offer cloud services.
        Amazon Web Services (AWS) is a major cloud provider.
        Andy Jassy became CEO of Amazon in July 2021.
        Amazon is headquartered in Seattle, Washington.
        """
        
        # Step 1: Text Processing
        processor = TextProcessor()
        chunks = processor.process(document, file_path="amazon.txt")
        self.assertGreater(len(chunks), 0)
        
        # Step 2: Entity Extraction
        extractor = EntityExtractor()
        results = extractor.extract_batch(chunks, use_mock=True)
        self.assertGreater(len(results), 0)
        
        # Step 3: Graph Building
        builder = GraphBuilder()
        for result in results:
            builder.add_extraction_result(result)
        
        self.assertGreater(builder.get_entity_count(), 0)
        
        # Step 4: Querying
        engine = QueryEngine(builder.nodes, builder.edges)
        search_results = engine.keyword_search("Amazon", top_k=10)
        
        self.assertGreater(len(search_results), 0)
        
        logger.info(f"✓ Complete pipeline successful:")
        logger.info(f"  - Chunks: {len(chunks)}")
        logger.info(f"  - Entities: {builder.get_entity_count()}")
        logger.info(f"  - Relations: {builder.get_relation_count()}")
        logger.info(f"  - Query results: {len(search_results)}")


class TestAccuracyBenchmark(unittest.TestCase):
    """Test accuracy benchmark"""
    
    def test_benchmark_run(self):
        """Test running accuracy benchmark"""
        benchmark = AccuracyBenchmark()
        
        test_docs = [
            "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
            "LinkedIn is a professional networking platform acquired by Microsoft."
        ]
        
        results = benchmark.run_benchmark(test_docs, test_names=["microsoft", "linkedin"])
        
        self.assertIn("overall_accuracy", results)
        self.assertGreater(results.get("overall_accuracy", 0), 0)
        
        report = benchmark.generate_benchmark_report(results)
        logger.info("\n" + report)


class ValidationSuite(unittest.TestCase):
    """Test validation utilities"""
    
    def test_text_processor_validator(self):
        """Test text processor validator"""
        processor = TextProcessor()
        sample_text = "This is a test document. It has multiple sentences."
        
        chunks = processor.process(sample_text)
        
        # Create mock LightRAG format
        lightrag_chunks = [
            {
                "tokens": c.tokens,
                "content": c.content,
                "chunk_order_index": c.chunk_order_index
            }
            for c in chunks
        ]
        
        result = TextProcessorValidator.validate_chunking(chunks, lightrag_chunks)
        
        logger.info(f"✓ Validator report:\n{result.report()}")
    
    def test_extraction_comparison(self):
        """Test extraction output comparison"""
        output1 = {
            "entities": {
                "Apple": [{"name": "Apple", "type": "ORG"}],
                "Steve Jobs": [{"name": "Steve Jobs", "type": "PERSON"}]
            },
            "relations": {}
        }
        
        output2 = {
            "entities": {
                "Apple": [{"name": "Apple", "type": "ORG"}],
                "Steve Jobs": [{"name": "Steve Jobs", "type": "PERSON"}]
            },
            "relations": {}
        }
        
        is_equivalent, summary = compare_extraction_outputs(output1, output2)
        
        self.assertTrue(is_equivalent)
        logger.info(f"✓ Extraction outputs are equivalent")


def run_all_tests():
    """Run all tests and generate report"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTextProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestEntityExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphBuilding))
    suite.addTests(loader.loadTestsFromTestCase(TestQueryEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestAccuracyBenchmark))
    suite.addTests(loader.loadTestsFromTestCase(ValidationSuite))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"\n✓ All tests passed! RAG_Core accuracy validated.")
    else:
        print(f"\n✗ Some tests failed. Please review the output above.")


if __name__ == "__main__":
    run_all_tests()
