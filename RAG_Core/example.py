"""
Complete example demonstrating RAG Core functionality

This script shows how to:
1. Clean and chunk text
2. Extract entities and relations
3. Build a knowledge graph
4. Query the graph
"""

import logging
from RAG_Core import (
    TextProcessor,
    EntityExtractor,
    GraphBuilder,
)
from RAG_Core.query_engine import QueryEngine
from RAG_Core.config import (
    ChunkConfig,
    ExtractionConfig,
    GraphConfig,
    QueryConfig,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete RAG Core example"""
    
    logger.info("=" * 60)
    logger.info("RAG Core - Complete Example")
    logger.info("=" * 60)
    
    # =========================================================================
    # 1. SETUP CONFIGURATION
    # =========================================================================
    logger.info("\n[1] Setting up configuration...")
    
    chunk_config = ChunkConfig(
        chunk_token_size=512,
        chunk_overlap_token_size=50,
        split_by_character="\n\n"
    )
    
    extraction_config = ExtractionConfig(
        entity_types=["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "CONCEPT"],
        max_entities_per_chunk=100
    )
    
    graph_config = GraphConfig(
        max_entity_descriptions=5,
        max_relation_descriptions=5
    )
    
    query_config = QueryConfig(
        top_k=10,
        top_k_chunks=5
    )
    
    logger.info("✓ Configuration created")
    
    # =========================================================================
    # 2. CREATE COMPONENTS
    # =========================================================================
    logger.info("\n[2] Creating RAG Core components...")
    
    processor = TextProcessor(chunk_config)
    extractor = EntityExtractor(extraction_config)
    builder = GraphBuilder(graph_config)
    
    logger.info("✓ Components created")
    
    # =========================================================================
    # 3. SAMPLE DOCUMENT
    # =========================================================================
    logger.info("\n[3] Loading sample document...")
    
    document = """
    Apple Inc. is an American multinational technology company that designs, 
    develops, and sells consumer electronics, computer software, and online services.
    
    Apple was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne.
    Steve Jobs served as the company's CEO until his death in 2011.
    Tim Cook became CEO in August 2011 and continues to serve in this position.
    
    The company is headquartered in Cupertino, California, USA.
    Apple has offices in many countries including Japan, China, and the United Kingdom.
    
    Apple's main products include the iPhone, iPad, Mac computers, Apple Watch, and Apple TV.
    The iPhone is the company's flagship product and the best-selling smartphone worldwide.
    
    In 2024, Apple announced its artificial intelligence integration across its product ecosystem.
    The company is investing heavily in machine learning and AI research.
    
    Apple is also known for its retail stores and exceptional customer service.
    The Apple Store experience has revolutionized how consumers interact with technology.
    """
    
    logger.info(f"Document loaded: {len(document)} characters")
    
    # =========================================================================
    # 4. TEXT PROCESSING
    # =========================================================================
    logger.info("\n[4] Processing text (clean & chunk)...")
    
    chunks = processor.process(
        text=document,
        file_path="apple_overview.txt",
        chunk_id_prefix="apple"
    )
    
    logger.info(f"✓ Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:3], 1):
        logger.info(f"\n   Chunk {i} (ID: {chunk.chunk_id}):")
        logger.info(f"   - Tokens: {chunk.tokens}")
        logger.info(f"   - Content: {chunk.content[:80]}...")
    
    # =========================================================================
    # 5. ENTITY & RELATION EXTRACTION
    # =========================================================================
    logger.info("\n[5] Extracting entities and relations...")
    
    extraction_results = extractor.extract_batch(chunks, use_mock=True)
    
    logger.info(f"✓ Extracted from {len(extraction_results)} chunks")
    
    # Print extraction summary
    total_entities = sum(len(result.entities) for result in extraction_results)
    total_relations = sum(len(result.relations) for result in extraction_results)
    
    logger.info(f"   - Total entity types found: {total_entities}")
    logger.info(f"   - Total relation types found: {total_relations}")
    
    # Show sample entities
    if extraction_results and extraction_results[0].entities:
        sample_result = extraction_results[0]
        logger.info(f"\n   Sample entities from first chunk:")
        for entity_name, entities in list(sample_result.entities.items())[:3]:
            entity = entities[0]
            logger.info(f"   - {entity_name} ({entity.type})")
    
    # =========================================================================
    # 6. BUILD KNOWLEDGE GRAPH
    # =========================================================================
    logger.info("\n[6] Building knowledge graph...")
    
    for i, result in enumerate(extraction_results, 1):
        logger.info(f"   Processing extraction result {i}/{len(extraction_results)}")
        builder.add_extraction_result(result)
    
    entity_count = builder.get_entity_count()
    relation_count = builder.get_relation_count()
    
    logger.info(f"✓ Knowledge graph built:")
    logger.info(f"   - Entities: {entity_count}")
    logger.info(f"   - Relations: {relation_count}")
    
    # =========================================================================
    # 7. QUERY ENGINE SETUP
    # =========================================================================
    logger.info("\n[7] Setting up query engine...")
    
    engine = QueryEngine(builder.nodes, builder.edges, query_config)
    
    logger.info("✓ Query engine ready")
    
    # =========================================================================
    # 8. KEYWORD SEARCH
    # =========================================================================
    logger.info("\n[8] Keyword Search: 'Apple'")
    logger.info("-" * 60)
    
    search_results = engine.keyword_search("Apple", top_k=5)
    
    if search_results:
        for rank, (entity_id, score) in enumerate(search_results, 1):
            node = builder.get_node(entity_id)
            logger.info(f"\n   {rank}. {entity_id} (Score: {score:.3f})")
            if node:
                logger.info(f"      Type: {node.entity_type}")
                logger.info(f"      Description: {node.description[:60]}...")
    else:
        logger.info("   No results found")
    
    # =========================================================================
    # 9. ENTITY SEARCH
    # =========================================================================
    logger.info("\n[9] Entity Search with Neighbors")
    logger.info("-" * 60)
    
    if builder.nodes:
        # Find an organization entity
        org_entity = None
        for entity_id, node in builder.nodes.items():
            if node.entity_type == "ORGANIZATION":
                org_entity = entity_id
                break
        
        if org_entity:
            query_result = engine.entity_search(org_entity, max_depth=1)
            
            logger.info(f"\n   Entity: {org_entity}")
            logger.info(f"   Found: {len(query_result.nodes)} nodes, {len(query_result.edges)} edges")
            
            if query_result.nodes:
                logger.info(f"\n   Related entities:")
                for node in query_result.nodes[:5]:
                    logger.info(f"   - {node.entity_id} ({node.entity_type})")
    
    # =========================================================================
    # 10. GRAPH STATISTICS
    # =========================================================================
    logger.info("\n[10] Graph Statistics")
    logger.info("-" * 60)
    
    stats = engine.get_graph_statistics()
    
    logger.info(f"\n   Nodes: {stats['num_nodes']}")
    logger.info(f"   Edges: {stats['num_edges']}")
    logger.info(f"   Density: {stats['density']:.4f}")
    logger.info(f"   Avg Degree: {stats['avg_degree']:.2f}")
    
    logger.info(f"\n   Entity Types:")
    for entity_type, count in stats['entity_types'].items():
        logger.info(f"   - {entity_type}: {count}")
    
    if stats['most_connected']:
        logger.info(f"\n   Most Connected Entities:")
        for item in stats['most_connected']:
            logger.info(f"   - {item['entity_id']}: {item['degree']} connections")
    
    # =========================================================================
    # 11. ENTITY CONTEXT
    # =========================================================================
    logger.info("\n[11] Entity Context")
    logger.info("-" * 60)
    
    if builder.nodes:
        entity_ids = list(builder.nodes.keys())
        if entity_ids:
            test_entity = entity_ids[0]
            context = engine.get_entity_context(test_entity, context_size=3)
            
            if context:
                logger.info(f"\n   Entity: {context['entity_id']}")
                logger.info(f"   Type: {context['entity_type']}")
                logger.info(f"   Description: {context['description'][:100]}...")
                
                if context['related_entities']:
                    logger.info(f"\n   Related Entities:")
                    for rel in context['related_entities']:
                        logger.info(f"   - {rel['entity_id']} (weight: {rel['weight']:.2f})")
    
    # =========================================================================
    # 12. SIMILARITY COMPUTATION
    # =========================================================================
    logger.info("\n[12] Entity Similarity")
    logger.info("-" * 60)
    
    if len(builder.nodes) >= 2:
        entity_ids = list(builder.nodes.keys())
        entity1 = entity_ids[0]
        entity2 = entity_ids[1] if len(entity_ids) > 1 else entity_ids[0]
        
        similarity = engine.compute_similarity(entity1, entity2)
        
        logger.info(f"\n   Similarity between:")
        logger.info(f"   - {entity1}")
        logger.info(f"   - {entity2}")
        logger.info(f"   Score: {similarity:.3f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✓ Processed {len(chunks)} text chunks")
    logger.info(f"✓ Extracted entities and relations")
    logger.info(f"✓ Built knowledge graph with {entity_count} entities and {relation_count} relations")
    logger.info(f"✓ Performed multiple types of graph queries")
    logger.info("=" * 60)
    logger.info("\nRAG Core example completed successfully! 🚀\n")


if __name__ == "__main__":
    main()
