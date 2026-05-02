#!/usr/bin/env python3
"""
LightRAG Demo - Local Ollama + Neo4j Setup
Hoàn toàn chạy local, miễn phí, không cần API key
"""

import os
import sys
import asyncio
import logging
from functools import partial
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create working directory
WORKING_DIR = "./internal_docs"
Path(WORKING_DIR).mkdir(exist_ok=True)


async def initialize_rag():
    """Initialize LightRAG with local Ollama"""
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
    
    logger.info("Initializing LightRAG...")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "mistral"),
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {"num_ctx": int(os.getenv("OLLAMA_LLM_NUM_CTX", "32768"))},
            "timeout": int(os.getenv("TIMEOUT", "300")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            max_token_size=int(os.getenv("OLLAMA_EMBEDDING_NUM_CTX", "8192")),
            func=partial(
                ollama_embed.func,
                embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
    )
    
    # Initialize storages
    await rag.initialize_storages()
    logger.info("✅ LightRAG initialized successfully!")
    
    return rag


async def insert_document(rag, file_path: str):
    """Insert a document into the knowledge graph"""
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        return False
    
    logger.info(f"📄 Reading document: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    logger.info(f"🔄 Processing document ({len(content)} chars)...")
    await rag.ainsert(content)
    
    logger.info("✅ Document inserted into knowledge graph!")
    return True


async def query_knowledge(rag, query: str, query_mode: str = "mix"):
    """Query the knowledge graph"""
    logger.info(f"🔍 Querying: '{query}' (mode: {query_mode})")
    
    result = await rag.aquery(query, param=QueryParam(mode=query_mode))
    
    logger.info(f"📊 Query Result:\n{result}\n")
    return result


async def main():
    """Main demo function"""
    print("\n" + "="*60)
    print("🚀 LightRAG Local Demo - Ollama + Neo4j")
    print("="*60 + "\n")
    
    # Check prerequisites
    logger.info("📋 Checking prerequisites...")
    
    # Check Ollama
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        logger.info("✅ Ollama is running")
    except Exception as e:
        logger.error("❌ Ollama is not running!")
        logger.error("   Please start Ollama first:")
        logger.error("   $ ollama serve")
        logger.error("\n   Then pull a model:")
        logger.error("   $ ollama pull mistral")
        logger.error("   $ ollama pull nomic-embed-text")
        return
    
    # Check Neo4j (optional, will be handled by connection attempt)
    logger.info("✅ Prerequisites checked\n")
    
    # Initialize RAG
    try:
        rag = await initialize_rag()
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG: {e}")
        logger.error("\n   Make sure Docker containers are running:")
        logger.error("   $ docker-compose -f docker-compose.local.yml up -d")
        return
    
    # Demo workflow
    logger.info("\n" + "="*60)
    logger.info("📚 Demo Workflow")
    logger.info("="*60 + "\n")
    
    # Create sample document
    sample_doc_path = os.path.join(WORKING_DIR, "sample_doc.txt")
    if not os.path.exists(sample_doc_path):
        logger.info("📝 Creating sample document...")
        sample_content = """
        LightRAG là một hệ thống Retrieval-Augmented Generation (RAG) đơn giản và nhanh.
        
        Các đặc điểm chính:
        - Xây dựng knowledge graph từ tài liệu
        - Hỗ trợ nhiều chế độ truy vấn (local, global, hybrid)
        - Có thể sử dụng với nhiều backend lưu trữ
        - Tích hợp với các LLM khác nhau
        
        LightRAG được phát triển bởi Hong Kong University of Science and Technology.
        Nó được thiết kế để cải thiện hiệu suất truy xuất thông tin so với RAG truyền thống.
        """
        
        with open(sample_doc_path, "w", encoding="utf-8") as f:
            f.write(sample_content)
        logger.info(f"✅ Sample document created: {sample_doc_path}\n")
    
    # Insert document
    await insert_document(rag, sample_doc_path)
    
    # Query examples
    logger.info("\n" + "="*60)
    logger.info("🔍 Query Examples")
    logger.info("="*60 + "\n")
    
    queries = [
        "LightRAG là gì?",
        "Những đặc điểm chính của LightRAG?",
    ]
    
    for query in queries:
        try:
            await query_knowledge(rag, query)
            await asyncio.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Demo completed!")
    logger.info("="*60)
    logger.info("\n💡 Next steps:")
    logger.info("   1. Add your own documents to " + WORKING_DIR)
    logger.info("   2. Run: lightrag-server")
    logger.info("   3. Open: http://localhost:8000")
    logger.info("   4. Upload documents and query via WebUI\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
