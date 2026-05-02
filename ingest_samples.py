#!/usr/bin/env python3
"""
Simple ingest script - bypass WebUI to load sample chunks into LightRAG
"""

import asyncio
import json
import os
from pathlib import Path

# Load environment BEFORE importing lightrag
from dotenv import load_dotenv
load_dotenv(override=True)

# Ensure embedding function is set
os.environ.setdefault('EMBEDDING_MODEL', 'nomic-embed-text')
os.environ.setdefault('EMBEDDING_BINDING', 'ollama')

from lightrag import LightRAG

async def ingest_samples():
    """Ingest sample math chunks."""
    print("🚀 LightRAG - Direct Ingestion")
    print("=" * 50)
    
    working_dir = "./internal_docs"
    
    # Initialize RAG
    print("📂 Initializing LightRAG...")
    rag = LightRAG(working_dir=working_dir)
    await rag.initialize_storages()
    print("✓ LightRAG initialized")
    
    # Load and ingest chunks
    jsonl_files = list(Path(working_dir).glob("*.jsonl"))
    
    if not jsonl_files:
        print("❌ No JSONL files found")
        return
    
    print(f"\n📚 Found {len(jsonl_files)} file(s)")
    
    total_chunks = 0
    for jsonl_file in sorted(jsonl_files):
        print(f"\n📖 Ingesting: {jsonl_file.name}")
        
        chunk_count = 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    if text:
                        await rag.ainsert(text)
                        chunk_count += 1
                except json.JSONDecodeError:
                    continue
        
        print(f"   ✓ Ingested {chunk_count} chunks")
        total_chunks += chunk_count
    
    print(f"\n✅ Total ingested: {total_chunks} chunks")
    print("\n🎓 Ready to query! Sample questions:")
    print('   - "Hàm số bậc hai là gì?"')
    print('   - "Giải phương trình x^2 - 5x + 6 = 0"')

if __name__ == "__main__":
    asyncio.run(ingest_samples())
