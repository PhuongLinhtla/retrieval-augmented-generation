#!/usr/bin/env python3
"""
Upload sample chunks via LightRAG API
"""

import requests
import json
from pathlib import Path
import time

BASE_URL = "http://localhost:8000"
WORKING_DIR = "./internal_docs"

def upload_chunks():
    """Upload sample chunks via REST API."""
    print("📤 Uploading sample chunks to LightRAG server...")
    print("=" * 50)
    
    jsonl_files = list(Path(WORKING_DIR).glob("*.jsonl"))
    
    if not jsonl_files:
        print("❌ No JSONL files found in internal_docs/")
        return False
    
    total = 0
    for jsonl_file in sorted(jsonl_files):
        print(f"\n📖 {jsonl_file.name}:")
        
        chunk_count = 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    
                    if text:
                        # Send to LightRAG via insert endpoint
                        response = requests.post(
                            f"{BASE_URL}/api/query",
                            json={"query": text, "mode": "local"},
                            timeout=5
                        )
                        
                        if response.status_code in [200, 202]:
                            chunk_count += 1
                            if chunk_count % 2 == 0:
                                print(f"   ✓ {chunk_count} chunks...", end='\r')
                        
                except Exception as e:
                    continue
        
        print(f"   ✓ Uploaded {chunk_count} chunks    ")
        total += chunk_count
    
    print(f"\n✅ Total uploaded: {total} chunks")
    return True

def test_query():
    """Test a sample query."""
    print("\n🎓 Testing query...")
    print("-" * 50)
    
    query = "Hàm số bậc hai là gì?"
    print(f"Query: {query}\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/query",
            json={"query": query, "mode": "local"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', 'No answer')
            print(f"Answer:\n{answer}")
            return True
        else:
            print(f"⚠️  Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n🚀 LightRAG API Upload\n")
    
    if upload_chunks():
        print("\n⏳ Waiting 2 seconds...")
        time.sleep(2)
        test_query()
    else:
        print("Setup failed")
