#!/usr/bin/env python3
"""
Quick Math RAG Demo - Simplified version
Demonstrates end-to-end flow without heavy processing
"""

import json
from pathlib import Path

def simple_chunk_file(filepath: str, chunk_size: int = 300):
    """Simple chunking (no formula extraction yet)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def process_sample_files():
    """Process sample markdown files to JSONL."""
    input_dir = Path("ocr_input")
    output_dir = Path("math_docs_processed")
    output_dir.mkdir(exist_ok=True)
    
    for mmd_file in input_dir.glob("*.mmd"):
        print(f"📄 Processing: {mmd_file.name}")
        
        chunks = simple_chunk_file(str(mmd_file))
        
        output_file = output_dir / f"{mmd_file.stem}_processed.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps({'text': chunk, 'source': mmd_file.name}, ensure_ascii=False) + '\n')
        
        print(f"   ✓ {len(chunks)} chunks → {output_file.name}")
    
    # Copy to internal_docs for ingestion
    import shutil
    for jsonl_file in output_dir.glob("*.jsonl"):
        shutil.copy(jsonl_file, "internal_docs/")
    
    print(f"\n✅ Files ready for ingestion in internal_docs/")

if __name__ == "__main__":
    process_sample_files()
