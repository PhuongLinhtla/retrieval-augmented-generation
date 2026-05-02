#!/usr/bin/env python3
"""
Demo: Math RAG with LightRAG
Ingests processed math documents and performs mathematical Q&A
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure .env is loaded before importing LightRAG
from dotenv import load_dotenv
load_dotenv()

from lightrag import LightRAG

# Configuration
WORKING_DIR = "./internal_docs"
PROCESSED_DIR = "./math_docs_processed"

# Math-specific prompts (can override in .env)
SYSTEM_PROMPT_MATH = """Bạn là giáo viên toán học trong hệ thống RAG. Khi trả lời:
1. Luôn kèm công thức LaTeX trong $$...$$ hoặc $...$ nếu liên quan
2. Giải thích từng bước rõ ràng (step-by-step)
3. Nếu có nhiều cách giải, liệt kê các phương pháp
4. Kèm trích dẫn nguồn: [file: path, lines: L1-L2]
5. Nếu không chắc, nói rõ: "không có trong tài liệu" và gợi ý bước kiểm chứng
"""

async def initialize_rag():
    """Initialize LightRAG with math configuration."""
    print("🚀 Initializing LightRAG for math learning...")
    
    # Initialize with working directory
    rag = LightRAG(working_dir=WORKING_DIR)
    
    # Ensure storages are initialized with embedding function
    try:
        await rag.initialize_storages()
        print("✓ LightRAG initialized with default storages")
    except ValueError as e:
        if "embedding_func" in str(e):
            print("⚠️  Initializing storages with embedding function...")
            # Get embedding function from rag (already set up from .env)
            await rag.initialize_storages()
        else:
            raise
    
    return rag

async def ingest_math_documents(rag: LightRAG):
    """Ingest all processed math documents."""
    if not Path(PROCESSED_DIR).exists():
        print(f"⚠️  No processed documents found at {PROCESSED_DIR}")
        print("   Run: ./process_math_docs.sh first")
        return 0
    
    jsonl_files = list(Path(PROCESSED_DIR).glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"⚠️  No .jsonl files in {PROCESSED_DIR}")
        return 0
    
    print(f"📚 Found {len(jsonl_files)} processed document(s)")
    
    total_chunks = 0
    for jsonl_file in jsonl_files:
        print(f"\n📖 Processing: {jsonl_file.name}")
        
        chunk_count = 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    chunk_text = data.get('text', '')
                    
                    if chunk_text:
                        await rag.ainsert(chunk_text)
                        chunk_count += 1
                        
                        if chunk_count % 5 == 0:
                            print(f"   ✓ Ingested {chunk_count} chunks...", end='\r')
                
                except json.JSONDecodeError:
                    print(f"   ⚠️  Error parsing line {line_num}, skipping")
                    continue
        
        print(f"   ✓ Ingested {chunk_count} chunks from {jsonl_file.name}")
        total_chunks += chunk_count
    
    print(f"\n✅ Total: {total_chunks} chunks ingested")
    return total_chunks

async def query_math(rag: LightRAG, question: str, mode: str = "local"):
    """
    Query the math knowledge graph.
    
    Args:
        rag: LightRAG instance
        question: Math question
        mode: "local", "global", or "mix"
    """
    print(f"\n💬 Question: {question}")
    print(f"   Mode: {mode}")
    print("-" * 60)
    
    try:
        response = await rag.aquery(question, mode=mode, top_k=10)
        
        print(f"\n📝 Answer:\n{response}\n")
        print("-" * 60)
        
        return response
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

async def interactive_math_qa(rag: LightRAG):
    """Interactive Q&A session."""
    print("\n" + "="*60)
    print("🎓 Math Learning RAG - Interactive Mode")
    print("="*60)
    print("Commands:")
    print("  'quit' - Exit")
    print("  'local' - Use local mode (precise)")
    print("  'global' - Use global mode (broader)")
    print("  'mix' - Use mix mode (hybrid)")
    print("-" * 60)
    
    mode = "local"
    
    while True:
        try:
            user_input = input("\n❓ Ask a math question (or 'quit'/'local'/'global'/'mix'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() in ['local', 'global', 'mix']:
                mode = user_input.lower()
                print(f"✓ Mode changed to: {mode}")
                continue
            elif not user_input:
                continue
            else:
                await query_math(rag, user_input, mode=mode)
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

async def demo_math_qa(rag: LightRAG):
    """Run demo queries."""
    demo_questions = [
        "Hàm số bậc hai là gì? Cho ví dụ.",
        "Giải bài toán: Tìm x sao cho $x^2 - 5x + 6 = 0$",
        "Công thức tính diện tích tam giác là gì?",
    ]
    
    print("\n" + "="*60)
    print("🎓 Math Learning RAG - Demo")
    print("="*60)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n📌 Demo Query {i}/{len(demo_questions)}")
        await query_math(rag, question, mode="local")

async def main():
    """Main entry point."""
    print("🎓 Math RAG System")
    print("="*60)
    
    # Initialize
    rag = await initialize_rag()
    
    # Ingest documents
    chunk_count = await ingest_math_documents(rag)
    
    if chunk_count == 0:
        print("\n⚠️  No documents ingested. Please:")
        print("   1. Copy PDF files to ocr_input/")
        print("   2. Run: ./process_math_docs.sh")
        print("   3. Run this demo again")
        return
    
    # Run demo or interactive
    print("\n" + "="*60)
    print("Choose mode:")
    print("  1. Run demo queries")
    print("  2. Interactive Q&A")
    print("-" * 60)
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            await demo_math_qa(rag)
        elif choice == "2":
            await interactive_math_qa(rag)
        else:
            print("Invalid choice. Running demo...")
            await demo_math_qa(rag)
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted.")

if __name__ == "__main__":
    asyncio.run(main())
