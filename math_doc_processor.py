#!/usr/bin/env python3
"""
Math-aware document processor for LightRAG.
Extracts LaTeX formulas, chunks intelligently, and prepares for embedding.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

class MathDocProcessor:
    """Process documents with LaTeX formulas for RAG."""
    
    def __init__(self, output_dir: str = "./math_docs_processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Regex patterns
        self.inline_math_pattern = r'\$[^\$]+\$'  # $...$
        self.display_math_pattern = r'\$\$[\s\S]*?\$\$'  # $$...$$
        self.latex_block_pattern = r'\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}'
        
    def extract_formulas(self, text: str) -> List[Dict]:
        """Extract all LaTeX formulas from text."""
        formulas = []
        
        # Display math ($$...$$)
        for match in re.finditer(self.display_math_pattern, text):
            formulas.append({
                'type': 'display',
                'latex': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Inline math ($...$) - excluding those already in display
        for match in re.finditer(self.inline_math_pattern, text):
            # Skip if overlaps with display math
            if not any(f['start'] <= match.start() < f['end'] for f in formulas):
                formulas.append({
                    'type': 'inline',
                    'latex': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return sorted(formulas, key=lambda x: x['start'])
    
    def smart_chunk(self, text: str, chunk_size: int = 400, 
                    overlap: int = 50, min_formula_gap: int = 200) -> List[str]:
        """
        Smart chunking that keeps formulas with their context.
        
        Args:
            text: Input text with LaTeX formulas
            chunk_size: Target chunk size (in chars)
            overlap: Overlap between chunks
            min_formula_gap: Minimum gap between formula and chunk boundary
        """
        formulas = self.extract_formulas(text)
        chunks = []
        pos = 0
        
        while pos < len(text):
            # Find safe chunk end (avoid breaking formulas)
            end = min(pos + chunk_size, len(text))
            
            # Expand if formula too close to boundary
            nearby_formulas = [f for f in formulas 
                             if abs(f['start'] - end) < min_formula_gap]
            if nearby_formulas:
                # Extend chunk to include nearby formula
                end = max(f['end'] for f in nearby_formulas)
            
            # Ensure we don't exceed text length
            end = min(end, len(text))
            
            # Backtrack to sentence boundary if possible
            if end < len(text):
                last_sent = max(
                    text.rfind('.', pos, end),
                    text.rfind('!', pos, end),
                    text.rfind('?', pos, end)
                )
                if last_sent > pos + chunk_size * 0.5:
                    end = last_sent + 1
            
            chunk = text[pos:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move position with overlap
            pos = end - overlap
            if pos >= len(text):
                break
        
        return chunks
    
    def create_formula_index(self, text: str) -> Dict:
        """Create index of formulas with context."""
        formulas = self.extract_formulas(text)
        index = {}
        
        for i, formula_data in enumerate(formulas):
            start = formula_data['start']
            latex = formula_data['latex']
            
            # Get surrounding context (50 chars before/after)
            context_start = max(0, start - 50)
            context_end = min(len(text), start + len(latex) + 50)
            context = text[context_start:context_end].strip()
            
            # Convert LaTeX to plain description for search
            description = self._latex_to_description(latex)
            
            index[f"formula_{i}"] = {
                'latex': latex,
                'type': formula_data['type'],
                'context': context,
                'description': description,
                'pos': start
            }
        
        return index
    
    @staticmethod
    def _latex_to_description(latex: str) -> str:
        """Convert LaTeX to plain text description."""
        text = latex.replace('$', '').strip()
        
        # Common substitutions
        replacements = {
            r'\frac': 'phân số',
            r'\sqrt': 'căn bậc',
            r'\int': 'tích phân',
            r'\sum': 'tổng',
            r'\cdot': 'nhân',
            r'\times': 'nhân',
            r'\div': 'chia',
            r'\\': 'dòng mới',
            r'\alpha': 'alpha',
            r'\beta': 'beta',
            r'\gamma': 'gamma',
            r'\theta': 'theta',
            r'\pi': 'pi',
            r'_{': 'chỉ số',
            r'^{': 'mũ',
        }
        
        for pattern, replace in replacements.items():
            text = text.replace(pattern, f" {replace} ")
        
        # Remove remaining LaTeX
        text = re.sub(r'[{}\\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text or latex
    
    def process_file(self, input_file: str, output_format: str = 'json') -> Dict:
        """
        Process a file and return chunks + formula index.
        
        Args:
            input_file: Path to markdown or text file (from Nougat output)
            output_format: 'json' or 'jsonl'
        
        Returns:
            Dict with chunks and formula_index
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = self.smart_chunk(text)
        formula_index = self.create_formula_index(text)
        
        result = {
            'source_file': str(input_file),
            'total_chunks': len(chunks),
            'total_formulas': len(formula_index),
            'chunks': chunks,
            'formula_index': formula_index
        }
        
        # Save output
        output_file = self.output_dir / f"{Path(input_file).stem}_processed.{output_format}"
        
        if output_format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        elif output_format == 'jsonl':
            with open(output_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps({'text': chunk, 'source': str(input_file)}, 
                                      ensure_ascii=False) + '\n')
        
        print(f"✅ Processed {input_file}")
        print(f"   Chunks: {len(chunks)}, Formulas: {len(formula_index)}")
        print(f"   Output: {output_file}")
        
        return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python math_doc_processor.py <input_file.mmd|txt>")
        sys.exit(1)
    
    processor = MathDocProcessor()
    result = processor.process_file(sys.argv[1], output_format='jsonl')
    
    # Print sample
    print(f"\n📄 Sample chunk 1:")
    print(result['chunks'][0][:300] + "...")
    
    print(f"\n📐 Sample formula:")
    if result['formula_index']:
        first_formula = list(result['formula_index'].values())[0]
        print(f"   LaTeX: {first_formula['latex']}")
        print(f"   Description: {first_formula['description']}")
