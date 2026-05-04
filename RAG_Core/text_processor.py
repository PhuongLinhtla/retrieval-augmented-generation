"""Text processing: cleaning and chunking"""

import logging
import hashlib
from typing import List, Optional
from dataclasses import asdict

from .types import Chunk
from .config import ChunkConfig
from .utils import SimpleTokenizer, TextCleaner, compute_hash_id

logger = logging.getLogger(__name__)


class TextProcessor:
    """Process and chunk raw text"""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """Initialize text processor
        
        Args:
            config: ChunkConfig instance with chunking parameters
        """
        self.config = config or ChunkConfig()
        self.tokenizer = SimpleTokenizer()
        self.cleaner = TextCleaner()
    
    def clean_text(self, text: str) -> str:
        """Clean raw text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        logger.info(f"Cleaning text (length: {len(text)})")
        cleaned = self.cleaner.clean(text)
        logger.debug(f"Cleaned text length: {len(cleaned)}")
        return cleaned
    
    def chunk_text(
        self,
        text: str,
        file_path: str = "unknown_source",
        chunk_id_prefix: str = ""
    ) -> List[Chunk]:
        """Chunk text into overlapping chunks
        
        This function chunks text by either:
        1. Character markers (if split_by_character is specified)
        2. Token size (sliding window with overlap)
        
        Args:
            text: Text to chunk
            file_path: Source file path for tracking
            chunk_id_prefix: Prefix for chunk IDs
            
        Returns:
            List of Chunk objects
            
        Raises:
            ValueError: If a chunk exceeds token limit with split_by_character_only=True
        """
        logger.info(f"Chunking text (length: {len(text)})")
        
        text = self.clean_text(text)
        chunks = []
        
        if self.config.split_by_character and not self.config.split_by_character_only:
            # Split by character first, then by token size if needed
            chunks = self._chunk_with_character_split(text, file_path, chunk_id_prefix)
        elif self.config.split_by_character and self.config.split_by_character_only:
            # Only split by character, raise error if chunk too large
            chunks = self._chunk_character_only(text, file_path, chunk_id_prefix)
        else:
            # Only split by token size
            chunks = self._chunk_by_tokens(text, file_path, chunk_id_prefix)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _chunk_by_tokens(
        self,
        text: str,
        file_path: str,
        chunk_id_prefix: str
    ) -> List[Chunk]:
        """Chunk text by token size with overlap
        
        Args:
            text: Text to chunk
            file_path: Source file path
            chunk_id_prefix: Prefix for chunk IDs
            
        Returns:
            List of Chunk objects
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        chunk_size = self.config.chunk_token_size
        overlap = self.config.chunk_overlap_token_size
        step = chunk_size - overlap
        
        chunk_order_index = 0
        for start_idx in range(0, len(tokens), step):
            end_idx = min(start_idx + chunk_size, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk ID
            chunk_hash = compute_hash_id(chunk_text, prefix="chunk-")
            if chunk_id_prefix:
                chunk_id = f"{chunk_id_prefix}_{chunk_order_index}"
            else:
                chunk_id = chunk_hash
            
            chunk = Chunk(
                chunk_id=chunk_id,
                content=chunk_text.strip(),
                tokens=len(chunk_tokens),
                chunk_order_index=chunk_order_index,
                file_path=file_path,
                metadata={
                    "start_token": start_idx,
                    "end_token": end_idx,
                    "overlap_with_next": overlap if end_idx < len(tokens) else 0
                }
            )
            
            chunks.append(chunk)
            chunk_order_index += 1
        
        return chunks
    
    def _chunk_with_character_split(
        self,
        text: str,
        file_path: str,
        chunk_id_prefix: str
    ) -> List[Chunk]:
        """Chunk by character marker, then by tokens if needed
        
        Args:
            text: Text to chunk
            file_path: Source file path
            chunk_id_prefix: Prefix for chunk IDs
            
        Returns:
            List of Chunk objects
        """
        # Split by character marker
        raw_chunks = text.split(self.config.split_by_character)
        
        # Process each section
        all_chunks = []
        chunk_order_index = 0
        
        for section in raw_chunks:
            section_tokens = self.tokenizer.encode(section)
            chunk_size = self.config.chunk_token_size
            
            if len(section_tokens) <= chunk_size:
                # Section fits in one chunk
                chunk = Chunk(
                    chunk_id=f"{chunk_id_prefix}_{chunk_order_index}" if chunk_id_prefix else compute_hash_id(section),
                    content=section.strip(),
                    tokens=len(section_tokens),
                    chunk_order_index=chunk_order_index,
                    file_path=file_path
                )
                all_chunks.append(chunk)
                chunk_order_index += 1
            else:
                # Section too large, split by tokens
                overlap = self.config.chunk_overlap_token_size
                step = chunk_size - overlap
                
                for start_idx in range(0, len(section_tokens), step):
                    end_idx = min(start_idx + chunk_size, len(section_tokens))
                    chunk_tokens = section_tokens[start_idx:end_idx]
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    
                    chunk = Chunk(
                        chunk_id=f"{chunk_id_prefix}_{chunk_order_index}" if chunk_id_prefix else compute_hash_id(chunk_text),
                        content=chunk_text.strip(),
                        tokens=len(chunk_tokens),
                        chunk_order_index=chunk_order_index,
                        file_path=file_path
                    )
                    all_chunks.append(chunk)
                    chunk_order_index += 1
        
        return all_chunks
    
    def _chunk_character_only(
        self,
        text: str,
        file_path: str,
        chunk_id_prefix: str
    ) -> List[Chunk]:
        """Chunk only by character marker, raise error if chunk too large
        
        Args:
            text: Text to chunk
            file_path: Source file path
            chunk_id_prefix: Prefix for chunk IDs
            
        Returns:
            List of Chunk objects
            
        Raises:
            ValueError: If any chunk exceeds token limit
        """
        raw_chunks = text.split(self.config.split_by_character)
        chunks = []
        chunk_order_index = 0
        
        for section in raw_chunks:
            section_tokens = self.tokenizer.encode(section)
            
            # Verify chunk is not too large
            if len(section_tokens) > self.config.chunk_token_size:
                error_msg = (
                    f"Chunk too large ({len(section_tokens)} > {self.config.chunk_token_size} tokens). "
                    f"Preview: {section[:100]}..."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            chunk = Chunk(
                chunk_id=f"{chunk_id_prefix}_{chunk_order_index}" if chunk_id_prefix else compute_hash_id(section),
                content=section.strip(),
                tokens=len(section_tokens),
                chunk_order_index=chunk_order_index,
                file_path=file_path
            )
            chunks.append(chunk)
            chunk_order_index += 1
        
        return chunks
    
    def process(
        self,
        text: str,
        file_path: str = "unknown_source",
        chunk_id_prefix: str = ""
    ) -> List[Chunk]:
        """End-to-end text processing: clean + chunk
        
        Args:
            text: Raw text to process
            file_path: Source file path
            chunk_id_prefix: Prefix for chunk IDs
            
        Returns:
            List of processed Chunk objects
        """
        return self.chunk_text(text, file_path, chunk_id_prefix)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create processor with custom config
    config = ChunkConfig(
        chunk_token_size=512,
        chunk_overlap_token_size=50,
        split_by_character="\n\n"
    )
    processor = TextProcessor(config)
    
    # Sample text
    sample_text = """
    This is a sample document about machine learning.
    Machine learning is a subset of artificial intelligence.
    
    It includes various techniques like supervised learning and unsupervised learning.
    Deep learning is a specialized branch of machine learning.
    
    Neural networks are the foundation of deep learning.
    They are inspired by biological neural systems.
    """
    
    # Process text
    chunks = processor.process(sample_text, file_path="sample.txt", chunk_id_prefix="doc1")
    
    # Print results
    for chunk in chunks:
        print(f"\nChunk {chunk.chunk_id}:")
        print(f"  Tokens: {chunk.tokens}")
        print(f"  Content: {chunk.content[:100]}...")
