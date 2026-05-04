"""Utility functions for RAG Core"""

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """Simple tokenizer using word splitting"""
    
    @staticmethod
    def encode(text: str) -> List[str]:
        """Tokenize text into tokens"""
        # Simple: split by whitespace
        return text.split()
    
    @staticmethod
    def decode(tokens: List[str]) -> str:
        """Convert tokens back to text"""
        return " ".join(tokens)
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count number of tokens"""
        return len(SimpleTokenizer.encode(text))


class TextCleaner:
    """Clean and normalize text"""
    
    # Patterns for cleaning
    CONTROL_CHARS = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    MULTIPLE_SPACES = re.compile(r' +')
    MULTIPLE_NEWLINES = re.compile(r'\n\n+')
    
    @staticmethod
    def clean(text: str) -> str:
        """Clean text by removing unwanted characters"""
        # Remove control characters
        text = TextCleaner.CONTROL_CHARS.sub('', text)
        
        # Replace multiple spaces with single space
        text = TextCleaner.MULTIPLE_SPACES.sub(' ', text)
        
        # Replace multiple newlines with double newline
        text = TextCleaner.MULTIPLE_NEWLINES.sub('\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def sanitize(text: str, remove_quotes: bool = False) -> str:
        """Sanitize extracted text"""
        # Remove leading/trailing quotes
        if remove_quotes:
            text = text.strip('"\'')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text


def compute_hash_id(text: str, prefix: str = "") -> str:
    """Compute hash ID for text"""
    hash_obj = hashlib.md5(text.encode())
    hash_id = hash_obj.hexdigest()
    return f"{prefix}{hash_id}" if prefix else hash_id


def merge_descriptions(descriptions: List[str], separator: str = " | ") -> str:
    """Merge multiple descriptions into one"""
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for desc in descriptions:
        if desc and desc not in seen:
            unique.append(desc)
            seen.add(desc)
    
    return separator.join(unique) if unique else ""


def sanitize_entity_name(name: str, max_length: int = 128) -> str:
    """Sanitize entity name"""
    # Remove quotes
    name = name.strip('"\'')
    
    # Replace multiple spaces with single space
    name = re.sub(r' +', ' ', name)
    
    # Remove special characters that cause issues
    name = re.sub(r'[\|\<\>\[\]\{\}]', '', name)
    
    # Truncate if too long
    if len(name) > max_length:
        name = name[:max_length]
        logger.warning(f"Entity name truncated: {name}...")
    
    return name.strip()


def sanitize_entity_type(entity_type: str) -> str:
    """Sanitize entity type"""
    # Convert to uppercase
    entity_type = entity_type.upper()
    
    # Remove spaces and special chars
    entity_type = re.sub(r'[\s\-_]', '', entity_type)
    
    # Handle comma-separated types (take first)
    if ',' in entity_type:
        entity_type = entity_type.split(',')[0]
    
    return entity_type.strip()


def parse_llm_output(output: str, delimiter: str = "<|#|>") -> List[List[str]]:
    """Parse LLM output into records
    
    Expected format:
    entity<|#|>name<|#|>type<|#|>description
    relation<|#|>source<|#|>target<|#|>keywords<|#|>description
    """
    records = []
    lines = output.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Split by delimiter
        parts = line.split(delimiter)
        if len(parts) >= 2:
            records.append(parts)
    
    return records


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text"""
    import json
    
    # Try to find JSON-like structure
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def split_text_by_delimiter(text: str, delimiters: List[str]) -> List[str]:
    """Split text by multiple delimiters while preserving order"""
    if not delimiters:
        return [text]
    
    # Create regex pattern from delimiters
    pattern = '|'.join(re.escape(d) for d in delimiters)
    parts = re.split(pattern, text)
    
    return [p.strip() for p in parts if p.strip()]


def format_timestamp() -> int:
    """Get current timestamp"""
    return int(datetime.now().timestamp())


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Batch items into groups"""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def deduplicate_with_order(items: List[Any]) -> List[Any]:
    """Deduplicate while preserving order"""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
