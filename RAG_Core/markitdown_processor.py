"""MarkItDown processing: multi-format file conversion to Markdown"""

import logging
import os
from typing import Optional

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

logger = logging.getLogger(__name__)

class MarkItDownProcessor:
    """Process various file formats and convert to Markdown using Microsoft MarkItDown"""
    
    def __init__(self, llm_client: Optional[object] = None, llm_model: Optional[str] = None):
        """Initialize MarkItDown processor
        
        Args:
            llm_client: Optional LLM client for advanced extraction (e.g., image description)
            llm_model: Optional LLM model name
        """
        if MarkItDown is None:
            logger.error("markitdown package not found. Please install it with 'pip install markitdown'")
            self.md = None
        else:
            self.md = MarkItDown(llm_client=llm_client, llm_model=llm_model)
    
    def is_available(self) -> bool:
        """Check if MarkItDown is available"""
        return self.md is not None
    
    def convert(self, file_path: str) -> str:
        """Convert a file to Markdown
        
        Args:
            file_path: Path to the file to convert
            
        Returns:
            Converted Markdown content as a string
        """
        if not self.md:
            logger.error("MarkItDown not initialized")
            return f"Error: MarkItDown not available. File: {file_path}"
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return f"Error: File not found: {file_path}"
            
        try:
            logger.info(f"Converting file to Markdown: {file_path}")
            result = self.md.convert(file_path)
            return result.text_content
        except Exception as e:
            logger.error(f"Error converting file {file_path}: {e}")
            return f"Error converting file: {str(e)}"

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = MarkItDownProcessor()
    if processor.is_available():
        # Test with a simple text file if exists, or just print status
        print("MarkItDown is available and ready.")
    else:
        print("MarkItDown is NOT available.")
