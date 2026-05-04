"""LLM client for RAG_Core"""

import logging
import requests
import json
import os
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class LLMClient:
    """Base class for LLM clients"""
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError

class OllamaClient(LLMClient):
    """Client for Ollama local LLM"""
    
    def __init__(self, model: str = "mistral", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.host}/api/chat"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        try:
            logger.info(f"Calling Ollama ({self.model}) with prompt length: {len(prompt)}")
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"Error: Failed to connect to Ollama at {self.host}. {str(e)}"

def get_llm_client(purpose: str = "general") -> LLMClient:
    """Get LLM client based on environment or defaults
    
    Args:
        purpose: 'general' or 'translation'
    """
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    if purpose == "translation":
        # Using the model pulled by the user (ALMA)
        model = os.getenv("OLLAMA_TRANSLATION_MODEL", "winkefinger/alma-13b")
    else:
        model = os.getenv("OLLAMA_MODEL", "mistral")
        
    return OllamaClient(model=model, host=host)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = get_llm_client()
    print(f"Testing LLM client ({client.__class__.__name__})...")
    response = client.generate("Say hello!")
    print(f"Response: {response}")
