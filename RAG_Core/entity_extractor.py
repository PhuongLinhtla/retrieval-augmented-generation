"""Entity and relation extraction"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .types import Entity, Relation, ExtractionResult, Chunk
from .config import ExtractionConfig
from .utils import (
    sanitize_entity_name, 
    sanitize_entity_type, 
    parse_llm_output,
    format_timestamp
)
from .prompts import PROMPTS
import json

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities and relations from text chunks"""
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """Initialize entity extractor
        
        Args:
            config: ExtractionConfig instance
        """
        self.config = config or ExtractionConfig()
    
    def extract_from_chunk(
        self,
        chunk: Chunk,
        llm_func: Optional[callable] = None,
        use_mock: bool = True
    ) -> ExtractionResult:
        """Extract entities and relations from a single chunk
        
        Args:
            chunk: Chunk to extract from
            llm_func: LLM function for extraction (if None, uses mock)
            use_mock: Use mock extraction if llm_func is None
            
        Returns:
            ExtractionResult containing entities and relations
        """
        logger.info(f"Extracting from chunk: {chunk.chunk_id}")
        
        if llm_func is None:
            if use_mock:
                return self._mock_extract(chunk)
            else:
                raise ValueError("llm_func required when use_mock=False")
        
        # Get LLM output
        prompt = self._build_prompt(chunk)
        llm_output = llm_func(prompt)
        
        # Parse output
        result = self._parse_extraction_output(llm_output, chunk)
        
        # GLEANING (if configured)
        if self.config.max_gleaning > 0:
            for _ in range(self.config.max_gleaning):
                continue_prompt = PROMPTS.get("entity_continue_extraction_user_prompt", "Continue identifying missed entities...").format(
                    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
                    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
                    language=self.config.language
                )
                # This is a simplified gleaning: just one more pass
                gleaned_output = llm_func(prompt + "\n" + llm_output + "\n" + continue_prompt)
                gleaned_result = self._parse_extraction_output(gleaned_output, chunk)
                result.entities.extend(gleaned_result.entities)
                result.relations.extend(gleaned_result.relations)
        
        return result
    
    def extract_batch(
        self,
        chunks: List[Chunk],
        llm_func: Optional[callable] = None,
        use_mock: bool = True
    ) -> List[ExtractionResult]:
        """Extract from multiple chunks
        
        Args:
            chunks: List of chunks to extract from
            llm_func: LLM function for extraction
            use_mock: Use mock extraction if llm_func is None
            
        Returns:
            List of ExtractionResult objects
        """
        results = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
            result = self.extract_from_chunk(chunk, llm_func, use_mock)
            results.append(result)
        
        return results
    
    def extract_keywords(self, query: str, llm_func: callable) -> Dict[str, List[str]]:
        """Extract high-level and low-level keywords from query
        
        Args:
            query: User query
            llm_func: LLM function
            
        Returns:
            Dict with 'high_level_keywords' and 'low_level_keywords'
        """
        language = getattr(self.config, "language", "English")
        prompt = PROMPTS["keywords_extraction"].format(
            query=query,
            language=language,
            examples="\n".join(PROMPTS.get("keywords_extraction_examples", []))
        )
        output = llm_func(prompt)
        
        try:
            # Clean output in case LLM added markdown fences
            clean_output = output.strip()
            if clean_output.startswith("```"):
                clean_output = clean_output.split("\n", 1)[1]
            if clean_output.endswith("```"):
                clean_output = clean_output.rsplit("\n", 1)[0]
            
            data = json.loads(clean_output)
            return {
                "high_level_keywords": data.get("high_level_keywords", []),
                "low_level_keywords": data.get("low_level_keywords", [])
            }
        except Exception as e:
            logger.error(f"Failed to parse keywords: {e}. Output was: {output}")
            return {"high_level_keywords": [query], "low_level_keywords": [query]}

    def _build_prompt(self, chunk: Chunk) -> str:
        """Build extraction prompt from chunk"""
        entity_types = self.config.entity_types or ["Person", "Organization", "Location", "Event", "Concept"]
        
        language = getattr(self.config, "language", "English")
        
        system_prompt = PROMPTS["entity_extraction_system_prompt"].format(
            entity_types=json.dumps(entity_types),
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            language=language,
            examples="\n".join(PROMPTS.get("entity_extraction_examples", []))
        )
        
        user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
            entity_types=json.dumps(entity_types),
            input_text=chunk.content,
            language=language,
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        )
        
        return system_prompt + "\n\n" + user_prompt
    
    def _parse_extraction_output(
        self,
        output: str,
        chunk: Chunk
    ) -> ExtractionResult:
        """Parse LLM output into entities and relations
        
        Args:
            output: LLM output text
            chunk: Source chunk
            
        Returns:
            ExtractionResult object
        """
        result = ExtractionResult(chunk_id=chunk.chunk_id, timestamp=format_timestamp())
        
        # Parse records from output
        records = parse_llm_output(output)
        
        for record in records:
            if len(record) < 2:
                continue
            
            record_type = record[0].strip().lower()
            
            if "entity" in record_type and len(record) >= 4:
                entity = self._parse_entity(record, chunk)
                if entity:
                    if entity.name not in result.entities:
                        result.entities[entity.name] = []
                    result.entities[entity.name].append(entity)
            
            elif "relation" in record_type and len(record) >= 5:
                relation = self._parse_relation(record, chunk)
                if relation:
                    key = tuple(sorted([relation.src_id, relation.tgt_id]))
                    if key not in result.relations:
                        result.relations[key] = []
                    result.relations[key].append(relation)
        
        logger.info(f"Extracted {len(result.entities)} entities and {len(result.relations)} relations")
        return result
    
    def _parse_entity(self, record: List[str], chunk: Chunk) -> Optional[Entity]:
        """Parse entity from record
        
        Args:
            record: Record parts from parsed output
            chunk: Source chunk
            
        Returns:
            Entity object or None if invalid
        """
        try:
            name = sanitize_entity_name(record[1])
            entity_type = sanitize_entity_type(record[2])
            description = record[3].strip() if len(record) > 3 else ""
            
            if not name or not description:
                logger.debug(f"Skipping invalid entity: name={name}, desc={description}")
                return None
            
            return Entity(
                name=name,
                type=entity_type,
                description=description,
                source_id=chunk.chunk_id,
                file_path=chunk.file_path,
                timestamp=format_timestamp()
            )
        except Exception as e:
            logger.warning(f"Error parsing entity from record {record}: {e}")
            return None
    
    def _parse_relation(self, record: List[str], chunk: Chunk) -> Optional[Relation]:
        """Parse relation from record
        
        Args:
            record: Record parts from parsed output
            chunk: Source chunk
            
        Returns:
            Relation object or None if invalid
        """
        try:
            src_id = sanitize_entity_name(record[1])
            tgt_id = sanitize_entity_name(record[2])
            keywords = record[3].strip() if len(record) > 3 else ""
            description = record[4].strip() if len(record) > 4 else ""
            
            if not src_id or not tgt_id or not description:
                logger.debug(f"Skipping invalid relation: src={src_id}, tgt={tgt_id}, desc={description}")
                return None
            
            if src_id == tgt_id:
                logger.debug(f"Skipping self-relation: {src_id}")
                return None
            
            # Extract weight if present
            weight = 1.0
            if len(record) > 5:
                try:
                    weight = float(record[5])
                except (ValueError, IndexError):
                    weight = 1.0
            
            return Relation(
                src_id=src_id,
                tgt_id=tgt_id,
                description=description,
                keywords=keywords,
                weight=weight,
                source_id=chunk.chunk_id,
                file_path=chunk.file_path,
                timestamp=format_timestamp()
            )
        except Exception as e:
            logger.warning(f"Error parsing relation from record {record}: {e}")
            return None
    
    def _mock_extract(self, chunk: Chunk) -> ExtractionResult:
        """Generate mock extraction for testing
        
        Args:
            chunk: Chunk to extract from
            
        Returns:
            Mock ExtractionResult
        """
        result = ExtractionResult(chunk_id=chunk.chunk_id, timestamp=format_timestamp())
        
        # Simple mock: extract capitalized words as entities
        words = chunk.content.split()
        entities_found = set()
        
        for word in words:
            # Simple heuristic: capitalized words might be entities
            if word and word[0].isupper() and len(word) > 2:
                clean_word = word.rstrip('.,;:')
                if clean_word not in entities_found:
                    entity = Entity(
                        name=clean_word,
                        type="CONCEPT",
                        description=f"Found in {chunk.file_path}",
                        source_id=chunk.chunk_id,
                        file_path=chunk.file_path,
                        timestamp=format_timestamp()
                    )
                    result.entities.setdefault(clean_word, []).append(entity)
                    entities_found.add(clean_word)
        
        # Mock relations between first and second entity
        entity_names = list(result.entities.keys())
        if len(entity_names) >= 2:
            relation = Relation(
                src_id=entity_names[0],
                tgt_id=entity_names[1],
                description=f"Related through {chunk.file_path}",
                keywords="related",
                weight=0.5,
                source_id=chunk.chunk_id,
                file_path=chunk.file_path,
                timestamp=format_timestamp()
            )
            key = tuple(sorted([relation.src_id, relation.tgt_id]))
            result.relations[key] = [relation]
        
        return result


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from .text_processor import TextProcessor
    
    # Create text processor and extractor
    processor = TextProcessor()
    extractor = EntityExtractor()
    
    # Sample text
    sample_text = """
    John Smith works at Google in Mountain View.
    He is a software engineer specializing in machine learning.
    Google was founded by Larry Page and Sergey Brin.
    """
    
    # Process text
    chunks = processor.process(sample_text)
    
    # Extract entities and relations
    for chunk in chunks:
        result = extractor.extract_from_chunk(chunk, use_mock=True)
        
        print(f"\nChunk: {chunk.chunk_id}")
        print(f"Entities: {list(result.entities.keys())}")
        print(f"Relations: {list(result.relations.keys())}")
        
        for entity_name, entities in result.entities.items():
            for entity in entities:
                print(f"  - {entity.name} ({entity.type}): {entity.description}")
