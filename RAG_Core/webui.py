"""Web UI for RAG_Core pipeline.

Run:
    python webui.py

Then open:
    http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

# Add parent directory to sys.path to allow importing RAG_Core
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from flask import Flask, render_template, request
import tempfile
import json

from RAG_Core import pdf_processor  # noqa: E402
from RAG_Core import EntityExtractor, GraphBuilder, TextProcessor, MarkItDownProcessor  # noqa: E402
from RAG_Core.config import ChunkConfig, ExtractionConfig, GraphConfig, QueryConfig, QueryMode  # noqa: E402
from RAG_Core.query_engine import QueryEngine  # noqa: E402


app = Flask(__name__, template_folder="templates", static_folder="static")


def _safe_int(value: str, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def _safe_float(value: str, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def run_pipeline(form: Dict[str, Any]) -> Dict[str, Any]:
    text = (form.get("text") or "").strip()
    source_name = (form.get("source_name") or "web_input.txt").strip() or "web_input.txt"
    keyword_query = (form.get("keyword_query") or "").strip()
    entity_query = (form.get("entity_query") or "").strip()
    nl_question = (form.get("nl_question") or "").strip()

    chunk_size = _safe_int(form.get("chunk_size"), default=512, min_value=64, max_value=4000)
    chunk_overlap = _safe_int(form.get("chunk_overlap"), default=50, min_value=0, max_value=2000)
    top_k = _safe_int(form.get("top_k"), default=8, min_value=1, max_value=50)
    max_depth = _safe_int(form.get("max_depth"), default=2, min_value=1, max_value=4)
    similarity_threshold = _safe_float(form.get("similarity_threshold"), default=0.5, min_value=0.0, max_value=1.0)
    use_translation = form.get("use_translation") == "on"
    query_mode_val = form.get("query_mode") or "hybrid"
    try:
        query_mode = QueryMode(query_mode_val)
    except ValueError:
        query_mode = QueryMode.HYBRID

    if not text:
        raise ValueError("Please paste text content or upload a text file.")

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)

    chunk_config = ChunkConfig(
        chunk_token_size=chunk_size,
        chunk_overlap_token_size=chunk_overlap,
        split_by_character="\n\n",
    )
    extraction_config = ExtractionConfig()
    graph_config = GraphConfig()
    query_config = QueryConfig(
        top_k=top_k, 
        cosine_threshold=similarity_threshold,
        mode=query_mode,
        use_translation=use_translation
    )

    processor = TextProcessor(chunk_config)
    extractor = EntityExtractor(extraction_config)
    builder = GraphBuilder(graph_config)

    chunks = processor.process(text=text, file_path=source_name, chunk_id_prefix="web")
    extraction_results = extractor.extract_batch(chunks, use_mock=True)

    for result in extraction_results:
        builder.add_extraction_result(result)

    # Convert chunks list to dict for easier lookup
    chunks_dict = {c.chunk_id: c for c in chunks}
    engine = QueryEngine(builder.nodes, builder.edges, query_config, chunks=chunks_dict)

    entity_rows: List[Dict[str, Any]] = []
    for node in builder.nodes.values():
        entity_rows.append(
            {
                "entity_id": node.entity_id,
                "entity_type": node.entity_type,
                "description": node.description,
                "source_count": len(node.source_ids),
            }
        )

    relation_rows: List[Dict[str, Any]] = []
    for edge in builder.edges.values():
        relation_rows.append(
            {
                "src": edge.src_id,
                "tgt": edge.tgt_id,
                "keywords": edge.keywords,
                "weight": round(edge.weight, 3),
                "description": edge.description,
            }
        )

    entity_rows.sort(key=lambda item: item["entity_id"].lower())
    relation_rows.sort(key=lambda item: (item["src"].lower(), item["tgt"].lower()))

    keyword_results: List[Tuple[str, float]] = []
    if keyword_query:
        keyword_results = engine.keyword_search(keyword_query, top_k=top_k)

    entity_result = None
    if entity_query:
        entity_result = engine.entity_search(entity_query, max_depth=max_depth)

    llm_answer = None
    if nl_question:
        try:
            llm_answer = engine.answer(nl_question)
        except Exception as e:
            llm_answer = f"Error generating answer: {e}"

    stats = engine.get_graph_statistics()
    # attach any pre-extracted formulas (from PDF OCR / MathPix)
    formulas: List[Dict[str, Any]] = form.get("_formulas") or []

    return {
        "text_preview": text[:600],
        "text_length": len(text),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "entity_rows": entity_rows,
        "relation_rows": relation_rows,
        "formulas": formulas,
        "keyword_query": keyword_query,
        "keyword_results": keyword_results,
        "entity_query": entity_query,
        "entity_result": entity_result,
        "nl_question": nl_question,
        "llm_answer": llm_answer,
        "stats": stats,
        "source_name": source_name,
        "config_echo": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "max_depth": max_depth,
            "similarity_threshold": similarity_threshold,
        },
    }


@app.route("/", methods=["GET", "POST"])
def index():
    error = ""
    output = None

    default_form = {
        "text": "",
        "source_name": "web_input.txt",
        "keyword_query": "",
        "entity_query": "",
        "chunk_size": "512",
        "chunk_overlap": "50",
        "top_k": "8",
        "max_depth": "2",
        "similarity_threshold": "0.5",
        "nl_question": "",
        "query_mode": "hybrid",
        "use_translation": "off",
    }

    form_values = dict(default_form)

    if request.method == "POST":
        form_values.update({k: v for k, v in request.form.items()})

        file_obj = request.files.get("text_file")
        if file_obj and file_obj.filename:
            filename = file_obj.filename
            
            # Use MarkItDown for all file types
            processor_md = MarkItDownProcessor()
            
            if processor_md.is_available():
                # Write uploaded file to a temporary file for MarkItDown
                suffix = os.path.splitext(filename)[1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                try:
                    tmp.write(file_obj.read())
                    tmp.flush()
                    tmp.close()
                    
                    # Convert to Markdown
                    text_extracted = processor_md.convert(tmp.name)
                    form_values["text"] = text_extracted
                    if not form_values.get("source_name"):
                        form_values["source_name"] = filename
                        
                    # If it's a PDF, we might still want to try the old pdf_processor for formulas
                    # but for now let's prioritize MarkItDown's clean output
                    if suffix.lower() == ".pdf":
                        # Optional: merge formulas if needed, or just let MarkItDown handle it
                        pass
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass
            else:
                # Fallback to simple text decoding if MarkItDown is not available
                try:
                    raw = file_obj.read()
                    decoded = raw.decode("utf-8", errors="ignore")
                    form_values["text"] = decoded
                    if not form_values.get("source_name"):
                        form_values["source_name"] = filename
                except Exception as e:
                    error = f"MarkItDown not available and text decoding failed: {e}"

        try:
            output = run_pipeline(form_values)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        error=error,
        output=output,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)
