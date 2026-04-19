from datetime import datetime
import os
from pathlib import Path

import streamlit as st

from rag import RAGPipeline, load_settings


st.set_page_config(page_title="RAG Learning Assistant", layout="wide")
st.title("RAG Learning Assistant - De tai 9")
st.caption(
    "Hybrid Dense+Sparse retrieval, RRF fusion, rerank va confidence gate. "
    "Nguon bang chung duoc hien thi ben duoi moi cau tra loi."
)


@st.cache_resource(show_spinner="Dang khoi tao he thong RAG (co the mat 1-3 phut o lan dau)...")
def build_pipeline(provider: str) -> RAGPipeline:
    settings = load_settings(llm_provider=provider)
    return RAGPipeline(settings)


def _save_uploaded_files(uploaded_files: list, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for uploaded in uploaded_files:
        target = destination / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        saved_paths.append(target)

    return saved_paths


def _serialize_contexts(contexts) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for item in contexts:
        location_number = item.chunk.parent_page_number or item.chunk.page_number
        doc_type = (item.chunk.doc_type or "").strip().lower()
        location_label = "slide" if doc_type == "slide" else "page"
        serialized.append(
            {
                "source_name": item.chunk.source_name,
                "source_path": item.chunk.source_path,
                "page_number": item.chunk.page_number,
                "location_number": location_number,
                "location_label": location_label,
                "chunk_index": item.chunk.chunk_index,
                "content": item.chunk.parent_content or item.chunk.content,
                "child_content": item.chunk.content,
                "parent_title": item.chunk.parent_title,
                "section_path": item.chunk.section_path,
                "title": item.chunk.title,
                "subject": item.chunk.subject,
                "grade": item.chunk.grade,
                "chapter": item.chunk.chapter,
                "published_year": item.chunk.published_year,
                "source": item.chunk.source,
                "file_type": item.chunk.file_type,
                "doc_type": item.chunk.doc_type,
                "similarity": item.similarity,
                "lexical_score": item.lexical_score,
                "dense_score": item.dense_score,
                "sparse_score": item.sparse_score,
                "rrf_score": item.rrf_score,
                "rerank_score": item.rerank_score,
                "final_score": item.final_score,
            }
        )
    return serialized


def _render_context_block(contexts: list[dict[str, object]], expanded: bool = False) -> None:
    with st.expander("Nguon tham khao (Context)", expanded=expanded):
        if not contexts:
            st.info("No context found for this answer.")
            return

        for idx, ctx in enumerate(contexts, start=1):
            st.markdown(
                f"**[{idx}] {ctx['source_name']} | {ctx['location_label']} {ctx['location_number']} | "
                f"Type: {ctx.get('doc_type') or ctx.get('file_type') or 'document'} | "
                f"Final: {float(ctx['final_score']):.3f}**"
            )
            st.caption(
                " | ".join(
                    [
                        f"Mon: {ctx.get('subject') or '-'}",
                        f"Lop: {ctx.get('grade') or '-'}",
                        f"Chuong: {ctx.get('chapter') or '-'}",
                        f"Nam XB: {ctx.get('published_year') or '-'}",
                    ]
                )
            )
            parent_title = str(ctx.get("parent_title") or "").strip()
            if parent_title:
                st.caption(f"Parent section: {parent_title}")
            section_path = str(ctx.get("section_path") or "").strip()
            if section_path and section_path != parent_title:
                st.caption(f"Section path: {section_path}")
            st.write(str(ctx["content"]))
            child_content = " ".join(str(ctx.get("child_content") or "").split())
            if child_content and child_content != " ".join(str(ctx["content"]).split()):
                preview = child_content if len(child_content) <= 260 else f"{child_content[:260].rstrip()}..."
                st.caption(f"Matched child chunk: {preview}")
            st.caption(
                f"Dense: {float(ctx.get('dense_score', 0.0)):.3f} | "
                f"Sparse: {float(ctx.get('sparse_score', 0.0)):.3f} | "
                f"RRF: {float(ctx.get('rrf_score', 0.0)):.3f} | "
                f"Rerank: {float(ctx.get('rerank_score', 0.0)):.3f} | "
                f"Lexical: {float(ctx['lexical_score']):.3f} | "
                f"Final: {float(ctx['final_score']):.3f}"
            )


def _render_documents(pipeline: RAGPipeline) -> None:
    documents = pipeline.list_documents()
    if not documents:
        st.info("Chua co tai lieu nao duoc index.")
        return

    rows = [
        {
            "id": doc["id"],
            "file": doc["source_name"],
            "type": doc["file_type"],
            "doc_type": doc.get("doc_type", ""),
            "title": doc.get("title", ""),
            "subject": doc.get("subject", ""),
            "grade": doc.get("grade", ""),
            "chapter": doc.get("chapter", ""),
            "published_year": doc.get("published_year", ""),
            "source": doc.get("source", ""),
            "pages": doc["total_pages"],
            "chunks": doc["chunk_count"],
            "updated": doc["updated_at"],
        }
        for doc in documents
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


provider_options = ["ollama", "local", "openai", "gemini"]
default_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
if default_provider not in provider_options:
    default_provider = "ollama"
provider = st.sidebar.selectbox(
    "LLM provider",
    provider_options,
    index=provider_options.index(default_provider),
)
st.sidebar.caption(
    "Neu la lan dau, hay doi model embedding va model Ollama tai xong truoc khi chat."
)

try:
    pipeline = build_pipeline(provider)
except Exception as exc:
    st.error(
        "Khong the khoi tao RAG engine. "
        "Hay kiem tra terminal va thu chay lai app."
    )
    st.exception(exc)
    st.stop()

settings = pipeline.settings
st.sidebar.caption(f"Vector backend: {pipeline.vector_backend().upper()}")
st.sidebar.caption(f"LLM chinh: {settings.ollama_llm_model}")
if settings.ollama_fallback_model:
    st.sidebar.caption(f"LLM fallback: {settings.ollama_fallback_model}")
st.sidebar.caption(
    f"CTX: {settings.ollama_num_ctx} | Temp: {settings.ollama_temperature:.2f}"
)
st.sidebar.caption(
    f"Hybrid: {'ON' if settings.use_hybrid_retrieval else 'OFF'} | "
    f"Reranker: {'ON' if settings.use_reranker else 'OFF'}"
)
st.sidebar.caption(
    f"Dense/Sparse/Fusion/RerankK: {settings.top_k_dense}/{settings.top_k_sparse}/"
    f"{settings.fusion_top_k}/{settings.rerank_top_k}"
)
st.sidebar.caption(
    f"Evidence chunks: {settings.evidence_chunk_min}-{settings.evidence_chunk_max} | "
    f"Evidence tokens: {settings.evidence_min_tokens}-{settings.evidence_max_tokens}"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Indexing")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/TXT/MD files",
    type=["pdf", "txt", "md", "markdown", "log"],
    accept_multiple_files=True,
)

if st.sidebar.button("Embed uploaded files", width="stretch"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one file.")
    else:
        saved = _save_uploaded_files(uploaded_files, settings.documents_dir)
        with st.spinner("Embedding and indexing documents..."):
            summary = pipeline.ingest_files(saved)
        st.session_state["last_ingest_summary"] = summary

if st.sidebar.button("Index documents folder", width="stretch"):
    with st.spinner("Indexing from local documents folder..."):
        summary = pipeline.ingest_folder(settings.documents_dir)
    st.session_state["last_ingest_summary"] = summary

if "last_ingest_summary" in st.session_state:
    report = st.session_state["last_ingest_summary"]
    st.sidebar.success(
        "Indexed: "
        f"{report['indexed_files']} file(s), "
        f"{report['indexed_chunks']} child chunk(s), "
        f"{report.get('indexed_parent_chunks', 0)} parent chunk(s)."
    )
    if report.get("errors"):
        with st.sidebar.expander("Indexing errors", expanded=False):
            for err in report["errors"]:
                st.write(str(err))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Uu tien Ollama de chay local VRAM thap. OpenAI/Gemini la tuy chon mo rong."
)

st.subheader("Indexed documents")
_render_documents(pipeline)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {
            "role": "assistant",
            "content": (
                "Xin chao. Minh la tro ly hoc tap RAG. "
                "Hay dat cau hoi sau khi ban da index tai lieu noi bo."
            ),
            "contexts": [],
            "provider": "local",
            "confidence": 0.0,
            "intent": "general",
            "needs_clarification": False,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
    ]

for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_context_block(message.get("contexts", []), expanded=False)
            provider_used = message.get("provider", "local")
            confidence = float(message.get("confidence", 0.0))
            intent = str(message.get("intent", "general"))
            need_clarify = bool(message.get("needs_clarification", False))
            st.caption(
                f"Provider: {provider_used} | Intent: {intent} | Confidence: {confidence:.3f}"
                + (" | Clarification needed" if need_clarify else "")
            )

question = st.chat_input("Dat cau hoi ve tai lieu da nap vao he thong...")
if question:
    st.session_state["chat_history"].append(
        {
            "role": "user",
            "content": question,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Dang truy xuat vector va tao cau tra loi..."):
            result = pipeline.ask(question)

        st.markdown(result.answer)
        contexts = _serialize_contexts(result.contexts)
        _render_context_block(contexts, expanded=False)
        st.caption(
            f"Provider: {result.provider} | Intent: {result.intent} | "
            f"Confidence: {result.confidence:.3f}"
            + (" | Clarification needed" if result.needs_clarification else "")
        )

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": result.answer,
            "contexts": contexts,
            "provider": result.provider,
            "confidence": result.confidence,
            "intent": result.intent,
            "needs_clarification": result.needs_clarification,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
    )
