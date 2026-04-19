from datetime import datetime
from pathlib import Path

import streamlit as st

from rag import RAGPipeline, load_settings


st.set_page_config(page_title="RAG Learning Assistant", layout="wide")
st.title("RAG Learning Assistant - De tai 9")
st.caption(
    "ANN + Cosine Retrieval with source tracing. "
    "Top-3 context chunks are shown below each answer."
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
        serialized.append(
            {
                "source_name": item.chunk.source_name,
                "source_path": item.chunk.source_path,
                "page_number": item.chunk.page_number,
                "chunk_index": item.chunk.chunk_index,
                "content": item.chunk.content,
                "similarity": item.similarity,
                "lexical_score": item.lexical_score,
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
                f"**[{idx}] {ctx['source_name']} | page {ctx['page_number']} | "
                f"Similarity: {float(ctx['similarity']):.3f}**"
            )
            st.write(str(ctx["content"]))
            st.caption(
                f"Lexical: {float(ctx['lexical_score']):.3f} | "
                f"Final score: {float(ctx['final_score']):.3f}"
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
            "pages": doc["total_pages"],
            "chunks": doc["chunk_count"],
            "updated": doc["updated_at"],
        }
        for doc in documents
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


provider = st.sidebar.selectbox("LLM provider", ["local", "openai", "gemini"], index=0)
st.sidebar.caption("Neu chua thay o chat, vui long doi he thong tai model embedding xong.")

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

st.sidebar.markdown("---")
st.sidebar.subheader("Indexing")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/TXT/MD files",
    type=["pdf", "txt", "md", "markdown", "log"],
    accept_multiple_files=True,
)

if st.sidebar.button("Embed uploaded files", use_container_width=True):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one file.")
    else:
        saved = _save_uploaded_files(uploaded_files, settings.documents_dir)
        with st.spinner("Embedding and indexing documents..."):
            summary = pipeline.ingest_files(saved)
        st.session_state["last_ingest_summary"] = summary

if st.sidebar.button("Index documents folder", use_container_width=True):
    with st.spinner("Indexing from local documents folder..."):
        summary = pipeline.ingest_folder(settings.documents_dir)
    st.session_state["last_ingest_summary"] = summary

if "last_ingest_summary" in st.session_state:
    report = st.session_state["last_ingest_summary"]
    st.sidebar.success(
        "Indexed: "
        f"{report['indexed_files']} file(s), "
        f"{report['indexed_chunks']} chunk(s)."
    )
    if report.get("errors"):
        with st.sidebar.expander("Indexing errors", expanded=False):
            for err in report["errors"]:
                st.write(str(err))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Set OPENAI_API_KEY or GEMINI_API_KEY in .env to enable external LLM grounding."
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
            "time": datetime.now().isoformat(timespec="seconds"),
        }
    ]

for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_context_block(message.get("contexts", []), expanded=False)
            provider_used = message.get("provider", "local")
            st.caption(f"Provider: {provider_used}")

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
        st.caption(f"Provider: {result.provider}")

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": result.answer,
            "contexts": contexts,
            "provider": result.provider,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
    )
