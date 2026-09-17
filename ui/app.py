"""
Streamlit UI for the RAG Application.

Communicates with the FastAPI backend over HTTP — no direct imports
of backend code. Provides document upload/ingestion and chat interface.
"""

import os

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Defaults to localhost for running the UI directly on the host; the
# Docker Compose stack overrides this to the "api" service's container
# name, since "localhost" inside the UI container would otherwise point
# at itself rather than the API container.
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 120.0  # seconds — LLM calls can be slow on free tier


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Application",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Source cards */
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #4a90d9;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85em;
    }
    .source-card .source-header {
        font-weight: 600;
        color: #4a90d9;
        margin-bottom: 4px;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .status-healthy { background: #d4edda; color: #155724; }
    .status-degraded { background: #fff3cd; color: #856404; }

    /* Metadata pills */
    .meta-pill {
        display: inline-block;
        background: #e9ecef;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75em;
        margin-right: 6px;
        color: #495057;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — Document Ingestion & Health
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📁 Document Management")

    # Health check
    st.subheader("System Status")
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            status_class = "status-healthy" if data["status"] == "healthy" else "status-degraded"
            st.markdown(
                f'<span class="status-badge {status_class}">{data["status"].upper()}</span>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            col1.metric("Qdrant", "✅" if data["qdrant"] else "❌")
            col2.metric("Redis", "✅" if data["redis"] else "❌")
        else:
            st.error(f"API returned {resp.status_code}")
    except httpx.ConnectError:
        st.error("❌ Cannot connect to API at " + API_BASE)
    except Exception as e:
        st.error(f"Health check failed: {e}")

    st.divider()

    # File upload
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md"],
        help="Supported formats: PDF, TXT, Markdown",
    )

    if uploaded_file is not None:
        if st.button("🚀 Ingest Document", use_container_width=True):
            with st.spinner("Ingesting document..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    }
                    resp = httpx.post(
                        f"{API_BASE}/ingest",
                        files=files,
                        timeout=TIMEOUT,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(
                            f"✅ Ingested **{result['filename']}**\n\n"
                            f"- Chunks: {result['chunk_count']}\n"
                            f"- Type: {result['doc_type']}\n"
                            f"- Source ID: `{result['source_id'][:12]}...`"
                        )
                    else:
                        error = resp.json().get("detail", resp.text)
                        st.error(f"Ingestion failed: {error}")
                except httpx.ConnectError:
                    st.error("Cannot connect to API. Is the backend running?")
                except Exception as e:
                    st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Main area — Chat Interface
# ---------------------------------------------------------------------------

st.title("🔍 RAG Chat")
st.caption("Ask questions about your ingested documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-header">{src["filename"] or src["source_id"][:12]} '
                        f'(chunk {src["chunk_index"]}, score: {src["score"]:.3f})</div>'
                        f'{src["content"][:300]}...'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Show metadata pills
            if message.get("metadata"):
                meta = message["metadata"]
                pills = []
                pills.append(f'<span class="meta-pill">🤖 {meta.get("provider", "")}</span>')
                pills.append(f'<span class="meta-pill">⏱️ {meta.get("latency_ms", 0):.0f}ms</span>')
                if meta.get("cached"):
                    pills.append('<span class="meta-pill">💾 cached</span>')
                pills.append(
                    f'<span class="meta-pill">🔑 {meta.get("request_id", "")[:8]}...</span>'
                )
                st.markdown(" ".join(pills), unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/chat",
                    json={"query": prompt, "top_k": 5},
                    timeout=TIMEOUT,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    st.markdown(answer)

                    # Show sources
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander("📄 View Sources", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<div class="source-header">'
                                    f'{src["filename"] or src["source_id"][:12]} '
                                    f'(chunk {src["chunk_index"]}, score: {src["score"]:.3f})'
                                    f'</div>'
                                    f'{src["content"][:300]}...'
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    # Metadata pills
                    meta = {
                        "provider": data.get("llm_provider", ""),
                        "model": data.get("llm_model", ""),
                        "latency_ms": data.get("latency_ms", 0),
                        "cached": data.get("cached", False),
                        "request_id": data.get("request_id", ""),
                    }
                    pills = []
                    pills.append(f'<span class="meta-pill">🤖 {meta["provider"]}</span>')
                    pills.append(f'<span class="meta-pill">⏱️ {meta["latency_ms"]:.0f}ms</span>')
                    if meta["cached"]:
                        pills.append('<span class="meta-pill">💾 cached</span>')
                    pills.append(
                        f'<span class="meta-pill">🔑 {meta["request_id"][:8]}...</span>'
                    )
                    st.markdown(" ".join(pills), unsafe_allow_html=True)

                    # Save to chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "metadata": meta,
                        }
                    )
                else:
                    error = resp.json().get("detail", resp.text)
                    st.error(f"Error: {error}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"❌ Error: {error}"}
                    )

            except httpx.ConnectError:
                msg = "❌ Cannot connect to API. Make sure the backend is running on port 8000."
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            except Exception as e:
                msg = f"❌ Unexpected error: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
