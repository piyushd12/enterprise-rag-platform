import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Enterprise RAG Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "token" not in st.session_state:
    st.session_state.token = None
if "workspace_id" not in st.session_state:
    st.session_state.workspace_id = None
if "workspace_name" not in st.session_state:
    st.session_state.workspace_name = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_get(path: str) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", headers=auth_headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, data: dict) -> dict | None:
    try:
        r = httpx.post(
            f"{API_BASE}{path}",
            json=data,
            headers=auth_headers(),
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def _load_conversation(conversation_id: str) -> None:
    messages_data = api_get(
        f"/workspaces/{st.session_state.workspace_id}"
        f"/conversations/{conversation_id}/messages"
    )

    if messages_data is None:
        st.error("Could not load conversation messages.")
        return

    restored_messages = []
    for msg in messages_data:
        restored_messages.append({
            "role": msg["role"],
            "content": msg["content"],
            "sources": msg.get("sources") or [],
        })

    st.session_state.conversation_id = conversation_id
    st.session_state.messages = restored_messages
    st.rerun()


with st.sidebar:
    st.title("🧠 RAG Platform")
    st.divider()

    if not st.session_state.token:
        st.subheader("🔐 Login")
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", use_container_width=True):
                try:
                    r = httpx.post(
                        f"{API_BASE}/auth/login",
                        json={"email": email, "password": password},
                        timeout=10,
                    )
                    if r.status_code == 200:
                        st.session_state.token = r.json()["access_token"]
                        st.success("Logged in!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                except Exception as e:
                    st.error(f"Could not reach API: {e}")

        with col2:
            if st.button("Register", use_container_width=True):
                if email and password:
                    try:
                        r = httpx.post(
                            f"{API_BASE}/auth/register",
                            json={
                                "email": email,
                                "password": password,
                                "full_name": email.split("@")[0],
                            },
                            timeout=10,
                        )
                        if r.status_code == 201:
                            st.success("Account created! Please log in.")
                        else:
                            st.error(r.json().get("detail", "Registration failed"))
                    except Exception as e:
                        st.error(f"Error: {e}")

    else:
        st.success(f"✅ Logged in")

        if st.button("Logout", use_container_width=True):
            for key in ["token", "workspace_id", "workspace_name",
                        "conversation_id", "messages"]:
                st.session_state[key] = None if key != "messages" else []
            st.rerun()

        st.divider()
        st.subheader("📁 Workspace")

        workspaces = api_get("/workspaces")
        if workspaces is not None:
            if not workspaces:
                new_name = st.text_input("Create workspace", placeholder="My Workspace")
                if st.button("Create", use_container_width=True) and new_name:
                    result = api_post("/workspaces", {"name": new_name})
                    if result:
                        st.session_state.workspace_id = result["id"]
                        st.session_state.workspace_name = result["name"]
                        st.rerun()
            else:
                workspace_names = {w["name"]: w["id"] for w in workspaces}
                selected_name = st.selectbox(
                    "Select workspace",
                    options=list(workspace_names.keys()),
                )
                selected_id = workspace_names[selected_name]

                if selected_id != st.session_state.workspace_id:
                    st.session_state.workspace_id = selected_id
                    st.session_state.workspace_name = selected_name
                    st.session_state.conversation_id = None
                    st.session_state.messages = []
                    st.rerun()

        if st.session_state.workspace_id:
            st.divider()
            st.subheader("📄 Upload Document")

            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "docx", "png", "jpg", "jpeg"],
                help="Supported: PDF, DOCX, PNG, JPEG",
            )

            if uploaded_file and st.button("Upload & Process", use_container_width=True):
                with st.spinner("Uploading..."):
                    try:
                        r = httpx.post(
                            f"{API_BASE}/workspaces/{st.session_state.workspace_id}/documents",
                            files={"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)},
                            headers=auth_headers(),
                            timeout=30,
                        )
                        if r.status_code == 202:
                            st.success(
                                f"✅ '{uploaded_file.name}' uploaded! "
                                f"Processing in background — wait ~30 seconds before chatting."
                            )
                        else:
                            st.error(f"Upload failed: {r.text}")
                    except Exception as e:
                        st.error(f"Upload error: {e}")

            if st.button("🔄 Refresh document status", use_container_width=True):
                st.rerun()

            docs = api_get(
                f"/workspaces/{st.session_state.workspace_id}/documents"
            )
            if docs:
                st.caption(f"Documents ({len(docs)} total):")
                for doc in docs[:5]:
                    status_emoji = {
                        "chunked": "✅",
                        "extracted": "📝",
                        "processing": "⏳",
                        "pending": "🕐",
                        "failed": "❌",
                    }.get(doc["status"], "❓")
                    st.caption(f"{status_emoji} {doc['title'][:30]} — {doc['status']}")

        if st.session_state.workspace_id:
            st.divider()

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("➕ New Chat", use_container_width=True):
                    st.session_state.conversation_id = None
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()

            st.markdown("**Previous Conversations**")

            conversations = api_get(
                f"/workspaces/{st.session_state.workspace_id}/conversations"
            )

            if conversations is None:
                st.caption("Could not load conversations.")
            elif not conversations:
                st.caption("No conversations yet. Ask your first question!")
            else:
                for convo in conversations[:15]: 
                    is_active = convo["id"] == st.session_state.conversation_id
                    title = convo.get("title") or "Untitled"
                    display_title = title[:35] + "..." if len(title) > 35 else title
                    label = f"{'▶ ' if is_active else ''}{display_title}"

                    if st.button(
                        label,
                        key=f"conv_{convo['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        if not is_active:
                            _load_conversation(convo["id"])


if not st.session_state.token:
    st.title("Welcome to the Enterprise RAG Platform")
    st.info("👈 Please log in using the sidebar to get started.")

elif not st.session_state.workspace_id:
    st.title("🧠 Enterprise RAG Platform")
    st.info("👈 Please select or create a workspace in the sidebar.")

else:
    st.title(f"💬 {st.session_state.workspace_name}")

    if st.session_state.conversation_id:
        st.caption(f"Conversation: `{st.session_state.conversation_id[:16]}...`")
    else:
        st.caption("New conversation — ask your first question below")

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} chunks used)"):
                    for i, source in enumerate(msg["sources"]):
                        st.markdown(
                            f"**Source {i + 1}** · Page {source['page_num']} "
                            f"· Score: `{source['score']:.2f}`"
                        )
                        st.text(source["chunk_text"][:300] + "...")
                        if i < len(msg["sources"]) - 1:
                            st.divider()

    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "sources": None,
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = api_post(
                    f"/workspaces/{st.session_state.workspace_id}/chat",
                    {
                        "query": prompt,
                        "conversation_id": st.session_state.conversation_id,
                    },
                )

            if result:
                st.session_state.conversation_id = result["conversation_id"]

                st.markdown(result["answer"])

                if result.get("sources"):
                    with st.expander(
                        f"📚 Sources ({result['chunks_used']} chunks used, "
                        f"{result['chunks_retrieved']} retrieved)"
                    ):
                        for i, source in enumerate(result["sources"]):
                            st.markdown(
                                f"**Source {i + 1}** · Page {source['page_num']} "
                                f"· Score: `{source['score']:.2f}`"
                            )
                            st.text(source["chunk_text"][:300] + "...")
                            if i < len(result["sources"]) - 1:
                                st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                })
            else:
                st.error("Failed to get a response. Check the backend logs.")