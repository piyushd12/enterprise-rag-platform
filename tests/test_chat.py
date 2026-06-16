from unittest.mock import AsyncMock, patch
from httpx import AsyncClient


async def _create_workspace_and_login(
    client: AsyncClient,
    email: str = "chatuser@example.com",
) -> tuple[dict, str]:
    """Register, login, and create workspace. Returns (headers, workspace_id)."""
    await client.post("/auth/register", json={
        "email": email, "password": "password123", "full_name": "Chat User"
    })
    login = await client.post("/auth/login", json={
        "email": email, "password": "password123"
    })
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    ws = await client.post("/workspaces", json={"name": "Chat WS"}, headers=headers)
    return headers, ws.json()["id"]


async def test_chat_creates_new_conversation(client: AsyncClient):
    headers, workspace_id = await _create_workspace_and_login(client)

    mock_result = {
        "answer": "This document is about AI.",
        "sources": [{"document_id": "doc-1", "chunk_text": "AI content", "page_num": 1, "score": 0.9, "chunk_index": 0}],
        "tokens_used": 150,
        "chunks_retrieved": 3,
        "chunks_used": 1,
    }

    with patch("app.routers.chat.run_rag_pipeline", new_callable=AsyncMock) as mock_pipeline, \
         patch("app.routers.chat.workspace_has_chunked_documents", new_callable=AsyncMock) as mock_check:

        mock_check.return_value = True
        mock_pipeline.return_value = mock_result

        response = await client.post(
            f"/workspaces/{workspace_id}/chat",
            json={"query": "What is this about?", "conversation_id": None},
            headers=headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert "message_id" in data
    assert data["answer"] == "This document is about AI."
    assert len(data["sources"]) == 1
    assert data["chunks_retrieved"] == 3
    assert data["chunks_used"] == 1


async def test_chat_requires_auth(client: AsyncClient):
    response = await client.post(
        "/workspaces/some-id/chat",
        json={"query": "Hello?"},
    )
    assert response.status_code == 40140


async def test_chat_continues_existing_conversation(client: AsyncClient):
    headers, workspace_id = await _create_workspace_and_login(
        client, "chatcontinue@example.com"
    )

    mock_result = {
        "answer": "Answer.", "sources": [],
        "tokens_used": 50, "chunks_retrieved": 0, "chunks_used": 0,
    }

    with patch("app.routers.chat.run_rag_pipeline", new_callable=AsyncMock) as mock, \
         patch("app.routers.chat.workspace_has_chunked_documents", new_callable=AsyncMock) as mock_check:

        mock_check.return_value = True
        mock.return_value = mock_result

        r1 = await client.post(
            f"/workspaces/{workspace_id}/chat",
            json={"query": "First question", "conversation_id": None},
            headers=headers,
        )
        assert r1.status_code == 200
        conv_id = r1.json()["conversation_id"]

        r2 = await client.post(
            f"/workspaces/{workspace_id}/chat",
            json={"query": "Follow-up question", "conversation_id": conv_id},
            headers=headers,
        )
        assert r2.status_code == 200
        assert r2.json()["conversation_id"] == conv_id


async def test_chat_returns_422_when_no_documents(client: AsyncClient):
    headers, workspace_id = await _create_workspace_and_login(
        client, "nodocs@example.com"
    )

    with patch("app.routers.chat.workspace_has_chunked_documents", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = False

        response = await client.post(
            f"/workspaces/{workspace_id}/chat",
            json={"query": "What does my document say?"},
            headers=headers,
        )

    assert response.status_code == 422
    assert "No indexed documents" in response.json()["detail"]


async def test_list_conversations(client: AsyncClient):
    headers, workspace_id = await _create_workspace_and_login(
        client, "listconv@example.com"
    )

    mock_result = {
        "answer": "A.", "sources": [],
        "tokens_used": 10, "chunks_retrieved": 0, "chunks_used": 0,
    }

    with patch("app.routers.chat.run_rag_pipeline", new_callable=AsyncMock) as mock, \
         patch("app.routers.chat.workspace_has_chunked_documents", new_callable=AsyncMock) as mock_check:

        mock_check.return_value = True
        mock.return_value = mock_result

        await client.post(
            f"/workspaces/{workspace_id}/chat",
            json={"query": "Question 1", "conversation_id": None},
            headers=headers,
        )
        await client.post(
            f"/workspaces/{workspace_id}/chat",
            json={"query": "Question 2", "conversation_id": None},
            headers=headers,
        )

    response = await client.get(
        f"/workspaces/{workspace_id}/conversations",
        headers=headers,
    )
    assert response.status_code == 200
    assert len(response.json()) == 2