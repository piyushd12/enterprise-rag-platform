from unittest.mock import MagicMock, patch
from httpx import AsyncClient


async def _setup_workspace(client: AsyncClient, headers: dict) -> str:
    r = await client.post(
        "/workspaces",
        json={"name": "Search Test Workspace"},
        headers=headers
    )
    assert r.status_code == 201
    return r.json()["id"]


async def test_search_returns_results(
    client: AsyncClient,
    auth_headers: dict,
):
    workspace_id = await _setup_workspace(client, auth_headers)

    mock_results = [
        {
            "chunk_text": "Paris is the capital of France.",
            "chunk_id": "doc_0",
            "document_id": "doc-001",
            "page_num": 1,
            "chunk_index": 0,
            "score": 0.89,
        }
    ]

    with patch("app.routers.search.embedder") as mock_embedder, \
         patch("app.routers.search.vector_store") as mock_vs:

        mock_embedder.embed_query.return_value = [0.1] * 384
        mock_vs.search.return_value = mock_results

        response = await client.post(
            f"/workspaces/{workspace_id}/search",
            json={"query": "What is the capital of France?", "top_k": 5},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is the capital of France?"
    assert data["total_found"] == 1
    assert data["results"][0]["chunk_text"] == "Paris is the capital of France."
    assert data["results"][0]["score"] == 0.89


async def test_search_requires_auth(client: AsyncClient):
    response = await client.post(
        "/workspaces/some-id/search",
        json={"query": "some query", "top_k": 5},
    )
    assert response.status_code == 401


async def test_search_rejects_short_query(
    client: AsyncClient,
    auth_headers: dict,
):
    workspace_id = await _setup_workspace(client, auth_headers)

    response = await client.post(
        f"/workspaces/{workspace_id}/search",
        json={"query": "hi", "top_k": 5},
        headers=auth_headers,
    )
    assert response.status_code == 422


async def test_search_returns_empty_for_no_matches(
    client: AsyncClient,
    auth_headers: dict,
):
    workspace_id = await _setup_workspace(client, auth_headers)

    with patch("app.routers.search.embedder") as mock_embedder, \
         patch("app.routers.search.vector_store") as mock_vs:

        mock_embedder.embed_query.return_value = [0.1] * 384
        mock_vs.search.return_value = []

        response = await client.post(
            f"/workspaces/{workspace_id}/search",
            json={"query": "completely unrelated nonsense query", "top_k": 5},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert response.json()["total_found"] == 0
    assert response.json()["results"] == []


async def test_search_uses_same_workspace_scope(
    client: AsyncClient,
    auth_headers: dict,
):
    workspace_id = await _setup_workspace(client, auth_headers)

    with patch("app.routers.search.embedder") as mock_embedder, \
         patch("app.routers.search.vector_store") as mock_vs:

        mock_embedder.embed_query.return_value = [0.0] * 384
        mock_vs.search.return_value = []

        await client.post(
            f"/workspaces/{workspace_id}/search",
            json={"query": "test query here", "top_k": 3},
            headers=auth_headers,
        )

        mock_vs.search.assert_called_once()
        call_kwargs = mock_vs.search.call_args.kwargs
        assert call_kwargs["workspace_id"] == workspace_id
        assert call_kwargs["top_k"] == 3