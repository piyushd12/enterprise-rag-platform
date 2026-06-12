from httpx import AsyncClient


async def test_create_workspace(client: AsyncClient, auth_headers: dict):
    response = await client.post(
        "/workspaces",
        json={"name": "My Test Workspace", "description": "For testing"},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "My Test Workspace"
    assert data["slug"] == "my-test-workspace"
    assert "id" in data


async def test_create_workspace_requires_auth(client: AsyncClient):
    response = await client.post(
        "/workspaces",
        json={"name": "Unauthorized Workspace"},
        # no auth headers
    )
    assert response.status_code == 401


async def test_list_workspaces_shows_only_mine(client: AsyncClient, auth_headers: dict):
    # Create 2 workspaces
    await client.post("/workspaces", json={"name": "Workspace A"}, headers=auth_headers)
    await client.post("/workspaces", json={"name": "Workspace B"}, headers=auth_headers)

    response = await client.get("/workspaces", headers=auth_headers)
    assert response.status_code == 200
    workspaces = response.json()
    assert len(workspaces) >= 2
    names = [w["name"] for w in workspaces]
    assert "Workspace A" in names
    assert "Workspace B" in names


async def test_get_workspace_by_id(client: AsyncClient, auth_headers: dict):
    # Create a workspace
    create_response = await client.post(
        "/workspaces",
        json={"name": "Fetchable Workspace"},
        headers=auth_headers,
    )
    workspace_id = create_response.json()["id"]

    # Fetch it by ID
    response = await client.get(f"/workspaces/{workspace_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["id"] == workspace_id


async def test_get_nonexistent_workspace_returns_404(client: AsyncClient, auth_headers: dict):
    response = await client.get("/workspaces/nonexistent-id-12345", headers=auth_headers)
    assert response.status_code == 404