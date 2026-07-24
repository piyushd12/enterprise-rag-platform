from unittest.mock import MagicMock, patch
from httpx import AsyncClient


async def _create_workspace(client: AsyncClient, headers: dict, name: str = "Test WS") -> str:
    r = await client.post("/workspaces", json={"name": name}, headers=headers)
    assert r.status_code == 201
    return r.json()["id"]


async def test_upload_pdf_returns_202(
    client: AsyncClient,
    auth_headers: dict,
    sample_pdf_bytes: bytes,
):
    workspace_id = await _create_workspace(client, auth_headers)

    with patch("app.routers.documents.StorageService") as mock_storage_cls, \
         patch("app.routers.documents.process_document") as mock_task:

        mock_storage_cls.return_value.upload_file.return_value = "workspaces/test/doc.pdf"
        mock_task.delay.return_value = MagicMock()

        response = await client.post(
            f"/workspaces/{workspace_id}/documents",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            headers=auth_headers,
        )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "pending"
    assert data["file_name"] == "test.pdf"
    assert data["file_type"] == "application/pdf"
    assert data["workspace_id"] == workspace_id
    mock_task.delay.assert_called_once_with(data["id"])


async def test_upload_requires_auth(client: AsyncClient, sample_pdf_bytes: bytes):
    response = await client.post(
        "/workspaces/some-workspace-id/documents",
        files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 401


async def test_upload_rejects_unsupported_type(
    client: AsyncClient,
    auth_headers: dict,
):
    workspace_id = await _create_workspace(client, auth_headers, "Type Test WS")

    response = await client.post(
        f"/workspaces/{workspace_id}/documents",
        files={"file": ("archive.zip", b"fake zip content", "application/zip")},
        headers=auth_headers,
    )
    assert response.status_code == 415


async def test_upload_rejects_empty_file(
    client: AsyncClient,
    auth_headers: dict,
):
    workspace_id = await _create_workspace(client, auth_headers, "Empty File WS")

    with patch("app.routers.documents.StorageService"):
        response = await client.post(
            f"/workspaces/{workspace_id}/documents",
            files={"file": ("empty.pdf", b"", "application/pdf")},
            headers=auth_headers,
        )
    assert response.status_code == 400


async def test_list_documents(
    client: AsyncClient,
    auth_headers: dict,
    sample_pdf_bytes: bytes,
):
    workspace_id = await _create_workspace(client, auth_headers, "List Test WS")

    for i in range(2):
        with patch("app.routers.documents.StorageService") as m, \
             patch("app.routers.documents.process_document"):
            m.return_value.upload_file.return_value = f"test/doc{i}.pdf"
            await client.post(
                f"/workspaces/{workspace_id}/documents",
                files={"file": (f"doc{i}.pdf", sample_pdf_bytes, "application/pdf")},
                headers=auth_headers,
            )

    response = await client.get(
        f"/workspaces/{workspace_id}/documents",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert len(response.json()) == 2


async def test_get_document_status(
    client: AsyncClient,
    auth_headers: dict,
    sample_pdf_bytes: bytes,
):
    workspace_id = await _create_workspace(client, auth_headers, "Status Test WS")

    with patch("app.routers.documents.StorageService") as m, \
         patch("app.routers.documents.process_document"):
        m.return_value.upload_file.return_value = "test/status_doc.pdf"
        upload_response = await client.post(
            f"/workspaces/{workspace_id}/documents",
            files={"file": ("status.pdf", sample_pdf_bytes, "application/pdf")},
            headers=auth_headers,
        )
    document_id = upload_response.json()["id"]

    response = await client.get(
        f"/workspaces/{workspace_id}/documents/{document_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["id"] == document_id
    assert response.json()["status"] == "pending"


async def test_get_document_not_found(client: AsyncClient, auth_headers: dict):
    workspace_id = await _create_workspace(client, auth_headers, "Not Found WS")
    response = await client.get(
        f"/workspaces/{workspace_id}/documents/nonexistent-id",
        headers=auth_headers,
    )
    assert response.status_code == 404