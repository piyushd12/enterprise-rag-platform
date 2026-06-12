import pytest
from httpx import AsyncClient


async def test_register_creates_user(client: AsyncClient):
    response = await client.post("/auth/register", json={
        "email": "alice@example.com",
        "password": "alicepassword",
        "full_name": "Alice Smith",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "alice@example.com"
    assert data["full_name"] == "Alice Smith"
    assert "hashed_password" not in data   # never expose password hash


async def test_register_duplicate_email_fails(client: AsyncClient):
    payload = {"email": "bob@example.com", "password": "bobpass123", "full_name": "Bob"}
    await client.post("/auth/register", json=payload)   # first registration
    response = await client.post("/auth/register", json=payload)   # duplicate
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


async def test_register_weak_password_fails(client: AsyncClient):
    response = await client.post("/auth/register", json={
        "email": "charlie@example.com",
        "password": "short",     # less than 8 characters
        "full_name": "Charlie",
    })
    assert response.status_code == 422   # Pydantic validation error


async def test_login_returns_token(client: AsyncClient):
    await client.post("/auth/register", json={
        "email": "dana@example.com",
        "password": "danapassword",
        "full_name": "Dana",
    })
    response = await client.post("/auth/login", json={
        "email": "dana@example.com",
        "password": "danapassword",
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert len(data["access_token"]) > 50 

async def test_login_wrong_password_fails(client: AsyncClient):
    await client.post("/auth/register", json={
        "email": "eve@example.com",
        "password": "correctpassword",
        "full_name": "Eve",
    })
    response = await client.post("/auth/login", json={
        "email": "eve@example.com",
        "password": "wrongpassword",
    })
    assert response.status_code == 401


async def test_get_me_with_valid_token(client: AsyncClient, auth_headers: dict):
    response = await client.get("/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["email"] == "testuser@example.com"


async def test_get_me_without_token_fails(client: AsyncClient):
    response = await client.get("/auth/me")
    assert response.status_code == 401   # no Authorization header at all