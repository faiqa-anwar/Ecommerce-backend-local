import pytest

from tests.conftest import auth_headers, register_and_login

pytestmark = pytest.mark.asyncio


async def test_register_creates_a_store_automatically(client):
    token = await register_and_login(client, "owner@shop.com")
    r = await client.get("/stores/me", headers=auth_headers(token))
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "owner@shop.com's store"
    assert len(body["public_api_key"]) > 10


async def test_duplicate_registration_is_rejected(client):
    await client.post("/auth/register", json={"email": "dupe@shop.com", "password": "SuperSecret123!"})
    r = await client.post("/auth/register", json={"email": "dupe@shop.com", "password": "SuperSecret123!"})
    assert r.status_code == 400


async def test_login_with_wrong_password_fails(client):
    await client.post("/auth/register", json={"email": "user@shop.com", "password": "SuperSecret123!"})
    r = await client.post(
        "/auth/jwt/login",
        data={"username": "user@shop.com", "password": "WrongPassword!"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 400


async def test_login_with_unknown_email_fails(client):
    r = await client.post(
        "/auth/jwt/login",
        data={"username": "nobody@nowhere.com", "password": "whatever123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 400


async def test_protected_route_requires_token(client):
    r = await client.get("/products")
    assert r.status_code == 401


async def test_protected_route_rejects_garbage_token(client):
    r = await client.get("/products", headers=auth_headers("this.is.not.a.valid.jwt"))
    assert r.status_code == 401


async def test_weak_password_is_rejected_by_validation(client):
    # fastapi-users enforces a minimum length by default via pydantic schema.
    r = await client.post("/auth/register", json={"email": "weak@shop.com", "password": "a"})
    assert r.status_code in (400, 422)


async def test_malformed_email_is_rejected(client):
    r = await client.post("/auth/register", json={"email": "not-an-email", "password": "SuperSecret123!"})
    assert r.status_code == 422
