import pytest

from tests.conftest import auth_headers, register_and_login

pytestmark = pytest.mark.asyncio


async def _create_product(client, token, sku="SKU-1", **overrides):
    payload = {"sku": sku, "name": "Blue Widget", "description": "A widget.", "price": 9.99, "stock_quantity": 10}
    payload.update(overrides)
    return await client.post("/products", json=payload, headers=auth_headers(token))


async def test_create_and_list_product(client):
    token = await register_and_login(client, "a@shop.com")
    r = await _create_product(client, token)
    assert r.status_code == 201
    assert r.json()["sku"] == "SKU-1"

    r = await client.get("/products", headers=auth_headers(token))
    assert r.status_code == 200
    assert len(r.json()) == 1


async def test_duplicate_sku_within_same_store_rejected(client):
    token = await register_and_login(client, "b@shop.com")
    await _create_product(client, token, sku="DUP")
    r = await _create_product(client, token, sku="DUP")
    assert r.status_code == 400


async def test_same_sku_allowed_across_different_stores(client):
    token_a = await register_and_login(client, "storea@shop.com")
    token_b = await register_and_login(client, "storeb@shop.com")
    r1 = await _create_product(client, token_a, sku="SHARED")
    r2 = await _create_product(client, token_b, sku="SHARED")
    assert r1.status_code == 201
    assert r2.status_code == 201


async def test_store_cannot_see_another_stores_products(client):
    token_a = await register_and_login(client, "iso-a@shop.com")
    token_b = await register_and_login(client, "iso-b@shop.com")
    created = await _create_product(client, token_a, sku="ISO-1")
    product_id = created.json()["id"]

    r = await client.get(f"/products/{product_id}", headers=auth_headers(token_b))
    assert r.status_code == 404


async def test_negative_price_rejected(client):
    token = await register_and_login(client, "neg@shop.com")
    r = await _create_product(client, token, price=-5)
    assert r.status_code == 422


async def test_negative_stock_rejected(client):
    token = await register_and_login(client, "negstock@shop.com")
    r = await _create_product(client, token, stock_quantity=-1)
    assert r.status_code == 422


async def test_update_stock(client):
    token = await register_and_login(client, "stock@shop.com")
    created = await _create_product(client, token)
    product_id = created.json()["id"]

    r = await client.patch(
        f"/products/{product_id}/stock", json={"stock_quantity": 42}, headers=auth_headers(token)
    )
    assert r.status_code == 200
    assert r.json()["stock_quantity"] == 42


async def test_get_nonexistent_product_404(client):
    token = await register_and_login(client, "missing@shop.com")
    r = await client.get("/products/999999", headers=auth_headers(token))
    assert r.status_code == 404
