import pytest

from tests.conftest import auth_headers, register_and_login

pytestmark = pytest.mark.asyncio


async def test_create_order_computes_total(client):
    token = await register_and_login(client, "orders1@shop.com")
    p1 = await client.post(
        "/products", json={"sku": "A", "name": "Item A", "price": 10.0, "stock_quantity": 5},
        headers=auth_headers(token),
    )
    p2 = await client.post(
        "/products", json={"sku": "B", "name": "Item B", "price": 5.5, "stock_quantity": 5},
        headers=auth_headers(token),
    )
    order_payload = {
        "customer_email": "buyer@example.com",
        "customer_name": "Buyer",
        "items": [
            {"product_id": p1.json()["id"], "quantity": 2},
            {"product_id": p2.json()["id"], "quantity": 1},
        ],
    }
    r = await client.post("/orders", json=order_payload, headers=auth_headers(token))
    assert r.status_code == 201
    assert r.json()["total_amount"] == 25.5  # 2*10 + 1*5.5


async def test_create_order_with_unknown_product_rejected(client):
    token = await register_and_login(client, "orders2@shop.com")
    r = await client.post(
        "/orders",
        json={"customer_email": "b@e.com", "items": [{"product_id": 999999, "quantity": 1}]},
        headers=auth_headers(token),
    )
    assert r.status_code == 400


async def test_create_order_with_no_items_rejected(client):
    token = await register_and_login(client, "orders3@shop.com")
    r = await client.post(
        "/orders", json={"customer_email": "b@e.com", "items": []}, headers=auth_headers(token)
    )
    assert r.status_code == 422


async def test_create_order_cannot_use_another_stores_product(client):
    token_a = await register_and_login(client, "ordersiso-a@shop.com")
    token_b = await register_and_login(client, "ordersiso-b@shop.com")
    p = await client.post(
        "/products", json={"sku": "X", "name": "X", "price": 1, "stock_quantity": 1},
        headers=auth_headers(token_a),
    )
    r = await client.post(
        "/orders",
        json={"customer_email": "b@e.com", "items": [{"product_id": p.json()["id"], "quantity": 1}]},
        headers=auth_headers(token_b),
    )
    assert r.status_code == 400


async def test_list_orders_scoped_to_store(client):
    token_a = await register_and_login(client, "orderslist-a@shop.com")
    token_b = await register_and_login(client, "orderslist-b@shop.com")
    p = await client.post(
        "/products", json={"sku": "Y", "name": "Y", "price": 1, "stock_quantity": 1},
        headers=auth_headers(token_a),
    )
    await client.post(
        "/orders",
        json={"customer_email": "b@e.com", "items": [{"product_id": p.json()["id"], "quantity": 1}]},
        headers=auth_headers(token_a),
    )
    r = await client.get("/orders", headers=auth_headers(token_b))
    assert r.json() == []
