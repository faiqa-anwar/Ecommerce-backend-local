import pytest

from tests.conftest import auth_headers, register_and_login

pytestmark = pytest.mark.asyncio


async def test_add_document_chunks_and_lists_it(client):
    token = await register_and_login(client, "kb@shop.com")
    r = await client.post(
        "/knowledge-base/documents",
        json={"title": "Return Policy", "content": "You can return items within 30 days of purchase."},
        headers=auth_headers(token),
    )
    assert r.status_code == 201
    assert r.json()["chunks_created"] >= 1

    r = await client.get("/knowledge-base", headers=auth_headers(token))
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.json()[0]["title"] == "Return Policy"


async def test_long_document_is_split_into_multiple_chunks(client):
    token = await register_and_login(client, "kblong@shop.com")
    long_text = "This is a sentence about our shipping policy. " * 60  # well over the 500-char chunk size
    r = await client.post(
        "/knowledge-base/documents",
        json={"title": "Shipping Policy", "content": long_text},
        headers=auth_headers(token),
    )
    assert r.status_code == 201
    assert r.json()["chunks_created"] > 1


async def test_empty_document_content_rejected(client):
    token = await register_and_login(client, "kbempty@shop.com")
    r = await client.post(
        "/knowledge-base/documents",
        json={"title": "Empty", "content": "   "},
        headers=auth_headers(token),
    )
    # Pydantic min_length=1 catches an empty string, but a whitespace-only
    # string still passes schema validation -- make sure it doesn't create
    # a bogus empty chunk downstream.
    assert r.status_code == 201
    assert r.json()["chunks_created"] == 0


async def test_sync_products_indexes_catalog(client):
    token = await register_and_login(client, "kbsync@shop.com")
    await client.post(
        "/products",
        json={"sku": "P1", "name": "Noise Cancelling Headphones", "description": "Great for travel.", "price": 199.99, "stock_quantity": 5},
        headers=auth_headers(token),
    )
    r = await client.post("/knowledge-base/sync-products", headers=auth_headers(token))
    assert r.status_code == 200
    assert r.json()["products_indexed"] == 1

    r = await client.get("/knowledge-base", headers=auth_headers(token))
    assert any(c["source_type"] == "product" for c in r.json())


async def test_sync_products_replaces_stale_entries_not_duplicates(client):
    token = await register_and_login(client, "kbresync@shop.com")
    await client.post(
        "/products",
        json={"sku": "P1", "name": "Widget", "description": "d", "price": 1, "stock_quantity": 1},
        headers=auth_headers(token),
    )
    await client.post("/knowledge-base/sync-products", headers=auth_headers(token))
    await client.post("/knowledge-base/sync-products", headers=auth_headers(token))  # sync twice

    r = await client.get("/knowledge-base", headers=auth_headers(token))
    product_chunks = [c for c in r.json() if c["source_type"] == "product"]
    assert len(product_chunks) == 1  # not 2


async def test_search_finds_relevant_chunk(client):
    token = await register_and_login(client, "kbsearch@shop.com")
    await client.post(
        "/knowledge-base/documents",
        json={"title": "Warranty", "content": "All electronics come with a 1 year warranty."},
        headers=auth_headers(token),
    )
    r = await client.post(
        "/knowledge-base/search", json={"query": "warranty", "top_k": 3}, headers=auth_headers(token)
    )
    assert r.status_code == 200
    assert len(r.json()) >= 1


async def test_search_on_empty_knowledge_base_returns_empty_list(client):
    token = await register_and_login(client, "kbnoresults@shop.com")
    r = await client.post(
        "/knowledge-base/search", json={"query": "anything"}, headers=auth_headers(token)
    )
    assert r.status_code == 200
    assert r.json() == []


async def test_delete_chunk(client):
    token = await register_and_login(client, "kbdelete@shop.com")
    created = await client.post(
        "/knowledge-base/documents",
        json={"title": "Temp", "content": "Temporary note."},
        headers=auth_headers(token),
    )
    r = await client.get("/knowledge-base", headers=auth_headers(token))
    chunk_id = r.json()[0]["id"]

    r = await client.delete(f"/knowledge-base/{chunk_id}", headers=auth_headers(token))
    assert r.status_code == 204

    r = await client.get("/knowledge-base", headers=auth_headers(token))
    assert r.json() == []


async def test_cannot_delete_another_stores_chunk(client):
    token_a = await register_and_login(client, "kbiso-a@shop.com")
    token_b = await register_and_login(client, "kbiso-b@shop.com")
    await client.post(
        "/knowledge-base/documents", json={"title": "A doc", "content": "secret"}, headers=auth_headers(token_a)
    )
    r = await client.get("/knowledge-base", headers=auth_headers(token_a))
    chunk_id = r.json()[0]["id"]

    r = await client.delete(f"/knowledge-base/{chunk_id}", headers=auth_headers(token_b))
    assert r.status_code == 404
