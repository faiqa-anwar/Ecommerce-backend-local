"""Tests for the local-LLM (Ollama) integration: error mapping, tool fallback,
the customer-email hint, order status updates, and the web UI route."""
import httpx
import openai
import pytest

import app.llm as llm_module
from tests.conftest import auth_headers, register_and_login

pytestmark = pytest.mark.asyncio

_REQ = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")


async def _store_key(client, token) -> str:
    return (await client.get("/stores/me", headers=auth_headers(token))).json()["public_api_key"]


def _raise(exc):
    async def _create(*args, **kwargs):
        raise exc

    return _create


async def test_chat_returns_503_with_hint_when_ollama_not_running(client, monkeypatch):
    token = await register_and_login(client, "down@shop.com")
    key = await _store_key(client, token)
    monkeypatch.setattr(llm_module._client.chat.completions, "create", _raise(openai.APIConnectionError(request=_REQ)))

    r = await client.post("/chat", json={"message": "hello"}, headers={"X-Store-Key": key})
    assert r.status_code == 503
    assert "ollama" in r.json()["detail"].lower()


async def test_chat_returns_503_with_pull_hint_when_model_missing(client, monkeypatch):
    token = await register_and_login(client, "nomodel@shop.com")
    key = await _store_key(client, token)
    exc = openai.NotFoundError("model not found", response=httpx.Response(404, request=_REQ), body=None)
    monkeypatch.setattr(llm_module._client.chat.completions, "create", _raise(exc))

    r = await client.post("/chat", json={"message": "hello"}, headers={"X-Store-Key": key})
    assert r.status_code == 503
    assert "ollama pull" in r.json()["detail"]


async def test_chat_returns_503_on_timeout(client, monkeypatch):
    token = await register_and_login(client, "slow@shop.com")
    key = await _store_key(client, token)
    monkeypatch.setattr(llm_module._client.chat.completions, "create", _raise(openai.APITimeoutError(request=_REQ)))

    r = await client.post("/chat", json={"message": "hello"}, headers={"X-Store-Key": key})
    assert r.status_code == 503
    assert "too long" in r.json()["detail"]


async def test_model_without_tool_support_falls_back_to_plain_chat(client, monkeypatch):
    token = await register_and_login(client, "notools@shop.com")
    key = await _store_key(client, token)
    seen = []

    class _Msg:
        content = "Plain answer without tools."
        tool_calls = None

    class _Resp:
        choices = [type("C", (), {"message": _Msg()})()]

    async def _create(*args, **kwargs):
        seen.append("tools" in kwargs)
        if "tools" in kwargs:
            raise openai.BadRequestError(
                "registry.ollama.ai/library/tinymodel does not support tools",
                response=httpx.Response(400, request=_REQ),
                body=None,
            )
        return _Resp()

    monkeypatch.setattr(llm_module._client.chat.completions, "create", _create)
    r = await client.post("/chat", json={"message": "hi"}, headers={"X-Store-Key": key})
    assert r.status_code == 200
    assert r.json()["reply"] == "Plain answer without tools."
    assert seen == [True, False]  # tried with tools, retried without


async def test_unrelated_bad_request_is_not_swallowed(client, monkeypatch):
    token = await register_and_login(client, "badreq@shop.com")
    key = await _store_key(client, token)
    exc = openai.BadRequestError("context length exceeded", response=httpx.Response(400, request=_REQ), body=None)
    monkeypatch.setattr(llm_module._client.chat.completions, "create", _raise(exc))
    with pytest.raises(openai.BadRequestError):
        await client.post("/chat", json={"message": "hi"}, headers={"X-Store-Key": key})


async def test_customer_email_is_passed_to_the_model_as_context(client, monkeypatch):
    token = await register_and_login(client, "emailctx@shop.com")
    key = await _store_key(client, token)
    captured = {}

    class _Msg:
        content = "ok"
        tool_calls = None

    class _Resp:
        choices = [type("C", (), {"message": _Msg()})()]

    async def _create(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        return _Resp()

    monkeypatch.setattr(llm_module._client.chat.completions, "create", _create)
    await client.post(
        "/chat", json={"message": "where's my order?", "customer_email": "jane@example.com"}, headers={"X-Store-Key": key}
    )
    assert any("jane@example.com" in m["content"] for m in captured["messages"] if m["role"] == "system")


async def test_return_flow_end_to_end_via_chat(client, mock_llm):
    """Admin ships an order -> customer asks to return it in chat -> model calls
    initiate_return -> order is really updated in the database."""
    token = await register_and_login(client, "returnflow@shop.com")
    key = await _store_key(client, token)
    p = await client.post(
        "/products", json={"sku": "R1", "name": "Jacket", "price": 80, "stock_quantity": 4}, headers=auth_headers(token)
    )
    o = await client.post(
        "/orders",
        json={"customer_email": "kim@example.com", "items": [{"product_id": p.json()["id"], "quantity": 1}]},
        headers=auth_headers(token),
    )
    oid = o.json()["id"]
    await client.patch(f"/orders/{oid}", json={"status": "shipped", "tracking_number": "TRK9"}, headers=auth_headers(token))

    mock_llm.script(
        [
            {"tool": "initiate_return", "args": {"order_id": oid, "customer_email": "kim@example.com", "reason": "too small"}},
            {"text": "Your return has been started."},
        ]
    )
    r = await client.post(
        "/chat", json={"message": f"please return order {oid}, I'm kim@example.com"}, headers={"X-Store-Key": key}
    )
    assert r.json()["reply"] == "Your return has been started."

    got = await client.get(f"/orders/{oid}", headers=auth_headers(token))
    assert got.json()["status"] == "return_requested"
    assert got.json()["return_reason"] == "too small"


async def test_return_with_wrong_email_does_not_change_order(client, mock_llm):
    token = await register_and_login(client, "returnbad@shop.com")
    key = await _store_key(client, token)
    p = await client.post(
        "/products", json={"sku": "R2", "name": "Hat", "price": 10, "stock_quantity": 4}, headers=auth_headers(token)
    )
    o = await client.post(
        "/orders",
        json={"customer_email": "owner@example.com", "items": [{"product_id": p.json()["id"], "quantity": 1}]},
        headers=auth_headers(token),
    )
    oid = o.json()["id"]
    await client.patch(f"/orders/{oid}", json={"status": "delivered"}, headers=auth_headers(token))

    mock_llm.script(
        [
            {"tool": "initiate_return", "args": {"order_id": oid, "customer_email": "thief@evil.com", "reason": "x"}},
            {"text": "I couldn't find that order."},
        ]
    )
    await client.post("/chat", json={"message": f"return order {oid}"}, headers={"X-Store-Key": key})
    got = await client.get(f"/orders/{oid}", headers=auth_headers(token))
    assert got.json()["status"] == "delivered"


async def test_order_patch_sets_status_and_tracking(client):
    token = await register_and_login(client, "patch1@shop.com")
    p = await client.post("/products", json={"sku": "P", "name": "P", "price": 1, "stock_quantity": 1}, headers=auth_headers(token))
    o = await client.post(
        "/orders", json={"customer_email": "a@b.com", "items": [{"product_id": p.json()["id"], "quantity": 1}]}, headers=auth_headers(token)
    )
    r = await client.patch(
        f"/orders/{o.json()['id']}", json={"status": "shipped", "tracking_number": "ABC123"}, headers=auth_headers(token)
    )
    assert r.status_code == 200
    assert r.json()["status"] == "shipped"
    assert r.json()["tracking_number"] == "ABC123"


async def test_order_patch_rejects_invalid_status(client):
    token = await register_and_login(client, "patch2@shop.com")
    p = await client.post("/products", json={"sku": "P", "name": "P", "price": 1, "stock_quantity": 1}, headers=auth_headers(token))
    o = await client.post(
        "/orders", json={"customer_email": "a@b.com", "items": [{"product_id": p.json()["id"], "quantity": 1}]}, headers=auth_headers(token)
    )
    r = await client.patch(f"/orders/{o.json()['id']}", json={"status": "teleported"}, headers=auth_headers(token))
    assert r.status_code == 422


async def test_order_patch_other_store_or_missing_is_404(client):
    token_a = await register_and_login(client, "patchiso-a@shop.com")
    token_b = await register_and_login(client, "patchiso-b@shop.com")
    p = await client.post("/products", json={"sku": "P", "name": "P", "price": 1, "stock_quantity": 1}, headers=auth_headers(token_a))
    o = await client.post(
        "/orders", json={"customer_email": "a@b.com", "items": [{"product_id": p.json()["id"], "quantity": 1}]}, headers=auth_headers(token_a)
    )
    assert (await client.patch(f"/orders/{o.json()['id']}", json={"status": "shipped"}, headers=auth_headers(token_b))).status_code == 404
    assert (await client.patch("/orders/99999", json={"status": "shipped"}, headers=auth_headers(token_a))).status_code == 404


async def test_order_patch_requires_auth(client):
    r = await client.patch("/orders/1", json={"status": "shipped"})
    assert r.status_code == 401


async def test_web_ui_is_served_at_root(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Store Assistant" in r.text
