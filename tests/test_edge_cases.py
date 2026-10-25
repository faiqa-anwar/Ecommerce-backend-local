import pytest

from tests.conftest import auth_headers, register_and_login

pytestmark = pytest.mark.asyncio


async def test_health_check(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_infinite_tool_call_loop_is_capped(client, mock_llm):
    """If the model just keeps requesting tools forever, the request must
    still return in bounded time with a graceful message, not hang."""
    token = await register_and_login(client, "loop@shop.com")
    r = await client.get("/stores/me", headers=auth_headers(token))
    key = r.json()["public_api_key"]

    # Script more tool-call steps than LLM_MAX_TOOL_ITERATIONS (default 5).
    mock_llm.script([{"tool": "check_inventory", "args": {"sku": "X"}, "id": f"call_{i}"} for i in range(10)])

    r = await client.post("/chat", json={"message": "loop please"}, headers={"X-Store-Key": key})
    assert r.status_code == 200
    assert "couldn't finish" in r.json()["reply"] or "try again" in r.json()["reply"]


async def test_tool_call_with_malformed_json_arguments_does_not_crash(client, mock_llm, monkeypatch):
    """Simulate the LLM returning syntactically invalid JSON in a tool call
    argument string -- app/llm.py must catch this, not 500."""
    token = await register_and_login(client, "badjson@shop.com")
    r = await client.get("/stores/me", headers=auth_headers(token))
    key = r.json()["public_api_key"]

    import json as _json

    import app.llm as llm_module

    class _BadFunction:
        name = "check_inventory"
        arguments = "{not valid json"

    class _BadToolCall:
        id = "call_bad"
        function = _BadFunction()

    class _BadMessage:
        content = ""
        tool_calls = [_BadToolCall()]

    class _FinalMessage:
        content = "Sorry, something went wrong looking that up."
        tool_calls = None

    class _Choice:
        def __init__(self, message):
            self.message = message

    class _Response:
        def __init__(self, message):
            self.choices = [_Choice(message)]

    calls = {"n": 0}

    async def _fake_create(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(_BadMessage())
        return _Response(_FinalMessage())

    monkeypatch.setattr(llm_module._client.chat.completions, "create", _fake_create)

    r = await client.post("/chat", json={"message": "check stock"}, headers={"X-Store-Key": key})
    assert r.status_code == 200
    assert r.json()["reply"] == "Sorry, something went wrong looking that up."


async def test_extremely_long_message_rejected(client, mock_llm):
    token = await register_and_login(client, "toolong@shop.com")
    r = await client.get("/stores/me", headers=auth_headers(token))
    key = r.json()["public_api_key"]

    r = await client.post("/chat", json={"message": "x" * 5000}, headers={"X-Store-Key": key})
    assert r.status_code == 422


async def test_products_missing_required_field_returns_422(client):
    token = await register_and_login(client, "missingfield@shop.com")
    r = await client.post("/products", json={"name": "No SKU or price"}, headers=auth_headers(token))
    assert r.status_code == 422
    assert "detail" in r.json()


async def test_wrong_content_type_json_body_handled(client):
    token = await register_and_login(client, "badcontent@shop.com")
    r = await client.post(
        "/products",
        content="not json at all",
        headers={**auth_headers(token), "Content-Type": "application/json"},
    )
    assert r.status_code == 422


async def test_nonexistent_route_returns_404(client):
    r = await client.get("/this-route-does-not-exist")
    assert r.status_code == 404
