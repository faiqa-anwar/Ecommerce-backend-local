import pytest

from tests.conftest import auth_headers, register_and_login

pytestmark = pytest.mark.asyncio


async def _get_store_key(client, token) -> str:
    r = await client.get("/stores/me", headers=auth_headers(token))
    return r.json()["public_api_key"]


async def test_chat_without_store_key_rejected(client, mock_llm):
    r = await client.post("/chat", json={"message": "hi"})
    assert r.status_code in (401, 422)  # 422 if header missing entirely, 401 if invalid


async def test_chat_with_invalid_store_key_rejected(client, mock_llm):
    r = await client.post("/chat", json={"message": "hi"}, headers={"X-Store-Key": "bogus"})
    assert r.status_code == 401


async def test_chat_creates_conversation_and_persists_messages(client, mock_llm):
    token = await register_and_login(client, "chat1@shop.com")
    key = await _get_store_key(client, token)
    mock_llm.script([{"text": "Hello! How can I help you today?"}])

    r = await client.post("/chat", json={"message": "Hi there"}, headers={"X-Store-Key": key})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "Hello! How can I help you today?"
    conversation_id = body["conversation_id"]

    r = await client.get(f"/conversations/{conversation_id}/messages", headers=auth_headers(token))
    roles = [m["role"] for m in r.json()]
    assert roles == ["user", "assistant"]


async def test_chat_continues_existing_conversation(client, mock_llm):
    token = await register_and_login(client, "chat2@shop.com")
    key = await _get_store_key(client, token)

    mock_llm.script([{"text": "Sure, what's your order number?"}])
    r1 = await client.post("/chat", json={"message": "I want to return something"}, headers={"X-Store-Key": key})
    conv_id = r1.json()["conversation_id"]

    mock_llm.script([{"text": "Got it, looking into order 123."}])
    r2 = await client.post(
        "/chat",
        json={"message": "It's order 123", "conversation_id": conv_id},
        headers={"X-Store-Key": key},
    )
    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == conv_id

    r = await client.get(f"/conversations/{conv_id}/messages", headers=auth_headers(token))
    assert len(r.json()) == 4  # 2 user + 2 assistant


async def test_chat_with_conversation_id_from_another_store_rejected(client, mock_llm):
    token_a = await register_and_login(client, "chatiso-a@shop.com")
    key_a = await _get_store_key(client, token_a)
    token_b = await register_and_login(client, "chatiso-b@shop.com")
    key_b = await _get_store_key(client, token_b)

    mock_llm.script([{"text": "hi"}])
    r1 = await client.post("/chat", json={"message": "hello"}, headers={"X-Store-Key": key_a})
    conv_id = r1.json()["conversation_id"]

    r2 = await client.post(
        "/chat", json={"message": "hijack attempt", "conversation_id": conv_id}, headers={"X-Store-Key": key_b}
    )
    assert r2.status_code == 404


async def test_chat_blank_message_rejected(client, mock_llm):
    token = await register_and_login(client, "chatblank@shop.com")
    key = await _get_store_key(client, token)
    r = await client.post("/chat", json={"message": "   "}, headers={"X-Store-Key": key})
    assert r.status_code == 422


async def test_chat_uses_tool_call_end_to_end(client, mock_llm):
    """Full loop: LLM asks for a tool, gets a real result from the real DB,
    then answers using it."""
    token = await register_and_login(client, "chattool@shop.com")
    key = await _get_store_key(client, token)
    await client.post(
        "/products",
        json={"sku": "TOOL-1", "name": "Test Widget", "price": 12.0, "stock_quantity": 3},
        headers=auth_headers(token),
    )

    mock_llm.script(
        [
            {"tool": "check_inventory", "args": {"sku": "TOOL-1"}, "id": "call_1"},
            {"text": "We have 3 of those in stock."},
        ]
    )
    r = await client.post(
        "/chat", json={"message": "how many Test Widgets do you have?"}, headers={"X-Store-Key": key}
    )
    assert r.status_code == 200
    assert r.json()["reply"] == "We have 3 of those in stock."


async def test_chat_rate_limit_enforced(client, mock_llm, monkeypatch):
    token = await register_and_login(client, "ratelimit@shop.com")
    key = await _get_store_key(client, token)

    import app.rate_limit as rate_limit_module

    monkeypatch.setattr(rate_limit_module.chat_rate_limiter, "limit", 2)

    mock_llm.script([{"text": "ok"}, {"text": "ok"}, {"text": "ok"}])
    r1 = await client.post("/chat", json={"message": "one"}, headers={"X-Store-Key": key})
    r2 = await client.post("/chat", json={"message": "two"}, headers={"X-Store-Key": key})
    r3 = await client.post("/chat", json={"message": "three"}, headers={"X-Store-Key": key})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


async def test_only_store_admin_can_view_own_conversations(client, mock_llm):
    token_a = await register_and_login(client, "convview-a@shop.com")
    key_a = await _get_store_key(client, token_a)
    token_b = await register_and_login(client, "convview-b@shop.com")

    mock_llm.script([{"text": "hi"}])
    await client.post("/chat", json={"message": "hello"}, headers={"X-Store-Key": key_a})

    r = await client.get("/conversations", headers=auth_headers(token_b))
    assert r.json() == []  # store B sees none of store A's conversations
