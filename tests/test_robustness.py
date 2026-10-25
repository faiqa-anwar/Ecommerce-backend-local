import pytest
from tests.conftest import auth_headers, register_and_login

pytestmark = pytest.mark.asyncio


async def _setup(client):
    token = await register_and_login(client, "repro@shop.com")
    key = (await client.get("/stores/me", headers=auth_headers(token))).json()["public_api_key"]
    return token, key


@pytest.mark.parametrize("bad_id", ["TRK-3-DEMO", "3", "#3", 99999999999, None])
async def test_odd_order_id_from_model_does_not_500(client, mock_llm, bad_id):
    _, key = await _setup(client)
    mock_llm.script([
        {"tool": "get_order_status", "args": {"order_id": bad_id, "customer_email": "a@b.com"}},
        {"text": "Could you confirm your order number?"},
    ])
    r = await client.post("/chat", json={"message": "status?"}, headers={"X-Store-Key": key})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("text", ['{"name":"get_order_status","parameters{"}}', '{"name": "nope", "parameters": {}}'])
async def test_tool_call_written_as_text_is_not_shown_to_customer(client, mock_llm, text):
    _, key = await _setup(client)
    mock_llm.script([{"text": text}, {"text": "Sure, what is your order number and email?"}])
    r = await client.post("/chat", json={"message": "status?"}, headers={"X-Store-Key": key})
    assert r.status_code == 200
    assert not r.json()["reply"].lstrip().startswith("{")


async def test_valid_tool_call_written_as_text_is_executed(client, mock_llm):
    token, key = await _setup(client)
    p = await client.post("/products", json={"sku": "T1", "name": "Mug", "price": 5, "stock_quantity": 7}, headers=auth_headers(token))
    mock_llm.script([
        {"text": '```json\n{"name": "check_inventory", "parameters": {"sku": "T1"}}\n```'},
        {"text": "We have 7 mugs in stock."},
    ])
    r = await client.post("/chat", json={"message": "mugs in stock?"}, headers={"X-Store-Key": key})
    assert r.json()["reply"] == "We have 7 mugs in stock."


async def test_tracking_number_lookup_needs_matching_email(client, mock_llm):
    token, key = await _setup(client)
    p = await client.post("/products", json={"sku": "T2", "name": "Cap", "price": 9, "stock_quantity": 3}, headers=auth_headers(token))
    o = await client.post("/orders", json={"customer_email": "kim@example.com", "items": [{"product_id": p.json()["id"], "quantity": 1}]}, headers=auth_headers(token))
    await client.patch(f"/orders/{o.json()['id']}", json={"status": "shipped", "tracking_number": "TRK-3-DEMO"}, headers=auth_headers(token))
    from app.tools import get_order_status
    from app.database import async_session_maker
    from sqlalchemy import select
    from app.models import Store
    async with async_session_maker() as s:
        store_id = (await s.execute(select(Store.id))).scalars().first()
        ok = await get_order_status(s, store_id, "TRK-3-DEMO", "KIM@example.com")      # tracking passed as order_id
        ok2 = await get_order_status(s, store_id, None, "kim@example.com", "TRK-3-DEMO")
        bad = await get_order_status(s, store_id, None, "other@example.com", "TRK-3-DEMO")
        missing_email = await get_order_status(s, store_id, "TRK-3-DEMO", "")
    assert ok["status"] == "shipped" and ok2["status"] == "shipped"
    assert "error" in bad and "status" not in bad
    assert "error" in missing_email
