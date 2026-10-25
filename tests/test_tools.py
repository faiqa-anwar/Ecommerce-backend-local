import pytest

from app.models import Order, OrderStatus, Product, Store
from app.tools import check_inventory, execute_tool, get_order_status, initiate_return, search_products

pytestmark = pytest.mark.asyncio


async def _make_store(session) -> Store:
    store = Store(name="Test Store")
    session.add(store)
    await session.flush()
    return store


async def test_check_inventory_known_sku(db_session):
    store = await _make_store(db_session)
    db_session.add(Product(store_id=store.id, sku="ABC", name="Thing", price=1, stock_quantity=7))
    await db_session.commit()

    result = await check_inventory(db_session, store.id, "ABC")
    assert result["stock_quantity"] == 7


async def test_check_inventory_unknown_sku_returns_error_not_crash(db_session):
    store = await _make_store(db_session)
    await db_session.commit()

    result = await check_inventory(db_session, store.id, "DOES-NOT-EXIST")
    assert "error" in result


async def test_search_products_case_insensitive_partial_match(db_session):
    store = await _make_store(db_session)
    db_session.add(Product(store_id=store.id, sku="X1", name="Red Running Shoes", price=50, stock_quantity=3))
    await db_session.commit()

    result = await search_products(db_session, store.id, "running")
    assert len(result["results"]) == 1


async def test_search_products_no_match_returns_empty_not_error(db_session):
    store = await _make_store(db_session)
    await db_session.commit()

    result = await search_products(db_session, store.id, "nonexistent thing")
    assert result["results"] == []


async def test_get_order_status_correct_email(db_session):
    store = await _make_store(db_session)
    order = Order(store_id=store.id, customer_email="jane@example.com", status=OrderStatus.shipped, tracking_number="TRK123")
    db_session.add(order)
    await db_session.commit()

    result = await get_order_status(db_session, store.id, order.id, "jane@example.com")
    assert result["status"] == "shipped"
    assert result["tracking_number"] == "TRK123"


async def test_get_order_status_email_mismatch_denied(db_session):
    store = await _make_store(db_session)
    order = Order(store_id=store.id, customer_email="jane@example.com", status=OrderStatus.shipped)
    db_session.add(order)
    await db_session.commit()

    result = await get_order_status(db_session, store.id, order.id, "attacker@evil.com")
    assert "error" in result
    assert "status" not in result  # no leakage of order details


async def test_get_order_status_wrong_store_denied(db_session):
    """An order_id is just an integer -- make sure a tool can't be used to
    read another store's order by guessing IDs, even with the right email."""
    store_a = await _make_store(db_session)
    store_b = await _make_store(db_session)
    order = Order(store_id=store_a.id, customer_email="jane@example.com", status=OrderStatus.shipped)
    db_session.add(order)
    await db_session.commit()

    result = await get_order_status(db_session, store_b.id, order.id, "jane@example.com")
    assert "error" in result


async def test_get_order_status_email_is_case_insensitive(db_session):
    store = await _make_store(db_session)
    order = Order(store_id=store.id, customer_email="Jane@Example.com", status=OrderStatus.paid)
    db_session.add(order)
    await db_session.commit()

    result = await get_order_status(db_session, store.id, order.id, "jane@example.com")
    assert result["status"] == "paid"


async def test_initiate_return_on_delivered_order_succeeds(db_session):
    store = await _make_store(db_session)
    order = Order(store_id=store.id, customer_email="a@b.com", status=OrderStatus.delivered)
    db_session.add(order)
    await db_session.commit()

    result = await initiate_return(db_session, store.id, order.id, "a@b.com", "Wrong size")
    assert result["status"] == "return_requested"


async def test_initiate_return_on_pending_order_rejected(db_session):
    store = await _make_store(db_session)
    order = Order(store_id=store.id, customer_email="a@b.com", status=OrderStatus.pending)
    db_session.add(order)
    await db_session.commit()

    result = await initiate_return(db_session, store.id, order.id, "a@b.com", "Changed my mind")
    assert "error" in result


async def test_execute_tool_unknown_name_returns_error(db_session):
    store = await _make_store(db_session)
    await db_session.commit()
    result = await execute_tool("delete_the_database", {}, db_session, store.id)
    assert "error" in result


async def test_execute_tool_missing_required_argument_returns_error_not_exception(db_session):
    store = await _make_store(db_session)
    await db_session.commit()
    # get_order_status requires order_id and customer_email -- give neither.
    result = await execute_tool("get_order_status", {}, db_session, store.id)
    assert "error" in result
