"""
Tools the LLM can call mid-conversation. Each tool is:
  1. Declared in TOOL_DEFINITIONS (OpenAI/OpenRouter function-calling schema)
  2. Implemented as an async function taking (session, store_id, **kwargs)

Security note: these run against a customer-facing endpoint with NO
customer login, so every tool that reads or writes order data must verify
the caller actually knows something private about that order (here: the
email on file) before returning or changing anything. Never trust an
order_id alone.
"""
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Order, OrderStatus, Product

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the store's product catalog by name, category or keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords."},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check current stock quantity for a product by SKU.",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Look up an order's status and tracking info. Needs the customer's email "
                "AND either the order number or the tracking number (e.g. TRK-3-DEMO)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_email": {"type": "string", "description": "Email the order was placed with."},
                    "order_id": {"type": "integer", "description": "Order number, digits only."},
                    "tracking_number": {"type": "string", "description": "Tracking number, if no order number."},
                },
                "required": ["customer_email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "initiate_return",
            "description": (
                "Start a return for a shipped or delivered order. Needs the customer's email, "
                "the order number (or tracking number), and a reason."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_email": {"type": "string"},
                    "reason": {"type": "string"},
                    "order_id": {"type": "integer", "description": "Order number, digits only."},
                    "tracking_number": {"type": "string"},
                },
                "required": ["customer_email", "reason"],
            },
        },
    },
]

_RETURN_ELIGIBLE_STATUSES = {OrderStatus.shipped, OrderStatus.delivered}


async def search_products(session: AsyncSession, store_id: int, query: str, max_results: int = 5) -> dict:
    max_results = max(1, min(int(max_results or 5), 20))
    like = f"%{query.strip()}%"
    stmt = (
        select(Product)
        .where(
            Product.store_id == store_id,
            (Product.name.ilike(like)) | (Product.description.ilike(like)) | (Product.category.ilike(like)),
        )
        .limit(max_results)
    )
    result = await session.execute(stmt)
    products = result.scalars().all()
    if not products:
        return {"results": [], "message": "No matching products found."}
    return {
        "results": [
            {
                "id": p.id,
                "sku": p.sku,
                "name": p.name,
                "price": float(p.price),
                "in_stock": p.stock_quantity > 0,
                "stock_quantity": p.stock_quantity,
            }
            for p in products
        ]
    }


async def check_inventory(session: AsyncSession, store_id: int, sku: str) -> dict:
    stmt = select(Product).where(Product.store_id == store_id, Product.sku == sku.strip())
    result = await session.execute(stmt)
    product = result.scalar_one_or_none()
    if product is None:
        return {"error": f"No product found with SKU '{sku}'."}
    return {"sku": product.sku, "name": product.name, "stock_quantity": product.stock_quantity}


_MAX_INT32 = 2**31 - 1
_NO_MATCH = {"error": "No matching order found for those details."}


def _parse_order_id(value) -> int | None:
    """Models often send order ids as "3", "#3" or even junk. Accept what is
    clearly a number, return None for everything else (never raise)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        value = int(text) if text.isdigit() else None
    if isinstance(value, int) and 0 < value <= _MAX_INT32:
        return value
    return None


async def _find_order(session: AsyncSession, store_id: int, order_id, tracking_number, customer_email) -> Order | None:
    """Find an order by number or tracking number, only if the email matches.
    Same None result for "no such order" and "wrong email" so callers can't
    probe which order numbers exist."""
    email = (customer_email or "").strip().lower()
    if not email:
        return None
    parsed_id = _parse_order_id(order_id)
    tracking = (tracking_number or "").strip()
    # A non-numeric "order id" (e.g. "TRK-3-DEMO") is really a tracking number.
    if parsed_id is None and not tracking and isinstance(order_id, str) and order_id.strip():
        tracking = order_id.strip()

    if parsed_id is not None:
        stmt = select(Order).where(Order.id == parsed_id, Order.store_id == store_id)
    elif tracking:
        stmt = select(Order).where(Order.tracking_number == tracking, Order.store_id == store_id)
    else:
        return None
    order = (await session.execute(stmt)).scalars().first()
    if order is None or order.customer_email.lower() != email:
        return None
    return order


async def get_order_status(
    session: AsyncSession, store_id: int, order_id=None, customer_email: str = "", tracking_number: str | None = None
) -> dict:
    if not (customer_email or "").strip():
        return {"error": "customer_email is required. Ask the customer for the email on the order."}
    if order_id in (None, "") and not tracking_number:
        return {"error": "Need an order number or tracking number. Ask the customer for one."}
    order = await _find_order(session, store_id, order_id, tracking_number, customer_email)
    if order is None:
        return _NO_MATCH
    return {
        "order_id": order.id,
        "status": order.status.value,
        "tracking_number": order.tracking_number,
        "total_amount": float(order.total_amount),
    }


async def initiate_return(
    session: AsyncSession,
    store_id: int,
    order_id=None,
    customer_email: str = "",
    reason: str = "",
    tracking_number: str | None = None,
) -> dict:
    if not (customer_email or "").strip():
        return {"error": "customer_email is required. Ask the customer for the email on the order."}
    if not (reason or "").strip():
        return {"error": "A reason for the return is required. Ask the customer why."}
    order = await _find_order(session, store_id, order_id, tracking_number, customer_email)
    if order is None:
        return _NO_MATCH
    if order.status not in _RETURN_ELIGIBLE_STATUSES:
        return {
            "error": (
                f"Order is currently '{order.status.value}' and isn't eligible for return "
                "(must be shipped or delivered)."
            )
        }
    order.status = OrderStatus.return_requested
    order.return_reason = reason.strip()[:1000]
    session.add(order)
    await session.commit()
    return {"order_id": order.id, "status": order.status.value, "message": "Return initiated."}


_TOOL_IMPLEMENTATIONS = {
    "search_products": search_products,
    "check_inventory": check_inventory,
    "get_order_status": get_order_status,
    "initiate_return": initiate_return,
}


TOOL_NAMES = set(_TOOL_IMPLEMENTATIONS)


async def execute_tool(name: str, arguments: dict, session: AsyncSession, store_id: int) -> dict:
    """Dispatch a tool call by name. Never raises -- any failure (unknown
    tool, bad arguments, DB error) is turned into an {"error": ...} dict so
    it can be fed straight back to the LLM as a tool result instead of
    crashing the request."""
    impl = _TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        return await impl(session, store_id, **arguments)
    except TypeError as exc:
        logger.warning("Tool '%s' called with bad arguments %s: %s", name, arguments, exc)
        return {"error": f"Invalid arguments for tool '{name}': {exc}"}
    except Exception as exc:  # noqa: BLE001 - intentionally broad: tools must never 500 the request
        logger.exception("Tool '%s' raised an unexpected error", name)
        # A failed query leaves the DB transaction aborted; reset it so the
        # rest of the request (saving the chat messages) still works.
        await session.rollback()
        return {"error": f"Internal error running tool '{name}': {exc}"}
