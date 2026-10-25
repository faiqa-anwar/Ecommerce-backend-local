import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/ecommerce_test_db"
)
os.environ.setdefault("SECRET_KEY", "test-secret")

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.database import Base, engine, async_session_maker
from app.main import app


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Every test starts from a clean schema. Simplicity over speed: for a
    project this size, dropping/recreating tables per test is fast enough
    and removes an entire class of test-order-dependence bugs.

    We dispose the engine's connection pool first: pytest-asyncio gives
    each test function its own event loop by default, but our SQLAlchemy
    `engine` is a single module-level object created once. Without this
    dispose, it can hand back a pooled asyncpg connection that belongs to
    a *previous* test's (now-closed) event loop, which asyncpg rejects.
    """
    await engine.dispose()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def client():
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest_asyncio.fixture
async def db_session():
    async with async_session_maker() as session:
        yield session


async def register_and_login(client: AsyncClient, email: str, password: str = "SuperSecret123!") -> str:
    """Helper: register a new admin (which auto-creates their store) and
    return a bearer token."""
    r = await client.post("/auth/register", json={"email": email, "password": password})
    assert r.status_code == 201, r.text

    r = await client.post(
        "/auth/jwt/login",
        data={"username": email, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_llm(monkeypatch):
    """Stub out the actual OpenRouter call so tests never need a real API
    key or network access, while still exercising the full tool-calling
    loop in app/llm.py against real tool implementations and a real DB.

    Call `mock_llm.script([...])` with a list of steps, each either:
      {"text": "final answer"}                                    -> ends the loop
      {"tool": "tool_name", "args": {...}, "id": "call_1"}        -> requests a tool call

    Each call to the fake LLM pops the next step. If the script runs out,
    it keeps returning a generic final text answer (so tests that don't
    care about the exact reply don't need to script every step).
    """
    import json as _json

    class _FakeFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = _json.dumps(arguments)

    class _FakeToolCall:
        def __init__(self, call_id, name, arguments):
            self.id = call_id
            self.function = _FakeFunction(name, arguments)

    class _FakeMessage:
        def __init__(self, content, tool_calls):
            self.content = content
            self.tool_calls = tool_calls

    class _FakeChoice:
        def __init__(self, message):
            self.message = message

    class _FakeResponse:
        def __init__(self, message):
            self.choices = [_FakeChoice(message)]

    state = {"queue": []}

    async def _fake_create(*args, **kwargs):
        if state["queue"]:
            step = state["queue"].pop(0)
        else:
            step = {"text": "Sure, happy to help!"}

        if "tool" in step:
            tool_call = _FakeToolCall(step.get("id", "call_1"), step["tool"], step.get("args", {}))
            message = _FakeMessage(step.get("content", ""), [tool_call])
        else:
            message = _FakeMessage(step["text"], None)
        return _FakeResponse(message)

    import app.llm as llm_module

    monkeypatch.setattr(llm_module._client.chat.completions, "create", _fake_create)

    class _MockLLM:
        def script(self, steps: list[dict]) -> None:
            state["queue"] = list(steps)

        def calls_remaining(self) -> int:
            return len(state["queue"])

    return _MockLLM()
