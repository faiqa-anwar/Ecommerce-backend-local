import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app.auth import auth_backend, fastapi_users
from app.database import create_all_tables
from app.routers import conversations, health, knowledge_base, orders, products, stores
from app.schemas import UserCreate, UserRead, UserUpdate

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dev/test convenience -- see app/database.py for why this isn't
    # Alembic-based yet.
    await create_all_tables()
    yield


app = FastAPI(title="E-commerce AI Support Backend", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError) -> JSONResponse:
    # Slightly friendlier 422 payload than FastAPI's default, still machine-readable.
    # jsonable_encoder is required here: pydantic's error dicts can carry the
    # raw exception instance in `ctx`, which json.dumps can't serialize on its own.
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors()), "message": "Invalid request body."},
    )


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def web_ui() -> FileResponse:
    """The browser UI (chat + admin panels). API docs stay at /docs."""
    return FileResponse(STATIC_DIR / "index.html")


app.include_router(health.router)
app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"]
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate), prefix="/users", tags=["users"]
)
app.include_router(stores.router)
app.include_router(products.router)
app.include_router(orders.router)
app.include_router(knowledge_base.router)
app.include_router(conversations.router)
