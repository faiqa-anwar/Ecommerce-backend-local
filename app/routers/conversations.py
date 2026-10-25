from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import current_active_user
from app.config import settings
from app.database import get_async_session
from app.llm import SYSTEM_PROMPT, LLMUnavailableError, run_chat_with_tools
from app.models import Conversation, Message, MessageRole, Store, User
from app.rag import search_knowledge_base
from app.rate_limit import RateLimitExceeded, chat_rate_limiter
from app.routers.products import _require_store
from app.schemas import ChatIn, ChatOut, ConversationOut, MessageOut

router = APIRouter(tags=["conversations"])


async def get_store_from_public_key(
    x_store_key: str = Header(..., description="Store's public chat API key"),
    session: AsyncSession = Depends(get_async_session),
) -> Store:
    result = await session.execute(select(Store).where(Store.public_api_key == x_store_key))
    store = result.scalar_one_or_none()
    if store is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid store key.")
    return store


@router.post("/chat", response_model=ChatOut)
async def chat(
    payload: ChatIn,
    store: Store = Depends(get_store_from_public_key),
    session: AsyncSession = Depends(get_async_session),
) -> ChatOut:
    try:
        chat_rate_limiter.check(store.public_api_key)
    except RateLimitExceeded as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from exc

    # Resolve or create the conversation.
    if payload.conversation_id is not None:
        result = await session.execute(
            select(Conversation).where(
                Conversation.id == payload.conversation_id, Conversation.store_id == store.id
            )
        )
        conversation = result.scalar_one_or_none()
        if conversation is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found for this store.")
    else:
        conversation = Conversation(
            store_id=store.id,
            customer_identifier=payload.customer_identifier or payload.customer_email or "anonymous",
        )
        session.add(conversation)
        await session.flush()

    # Short-term memory: pull the last N turns for context.
    history_result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.id.desc())
        .limit(settings.MAX_HISTORY_MESSAGES)
    )
    history = list(reversed(history_result.scalars().all()))

    # RAG: ground the answer in this store's knowledge base, if it has one.
    retrieved = await search_knowledge_base(session, store.id, payload.message)
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if retrieved:
        context_block = "\n---\n".join(chunk.content for chunk, _score in retrieved)
        llm_messages.append(
            {"role": "system", "content": f"Relevant store knowledge:\n{context_block}"}
        )
    if payload.customer_email:
        # Unverified, customer-supplied -- same trust level as them typing it
        # in the chat. The order tools still require it to match the order.
        llm_messages.append(
            {"role": "system", "content": f"The customer says their email is {payload.customer_email}."}
        )
    llm_messages += [{"role": m.role.value, "content": m.content} for m in history]
    llm_messages.append({"role": "user", "content": payload.message})

    try:
        reply = await run_chat_with_tools(llm_messages, session, store.id)
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    session.add(Message(conversation_id=conversation.id, role=MessageRole.user, content=payload.message))
    session.add(Message(conversation_id=conversation.id, role=MessageRole.assistant, content=reply))
    await session.commit()

    return ChatOut(conversation_id=conversation.id, reply=reply)


# ---------- Admin-side conversation inspection (JWT-protected) ----------
@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[Conversation]:
    store_id = _require_store(user)
    result = await session.execute(select(Conversation).where(Conversation.store_id == store_id))
    return list(result.scalars().all())


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def get_conversation_messages(
    conversation_id: int,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> list[Message]:
    store_id = _require_store(user)
    conv_result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.store_id == store_id)
    )
    if conv_result.scalar_one_or_none() is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")

    result = await session.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
    )
    return list(result.scalars().all())


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    store_id = _require_store(user)
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.store_id == store_id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
    await session.delete(conversation)
    await session.commit()
