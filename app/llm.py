import json
import logging

import openai
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.tools import TOOL_DEFINITIONS, TOOL_NAMES, execute_tool

logger = logging.getLogger(__name__)

# Ollama (and most local servers) speak the OpenAI Chat Completions API, so
# the official openai SDK works unmodified -- just point base_url at it.
_client = AsyncOpenAI(
    api_key=settings.LLM_API_KEY or "ollama",
    base_url=settings.LLM_BASE_URL,
    timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
    max_retries=0,  # a slow local model shouldn't be silently retried
)

SYSTEM_PROMPT = (
    "You are a helpful, honest customer support assistant for an online store. "
    "Use the provided store knowledge and tools to answer questions about products, "
    "orders, and policies. To look up or change an order you need both the order number "
    "and the customer's email -- ask for whichever is missing. If you don't know something "
    "and no tool or knowledge snippet covers it, say so plainly instead of guessing. Never "
    "invent an order status, tracking number, or price. Keep replies short and friendly. "
    "Never write JSON or tool syntax in your reply: use the tools, then answer in plain sentences."
)


class LLMUnavailableError(Exception):
    """The local LLM can't be used right now (not running, model not pulled,
    timed out). The message is safe to show to the user."""


def _explain_connection_error(exc: Exception) -> LLMUnavailableError:
    if isinstance(exc, openai.APITimeoutError):
        return LLMUnavailableError(
            "The local model took too long to answer. Try a smaller model "
            "(e.g. `ollama pull llama3.2`) or raise LLM_REQUEST_TIMEOUT_SECONDS."
        )
    return LLMUnavailableError(
        f"Can't reach the local LLM at {settings.LLM_BASE_URL}. "
        "Is Ollama running? Start it with `ollama serve` (or open the Ollama app)."
    )


_NUDGE = (
    "Your last message was not a valid tool call and not a proper answer. Either call a tool "
    "with valid arguments, or reply to the customer in plain sentences with no JSON."
)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _looks_like_tool_call(content: str) -> bool:
    text = _strip_fences(content)
    return text.startswith("{") and '"name"' in text


def _parse_text_tool_call(content: str) -> tuple[str, dict] | None:
    """Small local models sometimes write {"name": ..., "parameters": {...}} as
    plain text instead of using the tool-call channel. Recover it if it's valid."""
    try:
        data = json.loads(_strip_fences(content))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("name") not in TOOL_NAMES:
        return None
    args = data.get("parameters", data.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return (data["name"], args) if isinstance(args, dict) else None


async def _create_completion(messages: list[dict], use_tools: bool):
    kwargs: dict = {"model": settings.LLM_MODEL, "messages": messages}
    if use_tools:
        kwargs["tools"] = TOOL_DEFINITIONS
        kwargs["tool_choice"] = "auto"
    try:
        return await _client.chat.completions.create(**kwargs)
    except openai.NotFoundError as exc:
        raise LLMUnavailableError(
            f"Model '{settings.LLM_MODEL}' isn't available locally. "
            f"Download it with: ollama pull {settings.LLM_MODEL}"
        ) from exc
    except openai.APIConnectionError as exc:  # includes timeouts
        raise _explain_connection_error(exc) from exc


async def run_chat_with_tools(
    messages: list[dict],
    session: AsyncSession,
    store_id: int,
    max_iterations: int | None = None,
) -> str:
    """Standard tool-calling loop: ask the model; if it wants tools, run them
    and feed results back, until it gives a final answer or we hit the
    iteration cap (a required safety valve against endless tool loops)."""
    max_iterations = max_iterations or settings.LLM_MAX_TOOL_ITERATIONS
    working_messages = list(messages)
    use_tools = True

    for _ in range(max_iterations):
        try:
            response = await _create_completion(working_messages, use_tools)
        except openai.BadRequestError as exc:
            # Some local models don't support tool calling at all. Degrade to
            # a plain knowledge-base chatbot instead of failing the request.
            if use_tools and "tool" in str(exc).lower():
                logger.warning("Model '%s' rejected tools; continuing without them.", settings.LLM_MODEL)
                use_tools = False
                continue
            raise

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            content = assistant_message.content or ""
            if use_tools and _looks_like_tool_call(content):
                parsed = _parse_text_tool_call(content)
                if parsed is None:
                    # Garbled tool call: never show it to the customer; ask the model to retry.
                    working_messages += [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": _NUDGE},
                    ]
                    continue
                name, args = parsed
                call_id = f"call_text_{len(working_messages)}"
                working_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                        ],
                    }
                )
                result = await execute_tool(name, args, session, store_id)
                working_messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
                continue
            return content or "I'm not sure how to answer that -- could you rephrase?"

        working_messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in assistant_message.tool_calls
                ],
            }
        )

        for tool_call in assistant_message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError):
                result = {"error": "The tool arguments were not a valid JSON object."}
            else:
                result = await execute_tool(tool_call.function.name, arguments, session, store_id)

            working_messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
            )

    logger.warning("Hit max tool-call iterations (%s) without a final answer.", max_iterations)
    return (
        "I looked into that but couldn't finish -- could you try asking again, "
        "maybe with a bit more detail?"
    )
