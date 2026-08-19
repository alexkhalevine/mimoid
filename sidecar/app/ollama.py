import json
from collections.abc import AsyncIterator

import httpx

from . import config


async def list_models() -> list[dict]:
    async with httpx.AsyncClient(base_url=config.OLLAMA_BASE_URL, timeout=10) as client:
        response = await client.get("/api/tags")
        response.raise_for_status()
        return response.json().get("models", [])


async def pull_model(name: str) -> AsyncIterator[bytes]:
    """Streams Ollama's NDJSON pull-progress lines through unchanged."""
    async with (
        httpx.AsyncClient(base_url=config.OLLAMA_BASE_URL, timeout=None) as client,
        client.stream("POST", "/api/pull", json={"name": name}) as response,
    ):
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line:
                yield line.encode() + b"\n"


async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient(
        base_url=config.OLLAMA_BASE_URL, timeout=config.OLLAMA_EMBED_TIMEOUT
    ) as client:
        response = await client.post(
            "/api/embeddings",
            # keep_alive keeps nomic-embed-text resident between turns, so it
            # isn't evicted by the warm chat model and re-swapped (slowly) on
            # every message -- the swap happens once, not per query.
            json={
                "model": config.EMBEDDING_MODEL,
                "prompt": text,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
            },
        )
        response.raise_for_status()
        return response.json()["embedding"]


def _generation_options() -> dict:
    return {
        "num_ctx": config.OLLAMA_NUM_CTX,
        "temperature": config.OLLAMA_TEMPERATURE,
        "repeat_penalty": config.OLLAMA_REPEAT_PENALTY,
    }


async def chat(messages: list[dict], model: str | None = None) -> str:
    """Non-streaming chat completion, for one-off generations (e.g. the
    persona A/B comparison, or the eval harness's local judge) rather than
    the live chat UI. Defaults to the day-to-day model; callers like the eval
    harness can point this at a different local model (e.g. a stronger judge)
    without affecting the default chat path."""
    payload = {
        "model": model or config.DEFAULT_MODEL,
        "messages": messages,
        "stream": False,
        "options": _generation_options(),
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    async with httpx.AsyncClient(base_url=config.OLLAMA_BASE_URL, timeout=None) as client:
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]


async def chat_stream(messages: list[dict], tools: list[dict] | None = None) -> AsyncIterator[dict]:
    """Streams one assistant turn, yielding either content deltas or the
    tool calls the model wants to make.

    Yields `{"type": "token", "text": ...}` for reply text and
    `{"type": "tool_calls", "calls": [...]}` when the model asks for tools.
    A tool-calling model emits tool calls *instead of* content for a turn,
    so an ordinary tool-free reply streams exactly as it always did.

    Tools are passed to the same streaming request rather than to a separate
    non-streaming "should I use a tool?" pass. That alternative would have to
    generate the entire reply unstreamed on every tool-free turn and then
    throw it away to regenerate it streamed -- roughly doubling the wait on
    the majority of messages, which use no tool at all. This way tools cost
    nothing until one is actually called.
    """
    payload = {
        "model": config.DEFAULT_MODEL,
        "messages": messages,
        "stream": True,
        "options": _generation_options(),
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    if tools:
        payload["tools"] = tools
    async with (
        httpx.AsyncClient(base_url=config.OLLAMA_BASE_URL, timeout=None) as client,
        client.stream("POST", "/api/chat", json=payload) as response,
    ):
        response.raise_for_status()
        pending_calls: list[dict] = []
        async for line in response.aiter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            message = chunk.get("message", {})
            content = message.get("content", "")
            if content:
                yield {"type": "token", "text": content}
            # Ollama can spread tool calls over several chunks, so collect
            # them and emit one round at the end rather than dispatching a
            # half-built call.
            if message.get("tool_calls"):
                pending_calls.extend(message["tool_calls"])
            if chunk.get("done"):
                break
        if pending_calls:
            yield {"type": "tool_calls", "calls": pending_calls}
