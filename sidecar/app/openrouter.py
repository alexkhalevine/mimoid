import httpx

from . import config, db

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterUnavailable(Exception):
    """Raised when no OpenRouter API key is configured."""


def _resolve_api_key() -> str | None:
    """A key saved in-app (via the credentials modal) takes precedence over
    the MIMOID_OPENROUTER_API_KEY env var, which remains a fallback
    default for advanced/dev setups."""
    return db.get_setting("openrouter_api_key", "") or config.OPENROUTER_API_KEY


def get_status() -> dict:
    return {"has_api_key": bool(_resolve_api_key())}


def set_api_key(api_key: str) -> dict:
    db.set_setting("openrouter_api_key", api_key.strip())
    return get_status()


async def chat_completion(messages: list[dict], model: str | None = None) -> str:
    api_key = _resolve_api_key()
    if not api_key:
        raise OpenRouterUnavailable("no OpenRouter API key is configured")

    async with httpx.AsyncClient(base_url=OPENROUTER_BASE_URL, timeout=120) as client:
        response = await client.post(
            "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model or config.OPENROUTER_MODEL, "messages": messages},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
