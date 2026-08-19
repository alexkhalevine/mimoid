"""Weather lookups for the twin's weather tool, via Open-Meteo
(https://open-meteo.com) -- no API key, free for non-commercial use, which
is exactly this project.

network_activity.track() is called from *this* module rather than from its
caller (tools.py/main.py), unlike openrouter.py's and backup.py's cloud
calls, which are tracked at their call site in main.py. Those are
unconditional: main.py already knows a network call is about to happen the
moment it decides to call them. Weather is cached, so only the code that
just checked the cache knows whether a given call actually reaches the
network -- tracking one layer up would either mislabel a cache hit as
internet access, or need the hit/miss result threaded back out through an
extra return value. Owning the cache and the tracking in the same place is
simpler than either.
"""
import json
import logging
import time

import httpx

from . import config, db, network_activity

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Open-Meteo's documented WMO weather codes (a fixed, external vocabulary --
# see https://open-meteo.com/en/docs), collapsed to the granularity a spoken
# reply actually needs.
WMO_CODE_TEXT: dict[int, str] = {
    0: "clear sky",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _weather_text(code: float | None) -> str:
    if code is None:
        return "unknown conditions"
    return WMO_CODE_TEXT.get(int(code), f"unusual conditions (code {int(code)})")


# location key (lowercased configured name, or "lat,lon" for a geocoded
# lookup) -> (expires_at monotonic seconds, formatted result). Process-
# lifetime only, like ollama.py's and persona.py's other in-memory state --
# this is a desktop app with one sidecar process, not a service with
# multiple workers to keep consistent.
_cache: dict[str, tuple[float, str]] = {}


def configured_locations() -> list[dict]:
    """The Tools tab's weather locations, parsed from the settings row. Also
    the single source of truth main.py's /tools GET reads back -- avoids
    reimplementing this fallback/corrupt-JSON handling a second time."""
    raw = db.get_setting("tools_weather_locations", "")
    if not raw:
        return config.DEFAULT_WEATHER_LOCATIONS
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("tools_weather_locations setting is corrupt JSON; using the default locations")
        return config.DEFAULT_WEATHER_LOCATIONS
    return parsed if isinstance(parsed, list) and parsed else config.DEFAULT_WEATHER_LOCATIONS


def configured_location_names() -> list[str]:
    """For tools.py to bake into the get_weather spec's description as a
    hint -- steers the model toward the places it doesn't need to geocode."""
    return [entry["name"] for entry in configured_locations()]


def _resolve_location(name: str) -> tuple[str, float, float] | None:
    """Case-insensitive substring match against the configured locations
    (so "vienna" or "in ostend" both match), or None if nothing configured
    matches -- the caller falls back to geocoding."""
    needle = name.strip().lower()
    for entry in configured_locations():
        if needle in entry["name"].lower() or entry["name"].lower() in needle:
            return entry["name"], entry["latitude"], entry["longitude"]
    return None


async def _geocode(name: str, client: httpx.AsyncClient) -> tuple[str, float, float] | None:
    with network_activity.track(f"Open-Meteo: looking up {name}"):
        response = await client.get(GEOCODING_URL, params={"name": name, "count": 1})
        response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None
    top = results[0]
    return top["name"], top["latitude"], top["longitude"]


def _format(name: str, current: dict, daily: dict) -> str:
    temp = current.get("temperature_2m")
    feels_like = current.get("apparent_temperature")
    precipitation = current.get("precipitation")
    wind = current.get("wind_speed_10m")
    conditions = _weather_text(current.get("weather_code"))

    high = (daily.get("temperature_2m_max") or [None])[0]
    low = (daily.get("temperature_2m_min") or [None])[0]

    parts = [f"In {name} right now: {conditions}, {temp}°C"]
    if feels_like is not None and feels_like != temp:
        parts.append(f"(feels like {feels_like}°C)")
    parts_str = " ".join(parts) + "."
    detail = []
    if wind is not None:
        detail.append(f"wind {wind} km/h")
    if precipitation is not None:
        detail.append(f"precipitation {precipitation} mm")
    if high is not None and low is not None:
        detail.append(f"today's range {low}–{high}°C")
    if detail:
        parts_str += " " + ", ".join(detail).capitalize() + "."
    return parts_str


async def get_weather(location: str) -> str:
    """Current conditions plus today's high/low for `location`, as prose.

    Tries the configured locations first (whatever the Tools tab has been
    set to add fast-pathed, geocoding-free lookups for), falling back to a
    live geocoding lookup for anything else the twin is asked about --
    including every location, until the owner adds any.
    """
    resolved = _resolve_location(location)
    cache_key = f"{resolved[0].lower()}" if resolved else f"geo:{location.strip().lower()}"

    cached = _cache.get(cache_key)
    if cached and cached[0] > time.monotonic():
        return cached[1]

    async with httpx.AsyncClient(timeout=config.WEATHER_HTTP_TIMEOUT) as client:
        if resolved:
            name, latitude, longitude = resolved
        else:
            geocoded = await _geocode(location, client)
            if geocoded is None:
                return f"Couldn't find a place called {location!r} to check the weather for."
            name, latitude, longitude = geocoded

        with network_activity.track(f"Open-Meteo: weather for {name}"):
            response = await client.get(
                FORECAST_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                    "daily": "temperature_2m_max,temperature_2m_min",
                    "timezone": "auto",
                },
            )
            response.raise_for_status()

    payload = response.json()
    result = _format(name, payload.get("current", {}), payload.get("daily", {}))
    _cache[cache_key] = (time.monotonic() + config.WEATHER_CACHE_TTL, result)
    return result
