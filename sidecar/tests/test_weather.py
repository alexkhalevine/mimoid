import asyncio
import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import httpx

from app import tools, weather

# A fixture location, deliberately not the same as any of the app's actual
# defaults -- these tests exercise the "configured location" fast path, not
# what DEFAULT_WEATHER_LOCATIONS happens to contain (which is empty; see
# config.py).
_FIXTURE_LOCATION = {"name": "Springfield", "latitude": 39.9967, "longitude": -83.0059}


def _forecast_response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "current": {
                "temperature_2m": 21.0,
                "apparent_temperature": 20.0,
                "precipitation": 0.0,
                "weather_code": 1,
                "wind_speed_10m": 8.0,
            },
            "daily": {"temperature_2m_max": [24.0], "temperature_2m_min": [14.0]},
        },
    )


def _geocoding_response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200, json={"results": [{"name": "Tokyo", "latitude": 35.6895, "longitude": 139.6917}]}
    )


def _empty_geocoding_response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"results": []})


class WmoCodeTests(unittest.TestCase):
    def test_known_code(self):
        self.assertEqual(weather._weather_text(61), "slight rain")

    def test_unknown_code_reports_the_raw_code(self):
        self.assertIn("123", weather._weather_text(123))

    def test_missing_code(self):
        self.assertEqual(weather._weather_text(None), "unknown conditions")


class FormatTests(unittest.TestCase):
    """_format is what the model actually reads back as the tool result, so
    it has to degrade gracefully when a field Open-Meteo could plausibly
    omit is missing, not just when everything is present."""

    def test_full_payload_mentions_every_field(self):
        text = weather._format(
            "Vienna",
            {
                "temperature_2m": 21.0,
                "apparent_temperature": 19.0,
                "precipitation": 0.2,
                "weather_code": 2,
                "wind_speed_10m": 12.0,
            },
            {"temperature_2m_max": [24.0], "temperature_2m_min": [14.0]},
        )
        self.assertIn("Vienna", text)
        self.assertIn("partly cloudy", text)
        self.assertIn("feels like 19.0", text)
        self.assertIn("14.0", text)
        self.assertIn("24.0", text)

    def test_feels_like_omitted_when_equal_to_actual_temperature(self):
        text = weather._format(
            "Ostend",
            {
                "temperature_2m": 15.0,
                "apparent_temperature": 15.0,
                "precipitation": None,
                "weather_code": 0,
                "wind_speed_10m": None,
            },
            {},
        )
        self.assertNotIn("feels like", text)

    def test_missing_daily_range_is_not_reported_as_zero(self):
        text = weather._format(
            "Ostend",
            {"temperature_2m": 15.0, "apparent_temperature": 16.0, "precipitation": 0, "weather_code": 3, "wind_speed_10m": 5.0},
            {},
        )
        self.assertNotIn("range", text)


class GetWeatherTests(unittest.TestCase):
    def setUp(self):
        weather._cache.clear()
        self.addCleanup(weather._cache.clear)

        self.settings: dict[str, str] = {"tools_weather_locations": json.dumps([_FIXTURE_LOCATION])}
        get_setting = patch.object(weather.db, "get_setting", side_effect=lambda k, d: self.settings.get(k, d))
        get_setting.start()
        self.addCleanup(get_setting.stop)

        self.track_calls: list[str] = []

        @contextmanager
        def fake_track(label: str):
            self.track_calls.append(label)
            yield

        track_patch = patch.object(weather.network_activity, "track", fake_track)
        track_patch.start()
        self.addCleanup(track_patch.stop)

    def _install_transport(self, handler) -> dict[str, int]:
        call_count = {"n": 0}
        # weather.httpx *is* the httpx module (same object this file imported),
        # so patching weather.httpx.AsyncClient also patches the name this
        # factory would otherwise close over -- capture the real class first
        # or the factory recurses into itself.
        real_async_client = httpx.AsyncClient

        def counting_handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return handler(request)

        def factory(*_args, **_kwargs) -> httpx.AsyncClient:
            return real_async_client(transport=httpx.MockTransport(counting_handler))

        patcher = patch.object(weather.httpx, "AsyncClient", factory)
        patcher.start()
        self.addCleanup(patcher.stop)
        return call_count

    def test_configured_location_hits_forecast_only(self):
        calls = self._install_transport(_forecast_response)
        result = asyncio.run(weather.get_weather("springfield"))
        self.assertIn("Springfield", result)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(self.track_calls, ["Open-Meteo: weather for Springfield"])

    def test_cache_hit_skips_network_and_tracking(self):
        """The whole point of caching weather: a second ask about the same
        city in one conversation shouldn't refetch, and -- since nothing
        left the machine -- must not appear in the footer's internet-access
        history either."""
        calls = self._install_transport(_forecast_response)
        asyncio.run(weather.get_weather("Springfield"))
        asyncio.run(weather.get_weather("Springfield"))
        self.assertEqual(calls["n"], 1)
        self.assertEqual(len(self.track_calls), 1)

    def test_cache_expiry_refetches(self):
        calls = self._install_transport(_forecast_response)
        asyncio.run(weather.get_weather("Springfield"))
        _expires_at, cached_value = weather._cache["springfield"]
        weather._cache["springfield"] = (0.0, cached_value)  # force expiry without sleeping
        asyncio.run(weather.get_weather("Springfield"))
        self.assertEqual(calls["n"], 2)

    def test_geocoding_fallback_for_unconfigured_name(self):
        responses = iter([_geocoding_response, _forecast_response])
        calls = self._install_transport(lambda request: next(responses)(request))
        result = asyncio.run(weather.get_weather("Tokyo"))
        self.assertIn("Tokyo", result)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(
            self.track_calls, ["Open-Meteo: looking up Tokyo", "Open-Meteo: weather for Tokyo"]
        )

    def test_unknown_place_returns_friendly_message_and_is_not_cached(self):
        self._install_transport(_empty_geocoding_response)
        result = asyncio.run(weather.get_weather("Nowhereville"))
        self.assertIn("Nowhereville", result)
        self.assertIn("Couldn't find", result)
        self.assertEqual(weather._cache, {})

    def test_configured_locations_falls_back_to_defaults_on_corrupt_json(self):
        # No default city ships with the app (see config.DEFAULT_WEATHER_LOCATIONS),
        # so the fallback from corrupt settings is an empty list, not a
        # pre-picked place.
        self.settings["tools_weather_locations"] = "{not json"
        self.assertEqual(weather.configured_location_names(), [])

    def test_configured_locations_falls_back_to_defaults_on_empty_list(self):
        # An empty list would leave every lookup falling back to geocoding --
        # main.py's PUT /tools already rejects this at the edit, but the
        # fallback stays here too as a second line of defence.
        self.settings["tools_weather_locations"] = "[]"
        self.assertEqual(weather.configured_location_names(), [])


class AvailableSpecsWeatherTests(unittest.TestCase):
    def setUp(self):
        self.store: dict[str, str] = {}
        get = patch.object(tools.db, "get_setting", side_effect=lambda k, d: self.store.get(k, d))
        put = patch.object(tools.db, "set_setting", side_effect=self.store.__setitem__)
        get.start()
        put.start()
        self.addCleanup(get.stop)
        self.addCleanup(put.stop)

    def test_weather_spec_bakes_in_configured_location_names(self):
        self.store["tools_weather_locations"] = json.dumps([_FIXTURE_LOCATION])
        specs = tools.available_specs()
        weather_spec = next(s for s in specs if s["function"]["name"] == "get_weather")
        self.assertIn("Springfield", weather_spec["function"]["description"])

    def test_weather_spec_with_no_configured_locations_omits_the_hint_cleanly(self):
        # No default city ships with the app, so a fresh install (or one
        # where locations were somehow cleared) must not render a dangling
        # "Configured: ." in the description sent to the model.
        specs = tools.available_specs()
        weather_spec = next(s for s in specs if s["function"]["name"] == "get_weather")
        self.assertNotIn("Configured:", weather_spec["function"]["description"])
        self.assertNotIn("{locations_hint}", weather_spec["function"]["description"])

    def test_disabled_weather_is_not_offered(self):
        tools.set_feature_enabled(tools.FEATURE_WEATHER, False)
        names = [spec["function"]["name"] for spec in tools.available_specs()]
        self.assertNotIn("get_weather", names)

    def test_available_specs_does_not_mutate_the_registry(self):
        """available_specs() formats {locations_hint} into a *copy* of the
        spec. If it ever formatted SPECS in place, the second call would
        format an already-formatted string and silently corrupt the
        description."""
        tools.available_specs()
        tools.available_specs()
        raw_entry = next(entry for entry in tools.SPECS if entry["spec"]["function"]["name"] == "get_weather")
        self.assertIn("{locations_hint}", raw_entry["spec"]["function"]["description"])


if __name__ == "__main__":
    unittest.main()
