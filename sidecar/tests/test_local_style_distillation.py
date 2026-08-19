import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException

from app import main


class LocalStyleDistillationTests(unittest.IsolatedAsyncioTestCase):
    async def test_saves_locally_generated_guide(self):
        guide = {"content": "Write plainly.", "updated_at": "2026-07-23T00:00:00+00:00"}
        with (
            patch.object(main.db, "list_style_entries", return_value=[{"content": "Sample"}]),
            patch.object(main.persona, "distill_style_guide_locally", AsyncMock(return_value="Write plainly.")),
            patch.object(main.db, "set_style_guide", return_value=guide) as set_guide,
        ):
            result = await main.distill_style_guide_locally()

        self.assertEqual(result, guide)
        set_guide.assert_called_once_with("Write plainly.", entry_count=1)

    async def test_rejects_an_empty_corpus(self):
        with (
            patch.object(main.db, "list_style_entries", return_value=[]),
            self.assertRaises(HTTPException) as caught,
        ):
            await main.distill_style_guide_locally()

        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.detail, "Add some style examples first.")

    async def test_returns_ollama_unavailable_when_local_generation_fails(self):
        with (
            patch.object(main.db, "list_style_entries", return_value=[{"content": "Sample"}]),
            patch.object(
                main.persona,
                "distill_style_guide_locally",
                AsyncMock(side_effect=httpx.ConnectError("offline")),
            ),
            self.assertRaises(HTTPException) as caught,
        ):
            await main.distill_style_guide_locally()

        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.detail, "Ollama is not reachable. Start it and try again.")
