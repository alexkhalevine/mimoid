import re
import unittest

from app import config


class BuildPersonaPromptTests(unittest.TestCase):
    """The base persona instruction used to hardcode "Alex" (and the
    gendered "him"/"his" that went with it) -- now it's parameterized by
    whatever name the owner set during first-run onboarding, and repeats the
    name instead of a pronoun so it doesn't hardcode a gender either."""

    def test_owner_name_is_woven_in(self):
        prompt = config.build_persona_prompt("Jordan")
        self.assertIn("You are Jordan.", prompt)
        self.assertIn("you ARE Jordan", prompt)
        self.assertIn("the way Jordan would", prompt)

    def test_no_leftover_alex(self):
        prompt = config.build_persona_prompt("Jordan")
        self.assertNotIn("Alex", prompt)

    def test_no_gendered_pronoun(self):
        prompt = config.build_persona_prompt("Jordan")
        for pronoun in ("him", "his", "he", "she", "her"):
            self.assertNotRegex(prompt.lower(), rf"\b{re.escape(pronoun)}\b")

    def test_empty_name_falls_back_to_generic_phrasing_without_crashing(self):
        prompt = config.build_persona_prompt("")
        self.assertNotIn("Alex", prompt)
        self.assertIn("you ARE them", prompt)
        # No broken "You are ." from an empty interpolation.
        self.assertNotIn("You are .", prompt)


if __name__ == "__main__":
    unittest.main()
