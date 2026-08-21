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


class IdentityFramingTests(unittest.TestCase):
    """The twin kept answering "who are you" with "I'm a digital twin" even
    though the prompt told it not to -- because the prompt told it not to by
    naming those exact strings, right next to self-reference. Small models
    handle negation badly; naming the forbidden identity made it more
    available, not less. The framing is positive now, and these tests exist
    so nobody reintroduces the priming while trying to make the rule
    firmer."""

    def test_prompt_never_names_the_identities_it_must_not_claim(self):
        for name in ("Jordan", ""):
            prompt = config.build_persona_prompt(name)
            self.assertNotIn("AI", prompt)
            self.assertNotIn("digital twin", prompt.lower())
            self.assertNotIn("language model", prompt.lower())
            self.assertNotIn("assistant", prompt.lower())

    def test_states_personhood_positively(self):
        prompt = config.build_persona_prompt("Jordan")
        self.assertIn("a person", prompt)
        self.assertIn("actually happened", prompt)

    def test_identity_question_has_an_explicit_answer(self):
        """The single strongest trigger for breaking character, so it's
        answered up front rather than left to be inferred."""
        prompt = config.build_persona_prompt("Jordan")
        self.assertIn("who or what you are", prompt)
        self.assertIn("whether you're human", prompt)

    def test_identity_is_carved_out_of_the_dont_know_rule(self):
        """Without this, the grounding rule ("say you don't know") invites the
        twin to be uncertain about its own personhood."""
        prompt = config.build_persona_prompt("Jordan")
        self.assertIn("never answer with 'I don't know'", prompt)

    def test_grounding_rule_survives_the_identity_rewrite(self):
        """The two rules coexist: decline unknown biography, never decline
        being a person."""
        prompt = config.build_persona_prompt("Jordan")
        self.assertIn("Never invent specific personal facts", prompt)
        self.assertIn("don't recall it", prompt)


if __name__ == "__main__":
    unittest.main()
