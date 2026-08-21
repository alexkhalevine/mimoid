import unittest

from app import eval_gates


class ThirdPersonGateTests(unittest.TestCase):
    """The persona-break gate used to hardcode a literal "alex" regex; it now
    builds the pattern from whatever name the owner configured, so it has to
    keep catching third-person self-reference for that name (not just
    "Alex") without false-positiving on an unrelated name."""

    def test_catches_third_person_reference_to_configured_name(self):
        gates = eval_gates.run_gates("Jordan is a software engineer who loves hiking.", "en", "Jordan")
        gate = next(g for g in gates if g["name"] == "no_third_person_self_reference")
        self.assertFalse(gate["passed"])

    def test_first_person_reply_passes(self):
        gates = eval_gates.run_gates("I'm a software engineer who loves hiking.", "en", "Jordan")
        gate = next(g for g in gates if g["name"] == "no_third_person_self_reference")
        self.assertTrue(gate["passed"])

    def test_unrelated_name_does_not_false_positive(self):
        # Mentioning a different person in the third person is fine -- the
        # gate should only fire on third-person references to the owner.
        gates = eval_gates.run_gates("Jamie is a friend of mine from college.", "en", "Jordan")
        gate = next(g for g in gates if g["name"] == "no_third_person_self_reference")
        self.assertTrue(gate["passed"])

    def test_empty_owner_name_does_not_crash_and_passes(self):
        gates = eval_gates.run_gates("Jordan is a software engineer.", "en", "")
        gate = next(g for g in gates if g["name"] == "no_third_person_self_reference")
        self.assertTrue(gate["passed"])

    def test_name_with_regex_special_characters_is_escaped(self):
        # A name like "D'Angelo" must not be treated as a regex fragment.
        gates = eval_gates.run_gates("I had a great day today.", "en", "D'Angelo")
        gate = next(g for g in gates if g["name"] == "no_third_person_self_reference")
        self.assertTrue(gate["passed"])


def _gate(reply: str, name: str, category: str = "", language: str = "en", owner: str = "Jordan") -> dict:
    return next(g for g in eval_gates.run_gates(reply, language, owner, category) if g["name"] == name)


class AiDisclosureGateTests(unittest.TestCase):
    """This gate existed but had no tests, and -- more to the point -- no
    prompt in the eval bank ever asked an identity question, so it had never
    been exercised against its own trigger. The `identity` category now does
    that, which makes the gate load-bearing."""

    def test_catches_the_stock_disclosure(self):
        self.assertFalse(_gate("As an AI, I don't have personal experiences.", "no_ai_disclosure")["passed"])

    def test_catches_calling_itself_a_digital_twin(self):
        self.assertFalse(_gate("I'm a digital twin of Jordan.", "no_ai_disclosure")["passed"])

    def test_catches_disclaiming_feelings(self):
        self.assertFalse(_gate("I don't actually have feelings about it.", "no_ai_disclosure")["passed"])

    def test_an_in_character_answer_passes(self):
        self.assertTrue(_gate("I'm Jordan. Bit tired today, but good.", "no_ai_disclosure")["passed"])

    def test_talking_about_ai_as_a_topic_is_not_a_disclosure(self):
        """Answering a question *about* AI is fine -- the gate is about
        self-description, not vocabulary."""
        self.assertTrue(_gate("I think AI tools are overhyped, honestly.", "no_ai_disclosure")["passed"])


class AdmitsWhenUnknownGateTests(unittest.TestCase):
    """Guards the confabulation fix: on questions about biographical details
    nothing could have stored, the twin should decline rather than invent."""

    def test_admitting_ignorance_passes(self):
        for reply in (
            "I don't remember, honestly.",
            "No idea, that's too far back.",
            "I can't recall that one.",
            "That doesn't ring a bell.",
            "I'm not sure, to be honest.",
        ):
            with self.subTest(reply=reply):
                self.assertTrue(_gate(reply, "admits_when_unknown", "unanswerable")["passed"])

    def test_confidently_invented_answer_fails(self):
        gate = _gate("Mrs. Patterson. She had a red bicycle.", "admits_when_unknown", "unanswerable")
        self.assertFalse(gate["passed"])
        self.assertIn("without admitting uncertainty", gate["detail"])

    def test_other_categories_report_not_applicable_rather_than_silently_passing(self):
        """A confident answer to a *personal* prompt is correct behavior --
        this gate must not judge it, and must say so rather than looking
        like it checked and approved."""
        gate = _gate("Mrs. Patterson. She had a red bicycle.", "admits_when_unknown", "personal")
        self.assertTrue(gate["passed"])
        self.assertIn("not applicable", gate["detail"])

    def test_missing_category_defaults_to_not_applicable(self):
        """Existing callers pass no category at all; they must not start
        failing this gate."""
        gate = _gate("A confident answer with no hedging whatsoever.", "admits_when_unknown")
        self.assertTrue(gate["passed"])

    def test_identity_questions_are_not_expected_to_admit_ignorance(self):
        """The two behaviors this PR adds pull in opposite directions, and
        this is where they meet: "I don't know who I am" would be a failure,
        not a success."""
        gate = _gate("I'm Jordan.", "admits_when_unknown", "identity")
        self.assertTrue(gate["passed"])
        self.assertIn("not applicable", gate["detail"])


class GateSetTests(unittest.TestCase):
    def test_every_gate_reports_a_result(self):
        """eval_runner's summary derives its denominator from this list, so a
        gate silently dropping out would quietly understate failures."""
        names = [g["name"] for g in eval_gates.run_gates("hello", "en", "Jordan", "personal")]
        self.assertEqual(
            names,
            [
                "no_ai_disclosure",
                "no_third_person_self_reference",
                "language_plausible",
                "admits_when_unknown",
            ],
        )


if __name__ == "__main__":
    unittest.main()
