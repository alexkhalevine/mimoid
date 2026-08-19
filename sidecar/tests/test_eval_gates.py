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


if __name__ == "__main__":
    unittest.main()
