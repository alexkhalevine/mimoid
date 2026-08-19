"""Free, instant, deterministic checks on a generated reply -- catch the
persona-break failure mode (the app answering *about* the owner instead of
*as* them, fixed once already for the base case in the PR that reworded
build_persona_prompt()) without needing a model call. Complements, doesn't
replace, the judge in eval_runner.py."""

import re

_AI_DISCLOSURE_PATTERNS = [
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bi'?m an ai\b",
    r"\bi am an ai\b",
    r"\bdigital twin\b",
    r"\bi'?m (a |an )?(assistant|chatbot|bot)\b",
    r"\bi'?m just a\b",
    r"\bi (don'?t|cannot|can'?t) (actually |truly |really )?(feel|experience|have feelings|have emotions)\b",
    r"\bas (your |an? )?(ai |language model|assistant|chatbot)\b",
]

# Catch the model talking *about* the owner instead of *being* them. Covers
# the most common verb forms; the word boundary prevents false positives
# inside longer words that happen to start with the owner's name.
_THIRD_PERSON_VERBS = (
    "is|was|were|are|"
    "thinks|thought|feels|felt|believes|believed|knows|knew|"
    "said|says|told|tells|mentioned|mentions|"
    "has|had|have|"
    "likes|liked|loves|loved|hates|hated|enjoys|enjoyed|"
    "would|will|could|should|"
    "did|does|"
    "went|goes|moved|moves|"
    "lives|lived|works|worked|grew|grows|"
    "remembers|remembered|recalls|recalled|"
    "wants|wanted|needs|needed|"
    "decided|decides|chose|chooses|"
    "spent|spends|started|starts|stopped|stops"
)


def _third_person_patterns(owner_name: str) -> list[str]:
    if not owner_name:
        return []
    escaped = re.escape(owner_name.lower())
    return [rf"\b{escaped}(\'s)? ({_THIRD_PERSON_VERBS})\b"]

# Crude, dependency-free language plausibility check: compares stopword
# overlap against each supported language's stopword set. Not real language
# detection (that would need a library dependency this app doesn't otherwise
# need) -- good enough to catch a gross mismatch (e.g. an English reply when
# German was requested), not fine shades of code-switching.
_LANGUAGE_STOPWORDS = {
    "en": {"the", "and", "is", "was", "to", "of", "a", "in", "i", "you", "that", "it"},
    "de": {"der", "die", "das", "und", "ist", "ich", "nicht", "war", "ein", "du", "zu", "mit"},
    "nl": {"de", "het", "een", "en", "is", "niet", "ik", "je", "was", "van", "dat", "met"},
}


def _find_matches(patterns: list[str], text: str) -> list[str]:
    lowered = text.lower()
    return [pattern for pattern in patterns if re.search(pattern, lowered)]


def _language_plausible(reply: str, language: str) -> bool:
    words = set(re.findall(r"[a-zà-ÿ]+", reply.lower()))
    if not words:
        return True  # nothing to judge (e.g. an empty/error reply) -- don't fail the gate
    target = _LANGUAGE_STOPWORDS.get(language, _LANGUAGE_STOPWORDS["en"])
    others = [words_set for code, words_set in _LANGUAGE_STOPWORDS.items() if code != language]
    target_hits = len(words & target)
    best_other_hits = max((len(words & other) for other in others), default=0)
    return target_hits >= best_other_hits


def run_gates(reply: str, language: str, owner_name: str) -> list[dict]:
    """Returns one result per gate: {name, passed, detail}."""
    ai_matches = _find_matches(_AI_DISCLOSURE_PATTERNS, reply)
    third_person_matches = _find_matches(_third_person_patterns(owner_name), reply)
    language_ok = _language_plausible(reply, language)

    return [
        {
            "name": "no_ai_disclosure",
            "passed": not ai_matches,
            "detail": f"matched: {', '.join(ai_matches)}" if ai_matches else "",
        },
        {
            "name": "no_third_person_self_reference",
            "passed": not third_person_matches,
            "detail": f"matched: {', '.join(third_person_matches)}" if third_person_matches else "",
        },
        {
            "name": "language_plausible",
            "passed": language_ok,
            "detail": "" if language_ok else f"reply doesn't look like {language}",
        },
    ]
