import asyncio
import logging
import random

import httpx

from . import config, db, memory, ollama, openrouter, tools

logger = logging.getLogger(__name__)

# Persona-check "arms" -- ways of assembling the persona system prompt, so a
# duel compares two genuinely different setups instead of persona-vs-a-
# generic-strawman (which always won for the persona and taught nothing).
ARM_GUIDE_FEWSHOT = "guide_fewshot"
ARM_GUIDE_ONLY = "guide_only"
ARM_FEWSHOT_ONLY = "fewshot_only"
ARM_PLAIN = "plain"

# What send_message() actually uses for live chat today -- the duel's
# headline metric is this arm's win rate against whichever challenger it
# faces, i.e. "is what I actually ship winning?"
DEFAULT_ARM = ARM_GUIDE_FEWSHOT

ARM_LABELS = {
    ARM_GUIDE_FEWSHOT: "Style guide + matched examples",
    ARM_GUIDE_ONLY: "Style guide only",
    ARM_FEWSHOT_ONLY: "Matched examples only",
    ARM_PLAIN: "No style layer",
}

_ARM_SETTINGS = {
    ARM_GUIDE_FEWSHOT: {"use_guide": True, "use_examples": True},
    ARM_GUIDE_ONLY: {"use_guide": True, "use_examples": False},
    ARM_FEWSHOT_ONLY: {"use_guide": False, "use_examples": True},
    ARM_PLAIN: {"use_guide": False, "use_examples": False},
}


class DuelUnavailable(Exception):
    """Raised when fewer than two arms would produce genuinely distinct
    system prompts for this specific question (e.g. no style examples were
    retrieved and there's no distilled guide either, so only `plain` is
    possible)."""


def eligible_arms(has_guide: bool, has_examples: bool) -> list[str]:
    """Which arms are meaningfully distinct given what was actually
    retrieved for *this* question -- not just whether the corpus has any
    guide/examples at all. guide_fewshot degenerates into guide_only the
    moment no examples were retrieved for this particular prompt, so it
    must not be offered as if it were a separate arm in that case."""
    arms = [ARM_PLAIN]
    if has_examples:
        arms.append(ARM_FEWSHOT_ONLY)
    if has_guide:
        arms.append(ARM_GUIDE_ONLY)
        if has_examples:
            arms.append(ARM_GUIDE_FEWSHOT)
    return arms


DISTILL_SYSTEM_PROMPT = (
    "You analyze a person's writing samples and produce a compact style "
    "guide another AI will use to actually WRITE as this person -- not "
    "describe them. Address the guide directly to that AI in second person "
    "(\"You write short, punchy sentences...\", \"You rarely use "
    "exclamation points...\"), as direct instructions for how to write, not "
    "a profile of the person. Cover: sentence length and rhythm, vocabulary "
    "level, tone, how they structure answers, and what they typically don't "
    "do. Include a short do/don't list. Quote 3-5 characteristic phrases or "
    "verbal habits verbatim from the samples, if any recur. Keep the whole "
    "guide under 200 words. Output only the guide, no preamble."
)


# Shared framing for every place style examples get injected (guide few-shot,
# relevance-matched fallback, and the last-resort recency dump below). Small
# local models tend to treat injected examples as facts to report or a
# transcript to describe/analyze rather than pure voice reference -- this
# spells out the distinction explicitly instead of the vaguer "match this
# voice"/"how you write and talk" wording that didn't stop that.
_STYLE_EXAMPLES_FRAMING = (
    "The lines below are SAMPLES of how you write -- they show your tone, "
    "sentence rhythm, and word choice, nothing more. They are NOT facts "
    "about you, NOT memories, and NOT part of this conversation. Never "
    "repeat their wording, never treat what they say as true or as "
    "something being asked about, and never describe or analyze them. Let "
    "them shape only HOW you phrase your answer to the real question:"
)


def _format_style_examples(entries: list[dict]) -> str:
    lines = []
    for entry in entries:
        if entry["kind"] == "qa" and entry["prompt"]:
            lines.append(f"Q: {entry['prompt']}\nA: {entry['content']}")
        else:
            lines.append(entry["content"])
    return "\n\n---\n\n".join(lines)


async def _style_section(
    style_examples: list[dict] | None = None,
    use_guide: bool = True,
    use_examples: bool = True,
) -> str:
    guide = db.get_style_guide() if use_guide else None
    if guide:
        # A guide alone is rules with zero demonstrations. Layering a few
        # examples retrieved for *this* question on top -- rather than
        # replacing them with the guide, like before -- gives the model both
        # the rules and live, relevant demonstrations to match.
        parts = [f"Your writing style:\n{guide['content']}"]
        if use_examples and style_examples:
            parts.append(f"{_STYLE_EXAMPLES_FRAMING}\n{_format_style_examples(style_examples)}")
        return "\n\n".join(parts)

    # No distilled guide (or this arm deliberately excludes it): prefer
    # semantically-relevant examples when available (retrieved by ChromaDB
    # against the current question -- the same examples that would
    # accompany a guide). If none were retrieved (empty collection or no
    # meaningful match), fall back to the newest raw entries up to the cap
    # so there's at least some style signal.
    if use_examples and style_examples:
        return f"{_STYLE_EXAMPLES_FRAMING}\n{_format_style_examples(style_examples)}"

    if not use_examples:
        # An arm that deliberately excludes examples (plain, or guide_only
        # with no guide actually available) must not fall through to the raw
        # recency dump below -- that would silently reintroduce a style
        # signal the arm exists to test the absence of.
        return ""

    entries = db.list_style_entries()
    if not entries:
        return ""
    if len(entries) > config.STYLE_EXAMPLES_FALLBACK_CAP:
        logger.warning(
            "%d style entries but no distilled guide -- only injecting the newest %d "
            "into the system prompt. Distill a style guide in Train mode to use the full corpus.",
            len(entries),
            config.STYLE_EXAMPLES_FALLBACK_CAP,
        )
        entries = entries[: config.STYLE_EXAMPLES_FALLBACK_CAP]
    return f"{_STYLE_EXAMPLES_FRAMING}\n{_format_style_examples(entries)}"


async def build_system_prompt(
    relevant_memories: list[dict],
    history_snippets: list[dict] | None = None,
    language: str = config.DEFAULT_LANGUAGE,
    style_examples: list[dict] | None = None,
    use_guide: bool = True,
    use_examples: bool = True,
    owner_name: str = config.DEFAULT_OWNER_NAME,
) -> str:
    parts = [config.build_persona_prompt(owner_name)]

    # Belt-and-braces alongside the get_current_datetime tool: a 7B model
    # will often answer "what's today?" straight from training data without
    # ever calling a tool, and being confidently a year out is exactly the
    # kind of ungrounded claim the persona prompt above promises not to make.
    # One always-present line costs ~15 tokens and makes the common case
    # right for free -- no tool round trip at all. The tool still earns its
    # place for what a static line can't do (date arithmetic).
    #
    # Placed *after* the persona intro, and minute-rounded by
    # tools.format_datetime, both for the same reason: this string is the
    # only part of the system prompt that changes on its own, and anything
    # below the point where it changes has to be re-prefilled by Ollama.
    # Keeping it out of position zero preserves the cached persona prefix,
    # and minute granularity means back-to-back turns reuse the cache
    # instead of busting it on every single message.
    parts.append(f"The current date and time is {tools.format_datetime(tools.now())}.")

    if language != config.DEFAULT_LANGUAGE:
        language_name = config.SUPPORTED_LANGUAGES.get(language, language)
        parts.append(f"Respond in {language_name}, regardless of the language of the style examples below.")

    # Memories come before the style section: the style block can be large
    # (a distilled guide plus retrieved few-shot examples, or up to 40 raw
    # entries), and burying the twin's actual memories beneath it made a small
    # model less likely to draw on them. Persona identity -> who you are and
    # what you remember -> how you write -> world-knowledge you can retell.
    memories = memory.format_memories_section(relevant_memories)
    if memories:
        parts.append(memories)

    style = await _style_section(style_examples, use_guide=use_guide, use_examples=use_examples)
    if style:
        parts.append(style)

    history = memory.format_history_section(history_snippets or [])
    if history:
        parts.append(history)

    return "\n\n".join(parts)


# Both distill functions below read style_entries fresh every time and
# write style_guide -- nothing ever reads style_guide back into
# style_entries. The guide IS model output, but it's a derived compression
# of the human-authored corpus, recomputed from that corpus on every
# distill, never a source that feeds itself or the corpus in turn. That
# distinction is what keeps this a one-way pipeline (corpus -> guide)
# instead of a self-training loop -- see the source-allowlist invariant in
# main.py's create_style_entry, which is what keeps the corpus side of that
# pipeline human-only.
async def distill_style_guide() -> str:
    entries = db.list_style_entries()
    corpus = _format_style_examples(entries)
    messages = [
        {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
        {"role": "user", "content": corpus},
    ]
    return await openrouter.chat_completion(messages)


# Rough chars-per-token estimate and context-budget fraction for capping the
# corpus sent to a local model for distillation -- mirrors the idea behind
# main.py's _windowed_history (same estimate, same "reserve room for the
# reply" logic), applied to the style corpus instead of chat history. Local
# models have a much smaller context window than OpenRouter's cloud models
# (llama3's default here is 8192 tokens vs. e.g. Claude Haiku's 200k), so an
# uncapped corpus would silently get truncated by Ollama itself with no
# signal that anything was cut -- this caps it explicitly and logs when it
# does, instead.
_DISTILL_CHARS_PER_TOKEN_ESTIMATE = 3.5
_DISTILL_CONTEXT_FRACTION = 0.75


def _cap_corpus_for_local_distillation(entries: list[dict]) -> list[dict]:
    """Keeps the newest entries (db.list_style_entries() is already
    newest-first) that fit within a character budget derived from
    OLLAMA_NUM_CTX, reserving room for DISTILL_SYSTEM_PROMPT and the
    model's own reply."""
    budget_chars = int(
        config.OLLAMA_NUM_CTX * _DISTILL_CONTEXT_FRACTION * _DISTILL_CHARS_PER_TOKEN_ESTIMATE
    ) - len(DISTILL_SYSTEM_PROMPT)
    if budget_chars <= 0:
        return entries[:1]

    kept: list[dict] = []
    used_chars = 0
    for entry in entries:
        length = len(entry["content"]) + len(entry.get("prompt") or "")
        if kept and used_chars + length > budget_chars:
            break
        kept.append(entry)
        used_chars += length

    if len(kept) < len(entries):
        logger.warning(
            "%d style entries but only %d fit the local model's context window "
            "(OLLAMA_NUM_CTX=%d) -- distilling from the newest %d only. The "
            "OpenRouter cloud option isn't limited this way.",
            len(entries),
            len(kept),
            config.OLLAMA_NUM_CTX,
            len(kept),
        )
    return kept


async def distill_style_guide_locally() -> str:
    entries = _cap_corpus_for_local_distillation(db.list_style_entries())
    corpus = _format_style_examples(entries)
    messages = [
        {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
        {"role": "user", "content": corpus},
    ]
    return await ollama.chat(messages)


def _pick_matchup(eligible: list[str]) -> tuple[str, str]:
    """Default arm vs. the eligible challenger with the fewest recorded
    appearances (round-robin, so no single challenger starves against the
    default). Falls back to the two least-appeared eligible arms if the
    default itself isn't eligible for this prompt (e.g. no examples were
    retrieved and the default requires them)."""
    appearances = db.persona_duel_arm_appearances()
    ranked = sorted(eligible, key=lambda arm: appearances.get(arm, 0))
    if DEFAULT_ARM in eligible:
        challenger = next(arm for arm in ranked if arm != DEFAULT_ARM)
        return DEFAULT_ARM, challenger
    return ranked[0], ranked[1]


async def compare_arms(
    prompt: str, language: str = config.DEFAULT_LANGUAGE, owner_name: str = config.DEFAULT_OWNER_NAME
) -> dict:
    """Retrieves once, picks a blind matchup between two persona-prompt
    arms, generates both answers concurrently, and returns everything the
    caller needs to persist + show the blind pair. Raises DuelUnavailable if
    fewer than two arms would produce genuinely distinct prompts for this
    question."""
    try:
        query_embedding = await ollama.embed(prompt)
        relevant_memories = await memory.search_memories(prompt, query_embedding=query_embedding)
        style_examples = await memory.search_style_examples(prompt, query_embedding=query_embedding)
    except httpx.HTTPError:
        relevant_memories, style_examples = [], []

    has_guide = db.get_style_guide() is not None
    eligible = eligible_arms(has_guide, bool(style_examples))
    if len(eligible) < 2:
        raise DuelUnavailable

    arm_x, arm_y = _pick_matchup(eligible)
    # Randomize which physical slot (a/b) each arm lands in -- keeps the
    # comparison blind, same as the old persona_side shuffle.
    slot_a_arm, slot_b_arm = random.sample([arm_x, arm_y], 2)

    async def _answer(arm: str) -> str:
        system_prompt = await build_system_prompt(
            relevant_memories, [], language, style_examples, owner_name=owner_name, **_ARM_SETTINGS[arm]
        )
        return await ollama.chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        )

    answer_a, answer_b = await asyncio.gather(_answer(slot_a_arm), _answer(slot_b_arm))
    return {
        "arm_a": slot_a_arm,
        "arm_b": slot_b_arm,
        "answer_a": answer_a,
        "answer_b": answer_b,
    }
