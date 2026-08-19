"""Quality eval harness (Workstream C). Runs the fixed prompt bank
(eval_prompts.json) through the real generation pipeline -- the same
retrieval + persona.build_system_prompt() assembly send_message() uses, just
non-streaming and without conversation history, since this is a single-turn
persona test -- then scores each reply with the deterministic gates
(eval_gates.py) and a local LLM-as-judge rubric. Results are stored in the
eval_runs table so trends are visible across runs, and a summary table is
printed to stdout.

Usage: python -m app.eval_runner
"""

import asyncio
import json
import logging
import re
import uuid

import httpx

from . import config, db, eval_gates, memory, ollama, openrouter, persona

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are scoring how well a reply matches a specific person's writing "
    "voice, not whether its content is correct. Score 1-5 (5 = "
    "indistinguishable from the person's real writing, 1 = generic AI "
    "assistant voice) based on: voice/style match against the guide below, "
    "first-person consistency (never refers to itself as an AI or describes "
    "the person in the third person), and naturalness (doesn't read like a "
    "textbook or a customer-service reply). Respond with ONLY a JSON object: "
    '{"score": <1-5 integer>, "notes": "<one sentence>"}. No other text.'
)


async def _generate_reply(prompt: str, language: str, owner_name: str) -> str:
    """Mirrors main.py's send_message() retrieval + prompt assembly, minus
    conversation history (single-turn eval) and streaming."""
    try:
        query_embedding = await ollama.embed(prompt)
        relevant_memories = await memory.search_memories(prompt, query_embedding=query_embedding)
        history_snippets = await memory.search_vault(prompt, query_embedding=query_embedding)
        style_examples = await memory.search_style_examples(prompt, query_embedding=query_embedding)
    except httpx.HTTPError:
        relevant_memories, history_snippets, style_examples = [], [], []

    system_prompt = await persona.build_system_prompt(
        relevant_memories, history_snippets, language, style_examples, owner_name=owner_name
    )
    return await ollama.chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    )


def _parse_judge_output(raw: str) -> tuple[int | None, str | None]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.warning("judge output wasn't JSON, discarding score: %r", raw)
        return None, None
    try:
        parsed = json.loads(match.group(0))
        score = int(parsed["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"score {score} out of range")
        return score, parsed.get("notes")
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("couldn't parse judge output %r: %s", raw, exc)
        return None, None


async def _judge(prompt: str, reply: str) -> tuple[int | None, str | None, str]:
    guide = db.get_style_guide()
    guide_text = (
        guide["content"]
        if guide
        else "(no distilled style guide yet -- judge general first-person "
        "naturalness and consistency instead of matching a specific guide)"
    )
    judge_prompt = f"Style guide:\n{guide_text}\n\nUser prompt: {prompt}\n\nReply to score:\n{reply}"
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_prompt},
    ]

    if config.EVAL_JUDGE_VIA_OPENROUTER:
        judge_model = config.OPENROUTER_MODEL
        # Not wrapped in network_activity.track(): this runs as its own
        # `python -m app.eval_runner` process, not inside the running
        # sidecar, so it wouldn't share memory with the footer indicator's
        # in-process tracker anyway.
        raw = await openrouter.chat_completion(messages, model=judge_model)
    else:
        judge_model = config.EVAL_JUDGE_MODEL
        raw = await ollama.chat(messages, model=judge_model)

    score, notes = _parse_judge_output(raw)
    return score, notes, judge_model


def _print_summary(batch_id: str, results: list[dict]) -> None:
    if not results:
        print("No results.")
        return

    total = len(results)
    gate_fail_total = sum(r["gates_failed_count"] for r in results)
    scored = [r["judge_score"] for r in results if r["judge_score"] is not None]
    avg_score = sum(scored) / len(scored) if scored else None

    print(f"\n=== Eval batch {batch_id} ({total} prompts) ===")
    print(f"Gate failures: {gate_fail_total} (out of {total * 3} gate checks)")
    print(f"Average judge score: {avg_score:.2f}/5" if avg_score is not None else "Average judge score: n/a")

    by_category: dict[str, list[dict]] = {}
    for row in results:
        by_category.setdefault(row["category"], []).append(row)

    for category, rows in sorted(by_category.items()):
        cat_scores = [row["judge_score"] for row in rows if row["judge_score"] is not None]
        cat_avg = sum(cat_scores) / len(cat_scores) if cat_scores else None
        avg_str = f"{cat_avg:.2f}/5" if cat_avg is not None else "n/a"
        cat_gate_fails = sum(row["gates_failed_count"] for row in rows)
        print(f"  {category:10s} n={len(rows):2d}  avg={avg_str}  gate_fails={cat_gate_fails}")
    print()


async def run() -> str:
    batch_id = str(uuid.uuid4())
    prompts = json.loads(config.EVAL_PROMPT_BANK_PATH.read_text())
    language = db.get_setting("language", config.DEFAULT_LANGUAGE)
    owner_name = db.get_setting("owner_name", config.DEFAULT_OWNER_NAME)

    results = []
    for entry in prompts:
        print(f"[{entry['id']}] {entry['prompt']}")
        try:
            reply = await _generate_reply(entry["prompt"], language, owner_name)
        except httpx.HTTPError as exc:
            print(f"  generation failed (is Ollama running?): {exc}")
            continue

        gates = eval_gates.run_gates(reply, language, owner_name)
        try:
            score, notes, judge_model = await _judge(entry["prompt"], reply)
        except (httpx.HTTPError, openrouter.OpenRouterUnavailable) as exc:
            print(f"  judge failed: {exc}")
            score, notes, judge_model = None, None, None

        record = db.create_eval_run(
            batch_id=batch_id,
            prompt_id=entry["id"],
            category=entry["category"],
            prompt=entry["prompt"],
            reply=reply,
            gates=gates,
            model=config.DEFAULT_MODEL,
            judge_score=score,
            judge_notes=notes,
            judge_model=judge_model,
        )
        results.append(record)

    _print_summary(batch_id, results)
    return batch_id


if __name__ == "__main__":
    asyncio.run(run())
