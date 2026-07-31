"""BYOB financial-answer benchmarks scored by the Cognitivo component-based rubric."""
import json, re

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput

SYSTEM = "detailed thinking off\nYou are the domain answer-synthesis model in a financial-market agent. Given a question, and any verified facts supplied above it, write ONE direct answer that states every value the question asks for, using only the supplied facts. Do not invent figures, do not describe your reasoning, and do not hedge -- output only the final answer."

JUDGE_SYSTEM = "detailed thinking off\n/no_think\nYou are a strict evaluation judge. Reply with only the JSON object, with no preamble and no reasoning outside the JSON."

RUBRIC = "You are grading a candidate ANSWER against the official Cognitivo Hackathon component-based rubric. Grade strictly.\n\nExpected facts, each with the points it is worth:\n{facts}\n\nTolerance note: {tolerance}\n\nCandidate answer to grade:\n{response}\n\nFor every expected fact, decide whether the candidate answer EXPLICITLY states it. Accept equivalent date formats, harmless numeric formatting differences, and synonyms that preserve meaning, honoring the tolerance note. A fact FAILS if it is missing, contradicted, or replaced with an invented value. A vague, hedged, or placeholder answer that does not commit to the value FAILS. Award each fact's points in full or not at all -- there is no partial credit within a single fact.\nSum the points of every satisfied fact.\nReply with only a JSON object, starting with {{ and ending with }}:\n{{\"score\": <sum of satisfied points, max {max_score}>, \"reasoning\": \"<which facts matched or failed, briefly>\"}}"

_SCORE_RE = re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)')


def _build_prompt(spec, response):
    facts = "\n".join(
        f"- [{f['points']} points] {f['expected_fact']}" for f in spec["expected_facts"]
    )
    return RUBRIC.format(facts=facts, tolerance=spec.get("tolerance_note") or "none",
                         response=response.strip() or "(the model returned nothing)",
                         max_score=spec["max_score"])


def _parse(text, max_score):
    """Pull {"score": ..., "reasoning": ...} out of the judge reply.

    A reply we cannot parse scores None, not 0. A reasoning model that overruns its
    token budget before emitting JSON is a grading failure, and silently recording it
    as a zero would understate the model under test.
    """
    match = None
    for match in _SCORE_RE.finditer(text):   # last match wins: models often restate
        pass
    if match is None:
        return {"score": None, "normalized": None, "reasoning": text[:500].strip(),
                "max_score": max_score, "parse_error": True}
    score = max(0.0, min(float(match.group(1)), float(max_score)))
    reason = ""
    rmatch = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if rmatch:
        reason = rmatch.group(1).encode().decode("unicode_escape", "replace")
    return {"score": score, "normalized": score / max_score if max_score else 0.0,
            "reasoning": reason, "max_score": max_score}


_JUDGE_ATTEMPTS = 3


def _make_judge_fn(spec, response):
    async def _run(judge_client):
        prompt = _build_prompt(spec, response)
        judged = {"score": None, "normalized": None, "max_score": spec["max_score"],
                  "error": "judge never ran"}
        for attempt in range(_JUDGE_ATTEMPTS):
            try:
                reply = await judge_client.chat(prompt=prompt, system=JUDGE_SYSTEM)
            except Exception as exc:
                judged = {"score": None, "normalized": None, "max_score": spec["max_score"],
                          "error": f"{type(exc).__name__}: {exc}"}
                continue
            judged = _parse(reply.content, spec["max_score"])
            judged["judge_model"] = getattr(reply, "model", None)
            judged["attempts"] = attempt + 1
            if not judged.get("parse_error"):
                break
        # reward=None leaves the sample's reward untouched rather than counting an
        # ungradeable sample as a zero in nel's own aggregate.
        return {"reward": judged["normalized"], "judge": judged}
    return _run


@scorer
def judge_defer(sample: ScorerInput) -> dict:
    # `_judge_fn` is popped by the eval loop and awaited with the configured judge
    # client; `needs_judge` is what triggers that branch at all.
    spec = json.loads(sample.target)
    return {"correct": False, "needs_judge": True,
            "extracted": sample.response[:500],
            "_judge_fn": _make_judge_fn(spec, sample.response)}


for _name, _ds in [("fin-qa-judge-ungrounded", "data/fin-qa-judge-ungrounded.jsonl"),
                   ("fin-qa-judge-grounded", "data/fin-qa-judge-grounded.jsonl")]:
    benchmark(name=_name, dataset=_ds, prompt="{prompt}",
              target_field="completion", system_prompt=SYSTEM)(judge_defer)
