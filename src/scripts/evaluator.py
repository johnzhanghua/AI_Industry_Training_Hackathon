#!/home/cognitivo_g03/miniforge/bin python3
"""Evaluating Nemotron on the Cognitivo Financial Market Signal Task.

Converted from 02-Evaluator_notebook.ipynb for headless/remote runs.

The submitted agent architecture is:
    question -> Qwen agent-brain plans and emits tool calls
             -> agent runtime executes query_data / retrieve
             -> tool results return to Qwen until reasoning is complete
             -> fine-tuned Nemotron synthesizes the final answer
             -> POST /query returns {"answer": "..."}

This script evaluates the last step only -- whether Nemotron, given a question
and the verified facts an upstream tool call would have produced, writes the
direct, fact-complete `answer` the hackathon actually grades. It compares an
ungrounded condition (question only, the Brief's "no tool use" failure mode)
against a grounded condition (question + verified facts, i.e. what Nemotron
actually receives from the agent runtime), using both cheap similarity metrics
(BLEU/ROUGE/F1) and a component-based LLM judge that mirrors the hackathon's
own grading rubric.

Prerequisites: a Nemotron NIM/OpenAI-compatible endpoint serving
nvidia/llama-3.1-nemotron-nano-8b-v1 (see 00-NIM-Setup-DGX-Spark.ipynb and the
Participant Package's Setup Instructions), and the NeMo Evaluator SDK
installed per 00b-NeMo-Evaluator-SDK-Setup-DGX-Spark.ipynb.

Run against a different endpoint (e.g. the fine-tuned `domain-ft` alias on
port 8001) by setting NIM_HOST / MODEL_ID env vars before running, to produce
the base-vs-fine-tuned comparison required for scoring.
"""


# %% Configuration
import os, sys, subprocess, json, glob, shutil, math
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
import pandas as pd
from pprint import pp

NIM_HOST = os.environ.get("NIM_HOST", "http://localhost:4000")
MODEL_ID = os.environ.get("MODEL_ID", "domain-ft")
# Grade with a different model than the one under test. domain-ft judging its own
# output scored obviously-wrong answers at 0.6-0.8; agent-brain is the larger model
# already being served, so it costs nothing extra to use it as the grader.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "agent-brain")
CHAT_URL = f"{NIM_HOST}/v1/chat/completions"
DATA_DIR = "data"
QUESTIONS_PATH = Path("../../Participant_Package/public_questions.jsonl")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
# Final artefacts are tagged with the model under test so a base run and a
# fine-tuned run can sit side by side instead of overwriting each other.
RUN_TAG = os.environ.get("RUN_TAG", MODEL_ID)

os.makedirs("nel_benchmarks", exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
_NEL = str(Path(sys.executable).parent / "nel")

SYSTEM = ("detailed thinking off\n"
          "You are the domain answer-synthesis model in a financial-market agent. Given a "
          "question, and any verified facts supplied above it, write ONE direct answer that "
          "states every value the question asks for, using only the supplied facts. Do not "
          "invent figures, do not describe your reasoning, and do not hedge -- output only "
          "the final answer.")

# NOTE: nel's built-in judge metric ignores the `rubric`/`max_score` keys in the YAML --
# engine/eval_loop.py calls judge_score() with no config=, so JudgeScoringConfig() defaults
# apply (a generic "score 1-5" rubric). To actually grade against the hackathon's component
# rubric we ship our own judge closure in scoring_details["_judge_fn"], which the eval loop
# prefers over its default path. This template is what that closure sends.
RUBRIC = (
    "You are grading a candidate ANSWER against the official Cognitivo Hackathon "
    "component-based rubric. Grade strictly.\n\n"
    "Expected facts, each with the points it is worth:\n{facts}\n\n"
    "Tolerance note: {tolerance}\n\n"
    "Candidate answer to grade:\n{response}\n\n"
    "For every expected fact, decide whether the candidate answer EXPLICITLY states it. "
    "Accept equivalent date formats, harmless numeric formatting differences, and synonyms "
    "that preserve meaning, honoring the tolerance note. A fact FAILS if it is missing, "
    "contradicted, or replaced with an invented value. A vague, hedged, or placeholder "
    "answer that does not commit to the value FAILS. Award each fact's points in full or "
    "not at all -- there is no partial credit within a single fact.\n"
    "Sum the points of every satisfied fact.\n"
    'Reply with only a JSON object, starting with {{ and ending with }}:\n'
    '{{"score": <sum of satisfied points, max {max_score}>, '
    '"reasoning": "<which facts matched or failed, briefly>"}}'
)

# Both reasoning-suppression conventions are included so this works whichever model
# grades: Nemotron reads "detailed thinking off", Qwen (agent-brain) reads "/no_think".
# Left in, Qwen's reasoning trace can run past the token limit before it emits any JSON,
# which the parser can only score as 0.
JUDGE_SYSTEM = ("detailed thinking off\n/no_think\n"
                "You are a strict evaluation judge. Reply with only the JSON object, "
                "with no preamble and no reasoning outside the JSON.")


def _nel(args):
    """Run the nel CLI from the active environment, streaming output live."""
    cmd = [_NEL] + args
    print("\u2192", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


def run_bench(bench, out_dir, max_problems=None, max_tokens=256, system_prompt=None):
    """Run one benchmark against the local model endpoint."""
    shutil.rmtree(out_dir, ignore_errors=True)
    args = ["eval", "run", "--bench", bench,
            "--model-url", CHAT_URL, "--model-id", MODEL_ID, "--api-key", "dummy",
            "--max-tokens", str(max_tokens), "--output-dir", out_dir]
    if max_problems is not None:
        args += ["--max-problems", str(max_problems)]
    if system_prompt:
        args += ["--system-prompt", system_prompt]
    _nel(args)
    return out_dir


def run_config(cfg_path, out_dir):
    """Run a full YAML config (used for LLM-as-a-judge).

    Clears out_dir first, as run_bench does: nel writes a timestamped eval-*.json per
    run, so a stale report from a previous run would otherwise linger and load_scores
    could pick it up instead of the current one.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    _nel(["eval", "run", cfg_path])


def load_results(out_dir):
    """Load per-sample records from results.jsonl."""
    files = glob.glob(os.path.join(out_dir, "**", "results.jsonl"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No results.jsonl under {out_dir}")
    with open(files[0]) as f:
        return [json.loads(line) for line in f]


def load_scores(out_dir):
    """Load aggregate scores from the eval report."""
    files = glob.glob(os.path.join(out_dir, "**", "eval-*.json"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No eval report under {out_dir}")
    with open(files[0]) as f:
        return json.load(f)["benchmark"]["scores"]


def mean_metrics(rows, keys):
    """Mean of named scoring_details fields across samples."""
    out = {}
    for key in keys:
        values = [row["scoring_details"].get(key) for row in rows
                  if isinstance(row.get("scoring_details"), dict)
                  and isinstance(row["scoring_details"].get(key), (int, float))]
        out[key] = sum(values) / len(values) if values else float("nan")
    return out


def read_questions(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def build_grounding_context(q):
    """Stand-in for the agent runtime's tool_trace results, taken from grading.components."""
    lines = ["Verified facts returned by upstream data-query/retrieval tools:"]
    for c in q["grading"]["components"]:
        lines.append(f"- {c['expected_fact']}")
    tolerance = q["grading"].get("tolerance_note")
    if tolerance:
        lines.append(f"(Tolerance: {tolerance})")
    return "\n".join(lines)


def make_prompt(q, grounded):
    if grounded:
        return f"{build_grounding_context(q)}\n\nQuestion: {q['prompt']}\nAnswer:"
    return f"Question: {q['prompt']}\nAnswer:"


def write_eval_files(qs):
    """Write ungrounded/grounded x similarity/judge JSONL files, and a prompt->meta lookup
    used to join results back to each question's id/difficulty."""
    sim_paths = {False: os.path.join(DATA_DIR, "fin-qa-eval-ungrounded.jsonl"),
                 True:  os.path.join(DATA_DIR, "fin-qa-eval-grounded.jsonl")}
    judge_paths = {False: os.path.join(DATA_DIR, "fin-qa-judge-ungrounded.jsonl"),
                   True:  os.path.join(DATA_DIR, "fin-qa-judge-grounded.jsonl")}
    prompt_meta = {}

    for grounded in (False, True):
        with open(sim_paths[grounded], "w") as fs, open(judge_paths[grounded], "w") as fj:
            for q in qs:
                prompt = make_prompt(q, grounded)
                fs.write(json.dumps({
                    "prompt": prompt, "completion": q["reference_answer"],
                    "category": "summarization",
                }) + "\n")
                grading_spec = {
                    "expected_facts": [
                        {"expected_fact": c["expected_fact"], "points": c["points"]}
                        for c in q["grading"]["components"]
                    ],
                    "max_score": q["grading"]["max_score"],
                    "tolerance_note": q["grading"].get("tolerance_note", ""),
                }
                fj.write(json.dumps({
                    "prompt": prompt, "completion": json.dumps(grading_spec),
                    "category": "summarization",
                }) + "\n")
                prompt_meta[prompt] = {
                    "id": q["id"], "difficulty": q["difficulty"],
                    "datasets": ",".join(q["datasets"]), "grounded": grounded,
                }
    written = list(sim_paths.values()) + list(judge_paths.values())
    print(f"Wrote {len(qs)} rows x 2 conditions to {written}")
    return prompt_meta


def write_similarity_benchmark_module():
    """Write the BYOB benchmark module (ungrounded & grounded) with BLEU/ROUGE/F1 scoring."""
    Path("nel_benchmarks/fin_qa_similarity.py").write_text('''"""BYOB financial-answer benchmarks (ungrounded & grounded) with BLEU/ROUGE/F1 scoring."""
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput
from rouge_score import rouge_scorer
import sacrebleu

_rs = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

# Llama-3.1-Nemotron-Nano-8B-v1 toggles its reasoning trace with a "detailed thinking
# on/off" system-prompt convention (different from the "/no_think" convention used by
# nemotron-nano-9b-v2 in the original template) -- check the model card if your build
# behaves differently. We turn it off so the reply is the final answer, not a trace.
SYSTEM = ("detailed thinking off\\n"
          "You are the domain answer-synthesis model in a financial-market agent. Given a "
          "question, and any verified facts supplied above it, write ONE direct answer that "
          "states every value the question asks for, using only the supplied facts. Do not "
          "invent figures, do not describe your reasoning, and do not hedge -- output only "
          "the final answer.")

@scorer
def fin_scorer(sample: ScorerInput) -> dict:
    ref = str(sample.target).strip()
    hyp = sample.response.strip().split("\\n")[0].strip().strip(\'"\')
    r = _rs.score(ref, hyp)
    bleu = sacrebleu.sentence_bleu(hyp, [ref]).score / 100.0
    rt, ht = set(ref.lower().split()), set(hyp.lower().split())
    inter = len(rt & ht)
    prec = inter / len(ht) if ht else 0.0
    rec = inter / len(rt) if rt else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"reward": r["rougeL"].fmeasure, "bleu": bleu,
            "rouge1": r["rouge1"].fmeasure, "rougeL": r["rougeL"].fmeasure, "f1": f1}

benchmark(name="fin-qa-ungrounded", dataset="data/fin-qa-eval-ungrounded.jsonl", prompt="{prompt}",
          target_field="completion", system_prompt=SYSTEM)(fin_scorer)
benchmark(name="fin-qa-grounded", dataset="data/fin-qa-eval-grounded.jsonl", prompt="{prompt}",
          target_field="completion", system_prompt=SYSTEM)(fin_scorer)
''')


def write_judge_benchmark_module():
    """Write the BYOB benchmark module that scores via the component-based LLM judge.

    Scoring runs through our own `_judge_fn` closure rather than nel's built-in judge
    metric, because that built-in path drops the configured rubric and max_score (see
    the note on RUBRIC above). The closure grades each answer against the question's
    own `expected_facts`/`points` and normalizes by that question's `max_score`.
    """
    Path("nel_benchmarks/fin_qa_judge.py").write_text('''"""BYOB financial-answer benchmarks scored by the Cognitivo component-based rubric."""
import json, re

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput

SYSTEM = ''' + json.dumps(SYSTEM) + '''

JUDGE_SYSTEM = ''' + json.dumps(JUDGE_SYSTEM) + '''

RUBRIC = ''' + json.dumps(RUBRIC) + '''

_SCORE_RE = re.compile(r'"score"\\s*:\\s*(-?\\d+(?:\\.\\d+)?)')


def _build_prompt(spec, response):
    facts = "\\n".join(
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
    rmatch = re.search(r'"reasoning"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)"', text)
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
''')


def make_judge_config(bench, out_dir, num_questions, max_problems=None):
    svc = {"type": "api", "url": CHAT_URL, "protocol": "chat_completions",
           "model": MODEL_ID, "api_key": "dummy"}
    judge_svc = dict(svc, model=JUDGE_MODEL)
    cfg = {
        "services": {"solver": svc, "judge": judge_svc},
        "sandboxes": {"none": {"type": "none"}},
        "benchmarks": [{
            "name": f"nel_benchmarks/fin_qa_judge.py:{bench}",
            "solver": {"type": "simple", "service": "solver", "system_prompt": "detailed thinking off"},
            "max_problems": max_problems if max_problems is not None else num_questions,
            "sandbox": {"type": "none"},
            # This metric block exists so the orchestrator builds a judge client from
            # `service: judge` (_find_judge_client). Its rubric/max_score would be
            # ignored, so they are deliberately not set here -- the real rubric and
            # per-question max_score live in the _judge_fn closure in fin_qa_judge.py.
            "scoring": {"include_defaults": True, "metrics": [{
                "type": "judge", "name": "quality", "service": "judge",
                "reference_free": False,
                "allow_self_judge": JUDGE_MODEL == MODEL_ID}]},
            "params": {}}],
        "cluster": {"type": "local"},
        "output": {"dir": out_dir, "timestamped": False},
    }
    path = f"nel_benchmarks/{bench}.yaml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    return path


SCORE_COLS = ["id", "difficulty", "condition", "score", "normalized"]


def judge_score_table(out_dir, condition, prompt_meta):
    rows = load_results(out_dir)
    recs = []
    for r in rows:
        meta = prompt_meta.get(r["prompt"], {})
        j = (r.get("scoring_details") or {}).get("judge") or {}
        recs.append({"id": meta.get("id"), "difficulty": meta.get("difficulty"),
                     "condition": condition, "score": j.get("score"),
                     "normalized": j.get("normalized"),
                     "datasets": meta.get("datasets"),
                     "max_score": j.get("max_score"),
                     "ungraded": bool(j.get("parse_error") or j.get("error")),
                     "answer": (r.get("model_response") or "").strip(),
                     "judge_reasoning": j.get("reasoning")})
    df = pd.DataFrame(recs)
    n_bad = int(df["ungraded"].sum()) if len(df) else 0
    if n_bad:
        print(f"WARNING: {n_bad}/{len(df)} {condition} samples could not be graded "
              f"(judge returned no parsable score); excluded from means: "
              f"{sorted(df.loc[df.ungraded, 'id'].dropna())}")
    return df


def _jsonable(value):
    """NaN/NumPy-safe scalar conversion so json.dump emits valid JSON."""
    if value is None:
        return None
    if hasattr(value, "item"):          # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _records(df):
    return [{k: _jsonable(v) for k, v in row.items()} for row in df.to_dict("records")]


def write_final_results(questions, sim_df, all_scores, overall, by_difficulty):
    """Write the run's headline numbers to results/final_results_<tag>.{json,md}.

    The JSON is the machine-readable artefact (diff two runs to show the
    fine-tune's effect); the Markdown is the human-readable summary to paste
    into the submission write-up.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    per_condition = {c: _jsonable(v) for c, v in overall.items()}
    judge_delta = None
    if "grounded" in per_condition and "ungrounded" in per_condition:
        if per_condition["grounded"] is not None and per_condition["ungrounded"] is not None:
            judge_delta = per_condition["grounded"] - per_condition["ungrounded"]

    payload = {
        "run": {
            "timestamp_utc": timestamp,
            "nim_host": NIM_HOST,
            "model_id": MODEL_ID,
            "judge_model": JUDGE_MODEL,
            "run_tag": RUN_TAG,
            "questions_path": str(QUESTIONS_PATH),
            "n_questions": len(questions),
        },
        "similarity": {
            "metrics": {metric: {col: _jsonable(sim_df.loc[metric, col])
                                 for col in sim_df.columns}
                        for metric in sim_df.index},
        },
        "judge": {
            "rubric": "cognitivo component-based; normalized per question by its own max_score",
            "ungraded_samples": int(all_scores["ungraded"].sum()),
            "overall_normalized": per_condition,
            "grounding_delta": judge_delta,
            "by_difficulty": _records(by_difficulty.reset_index()),
            "per_question": _records(all_scores),
        },
    }

    json_path = RESULTS_DIR / f"final_results_{RUN_TAG}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    md_path = RESULTS_DIR / f"final_results_{RUN_TAG}.md"
    md = [
        f"# Evaluation results -- `{MODEL_ID}`",
        "",
        f"- Run (UTC): {timestamp}",
        f"- Endpoint: {NIM_HOST}",
        f"- Questions: {QUESTIONS_PATH} ({len(questions)} public questions)",
        f"- Judge: {JUDGE_MODEL} (component-based rubric)",
        "",
        "## Similarity metrics (BLEU / ROUGE / F1)",
        "",
        "```", sim_df.round(4).to_string(), "```",
        "",
        "## Component-based LLM judge (normalized, 0-1)",
        "",
        "```", overall.round(3).to_string(), "```",
        "",
        f"Grounding delta: {judge_delta:+.3f}" if judge_delta is not None
        else "Grounding delta: n/a",
        "",
        "### By difficulty",
        "",
        "```", by_difficulty.round(3).to_string(), "```",
        "",
        "### Per question",
        "",
        "```", all_scores[SCORE_COLS].to_string(index=False), "```",
        "",
    ]
    md_path.write_text("\n".join(md))

    print(f"\nWrote final results to {json_path} and {md_path}")
    return json_path, md_path


def main():
    # --- Config / environment sanity checks ---------------------------------
    if not Path(_NEL).is_file():
        raise FileNotFoundError(
            f"NeMo Evaluator CLI not found at {_NEL}. Run ./setup-dgx-spark.sh first."
        )
    os.environ.update(NIM_HOST=NIM_HOST, MODEL_ID=MODEL_ID)
    print("NIM_HOST =", NIM_HOST, "| MODEL_ID =", MODEL_ID)
    print("data dir =", DATA_DIR)
    print("questions =", QUESTIONS_PATH, "(exists:", QUESTIONS_PATH.is_file(), ")")
    print("nel      =", _NEL)

    models_response = requests.get(f"{NIM_HOST}/v1/models", timeout=5)
    models_response.raise_for_status()
    served_models = [m["id"] for m in models_response.json()["data"]]
    print("Available models:", served_models)
    assert MODEL_ID in served_models, f"Expected {MODEL_ID} to be available, but endpoint serves {served_models}."
    # The endpoint may be a multi-model proxy; requests are routed by --model-id, so
    # availability is all that matters -- not which model happens to be listed first.
    print("Evaluating model:", MODEL_ID)
    _nel(["--version"])

    # --- Load public calibration questions -----------------------------------
    questions = read_questions(QUESTIONS_PATH)
    print(f"Loaded {len(questions)} public questions")

    overview = pd.DataFrame([
        {"id": q["id"], "difficulty": q["difficulty"], "datasets": ",".join(q["datasets"]),
         "scope": q["dataset_scope"], "components": len(q["grading"]["components"])}
        for q in questions
    ])
    print(overview.to_string(index=False))
    pp(questions[0])

    print("--- ungrounded ---")
    print(make_prompt(questions[0], grounded=False))
    print("\n--- grounded ---")
    print(make_prompt(questions[0], grounded=True))

    prompt_meta = write_eval_files(questions)

    # --- Similarity metrics (BLEU / ROUGE / F1) ------------------------------
    write_similarity_benchmark_module()

    run_bench("nel_benchmarks/fin_qa_similarity.py:fin-qa-ungrounded",
              "results/sim_ungrounded", max_problems=len(questions), max_tokens=256)
    run_bench("nel_benchmarks/fin_qa_similarity.py:fin-qa-grounded",
              "results/sim_grounded", max_problems=len(questions), max_tokens=256)

    keys = ["bleu", "rouge1", "rougeL", "f1"]
    ungrounded = mean_metrics(load_results("results/sim_ungrounded"), keys)
    grounded = mean_metrics(load_results("results/sim_grounded"), keys)

    sim_df = pd.DataFrame({"ungrounded": ungrounded, "grounded": grounded})
    sim_df["delta"] = sim_df["grounded"] - sim_df["ungrounded"]
    print(sim_df.round(4))

    # --- Component-based LLM judge -------------------------------------------
    write_judge_benchmark_module()

    cfg_ungrounded = make_judge_config("fin-qa-judge-ungrounded", "results/judge_ungrounded", len(questions))
    run_config(cfg_ungrounded, "results/judge_ungrounded")
    ungrounded_scores = judge_score_table("results/judge_ungrounded", "ungrounded", prompt_meta)
    print(f"Ungrounded judge mean: {ungrounded_scores['normalized'].mean():.3f}  "
          f"(n={len(ungrounded_scores)})\n")
    print(ungrounded_scores[SCORE_COLS].to_string(index=False))

    rows = load_results("results/judge_ungrounded")
    sample = rows[0]
    print("\nQuestion:", sample["prompt"].split("Question:")[-1][:120].strip(), "...")
    print("Model answer :", sample["model_response"].strip().splitlines()[0])
    _j = sample["scoring_details"]["judge"]
    print("Judge score  :", _j["score"], "/", _j.get("max_score"),
          "->", round(_j["normalized"], 2))
    print("Judge says   :", sample["scoring_details"]["judge"]["reasoning"][:200], "...")

    cfg_grounded = make_judge_config("fin-qa-judge-grounded", "results/judge_grounded", len(questions))
    run_config(cfg_grounded, "results/judge_grounded")
    grounded_scores = judge_score_table("results/judge_grounded", "grounded", prompt_meta)
    print(f"Grounded judge mean: {grounded_scores['normalized'].mean():.3f}  "
          f"(n={len(grounded_scores)})")

    all_scores = pd.concat([ungrounded_scores, grounded_scores], ignore_index=True)

    overall = all_scores.groupby("condition")["normalized"].mean().rename("mean_normalized")
    print("Overall:\n", overall.round(3), "\n")

    by_difficulty = all_scores.pivot_table(index="difficulty", columns="condition",
                                            values="normalized", aggfunc="mean")
    by_difficulty["delta"] = by_difficulty.get("grounded") - by_difficulty.get("ungrounded")
    print("By difficulty:\n", by_difficulty.round(3))

    write_final_results(questions, sim_df, all_scores, overall, by_difficulty)


if __name__ == "__main__":
    main()

