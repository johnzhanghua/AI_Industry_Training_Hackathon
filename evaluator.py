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
import os, sys, subprocess, json, glob, shutil
from pathlib import Path

import requests
import yaml
import pandas as pd
from pprint import pp

NIM_HOST = os.environ.get("NIM_HOST", "http://localhost:8000")
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/llama-3.1-nemotron-nano-8b-v1")
CHAT_URL = f"{NIM_HOST}/v1/chat/completions"
DATA_DIR = "data"
QUESTIONS_PATH = Path("../../Participant_Package/public_questions.jsonl")

os.makedirs("nel_benchmarks", exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
_NEL = str(Path(sys.executable).parent / "nel")

SYSTEM = ("detailed thinking off\n"
          "You are the domain answer-synthesis model in a financial-market agent. Given a "
          "question, and any verified facts supplied above it, write ONE direct answer that "
          "states every value the question asks for, using only the supplied facts. Do not "
          "invent figures, do not describe your reasoning, and do not hedge -- output only "
          "the final answer.")

RUBRIC = (
    "You are grading a candidate ANSWER against the official Cognitivo Hackathon "
    "component-based rubric.\n{instruction}\n"
    "Candidate answer to grade:\n{response}\n{reference_section}\n"
    "The reference above is a JSON object with an `expected_facts` list (each entry has an "
    "`expected_fact` and its `points`), a `max_score`, and a `tolerance_note`.\n"
    "For every entry in `expected_facts`, decide whether the candidate answer states that fact. "
    "Accept equivalent date formats, harmless numeric formatting differences, and synonyms that "
    "preserve meaning, honoring `tolerance_note`. A fact fails if it is missing, contradicted, or "
    "replaced with an invented value.\n"
    "Sum the `points` of every satisfied fact into a single total out of `max_score`.\n"
    'Reply with only a JSON object: {{"score": <total out of max_score>, '
    '"reasoning": "<which facts matched or failed, briefly>"}}'
)


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


def run_config(cfg_path):
    """Run a full YAML config (used for LLM-as-a-judge)."""
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
    """Write the BYOB benchmark module that defers scoring to the component-based LLM judge."""
    Path("nel_benchmarks/fin_qa_judge.py").write_text('''"""BYOB financial-answer benchmarks that defer scoring to a component-based LLM judge."""
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, needs_judge

SYSTEM = ("detailed thinking off\\n"
          "You are the domain answer-synthesis model in a financial-market agent. Given a "
          "question, and any verified facts supplied above it, write ONE direct answer that "
          "states every value the question asks for, using only the supplied facts. Do not "
          "invent figures, do not describe your reasoning, and do not hedge -- output only "
          "the final answer.")

@scorer
def judge_defer(sample: ScorerInput) -> dict:
    # Returning needs_judge(...) tells the eval loop to score this sample with the LLM judge
    # configured in the eval config (see make_judge_config).
    return needs_judge(sample)

for _name, _ds in [("fin-qa-judge-ungrounded", "data/fin-qa-judge-ungrounded.jsonl"),
                   ("fin-qa-judge-grounded", "data/fin-qa-judge-grounded.jsonl")]:
    benchmark(name=_name, dataset=_ds, prompt="{prompt}",
              target_field="completion", system_prompt=SYSTEM)(judge_defer)
''')


def make_judge_config(bench, out_dir, num_questions, max_problems=None):
    svc = {"type": "api", "url": CHAT_URL, "protocol": "chat_completions",
           "model": MODEL_ID, "api_key": "dummy"}
    cfg = {
        "services": {"solver": svc, "judge": dict(svc)},
        "sandboxes": {"none": {"type": "none"}},
        "benchmarks": [{
            "name": f"nel_benchmarks/fin_qa_judge.py:{bench}",
            "solver": {"type": "simple", "service": "solver", "system_prompt": "detailed thinking off"},
            "max_problems": max_problems if max_problems is not None else num_questions,
            "sandbox": {"type": "none"},
            "scoring": {"include_defaults": True, "metrics": [{
                "type": "judge", "name": "quality", "service": "judge",
                "reference_free": False, "allow_self_judge": True, "max_score": 10,
                "system_prompt": "detailed thinking off\nReply only with the JSON object.",
                "rubric": RUBRIC}]},
            "params": {}}],
        "cluster": {"type": "local"},
        "output": {"dir": out_dir, "timestamped": False},
    }
    path = f"nel_benchmarks/{bench}.yaml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    return path


def judge_score_table(out_dir, condition, prompt_meta):
    rows = load_results(out_dir)
    recs = []
    for r in rows:
        meta = prompt_meta.get(r["prompt"], {})
        j = (r.get("scoring_details") or {}).get("judge") or {}
        recs.append({"id": meta.get("id"), "difficulty": meta.get("difficulty"),
                     "condition": condition, "score": j.get("score"),
                     "normalized": j.get("normalized")})
    return pd.DataFrame(recs)


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
    served_model = models_response.json()["data"][0]["id"]
    print("Serving model:", served_model)
    assert served_model == MODEL_ID, f"Expected {MODEL_ID}, but the endpoint serves {served_model}."
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
    run_config(cfg_ungrounded)
    ungrounded_scores = judge_score_table("results/judge_ungrounded", "ungrounded", prompt_meta)
    print(f"Ungrounded judge mean: {ungrounded_scores['normalized'].mean():.3f}  "
          f"(n={len(ungrounded_scores)})\n")
    print(ungrounded_scores.to_string(index=False))

    rows = load_results("results/judge_ungrounded")
    sample = rows[0]
    print("\nQuestion:", sample["prompt"].split("Question:")[-1][:120].strip(), "...")
    print("Model answer :", sample["model_response"].strip().splitlines()[0])
    print("Judge score  :", sample["scoring_details"]["judge"]["score"],
          "/10 ->", round(sample["scoring_details"]["judge"]["normalized"], 2))
    print("Judge says   :", sample["scoring_details"]["judge"]["reasoning"][:200], "...")

    cfg_grounded = make_judge_config("fin-qa-judge-grounded", "results/judge_grounded", len(questions))
    run_config(cfg_grounded)
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


if __name__ == "__main__":
    main()

