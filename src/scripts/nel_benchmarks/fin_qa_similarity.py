"""BYOB financial-answer benchmarks (ungrounded & grounded) with BLEU/ROUGE/F1 scoring."""
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput
from rouge_score import rouge_scorer
import sacrebleu

_rs = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

# Llama-3.1-Nemotron-Nano-8B-v1 toggles its reasoning trace with a "detailed thinking
# on/off" system-prompt convention (different from the "/no_think" convention used by
# nemotron-nano-9b-v2 in the original template) -- check the model card if your build
# behaves differently. We turn it off so the reply is the final answer, not a trace.
SYSTEM = ("detailed thinking off\n"
          "You are the domain answer-synthesis model in a financial-market agent. Given a "
          "question, and any verified facts supplied above it, write ONE direct answer that "
          "states every value the question asks for, using only the supplied facts. Do not "
          "invent figures, do not describe your reasoning, and do not hedge -- output only "
          "the final answer.")

@scorer
def fin_scorer(sample: ScorerInput) -> dict:
    ref = str(sample.target).strip()
    hyp = sample.response.strip().split("\n")[0].strip().strip('"')
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
