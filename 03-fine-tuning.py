#!/usr/bin/env python3
"""LoRA fine-tuning of Nemotron-8B for financial answer synthesis on DGX Spark.

Script form of `03-Customizer.ipynb` — same pipeline, runnable headless under
tmux so it survives an SSH disconnect.

Stages:
    data     build data/{train,val,test}.jsonl from `data set/` and inspect it
    recipe   write the AutoModel YAML recipe and the in-container launcher
    train    stop the NIM, run LoRA training, restart the NIM
    curves   summarise checkpoints/{training,validation}.jsonl
    compare  generate base vs LoRA answers on held-out rows and score them

Usage:
    # everything, from the repository root
    python3 03-fine-tuning.py

    # long run under tmux
    tmux new-session -s finetune \\
      "python3 03-fine-tuning.py 2>&1 | tee logs/finetune.log"

    # single stage
    python3 03-fine-tuning.py --stage curves

Hyperparameter defaults follow the handout's confirmed baseline
(Participant_Package/handout/01_training_guide.md). Two are hard constraints,
not preferences: lr 1e-4 spikes the loss at warmup, and sequences longer than
512 OOM on a single node.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STAGES = ["data", "recipe", "train", "curves", "compare"]

# Where a local --base-model directory is mounted inside the container.
MODEL_MOUNT = "/models"

# Weights are pre-staged under MODELS_DIR on each node, so training never
# reaches for huggingface.co. --base-model names one of them.
MODELS_DIR = "~/local-llm-setup/models"
DEFAULT_BASE_MODEL = "Llama-3.1-Nemotron-Nano-8B-v1"

# Written into the recipe and read back by the verification step.
LORA_RANK = 32
LORA_ALPHA = 32
LEARNING_RATE = "5.0e-5"
SEQ_LENGTH = 512
MAX_STEPS = 100
CKPT_EVERY = 20
LOCAL_BATCH = 2
GLOBAL_BATCH = 8


def log(message: str) -> None:
    print(f"\n=== {message}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("→ " + " ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd, check=True, **kwargs)
    except FileNotFoundError:
        raise SystemExit(f"{cmd[0]} not found on PATH") from None


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, **kwargs)


# ---------------------------------------------------------------------------
# Stage: data
# ---------------------------------------------------------------------------


def stage_data(args) -> dict[str, Path]:
    log("Stage 1 — build and inspect the training data")
    out_dir = Path(args.out_dir)
    splits = {name: out_dir / f"{name}.jsonl" for name in ("train", "val", "test")}

    if args.rebuild_data or not all(p.is_file() for p in splits.values()):
        prepare = Path(__file__).with_name("prepare_finance_data.py")
        if not prepare.is_file():
            raise SystemExit(f"data preparation script not found: {prepare}")
        run(
            [
                sys.executable,
                str(prepare),
                "--data-root", args.data_root,
                "--out-dir", str(out_dir),
                "--afr-share", str(args.afr_share),
                "--total", str(args.examples),
            ]
        )
    else:
        print("splits already present — pass --rebuild-data to regenerate")

    for name, path in splits.items():
        rows = read_jsonl(path)
        headlines = sum(1 for r in rows if "Write the headline" in r["prompt"])
        print(
            f"{name:6s}: {len(rows):6,d} rows | "
            f"{len(rows) - headlines:5,d} analytic / {headlines:5,d} headline"
        )

    example = next(
        r
        for r in read_jsonl(splits["train"])
        if "Write the headline" not in r["prompt"]
    )
    print("\nExample analytic pair\n" + "-" * 60)
    print(example["prompt"])
    print("-" * 60)
    print("completion:", example["completion"])
    return splits


# ---------------------------------------------------------------------------
# Stage: recipe
# ---------------------------------------------------------------------------

LAUNCHER = '''"""DGX Spark launcher for NeMo AutoModel 26.06.

Patches one transformers-5.x incompatibility: AutoModel's initialize_weights()
calls LlamaRMSNorm.reset_parameters(), which transformers 5.x removed. We re-add
it (RMSNorm init = weights set to 1), then run the stock entrypoint unchanged.
"""
import torch, runpy
from transformers.models.llama.modeling_llama import LlamaRMSNorm

def _reset(self):
    with torch.no_grad():
        self.weight.fill_(1.0)

if not hasattr(LlamaRMSNorm, "reset_parameters"):
    LlamaRMSNorm.reset_parameters = _reset
    print("[spark-patch] added LlamaRMSNorm.reset_parameters")

runpy.run_path("/opt/Automodel/examples/llm_finetune/finetune.py", run_name="__main__")
'''


def recipe_yaml(args) -> str:
    model_path = container_model_path(args)
    dataset_block = """  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /workspace/{path}
  split: {split}
  column_mapping:
    question: prompt
    answer: completion
  seq_length: {seq}
  answer_only_loss_mask: true
  padding: do_not_pad
  truncation: longest_first"""

    return f"""# LoRA (PEFT) fine-tuning of {model_path} for financial answer synthesis
# over the AFR / ASX / RBA corpus, on a single DGX Spark GPU.
#
# Generated by src/scripts/03-fine-tuning.py — edit that, not this file.
#
# Hyperparameters follow the handout's confirmed baseline:
#   LORA_RANK {LORA_RANK} | LR {LEARNING_RATE} | MAX_SEQ_LEN {SEQ_LENGTH}
#   MAX_STEPS {args.steps} | CKPT_EVERY {CKPT_EVERY}
#   BATCH_SIZE {LOCAL_BATCH} x GRAD_ACCUM {GLOBAL_BATCH // LOCAL_BATCH} = effective {GLOBAL_BATCH}
# lr 1e-4 spikes the loss at warmup, and seq > 512 OOMs on one node — do not raise either.
recipe: TrainFinetuneRecipeForNextTokenPrediction

step_scheduler:
  global_batch_size: {GLOBAL_BATCH}
  local_batch_size: {LOCAL_BATCH}
  num_epochs: 1
  max_steps: {args.steps}
  val_every_steps: {CKPT_EVERY}
  ckpt_every_steps: {CKPT_EVERY}

dist_env:
  backend: nccl
  timeout_minutes: 20

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111
  ranked: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: {model_path}
  torch_dtype: bf16

checkpoint:
  enabled: true
  checkpoint_dir: /workspace/{args.checkpoint_dir}
  model_save_format: safetensors
  save_consolidated: true

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: '*_proj'
  dim: {LORA_RANK}
  alpha: {LORA_ALPHA}
  dropout: 0.1
  use_triton: true

distributed:
  # AutoModel instantiates this block like any other, so it needs a _target_ —
  # a bare `strategy: fsdp2` fails with "No _target_ found to instantiate".
  # dp_size must be `null`, not `none`: YAML reads `none` as the string "none".
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: null
  tp_size: 1
  cp_size: 1
  sequence_parallel: false

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataset:
{dataset_block.format(path=f"{args.out_dir}/train.jsonl", split="train", seq=SEQ_LENGTH)}

validation_dataset:
{dataset_block.format(path=f"{args.out_dir}/val.jsonl", split="validation", seq=SEQ_LENGTH)}

packed_sequence:
  packed_sequence_size: 0

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: true

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

optimizer:
  _target_: torch.optim.Adam
  lr: {LEARNING_RATE}
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1.0e-8
"""


def stage_recipe(args) -> Path:
    log("Stage 2 — write the training recipe and launcher")
    recipe_dir = Path(args.recipe).parent
    recipe_dir.mkdir(parents=True, exist_ok=True)

    Path(args.recipe).write_text(recipe_yaml(args))
    print("wrote", args.recipe)

    launcher = recipe_dir / "spark_finetune.py"
    launcher.write_text(LAUNCHER)
    print("wrote", launcher)

    # Verify the recipe parses and agrees with the CLI arguments, so a typo
    # fails here rather than 20 minutes into a container run.
    try:
        import yaml
    except ImportError:
        print("note: pyyaml not installed — skipping recipe verification")
        return Path(args.recipe)

    with open(args.recipe, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    recipe_model = cfg["model"]["pretrained_model_name_or_path"]
    if recipe_model != container_model_path(args):
        raise SystemExit(
            f"recipe trains {recipe_model} but --base-model is {args.base_model}"
        )
    sched = cfg["step_scheduler"]
    print(
        f"Recipe OK — model: {recipe_model}\n"
        f"LoRA rank {cfg['peft']['dim']} alpha {cfg['peft']['alpha']} "
        f"| lr {cfg['optimizer']['lr']} | seq {cfg['dataset']['seq_length']}\n"
        f"steps {sched['max_steps']} "
        f"| batch {sched['local_batch_size']} x accum "
        f"{sched['global_batch_size'] // sched['local_batch_size']} "
        f"| ckpt every {sched['ckpt_every_steps']}\n"
        f"train data {cfg['dataset']['path_or_dataset_id']}"
    )
    return Path(args.recipe)


# ---------------------------------------------------------------------------
# NIM lifecycle — training needs the GPU memory the NIM reserves (~84 GB)
# ---------------------------------------------------------------------------


def nim_running(container: str) -> bool:
    try:
        out = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name=^{container}$"],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        raise SystemExit("docker not found on PATH — this stage runs in a container") from None
    return bool(out.stdout.strip())


def stop_nim(container: str) -> bool:
    was_running = nim_running(container)
    if was_running:
        run(["docker", "stop", container], stdout=subprocess.DEVNULL)
        print(f"Stopped {container} to free the GPU.")
    else:
        print(f"{container} was already stopped; it will be left stopped.")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv"],
        check=False,
    )
    return was_running


def start_nim(container: str, host: str, timeout_secs: int = 600) -> None:
    run(["docker", "start", container], stdout=subprocess.DEVNULL)
    try:
        import requests
    except ImportError:
        print("note: requests not installed — not waiting for readiness")
        return

    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{host}/v1/health/ready", timeout=3).status_code == 200:
                print("NIM ready.")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(
        f"NIM did not restart within {timeout_secs // 60} minutes. "
        f"Run: docker logs {container}"
    )


# ---------------------------------------------------------------------------
# Stage: train
# ---------------------------------------------------------------------------


def local_model_dir(args) -> Path | None:
    """Host directory holding the base weights, or None if there isn't one.

    --base-model is a name under --models-dir. An absolute path still resolves,
    because `Path("/a") / "/b"` is "/b".
    """
    path = Path(args.models_dir).expanduser() / Path(args.base_model).expanduser()
    return path.resolve() if path.is_dir() else None


def container_model_path(args) -> str:
    """How the base model is addressed *inside* the container."""
    local = local_model_dir(args)
    if local is None:
        raise SystemExit(
            f"base model {args.base_model!r} not found under "
            f"{Path(args.models_dir).expanduser()}"
        )
    return f"{MODEL_MOUNT}/{local.name}"


def docker_cmd(args, script: str, extra_env: dict[str, str] | None = None) -> list[str]:
    repo = str(Path.cwd())
    hf_cache = str(Path.home() / ".cache" / "huggingface")
    cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--shm-size", "8g",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-v", f"{repo}:/workspace", "-w", "/workspace",
        "-v", f"{hf_cache}:/root/.cache/huggingface",
    ]
    # The base model lives outside the repo, so it needs its own mount. Offline
    # mode then makes a silent fall-through to the network impossible rather
    # than merely unlikely — these nodes have no outbound connectivity.
    local = local_model_dir(args)
    if local is None:
        raise SystemExit(
            f"base model {args.base_model!r} not found under "
            f"{Path(args.models_dir).expanduser()}"
        )
    cmd += [
        "-v", f"{local}:{MODEL_MOUNT}/{local.name}:ro",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "TRANSFORMERS_OFFLINE=1",
    ]
    for key, value in (extra_env or {}).items():
        cmd += ["-e", f"{key}={value}"]
    return cmd + [args.image, "python3", script]


def stage_train(args) -> None:
    log("Stage 3 — LoRA fine-tuning")
    was_running = stop_nim(args.nim_container)

    recipe_dir = Path(args.recipe).parent
    cmd = docker_cmd(args, f"/workspace/{recipe_dir}/spark_finetune.py") + [
        "-c", f"/workspace/{args.recipe}",
        "--nproc-per-node", "1",
    ]
    try:
        run(cmd)
    finally:
        # Always hand the GPU back, including on Ctrl-C or a failed run.
        if was_running:
            print("\nRestarting the NIM…", flush=True)
            start_nim(args.nim_container, args.nim_host)


# ---------------------------------------------------------------------------
# Stage: curves
# ---------------------------------------------------------------------------


def stage_curves(args) -> None:
    log("Stage 4 — training curves")
    ckpt = Path(args.checkpoint_dir)
    train_log, val_log = ckpt / "training.jsonl", ckpt / "validation.jsonl"
    if not train_log.is_file():
        raise SystemExit(f"{train_log} not found — has training run?")

    train = read_jsonl(train_log)
    val = read_jsonl(val_log) if val_log.is_file() else []
    if not train:
        raise SystemExit(f"{train_log} is empty — training logged no steps")

    print(f"steps logged: {len(train)} | final train loss: {train[-1]['loss']:.4f}")
    if val:
        best = min(val, key=lambda v: v["val_loss"])
        print(f"best val loss: {best['val_loss']:.4f} at step {best['step']}")
        for entry in val:
            print(f"  step {entry['step']:>4}: val_loss {entry['val_loss']:.4f}")

    for link in (ckpt / "LATEST", ckpt / "LOWEST_VAL"):
        if link.is_symlink():
            print(f"{link} -> {os.readlink(link)}")

    # A PNG is written when matplotlib is available; the numbers above are the
    # authoritative output either way.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(9, 5))
    plt.plot([t["step"] for t in train], [t["loss"] for t in train],
             label="train loss", alpha=0.7)
    if val:
        plt.scatter([v["step"] for v in val], [v["val_loss"] for v in val],
                    color="red", label="val loss", zorder=5)
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("LoRA fine-tuning loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = Path("logs") / "loss_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


# ---------------------------------------------------------------------------
# Stage: compare
# ---------------------------------------------------------------------------

COMPARE_INFER = '''"""Generate base and LoRA answers for held-out rows, in-container."""
import os, json, glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.environ["BASE_MODEL"]
CKPT = os.environ.get("CKPT_DIR", "/workspace/checkpoints")
PROMPTS_PATH = os.environ["PROMPTS_PATH"]
OUT_PATH = os.environ["OUT_PATH"]

candidates = [p for p in [os.path.join(CKPT, "LOWEST_VAL", "model"),
                          os.path.join(CKPT, "LATEST", "model")] if os.path.isdir(p)]
if not candidates:
    candidates = sorted(glob.glob(os.path.join(CKPT, "**", "model"), recursive=True))
if not candidates:
    raise FileNotFoundError(f"No LoRA adapter found under {CKPT}")
adapter = candidates[0]
print("Base:", BASE, "| Adapter:", adapter, flush=True)

tokenizer = AutoTokenizer.from_pretrained(adapter)
base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="cuda"
).eval()
model = PeftModel.from_pretrained(base, adapter).eval()

# Must match synthesis_node in src/agent.py verbatim: the comparison is only
# meaningful if it scores the model under the prompt it serves behind.
SYSTEM = ("You are an expert financial analysis synthesizer. Generate a direct, grounded answer "
          "to the question based strictly on the provided context (which is formatted as JSON blocks).")
with open(PROMPTS_PATH) as handle:
    prompts = json.load(handle)

def generate(active_model, label):
    answers = []
    for index, prompt in enumerate(prompts, 1):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_ids = encoded["input_ids"].to("cuda")
        with torch.no_grad():
            output = active_model.generate(
                input_ids, max_length=input_ids.shape[1] + 96, do_sample=False
            )
        text = tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        # References are a single line; take the first non-empty line so a base
        # model that opens with a blank line is not scored as an empty answer.
        lines = [line for line in text.splitlines() if line.strip()]
        answers.append(lines[0].strip() if lines else "")
        if index % 10 == 0 or index == len(prompts):
            print(f"{label}: {index}/{len(prompts)}", flush=True)
    return answers

fine_tuned = generate(model, "LoRA")
with model.disable_adapter():
    base_answers = generate(model, "base")

with open(OUT_PATH, "w") as handle:
    json.dump({"base": base_answers, "ft": fine_tuned}, handle)
print("wrote", OUT_PATH, flush=True)
'''


def token_f1(reference: str, prediction: str) -> float:
    ref = set(reference.lower().split())
    pred = set(prediction.lower().split())
    overlap = len(ref & pred)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(ref) if ref else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def stage_compare(args, splits: dict[str, Path]) -> None:
    log("Stage 5 — base vs LoRA on held-out rows")
    recipe_dir = Path(args.recipe).parent

    if not splits["test"].is_file():
        raise SystemExit(f"{splits['test']} not found — run the `data` stage first")

    rows = read_jsonl(splits["test"])[: args.compare_n]
    references = [r["completion"] for r in rows]
    prompts_path = recipe_dir / "compare_prompts.json"
    out_path = recipe_dir / "compare_out.json"
    write_json(prompts_path, [r["prompt"] for r in rows])
    print(f"prepared {len(rows)} held-out prompts (of the full test split)")

    (recipe_dir / "compare_infer.py").write_text(COMPARE_INFER)

    was_running = stop_nim(args.nim_container)
    try:
        run(
            docker_cmd(
                args,
                f"/workspace/{recipe_dir}/compare_infer.py",
                {
                    "BASE_MODEL": container_model_path(args),
                    "CKPT_DIR": f"/workspace/{args.checkpoint_dir}",
                    "PROMPTS_PATH": f"/workspace/{prompts_path}",
                    "OUT_PATH": f"/workspace/{out_path}",
                },
            )
        )
    finally:
        if was_running:
            print("\nRestarting the NIM…", flush=True)
            start_nim(args.nim_container, args.nim_host)

    with open(out_path, encoding="utf-8") as handle:
        out = json.load(handle)
    if not (len(out["base"]) == len(out["ft"]) == len(references)):
        raise SystemExit("comparison output length does not match the references")

    try:
        from rouge_score import rouge_scorer
        import sacrebleu
    except ImportError:
        print("note: rouge-score/sacrebleu not installed — printing token F1 only")
        rouge_scorer = sacrebleu = None

    def metrics(predictions: list[str]) -> dict[str, float]:
        scores = {
            "token_f1": sum(token_f1(r, p) for r, p in zip(references, predictions))
            / len(references)
        }
        if rouge_scorer is not None:
            scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
            rows_ = [scorer.score(r, p) for r, p in zip(references, predictions)]
            scores["rouge1"] = sum(s["rouge1"].fmeasure for s in rows_) / len(rows_)
            scores["rougeL"] = sum(s["rougeL"].fmeasure for s in rows_) / len(rows_)
            scores["bleu"] = sacrebleu.corpus_bleu(predictions, [references]).score / 100.0
        return scores

    base_scores, lora_scores = metrics(out["base"]), metrics(out["ft"])

    def question_of(prompt: str) -> str:
        for line in prompt.splitlines():
            if line.startswith("Question:"):
                return line[len("Question:"):].strip()
        return prompt.splitlines()[-1]

    print("\nFirst 5 held-out rows\n" + "=" * 72)
    for row, base_answer, lora_answer in list(
        zip(rows, out["base"], out["ft"])
    )[:5]:
        print("Q:", question_of(row["prompt"])[:100])
        print("  reference:", row["completion"][:110])
        print("  base     :", base_answer[:110])
        print("  LoRA     :", lora_answer[:110])
        print("-" * 72)

    print(f"\n{'metric':<10} {'base':>9} {'LoRA':>9} {'delta':>9}")
    for key in base_scores:
        delta = lora_scores[key] - base_scores[key]
        print(f"{key:<10} {base_scores[key]:>9.4f} {lora_scores[key]:>9.4f} {delta:>+9.4f}")

    report = Path("logs") / "compare_metrics.json"
    write_json(
        report,
        {"n": len(rows), "base": base_scores, "fine_tuned": lora_scores},
        indent=2,
    )
    print("\nwrote", report)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", choices=STAGES, action="append",
                    help="run only these stages (repeatable); default is all")
    ap.add_argument("--data-root", default="data set")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--afr-share", type=float, default=0.5)
    ap.add_argument("--examples", type=int, default=60_000,
                    help="target training pairs across all splits (48k/6k/6k)")
    ap.add_argument("--rebuild-data", action="store_true")
    ap.add_argument("--recipe", default="automodel_recipes/finance_lora.yaml")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                    help="host-side checkpoint directory, relative to the repository root")
    ap.add_argument("--models-dir", default=MODELS_DIR,
                    help="host directory the pre-staged model weights live under")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help="model name under --models-dir; mounted read-only at "
                         f"{MODEL_MOUNT}/<name>")
    ap.add_argument("--image", default="nvcr.io/nvidia/nemo:25.09")
    ap.add_argument("--nim-container", default="nemo")
    ap.add_argument("--nim-host", default="http://localhost:8001")
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    ap.add_argument("--compare-n", type=int, default=50,
                    help="held-out rows to compare (the guide reports on 50)")
    args = ap.parse_args()

    stages = args.stage or STAGES

    if not Path(args.data_root).is_dir() and "data" in stages:
        raise SystemExit(
            f"data root not found: {args.data_root!r} — run from the repository root"
        )

    # The base model must be on disk: these nodes have no outbound network, so a
    # name that resolves to nothing can only fail later at load time.
    models_root = Path(args.models_dir).expanduser()
    local = local_model_dir(args)
    if not local:
        available = (
            sorted(p.name for p in models_root.iterdir() if p.is_dir())
            if models_root.is_dir() else []
        )
        raise SystemExit(
            f"base model {args.base_model!r} not found under {models_root}\n"
            + (f"available: {', '.join(available)}" if available
               else f"{models_root} holds no model directories")
        )
    missing = [f for f in ("config.json", "model.safetensors.index.json")
               if not (local / f).is_file()]
    if missing:
        raise SystemExit(
            f"{local} is not a usable model directory — missing {', '.join(missing)}"
        )

    print("base model :", local, f"→ {container_model_path(args)}")
    print("image      :", args.image)
    print("recipe     :", args.recipe)
    print("stages     :", ", ".join(stages))

    # Later stages need the split paths even when `data` is skipped.
    splits = {
        name: Path(args.out_dir) / f"{name}.jsonl"
        for name in ("train", "val", "test")
    }

    if "data" in stages:
        splits = stage_data(args)
    if "recipe" in stages:
        stage_recipe(args)
    if "train" in stages:
        stage_train(args)
    if "curves" in stages:
        stage_curves(args)
    if "compare" in stages:
        stage_compare(args, splits)

    log("Done")


if __name__ == "__main__":
    main()
