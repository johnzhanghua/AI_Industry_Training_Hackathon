#!/usr/bin/env bash
# setup-dgx-spark.sh
#
# One-shot setup for the LLM Evaluation & Customization workshop on a clean
# DGX Spark. Replaces the three setup notebooks (00, 00b, 00c).
#
# Steps:
#   1. Verify GPU, Docker, and GPU-in-container access (CDI or nvidia runtime)
#   2. Prompt for your NGC API key and log in to nvcr.io
#   3. Install Miniforge if conda is missing, create the `llm-eval-workshop` env
#      (Python 3.12), install all deps, and register its Jupyter kernel
#   4. Launch the Nemotron Nano 9B NIM on localhost:8000 (or reuse a running one)
#   5. Wait for the NIM to become healthy
#   6. Pull the NeMo AutoModel container image (for nb 03)
#   7. Run live NIM, evaluator, Jupyter, and GPU-container smoke tests
#
# Python 3.12 is required because nemo-evaluator>=0.3 declares >=3.12; every other
# workshop package also supports 3.12, so one environment covers all notebooks.
#
# Re-running is safe — each step checks whether it is already done before acting.
#
# Usage:
#   ./setup-dgx-spark.sh
#
# Override defaults via env vars, e.g.:
#   NIM_PORT=8001 HEALTH_TIMEOUT_SECS=2400 ./setup-dgx-spark.sh

set -Eeuo pipefail

# ── Configuration (override via env vars) ─────────────────────────────────────
NIM_IMAGE="${NIM_IMAGE:-nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:latest}"
NIM_CONTAINER="${NIM_CONTAINER:-nemotron-nano}"
NIM_PORT="${NIM_PORT:-8000}"
NIM_CACHE="${NIM_CACHE:-$HOME/.cache/nim}"
AUTOMODEL_IMAGE="${AUTOMODEL_IMAGE:-nvcr.io/nvidia/nemo-automodel:26.06}"
WORKSHOP_ENV="${WORKSHOP_ENV:-llm-eval-workshop}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"   # where to install conda if missing
AUTO_INSTALL_CONDA="${AUTO_INSTALL_CONDA:-1}"        # set 0 to fail instead of installing
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-1800}"   # cold model download can be slow
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-70}"            # images + model caches from scratch
NIM_URL="http://localhost:${NIM_PORT}"
# ───────────────────────────────────────────────────────────────────────────────

# ── Output helpers ────────────────────────────────────────────────────────────
BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${BOLD}${GREEN}══ $* ${NC}"; }
log()  { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }
die()  { echo -e "  ${RED}✗ ERROR:${NC} $*" >&2; exit 1; }
trap 'die "unexpected failure at line ${LINENO} (exit $?). See the output above."' ERR

# ── Small utilities (keep the steps below free of duplicated logic) ───────────
have()              { command -v "$1" >/dev/null 2>&1; }
conda_has_env()     { "$CONDA_CMD" env list | awk '{print $1}' | grep -qx "$1"; }
container_running() { [[ -n "$(docker ps  -q  --filter "name=^${1}$")" ]]; }
container_exists()  { [[ -n "$(docker ps -aq  --filter "name=^${1}$")" ]]; }
nim_model_id()      {
  local response
  response=$(curl -sf --max-time 5 "${NIM_URL}/v1/models" 2>/dev/null) || return 1
  "$CONDA_CMD" run -n "$WORKSHOP_ENV" python -c \
    "import sys,json; print(json.loads(sys.argv[1])['data'][0]['id'])" "$response" 2>/dev/null
}
retry() {  # retry <attempts> <command...> — for flaky network operations
  local attempts=$1; shift
  local i
  for ((i = 1; i <= attempts; i++)); do
    "$@" && return 0
    warn "attempt ${i}/${attempts} failed: $*"
    sleep 5
  done
  return 1
}

# Locate a usable conda: $CONDA_EXE, then PATH, then the usual install prefixes
# (a conda that was installed but never `conda init`-ed is not on PATH).
find_conda() {
  local candidate
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    echo "$CONDA_EXE"; return 0
  fi
  if candidate=$(command -v conda 2>/dev/null); then
    echo "$candidate"; return 0
  fi
  for candidate in "$MINIFORGE_DIR" "$HOME/miniforge3" "$HOME/miniconda3" \
                   "$HOME/anaconda3" /opt/conda /opt/miniforge3; do
    [[ -x "${candidate}/bin/conda" ]] && { echo "${candidate}/bin/conda"; return 0; }
  done
  return 1
}

# Install Miniforge (conda-forge's minimal conda) non-interactively.
# Miniforge is used rather than Miniconda: it defaults to the conda-forge channel
# and carries no commercial licensing restrictions.
install_miniforge() {
  local arch installer url tmp
  arch="$(uname -m)"
  case "$arch" in
    aarch64|arm64) installer="Miniforge3-Linux-aarch64.sh" ;;
    x86_64)        installer="Miniforge3-Linux-x86_64.sh"  ;;
    *) die "No Miniforge build for architecture '$arch'. Install conda manually: https://github.com/conda-forge/miniforge" ;;
  esac
  url="https://github.com/conda-forge/miniforge/releases/latest/download/${installer}"

  have curl || die "curl is required to download Miniforge. Install it: sudo apt-get install -y curl"

  [[ -e "$MINIFORGE_DIR" ]] \
    && die "'$MINIFORGE_DIR' already exists but has no bin/conda. Remove it or set MINIFORGE_DIR to a different path."

  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064  # expand $tmp now, not at trap time
  trap "rm -rf '$tmp'" RETURN

  log "Downloading $installer ..."
  retry 3 curl -fsSL -o "${tmp}/${installer}" "$url" \
    || die "Failed to download Miniforge from $url — check your network."

  log "Installing Miniforge to $MINIFORGE_DIR (batch mode) ..."
  bash "${tmp}/${installer}" -b -p "$MINIFORGE_DIR" >/dev/null \
    || die "Miniforge installer failed."

  [[ -x "${MINIFORGE_DIR}/bin/conda" ]] \
    || die "Miniforge install finished but ${MINIFORGE_DIR}/bin/conda is missing."

  # Make `conda activate` work in the user's future interactive shells. This edits
  # ~/.bashrc; the script itself only ever uses `conda run`, so it is a convenience.
  if "${MINIFORGE_DIR}/bin/conda" init bash >/dev/null 2>&1; then
    log "Ran 'conda init bash' — open a new shell (or 'source ~/.bashrc') to use 'conda activate'"
  else
    warn "'conda init bash' failed — activate manually with: source ${MINIFORGE_DIR}/bin/activate"
  fi

  log "Miniforge installed"
}

# Run from the repo root so relative paths (composer/requirements.txt) resolve
# no matter where the script is invoked from.
cd "$(dirname "$(readlink -f "$0")")"
REQUIREMENTS="composer/requirements.txt"
[[ -f "$REQUIREMENTS" ]] \
  || die "$REQUIREMENTS not found — run this script from inside the repo checkout."

# ── Step 1: Verify environment ────────────────────────────────────────────────
step "Step 1 — Verify the DGX Spark environment"

have nvidia-smi || die "nvidia-smi not found — is this a DGX Spark with drivers installed?"
have docker     || die "Docker not found. Install Docker and add yourself to the docker group."
have curl       || die "curl not found. Install the standard DGX OS curl package, then re-run."
docker info >/dev/null 2>&1 \
  || die "Cannot talk to the Docker daemon. Is it running, and are you in the 'docker' group?"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || die "nvidia-smi failed to query the GPU."

# GPU-in-container access: prefer CDI (recommended on DGX Spark), else nvidia runtime.
# GPU_ARGS is an array so the flags never rely on fragile word-splitting.
if have nvidia-ctk && nvidia-ctk cdi list 2>/dev/null | grep -q "nvidia.com/gpu=all"; then
  GPU_ARGS=(--device nvidia.com/gpu=all)
  log "GPU access: CDI (--device nvidia.com/gpu=all)"
elif docker info 2>/dev/null | grep -qi "Runtimes:.*nvidia"; then
  GPU_ARGS=(--runtime=nvidia --gpus all)
  warn "CDI not found — using the legacy nvidia runtime (--runtime=nvidia --gpus all)"
  warn "To enable CDI: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
else
  die "No GPU-in-Docker mechanism found (tried CDI and nvidia runtime). See guide/DGX-Spark-Setup.md."
fi

[[ "$(uname -m)" == "aarch64" ]] \
  || warn "Expected aarch64, got $(uname -m). The DGX Spark NIM image is aarch64-only."

# A first run needs roughly 22 GB for the NIM image, 28 GB for AutoModel, and
# 14 GB for model caches. Warn rather than fail because some/all may be cached.
FREE_DISK_GB=$(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}')
if (( FREE_DISK_GB < MIN_FREE_DISK_GB )); then
  warn "Only ${FREE_DISK_GB} GB free; a fully uncached setup needs about ${MIN_FREE_DISK_GB} GB."
else
  log "Disk space: ${FREE_DISK_GB} GB free"
fi

# ── Step 2: NGC API key and registry login ────────────────────────────────────
step "Step 2 — NGC API key and nvcr.io login"

if [[ -z "${NGC_API_KEY:-}" ]]; then
  [[ -t 0 ]] || die "NGC_API_KEY is not set and no terminal is available to prompt. Export it and re-run."
  read -rsp "  Enter your NGC API key (nvapi-...): " NGC_API_KEY; echo
fi
export NGC_API_KEY

[[ "$NGC_API_KEY" == nvapi-* ]] \
  || die "That doesn't look like an NGC API key (expected it to start with 'nvapi-')."

if echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin >/dev/null 2>&1; then
  log "Logged in to nvcr.io"
else
  die "docker login to nvcr.io failed — check that the API key is valid and has NIM access."
fi

# ── Step 3: Single workshop conda env (Python 3.12) ───────────────────────────
step "Step 3 — Workshop conda env ($WORKSHOP_ENV, Python 3.12)"

if CONDA_CMD="$(find_conda)"; then
  log "Found conda: $CONDA_CMD"
elif [[ "$AUTO_INSTALL_CONDA" == "1" ]]; then
  warn "conda not found — installing Miniforge to $MINIFORGE_DIR"
  warn "(set AUTO_INSTALL_CONDA=0 to skip this and install conda yourself)"
  install_miniforge
  CONDA_CMD="${MINIFORGE_DIR}/bin/conda"
else
  die "conda not found and AUTO_INSTALL_CONDA=0. Install Miniforge: https://github.com/conda-forge/miniforge"
fi

if conda_has_env "$WORKSHOP_ENV"; then
  PY_VER=$("$CONDA_CMD" run -n "$WORKSHOP_ENV" python -c \
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
  if "$CONDA_CMD" run -n "$WORKSHOP_ENV" python -c \
       "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 12) else 1)" 2>/dev/null; then
    log "Env '$WORKSHOP_ENV' exists (Python $PY_VER) — reusing it"
  else
    die "Env '$WORKSHOP_ENV' is Python $PY_VER — nemo-evaluator needs 3.12+. Recreate it: conda env remove -n $WORKSHOP_ENV, then re-run."
  fi
else
  log "Creating conda env '$WORKSHOP_ENV' (Python 3.12)..."
  "$CONDA_CMD" create -y -n "$WORKSHOP_ENV" python=3.12
fi

log "Installing dependencies from $REQUIREMENTS (lm-eval, nemo-evaluator, datasets, ...)"
"$CONDA_CMD" run -n "$WORKSHOP_ENV" python -m pip install -q --upgrade pip
"$CONDA_CMD" run -n "$WORKSHOP_ENV" python -m pip install -q -r "$REQUIREMENTS"
log "Dependencies installed"

# Register the Jupyter kernel so it appears in JupyterLab.
"$CONDA_CMD" run -n "$WORKSHOP_ENV" python -m ipykernel install --user \
  --name "$WORKSHOP_ENV" --display-name "Python ($WORKSHOP_ENV)" >/dev/null
log "Jupyter kernel registered as 'Python ($WORKSHOP_ENV)'"

NEL_VER=$("$CONDA_CMD" run -n "$WORKSHOP_ENV" nel --version 2>/dev/null || echo "not found")
if [[ "$NEL_VER" == "not found" ]]; then
  die "the 'nel' CLI is missing after install — check the pip output above."
fi
log "NeMo Evaluator: $NEL_VER"

# ── Step 4: Launch the Nemotron Nano NIM ──────────────────────────────────────
step "Step 4 — Launch the Nemotron Nano NIM"

mkdir -p "$NIM_CACHE"

if container_running "$NIM_CONTAINER"; then
  log "Container '$NIM_CONTAINER' is already running — reusing it"
elif container_exists "$NIM_CONTAINER"; then
  log "Container '$NIM_CONTAINER' exists but is stopped — starting it"
  docker start "$NIM_CONTAINER" >/dev/null
else
  log "Launching '$NIM_CONTAINER' from $NIM_IMAGE"
  log "(First run downloads and optimises the model — this can take several minutes)"
  docker run -d --name "$NIM_CONTAINER" \
      "${GPU_ARGS[@]}" \
      --shm-size=16GB \
      -e NGC_API_KEY \
      -v "${NIM_CACHE}:/opt/nim/.cache" \
      -u "$(id -u)" \
      -p "${NIM_PORT}:8000" \
      "$NIM_IMAGE" >/dev/null
  log "Container launched in detached mode"
fi

# ── Step 5: Wait for the NIM to be healthy ────────────────────────────────────
step "Step 5 — Waiting for the NIM to be ready (timeout ${HEALTH_TIMEOUT_SECS}s)"

MODEL_ID="unknown"
echo -n "  Polling ${NIM_URL}/v1/health/ready"
deadline=$((SECONDS + HEALTH_TIMEOUT_SECS))
while (( SECONDS < deadline )); do
  # If the container died (e.g. bad image / wrong arch), fail fast with the logs.
  if ! container_running "$NIM_CONTAINER"; then
    echo ""
    docker logs --tail 20 "$NIM_CONTAINER" 2>&1 || true
    die "Container '$NIM_CONTAINER' exited before becoming healthy (see logs above)."
  fi
  if curl -sf --max-time 5 "${NIM_URL}/v1/health/ready" >/dev/null 2>&1; then
    echo " ready!"
    MODEL_ID=$(nim_model_id || echo "unknown")
    log "NIM is serving: $MODEL_ID"
    break
  fi
  echo -n "."
  sleep 10
done
if [[ "$MODEL_ID" == "unknown" ]] && ! curl -sf --max-time 5 "${NIM_URL}/v1/health/ready" >/dev/null 2>&1; then
  echo ""
  die "NIM did not become healthy within ${HEALTH_TIMEOUT_SECS}s. Check: docker logs $NIM_CONTAINER"
fi

# ── Step 6: Pull the NeMo AutoModel container ─────────────────────────────────
step "Step 6 — NeMo AutoModel container"

if docker image inspect "$AUTOMODEL_IMAGE" >/dev/null 2>&1; then
  log "Image '$AUTOMODEL_IMAGE' already present — skipping pull"
else
  log "Pulling $AUTOMODEL_IMAGE (large image — may take a while)"
  retry 3 docker pull "$AUTOMODEL_IMAGE" \
    || die "docker pull failed after 3 attempts — check your network and NGC access."
  log "AutoModel container image ready"
fi

# ── Step 7: Live smoke tests ──────────────────────────────────────────────────
step "Step 7 — Live smoke tests"

log "NIM model: ${MODEL_ID}"

"$CONDA_CMD" run -n "$WORKSHOP_ENV" jupyter lab --version >/dev/null \
  || die "JupyterLab is missing after dependency installation."
log "JupyterLab is installed"

CHAT_BODY='{"model":"'"${MODEL_ID}"'","messages":[{"role":"system","content":"/no_think"},{"role":"user","content":"Reply with exactly: setup-ok"}],"max_tokens":32}'
CHAT_RESPONSE=$(curl -sf --max-time 60 "${NIM_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" -d "$CHAT_BODY") \
  || die "NIM chat-completion request failed."
"$CONDA_CMD" run -n "$WORKSHOP_ENV" python -c \
  "import sys,json; assert json.loads(sys.argv[1])['choices'][0]['message']['content'].strip()" \
  "$CHAT_RESPONSE" >/dev/null \
  || die "NIM chat-completion response was invalid."
log "NIM chat-completion endpoint works"

"$CONDA_CMD" run -n "$WORKSHOP_ENV" nel list >/dev/null 2>&1 \
  || die "NeMo Evaluator 'nel list' failed."
log "NeMo Evaluator CLI works"

SMOKE_DIR=$(mktemp -d)
if "$CONDA_CMD" run -n "$WORKSHOP_ENV" nel eval run --bench gsm8k \
     --model-url "${NIM_URL}/v1/chat/completions" \
     --model-id "$MODEL_ID" --api-key dummy \
     --max-problems 2 --max-tokens 2048 --output-dir "$SMOKE_DIR" >/dev/null 2>&1; then
  rm -rf "$SMOKE_DIR"
  log "NeMo Evaluator → NIM → score smoke test works"
else
  warn "Evaluator smoke artifacts kept at: $SMOKE_DIR"
  die "NeMo Evaluator could not complete a two-problem GSM8K run."
fi

docker run --rm "${GPU_ARGS[@]}" --shm-size=8g "$AUTOMODEL_IMAGE" \
  python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" \
  >/dev/null 2>&1 \
  || die "AutoModel container cannot access the GPU."
log "AutoModel container GPU access works"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}✓ Setup complete!${NC}"
echo ""
echo "  NIM running on : ${NIM_URL}  (model: ${MODEL_ID})"
echo "  Workshop env   : conda activate ${WORKSHOP_ENV}  (Python 3.12)"
echo "                   → one environment for all notebooks (01, 02, 03)"
echo "  conda          : ${CONDA_CMD}"
if [[ "${CONDA_CMD}" == "${MINIFORGE_DIR}/bin/conda" ]] && ! have conda; then
echo "                   (just installed — open a new shell or run:"
echo "                    source ${MINIFORGE_DIR}/bin/activate)"
fi
echo "  AutoModel image: ${AUTOMODEL_IMAGE}  →  used automatically by notebook 03"
echo ""
echo "  Next step:"
echo "    ${CONDA_CMD} run -n ${WORKSHOP_ENV} jupyter lab --ip 0.0.0.0 --no-browser"
echo ""
echo "  In JupyterLab, use the 'Python (${WORKSHOP_ENV})' kernel and run:"
echo "    01-NIM-Evaluation.ipynb  →  02-Evaluator_notebook.ipynb  →  03-Customizer.ipynb"
echo "  (The NIM is already running — you can skip notebook 00.)"
echo ""
