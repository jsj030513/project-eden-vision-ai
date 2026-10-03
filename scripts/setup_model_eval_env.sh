#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
  echo "Usage: bash scripts/setup_model_eval_env.sh [new-venv-path]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EVAL_VENV="${1:-${MODEL_EVAL_VENV:-/Users/dev/.venvs/eden-vision-ai-model-eval}}"
LOCK="$PROJECT_DIR/requirements/model-eval-lock.txt"

# Never reuse, replace, delete, or repair an existing environment implicitly.
if [[ -e "$EVAL_VENV" || -L "$EVAL_VENV" ]]; then
  echo "Refusing existing path: $EVAL_VENV. Choose a new path; nothing was changed." >&2
  exit 1
fi
command -v brew >/dev/null || { echo 'Homebrew is required.' >&2; exit 1; }
PY312="$(brew --prefix python@3.12)/bin/python3.12"
"$PY312" -I - <<'PY'
import platform, sys
if sys.version_info[:2] != (3, 12) or platform.machine() != 'arm64':
    raise SystemExit('Homebrew Python 3.12 / arm64 is required')
print('Base Python:', platform.python_version(), platform.machine())
if platform.python_version() != '3.12.14':
    print('WARNING: snapshot used Python 3.12.14; this patch version needs revalidation.')
PY

# mkdir exclusively reserves the target; failed installs are left for inspection.
mkdir -p -- "$(dirname -- "$EVAL_VENV")"
mkdir -- "$EVAL_VENV"
EVAL_VENV="$(cd -- "$EVAL_VENV" && pwd)"
"$PY312" -m venv "$EVAL_VENV"
EVAL_PYTHON="$EVAL_VENV/bin/python"
"$EVAL_PYTHON" -I -m pip --isolated install -r "$LOCK"
# Dependencies are already pinned. PEP 517 build dependencies stay in a temporary
# isolated build environment under the existing pyproject build-system bounds.
"$EVAL_PYTHON" -I -m pip --isolated install --no-deps -e "$PROJECT_DIR"
"$EVAL_PYTHON" -I -m pip --isolated check
"$EVAL_PYTHON" -I - "$LOCK" <<'PY'
import importlib.metadata as metadata
import pathlib, sys
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    name, expected = line.split('==')
    actual = metadata.version(name)
    if actual != expected:
        raise SystemExit(f'Lock mismatch: {name}: {actual} != {expected}')
print('All lock versions verified.')
PY

# Sourcing this file also activates the new environment in the caller's shell.
{
  printf 'source %q\n' "$EVAL_VENV/bin/activate"
  printf 'export HF_HOME=%q\n' '/Users/dev/Library/Caches/eden-vision-ai/huggingface'
  printf '# Optional download transport workaround, not an inference option:\n'
  printf '# export HF_HUB_DISABLE_XET=1\n'
} > "$EVAL_VENV/model-eval-env.sh"
printf 'Environment ready. Run: source %q\n' "$EVAL_VENV/model-eval-env.sh"
echo 'No model was downloaded or loaded. Verify native MPS separately.'
