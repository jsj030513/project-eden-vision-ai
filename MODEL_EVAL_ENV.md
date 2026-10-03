# Model evaluation environment

The service `.venv` is for FastAPI and tests. The separate model-eval venv is
for torch/Transformers experiments. AI packages are not production dependencies:
the API still uses DummyVisionModel, and no production model has been selected.

## Recorded environment

Snapshot date: 2026-09-23. Source of truth:
`/Users/dev/.venvs/eden-vision-ai-model-eval`.

- Homebrew Python **3.12.14**, native **arm64**, macOS Apple Silicon.
- pip **26.2.1**; torch **2.14.0**; transformers **5.17.0**.
- huggingface-hub **1.32.0**; tokenizers **0.23.2**; safetensors **0.8.0**.
- Native MPS built/available: **True/True**, checked outside the restricted sandbox.

`requirements/model-eval.txt` lists direct experiment dependencies. The retained
baseline scripts directly import huggingface-hub. Pillow and Eden's service
dependencies remain owned by `pyproject.toml`; they are not separately declared
as direct evaluation requirements.

`requirements/model-eval-lock.txt` is the complete `pip freeze --all` snapshot,
including transitive dependencies, project runtime dependencies, pip and setuptools.
The local editable project entry is excluded. This is a version snapshot for
Python 3.12/macOS arm64, not a cross-platform lock or an artifact-hash lock.
Keep the same project source when reproducing results; an editable install tracks
the working tree and does not itself freeze source code.

## Create a fresh environment

From the project root:

```sh
bash scripts/setup_model_eval_env.sh
```

The default destination is `/Users/dev/.venvs/eden-vision-ai-model-eval`.
**Any existing destination is refused**, including empty directories and symlinks.
Nothing is deleted or reused automatically. Use a new path for reproduction:

```sh
MODEL_EVAL_VENV=/tmp/eden-model-eval-repro-new \
  bash scripts/setup_model_eval_env.sh
source /tmp/eden-model-eval-repro-new/model-eval-env.sh
```

A single positional path also works and takes precedence over `MODEL_EVAL_VENV`.
The script locates `$(brew --prefix python@3.12)/bin/python3.12`, verifies 3.12/arm64,
and warns if the patch version differs from 3.12.14. Homebrew must already provide
the required interpreter; the script does not install or upgrade Python.

Installation uses the full lock, then installs this checkout editable with
`--no-deps`. All final locked versions and `pip check` are verified. Pip runs
isolated from user pip configuration/environment settings. Package-index network
access is needed unless the required artifacts are available locally. A failed
environment is left in place for inspection; use another fresh path for retry.

PEP 517 editable-build tooling is installed in pip's temporary build environment
according to the existing `setuptools>=77,<83` declaration. It is separate from
the snapshot's installed setuptools 84.0.0. Build-tool artifacts are not pinned
by this runtime snapshot; this workflow verifies the installed runtime rather
than claiming bit-identical wheels. `pyproject.toml` is unchanged.

## Cache and pinned checkpoints

The generated `model-eval-env.sh` activates the new venv and sets:

```sh
export HF_HOME=/Users/dev/Library/Caches/eden-vision-ai/huggingface
# Optional only for download transport problems:
export HF_HUB_DISABLE_XET=1
```

Xet disabling changes download transport, not model inference. Setup installs
Python packages only; it does not download model weights. The cached evaluation
runtime uses local files only and requires these existing pinned snapshots:

| Model | Model ID | Revision |
|---|---|---|
| CLIP | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| SigLIP2 | `google/siglip2-base-patch16-224` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |

Never evaluate floating `main`. The evaluation runner's fixed mappings remain
the executable source for these revisions. Missing snapshots fail explicitly;
package installation alone does not populate an empty model cache.

## Verify MPS without loading a model

```sh
python - <<'PY'
import platform
import importlib.metadata as metadata
import torch
print('Python:', platform.python_version(), platform.machine())
for name in ('torch', 'transformers', 'huggingface-hub', 'tokenizers', 'safetensors'):
    print(name, metadata.version(name))
print('MPS built:', torch.backends.mps.is_built())
print('MPS available:', torch.backends.mps.is_available())
PY
```

Expected: arm64, the versions above, MPS True/True. Sandboxed device access may
report unavailable; verify in a normal macOS terminal. Do not enable
`PYTORCH_ENABLE_MPS_FALLBACK=1` to conceal a native MPS failure.

## Validate, plan, and evaluate

Run from `/Users/dev/Developer/eden-vision-ai`:

```sh
python experiments/evaluate_models.py \
  --dataset /Users/dev/Datasets/eden-vision-eval --validate-only

python experiments/evaluate_models.py \
  --dataset /Users/dev/Datasets/eden-vision-eval \
  --model all --taxonomy all --dry-run

# Only after a person supplies real photos and canonical Ground Truth:
python experiments/evaluate_models.py \
  --dataset /Users/dev/Datasets/eden-vision-eval \
  --model all --taxonomy all
```

Validation and dry-run also work in the lightweight main `.venv`. Empty datasets
load no models and write no reports. Inference uses the model-eval venv, pinned
revisions and memory-cached taxonomy text embeddings. It does not choose a winner
or calibrate confidence. See `experiments/EVALUATION_RUNNER.md` for report and
annotation semantics. No real images or Ground Truth are generated by setup.

## Clean-room verification (2026-09-23)

The setup script created `/tmp/eden-model-eval-repro-verified` from Homebrew
Python, without copying the original venv. All **46 locked distributions** plus
editable Eden matched the original environment's **47 distribution versions**.
Python was 3.12.14/arm64 and native MPS was built/available True/True.
`pip check`, validate-only, and all/all dry-run passed with zero dataset entries;
no model was loaded, no inference ran, and no weights were downloaded.

The production dependency `pillow-heif==1.8.0` is installed from the native
`cp312-cp312-macosx_11_0_arm64` wheel alongside Pillow 12.3.0. A new lock
installation must therefore contain **47 locked distributions** plus editable
Eden; the historical clean-room count above predates this HEIF addition.

The original destination refusal and shell syntax check also passed. The first
attempt at `/tmp/eden-model-eval-repro` failed on sandboxed PyPI DNS access and
was left untouched; the verified retry used a separate fresh path with network
access. Main tests remained 202 passed with the existing two upstream warnings;
main torch/Transformers remained absent. Production source, taxonomy resources,
`pyproject.toml`, and the external evaluation dataset were hash-verified unchanged.
