# Local CLIP API operation

Use macOS Apple Silicon, Python 3.12, and the project `.venv`. The API-only
installation remains usable for legacy development. For CLIP, install the pinned
optional runtime dependencies:

```sh
./.venv/bin/python -m pip install -r requirements/model-runtime.txt
./.venv/bin/python -m pip check
```

Provision the pinned model snapshot separately before starting the service.
Startup never downloads missing files. The model is `openai/clip-vit-base-patch32`,
revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, using eden-v2 (78 concepts)
and `a photo of a {display_name}` with lowercase display names.

## Entrypoints

Legacy development retains DummyVisionModel, eden-v1 and schema 1.0:

```sh
./.venv/bin/python -m uvicorn eden_vision.main:app --host 127.0.0.1 --port 8090 --workers 1
```

The explicit CLIP production entrypoint is `eden_vision.production:app`:

```sh
./scripts/run_clip_api.sh
```

The launcher locates its own project root and uses that project's `.venv` Python.
It reads the existing Settings host/port/log-level convention, including `.env`;
HOST only permits `127.0.0.1`, PORT defaults to 8090. Do not expose it externally
without a separate deployment design.

Use **one worker**. Every worker loads a separate model and MPS cache; current
request serialization is validated for one app instance. The launcher explicitly
sets `workers=1` and does not use reload.

## Cache and device

Cache resolution is explicit runtime settings, then compatible existing environment,
then `$HOME/Library/Caches/eden-vision-ai/huggingface`; the hub directory is its
`hub` child. No username is hardcoded. The launcher supplies missing defaults and
sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before Python starts.
Conflicting existing offline settings, inconsistent cache paths, or already-imported
HF constants cause failure rather than silent override. Environment setup within
an app persists for its process; initialize compatible settings before importing HF.

MPS is required. CPU fallback is prohibited. `PYTORCH_ENABLE_MPS_FALLBACK=1`
causes failure; the launcher otherwise unsets it. Missing weights, unavailable MPS,
or load failures prevent startup. There is no dummy fallback in the CLIP app.

## Liveness and readiness

- `GET /health`: HTTP 200, `{"status":"ok","service":"eden-vision-ai"}`.
  This is liveness, not a model readiness check.
- `GET /ready`: HTTP 200 when the current mode is ready; HTTP 503 otherwise.
  CLIP reports mode `clip_ranking`, runtime flags and the loaded model ID/revision,
  taxonomy version and device. Legacy mode reports `legacy_classification` with
  ranking flags false. Cache paths, environment values and tensors are excluded.
  Readiness does not add an `X-Request-ID` response header.

## Classification response semantics

The CLIP app accepts the existing multipart `image` upload at
`POST /v1/vision/classify`. Its response is schema **1.1**, `status="ranked"`,
`predictions=[]`, Top-5 `ranking`, and `unknown_decision="not_applied"`.
Scores are `raw_image_text_logit`, not confidence or probability. Top-1 is a ranked
candidate, not an accepted classification. No acceptance, UNKNOWN, or quality
rejection policy is applied. The legacy app's schema 1.0 is unchanged.

## Readiness-only smoke

Set the cache/offline environment above and unset CPU fallback before running:

```sh
./.venv/bin/python scripts/smoke_clip_production_readiness.py --load
```

This explicitly loads local weights and text embeddings, calls only `/health` and
`/ready` through in-process ASGI, and closes the runtime. It never ranks images.
