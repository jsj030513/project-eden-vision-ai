#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export HF_HOME="${HF_HOME-$HOME/Library/Caches/eden-vision-ai/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE-$HF_HOME/hub}"
if [[ -z "$HF_HOME" || -z "$HF_HUB_CACHE" ]]; then
    echo 'Error: cache paths must not be empty' >&2
    exit 1
fi
if [[ "${HF_HUB_OFFLINE-1}" != '1' || "${TRANSFORMERS_OFFLINE-1}" != '1' ]]; then
    echo 'Error: offline settings conflict with required offline mode' >&2
    exit 1
fi
if [[ "${PYTORCH_ENABLE_MPS_FALLBACK-}" == '1' ]]; then
    echo 'Error: CPU fallback is prohibited' >&2
    exit 1
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset PYTORCH_ENABLE_MPS_FALLBACK
# Preserve Settings' loopback/port/log-level policy; one model per worker.
exec ./.venv/bin/python -c '
import uvicorn
from eden_vision.config import Settings
config = Settings()
uvicorn.run("eden_vision.production:app", host=config.host, port=config.port,
            log_level=config.log_level.lower(), workers=1)
'
