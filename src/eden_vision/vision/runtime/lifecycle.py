"""Explicit app-owned ranking lifecycle; no ML imports on the default API path.

Environment is process-wide, so incompatible app configurations fail rather than
silently rebinding imported HF constants. Each app/worker owns its own runtime.
HTTP ranking uses its app's classification_lock; that lock is neither global
nor cross-worker. This lifecycle module never schedules ranking itself.
"""
from collections.abc import Callable
from contextlib import asynccontextmanager
from importlib.resources import files
import os
from pathlib import Path
import sys
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from eden_vision.classification.taxonomy import Taxonomy, parse_taxonomy
from eden_vision.vision.types import ImageInput, RankingResult


class RankingRuntimeSettings(BaseModel):
    """Explicit composition only. Environment cannot enable a default app."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    enabled: Literal[True] = True
    model_id: Literal['openai/clip-vit-base-patch32'] = 'openai/clip-vit-base-patch32'
    revision: Literal['3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'] = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
    taxonomy_version: Literal['eden-v2'] = 'eden-v2'
    prompt_template: Literal['a photo of a {display_name}'] = 'a photo of a {display_name}'
    device: Literal['mps'] = 'mps'
    offline: Literal[True] = True
    hf_home: Path | None = None
    hf_hub_cache: Path | None = None

    @field_validator('hf_home', 'hf_hub_cache', mode='before')
    @classmethod
    def nonempty_path(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError('cache path must not be empty')
        return value

    @model_validator(mode='after')
    def resolve_paths(self):
        home = self.hf_home
        if home is None:
            value = os.environ.get('HF_HOME')
            if value is not None and not value.strip():
                raise ValueError('HF_HOME must not be empty')
            home = Path(value) if value is not None else Path.home() / 'Library/Caches/eden-vision-ai/huggingface'
        hub = self.hf_hub_cache
        if hub is None:
            value = os.environ.get('HF_HUB_CACHE')
            if value is not None and not value.strip():
                raise ValueError('HF_HUB_CACHE must not be empty')
            hub = Path(value) if value is not None else home / 'hub'
        home, hub = home.expanduser().resolve(), hub.expanduser().resolve()
        if hub != home / 'hub':
            raise ValueError('HF_HUB_CACHE must equal HF_HOME/hub for this composition')
        object.__setattr__(self, 'hf_home', home)
        object.__setattr__(self, 'hf_hub_cache', hub)
        return self


class RuntimeEnvironmentError(RuntimeError):
    """Conflicting process environment or already-initialized HF constants."""


def prepare_runtime_environment(settings: RankingRuntimeSettings):
    desired = {'HF_HOME': str(settings.hf_home), 'HF_HUB_CACHE': str(settings.hf_hub_cache),
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeEnvironmentError('PYTORCH_ENABLE_MPS_FALLBACK=1 is forbidden')
    for key, value in desired.items():
        actual = os.environ.get(key)
        if actual is not None and actual != value:
            raise RuntimeEnvironmentError(f'{key} conflicts: configured={value!r}, environment={actual!r}')
    # Never reload HF modules or repair their cached constants after import.
    constants = sys.modules.get('huggingface_hub.constants')
    if constants is not None:
        for key in ('HF_HOME', 'HF_HUB_CACHE'):
            if str(getattr(constants, key, None)) != desired[key]:
                raise RuntimeEnvironmentError(f'already-imported Hugging Face {key} conflicts; use a fresh process')
        if getattr(constants, 'HF_HUB_OFFLINE', None) is not True:
            raise RuntimeEnvironmentError('already-imported Hugging Face is not offline; use a fresh process')
    hub = sys.modules.get('transformers.utils.hub')
    if hub is not None and not hub.is_offline_mode():
        raise RuntimeEnvironmentError('already-imported Transformers is not offline; use a fresh process')
    # All checks precede any mutation. Values persist for this process, because
    # initialized HF modules would make restoring old values misleading.
    for key, value in desired.items():
        os.environ.setdefault(key, value)


class RankingRuntime(Protocol):
    @property
    def metadata(self) -> dict: ...
    def load(self) -> None: ...
    def close(self) -> None: ...
    def rank(self, image: ImageInput) -> RankingResult: ...


RankingRuntimeFactory = Callable[[Taxonomy, RankingRuntimeSettings], RankingRuntime]


def build_clip_ranking_runtime(taxonomy: Taxonomy, settings: RankingRuntimeSettings) -> RankingRuntime:
    prepare_runtime_environment(settings)
    from eden_vision.vision.runtime.clip import ClipRankingRuntime, SPEC
    if (settings.model_id, settings.revision, settings.prompt_template, settings.device) != (
            SPEC.model_id, SPEC.revision, SPEC.prompt_template, SPEC.device):
        raise ValueError('runtime spec differs from lifecycle configuration')
    return ClipRankingRuntime(taxonomy)


def load_ranking_taxonomy() -> Taxonomy:
    taxonomy = parse_taxonomy(files('eden_vision.resources').joinpath('taxonomy_v2.json').read_text())
    if taxonomy.taxonomy_version != 'eden-v2' or len(taxonomy.concepts) != 78:
        raise ValueError('CLIP composition requires eden-v2 / 78 candidates')
    return taxonomy


def initialize_ranking_state(app, enabled: bool):
    app.state.ranking_runtime_enabled = enabled
    app.state.ranking_runtime_ready = False
    app.state.ranking_runtime_startup_failed = False
    app.state.ranking_runtime = None


@asynccontextmanager
async def ranking_lifespan(app, factory: RankingRuntimeFactory | None,
                           settings: RankingRuntimeSettings | None):
    initialize_ranking_state(app, factory is not None)
    if factory is None:
        yield
        return
    runtime = None
    started = False
    failed = False
    try:
        settings = settings if settings is not None else RankingRuntimeSettings()
        # Revalidate even settings built with model_construct/model_copy.
        settings = RankingRuntimeSettings.model_validate(settings.model_dump())
        prepare_runtime_environment(settings)
        taxonomy = load_ranking_taxonomy()
        runtime = factory(taxonomy, settings)
        runtime.load()
        metadata = runtime.metadata
        if (metadata.get('ready') is not True or metadata.get('taxonomy_version') != 'eden-v2'
                or metadata.get('candidate_count') != 78):
            raise RuntimeError('ranking runtime did not become ready with eden-v2 / 78 candidates')
        app.state.ranking_runtime = runtime
        app.state.ranking_runtime_ready = True
        started = True
        yield
    except BaseException:
        failed = True
        if not started:
            app.state.ranking_runtime_startup_failed = True
        raise
    finally:
        app.state.ranking_runtime_ready = False
        app.state.ranking_runtime = None
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                if not failed:
                    raise
