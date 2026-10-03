"""Explicit CLIP entrypoint. Import composes an app; lifespan owns model loading."""
from eden_vision.config import Settings
from eden_vision.main import create_app
from eden_vision.vision.runtime.lifecycle import (
    RankingRuntimeFactory, RankingRuntimeSettings, build_clip_ranking_runtime,
)


def create_production_app(
    config: Settings | None = None, *,
    ranking_runtime_factory: RankingRuntimeFactory = build_clip_ranking_runtime,
    ranking_runtime_settings: RankingRuntimeSettings | None = None,
):
    if ranking_runtime_factory is None:
        raise ValueError('production requires an explicit ranking runtime')
    return create_app(config, ranking_runtime_factory=ranking_runtime_factory,
                      ranking_runtime_settings=ranking_runtime_settings)


app = create_production_app()
