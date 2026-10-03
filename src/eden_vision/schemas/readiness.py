"""Safe readiness projection. No paths, tensors, environment or error details."""
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    status: Literal['ready', 'not_ready']
    mode: Literal['legacy_classification', 'clip_ranking']
    ranking_runtime_enabled: bool = Field(strict=True)
    ranking_runtime_ready: bool = Field(strict=True)
    model_id: Literal['openai/clip-vit-base-patch32'] | None = None
    model_revision: Literal['3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'] | None = None
    taxonomy_version: Literal['eden-v2'] | None = None
    device: Literal['mps'] | None = None

    @model_validator(mode='after')
    def consistent(self) -> Self:
        metadata = (self.model_id, self.model_revision, self.taxonomy_version, self.device)
        if self.mode == 'legacy_classification':
            if self.ranking_runtime_enabled or self.ranking_runtime_ready or any(v is not None for v in metadata):
                raise ValueError('legacy readiness cannot claim a ranking runtime')
        else:
            if not self.ranking_runtime_enabled:
                raise ValueError('clip readiness requires configured ranking')
            if self.ranking_runtime_ready != (self.status == 'ready'):
                raise ValueError('clip status must match runtime readiness')
            if self.status == 'ready' and any(v is None for v in metadata):
                raise ValueError('ready CLIP metadata required')
        return self
