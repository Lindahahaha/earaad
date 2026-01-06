from typing import Sequence
from pydantic import BaseModel
import torch


class ModelInputArgs(BaseModel):
    fs: int
    window_length: int
    num_channels: int | None = None
    num_audio_features: int | None = None

    model_config = {"extra": "allow"}


class LossArgs(BaseModel):
    weight: Sequence[float | torch.Tensor] | None = None

    # 修改这里：添加 "arbitrary_types_allowed": True 以支持 torch.Tensor
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.weight is not None:
            self.weight = torch.Tensor(self.weight)  # type: ignore