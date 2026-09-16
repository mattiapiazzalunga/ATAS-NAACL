from abc import ABC, abstractmethod
from typing import Iterable, Optional

import torch
from torch import Tensor


class BaseVectorUpdateStrategy(ABC):
    @abstractmethod
    def init_layers(
            self,
            layers: Iterable[int],
            hidden_size: int,
    ) -> None:
        pass

    @abstractmethod
    def update(
            self,
            layer_idx: int,
            acts: Tensor,
            correctness_labels: Tensor,
            starts: Tensor,
            tokens_window: int,
            attention_mask: Optional[torch.Tensor],
            group_ids: Optional[torch.Tensor]
    ) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
