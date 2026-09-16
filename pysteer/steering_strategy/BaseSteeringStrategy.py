from abc import ABC, abstractmethod
from typing import List, Union, Dict

import torch
from torch import Tensor

AlphaSpec = Union[float, Dict[int, float]]


class BaseSteeringStrategy(ABC):
    _target_layers: List[int]

    @property
    @abstractmethod
    def target_layers(self) -> List[int]:
        pass

    @target_layers.setter
    @abstractmethod
    def target_layers(self, layers: List[int]) -> None:
        pass

    @abstractmethod
    def steer(
            self,
            layer_idx: int,
            hidden_states: Tensor,
            alpha: AlphaSpec,
            mask_tok: torch.Tensor
    ) -> Tensor:
        pass
