import logging
from typing import Dict, Iterable, Optional, List, Union, Set, Tuple

import torch
from torch import Tensor

from vector_update_strategy.BaseVectorUpdateStrategy import BaseVectorUpdateStrategy

logger = logging.getLogger(__name__)


class VectorMediator:

    def __init__(
            self,
            layers_to_extract: Union[Iterable[int], int],
            hidden_size: int,
            update_strategy: Optional[BaseVectorUpdateStrategy] = None
    ):
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {hidden_size}")

        self.hidden_size = hidden_size
        self.layers_to_extract = layers_to_extract
        self.update_strategy = update_strategy
        self._update_errors: List[Tuple[int, Exception]] = []

    def assert_ok(self) -> None:
        if self._update_errors:
            idx, e = self._update_errors[0]
            raise RuntimeError(f"Vector update failed on layer {idx}: {e}") from e

    def clear_errors(self) -> None:
        self._update_errors.clear()

    @property
    def layers_to_extract(self) -> List[int]:
        return self._layers_to_extract

    @layers_to_extract.setter
    def layers_to_extract(self, value: Union[Iterable[int], int]) -> None:
        if isinstance(value, int):
            value = [value]

        try:
            layer_set: Set[int] = {int(l) for l in value}
            if not layer_set:
                raise ValueError("At least one layer must be specified for extraction")
            self._layers_to_extract = sorted(layer_set)

            if getattr(self, "_update_strategy", None) is not None:
                self._update_strategy.init_layers(self._layers_to_extract,
                                                  self.hidden_size)
        except (TypeError, ValueError) as e:
            if isinstance(e, TypeError):
                raise TypeError("layers_to_extract must contain integer values") from e
            raise

    @property
    def update_strategy(self) -> Optional[BaseVectorUpdateStrategy]:
        return self._update_strategy

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @update_strategy.setter
    def update_strategy(self, value: Optional[BaseVectorUpdateStrategy]) -> None:
        self._update_strategy = value

        if self._update_strategy is not None:
            try:
                self._update_strategy.init_layers(
                    self.layers_to_extract,
                    self.hidden_size
                )
            except Exception as e:
                self._update_strategy = None
                raise RuntimeError(f"Failed to initialize update strategy: {str(e)}") from e

    @hidden_size.setter
    def hidden_size(self, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {value}")
        self._hidden_size = value

        if getattr(self, "_update_strategy", None) is not None:
            self._update_strategy.init_layers(self.layers_to_extract,
                                              self._hidden_size)

    def update_vectors(
            self,
            activations: Dict[int, Tensor],
            labels: Tensor,
            starts: Tensor,
            tokens_window: int,
            attention_mask,
            group_ids: Optional[torch.Tensor],
    ) -> None:
        if self.update_strategy is None:
            raise RuntimeError("VectorMediator.update_vectors: update_strategy is None.")
        if not activations:
            raise RuntimeError("VectorMediator.update_vectors: activations is empty.")

        with torch.inference_mode():
            for idx in self.layers_to_extract:
                if idx not in activations:
                    continue
                try:
                    dev = activations[idx].device
                    if torch.is_tensor(attention_mask) and attention_mask.device != dev:
                        logger.warning(
                            "VectorMediator: attention_mask on %s but activations[%d] on %s. "
                            "Consider moving masks/starts/labels to activations device.",
                            str(attention_mask.device), idx, str(dev)
                        )
                    if torch.is_tensor(starts) and starts.device != dev:
                        logger.warning(
                            "VectorMediator: starts on %s but activations[%d] on %s.",
                            str(starts.device), idx, str(dev)
                        )
                except Exception:
                    pass
                try:
                    self.update_strategy.update(
                        layer_idx=idx,
                        acts=activations[idx],
                        correctness_labels=labels,
                        starts=starts,
                        tokens_window=tokens_window,
                        attention_mask=attention_mask,
                        group_ids=group_ids,
                    )
                except Exception as e:
                    self._update_errors.append((idx, e))
                    logger.exception("Error updating vectors for layer %d", idx)
                    raise RuntimeError(f"Error updating vectors for layer {idx}: {e}") from e

    def reset(self) -> None:
        if self.update_strategy is not None:
            try:
                self.update_strategy.reset()
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("Error resetting update strategy")
                else:
                    logger.warning("Error resetting update strategy: %s", e)
