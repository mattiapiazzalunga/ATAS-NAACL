import logging
from typing import Dict, List, Iterable, Optional

import torch
from torch import Tensor

from utils.ModelUtils import ModelUtils
from vector_update_strategy.BaseVectorUpdateStrategy import BaseVectorUpdateStrategy

logger = logging.getLogger(__name__)


class CmdVectorMediator(BaseVectorUpdateStrategy):
    def __init__(
            self,
            correctness_threshold: float = 0.5,
            incorrectness_threshold: float = 0.5,
            use_last_token_for_response: bool = True,
    ):
        super().__init__()
        self._layers: List[int] = []
        self._hidden: Optional[int] = None
        self._correct_vectors: Dict[int, Tensor] = {}
        self._incorrect_vectors: Dict[int, Tensor] = {}
        self._sum_correct_vectors: Dict[int, Tensor] = {}
        self._sum_incorrect_vectors: Dict[int, Tensor] = {}
        self._correction_vectors: Dict[int, Tensor] = {}
        self._correct_weights: Dict[int, float] = {}
        self._incorrect_weights: Dict[int, float] = {}
        self._correctness_threshold: float = float(correctness_threshold)
        self._incorrectness_threshold: float = float(incorrectness_threshold)
        self._use_last_token_for_response: bool = bool(use_last_token_for_response)

    @property
    def sum_correct_vectors(self) -> Dict[int, Tensor]:
        return self._sum_correct_vectors

    @sum_correct_vectors.setter
    def sum_correct_vectors(self, value: Dict[int, Tensor]) -> None:
        self._sum_correct_vectors = value

    @property
    def sum_incorrect_vectors(self) -> Dict[int, Tensor]:
        return self._sum_incorrect_vectors

    @sum_incorrect_vectors.setter
    def sum_incorrect_vectors(self, value: Dict[int, Tensor]) -> None:
        self._sum_incorrect_vectors = value

    @property
    def correct_vectors(self) -> Dict[int, Tensor]:
        return self._correct_vectors

    @correct_vectors.setter
    def correct_vectors(self, value: Dict[int, Tensor]) -> None:
        self._correct_vectors = value

    @property
    def incorrect_vectors(self) -> Dict[int, Tensor]:
        return self._incorrect_vectors

    @incorrect_vectors.setter
    def incorrect_vectors(self, value: Dict[int, Tensor]) -> None:
        self._incorrect_vectors = value

    @property
    def correct_weights(self) -> Dict[int, float]:
        return self._correct_weights

    @correct_weights.setter
    def correct_weights(self, value: Dict[int, float]) -> None:
        self._correct_weights = value

    @property
    def incorrect_weights(self) -> Dict[int, float]:
        return self._incorrect_weights

    @incorrect_weights.setter
    def incorrect_weights(self, value: Dict[int, float]) -> None:
        self._incorrect_weights = value

    @property
    def layers(self) -> List[int]:
        return self._layers

    @property
    def correctness_threshold(self) -> float:
        return self._correctness_threshold

    @correctness_threshold.setter
    def correctness_threshold(self, thr: float) -> None:
        self._correctness_threshold = float(thr)

    @property
    def incorrectness_threshold(self) -> float:
        return self._incorrectness_threshold

    @incorrectness_threshold.setter
    def incorrectness_threshold(self, thr: float) -> None:
        self._incorrectness_threshold = float(thr)

    @property
    def correction_vectors(self) -> Dict[int, Tensor]:
        if isinstance(self._correction_vectors, dict) and len(self._correction_vectors) > 0:
            return self._correction_vectors
        out: Dict[int, Tensor] = {}
        for idx in set(self._correct_vectors).intersection(self._incorrect_vectors):
            out[idx] = (self._correct_vectors[idx] - self._incorrect_vectors[idx]).detach()
        return out

    def init_layers(
            self,
            layers: Iterable[int],
            hidden_size: int,
    ) -> None:
        self._layers = sorted({int(l) for l in layers})
        self._hidden = int(hidden_size)
        self._correction_vectors = {}
        self._correct_vectors = {
            idx: torch.zeros(hidden_size, dtype=torch.float32)
            for idx in self._layers
        }
        self._incorrect_vectors = {
            idx: torch.zeros(hidden_size, dtype=torch.float32)
            for idx in self._layers
        }
        self._correct_weights = dict.fromkeys(self._layers, 0.0)
        self._incorrect_weights = dict.fromkeys(self._layers, 0.0)
        self._sum_correct_vectors = {
            idx: torch.zeros(hidden_size, dtype=torch.float32)
            for idx in self._layers
        }
        self._sum_incorrect_vectors = {
            idx: torch.zeros(hidden_size, dtype=torch.float32)
            for idx in self._layers
        }

    def finalize(self) -> None:

        if self._correction_vectors:
            return
        cv: Dict[int, Tensor] = {}
        for idx in self._layers:
            if idx in self._correct_vectors and idx in self._incorrect_vectors:
                v = (self._correct_vectors[idx] - self._incorrect_vectors[idx]).detach().to("cpu", torch.float32)
                cv[int(idx)] = v.contiguous()
            else:
                logger.warning("finalize(): missing vectors for layer %d; skipping.", int(idx))
        self._correction_vectors = cv


        self._correct_vectors.clear()
        self._incorrect_vectors.clear()
        self._sum_correct_vectors.clear()
        self._sum_incorrect_vectors.clear()
        self._correct_weights.clear()
        self._incorrect_weights.clear()



    @torch.inference_mode()
    def update(self, layer_idx, acts, correctness_labels, starts,
               tokens_window, attention_mask, group_ids) -> None:
        try:
            hidden_size = self.correct_vectors[layer_idx].shape[0]
            acts = ModelUtils.sanitize_fp16(acts).to(torch.float32)
            acts = ModelUtils.ensure_bsh(acts, hidden_size, from_layout="BSH")
            dev = acts.device
            starts = starts.to(dev)
            correctness_labels = correctness_labels.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if group_ids is not None:
                _ = group_ids.to(dev)

            _, ss, _ = acts.shape
            seq = torch.arange(ss, device=acts.device).unsqueeze(0)
            mask = (seq >= starts.unsqueeze(1))
            if tokens_window > 0:
                mask &= seq < (starts + tokens_window).unsqueeze(1)

            if (attention_mask is not None
                    and attention_mask.dim() == 2
                    and attention_mask.shape == mask.shape):
                mask &= attention_mask.to(dtype=torch.bool, device=mask.device)

            valid = mask.any(dim=1)

            if not valid.any():
                try:
                    too_late = int((starts >= ss).sum().item()) if torch.is_tensor(starts) else -1
                    logger.warning(
                        "CmdVectorMediator.update: all samples masked out at layer %d "
                        "(ss=%d, tokens_window=%d). starts>=ss: %s. "
                        "This often means truncation removed the response span.",
                        int(layer_idx), int(ss), int(tokens_window), str(too_late)
                    )
                except Exception:
                    logger.warning(
                        "CmdVectorMediator.update: all samples masked out at layer %d (ss=%d).",
                        int(layer_idx), int(ss)
                    )
                return

            acts_sub = acts[valid]
            if self._use_last_token_for_response:
                m_bool = mask[valid].to(torch.bool)
                bvv, _, _ = acts_sub.shape
                pos = torch.arange(ss, device=acts.device).unsqueeze(0).expand(bvv, -1)
                last_pos = pos.masked_fill(~m_bool, -1).max(dim=1).values
                b_idx = torch.arange(bvv, device=acts.device)
                acts_tok = acts_sub[b_idx, last_pos, :]
            else:
                mask_sub = mask[valid].unsqueeze(-1)
                sum_acts = (acts_sub * mask_sub).sum(dim=1)
                den = mask_sub.sum(dim=1, dtype=torch.float32).clamp_min(1.0)
                acts_tok = sum_acts / den

            if acts_tok.size(-1) != self.correct_vectors[layer_idx].size(0):
                raise RuntimeError(f"hidden-dim mismatch while updating layer {layer_idx}")

            dev = acts.device
            for store in (self._sum_correct_vectors,
                          self._sum_incorrect_vectors,
                          self._correct_vectors,
                          self._incorrect_vectors):
                if store[layer_idx].device != dev:
                    store[layer_idx] = store[layer_idx].to(dev)

            labels = correctness_labels[valid].to(torch.float32)
            good_mask = labels >= self.correctness_threshold
            bad_mask = labels < self.incorrectness_threshold

            if good_mask.any():
                w = labels[good_mask]
                s = (acts_tok[good_mask] * w.unsqueeze(1)).sum(0)
                prev_v = self._sum_correct_vectors[layer_idx]
                prev_w = self.correct_weights[layer_idx]
                new_w = prev_w + w.sum().item()
                self._sum_correct_vectors[layer_idx] = prev_v + s
                self.correct_vectors[layer_idx] = self._sum_correct_vectors[layer_idx] / (new_w + 1e-9)
                self.correct_weights[layer_idx] = new_w

            if bad_mask.any():
                w = 1.0 - labels[bad_mask]
                s = (acts_tok[bad_mask] * w.unsqueeze(1)).sum(0)
                prev_v = self._sum_incorrect_vectors[layer_idx]
                prev_w = self.incorrect_weights[layer_idx]
                new_w = prev_w + w.sum().item()
                self._sum_incorrect_vectors[layer_idx] = (prev_v + s)
                self.incorrect_vectors[layer_idx] = self._sum_incorrect_vectors[layer_idx] / (new_w + 1e-9)
                self.incorrect_weights[layer_idx] = new_w

            logger.debug(
                "[%d] good=%d bad=%d mask_any=%s",
                layer_idx, good_mask.sum().item(), bad_mask.sum().item(), mask.any().item()
            )
        except Exception as e:
            logger.exception("Error updating vectors for layer %d", layer_idx)
            raise RuntimeError(f"CmdVectorMediator.update failed on layer {layer_idx}: {e}") from e

    def reset(self) -> None:
        if self._hidden is None:
            raise RuntimeError(
                "CmdVectorMediator not initialized: "
                "Call init_layers(layers, hidden_size) before reset()."
            )
        self.init_layers(self._layers, self._hidden)
