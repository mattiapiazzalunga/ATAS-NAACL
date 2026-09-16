import logging
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor
from utils.ModelUtils import ModelUtils
from vector_update_strategy.BaseVectorUpdateStrategy import BaseVectorUpdateStrategy

logger = logging.getLogger(__name__)


class CpcaVectorMediator(BaseVectorUpdateStrategy):

    def __init__(
            self,
            correctness_threshold: float = 0.5,
            incorrectness_threshold: float = 0.5,
            use_last_token_for_response: bool = True,
            eps: float = 1e-12,
    ):
        super().__init__()
        self._layers: List[int] = []
        self._hidden: Optional[int] = None

        self._thr_g = float(correctness_threshold)
        self._thr_b = float(incorrectness_threshold)
        self._use_last = bool(use_last_token_for_response)
        self._eps = float(eps)

        self._good: Dict[int, List[Tensor]] = {}
        self._bad: Dict[int, List[Tensor]] = {}
        self._correction_vectors: Dict[int, Tensor] = {}

    def init_layers(self, layers: Iterable[int], hidden_size: int) -> None:
        self._layers = sorted({int(l) for l in layers})
        self._hidden = int(hidden_size)
        self._good = {L: [] for L in self._layers}
        self._bad = {L: [] for L in self._layers}
        self._correction_vectors = {}

    @property
    def correction_vectors(self) -> Dict[int, Tensor]:
        return dict(self._correction_vectors)

    @staticmethod
    def _cap_append(lst: List[Tensor], x: Tensor) -> None:
        lst.append(x)

    @staticmethod
    def _masked_mean(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=hs.dtype).unsqueeze(-1)
        den = m.sum(dim=1).clamp_min(1.0)
        return (hs * m).sum(dim=1) / den

    @staticmethod
    def _last_token_repr(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bb, ss, hh = hs.shape
        m_bool = mask.to(torch.bool)
        pos = torch.arange(ss, device=hs.device).unsqueeze(0).expand(bb, -1)
        masked_pos = pos.masked_fill(~m_bool, -1)
        last_pos = masked_pos.max(dim=1).values

        out = torch.zeros((bb, hh), device=hs.device, dtype=hs.dtype)
        valid = last_pos >= 0
        if valid.any():
            b_idx = torch.arange(bb, device=hs.device)[valid]
            out[valid] = hs[b_idx, last_pos[valid], :]
        return out

    @torch.inference_mode()
    def update(
            self,
            layer_idx: int,
            acts: Tensor,
            correctness_labels: Tensor,
            starts: Tensor,
            tokens_window: int,
            attention_mask: Optional[torch.Tensor],
            group_ids: Optional[torch.Tensor],
    ) -> None:
        if self._hidden is None:
            raise RuntimeError("CpcaVectorMediator not initialized (missing hidden_size).")
        if int(layer_idx) not in self._good:
            return

        h = int(self._hidden)

        acts = ModelUtils.sanitize_fp16(acts).to(torch.float32)
        acts = ModelUtils.ensure_bsh(acts, h, from_layout="BSH")

        dev = acts.device
        starts = starts.to(dev, dtype=torch.int64)
        labs = correctness_labels.to(dev, dtype=torch.float32)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        _, ss, _ = acts.shape
        pos = torch.arange(ss, device=dev).unsqueeze(0)

        mask = (pos >= starts.unsqueeze(1))
        if tokens_window > 0:
            mask &= (pos < (starts + tokens_window).unsqueeze(1))

        if (
                attention_mask is not None
                and attention_mask.dim() == 2
                and attention_mask.shape == mask.shape
        ):
            mask &= attention_mask.to(dtype=torch.bool, device=dev)

        valid = mask.any(dim=1)
        if not bool(valid.any().item()):
            return

        acts_sub = acts[valid]
        mask_sub = mask[valid]
        labs_sub = labs[valid]

        if self._use_last:
            acts_tok = self._last_token_repr(acts_sub, mask_sub)
        else:
            acts_tok = self._masked_mean(acts_sub, mask_sub)

        good_mask = labs_sub >= self._thr_g
        bad_mask = labs_sub < self._thr_b

        if good_mask.any():
            xg = acts_tok[good_mask].detach().to("cpu", torch.float32).contiguous()
            for i in range(int(xg.size(0))):
                self._cap_append(self._good[int(layer_idx)], xg[i])

        if bad_mask.any():
            xb = acts_tok[bad_mask].detach().to("cpu", torch.float32).contiguous()
            for i in range(int(xb.size(0))):
                self._cap_append(self._bad[int(layer_idx)], xb[i])

    @staticmethod
    def _pc1(dd: torch.Tensor) -> torch.Tensor:
        if dd.ndim != 2 or int(dd.size(0)) <= 0:
            return torch.zeros((int(dd.size(-1)),), dtype=torch.float32)
        if int(dd.size(0)) == 1:
            return dd[0].to(torch.float32)

        dd = dd.to(torch.float32)

        _, _, vh = torch.linalg.svd(dd, full_matrices=False)
        v = vh[0, :]

        return v.to(torch.float32)

    def finalize(self) -> None:
        if self._correction_vectors:
            return
        if self._hidden is None:
            raise RuntimeError("CpcaVectorMediator not initialized.")

        cv: Dict[int, Tensor] = {}
        _ = self._eps

        for L in self._layers:
            g = self._good.get(int(L), [])
            b = self._bad.get(int(L), [])

            if len(g) == 0 or len(b) == 0:
                logger.warning(
                    "CpcaVectorMediator.finalize: insufficient samples at layer %d (good=%d bad=%d). "
                    "Setting correction vector to ZERO to keep layer present (ACTS-like behavior).",
                    int(L), len(g), len(b)
                )
                z = torch.zeros((int(self._hidden),), dtype=torch.float32)
                cv[int(L)] = z.detach().to("cpu", torch.float32).contiguous()
                continue

            xg = torch.stack(g, 0).to(torch.float32)
            xb = torch.stack(b, 0).to(torch.float32)

            mu_g = xg.mean(0)
            mu_b = xb.mean(0)

            mu = 0.5 * (mu_g + mu_b)
            xx = torch.cat([xg - mu, xb - mu], dim=0)

            v = self._pc1(xx)

            md = (mu_g - mu_b)
            if float((v * md).sum().item()) < 0.0:
                v = -v

            cv[int(L)] = v.detach().to("cpu", torch.float32).contiguous()

        self._correction_vectors = cv

        self._good.clear()
        self._bad.clear()

    def reset(self) -> None:
        if self._hidden is None:
            raise RuntimeError("CpcaVectorMediator not initialized.")
        self.init_layers(self._layers, self._hidden)