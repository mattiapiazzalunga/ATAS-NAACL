import logging
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from enums.MediatorGroupRepresentativeEnum import MediatorGroupRepresentativeEnum
from torch import Tensor
from utils.ModelUtils import ModelUtils
from vector_update_strategy.BaseVectorUpdateStrategy import BaseVectorUpdateStrategy

logger = logging.getLogger(__name__)


class AtasVectorMediator(BaseVectorUpdateStrategy):
    def __init__(
            self,
            correctness_threshold: float = 0.5,
            representative: MediatorGroupRepresentativeEnum = MediatorGroupRepresentativeEnum.PROTOTYPE,
            prototype_k: int = 3,
            prototype_max_iters: int = 15,
            use_last_token_for_prompt: bool = True,
            use_last_token_for_response: bool = True,
            *,
            master_prompt_layer: Optional[int] = None,
            correction_mode: str = "cmd",
            eps: float = 1e-12,
            fail_on_missing_prompt_samples: bool = True,
    ) -> None:
        super().__init__()
        self._layers: List[int] = []
        self._layers_set = set()
        self._hidden: Optional[int] = None
        self._thr = float(correctness_threshold)

        cm = str(correction_mode).strip().lower()
        if cm not in ("cmd", "cpca"):
            raise ValueError("correction_mode must be 'cmd' or 'cpca'")
        self._correction_mode: str = cm
        self._eps: float = float(eps)

        if representative not in (
                MediatorGroupRepresentativeEnum.MEAN,
                MediatorGroupRepresentativeEnum.PROTOTYPE,
        ):
            raise ValueError("representative must be 'mean' or 'prototype'")
        self._rep: MediatorGroupRepresentativeEnum = representative

        self._proto_k = max(1, int(prototype_k))
        self._proto_iters = max(1, int(prototype_max_iters))

        self._use_last_token_for_prompt: bool = bool(use_last_token_for_prompt)
        self._use_last_token_for_response: bool = bool(use_last_token_for_response)
        self._master_prompt_layer: Optional[int] = (
            int(master_prompt_layer) if master_prompt_layer is not None else None
        )
        self._fail_on_missing_prompt_samples: bool = bool(fail_on_missing_prompt_samples)

        self._sum_prompt: Dict[int, Dict[int, Tuple[Tensor, int]]] = {}
        self._sum_corr: Dict[int, Dict[int, Tuple[Tensor, int]]] = {}
        self._sum_inc: Dict[int, Dict[int, Tuple[Tensor, int]]] = {}
        self._sum_corr_sq: Dict[int, Dict[int, Tuple[Tensor, int]]] = {}
        self._sum_inc_sq: Dict[int, Dict[int, Tuple[Tensor, int]]] = {}

        self._lst_prompt: Dict[int, Dict[int, List[Tensor]]] = {}
        self._lst_corr: Dict[int, Dict[int, List[Tensor]]] = {}
        self._lst_inc: Dict[int, Dict[int, List[Tensor]]] = {}

        self._routing_centroids: Dict[int, Tensor] = {}
        self._correction_vectors: Dict[int, Tensor] = {}
        self._task_ids_order: List[int] = []
        self._finalized: bool = False

    def _is_mean_rep(self) -> bool:
        return self._rep == MediatorGroupRepresentativeEnum.MEAN

    def _is_list_rep(self) -> bool:
        return self._rep == MediatorGroupRepresentativeEnum.PROTOTYPE

    @property
    def use_last_token_for_prompt(self) -> bool:
        return self._use_last_token_for_prompt

    def init_layers(self, layers: Iterable[int], hidden_size: int) -> None:
        self._layers = sorted({int(l) for l in layers})
        self._layers_set = set(self._layers)
        self._hidden = int(hidden_size)

        if not self._layers:
            raise ValueError("AtasVectorMediator.init_layers: empty layers.")

        if self._master_prompt_layer is None:
            self._master_prompt_layer = int(self._layers[len(self._layers) // 2])

        prompt_layers = sorted(set(self._layers) | {int(self._master_prompt_layer)})

        self._sum_prompt = {L: {} for L in prompt_layers}
        self._sum_corr = {L: {} for L in self._layers}
        self._sum_inc = {L: {} for L in self._layers}
        self._sum_corr_sq = {L: {} for L in self._layers}
        self._sum_inc_sq = {L: {} for L in self._layers}

        self._lst_prompt = {L: {} for L in prompt_layers}
        self._lst_corr = {L: {} for L in self._layers}
        self._lst_inc = {L: {} for L in self._layers}

        self._routing_centroids.clear()
        self._correction_vectors.clear()
        self._task_ids_order = []
        self._finalized = False

    @torch.inference_mode()
    def update(
            self,
            layer_idx: int,
            acts: Tensor,
            correctness_labels: Tensor,
            starts: Tensor,
            tokens_window: int,
            attention_mask: Optional[Tensor] = None,
            group_ids: Optional[Tensor] = None,
    ) -> None:
        if self._hidden is None:
            raise RuntimeError("AtasVectorMediator not initialized (missing hidden_size).")
        if group_ids is None:
            return

        collect_prompt = int(layer_idx) == int(self._master_prompt_layer)
        collect_response = int(layer_idx) in self._layers_set

        if not collect_prompt and not collect_response:
            return

        acts = ModelUtils.sanitize_fp16(acts).to(torch.float32)
        acts = ModelUtils.ensure_bsh(acts, self._hidden, from_layout="BSH")

        bb, ss, hh = acts.shape
        dev = acts.device

        starts = starts.to(device=dev, dtype=torch.int64)
        correctness_labels = correctness_labels.to(device=dev, dtype=torch.float32)
        group_ids = group_ids.to(device=dev, dtype=torch.long)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=dev)

        pos = torch.arange(ss, device=dev).unsqueeze(0)

        mask_resp = pos >= starts.unsqueeze(1)
        if tokens_window > 0:
            mask_resp &= pos < (starts + tokens_window).unsqueeze(1)

        if (
                attention_mask is not None
                and attention_mask.dim() == 2
                and attention_mask.shape == mask_resp.shape
        ):
            non_pad = attention_mask.to(dtype=torch.bool)
        else:
            non_pad = torch.ones_like(mask_resp, dtype=torch.bool, device=dev)

        mask_resp &= non_pad
        mask_prompt = (non_pad & (pos < starts.unsqueeze(1))) if collect_prompt else None

        valid_prompt = (
            mask_prompt.any(dim=1)
            if collect_prompt and mask_prompt is not None
            else torch.zeros((bb,), dtype=torch.bool, device=dev)
        )
        valid_resp = (
            mask_resp.any(dim=1)
            if collect_response
            else torch.zeros((bb,), dtype=torch.bool, device=dev)
        )
        valid = (valid_prompt | valid_resp) if collect_prompt else valid_resp

        if not valid.any():
            try:
                too_late = int((starts >= ss).sum().item())
                logger.warning(
                    "AtasVectorMediator.update: all samples masked out at layer %d "
                    "(ss=%d, tokens_window=%d). starts>=ss: %d/%d. "
                    "This often means truncation removed the response span or starts is too large.",
                    int(layer_idx), int(ss), int(tokens_window), too_late, int(bb)
                )
            except Exception:
                logger.warning(
                    "AtasVectorMediator.update: all samples masked out at layer %d (ss=%d).",
                    int(layer_idx), int(ss)
                )
            return

        acts = acts[valid]
        if collect_prompt and mask_prompt is not None:
            mask_prompt = mask_prompt[valid]
        mask_resp = mask_resp[valid]
        correctness_labels = correctness_labels[valid]
        group_ids = group_ids[valid]

        bb = int(acts.size(0))
        valid_prompt = (
            mask_prompt.any(dim=1)
            if collect_prompt and mask_prompt is not None
            else torch.zeros((bb,), dtype=torch.bool, device=acts.device)
        )
        valid_resp = (
            mask_resp.any(dim=1)
            if collect_response
            else torch.zeros((bb,), dtype=torch.bool, device=acts.device)
        )

        def masked_mean(x: Tensor, m: Tensor) -> Tensor:
            m = m.unsqueeze(-1)
            den = m.sum(dim=1).clamp_min(1.0)
            return (x * m).sum(dim=1) / den

        def last_token_repr(x: Tensor, m: Tensor) -> Tensor:
            bbb, sss, hhh = x.shape
            m_bool = m.to(torch.bool)
            positions = torch.arange(sss, device=x.device).unsqueeze(0).expand(bbb, -1)
            masked_pos = positions.masked_fill(~m_bool, -1)
            last_pos = masked_pos.max(dim=1).values

            out = torch.zeros((bbb, hhh), device=x.device, dtype=x.dtype)
            validd = last_pos >= 0
            if validd.any():
                idx = last_pos[validd]
                b_idx = torch.arange(bbb, device=x.device)[validd]
                out[validd] = x[b_idx, idx, :]
            return out

        if collect_prompt and mask_prompt is not None:
            if self.use_last_token_for_prompt:
                prompt_vec = last_token_repr(acts, mask_prompt)
            else:
                prompt_vec = masked_mean(acts, mask_prompt)
        else:
            prompt_vec = None

        resp_vec = None
        if collect_response:
            if self._use_last_token_for_response:
                resp_vec = last_token_repr(acts, mask_resp)
            else:
                resp_vec = masked_mean(acts, mask_resp)

        if collect_response and not valid_resp.any():
            logger.warning(
                "AtasVectorMediator.update: all responses empty after masking at layer %d "
                "(likely truncation removed the response). Skipping response-centroid updates for this batch.",
                int(layer_idx)
            )

        labs_cpu = correctness_labels.detach().to(torch.float32).cpu()
        tid_cpu = group_ids.detach().to(torch.long).cpu()
        prompt_vec_cpu = prompt_vec.detach().to(torch.float32).cpu() if prompt_vec is not None else None
        resp_vec_cpu = resp_vec.detach().to(torch.float32).cpu() if resp_vec is not None else None
        z = torch.zeros(hh, dtype=torch.float32)

        if self._is_mean_rep():
            pv_norm = None
            if collect_prompt and prompt_vec_cpu is not None:
                pv_norm = prompt_vec_cpu / prompt_vec_cpu.norm(dim=1, keepdim=True).clamp_min(1e-12)

            tids = tid_cpu.flatten()
            unique_tids = torch.unique(tids, dim=0)

            vp = valid_prompt.to(torch.bool).cpu()
            vr = valid_resp.to(torch.bool).cpu()

            for tid in unique_tids.tolist():
                tid = int(tid)

                if collect_prompt and pv_norm is not None:
                    sel_p = ((tids == tid) & vp).nonzero(as_tuple=False).squeeze(1)
                    if sel_p.numel() > 0:
                        s, c = self._sum_prompt[layer_idx].get(tid, (z, 0))
                        self._sum_prompt[layer_idx][tid] = (
                            s + pv_norm.index_select(0, sel_p).sum(0),
                            c + int(sel_p.numel()),
                        )

                if collect_response:
                    sel_r = ((tids == tid) & vr).nonzero(as_tuple=False).squeeze(1)
                    if sel_r.numel() == 0:
                        continue

                    labs_sel = labs_cpu.index_select(0, sel_r)
                    rv_sel = resp_vec_cpu.index_select(0, sel_r)

                    good = (labs_sel >= self._thr).nonzero(as_tuple=False).squeeze(1)
                    bad = (labs_sel < self._thr).nonzero(as_tuple=False).squeeze(1)

                    if good.numel() > 0:
                        rvg = rv_sel.index_select(0, good)

                        s, c = self._sum_corr[layer_idx].get(tid, (z, 0))
                        self._sum_corr[layer_idx][tid] = (s + rvg.sum(0), c + int(good.numel()))

                        s2, c2 = self._sum_corr_sq[layer_idx].get(tid, (z, 0))
                        self._sum_corr_sq[layer_idx][tid] = (s2 + (rvg * rvg).sum(0), c2 + int(good.numel()))

                    if bad.numel() > 0:
                        rvb = rv_sel.index_select(0, bad)

                        s, c = self._sum_inc[layer_idx].get(tid, (z, 0))
                        self._sum_inc[layer_idx][tid] = (s + rvb.sum(0), c + int(bad.numel()))

                        s2, c2 = self._sum_inc_sq[layer_idx].get(tid, (z, 0))
                        self._sum_inc_sq[layer_idx][tid] = (s2 + (rvb * rvb).sum(0), c2 + int(bad.numel()))

                    if self._correction_mode == "cpca":
                        lc = self._lst_corr[layer_idx].setdefault(tid, [])
                        li = self._lst_inc[layer_idx].setdefault(tid, [])

                        if good.numel() > 0:
                            xg = rv_sel.index_select(0, good).to(torch.float32)
                            for ii in range(int(xg.size(0))):
                                lc.append(xg[ii].detach().cpu())

                        if bad.numel() > 0:
                            xb = rv_sel.index_select(0, bad).to(torch.float32)
                            for ii in range(int(xb.size(0))):
                                li.append(xb[ii].detach().cpu())

        else:
            for i in range(bb):
                tid = int(tid_cpu[i].item())

                lp = self._lst_prompt[layer_idx].setdefault(tid, []) if collect_prompt else None
                lc = self._lst_corr[layer_idx].setdefault(tid, []) if collect_response else None
                li = self._lst_inc[layer_idx].setdefault(tid, []) if collect_response else None

                if (
                        collect_prompt
                        and prompt_vec_cpu is not None
                        and lp is not None
                        and bool(valid_prompt[i].cpu().item())
                ):
                    lp.append(prompt_vec_cpu[i])

                if collect_response and bool(valid_resp[i].cpu().item()):
                    rv = resp_vec_cpu[i]
                    (lc if labs_cpu[i].item() >= self._thr else li).append(rv)

    @staticmethod
    def _spherical_kmeans(x: Tensor, k: int, iters: int) -> Tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.size(0) == 0:
            return torch.zeros(0, x.size(-1)), torch.empty(0, dtype=torch.long)

        x = x.to(torch.float32)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

        n, _ = x.shape
        k = max(1, min(k, n))
        idx = torch.randperm(n)[:k]
        centers = x[idx].clone()

        for _ in range(iters):
            sim = x @ centers.t()
            assign = sim.argmax(dim=1)
            new_centers = []
            for j in range(k):
                sel: Tensor = assign == j
                if not torch.any(sel):
                    new_centers.append(x[torch.randint(0, n, (1,)).item()])
                else:
                    c = x[sel].mean(0)
                    c = c / c.norm().clamp_min(1e-12)
                    new_centers.append(c)
            new_centers = torch.stack(new_centers, 0)
            if torch.allclose(new_centers, centers, atol=1e-5, rtol=0.0):
                centers = new_centers
                break
            centers = new_centers

        sim = x @ centers.t()
        assign = sim.argmax(dim=1)
        return centers, assign

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
        if self._finalized:
            return

        master_L = (
            int(self._master_prompt_layer)
            if self._master_prompt_layer is not None
            else int(self._layers[len(self._layers) // 2])
        )

        all_ids = set()

        if self._is_mean_rep():
            all_ids |= set(self._sum_prompt.get(master_L, {}).keys())
        else:
            all_ids |= set(self._lst_prompt.get(master_L, {}).keys())

        for L in self._layers:
            all_ids |= set(self._sum_corr.get(L, {}).keys()) | set(self._sum_inc.get(L, {}).keys())
            if self._correction_mode == "cpca":
                all_ids |= set(self._lst_corr.get(L, {}).keys()) | set(self._lst_inc.get(L, {}).keys())

        self._task_ids_order = sorted(all_ids)

        if not self._task_ids_order:
            raise RuntimeError(
                "AtasVectorMediator.finalize: no valid ATAS task_id collected. "
                "All samples were skipped or fully masked, so routing centroids cannot be built. "
                "Common causes: truncation removed prompt/response spans, starts>=seq_len, "
                "or the training batch contains no usable task_id rows."
            )

        hh = int(self._hidden)
        original_kk = len(self._task_ids_order)

        missing_prompt: List[int] = []
        if self._is_mean_rep():
            for tid in self._task_ids_order:
                _, c = self._sum_prompt.get(master_L, {}).get(int(tid), (torch.zeros(hh), 0))
                if int(c) <= 0:
                    missing_prompt.append(int(tid))
        else:
            for tid in self._task_ids_order:
                lp = self._lst_prompt.get(master_L, {}).get(int(tid), [])
                if not lp:
                    missing_prompt.append(int(tid))

        if missing_prompt:
            preview = missing_prompt[:20]
            more = "" if len(missing_prompt) <= 20 else f" (+{len(missing_prompt) - 20} more)"
            base_msg = (
                "AtasVectorMediator.finalize: some task_id have NO prompt samples on master_prompt_layer.\n"
                f"- master_prompt_layer={master_L}\n"
                f"- missing prompt samples for {len(missing_prompt)}/{original_kk} task_id: {preview}{more}\n"
                "Common causes: left-truncation removing the prompt, starts==0, or prompt_len==0 after truncation.\n"
                "Fix: preserve prompt tokens (prefer truncation_side='right' for prompt+response), "
                "skip samples where prompt_len==0, or increase max_len."
            )

            if self._fail_on_missing_prompt_samples:
                warnings.warn(base_msg, RuntimeWarning, stacklevel=2)
                logger.error("%s", base_msg)
                raise RuntimeError(base_msg)

            drop_msg = base_msg + "\nDropping these task_id from final ATAS tables to avoid zero routing centroids."
            warnings.warn(drop_msg, RuntimeWarning, stacklevel=2)
            logger.error("%s", drop_msg)

            missing_set = set(missing_prompt)
            self._task_ids_order = [tid for tid in self._task_ids_order if tid not in missing_set]

            if not self._task_ids_order:
                raise RuntimeError(
                    "AtasVectorMediator.finalize: all collected ATAS task_id were dropped because they had "
                    "no valid prompt samples on master_prompt_layer."
                )

        kk = len(self._task_ids_order)
        pc_master = torch.zeros((kk, hh), dtype=torch.float32)

        if self._is_mean_rep():
            for j, tid in enumerate(self._task_ids_order):
                s, c = self._sum_prompt.get(master_L, {}).get(tid, (torch.zeros(hh), 0))
                nrm = s.norm().clamp_min(1e-12)
                pc_master[j] = s / nrm if c > 0 and float(nrm.item()) > 0.0 else torch.zeros(hh)
        else:
            for j, tid in enumerate(self._task_ids_order):
                lp = self._lst_prompt.get(master_L, {}).get(tid, [])
                if lp:
                    xp = torch.stack(lp, 0).to(torch.float32)
                    centers, assign = self._spherical_kmeans(xp, k=self._proto_k, iters=self._proto_iters)
                    if centers.numel() > 0 and assign.numel() > 0:
                        counts = torch.bincount(assign, minlength=centers.size(0))
                        j_star = int(torch.argmax(counts).item())
                        pc_master[j] = centers[j_star]
                    else:
                        mean_p = xp.mean(0)
                        pc_master[j] = mean_p / mean_p.norm().clamp_min(1e-12)
                else:
                    pc_master[j] = torch.zeros(hh)

        pc_master = pc_master / pc_master.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self._routing_centroids[int(master_L)] = pc_master.detach().to("cpu", torch.float32).contiguous()

        for L in self._layers:
            cc = torch.zeros((kk, hh), dtype=torch.float32)
            ic = torch.zeros((kk, hh), dtype=torch.float32)

            def _mean_var_from_sums(sum_vec: Tensor, sum_sq_vec: Tensor, count: int) -> Tuple[Tensor, Tensor]:
                if count is None or count <= 0:
                    z = torch.zeros(hh, dtype=torch.float32)
                    return z, z
                mean_vec = sum_vec / float(count)
                var_vec = (sum_sq_vec / float(count)) - (mean_vec * mean_vec)
                var_vec = torch.clamp(var_vec, min=0.0)
                return mean_vec, var_vec

            if self._is_mean_rep():
                for j, tid in enumerate(self._task_ids_order):
                    s_c, n_c = self._sum_corr[L].get(tid, (torch.zeros(hh), 0))
                    s2_c, _ = self._sum_corr_sq[L].get(tid, (torch.zeros(hh), 0))
                    mu_c, _ = _mean_var_from_sums(s_c, s2_c, n_c)

                    s_i, n_i = self._sum_inc[L].get(tid, (torch.zeros(hh), 0))
                    s2_i, _ = self._sum_inc_sq[L].get(tid, (torch.zeros(hh), 0))
                    mu_i, _ = _mean_var_from_sums(s_i, s2_i, n_i)

                    cc[j] = mu_c
                    ic[j] = mu_i

            elif self._rep == MediatorGroupRepresentativeEnum.PROTOTYPE:
                for j, tid in enumerate(self._task_ids_order):
                    lc = self._lst_corr[L].get(tid, [])
                    li = self._lst_inc[L].get(tid, [])

                    if lc:
                        xc = torch.stack(lc, 0).to(torch.float32)
                        cc[j] = xc.mean(0)
                    if li:
                        xi = torch.stack(li, 0).to(torch.float32)
                        ic[j] = xi.mean(0)
            else:
                raise RuntimeError("Unsupported representative type.")

            if self._correction_mode != "cpca":
                dv = (cc - ic).detach().to("cpu", torch.float32).contiguous()
                self._correction_vectors[int(L)] = dv
            else:
                dv = torch.zeros((kk, hh), dtype=torch.float32)
                for j, tid in enumerate(self._task_ids_order):
                    good_list = self._lst_corr[L].get(int(tid), [])
                    bad_list = self._lst_inc[L].get(int(tid), [])

                    if not good_list or not bad_list:
                        continue

                    xg = torch.stack(good_list, 0).to(torch.float32)
                    xb = torch.stack(bad_list, 0).to(torch.float32)
                    if int(xg.size(0)) == 0 or int(xb.size(0)) == 0:
                        continue

                    mu_g = xg.mean(0)
                    mu_b = xb.mean(0)

                    mu = 0.5 * (mu_g + mu_b)
                    xx = torch.cat([xg - mu, xb - mu], dim=0)

                    v = self._pc1(xx)

                    md = mu_g - mu_b
                    if torch.dot(v, md).item() < 0.0:
                        v = -v

                    dv[j] = v

                self._correction_vectors[int(L)] = dv.detach().to("cpu", torch.float32).contiguous()

        self._finalized = True

        self._sum_prompt.clear()
        self._sum_corr.clear()
        self._sum_inc.clear()
        self._sum_corr_sq.clear()
        self._sum_inc_sq.clear()
        self._lst_prompt.clear()
        self._lst_corr.clear()
        self._lst_inc.clear()

    def reset(self) -> None:
        if self._hidden is None:
            raise RuntimeError(
                "AtasVectorMediator not initialized: "
                "Call init_layers(layers, hidden_size) before reset()."
            )
        self.init_layers(self._layers, self._hidden)

    def get_multitask_params(self) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], List[int]]:
        if not self._finalized:
            raise RuntimeError("AtasVectorMediator.finalize() not yet called.")
        return dict(self._routing_centroids), dict(self._correction_vectors), list(self._task_ids_order)
