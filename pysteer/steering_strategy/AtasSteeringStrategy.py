import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as ff
from steering_strategy.BaseSteeringStrategy import BaseSteeringStrategy, AlphaSpec
from torch import Tensor

logger = logging.getLogger(__name__)


class AtasSteeringStrategy(BaseSteeringStrategy):
    def __init__(
            self,
            routing_centroids: Dict[int, Tensor],
            correction_vectors: Dict[int, Tensor],
            task_ids: Optional[List[int]] = None,
            *,
            similarity_threshold: float = 0.0,
            use_last_token_for_prompt: bool = True,
            normalize_eps: float = 1e-12,
            master_prompt_layer: Optional[int] = None,
            cpu_store_dtype: torch.dtype = torch.float16,
            keep_raw_correction_vectors: bool = False,
            random_routing_seed: Optional[int] = None,
    ) -> None:
        eps = float(normalize_eps)
        self._normalize_eps = eps
        self._cpu_store_dtype = cpu_store_dtype
        self._keep_raw_correction_vectors = bool(keep_raw_correction_vectors)

        _cv_cpu = {
            int(k): v.detach().to("cpu", torch.float32).contiguous()
            for k, v in correction_vectors.items()
        }
        self._correction_vectors = _cv_cpu if self._keep_raw_correction_vectors else {}

        self.target_layers = sorted(_cv_cpu.keys())
        if not self.target_layers:
            raise ValueError("AtasSteeringStrategy: empty correction_vectors.")

        if master_prompt_layer is None:
            self.master_prompt_layer = int(self.target_layers[len(self.target_layers) // 2])
        else:
            self.master_prompt_layer = int(master_prompt_layer)

        if int(self.master_prompt_layer) not in routing_centroids:
            raise ValueError(
                f"routing_centroids missing master_prompt_layer={self.master_prompt_layer}. "
                f"available={sorted(int(k) for k in routing_centroids.keys())}"
            )

        pc = routing_centroids[int(self.master_prompt_layer)].detach().to("cpu", torch.float32).contiguous()
        if pc.ndim != 2 or int(pc.size(0)) == 0 or int(pc.size(1)) == 0:
            raise ValueError(
                "AtasSteeringStrategy: routing_centroids for master_prompt_layer is empty or invalid. "
                "Training produced zero valid ATAS tasks."
            )

        bad_rows = (pc.norm(dim=-1) <= eps).nonzero(as_tuple=False).squeeze(1).tolist()
        if bad_rows:
            preview = bad_rows[:20]
            more = "" if len(bad_rows) <= 20 else f" (+{len(bad_rows) - 20} more)"
            raise ValueError(
                "AtasSteeringStrategy: routing_centroids contains zero-norm rows on "
                f"master_prompt_layer={self.master_prompt_layer}. "
                "These tasks cannot be routed safely. "
                f"Bad rows: {preview}{more}. "
                "Drop tasks with missing prompt samples before building the strategy."
            )

        if task_ids is not None:
            tids = [int(t) for t in task_ids]
            if len(tids) == 0:
                raise ValueError(
                    "AtasSteeringStrategy: task_ids is empty. Training produced zero valid ATAS tasks."
                )
            if len(tids) != int(pc.size(0)):
                raise ValueError(
                    f"AtasSteeringStrategy: task_ids length ({len(tids)}) does not match "
                    f"routing_centroids rows ({int(pc.size(0))})."
                )
            self.task_ids = tids
        else:
            self.task_ids = None

        pc_n = ff.normalize(pc, dim=-1, eps=eps)
        self._pc_t_cpu = pc_n.transpose(0, 1).contiguous().to(dtype=self._cpu_store_dtype)

        self._dv_n_cpu: Dict[int, Tensor] = {}
        for k, dv in _cv_cpu.items():
            dn = dv.norm(dim=-1, keepdim=True).clamp_min(eps)
            self._dv_n_cpu[int(k)] = (dv / dn).contiguous().to(dtype=self._cpu_store_dtype)

        self._tensor_cache: Dict[tuple[int, str, int], Dict[str, Tensor]] = {}
        self.similarity_threshold = float(similarity_threshold)
        self.use_last_token_for_prompt = bool(use_last_token_for_prompt)
        self.random_routing_seed = None if random_routing_seed is None else int(random_routing_seed)
        self._random_generator: Optional[torch.Generator] = None
        if self.random_routing_seed is not None:
            self._random_generator = torch.Generator(device="cpu")
            self._random_generator.manual_seed(self.random_routing_seed)
        self._prompt_mu_routing: Optional[Tensor] = None
        self._routing_idx: Optional[Tensor] = None
        self._routing_sim: Optional[Tensor] = None
        self._chosen_global: Optional[Tensor] = None
        self._state_bsz: Optional[int] = None
        self._forced_task_ids_cpu: Optional[Tensor] = None
        self._forced_routing_idx_cpu: Optional[Tensor] = None

        self._alpha_table_cpu: Optional[Tensor] = None
        self._alpha_dev_cache: Dict[tuple[str, int], Tensor] = {}

    def clear_cache(self) -> None:
        self._tensor_cache.clear()
        self._alpha_table_cpu = None
        self._alpha_dev_cache.clear()

    def prepare_alpha(self, alpha: AlphaSpec) -> None:
        self._alpha_table_cpu = None
        self._alpha_dev_cache.clear()

        if not isinstance(alpha, dict):
            return

        if self.task_ids is None:
            raise RuntimeError("AtasSteeringStrategy.prepare_alpha: task_ids is None.")

        self._alpha_table_cpu = torch.tensor(
            [float(alpha[int(tid)]) for tid in self.task_ids],
            dtype=torch.float32,
            device="cpu",
        ).contiguous()

    def export_params(self) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Optional[List[int]]]:
        routing = {
            int(self.master_prompt_layer): (
                self._pc_t_cpu.transpose(0, 1)
                .detach()
                .to("cpu", torch.float32)
                .contiguous()
            )
        }
        corr = {
            int(k): v.detach().to("cpu", torch.float32).contiguous()
            for k, v in self.correction_vectors.items()
        }
        tids = None if self.task_ids is None else [int(t) for t in self.task_ids]
        return routing, corr, tids

    @property
    def correction_vectors(self) -> Dict[int, Tensor]:
        if self._correction_vectors:
            return {
                int(k): v.detach().to("cpu", torch.float32).contiguous()
                for k, v in self._correction_vectors.items()
            }

        return {
            int(k): v.detach().to("cpu", torch.float32).contiguous()
            for k, v in self._dv_n_cpu.items()
        }

    def _get_layer_tensors(self, layer_idx: int, device: torch.device) -> Dict[str, Tensor]:
        dev_idx = -1 if device.index is None else int(device.index)
        key = (int(layer_idx), device.type, dev_idx)
        cached = self._tensor_cache.get(key)
        if cached is not None:
            return cached

        work_dtype = torch.float32
        out: Dict[str, Tensor] = {}

        if int(layer_idx) in self._dv_n_cpu:
            out["dv_n"] = self._dv_n_cpu[int(layer_idx)].to(device=device, dtype=work_dtype)

        if int(layer_idx) == int(self.master_prompt_layer):
            out["pc_t"] = self._pc_t_cpu.to(device=device, dtype=work_dtype)

        self._tensor_cache[key] = out
        return out

    def _sample_random_routing_idx(self, bsz: int) -> Tensor:
        n_tasks = int(self._pc_t_cpu.size(1))
        if n_tasks <= 0:
            raise RuntimeError("AtasSteeringStrategy: cannot sample random routing from empty task table.")
        if self._random_generator is None:
            raise RuntimeError("AtasSteeringStrategy: random routing requested but generator is None.")
        return torch.randint(
            low=0,
            high=n_tasks,
            size=(int(bsz),),
            generator=self._random_generator,
            device="cpu",
        )

    def reseed_random_routing(self, seed: int) -> None:
        """Reset random routing so each evaluation task can use an independent deterministic stream."""
        self.random_routing_seed = int(seed)
        if self._random_generator is None:
            self._random_generator = torch.Generator(device="cpu")
        self._random_generator.manual_seed(self.random_routing_seed)

    def set_forced_task_ids(self, task_ids) -> None:
        """Force routing to externally predicted task IDs for the next generation calls."""
        if task_ids is None:
            self.clear_forced_task_ids()
            return
        if self.task_ids is None:
            raise RuntimeError("AtasSteeringStrategy: cannot force task IDs when task_ids is None.")

        forced = torch.as_tensor(task_ids, dtype=torch.long, device="cpu").flatten().contiguous()
        if forced.numel() == 0:
            raise ValueError("AtasSteeringStrategy: forced task IDs cannot be empty.")

        row_by_task_id = {int(task_id): row for row, task_id in enumerate(self.task_ids)}
        unknown = sorted({int(task_id) for task_id in forced.tolist()} - set(row_by_task_id))
        if unknown:
            raise ValueError(
                f"AtasSteeringStrategy: forced task IDs are not present in the routing table: {unknown}."
            )

        self._forced_task_ids_cpu = forced
        self._forced_routing_idx_cpu = torch.tensor(
            [row_by_task_id[int(task_id)] for task_id in forced.tolist()],
            dtype=torch.long,
            device="cpu",
        ).contiguous()

    def clear_forced_task_ids(self) -> None:
        self._forced_task_ids_cpu = None
        self._forced_routing_idx_cpu = None
        self._chosen_global = None

    def _forced_routing_idx(self, bsz: int, device: torch.device) -> Optional[Tensor]:
        forced = self._forced_routing_idx_cpu
        if forced is None:
            return None

        forced = forced.to(device=device, dtype=torch.long)
        if forced.size(0) == int(bsz):
            return forced
        if forced.size(0) == 1:
            return forced.expand(int(bsz))
        if int(bsz) % forced.size(0) == 0:
            return forced.repeat_interleave(int(bsz) // forced.size(0), dim=0)
        raise RuntimeError(
            "AtasSteeringStrategy: cannot broadcast forced routing IDs "
            f"from batch {forced.size(0)} to batch {int(bsz)}."
        )

    @property
    def target_layers(self) -> List[int]:
        return self._target_layers

    @target_layers.setter
    def target_layers(self, layers: List[int]) -> None:
        self._target_layers = layers

    @torch.inference_mode()
    def ingest_prompt(self, layer_idx: int, hidden_states: Tensor, prompt_mask: Tensor) -> None:
        if int(layer_idx) != int(self.master_prompt_layer):
            return

        if not (torch.is_tensor(hidden_states) and hidden_states.dim() == 3):
            return
        if not (torch.is_tensor(prompt_mask) and prompt_mask.dim() == 2):
            return
        if not prompt_mask.any():
            return

        hs = hidden_states.to(torch.float32)
        pmask = prompt_mask.to(hs.device)

        bb, ss, hh = hs.shape

        if self._state_bsz is not None and self._state_bsz != bb:
            self._routing_idx = None
            self._routing_sim = None
            self._chosen_global = None
        self._state_bsz = bb

        if self.use_last_token_for_prompt:
            pmask_bool = pmask.to(torch.bool)
            positions = torch.arange(ss, device=hs.device).unsqueeze(0).expand(bb, -1)
            masked_pos = positions.masked_fill(~pmask_bool, -1)
            last_pos = masked_pos.max(dim=1).values

            mu = torch.zeros((bb, hh), device=hs.device, dtype=hs.dtype)
            valid = last_pos >= 0
            if valid.any():
                idx2 = last_pos[valid]
                b_idx = torch.arange(bb, device=hs.device)[valid]
                mu[valid] = hs[b_idx, idx2, :]
        else:
            m = pmask.unsqueeze(-1)
            den = m.sum(dim=1).clamp_min(1.0)
            mu = (hs * m).sum(dim=1) / den

        self._prompt_mu_routing = mu

        forced_idx = self._forced_routing_idx(bb, hs.device)
        if forced_idx is not None:
            self._routing_idx = forced_idx.detach()
            self._routing_sim = None
            self._chosen_global = forced_idx.detach()
            return

        eps = self._normalize_eps
        dev = mu.device
        tt = self._get_layer_tensors(layer_idx, dev)
        pc_t = tt.get("pc_t", None)
        if pc_t is None or pc_t.ndim != 2 or int(pc_t.size(1)) == 0:
            raise RuntimeError(
                "AtasSteeringStrategy: empty routing table on master_prompt_layer. "
                "Training produced zero valid ATAS tasks."
            )

        mu_n = ff.normalize(mu.to(torch.float32), dim=-1, eps=eps)
        sims = mu_n @ pc_t
        sims_max, idx = sims.max(dim=-1)
        routing_idx = idx.detach()

        chosen = (
            self._sample_random_routing_idx(bb).to(device=hs.device)
            if self._random_generator is not None
            else routing_idx
        )
        self._routing_idx = routing_idx
        self._routing_sim = sims_max.detach()
        self._chosen_global = chosen.detach()

    @torch.inference_mode()
    def compute_reject_mask(self) -> Optional[torch.Tensor]:
        if self._forced_routing_idx_cpu is not None:
            return None

        thr = float(getattr(self, "similarity_threshold", 0.0) or 0.0)
        if thr <= 0.0:
            return None

        sims_max = self._routing_sim
        if sims_max is None:
            mu = self._prompt_mu_routing
            if mu is None:
                return None
            eps = self._normalize_eps
            tt = self._get_layer_tensors(int(self.master_prompt_layer), mu.device)
            pc_t = tt.get("pc_t", None)
            if pc_t is None or pc_t.ndim != 2 or int(pc_t.size(1)) == 0:
                raise RuntimeError(
                    "AtasSteeringStrategy: empty routing table on master_prompt_layer. "
                    "Training produced zero valid ATAS tasks."
                )
            mu_n = ff.normalize(mu.to(torch.float32), dim=-1, eps=eps)
            sims = mu_n @ pc_t
            sims_max = sims.max(dim=-1).values

        if not (torch.is_tensor(sims_max) and sims_max.dim() == 1):
            return None

        return sims_max.to(torch.float32) < thr

    @torch.inference_mode()
    def steer(self, layer_idx: int, hidden_states: Tensor, alpha: AlphaSpec, mask_tok: Tensor) -> Tensor:
        if layer_idx not in self.target_layers or not mask_tok.any():
            return hidden_states

        orig_dtype = hidden_states.dtype
        dev = hidden_states.device
        work_dtype = torch.float32

        hs = hidden_states.to(work_dtype).contiguous()

        chosen = self._chosen_global
        if chosen is None:
            forced_idx = self._forced_routing_idx(int(hs.size(0)), dev)
            if forced_idx is not None:
                chosen = forced_idx.detach()
                self._routing_idx = forced_idx.detach()
                self._routing_sim = None
                self._chosen_global = chosen

        if chosen is None:
            mu = self._prompt_mu_routing
            if mu is None:
                return hidden_states
            eps = self._normalize_eps
            tt_rl = self._get_layer_tensors(int(self.master_prompt_layer), mu.device)
            pc_t = tt_rl.get("pc_t", None)
            if pc_t is None or pc_t.ndim != 2 or int(pc_t.size(1)) == 0:
                raise RuntimeError(
                    "AtasSteeringStrategy: empty routing table on master_prompt_layer. "
                    "Training produced zero valid ATAS tasks."
                )
            mu_n = ff.normalize(mu.to(torch.float32), dim=-1, eps=eps)
            sims = mu_n @ pc_t
            routing_idx = sims.argmax(dim=-1).detach()
            chosen = (
                self._sample_random_routing_idx(int(mu.size(0))).to(device=mu.device)
                if self._random_generator is not None
                else routing_idx
            )
            self._routing_idx = routing_idx
            self._chosen_global = chosen

        bsz = int(hs.size(0))
        chosen = chosen.to(device=dev, dtype=torch.long)
        if chosen.size(0) != bsz:
            if chosen.size(0) == 1:
                chosen = chosen.expand(bsz)
            elif (bsz % chosen.size(0)) == 0:
                chosen = chosen.repeat_interleave(bsz // chosen.size(0), dim=0)
            else:
                raise RuntimeError(f"AtasSteeringStrategy: cannot broadcast chosen to bsz={bsz}")

        tt = self._get_layer_tensors(layer_idx, dev)
        dv_n = tt.get("dv_n", None)
        if dv_n is None:
            raise RuntimeError(
                f"AtasSteeringStrategy: missing correction vectors for target layer {layer_idx}."
            )
        delta = dv_n.index_select(0, chosen)

        if isinstance(alpha, dict):
            if self._alpha_table_cpu is None:
                self.prepare_alpha(alpha)

            dev_idx = -1 if dev.index is None else int(dev.index)
            k = (dev.type, dev_idx)
            alpha_table = self._alpha_dev_cache.get(k)
            if alpha_table is None:
                alpha_table = self._alpha_table_cpu.to(device=dev, dtype=work_dtype)
                self._alpha_dev_cache[k] = alpha_table

            alpha_b = alpha_table.index_select(0, chosen)
        else:
            alpha_b = torch.full((bsz,), float(alpha), device=dev, dtype=work_dtype)

        mask = mask_tok.to(dtype=torch.bool, device=dev)
        _, ss, _ = hs.shape

        if ss == 1:
            m = mask[:, 0]
            if m.any():
                hs[m, 0, :] += alpha_b[m].unsqueeze(1) * delta[m]
        else:
            b_idx, s_idx = mask.nonzero(as_tuple=True)
            if b_idx.numel() > 0:
                hs[b_idx, s_idx, :] += alpha_b.index_select(0, b_idx).unsqueeze(1) * delta.index_select(0, b_idx)

        if orig_dtype in (torch.float16, torch.bfloat16):
            finfo = torch.finfo(orig_dtype)
            hs.clamp_(min=-finfo.max, max=finfo.max)

        return hs.to(orig_dtype)

    def reset_prompt_state(self) -> None:
        self._prompt_mu_routing = None
        self._routing_idx = None
        self._routing_sim = None
        self._chosen_global = None
        self._state_bsz = None
