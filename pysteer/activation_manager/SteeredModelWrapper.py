from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Dict, Tuple, cast, Final, Callable, Optional, List

import torch
import torch.nn as nn

try:
    from transformers import LogitsProcessorList
except Exception:
    from transformers.generation import LogitsProcessorList

from enums.ApplyFromModeEnum import ApplyFromModeEnum
from steering_strategy.BaseSteeringStrategy import AlphaSpec
from steering_strategy.BaseSteeringStrategy import BaseSteeringStrategy
from utils.ModelUtils import ModelUtils

logger = logging.getLogger(__name__)

# Public so evaluation code can recognize the exact completion emitted by the
# ATAS hard-rejection path and score it according to the refusal policy.
ATAS_REFUSAL_TEXT = "I cannot provide an answer to this request due to policy reasons."


class _LazyForceTokensLogitsProcessor:
    def __init__(self, forced_token_ids: List[int], mask_getter: Callable[[], Optional[torch.Tensor]], start_len: int):
        self._forced = [int(x) for x in forced_token_ids]
        self._mask_getter = mask_getter
        self._start_len = int(start_len)
        self._cached_mask: Optional[torch.Tensor] = None

    def _broadcast_mask(self, bsz: int, device: torch.device) -> Optional[torch.Tensor]:
        m = self._cached_mask
        if m is None:
            m0 = self._mask_getter()
            if not torch.is_tensor(m0):
                return None
            self._cached_mask = m0.detach().to(dtype=torch.bool, device="cpu")
            m = self._cached_mask

        m = m.to(device=device, dtype=torch.bool)
        if m.numel() == 1:
            return m.expand(bsz)
        if m.size(0) == bsz:
            return m
        if m.size(0) == 1:
            return m.expand(bsz)
        if bsz % m.size(0) == 0:
            return m.repeat_interleave(bsz // m.size(0), dim=0)
        return None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        bsz = int(scores.size(0))
        dev = scores.device

        step = int(input_ids.size(1)) - self._start_len
        if step < 0:
            return scores

        m = self._broadcast_mask(bsz, dev)
        if m is None or (not bool(m.any().item())):
            return scores

        tok = self._forced[min(step, len(self._forced) - 1)]

        out = scores.clone()
        out[m] = -float("inf")
        out[m, tok] = 0.0
        return out


class SteeredModelWrapper(nn.Module):
    def __init__(
            self,
            base_model: nn.Module,
            strategy: Optional[BaseSteeringStrategy] = None,
            *,
            tokens_window: int = -1,
            alpha: AlphaSpec = 0.5,
            apply_from_token: int = -1,
            apply_from_mode: ApplyFromModeEnum = ApplyFromModeEnum.PROMPT_END,
            padding_side: str = "left",
            tokenizer=None,
    ):
        super().__init__()

        if not isinstance(base_model, nn.Module):
            raise TypeError("base_model must be an instance of torch.nn.Module")

        self._base_model: Final[nn.Module] = base_model

        self._cached_prompt_len: Optional[torch.Tensor] = None
        self._cached_seq_len: Optional[torch.Tensor] = None
        self._cached_left_pad: Optional[torch.Tensor] = None

        self._steering_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._hooks_active: bool = False

        self._padding_side: str = str(padding_side).lower()
        self._tokens_window: int = int(tokens_window)
        self._alpha: AlphaSpec = alpha
        self._apply_from_token: int = int(apply_from_token)
        self._apply_from_mode: ApplyFromModeEnum = apply_from_mode
        self._strategy: Optional[BaseSteeringStrategy] = None

        self._global_cursor: Optional[torch.Tensor] = None

        self._target_layers_set: set[int] = set()
        self._master_prompt_layer: Optional[int] = None
        self._master_is_target: bool = False
        self._is_atas_strategy: bool = False

        self._prompt_len_dev_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self._left_pad_dev_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self._starts_dev_cache: Dict[Tuple[str, int], torch.Tensor] = {}

        self._attn_cache_owner_id: Optional[int] = None
        self._attn_window_cache: Dict[Tuple[int, int, int, int, str, int], torch.Tensor] = {}

        base = ModelUtils.unwrap_model(self._base_model)
        self._pad_token_id = getattr(base.config, "pad_token_id", None)
        self._tokenizer = tokenizer

        self.cached_prompt_len = None
        self.cached_seq_len = None
        self.cached_left_pad = None
        self.cached_nonpad_2d: Optional[torch.Tensor] = None

        self.steering_hooks = []
        self.strategy = strategy
        self.alpha = alpha
        self.apply_from_token = int(apply_from_token)
        self.tokens_window = int(tokens_window)
        self.apply_from_mode = apply_from_mode
        self.padding_side = padding_side

    def steering_tensor_bytes(self) -> int:
        strat = self.strategy
        if strat is None:
            return 0

        tot = 0

        if hasattr(strat, "_dv_n_cpu"):
            dvn = getattr(strat, "_dv_n_cpu", None)
            if isinstance(dvn, dict):
                for v in dvn.values():
                    if torch.is_tensor(v):
                        tot += v.numel() * v.element_size()

        if hasattr(strat, "_pc_t_cpu"):
            pc = getattr(strat, "_pc_t_cpu", None)
            if torch.is_tensor(pc):
                tot += pc.numel() * pc.element_size()

        for name in ("prompt_centroids", "correct_centroids", "incorrect_centroids"):
            if hasattr(strat, name):
                d = getattr(strat, name)
                if isinstance(d, dict):
                    for v in d.values():
                        if torch.is_tensor(v):
                            tot += v.numel() * v.element_size()

        return int(tot)

    def _alpha_is_effectively_zero(self) -> bool:
        a = self.alpha
        if isinstance(a, dict):
            if not a:
                return True
            return all(math.isclose(float(v), 0.0, abs_tol=1e-9) for v in a.values())
        return math.isclose(float(a), 0.0, abs_tol=1e-9)

    def _get_eos_token_id(self) -> Optional[int]:
        tok = getattr(self, "_tokenizer", None)
        if tok is not None:
            eid = getattr(tok, "eos_token_id", None)
            if eid is not None:
                return int(eid)

        base = ModelUtils.unwrap_model(self._base_model)
        cfg = getattr(base, "config", None)
        if cfg is not None and getattr(cfg, "eos_token_id", None) is not None:
            return int(getattr(cfg, "eos_token_id"))

        gen_cfg = getattr(base, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "eos_token_id", None) is not None:
            return int(getattr(gen_cfg, "eos_token_id"))

        return None

    @staticmethod
    def _device_key(device: torch.device) -> Tuple[str, int]:
        return (device.type, -1 if device.index is None else int(device.index))

    def _clear_runtime_tensor_caches(self) -> None:
        self._prompt_len_dev_cache.clear()
        self._left_pad_dev_cache.clear()
        self._starts_dev_cache.clear()
        self._attn_window_cache.clear()
        self._attn_cache_owner_id = None

    def _sync_strategy_runtime_flags(self) -> None:
        strat = self._strategy
        if strat is None:
            self._target_layers_set = set()
            self._master_prompt_layer = None
            self._master_is_target = False
            self._is_atas_strategy = False
            return

        self._target_layers_set = {int(i) for i in getattr(strat, "target_layers", [])}
        mp = getattr(strat, "master_prompt_layer", None)
        self._master_prompt_layer = None if mp is None else int(mp)
        self._master_is_target = (
            self._master_prompt_layer in self._target_layers_set
            if self._master_prompt_layer is not None
            else False
        )
        self._is_atas_strategy = strat.__class__.__name__ == "AtasSteeringStrategy"

    def _get_cached_prompt_len_on(self, device: torch.device) -> torch.Tensor:
        if self.cached_prompt_len is None:
            raise RuntimeError("cached_prompt_len is None.")
        key = self._device_key(device)
        out = self._prompt_len_dev_cache.get(key)
        if out is None:
            out = self.cached_prompt_len.to(device=device, dtype=torch.int64)
            self._prompt_len_dev_cache[key] = out
        return out

    def _get_cached_left_pad_on(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.cached_left_pad is None:
            return None
        key = self._device_key(device)
        out = self._left_pad_dev_cache.get(key)
        if out is None:
            out = self.cached_left_pad.to(device=device, dtype=torch.int64)
            self._left_pad_dev_cache[key] = out
        return out

    def _get_cached_starts_on(self, device: torch.device) -> torch.Tensor:
        key = self._device_key(device)
        out = self._starts_dev_cache.get(key)
        if out is not None:
            return out

        prompt_len = self._get_cached_prompt_len_on(device)
        if self.apply_from_mode == ApplyFromModeEnum.PROMPT_END:
            out = prompt_len
        else:
            out = torch.full_like(prompt_len, int(self.apply_from_token))

        self._starts_dev_cache[key] = out
        return out

    @staticmethod
    def _attention_mask_to_nonpad_2d(
            attention_mask: Optional[torch.Tensor],
            *,
            device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        if not torch.is_tensor(attention_mask):
            return None

        am = attention_mask
        dev = device if device is not None else am.device

        if am.dtype == torch.bool:
            non_mask = am
        elif am.dtype.is_floating_point:
            if am.dim() <= 2:
                non_mask = am > 0.5
            else:
                non_mask = am >= 0
        else:
            non_mask = am != 0

        if non_mask.dim() == 4:
            x = non_mask
            if x.size(1) == 1:
                x = x.squeeze(1)
            else:
                x = x.any(dim=1)

            if x.dim() == 3:
                amk = x.any(dim=1)
            elif x.dim() == 2:
                amk = x
            else:
                return None

        elif non_mask.dim() == 3:
            x = non_mask
            amk = x.squeeze(1) if x.size(1) == 1 else x.any(dim=1)

        elif non_mask.dim() == 2:
            amk = non_mask

        elif non_mask.dim() == 1:
            amk = non_mask.unsqueeze(0)

        else:
            return None

        return amk.to(device=dev, dtype=torch.bool)

    @staticmethod
    def _infer_prompt_len_from_labels(
            *,
            labels: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            input_ids: Optional[torch.Tensor],
            pad_token_id: Optional[int],
    ) -> Optional[torch.Tensor]:
        if not (torch.is_tensor(labels) and labels.dim() == 2):
            return None

        dev = labels.device
        _, seq_len = labels.shape

        nonpad: Optional[torch.Tensor] = None
        if torch.is_tensor(attention_mask):
            am2d = SteeredModelWrapper._attention_mask_to_nonpad_2d(attention_mask, device=dev)
            if torch.is_tensor(am2d) and am2d.dim() == 2 and am2d.shape == labels.shape:
                nonpad = am2d.to(device=dev, dtype=torch.bool)
        elif (
                torch.is_tensor(input_ids)
                and input_ids.dim() == 2
                and input_ids.shape == labels.shape
                and pad_token_id is not None
        ):
            nonpad = input_ids.ne(int(pad_token_id)).to(device=dev, dtype=torch.bool)
        else:
            return None

        if nonpad is None:
            return None

        pos = torch.arange(seq_len, device=dev, dtype=torch.int64).unsqueeze(0).expand(labels.size(0), -1)
        resp = (labels != -100) & nonpad

        sentinel = torch.full_like(pos, seq_len)
        first_resp = torch.where(resp, pos, sentinel).min(dim=1).values

        prompt_len = (nonpad & (pos < first_resp.unsqueeze(1))).sum(dim=1, dtype=torch.int64)
        return prompt_len

    def _ensure_global_cursor(self, bsz: int, device: torch.device) -> torch.Tensor:
        if self._global_cursor is None:
            if self.cached_prompt_len is None:
                self._global_cursor = torch.zeros((bsz,), dtype=torch.int64, device=device)
            else:
                self._global_cursor = self._get_cached_prompt_len_on(device).clone()

        cur = self._global_cursor.to(device=device, dtype=torch.int64)
        if cur.size(0) != bsz:
            if cur.size(0) == 1:
                cur = cur.expand(bsz)
            elif bsz % cur.size(0) == 0:
                cur = cur.repeat_interleave(bsz // cur.size(0), dim=0)
            else:
                raise RuntimeError(
                    f"SteeredModelWrapper: cannot broadcast global cursor to batch. "
                    f"cursor={tuple(cur.shape)} bsz={bsz}"
                )

        self._global_cursor = cur
        return cur

    def _cache_nonpad_2d_from_inputs(
            self,
            *,
            attention_mask: Optional[torch.Tensor],
            input_ids: Optional[torch.Tensor],
    ) -> None:
        self._attn_window_cache.clear()
        self._attn_cache_owner_id = None

        am2d = self._attention_mask_to_nonpad_2d(attention_mask)
        if torch.is_tensor(am2d) and am2d.dim() == 2:
            self.cached_nonpad_2d = am2d.to(dtype=torch.bool)
            return

        pad_id = self._pad_token_id
        if torch.is_tensor(input_ids) and input_ids.dim() == 2 and pad_id is not None:
            self.cached_nonpad_2d = input_ids.ne(int(pad_id)).to(dtype=torch.bool)
            return

        self.cached_nonpad_2d = None

    def _get_nonpad_step_mask(
            self,
            *,
            bsz: int,
            ss: int,
            past_len: int,
            device: torch.device,
    ) -> torch.Tensor:
        m = self.cached_nonpad_2d
        if torch.is_tensor(m) and m.dim() == 2:
            m = m.to(device=device, dtype=torch.bool)
            if m.size(0) != bsz:
                if m.size(0) == 1:
                    m = m.expand(bsz, -1)
                elif bsz % m.size(0) == 0:
                    m = m.repeat_interleave(bsz // m.size(0), dim=0)
                else:
                    m = None

            if torch.is_tensor(m):
                s_total = int(m.size(1))
                start = max(0, min(int(past_len), s_total))
                end = max(0, min(int(past_len) + int(ss), s_total))
                win = m[:, start:end]

                if win.size(1) < ss:
                    logger.warning(
                        "SteeredModelWrapper: cached nonpad mask shorter than decode chunk "
                        "(got %d < ss=%d). Padding on the RIGHT to keep alignment.",
                        int(win.size(1)),
                        int(ss),
                    )
                    pad_is_valid = bool(int(past_len) >= int(s_total))
                    pad_val = True if pad_is_valid else False
                    pad = torch.full(
                        (win.size(0), ss - win.size(1)),
                        fill_value=pad_val,
                        dtype=torch.bool,
                        device=device,
                    )
                    win = torch.cat([win, pad], dim=1)
                elif win.size(1) > ss:
                    win = win[:, :ss]

                return win

        if self.padding_side == "left":
            lp = self._get_cached_left_pad_on(device)
            if torch.is_tensor(lp) and lp.size(0) != bsz:
                if lp.size(0) == 1:
                    lp = lp.expand(bsz)
                elif bsz % lp.size(0) == 0:
                    lp = lp.repeat_interleave(bsz // lp.size(0), dim=0)
                else:
                    lp = None

            if torch.is_tensor(lp):
                pos = torch.arange(past_len, past_len + ss, device=device, dtype=torch.int64).unsqueeze(0)
                return pos >= lp.unsqueeze(1)

        return torch.ones((bsz, ss), dtype=torch.bool, device=device)

    @property
    def base_model(self) -> nn.Module:
        return self._base_model

    @base_model.setter
    def base_model(self, value: nn.Module) -> None:
        raise AttributeError("SteeredModelWrapper.base_model is immutable after initialization.")

    @property
    def cached_left_pad(self) -> Optional[torch.Tensor]:
        return self._cached_left_pad

    @cached_left_pad.setter
    def cached_left_pad(self, value: Optional[torch.Tensor]) -> None:
        self._cached_left_pad = value
        self._clear_runtime_tensor_caches()

    @property
    def padding_side(self) -> str:
        return self._padding_side

    @padding_side.setter
    def padding_side(self, value: str) -> None:
        value = str(value).lower()
        if value not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'")
        self._padding_side = value
        self._clear_runtime_tensor_caches()

    @property
    def tokens_window(self) -> int:
        return self._tokens_window

    @tokens_window.setter
    def tokens_window(self, value: int) -> None:
        self._tokens_window = int(value)

    @property
    def steering_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        return self._steering_hooks

    @steering_hooks.setter
    def steering_hooks(self, value: List[torch.utils.hooks.RemovableHandle]) -> None:
        self._steering_hooks = value

    @property
    def cached_prompt_len(self) -> Optional[torch.Tensor]:
        return self._cached_prompt_len

    @cached_prompt_len.setter
    def cached_prompt_len(self, value: Optional[torch.Tensor]) -> None:
        self._cached_prompt_len = value
        self._clear_runtime_tensor_caches()

    @property
    def cached_seq_len(self) -> Optional[torch.Tensor]:
        return self._cached_seq_len

    @cached_seq_len.setter
    def cached_seq_len(self, value: Optional[torch.Tensor]) -> None:
        self._cached_seq_len = value

    @property
    def strategy(self) -> Optional[BaseSteeringStrategy]:
        return self._strategy

    @strategy.setter
    def strategy(self, value: Optional[BaseSteeringStrategy]) -> None:
        was_active = getattr(self, "_hooks_active", False)

        if was_active:
            self.remove_hooks()

        self._strategy = value
        self._sync_strategy_runtime_flags()
        self._validate_alpha_against_strategy()

        if self._strategy is not None and hasattr(self._strategy, "prepare_alpha"):
            self._strategy.prepare_alpha(self._alpha)

        if was_active and value is not None:
            self._register_hooks()

    @property
    def alpha(self) -> AlphaSpec:
        return self._alpha

    @alpha.setter
    def alpha(self, v: AlphaSpec) -> None:
        if isinstance(v, dict):
            self._alpha = {int(k): float(val) for k, val in v.items()}
        else:
            self._alpha = float(v)

        self._validate_alpha_against_strategy()

        strat = self.strategy
        if strat is not None and hasattr(strat, "prepare_alpha"):
            strat.prepare_alpha(self._alpha)

    def _validate_alpha_against_strategy(self) -> None:
        a = self._alpha
        if not isinstance(a, dict):
            return

        strat = self.strategy
        if strat is None:
            return

        if strat.__class__.__name__ != "AtasSteeringStrategy":
            raise TypeError("Per-task alpha dictionaries are supported only with AtasSteeringStrategy.")

        tids = getattr(strat, "task_ids", None)
        if tids is None:
            raise RuntimeError("Per-task alpha was requested, but strategy.task_ids is None.")

        tids_set = {int(t) for t in tids}
        keys = {int(k) for k in a.keys()}
        if len(a) != len(tids) or keys != tids_set:
            missing = sorted(list(tids_set - keys))
            extra = sorted(list(keys - tids_set))
            raise ValueError(
                "Invalid per-task alpha: it must contain every ATAS task_id exactly once.\n"
                f"- expected ({len(tids)}): {sorted(tids_set)}\n"
                f"- received ({len(a)}): {sorted(keys)}\n"
                f"- missing: {missing}\n"
                f"- extra: {extra}"
            )

    @property
    def apply_from_token(self) -> int:
        return self._apply_from_token

    @apply_from_token.setter
    def apply_from_token(self, v: int) -> None:
        self._apply_from_token = int(v)
        self._starts_dev_cache.clear()

    @property
    def apply_from_mode(self) -> ApplyFromModeEnum:
        return self._apply_from_mode

    @apply_from_mode.setter
    def apply_from_mode(self, m: ApplyFromModeEnum) -> None:
        if not isinstance(m, ApplyFromModeEnum):
            raise TypeError("apply_from_mode expects ApplyFromModeEnum")
        self._apply_from_mode = m
        self._starts_dev_cache.clear()

    @staticmethod
    def _extract_hidden_from_output(output: Any) -> Tuple[Optional[torch.Tensor], str, Any]:
        if torch.is_tensor(output):
            return output, "tensor", None
        if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
            return output[0], "tuple", None
        if isinstance(output, list) and len(output) > 0 and torch.is_tensor(output[0]):
            return output[0], "list", None

        for attr in ("last_hidden_state", "hidden_states", "hidden_state"):
            if hasattr(output, attr):
                t = getattr(output, attr)
                if torch.is_tensor(t):
                    return t, "attr", attr

        try:
            if isinstance(output, dict):
                for k in ("last_hidden_state", "hidden_states", "hidden_state"):
                    if k in output and torch.is_tensor(output[k]):
                        return output[k], "dict", k
        except Exception:
            pass

        try:
            t0 = output[0]
            if torch.is_tensor(t0):
                return t0, "getitem0", None
        except Exception:
            pass

        return None, "unknown", None

    @staticmethod
    def _replace_hidden_in_output(output: Any, new_hidden: torch.Tensor, kind: str, meta: Any) -> Any:
        if kind == "tensor":
            return new_hidden

        if kind == "tuple":
            return (new_hidden,) + output[1:]

        if kind == "list":
            out = list(output)
            out[0] = new_hidden
            return out

        if kind == "dict":
            key = cast(str, meta)
            out = cast(Dict[str, Any], dict(output))
            out[key] = new_hidden
            return out

        if kind == "attr":
            attr = meta
            if dataclasses.is_dataclass(output):
                try:
                    return dataclasses.replace(output, **{attr: new_hidden})
                except Exception:
                    pass

            try:
                data: Dict[str, Any]
                if hasattr(output, "to_dict") and callable(getattr(output, "to_dict")):
                    data = cast(Dict[str, Any], output.to_dict())
                else:
                    data = cast(Dict[str, Any], dict(output))
                data[attr] = new_hidden
                return output.__class__(**data)
            except Exception:
                pass

            try:
                setattr(output, attr, new_hidden)
                return output
            except Exception as e:
                raise RuntimeError(
                    f"SteeredModelWrapper: failed to replace hidden via attr='{attr}' "
                    f"for output type={type(output).__name__}: {e}"
                ) from e

        raise RuntimeError(
            f"SteeredModelWrapper: unsupported output container kind='{kind}' "
            f"(type={type(output).__name__}); cannot apply steering."
        )

    def _extract_2d_attention_mask(
            self,
            att_mask: Optional[torch.Tensor],
            _inputs,
            _kwargs,
            ss: int,
            past_len: int,
            device: torch.device,
    ) -> torch.Tensor:
        am = att_mask
        if am is None and isinstance(_inputs, (tuple, list)) and len(_inputs) >= 2 and torch.is_tensor(_inputs[1]):
            am = _inputs[1]

        if am is None:
            bsz = _inputs[0].size(0) if (isinstance(_inputs, (tuple, list)) and torch.is_tensor(_inputs[0])) else 1
            return torch.ones((bsz, ss), dtype=torch.bool, device=device)

        bsz = _inputs[0].size(0) if (isinstance(_inputs, (tuple, list)) and torch.is_tensor(_inputs[0])) else 1
        am_id = id(am)
        if self._attn_cache_owner_id != am_id:
            self._attn_window_cache.clear()
            self._attn_cache_owner_id = am_id

        dkey = self._device_key(device)
        cache_key = (am_id, int(past_len), int(ss), int(bsz), dkey[0], dkey[1])
        cached = self._attn_window_cache.get(cache_key)
        if cached is not None:
            return cached

        amk = SteeredModelWrapper._attention_mask_to_nonpad_2d(am, device=device)
        if amk is None:
            return torch.ones((bsz, ss), dtype=torch.bool, device=device)

        if amk.dim() == 2 and amk.size(1) == ss:
            out = amk
            self._attn_window_cache[cache_key] = out
            return out

        k_len = int(amk.size(1))
        start = max(0, min(int(past_len), k_len))
        end = max(0, min(int(past_len) + int(ss), k_len))
        window = amk[:, start:end]

        if window.size(1) < ss:
            pad_is_valid = bool(int(past_len) >= int(k_len))
            pad_val = True if pad_is_valid else False
            pad = torch.full(
                (window.size(0), ss - window.size(1)),
                fill_value=pad_val,
                dtype=torch.bool,
                device=device,
            )
            window = torch.cat([window, pad], dim=1)
        elif window.size(1) > ss:
            window = window[:, :ss]

        out = window.to(dtype=torch.bool, device=device)
        self._attn_window_cache[cache_key] = out
        return out

    def _make_hook(self, layer_idx: int):
        @torch.inference_mode()
        def hook(_module, args, kwargs, output):
            if kwargs is None:
                kwargs = {}
            elif not isinstance(kwargs, dict):
                raise RuntimeError(
                    f"SteeredModelWrapper: unexpected kwargs type in hook for layer {layer_idx}: "
                    f"{type(kwargs).__name__}"
                )

            strat = self._strategy
            _inputs = args

            hidden, kind, meta = self._extract_hidden_from_output(output)
            if hidden is None:
                raise RuntimeError(
                    f"SteeredModelWrapper: could not extract hidden states from output on layer {layer_idx} "
                    f"(kind={kind}, meta={meta})."
                )
            if hidden.dim() != 3:
                raise RuntimeError(
                    f"SteeredModelWrapper: expected hidden dim=3 on layer {layer_idx}, got shape={tuple(hidden.shape)}."
                )

            bsz, ss, _ = hidden.shape
            device = hidden.device

            is_target = int(layer_idx) in self._target_layers_set
            is_master = (
                    self._master_prompt_layer is not None
                    and int(layer_idx) == int(self._master_prompt_layer)
            )

            cursor_owner_layer = max(self._target_layers_set) if self._target_layers_set else int(layer_idx)
            is_cursor_owner = int(layer_idx) == int(cursor_owner_layer)

            att_mask = kwargs.get("attention_mask", None)
            if att_mask is None and isinstance(_inputs, (tuple, list)) and len(_inputs) >= 2 and torch.is_tensor(_inputs[1]):
                att_mask = _inputs[1]

            pkv = (
                    kwargs.get("past_key_value")
                    or kwargs.get("past_key_values")
                    or kwargs.get("layer_past")
            )
            has_pkv = pkv is not None and not (isinstance(pkv, (tuple, list)) and len(pkv) == 0)

            past_len = 0
            if torch.is_tensor(att_mask):
                try:
                    if att_mask.dim() == 2:
                        k_len = int(att_mask.size(1))
                    elif att_mask.dim() in (3, 4):
                        k_len = int(att_mask.size(-1))
                    else:
                        k_len = 0
                    if k_len >= ss and k_len > 0:
                        past_len = k_len - ss
                except Exception:
                    past_len = 0

            if past_len == 0:
                try:
                    if (
                            isinstance(pkv, (tuple, list))
                            and len(pkv) >= 2
                            and torch.is_tensor(pkv[0])
                            and pkv[0].dim() >= 3
                    ):
                        past_len = int(pkv[0].shape[-2])
                    elif isinstance(pkv, (tuple, list)) and len(pkv) > layer_idx:
                        pair = pkv[layer_idx]
                        if (
                                isinstance(pair, (tuple, list))
                                and len(pair) >= 2
                                and torch.is_tensor(pair[0])
                                and pair[0].dim() >= 3
                        ):
                            past_len = int(pair[0].shape[-2])
                    elif hasattr(pkv, "get_seq_length"):
                        past_len = int(pkv.get_seq_length())
                except Exception:
                    past_len = 0

            prefill_like = has_pkv and (ss > 1) and (past_len == ss)
            fallback_decode = (not has_pkv) and (past_len == 0) and (ss == 1) and (self._global_cursor is not None)
            is_decode = (not prefill_like) and (((past_len > 0) and (ss == 1 or past_len > ss)) or fallback_decode)
            past_len_eff = 0 if prefill_like else past_len
            has_reliable_past_len = bool(has_pkv or past_len_eff > 0)

            if is_decode:
                nonpad_step = torch.ones((bsz, ss), dtype=torch.bool, device=device)
            else:
                nonpad_step = self._get_nonpad_step_mask(
                    bsz=bsz,
                    ss=ss,
                    past_len=past_len_eff,
                    device=device,
                )

            nonpad_step &= self._extract_2d_attention_mask(
                att_mask, _inputs, kwargs, ss=ss, past_len=past_len_eff, device=device
            )

            if self.cached_prompt_len is None:
                raise RuntimeError(
                    f"SteeredModelWrapper: cached_prompt_len is None on layer {layer_idx}. "
                    f"Call wrapper.forward()/wrapper.generate() to initialize caches."
                )

            if is_decode and is_master and not is_target:
                return output

            starts_np = self._get_cached_starts_on(device)

            if starts_np.size(0) != bsz:
                if starts_np.size(0) == 1:
                    starts_np = starts_np.expand(bsz)
                elif bsz % starts_np.size(0) == 0:
                    factor = bsz // starts_np.size(0)
                    starts_np = starts_np.repeat_interleave(factor, dim=0)
                else:
                    raise RuntimeError(
                        f"SteeredModelWrapper: cannot broadcast starts to batch. "
                        f"starts={tuple(starts_np.shape)} bsz={bsz} layer={layer_idx}"
                    )

            if is_decode:
                if has_reliable_past_len:
                    if ss == 1:
                        tok_idx_np = torch.full((bsz, 1), past_len_eff, dtype=torch.int64, device=device)
                    else:
                        ar = torch.arange(ss, device=device, dtype=torch.int64).unsqueeze(0)
                        tok_idx_np = ar + int(past_len_eff)
                else:
                    cur = self._ensure_global_cursor(bsz=bsz, device=device)
                    if ss == 1:
                        tok_idx_np = cur.unsqueeze(1)
                    else:
                        ar = torch.arange(ss, device=device, dtype=torch.int64).unsqueeze(0)
                        tok_idx_np = cur.unsqueeze(1) + ar
            else:
                rank = nonpad_step.to(torch.int64).cumsum(dim=1) - 1
                tok_idx_np = rank

            try:
                if (not is_decode) and (strat is not None) and hasattr(strat, "ingest_prompt"):
                    rl = self._master_prompt_layer
                    if (rl is None) or (int(layer_idx) == int(rl)):
                        prompt_mask = nonpad_step & (tok_idx_np >= 0) & (tok_idx_np < starts_np.unsqueeze(1))
                        if prompt_mask.any():
                            strat.ingest_prompt(layer_idx, hidden, prompt_mask)
            except Exception as e:
                raise RuntimeError(f"Ingest_prompt failed on layer {layer_idx}: {e}") from e

            if is_master and not is_target:
                return output

            if strat is None or self._alpha_is_effectively_zero():
                if is_decode and (not has_reliable_past_len) and is_cursor_owner:
                    step_inc = nonpad_step.to(torch.int64).sum(dim=1)
                    self._global_cursor = self._ensure_global_cursor(bsz=bsz, device=device) + step_inc
                return output

            mask_tok = (tok_idx_np >= starts_np.unsqueeze(1)) & nonpad_step & (tok_idx_np >= 0)

            if self.tokens_window > 0:
                ends_np = starts_np + self.tokens_window
                mask_tok &= tok_idx_np < ends_np.unsqueeze(1)

            if not mask_tok.any():
                if is_decode and (not has_reliable_past_len) and is_cursor_owner:
                    step_inc = nonpad_step.to(torch.int64).sum(dim=1)
                    self._global_cursor = self._ensure_global_cursor(bsz=bsz, device=device) + step_inc
                return output

            steered = strat.steer(
                layer_idx,
                hidden_states=hidden,
                alpha=self.alpha,
                mask_tok=mask_tok,
            ).to(hidden.dtype)

            if is_decode and (not has_reliable_past_len) and is_cursor_owner:
                step_inc = nonpad_step.to(torch.int64).sum(dim=1)
                self._global_cursor = self._ensure_global_cursor(bsz=bsz, device=device) + step_inc

            return self._replace_hidden_in_output(output, steered, kind, meta)

        return hook
    def _register_hooks(self) -> None:
        if not self._strategy:
            return

        self.remove_hooks()
        layers = ModelUtils.find_transformer_layers(self.base_model)

        hook_layers = sorted(self._target_layers_set)
        if self._master_prompt_layer is not None:
            hook_layers = sorted(set(hook_layers) | {int(self._master_prompt_layer)})

        for idx in hook_layers:
            if idx < 0 or idx >= len(layers):
                raise IndexError(f"Layer index {idx} out of range (0–{len(layers) - 1})")
            try:
                h = layers[idx].register_forward_hook(self._make_hook(idx), with_kwargs=True)
            except TypeError as e:
                raise RuntimeError(
                    "This wrapper requires register_forward_hook(..., with_kwargs=True). "
                    "Please use torch >= 1.13 (recommended 2.x)."
                ) from e
            self._steering_hooks.append(h)

        self._hooks_active = True

    def remove_hooks(self) -> None:
        for h in self._steering_hooks:
            try:
                h.remove()
            except (RuntimeError, AttributeError):
                pass
        self._steering_hooks.clear()
        self._hooks_active = False

    def forward(self, *args, **kwargs):
        if self._strategy is not None and not getattr(self, "_hooks_active", False):
            raise RuntimeError(
                "SteeredModelWrapper used outside of a context manager. "
                "Use `with wrapper as m:` to activate steering hooks."
            )
        if self.apply_from_mode == ApplyFromModeEnum.FIXED and self.apply_from_token < 0:
            raise ValueError("apply_from_token must be >= 0 when apply_from_mode=FIXED.")

        self._attn_window_cache.clear()
        self._attn_cache_owner_id = None

        pkv = (
                kwargs.get("past_key_values")
                or kwargs.get("past_key_value")
                or kwargs.get("layer_past")
        )
        is_prefill = pkv is None or (isinstance(pkv, (tuple, list)) and len(pkv) == 0)
        if is_prefill and hasattr(self.strategy, "reset_prompt_state"):
            self.strategy.reset_prompt_state()

        prompt_lengths = kwargs.pop("prompt_lengths", None)
        left_pad_lengths = kwargs.pop("left_pad_lengths", None)

        if prompt_lengths is not None:
            if not torch.is_tensor(prompt_lengths):
                prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.int64)
            else:
                prompt_lengths = prompt_lengths.to(dtype=torch.int64)

            if self.cached_prompt_len is None or is_prefill:
                self.cached_prompt_len = prompt_lengths

            attn_tmp = kwargs.get("attention_mask", None)
            ids_tmp = kwargs.get("input_ids", None)
            attn_tmp_2d = self._attention_mask_to_nonpad_2d(attn_tmp) if torch.is_tensor(attn_tmp) else None

            if torch.is_tensor(attn_tmp_2d) and attn_tmp_2d.dim() == 2:
                self.cached_seq_len = torch.full_like(prompt_lengths, int(attn_tmp_2d.size(1)))
            elif torch.is_tensor(ids_tmp) and ids_tmp.dim() == 2:
                self.cached_seq_len = torch.full_like(prompt_lengths, int(ids_tmp.size(1)))

            if self.padding_side == "left":
                if left_pad_lengths is None:
                    raise ValueError(
                        "left_pad_lengths is required when padding_side='left' and prompt_lengths is provided."
                    )
                if not torch.is_tensor(left_pad_lengths):
                    left_pad_lengths = torch.tensor(left_pad_lengths, dtype=torch.int64)
                else:
                    left_pad_lengths = left_pad_lengths.to(dtype=torch.int64)
                self.cached_left_pad = left_pad_lengths
            else:
                self.cached_left_pad = torch.zeros_like(prompt_lengths)

            self._cache_nonpad_2d_from_inputs(
                attention_mask=attn_tmp_2d,
                input_ids=ids_tmp if torch.is_tensor(ids_tmp) else None,
            )

            if is_prefill:
                self._global_cursor = self._get_cached_prompt_len_on(torch.device(prompt_lengths.device)).clone()

            return self.base_model(*args, **kwargs)

        attn_raw = kwargs.get("attention_mask", None)
        attn: Optional[torch.Tensor] = attn_raw if torch.is_tensor(attn_raw) else None
        attn_2d = self._attention_mask_to_nonpad_2d(attn) if attn is not None else None
        labels = kwargs.get("labels", None)

        if attn_2d is not None:
            nonpad_len = attn_2d.sum(1, dtype=torch.int64).to(attn_2d.device)

            if self.cached_prompt_len is None or is_prefill:
                inferred = None
                if torch.is_tensor(labels):
                    inferred = self._infer_prompt_len_from_labels(
                        labels=labels,
                        attention_mask=attn_2d,
                        input_ids=kwargs.get("input_ids") if torch.is_tensor(kwargs.get("input_ids", None)) else None,
                        pad_token_id=self._pad_token_id,
                    )
                self.cached_prompt_len = inferred.to(attn_2d.device) if inferred is not None else nonpad_len

            self.cached_seq_len = torch.full_like(nonpad_len, int(attn_2d.size(1)))
            if self.padding_side == "left":
                self.cached_left_pad = int(attn_2d.size(1)) - nonpad_len
            else:
                self.cached_left_pad = torch.zeros_like(nonpad_len)

            self._cache_nonpad_2d_from_inputs(
                attention_mask=attn_2d,
                input_ids=kwargs.get("input_ids") if torch.is_tensor(kwargs.get("input_ids", None)) else None,
            )
        else:
            ids = kwargs.get("input_ids")
            if ids is None:
                raise ValueError("Missing attention_mask and input_ids; cannot infer prompt length.")

            dev = ids.device
            bsz, seq_len = ids.shape
            pad_id = self._pad_token_id

            if pad_id is not None:
                am = ids.ne(pad_id).to(dtype=torch.int64)
                nonpad_len = am.sum(1, dtype=torch.int64)
            else:
                nonpad_len = torch.full((bsz,), seq_len, dtype=torch.int64, device=dev)

            if self.cached_prompt_len is None or is_prefill:
                inferred = None
                if torch.is_tensor(labels):
                    inferred = self._infer_prompt_len_from_labels(
                        labels=labels,
                        attention_mask=None,
                        input_ids=ids,
                        pad_token_id=self._pad_token_id,
                    )
                self.cached_prompt_len = inferred.to(dev) if inferred is not None else nonpad_len

            self.cached_seq_len = torch.full_like(nonpad_len, seq_len)

            if self.padding_side == "left":
                if pad_id is not None:
                    self.cached_left_pad = seq_len - nonpad_len
                else:
                    self.cached_left_pad = torch.zeros_like(nonpad_len)
            else:
                self.cached_left_pad = torch.zeros_like(nonpad_len)

            self._cache_nonpad_2d_from_inputs(attention_mask=None, input_ids=ids)

        if is_prefill and self.cached_prompt_len is not None:
            self._global_cursor = self._get_cached_prompt_len_on(self.cached_prompt_len.device).clone()

        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if self._strategy is not None and not getattr(self, "_hooks_active", False):
            raise RuntimeError(
                "SteeredModelWrapper.generate() called outside of a context manager. "
                "Use `with wrapper as m:` to activate steering hooks."
            )
        if self.apply_from_mode == ApplyFromModeEnum.FIXED and self.apply_from_token < 0:
            raise ValueError("apply_from_token must be >= 0 when apply_from_mode=FIXED.")

        self._attn_window_cache.clear()
        self._attn_cache_owner_id = None

        nb = int(kwargs.get("num_beams", 1) or 1)
        if nb > 1:
            raise RuntimeError(
                "SteeredModelWrapper.generate(): beam search (num_beams>1) is not supported yet because "
                "cached prompt/left_pad masks are not reordered with beams. Use num_beams=1."
            )

        attn_raw = kwargs.get("attention_mask", None)
        attn = attn_raw if torch.is_tensor(attn_raw) else None
        attn_2d = self._attention_mask_to_nonpad_2d(attn) if attn is not None else None

        pkv = (
                kwargs.get("past_key_values")
                or kwargs.get("past_key_value")
                or kwargs.get("layer_past")
        )
        is_prefill = pkv is None or (isinstance(pkv, (tuple, list)) and len(pkv) == 0)

        if is_prefill and hasattr(self.strategy, "reset_prompt_state"):
            self.strategy.reset_prompt_state()

        self.cached_prompt_len = None
        self.cached_seq_len = None
        self.cached_left_pad = None
        self.cached_nonpad_2d = None
        self._global_cursor = None

        if attn_2d is not None:
            nonpad_len = attn_2d.sum(1, dtype=torch.int64).to(attn_2d.device)

            if self.cached_prompt_len is None or pkv is None:
                self.cached_prompt_len = nonpad_len

            self.cached_seq_len = torch.full_like(self.cached_prompt_len, int(attn_2d.size(1)))

            if self.padding_side == "left":
                self.cached_left_pad = attn_2d.size(1) - nonpad_len
            else:
                self.cached_left_pad = torch.zeros_like(nonpad_len)

            self._cache_nonpad_2d_from_inputs(
                attention_mask=attn_2d,
                input_ids=kwargs.get("input_ids") if torch.is_tensor(kwargs.get("input_ids", None)) else None,
            )
        else:
            ids = kwargs.get("input_ids")
            if ids is None:
                raise ValueError("Missing attention_mask and input_ids; cannot infer prompt length.")

            dev = ids.device
            bsz, seq_len = ids.shape
            pad_id = self._pad_token_id

            if self.cached_prompt_len is None or pkv is None:
                if pad_id is not None:
                    nonpad_len = ids.ne(pad_id).sum(1, dtype=torch.int64).to(dev)
                    self.cached_prompt_len = nonpad_len
                else:
                    nonpad_len = torch.full((bsz,), seq_len, dtype=torch.int64, device=dev)
                    self.cached_prompt_len = nonpad_len

            self.cached_seq_len = torch.full_like(self.cached_prompt_len, seq_len)

            if self.padding_side == "left":
                if pad_id is not None:
                    nonpad_len_now = ids.ne(pad_id).sum(1, dtype=torch.int64).to(dev)
                else:
                    nonpad_len_now = torch.full((bsz,), seq_len, dtype=torch.int64, device=dev)
                self.cached_left_pad = seq_len - nonpad_len_now
            else:
                self.cached_left_pad = torch.zeros_like(self.cached_prompt_len)

            self._cache_nonpad_2d_from_inputs(attention_mask=None, input_ids=ids)

        if self.cached_prompt_len is not None:
            self._global_cursor = self._get_cached_prompt_len_on(self.cached_prompt_len.device).clone()

        if (self.strategy is not None) and self._is_atas_strategy:
            if self._tokenizer is None:
                raise RuntimeError("SteeredModelWrapper: ATAS hard refusal requires passing tokenizer to the wrapper.")

            input_ids = kwargs.get("input_ids", None)
            if input_ids is None:
                if len(args) >= 1 and torch.is_tensor(args[0]):
                    input_ids = args[0]
                else:
                    raise RuntimeError("SteeredModelWrapper: hard ATAS refusal requires input_ids in generate().")

            start_len = int(input_ids.size(1))

            forced_text = ATAS_REFUSAL_TEXT
            enc = self._tokenizer(forced_text, add_special_tokens=False, return_tensors=None)
            forced_ids = enc.get("input_ids", [])
            if forced_ids and isinstance(forced_ids[0], list):
                forced_ids = forced_ids[0]
            forced_ids = [int(x) for x in forced_ids]

            eos_id = self._get_eos_token_id()
            if eos_id is not None:
                forced_ids = forced_ids + [int(eos_id)]

            need_new = len(forced_ids)
            if "max_new_tokens" in kwargs and kwargs["max_new_tokens"] is not None:
                try:
                    kwargs["max_new_tokens"] = max(int(kwargs["max_new_tokens"]), need_new)
                except Exception:
                    pass
            elif "max_length" in kwargs and kwargs["max_length"] is not None:
                try:
                    kwargs["max_length"] = max(int(kwargs["max_length"]), start_len + need_new)
                except Exception:
                    pass

            proc = _LazyForceTokensLogitsProcessor(
                forced_token_ids=forced_ids,
                mask_getter=self.strategy.compute_reject_mask,
                start_len=start_len,
            )

            lp = kwargs.get("logits_processor", None)
            if lp is None:
                kwargs["logits_processor"] = LogitsProcessorList([proc])
            elif isinstance(lp, LogitsProcessorList):
                kwargs["logits_processor"] = LogitsProcessorList(list(lp) + [proc])
            else:
                try:
                    kwargs["logits_processor"] = LogitsProcessorList(list(lp) + [proc])
                except Exception:
                    kwargs["logits_processor"] = LogitsProcessorList([lp, proc])

        return self.base_model.generate(*args, **kwargs)

    def reset_prompt_state(self):
        self.cached_prompt_len = None
        self.cached_seq_len = None
        self.cached_left_pad = None
        self.cached_nonpad_2d = None
        self._global_cursor = None
        self._clear_runtime_tensor_caches()
        if hasattr(self.strategy, "reset_prompt_state"):
            self.strategy.reset_prompt_state()

    def __del__(self):
        try:
            if hasattr(self, "_steering_hooks"):
                self.remove_hooks()
        except Exception:
            pass

    def __enter__(self):
        if self._strategy is None:
            self._hooks_active = True
            return self

        try:
            self._register_hooks()
            if hasattr(self._strategy, "prepare_alpha"):
                try:
                    self._strategy.prepare_alpha(self.alpha)
                except Exception:
                    pass
            return self
        except Exception:
            self.remove_hooks()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False
