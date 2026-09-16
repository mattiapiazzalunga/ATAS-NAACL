import contextlib
import logging
import warnings
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from utils.ModelUtils import ModelUtils

logger = logging.getLogger(__name__)


class ActivationExtractor:

    def __init__(
            self,
            model: Any,
            layers_to_extract: Union[Iterable[int], int],
            *,
            offload_to_cpu: bool = True,
            decode_chunk_max: int = 1,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("ActivationExtractor expects 'model' to be a torch.nn.Module")

        self._model = model
        self.layers_to_extract = layers_to_extract
        self.handles: List[RemovableHandle] = []
        self._offload_to_cpu = bool(offload_to_cpu)

        base = ModelUtils.unwrap_model(self._model)
        self._hidden_size = ModelUtils.get_hidden_size(base.config)
        self._layers = list(ModelUtils.find_transformer_layers(self._model))

        self._hook_errors: List[tuple[int, Exception]] = []
        self._chunks: Dict[int, List[Tensor]] = {}
        self._activations: Dict[int, Tensor] = {}

        self._seen_seq_len: Dict[int, int] = {}
        self._phase: Dict[int, str] = {}
        self._prefill_len: Dict[int, int] = {}
        self._decode_chunk_max: int = max(1, int(decode_chunk_max))

        for idx in self.layers_to_extract:
            if idx < 0 or idx >= len(self._layers):
                raise IndexError(
                    f"Layer index {idx} out of range for model with {len(self._layers)} layers."
                )

    def _reset_layer_state(self, idx: int) -> None:
        self._chunks.pop(idx, None)
        self._activations.pop(idx, None)
        self._seen_seq_len.pop(idx, None)
        self._phase.pop(idx, None)
        self._prefill_len.pop(idx, None)

    @property
    def model(self) -> Any:
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        raise AttributeError("ActivationExtractor.model is immutable after initialization.")

    @property
    def layers_to_extract(self) -> List[int]:
        return self._layers_to_extract

    @layers_to_extract.setter
    def layers_to_extract(self, value: Union[Iterable[int], int]) -> None:
        if isinstance(value, int):
            value = [value]
        self._layers_to_extract = sorted({int(i) for i in value})
        if not self._layers_to_extract:
            raise ValueError("At least one layer must be specified for extraction")

    @property
    def handles(self) -> List[RemovableHandle]:
        return self._handles

    @handles.setter
    def handles(self, value: List[RemovableHandle]) -> None:
        self._handles = value

    @property
    def activations(self) -> Dict[int, Tensor]:
        return self._activations

    def _make_hook(self, idx: int):
        @torch.inference_mode()
        def hook(_module, inputs, output):
            try:
                if hasattr(output, "last_hidden_state"):
                    t = output.last_hidden_state
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    t = output[0]
                else:
                    t = output

                if not torch.is_tensor(t):
                    raise RuntimeError(
                        f"ActivationExtractor: layer {idx} hook output is not a tensor "
                        f"(type={type(t).__name__})."
                    )

                t = ModelUtils.ensure_bsh(
                    t,
                    self._hidden_size,
                    from_layout="BSH",
                )

                if not (torch.is_tensor(t) and t.dim() == 3):
                    raise RuntimeError(
                        f"ActivationExtractor: layer {idx} expected 3D tensor (B,S,H), "
                        f"got {None if not torch.is_tensor(t) else tuple(t.shape)}."
                    )

                s = int(t.shape[1])

                already_have = bool(self._chunks.get(idx)) or (idx in self._activations)
                if already_have and s > self._decode_chunk_max:
                    msg = (
                        f"ActivationExtractor: detected seq_len={s} (> decode_chunk_max={self._decode_chunk_max}) "
                        f"on layer {idx} while previous activations are still accumulated. "
                        "This often means a new prefill happened without clear(). "
                        "Resetting accumulation for this layer to avoid mixing."
                    )
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    logger.warning("%s", msg)
                    self._reset_layer_state(idx)

                phase = self._phase.get(idx)
                if phase is None:
                    if s > 1:
                        self._phase[idx] = "prefill"
                        self._prefill_len[idx] = s
                    else:
                        self._phase[idx] = "decode"
                    self._seen_seq_len[idx] = s
                else:
                    if phase == "prefill":
                        prefill_s = self._prefill_len.get(idx, self._seen_seq_len.get(idx, s))

                        if s == prefill_s:
                            self._seen_seq_len[idx] = s
                        elif s <= self._decode_chunk_max:
                            self._phase[idx] = "decode"
                            self._seen_seq_len[idx] = s
                        else:
                            raise RuntimeError(
                                f"ActivationExtractor: layer {idx} saw unexpected seq_len change during prefill: "
                                f"prefill_seq_len={prefill_s} new_seq_len={s}. "
                                "Call clear() between independent forwards/generations."
                            )
                    else:
                        if s <= self._decode_chunk_max:
                            self._seen_seq_len[idx] = s
                        else:
                            msg = (
                                f"ActivationExtractor: layer {idx} saw seq_len={s} during decode phase; "
                                f"expected <= {self._decode_chunk_max}. "
                                "This likely indicates an accidental new prefill without clear(). "
                                "Resetting this layer state and treating this as a new prefill."
                            )
                            warnings.warn(msg, RuntimeWarning, stacklevel=2)
                            logger.warning("%s", msg)
                            self._reset_layer_state(idx)
                            self._phase[idx] = "prefill" if s > 1 else "decode"
                            if s > 1:
                                self._prefill_len[idx] = s
                            self._seen_seq_len[idx] = s

                t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).detach()

                if self._offload_to_cpu:
                    if t.device.type == "cuda":
                        t = t.to("cpu", non_blocking=True)
                    else:
                        t = t.to("cpu")

                chunks = self._chunks.setdefault(idx, [])
                prev = chunks[-1] if chunks else self._activations.get(idx, None)

                if (
                        prev is not None
                        and (
                        (not torch.is_tensor(prev))
                        or prev.dim() != 3
                        or t.shape[0] != prev.shape[0]
                        or t.shape[2] != prev.shape[2]
                )
                ):
                    msg = (
                        f"ActivationExtractor: shape mismatch on layer {idx}: "
                        f"prev={None if prev is None else tuple(prev.shape)} new={tuple(t.shape)}."
                    )
                    logger.warning("%s Resetting accumulation.", msg)
                    self._reset_layer_state(idx)
                    raise RuntimeError(msg)

                chunks.append(t)

            except Exception as e:
                self._reset_layer_state(idx)
                self._hook_errors.append((idx, e))
                raise
        return hook

    def assert_ok(self) -> None:
        if self._hook_errors:
            idx, e = self._hook_errors[0]
            raise RuntimeError(f"Activation hook failed on layer {idx}: {e}") from e

    def finalize(self, *, clear_chunks: bool = True) -> Dict[int, Tensor]:
        for idx, chunks in self._chunks.copy().items():
            if not chunks:
                continue

            if idx in self._activations:
                pieces: List[Tensor] = [self._activations.pop(idx)] + chunks
            else:
                pieces = chunks

            try:
                b0 = int(pieces[0].shape[0])
                h0 = int(pieces[0].shape[2])
            except Exception:
                if clear_chunks:
                    self._chunks.pop(idx, None)
                raise RuntimeError(f"ActivationExtractor.finalize: bad tensor shape at layer {idx}.")

            ok_mask: List[bool] = []
            for x in pieces:
                ok_mask.append(
                    torch.is_tensor(x)
                    and x.dim() == 3
                    and int(x.shape[0]) == b0
                    and int(x.shape[2]) == h0
                )

            ok = [pieces[i] for i, good in enumerate(ok_mask) if good]
            if len(ok) != len(pieces):
                bad_shapes = [
                    None if not torch.is_tensor(pieces[i]) else tuple(pieces[i].shape)
                    for i, good in enumerate(ok_mask) if not good
                ]
                raise RuntimeError(
                    f"ActivationExtractor.finalize: incompatible chunks at layer {idx}: {bad_shapes}"
                )

            if not ok:
                if clear_chunks:
                    self._chunks.pop(idx, None)
                continue

            self._activations[idx] = torch.cat(ok, dim=1) if len(ok) > 1 else ok[0]

            if clear_chunks:
                self._chunks.pop(idx, None)
            else:
                self._chunks[idx] = []

        if clear_chunks:
            self._chunks.clear()

        return self._activations

    def attach(self, *, clear_activations: bool = True) -> None:
        self._detach_hooks_only()
        if clear_activations:
            self.clear()

        self._layers = list(ModelUtils.find_transformer_layers(self._model))
        try:
            base = ModelUtils.unwrap_model(self._model)
            cfg = getattr(base, "config", None)
            expected = None
            for k in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
                if cfg is not None and hasattr(cfg, k):
                    expected = int(getattr(cfg, k))
                    break
            if expected is not None and 0 < expected != len(self._layers):
                logger.warning(
                    "ActivationExtractor: find_transformer_layers() found %d layers, "
                    "but config expects %d. Hooks may target wrong modules (pre/post norm mismatch risk).",
                    len(self._layers), expected
                )
        except Exception:
            pass
        n = len(self._layers)
        bad = [i for i in self.layers_to_extract if i < 0 or i >= n]
        if bad:
            raise IndexError(f"Layer index(es) {bad} out of range for model with {n} layers.")

        new_handles: List[RemovableHandle] = []
        try:
            for idx in self.layers_to_extract:
                block = self._layers[idx]
                h = block.register_forward_hook(self._make_hook(idx))
                new_handles.append(h)
            self.handles = new_handles
        except Exception as e:
            self.remove()
            raise RuntimeError(f"Failed to attach activation hooks: {e}") from e

    def clear(self) -> None:
        self._chunks.clear()
        self._activations.clear()
        self._hook_errors.clear()
        self._seen_seq_len.clear()
        self._phase.clear()
        self._prefill_len.clear()

    def _detach_hooks_only(self) -> None:
        for h in self.handles.copy():
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    def remove(self) -> None:
        self._detach_hooks_only()
        self.clear()

    def close(self) -> None:
        self.remove()

    def __enter__(self):
        self.attach(clear_activations=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False

    @contextlib.contextmanager
    def temporarily_disabled(self):
        was_on = bool(self.handles)
        if was_on:
            self._detach_hooks_only()
        try:
            yield
        finally:
            if was_on:
                self.attach(clear_activations=False)
