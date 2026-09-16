from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, cast

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from activation_manager.ActivationExtractor import ActivationExtractor
from activation_manager.SteeredModelWrapper import SteeredModelWrapper
from activation_manager.VectorMediator import VectorMediator
from enums.ApplyFromModeEnum import ApplyFromModeEnum
from enums.MediatorTypeEnum import MediatorTypeEnum
from enums.TaskTypeEnum import TaskTypeEnum
from steering_strategy.AtasSteeringStrategy import AtasSteeringStrategy
from steering_strategy.BaseSteeringStrategy import AlphaSpec
from steering_strategy.BaseSteeringStrategy import BaseSteeringStrategy
from steering_strategy.GeneralSteeringStrategy import GeneralSteeringStrategy
from utils.ModelUtils import ModelUtils
from vector_update_strategy.AtasVectorMediator import AtasVectorMediator
from vector_update_strategy.BaseVectorUpdateStrategy import BaseVectorUpdateStrategy
from vector_update_strategy.CmdVectorMediator import CmdVectorMediator
from vector_update_strategy.CpcaVectorMediator import CpcaVectorMediator

logger = logging.getLogger(__name__)


def _count_forward_hooks(model: nn.Module) -> int:
    base = ModelUtils.unwrap_model(model)
    tot = 0
    for m in base.modules():
        tot += len(getattr(m, "_forward_hooks", {}))
        tot += len(getattr(m, "_forward_pre_hooks", {}))
    return tot


def _assert_no_forward_hooks(model: nn.Module, where: str = "") -> None:
    base = ModelUtils.unwrap_model(model)
    n = _count_forward_hooks(base)
    if n != 0:
        raise RuntimeError(
            f"[HOOK LEAK] Found {n} forward/pre-forward hooks on base(unwrapped) model {where}. "
            "This can contaminate baseline generations."
        )


@contextmanager
def _attached_activation_extractor(extractor: ActivationExtractor):
    extractor.attach()
    try:
        yield
    finally:
        extractor.remove()


class PromptDataset(Dataset):
    def __init__(self, data_iter: Sequence[Dict[str, Any]]) -> None:
        self._data = list(data_iter)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._data[idx]


def identity_collate(batch):
    return batch


class Executor:
    def __init__(
            self,
            model: nn.Module,
            tokenizer,
            train_df: pd.DataFrame,
            mediator_type: MediatorTypeEnum = MediatorTypeEnum.CMD,
            layers_to_extract: Union[Iterable[int], int] = (0,),
            atas_master_layer: Optional[int] = None,
            batch_size: int = 4,
            padding_side: Optional[str] = None,
            truncation_side: Optional[str] = None,
            dtype: torch.dtype = torch.float16,
            alpha: AlphaSpec = 0.5,
            generation_kwargs: Optional[Dict[str, Any]] = None,
            atas_similarity_threshold: float = 0.0,
            *,
            tokens_window: int = -1,
            task_type: TaskTypeEnum = TaskTypeEnum.UNSUPERVISED,
            apply_from_token: int = -1,
            apply_from_mode: ApplyFromModeEnum = ApplyFromModeEnum.PROMPT_END,
            atas_use_last_token_for_prompt: bool = True,
            use_last_token_for_response: bool = True,
            decode_chunk_max: int = 1,
            atas_random_routing_seed: Optional[int] = None,
            free_training_artifacts_after_build: bool = True,
            max_prompt_tokens: Optional[int] = 1024,
            max_response_tokens: Optional[int] = 512,
            max_sequence_tokens: Optional[int] = 1536,
    ) -> None:
        self._generation_kwargs: Dict[str, Any] = {}
        self._vector_update_strategy: Optional[BaseVectorUpdateStrategy] = None
        self._activation_extractor: Optional[ActivationExtractor] = None
        self._vector_mediator: Optional[VectorMediator] = None
        self._steering_strategy: Optional[BaseSteeringStrategy] = None
        n_layers = len(ModelUtils.find_transformer_layers(model))
        if n_layers <= 0:
            raise RuntimeError("Cannot set default ATAS routing layer: no transformer layers found in model.")
        self.atas_master_layer = (n_layers // 2) if atas_master_layer is None else int(atas_master_layer)

        self.generation_kwargs = generation_kwargs or {}

        self.atas_similarity_threshold: float = float(atas_similarity_threshold)
        self.atas_use_last_token_for_prompt: bool = bool(atas_use_last_token_for_prompt)
        self.use_last_token_for_response: bool = bool(use_last_token_for_response)
        self.atas_random_routing_seed: Optional[int] = (
            None if atas_random_routing_seed is None
            else int(atas_random_routing_seed)
        )
        self.tokenizer = tokenizer
        self.padding_side = (tokenizer.padding_side if padding_side is None else padding_side).lower()
        self.truncation_side = (
            getattr(tokenizer, "truncation_side", "left") if truncation_side is None else truncation_side
        ).lower()

        self.model = model
        self._ensure_tokenizer_has_pad_token()

        self.layers_to_extract = layers_to_extract
        self.batch_size = batch_size
        self.alpha = alpha
        self.apply_from_token = apply_from_token
        self.task_type = task_type
        self.apply_from_mode = apply_from_mode
        self.mediator_type = mediator_type
        self.dtype = dtype
        self.tokens_window = int(tokens_window)

        self.train_df: pd.DataFrame = pd.DataFrame(train_df)
        self.unique_refs: List[Any] = []
        self.max_sequence_tokens: int = (
            1536 if max_sequence_tokens is None else max(1, int(max_sequence_tokens))
        )
        self.max_prompt_tokens: int = (
            1024 if max_prompt_tokens is None else max(1, int(max_prompt_tokens))
        )
        self.max_response_tokens: int = (
            512 if max_response_tokens is None else max(1, int(max_response_tokens))
        )
        if self.max_prompt_tokens + self.max_response_tokens > self.max_sequence_tokens:
            raise ValueError(
                "Executor token budgets are inconsistent: max_prompt_tokens + max_response_tokens "
                "must be <= max_sequence_tokens."
            )
        self.last_teacher_forcing_truncation_stats: Dict[str, int] = {
            "rows": 0,
            "responses_truncated": 0,
            "max_response_tokens_before_truncation": 0,
        }
        self.decode_chunk_max: int = max(1, int(decode_chunk_max))

        self.dataset = self.prepare_dataset_to_train(self.train_df)
        self._setup_vector_components()
        self._free_training_artifacts_after_build: bool = bool(free_training_artifacts_after_build)

    def _ensure_tokenizer_has_pad_token(self) -> None:
        tok = self.tokenizer
        if tok is None:
            raise ValueError("Executor requires a tokenizer.")

        if getattr(tok, "pad_token_id", None) is not None:
            return

        eos_token = getattr(tok, "eos_token", None)
        eos_token_id = getattr(tok, "eos_token_id", None)

        try:
            if eos_token is not None:
                tok.pad_token = eos_token
            elif eos_token_id is not None and hasattr(tok, "convert_ids_to_tokens"):
                eos_token = tok.convert_ids_to_tokens(int(eos_token_id))
                if eos_token is None:
                    raise ValueError("convert_ids_to_tokens returned None")
                tok.pad_token = eos_token
            else:
                raise ValueError("missing eos_token/eos_token_id")
        except Exception as e:
            raise ValueError(
                "Tokenizer has no pad_token. Set tokenizer.pad_token explicitly, "
                "or provide a tokenizer with eos_token so pad_token can be derived automatically."
            ) from e

        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            raise ValueError(
                "Tokenizer pad_token could not be initialized. Set tokenizer.pad_token explicitly."
            )

        try:
            base = ModelUtils.unwrap_model(self.model)
            cfg = getattr(base, "config", None)
            if cfg is not None and getattr(cfg, "pad_token_id", None) is None:
                cfg.pad_token_id = int(pad_id)

            gen_cfg = getattr(base, "generation_config", None)
            if gen_cfg is not None and getattr(gen_cfg, "pad_token_id", None) is None:
                gen_cfg.pad_token_id = int(pad_id)
        except Exception:
            pass

    @contextmanager
    def _with_tokenizer_sides(self):
        tok = self.tokenizer
        if tok is None:
            yield
            return

        self._ensure_tokenizer_has_pad_token()

        had_pad = hasattr(tok, "padding_side")
        had_trunc = hasattr(tok, "truncation_side")

        old_pad = tok.padding_side if had_pad else None
        old_trunc = tok.truncation_side if had_trunc else None

        try:
            if had_pad and old_pad != self.padding_side:
                tok.padding_side = self.padding_side
            if had_trunc and old_trunc != self.truncation_side:
                tok.truncation_side = self.truncation_side
            yield
        finally:
            if had_pad:
                tok.padding_side = old_pad
            if had_trunc:
                tok.truncation_side = old_trunc

    def _free_training_artifacts(self) -> None:
        try:
            if self.activation_extractor is not None:
                self.activation_extractor.remove()
        except Exception:
            pass
        self.activation_extractor = None
        self.vector_mediator = None
        self.vector_update_strategy = None
        self.dataset = None
        try:
            self.train_df = pd.DataFrame()
        except Exception:
            pass

    def _atas_enabled(self) -> bool:
        return self.mediator_type in (
            MediatorTypeEnum.ATAS_CMD,
            MediatorTypeEnum.ATAS_CPCA,
        )

    def _training_layers_to_extract(self) -> List[int]:
        layers = sorted({int(i) for i in self.layers_to_extract})
        if not self._atas_enabled():
            return layers

        n = len(ModelUtils.find_transformer_layers(self.model))
        ml = int(self.atas_master_layer)
        if ml < 0 or ml >= n:
            raise IndexError(f"atas_master_layer={ml} out of range for model with {n} layers.")

        return sorted(set(layers) | {ml})

    def _require_activations(self, atas_full: Dict[int, torch.Tensor]) -> None:
        if not atas_full:
            raise RuntimeError(
                "Activation extraction produced an empty dict. "
                "This indicates hooks did not capture any activations for the requested layers."
            )
        required_layers = (
            self.activation_extractor.layers_to_extract
            if self.activation_extractor is not None
            else self.layers_to_extract
        )
        missing = [i for i in required_layers if i not in atas_full]
        if missing:
            got = sorted(atas_full.keys())
            raise RuntimeError(
                f"Missing activations for requested layers {missing}. Got layers={got}. "
                "This is treated as fatal to avoid silent training failures."
            )

    def _require_group_ids_if_atas(self, group_ids: Optional[torch.Tensor]) -> None:
        if self.mediator_type in (
                MediatorTypeEnum.ATAS_CMD,
                MediatorTypeEnum.ATAS_CPCA,
        ) and group_ids is None:
            raise ValueError("ATAS requires a task_id for every row (group_ids=None).")

    def _encode_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        with self._with_tokenizer_sides():
            return self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_tokens,
                add_special_tokens=False,
            )

    def _calculate_prompts_lengths(self, prompts: List[str]) -> torch.Tensor:
        with self._with_tokenizer_sides():
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_tokens,
                add_special_tokens=False,
            )
        return enc["attention_mask"].sum(1, dtype=torch.int64)

    def _pad_from_ids(self, id_lists: List[List[int]]) -> Dict[str, torch.Tensor]:
        items = [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in id_lists]
        with self._with_tokenizer_sides():
            return self.tokenizer.pad(items, padding=True, return_tensors="pt")

    def _trim_generated_tokens(self, gen_ids: List[int]) -> List[int]:
        if not gen_ids:
            return gen_ids

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)

        if eos_id is not None:
            try:
                j = gen_ids.index(int(eos_id))
                gen_ids = gen_ids[: j]  # no EOS, so not j+1
            except ValueError:
                pass

        if pad_id is not None:
            while gen_ids and gen_ids[-1] == int(pad_id):
                gen_ids.pop()

        return gen_ids

    @staticmethod
    def _as_torch_device(v) -> torch.device:
        if isinstance(v, torch.device):
            return v
        if v is None:
            return torch.device("cpu")
        if isinstance(v, int):
            return torch.device(f"cuda:{v}") if torch.cuda.is_available() else torch.device("cpu")

        s = str(v).strip().lower()
        if s in {"disk", "meta", "offload", "nvme"}:
            return torch.device("cpu")
        if s.isdigit():
            return torch.device(f"cuda:{s}") if torch.cuda.is_available() else torch.device("cpu")
        if s == "cuda":
            return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        try:
            return torch.device(s)
        except (TypeError, ValueError):
            return torch.device("cpu")

    @staticmethod
    def _infer_input_device(model: nn.Module) -> torch.device:
        base = ModelUtils.unwrap_model(model)
        try:
            if hasattr(base, "get_input_embeddings"):
                emb = base.get_input_embeddings()
                if emb is not None:
                    return next(emb.parameters()).device
        except Exception:
            pass

        dm = getattr(base, "hf_device_map", None)
        if isinstance(dm, dict) and dm:
            preferred_suffixes = ("embed_tokens", "wte", "embeddings", "tok_embeddings", "word_embeddings")
            for k, v in dm.items():
                if any(k.endswith(suf) or suf in k for suf in preferred_suffixes):
                    return Executor._as_torch_device(v)
            for v in dm.values():
                dv = Executor._as_torch_device(v)
                if dv.type == "cuda":
                    return dv
            return Executor._as_torch_device(next(iter(dm.values())))

        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, m: nn.Module) -> None:
        if not isinstance(m, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        self._model = m

    @property
    def truncation_side(self) -> str:
        return self._truncation_side

    @truncation_side.setter
    def truncation_side(self, value: str) -> None:
        value = str(value).lower()
        if value not in ("left", "right"):
            raise ValueError("truncation_side must be 'left' or 'right'")
        self._truncation_side = value

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tok):
        self._tokenizer = tok

    @property
    def padding_side(self) -> str:
        return self._padding_side

    @padding_side.setter
    def padding_side(self, value: str) -> None:
        value = str(value).lower()
        if value not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'")
        self._padding_side = value

    @property
    def layers_to_extract(self) -> List[int]:
        return self._layers_to_extract

    @layers_to_extract.setter
    def layers_to_extract(self, value: Union[Iterable[int], int]) -> None:
        if isinstance(value, int):
            value = [value]
        cleaned = sorted({int(i) for i in value})
        if not cleaned:
            raise ValueError("At least one layer must be specified for extraction")
        self._layers_to_extract = cleaned

        n = len(ModelUtils.find_transformer_layers(self.model))
        bad = [i for i in cleaned if i < 0 or i >= n]
        if bad:
            raise IndexError(f"Layer index(es) {bad} out of range for model with {n} layers.")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size: int) -> None:
        self._batch_size = max(1, int(size))

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        if not isinstance(value, torch.dtype):
            raise TypeError("dtype must be an instance of torch.dtype")
        self._dtype = value

    @property
    def alpha(self) -> AlphaSpec:
        return self._alpha

    @alpha.setter
    def alpha(self, a: AlphaSpec) -> None:
        if isinstance(a, dict):
            self._alpha = {int(k): float(v) for k, v in a.items()}
        else:
            self._alpha = float(a)

    @property
    def apply_from_token(self) -> int:
        return self._apply_from_token

    @apply_from_token.setter
    def apply_from_token(self, ind: int) -> None:
        self._apply_from_token = int(ind)

    @property
    def apply_from_mode(self) -> ApplyFromModeEnum:
        return self._apply_from_mode

    @apply_from_mode.setter
    def apply_from_mode(self, mode: ApplyFromModeEnum) -> None:
        if not isinstance(mode, ApplyFromModeEnum):
            raise TypeError("apply_from_mode must be ApplyFromModeEnum")
        self._apply_from_mode = mode

    @property
    def task_type(self) -> TaskTypeEnum:
        return self._task_type

    @task_type.setter
    def task_type(self, t: TaskTypeEnum) -> None:
        if not isinstance(t, TaskTypeEnum):
            raise TypeError("task_type must be TaskTypeEnum")
        self._task_type = t

    @property
    def mediator_type(self) -> MediatorTypeEnum:
        return self._mediator_type

    @mediator_type.setter
    def mediator_type(self, m: MediatorTypeEnum) -> None:
        if not isinstance(m, MediatorTypeEnum):
            raise TypeError("mediator_type must be MediatorTypeEnum")
        self._mediator_type = m

    @property
    def tokens_window(self) -> int:
        return self._tokens_window

    @tokens_window.setter
    def tokens_window(self, value: int) -> None:
        self._tokens_window = int(value)

    @property
    def generation_kwargs(self) -> Dict[str, Any]:
        return self._generation_kwargs

    @generation_kwargs.setter
    def generation_kwargs(self, kw: Dict[str, Any]) -> None:
        self._generation_kwargs = dict(kw or {})

    @property
    def vector_update_strategy(self) -> Optional[BaseVectorUpdateStrategy]:
        return self._vector_update_strategy

    @vector_update_strategy.setter
    def vector_update_strategy(self, v: Optional[BaseVectorUpdateStrategy]) -> None:
        self._vector_update_strategy = v

    @property
    def activation_extractor(self) -> Optional[ActivationExtractor]:
        return self._activation_extractor

    @activation_extractor.setter
    def activation_extractor(self, a: Optional[ActivationExtractor]) -> None:
        self._activation_extractor = a

    @property
    def vector_mediator(self) -> Optional[VectorMediator]:
        return self._vector_mediator

    @vector_mediator.setter
    def vector_mediator(self, vm: Optional[VectorMediator]) -> None:
        self._vector_mediator = vm

    @property
    def steering_strategy(self) -> Optional[BaseSteeringStrategy]:
        return self._steering_strategy

    @steering_strategy.setter
    def steering_strategy(self, rs: Optional[BaseSteeringStrategy]) -> None:
        self._steering_strategy = rs

    def _spawn_subset_executor(self, subset_df: pd.DataFrame) -> "Executor":
        return Executor(
            model=self.model,
            tokenizer=self.tokenizer,
            train_df=pd.DataFrame(subset_df).copy(),
            mediator_type=self.mediator_type,
            layers_to_extract=self.layers_to_extract,
            atas_master_layer=self.atas_master_layer,
            batch_size=self.batch_size,
            padding_side=self.padding_side,
            truncation_side=self.truncation_side,
            dtype=self.dtype,
            alpha=0.0,
            generation_kwargs=dict(self.generation_kwargs),
            atas_similarity_threshold=self.atas_similarity_threshold,
            tokens_window=self.tokens_window,
            task_type=self.task_type,
            apply_from_token=self.apply_from_token,
            apply_from_mode=self.apply_from_mode,
            atas_use_last_token_for_prompt=self.atas_use_last_token_for_prompt,
            use_last_token_for_response=self.use_last_token_for_response,
            decode_chunk_max=self.decode_chunk_max,
            atas_random_routing_seed=self.atas_random_routing_seed,
            free_training_artifacts_after_build=True,
        )

    def _merge_single_task_into_atas(
            self,
            task_id: int,
            new_strategy: AtasSteeringStrategy,
            *,
            replace_if_exists: bool = False,
    ) -> None:
        current = self.steering_strategy
        if current is None:
            self.steering_strategy = new_strategy
            return

        if not isinstance(current, AtasSteeringStrategy):
            raise RuntimeError(
                "Current steering strategy is not ATAS; cannot incrementally add ATAS tasks."
            )

        old_routing, old_corr, old_tids = current.export_params()
        new_routing, new_corr, new_tids = new_strategy.export_params()

        if old_tids is None:
            raise RuntimeError("Existing ATAS strategy has task_ids=None.")
        if new_tids != [int(task_id)]:
            raise RuntimeError(
                f"Expected a single exported ATAS task [{int(task_id)}], got {new_tids}."
            )

        master_old = int(current.master_prompt_layer)
        master_new = int(new_strategy.master_prompt_layer)
        if master_old != master_new:
            raise RuntimeError(
                f"ATAS master layer mismatch: current={master_old} new={master_new}."
            )
        master = master_old

        if master not in old_routing or master not in new_routing:
            raise RuntimeError(
                f"Missing routing centroids for master_prompt_layer={master}."
            )

        required_layers = sorted({int(i) for i in self.layers_to_extract})
        missing_old = [L for L in required_layers if L not in old_corr]
        missing_new = [L for L in required_layers if L not in new_corr]
        if missing_old:
            raise RuntimeError(
                f"Existing ATAS correction vectors missing layers {missing_old}."
            )
        if missing_new:
            raise RuntimeError(
                f"New ATAS correction vectors missing layers {missing_new}."
            )

        pc_old = old_routing[master]
        pc_new = new_routing[master]
        if not (torch.is_tensor(pc_new) and pc_new.ndim == 2 and int(pc_new.size(0)) == 1):
            raise RuntimeError(
                f"New ATAS routing centroid must have shape [1, H], got {None if not torch.is_tensor(pc_new) else tuple(pc_new.shape)}."
            )

        if int(task_id) in old_tids:
            if not replace_if_exists:
                raise ValueError(
                    f"task_id={int(task_id)} already exists in ATAS. "
                    "Pass replace_if_exists=True to overwrite it."
                )

            idx = old_tids.index(int(task_id))
            merged_tids = list(old_tids)

            merged_pc = pc_old.clone()
            merged_pc[idx] = pc_new[0]

            merged_corr: Dict[int, torch.Tensor] = {}
            for L in required_layers:
                cur = old_corr[L].clone()
                new_row = new_corr[L]
                if not (torch.is_tensor(new_row) and new_row.ndim == 2 and int(new_row.size(0)) == 1):
                    raise RuntimeError(
                        f"New ATAS correction vector for layer {L} must have shape [1, H], "
                        f"got {None if not torch.is_tensor(new_row) else tuple(new_row.shape)}."
                    )
                cur[idx] = new_row[0]
                merged_corr[int(L)] = cur
        else:
            merged_tids = list(old_tids) + [int(task_id)]
            merged_pc = torch.cat([pc_old, pc_new], dim=0)

            merged_corr = {}
            for L in required_layers:
                new_row = new_corr[L]
                if not (torch.is_tensor(new_row) and new_row.ndim == 2 and int(new_row.size(0)) == 1):
                    raise RuntimeError(
                        f"New ATAS correction vector for layer {L} must have shape [1, H], "
                        f"got {None if not torch.is_tensor(new_row) else tuple(new_row.shape)}."
                    )
                merged_corr[int(L)] = torch.cat([old_corr[L], new_row], dim=0)

        self.steering_strategy = AtasSteeringStrategy(
            routing_centroids={master: merged_pc},
            correction_vectors=merged_corr,
            task_ids=merged_tids,
            similarity_threshold=float(current.similarity_threshold),
            use_last_token_for_prompt=bool(current.use_last_token_for_prompt),
            master_prompt_layer=master,
            random_routing_seed=getattr(current, "random_routing_seed", None),
        )

    def add_atas_task(
            self,
            task_id: int,
            task_df: pd.DataFrame,
            *,
            replace_if_exists: bool = False,
            alpha_for_new_task: Optional[float] = None,
    ) -> None:
        if self.mediator_type not in (MediatorTypeEnum.ATAS_CMD, MediatorTypeEnum.ATAS_CPCA):
            raise RuntimeError("add_atas_task() is supported only for ATAS mediators.")

        df = pd.DataFrame(task_df).copy()
        if df.empty:
            raise ValueError("task_df is empty.")

        required = {"prompt", "response", "reference"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for incremental ATAS task: {sorted(missing)}"
            )

        if "task_id" in df.columns:
            seen_ids = sorted({int(x) for x in df["task_id"].dropna().tolist()})
            bad_ids = [x for x in seen_ids if x != int(task_id)]
            if bad_ids:
                raise ValueError(
                    f"task_df contains task_id values different from the requested task_id={int(task_id)}: {bad_ids}"
                )

        df["task_id"] = int(task_id)

        temp = self._spawn_subset_executor(df)
        _ = temp.representation_extractor()

        if not isinstance(temp.steering_strategy, AtasSteeringStrategy):
            raise RuntimeError("Temporary subset training did not produce an ATAS strategy.")

        if self.steering_strategy is None:
            self.steering_strategy = temp.steering_strategy
        else:
            self._merge_single_task_into_atas(
                task_id=int(task_id),
                new_strategy=temp.steering_strategy,
                replace_if_exists=replace_if_exists,
            )

        if isinstance(self.alpha, dict):
            alpha_map = dict(self.alpha)
            if int(task_id) not in alpha_map or alpha_for_new_task is not None:
                default_alpha = (
                    float(alpha_for_new_task)
                    if alpha_for_new_task is not None
                    else (float(next(iter(alpha_map.values()))) if alpha_map else 0.5)
                )
                alpha_map[int(task_id)] = default_alpha
                self.alpha = alpha_map

        base_df = pd.DataFrame(self.train_df).copy()
        if replace_if_exists and "task_id" in base_df.columns:
            base_df = base_df[base_df["task_id"] != int(task_id)]
        self.train_df = pd.concat([base_df, df], ignore_index=True)

        if self.dataset is not None:
            self.dataset = self.prepare_dataset_to_train(self.train_df)

    def prepare_dataset_to_train(self, df: pd.DataFrame) -> DataLoader:
        df = pd.DataFrame(df)

        required = {"prompt", "reference"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if self.mediator_type in (MediatorTypeEnum.ATAS_CMD, MediatorTypeEnum.ATAS_CPCA):
            if "task_id" not in df.columns:
                raise ValueError("ATAS training requires a 'task_id' column.")
            if df["task_id"].isna().any():
                raise ValueError("ATAS training requires non-null 'task_id' for every row.")

        if self.task_type == TaskTypeEnum.UNSUPERVISED:
            if not df["reference"].isin([0, 1]).all():
                raise ValueError("For UNSUPERVISED mode the 'reference' column must contain only 0 or 1.")
            if "response" not in df.columns:
                raise ValueError("UNSUPERVISED training requires a 'response' column for every row.")
            if df["response"].isna().any():
                raise ValueError("UNSUPERVISED training requires 'response' to be present for every row.")
        else:
            raise Exception("No technique other than unsupervised supported")

        records: List[Dict[str, Any]] = df.to_dict("records")
        return DataLoader(
            PromptDataset(records),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=identity_collate,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )

    def _setup_vector_components(self) -> None:
        try:
            training_layers = self._training_layers_to_extract()
            if self.mediator_type == MediatorTypeEnum.ATAS_CMD:
                self.vector_update_strategy = AtasVectorMediator(
                    correctness_threshold=0.5,
                    correction_mode="cmd",
                    use_last_token_for_prompt=self.atas_use_last_token_for_prompt,
                    use_last_token_for_response=self.use_last_token_for_response,
                    master_prompt_layer=self.atas_master_layer,
                )
            elif self.mediator_type == MediatorTypeEnum.ATAS_CPCA:
                self.vector_update_strategy = AtasVectorMediator(
                    correctness_threshold=0.5,
                    correction_mode="cpca",
                    use_last_token_for_prompt=self.atas_use_last_token_for_prompt,
                    use_last_token_for_response=self.use_last_token_for_response,
                    master_prompt_layer=self.atas_master_layer,
                )
            elif self.mediator_type == MediatorTypeEnum.CPCA:
                self.vector_update_strategy = CpcaVectorMediator(
                    use_last_token_for_response=self.use_last_token_for_response,
                )
            else:
                self.vector_update_strategy = CmdVectorMediator(
                    use_last_token_for_response=self.use_last_token_for_response,
                )

            self.activation_extractor = ActivationExtractor(
                model=self.model,
                layers_to_extract=training_layers,
                offload_to_cpu=True,
                decode_chunk_max=self.decode_chunk_max,
            )

            layers = list(ModelUtils.find_transformer_layers(self.model))
            if not layers:
                raise RuntimeError("No transformer layers found: check ModelUtils.find_transformer_layers().")

            base = ModelUtils.unwrap_model(self.model)

            cfg_max = int(getattr(base.config, "max_position_embeddings", 0) or 0)
            tok_max = getattr(self.tokenizer, "model_max_length", None)
            tok_max_int = None
            try:
                if tok_max is not None:
                    tok_max_int = int(tok_max)
                    if tok_max_int >= 10 ** 9:
                        tok_max_int = None
            except Exception:
                tok_max_int = None

            requested_sequence_max = int(self.max_sequence_tokens)
            candidates = [
                x for x in (cfg_max, tok_max_int, requested_sequence_max)
                if isinstance(x, int) and x > 0
            ]
            self.max_sequence_tokens = min(candidates) if candidates else requested_sequence_max
            self.max_prompt_tokens = min(int(self.max_prompt_tokens), self.max_sequence_tokens)
            self.max_response_tokens = min(int(self.max_response_tokens), self.max_sequence_tokens)
            if self.max_prompt_tokens + self.max_response_tokens > self.max_sequence_tokens:
                raise ValueError(
                    "Resolved model context is smaller than the configured Executor prompt/response budgets."
                )

            arch = getattr(base.config, "is_encoder_decoder", False)
            if arch:
                raise NotImplementedError("Encoder-decoder architectures are not supported by this steering wrapper.")

            hidden_size = ModelUtils.get_hidden_size(base.config)
            self.vector_mediator = VectorMediator(
                layers_to_extract=training_layers,
                hidden_size=hidden_size,
                update_strategy=self.vector_update_strategy,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to set up vector components: {e}") from e

    def _build_steering_strategy(self) -> None:
        try:
            upd = self.vector_mediator.update_strategy if self.vector_mediator else None
            if upd is None:
                raise ValueError("Vector mediator update strategy is not initialized")

            if hasattr(upd, "get_multitask_params"):
                routing_c, corr_vecs, task_ids = upd.get_multitask_params()
                similarity_threshold = float(getattr(self, "atas_similarity_threshold", 0.0))
                requested_layers = sorted({int(i) for i in self.layers_to_extract})
                corr_vecs = {
                    int(k): v
                    for k, v in corr_vecs.items()
                    if int(k) in requested_layers
                }

                missing = [i for i in requested_layers if i not in corr_vecs]
                if missing:
                    raise RuntimeError(
                        f"ATAS correction vectors missing for requested layers {missing}. "
                        f"Available layers: {sorted(int(k) for k in corr_vecs.keys())}"
                    )
                self.steering_strategy = AtasSteeringStrategy(
                    routing_centroids=routing_c,
                    correction_vectors=corr_vecs,
                    task_ids=task_ids,
                    similarity_threshold=similarity_threshold,
                    use_last_token_for_prompt=self.atas_use_last_token_for_prompt,
                    master_prompt_layer=self.atas_master_layer,
                    random_routing_seed=self.atas_random_routing_seed,
                )
                return

            cv = getattr(upd, "correction_vectors", None)
            if not cv:
                raise ValueError("Correction vectors not available in update strategy")

            self.steering_strategy = GeneralSteeringStrategy(correction_vectors=cv)

        except Exception as e:
            raise RuntimeError(f"Failed to build steering strategy: {e}") from e

    def _run_training_epoch_for_loader(self, loader) -> None:
        if self.activation_extractor is None:
            raise RuntimeError("activation_extractor is None")
        if self.vector_mediator is None:
            raise RuntimeError("vector_mediator is None")

        model_device = self._infer_input_device(self.model)
        max_len = int(getattr(self, "max_sequence_tokens", 1536))
        max_response_len = int(getattr(self, "max_response_tokens", 512))
        rows_seen = 0
        responses_truncated = 0
        max_response_tokens_before_truncation = 0

        for raw_batch in tqdm(loader, desc="Extracting activations"):
            batch = cast(List[Dict[str, Any]], raw_batch)

            prefix_texts = [d["prompt"] for d in batch]

            enc = self._encode_prompts(prefix_texts)
            prompt_full_lens = enc["attention_mask"].sum(1, dtype=torch.int64)
            seq_len_prompt_pad = enc["input_ids"].size(1)

            base_prompts: Optional[List[str]] = None
            enc = {k: v.to(model_device) for k, v in enc.items()}

            if self.task_type == TaskTypeEnum.UNSUPERVISED:
                inp_ids = enc["input_ids"].detach().cpu()
                attn_inp = enc["attention_mask"].detach().cpu()
                prompt_lens_cpu = attn_inp.sum(1, dtype=torch.int64)

                resp_fields = [d.get("response", None) for d in batch]
                missing = [i for i, r in enumerate(resp_fields) if r is None or (isinstance(r, float) and pd.isna(r))]
                if missing:
                    meta = []
                    for i in missing[:10]:
                        d = batch[i]
                        meta.append({"i": i, "prompt_id": d.get("prompt_id"), "task_id": d.get("task_id")})
                    more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
                    raise ValueError(
                        "UNSUPERVISED training requires 'response' for every row (teacher forcing). "
                        f"Missing response in batch positions: {missing[:10]}{more}. Meta preview: {meta}."
                    )

                concat_lists: List[List[int]] = []
                kept_batch: List[Dict[str, Any]] = []
                kept_prompt_lens: List[int] = []
                for i in range(inp_ids.size(0)):
                    resp_ids: List[int] = []
                    prompt_tokens = inp_ids[i, attn_inp[i].to(torch.bool)].tolist()

                    r = resp_fields[i]
                    if isinstance(r, (list, tuple)) and all(isinstance(x, int) for x in r):
                        resp_ids = list(map(int, r))
                    else:
                        r = "" if r is None else str(r)
                        with self._with_tokenizer_sides():
                            resp_ids = self.tokenizer(r, add_special_tokens=False, return_tensors=None)["input_ids"]

                    out_tokens = self._trim_generated_tokens(list(resp_ids))
                    rows_seen += 1
                    max_response_tokens_before_truncation = max(
                        max_response_tokens_before_truncation,
                        len(out_tokens),
                    )
                    if len(out_tokens) > max_response_len:
                        # Keep the natural response prefix: this is exactly the portion the
                        # causal model can produce under the matching generation budget.
                        out_tokens = out_tokens[:max_response_len]
                        responses_truncated += 1
                    seq = prompt_tokens + out_tokens
                    if len(seq) > max_len:
                        dropped = len(seq) - max_len
                        pl = int(prompt_lens_cpu[i].item())

                        if self.truncation_side == "left":
                            seq = seq[-max_len:]
                            dropped_from_prompt = min(dropped, pl)
                            pl = max(0, pl - dropped_from_prompt)
                            prompt_lens_cpu[i] = pl
                            if dropped > int(prompt_lens_cpu[i].item()) + int(dropped_from_prompt):
                                logger.warning(
                                    "UNSUPERVISED: left-truncation removed part of the RESPONSE. "
                                    "Skipping sample i=%d (prompt_len=%d, dropped=%d, max_len=%d).",
                                    int(i), int(pl), int(dropped), int(max_len)
                                )
                                continue
                        else:
                            seq = seq[:max_len]
                            prompt_lens_cpu[i] = min(pl, max_len)
                    pl_now = int(prompt_lens_cpu[i].item())
                    if pl_now <= 0:
                        d = batch[i]
                        logger.warning(
                            "UNSUPERVISED: prompt_len became 0 after truncation; skipping sample i=%d "
                            "(prompt_id=%s, task_id=%s, max_len=%d, truncation_side=%s). "
                            "Routing prompt centroid would be invalid.",
                            int(i), str(d.get("prompt_id")), str(d.get("task_id")),
                            int(max_len), str(self.truncation_side),
                        )
                        continue
                    concat_lists.append(seq)
                    kept_batch.append(batch[i])
                    kept_prompt_lens.append(pl_now)

                self.last_teacher_forcing_truncation_stats = {
                    "rows": int(rows_seen),
                    "responses_truncated": int(responses_truncated),
                    "max_response_tokens_before_truncation": int(max_response_tokens_before_truncation),
                }

                if not concat_lists:
                    logger.warning(
                        "UNSUPERVISED: all samples in batch were skipped due to response truncation (max_len=%d).",
                        int(max_len)
                    )
                    continue

                batch = kept_batch
                prompt_lens_cpu = torch.tensor(kept_prompt_lens, dtype=torch.int64)
                enc_aug = self._pad_from_ids(concat_lists)
                enc_aug = {k: v.to(model_device) for k, v in enc_aug.items()}

                self.activation_extractor.clear()
                _ = self.model(
                    input_ids=enc_aug["input_ids"],
                    attention_mask=enc_aug["attention_mask"],
                    use_cache=False,
                )
                self.activation_extractor.assert_ok()

                atas_full = dict(self.activation_extractor.finalize(clear_chunks=True).items())
                self._require_activations(atas_full)

                device = next(iter(atas_full.values())).device

                attn_bool = enc_aug["attention_mask"].to(device=device, dtype=torch.bool)
                total_lens = enc_aug["attention_mask"].sum(1, dtype=torch.int64).to(device)

                s_pad = int(enc_aug["input_ids"].size(1))
                if self.padding_side == "left":
                    left_pad = (s_pad - total_lens).to(dtype=torch.int64)
                else:
                    left_pad = torch.zeros_like(total_lens, dtype=torch.int64)

                prompt_lens = prompt_lens_cpu.to(device=device, dtype=torch.int64)

                if self.apply_from_mode == ApplyFromModeEnum.FIXED:
                    if self.padding_side == "left":
                        starts = left_pad + int(self.apply_from_token)
                    else:
                        starts = torch.full_like(prompt_lens, int(self.apply_from_token), device=device)
                else:
                    if self.padding_side == "left":
                        starts = left_pad + prompt_lens
                    else:
                        starts = prompt_lens

                starts = starts.to(dtype=torch.int64)

                labels = torch.tensor([d["reference"] for d in batch], dtype=torch.float32, device=device)

                tid_s = [d.get("task_id") for d in batch]
                group_ids = (
                    torch.tensor([int(t) for t in tid_s], dtype=torch.long, device=device)
                    if tid_s and all(t is not None for t in tid_s)
                    else None
                )
                self._require_group_ids_if_atas(group_ids)

                self.vector_mediator.update_vectors(
                    activations=atas_full,
                    labels=labels,
                    starts=starts,
                    tokens_window=self.tokens_window,
                    attention_mask=attn_bool,
                    group_ids=group_ids,
                )

                self.activation_extractor.clear()
                del atas_full, enc_aug, enc
            else:
                raise Exception(
                    "No technique other than unsupervised supported"
                )

        if responses_truncated:
            logger.warning(
                "Executor teacher forcing capped %d/%d responses at %d tokens "
                "(longest before truncation: %d).",
                int(responses_truncated),
                int(rows_seen),
                int(max_response_len),
                int(max_response_tokens_before_truncation),
            )

    def representation_extractor(self) -> SteeredModelWrapper:
        _assert_no_forward_hooks(self.model, where="(Executor.representation_extractor: entry)")

        if self.steering_strategy is not None:
            return SteeredModelWrapper(
                base_model=self.model,
                tokenizer=self.tokenizer,
                strategy=self.steering_strategy,
                alpha=self.alpha,
                apply_from_token=self.apply_from_token,
                apply_from_mode=self.apply_from_mode,
                tokens_window=self.tokens_window,
                padding_side=self.padding_side,
            )

        if (
                self.vector_update_strategy is None
                or self.activation_extractor is None
                or self.vector_mediator is None
        ):
            self._setup_vector_components()
        else:
            if hasattr(self.vector_update_strategy, "reset"):
                self.vector_update_strategy.reset()
            self.vector_mediator.clear_errors()
            self.activation_extractor.remove()
            self.activation_extractor.clear()

        if self.dataset is None:
            raise RuntimeError(
                "Training dataset is not available and no cached steering strategy exists."
            )

        self.model.eval()
        with _attached_activation_extractor(self.activation_extractor), torch.inference_mode():
            self._run_training_epoch_for_loader(self.dataset)

        if hasattr(self.vector_update_strategy, "finalize"):
            self.vector_update_strategy.finalize()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self._build_steering_strategy()
        if hasattr(self.steering_strategy, "target_layers"):
            if set(self.steering_strategy.target_layers) != set(self.layers_to_extract):
                raise RuntimeError(
                    f"Strategy layers {self.steering_strategy.target_layers} != requested {self.layers_to_extract}"
                )

        _assert_no_forward_hooks(self.model, where="(Executor.representation_extractor: before return)")

        wrapper = SteeredModelWrapper(
            base_model=self.model,
            tokenizer=self.tokenizer,
            strategy=self.steering_strategy,
            alpha=self.alpha,
            apply_from_token=self.apply_from_token,
            apply_from_mode=self.apply_from_mode,
            tokens_window=self.tokens_window,
            padding_side=self.padding_side,
        )

        if self._free_training_artifacts_after_build:
            self._free_training_artifacts()

        return wrapper
