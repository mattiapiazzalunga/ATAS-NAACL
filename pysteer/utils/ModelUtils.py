from __future__ import annotations

import re
from typing import Callable, List, Optional

import torch
import torch.nn as nn


class ModelUtils:

    @staticmethod
    def sanitize_fp16(t: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(t.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _get_by_path(obj, path: str):
        cur = obj
        for p in path.split("."):
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur

    @staticmethod
    def ensure_bsh(
            t: torch.Tensor,
            hidden_size: int,
            *,
            strict: bool = True,
            from_layout: Optional[str] = None,
            batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        if t.ndim != 3:
            if strict:
                raise ValueError(f"Expected 3D tensor (B,S,H); got shape={tuple(t.shape)}")
            return t

        if t.size(-1) != hidden_size:
            if strict:
                raise ValueError(
                    f"Expected hidden_size={hidden_size} on last dim; got shape={tuple(t.shape)}"
                )
            return t.contiguous()

        if from_layout is not None:
            fl = str(from_layout).upper()
            if fl == "BSH":
                return t.contiguous()
            if fl == "SBH":
                return t.permute(1, 0, 2).contiguous()
            raise ValueError(f"Unsupported from_layout={from_layout!r}")

        if batch_size is not None:
            b0 = (t.size(0) == batch_size)
            b1 = (t.size(1) == batch_size)

            if b0 and not b1:
                return t.contiguous()
            if b1 and not b0:
                return t.permute(1, 0, 2).contiguous()

            if b0 and b1:
                raise ValueError(
                    f"Ambiguous tensor layout for shape={tuple(t.shape)} and batch_size={batch_size}: "
                    "both dim0 and dim1 match batch_size. Pass from_layout explicitly."
                )

            raise ValueError(
                f"Cannot infer BSH/SBH for shape={tuple(t.shape)} and batch_size={batch_size}."
            )

        if strict:
            raise ValueError(
                f"Ambiguous 3D tensor layout for shape={tuple(t.shape)}. "
                "Pass from_layout='BSH'/'SBH' explicitly. "
                "Using only batch_size is insufficient when dim0 == dim1 == batch_size."
            )

        return t.contiguous()

    @staticmethod
    def unwrap_model(model: nn.Module) -> nn.Module:
        m = model
        seen = set()
        while hasattr(m, "module") and getattr(m, "module") is not m and id(m) not in seen:
            seen.add(id(m))
            m = m.module
        return m

    @staticmethod
    def _residual_block_predicate(mod: nn.Module, path: str) -> bool:
        name = (path or "").lower()
        leaf = name.split(".")[-1]

        deny_suffixes = (
            "self_attn", "selfattention", "attention", "attn",
            "mlp", "ff", "ffn", "feed_forward",
            "layernorm", "rmsnorm", "norm",
            "proj", "o_proj", "k_proj", "q_proj", "v_proj",
            "moe", "experts", "expert", "router", "gate", "gating",
            "embed", "embedding", "dropout",
        )
        if any(leaf.endswith(suf) for suf in deny_suffixes):
            return False

        cname = mod.__class__.__name__.lower()

        allow_exact = {
            "llamadecoderlayer", "llama3decoderlayer",
            "mistraldecoderlayer",
            "openelmdecoderlayer",
            "qwen2decoderlayer", "qwen25decoderlayer", "qwen2_5decoderlayer",
            "qwen3decoderlayer",
            "gemma3decoderlayer", "gemmadecoderlayer", "gemmablock", "gemma3block",
            "mixtraldecoderlayer", "falcondecoderlayer", "optdecoderlayer",
            "bloomblock", "mptblock", "gptneoxlayer", "gptjblock",
            "gpt2block", "gptbigcodeblock", "deepseekdecoderlayer", "phi3decoderlayer",
        }
        if cname in allow_exact:
            return True

        has_attn = any(
            hasattr(mod, a) for a in (
                "self_attn", "self_attention", "attn", "attention"
            )
        )
        has_ff = any(
            hasattr(mod, a) for a in (
                "mlp", "feed_forward", "ff", "ffn", "dense_h_to_4h", "block_sparse_moe", "moe_layer"
            )
        )
        has_norm = any(
            hasattr(mod, a) for a in (
                "input_layernorm", "post_attention_layernorm",
                "norm", "rmsnorm", "layernorm", "ln1", "ln_1", "ln",
                "attn_norm", "ffn_norm"
            )
        )

        return has_attn and has_ff and has_norm

    @staticmethod
    def find_transformer_layers(
            model: nn.Module,
            *,
            predicate: Optional[Callable[[nn.Module, str], bool]] = None,
            allow_heuristics: bool = True,
    ) -> List[nn.Module]:
        base = ModelUtils.unwrap_model(model)

        config = getattr(base, "config", None)
        expected = int(
            getattr(config, "num_hidden_layers", 0)
            or getattr(config, "num_transformer_layers", 0)
            or 0
        )

        model_type = getattr(config, "model_type", None) if config is not None else None

        if model_type in {"llama", "qwen2", "qwen3"}:
            known = [
                "model.layers",
                "model.model.layers",
            ]
        elif model_type == "gemma3":
            known = [
                "model.layers",
                "language_model.model.layers",
                "language_model.layers",
                "model.model.layers",
            ]
        elif model_type == "openelm":
            known = [
                "transformer.layers",
                "model.transformer.layers",
            ]
        else:
            known = [
                "model.layers",
                "model.model.layers",
                "transformer.layers",
                "transformer.h",
                "gpt_neox.layers",
                "model.decoder.layers",
                "decoder.layers",
                "language_model.model.layers",
                "language_model.layers",
                "text_model.model.layers",
                "text_model.layers",
                "layers",
            ]

        for kp in known:
            layers = ModelUtils._get_by_path(base, kp)
            if isinstance(layers, (nn.ModuleList, list)) and layers and all(isinstance(x, nn.Module) for x in layers):
                if expected and len(layers) != expected:
                    continue
                return list(layers)

        if not allow_heuristics:
            raise RuntimeError("Could not locate transformer layers via known paths.")

        pred = predicate or ModelUtils._residual_block_predicate
        found: List[tuple[str, nn.Module]] = []

        def dfs(mod: nn.Module, path: str) -> bool:
            matched_child = False
            for child_name, child in mod.named_children():
                child_path = f"{path}.{child_name}" if path else child_name
                if dfs(child, child_path):
                    matched_child = True

            if pred(mod, path):
                if not matched_child:
                    found.append((path, mod))
                return True

            return matched_child

        dfs(base, "")

        def key(item):
            path, _ = item
            nums = [int(x) for x in re.findall(r"\b(\d+)\b", path)]
            return nums or [10 ** 9, hash(path) & 0xFFFF]

        found.sort(key=key)
        return [m for _, m in found]

    @staticmethod
    def get_hidden_size(config) -> int:
        for k in ("hidden_size", "n_embd", "d_model", "model_dim"):
            if hasattr(config, k):
                return int(getattr(config, k))
        raise AttributeError("Could not determine hidden size from config (no hidden_size/n_embd/d_model).")
