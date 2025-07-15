# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Inference‑only EXAONE‑4 model compatible with HuggingFace weights
(https://huggingface.co/LGAI-EXAONE/EXAONE-4.0‑xx)

Key differences from Qwen3:
  • QK‑Norm (RMSNorm per‑head) on query/key
  • Sliding‑Window(local) / global hybrid attention
  • Default GQA (num_key_value_heads < num_attention_heads)
"""

from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import Exaone4Config

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    DEFAULT_VOCAB_PADDING_SIZE,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .adapters import as_seq_cls_model
from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    make_layers,
    maybe_prefix,
    PPMissingLayer,
)

logger = init_logger(__name__)


# --------------------------------------------------------------------- #
# 1. Feed‑Forward Network (SwiGLU)                                      #
# --------------------------------------------------------------------- #
class Exaone4MLP(nn.Module):
    """gate_proj · up_proj → down_proj  (SwiGLU)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = RowParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        assert hidden_act == "silu", "EXAONE‑4 uses silu(Gated‑SwiGLU)."
        self.act_fn = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.act_fn(gate) * up
        x, _ = self.down_proj(x)
        return x


# --------------------------------------------------------------------- #
# 2. Attention                                                          #
# --------------------------------------------------------------------- #
class Exaone4Attention(nn.Module):
    """Decoder‑only attention with QK‑Norm, GQA, RoPE, SlidingWindow."""

    def __init__(
        self,
        config: Exaone4Config,
        cache_config: Optional[CacheConfig],
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        # KV heads: replicate or split
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # Projections
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Rotary positional emb.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10_000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # QK‑Norm
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK‑Norm per head
        q = self.q_norm(q.view(*q.shape[:-1], self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)).view(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v)
        out, _ = self.o_proj(attn_out)
        return out


# --------------------------------------------------------------------- #
# 3. Decoder Layer                                                      #
# --------------------------------------------------------------------- #
class Exaone4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Exaone4Config,
        cache_config: Optional[CacheConfig],
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Exaone4Attention(
            config, cache_config, quant_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = Exaone4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)

        # MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# --------------------------------------------------------------------- #
# 4. Transformer Stack                                                  #
# --------------------------------------------------------------------- #
@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Exaone4Model(nn.Module):
    """Stack of N decoder layers + RMSNorm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        cfg: Exaone4Config = vllm_config.model_config.hf_config
        quant = vllm_config.quant_config

        self.embed_tokens = ParallelLMHead(
            cfg.vocab_size,
            cfg.hidden_size,
            org_num_embeddings=cfg.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant,
            prefix=f"{prefix}.embed_tokens",
        )

        # create layers
        self.layers = nn.ModuleList([
            Exaone4DecoderLayer(
                cfg,
                cache_config=vllm_config.cache_config,
                quant_config=quant,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(cfg.num_hidden_layers)
        ])
        self.final_ln = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # util for PP
        self.make_empty_intermediate_tensors = lambda bs, sl: IntermediateTensors(bs, sl)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states = self.final_ln(hidden_states)

        if intermediate_tensors is not None:
            intermediate_tensors.hidden_states = hidden_states
            return intermediate_tensors
        return hidden_states


# --------------------------------------------------------------------- #
# 5. Causal LM wrapper (+ LoRA / PP / logits)                           #
# --------------------------------------------------------------------- #
class Exaone4ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """vLLM‑serving wrapper around Exaone4Model."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        quant = vllm_config.quant_config

        self.config = cfg
        self.model = Exaone4Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        # LM Head
        if get_pp_group().is_last_rank:
            if cfg.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    cfg.vocab_size,
                    cfg.hidden_size,
                    org_num_embeddings=cfg.vocab_size,
                    padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                    quant_config=quant,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(cfg.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    # ---- serving‑path helpers ------------------------------------------
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    # ---- weight loader --------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


# 선택적: classification 헤드가 필요하다면 utils.as_seq_cls_model 로 래핑

Exaone4ForSequenceClassification = as_seq_cls_model(Exaone4ForCausalLM)
