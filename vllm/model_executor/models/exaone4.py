# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import Exaone4Config

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


# ----------------------------------------------------------------------
# 1‑A. MLP (gated‑SwiGLU → down‑proj)
# ----------------------------------------------------------------------
class Exaone4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # gate & up proj 합친 2×intermediate (SwiGLU)
        self.gate_up_proj = RowParallelLinear(  # weight 패킹 안 함
            hidden_size,
            intermediate_size * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError("Exaone4 MLP 는 silu·gate 방식만 지원.")
        self.act_fn = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.act_fn(gate) * up  # SwiGLU
        x, _ = self.down_proj(x)
        return x


# ----------------------------------------------------------------------
# 1‑B. Attention (QK‑Norm + RoPE + SlidingWindow)
# ----------------------------------------------------------------------
class Exaone4Attention(nn.Module):
    """Decoder‑only attention 레이어 (GQA + Sliding Window)."""

    def __init__(
        self,
        config: Exaone4Config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10_000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        qkv_bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # QKV 병렬 프로젝션
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # RotaryEmbedding
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # vLLM core Attention 모듈
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
        # (b s h) → QKV split
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK‑Norm per head
        q = self.q_norm(q.view(*q.shape[:-1], self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)).view(k.shape)

        # RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Sliding‑window 여부는 config .sliding_window 가 None 이 아니면 내부에서 자동 처리
        attn_out = self.attn(q, k, v)
        out, _ = self.o_proj(attn_out)
        return out


# ----------------------------------------------------------------------
# 1‑C. Decoder Block
# ----------------------------------------------------------------------
class Exaone4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Exaone4Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn = Exaone4Attention(
            config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=getattr(config, "rope_theta", 10_000),
            rope_scaling=getattr(config, "rope_scaling", None),
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Exaone4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn(positions, self.ln_1(hidden_states))
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.ln_2(hidden_states))
        return residual + hidden_states


# ----------------------------------------------------------------------
# 1‑D. Transformer (stacked layers + RMSNorm final)
# ----------------------------------------------------------------------
class Exaone4Model(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        cfg: Exaone4Config = vllm_config.model_config.hf_config
        quant = vllm_config.quant_config

        self.config = cfg
        self.quant_config = quant
        self.vocab_size = cfg.vocab_size

        self.wte = ParallelLMHead(
            cfg.vocab_size,
            cfg.hidden_size,
            org_num_embeddings=cfg.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant,
            # input embedding 전용 head
        )

        self.layers = make_layers(
            lambda idx: Exaone4DecoderLayer(
                cfg,
                cache_config=vllm_config.cache_config,
                quant_config=quant,
                prefix=f"{prefix}.layers.{idx}",
            ),
            cfg.num_hidden_layers,
        )
        self.ln_f = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            cfg.num_hidden_layers,
            tp_size=get_tensor_model_parallel_world_size(),
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is None:
            hidden_states = self.get_input_embeddings(input_ids)
        else:
            hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)

        hidden_states = self.ln_f(hidden_states)

        if intermediate_tensors is not None:
            intermediate_tensors.hidden_states = hidden_states
            return intermediate_tensors
        return hidden_states


# ----------------------------------------------------------------------
# 1‑E. Causal LM head (tie weights, LoRA 호환)
# ----------------------------------------------------------------------
class Exaone4ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    embedding_modules = {"wte": "input_embeddings", "lm_head": "output_embeddings"}
    embedding_padding_modules = ["lm_head"]
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config: Exaone4Config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.lora_config = vllm_config.lora_config

        self.model = Exaone4Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = self.config.vocab_size
            if self.lora_config:
                self.unpadded_vocab_size += self.lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                self.config.hidden_size,
                org_num_embeddings=self.config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                if not self.lora_config
                else self.lora_config.lora_vocab_padding_size,
                quant_config=self.quant_config,
            )
            if self.config.tie_word_embeddings:
                self.lm_head.weight = self.model.wte.weight
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size,
                self.config.vocab_size,
                getattr(self.config, "logit_scale", 1.0),
            )
        else:
            self.lm_head = nn.Identity()  # PP front‑ranks placeholder

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    # --- forward / helper --------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    # --- weight‑loader -----------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        HF -> vLLM 이름 매핑은 대부분 자동 처리되지만
        • qkv_proj → q_proj/k_proj/v_proj 스택
        • gate_up_proj → gate_proj/up_proj
        만 추가 매핑.
        """
        loader = AutoWeightsLoader(
            self,
            stacked_params_mapping=self.packed_modules_mapping,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
