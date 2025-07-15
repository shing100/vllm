# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM용 EXAONE4 모델 구현 (HuggingFace EXAONE4, vLLM Exaone, Qwen3 구조 참고)
"""
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import Exaone4Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .adapters import as_seq_cls_model
from .interfaces import SupportsLoRA, SupportsPP
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix

logger = init_logger(__name__)


class Exaone4MLP(nn.Module):
    """
    HF EXAONE4 구조에 맞춘 MLP (gate_proj, up_proj, down_proj)
    """
    def __init__(self, config: Exaone4Config, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()
        self.gate_proj = RowParallelLinear(
            config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.gate_proj"
        )
        self.up_proj = RowParallelLinear(
            config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.up_proj"
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.mlp.down_proj"
        )
        self.act_fn = torch.nn.functional.silu if config.hidden_act == "silu" else torch.nn.functional.gelu

    def forward(self, x):
        gate = self.gate_proj(x)[0]
        up = self.up_proj(x)[0]
        return self.down_proj(self.act_fn(gate) * up)[0]


class Exaone4Attention(nn.Module):
    """
    HF EXAONE4 구조에 맞춘 Attention (q_proj, k_proj, v_proj, o_proj, qk-norm, rotary 등)
    """
    def __init__(
        self,
        config: Exaone4Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.q_proj = RowParallelLinear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.q_proj"
        )
        self.k_proj = RowParallelLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.k_proj"
        )
        self.v_proj = RowParallelLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.v_proj"
        )
        self.o_proj = RowParallelLinear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.self_attn.o_proj"
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        self.attn = Attention(
            self.num_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.attn",
            attn_type=AttentionType.DECODER,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states)[0]
        k = self.k_proj(hidden_states)[0]
        v = self.v_proj(hidden_states)[0]
        # qk-norm
        q = self.q_norm(q.view(*q.shape[:-1], self.num_attention_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(*k.shape[:-1], self.num_key_value_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output = self.o_proj(attn_output)[0]
        return output


class Exaone4DecoderLayer(nn.Module):
    """
    HF EXAONE4 구조에 맞춘 Transformer 디코더 레이어 (post-LN, residual, MLP, Attention)
    """
    def __init__(
        self,
        config: Exaone4Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Exaone4Attention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.mlp = Exaone4MLP(config, quant_config=quant_config, prefix=prefix)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
        attn_out = self.self_attn(positions, hidden_states)
        attn_out = self.post_attention_layernorm(attn_out)
        hidden_states = residual + attn_out
        residual = hidden_states
        # MLP
        mlp_out = self.mlp(hidden_states)
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        hidden_states = residual + mlp_out
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Exaone4Model(nn.Module):
    """
    HF EXAONE4 구조에 맞춘 모델 본체 (embedding, layer stack, final norm)
    """
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Exaone4DecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"model.layers.{i}",
            ) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class Exaone4ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "q_proj": ["q_proj"],
        "k_proj": ["k_proj"],
        "v_proj": ["v_proj"],
        "gate_proj": ["gate_proj"],
        "up_proj": ["up_proj"],
        "down_proj": ["down_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.model = Exaone4Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head")
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.get_input_embeddings

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


Exaone4ForSequenceClassification = as_seq_cls_model(Exaone4ForCausalLM)
