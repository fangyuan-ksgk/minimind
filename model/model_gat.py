from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             GAT Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

class GATConfig(PretrainedConfig):
    model_type = "gat"

    def __init__(
        self,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        hidden_size: int = 768,
        flex_kernel_options: Optional[dict] = None,
        memory_span: int = 1024,
        K: int = 4,
        L: int = 2,
        vocab_size_list: list = [64, 32],
        _compile: bool = True,
        **kwargs
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.flex_kernel_options = flex_kernel_options
        self.memory_span = memory_span
        self.K = K
        self.L = L
        self.vocab_size_list = vocab_size_list
        self._compile = _compile
        
        vocab_sizes = torch.tensor(self.vocab_size_list) + 1
        self.total_vocab_size = int(sum(vocab_sizes))

        super().__init__(**kwargs)

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             Dependencies
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def infer_level(indices: torch.Tensor, vocab_sizes: torch.Tensor, pad_token: int):
    indices_expanded = indices.unsqueeze(-1)
    levels = (indices_expanded < vocab_sizes.cumsum(dim=0)).int().argmax(dim=-1)
    padding_mask = (indices == pad_token)
    final_levels = torch.where(padding_mask, -1, levels.long())
    return final_levels

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().float()
            self.sin_cached = freqs.sin().float()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        assert x.ndim == 4
        d = x.shape[3]//2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, flex_kernel_options=None):
        super().__init__()
        assert dim % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.dim = dim
        self.n_head = n_head        
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lamb = nn.Parameter(torch.tensor(0.5))
        self.rotary = Rotary(dim // n_head)
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_()
        self.flex_kernel_options = flex_kernel_options

    def forward(self, x, block_mask=None, cache=None):
        B, T, C = x.shape
        D_head = C // self.n_head

        # Reshape and apply rotary embeddings before transposing
        q_unrot = self.c_q(x).view(B, T, self.n_head, D_head)
        k_unrot = self.c_k(x).view(B, T, self.n_head, D_head)
        v_untransposed = self.c_v(x).view(B, T, self.n_head, D_head)

        q = self.rotary(q_unrot).transpose(1, 2)
        k = self.rotary(k_unrot).transpose(1, 2)
        v = v_untransposed.transpose(1, 2)
        
        # Consistent shape for cache: (batch, heads, seq_len, head_dim)
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_cache = (k, v)
   
        # QK norm applied after rotary and caching
        q, k = norm(q), norm(k)
        
        y = flex_attention(q, k, v, block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)       
        y = self.c_proj(y)       
        return y, new_cache

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config.hidden_size, config.num_attention_heads, config.flex_kernel_options)
        self.mlp = MLP(config.hidden_size)
   
    def forward(self, x, block_mask, cache=None):
        # Simplified to a standard pre-norm residual block
        attn_output, new_cache = self.attn(norm(x), block_mask=block_mask, cache=cache)
        x = x + attn_output
        x = x + self.mlp(norm(x))
        return x, new_cache

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             GAT Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

class GATModel(nn.Module):
    def __init__(self, config: GATConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.total_vocab_size, config.hidden_size)
        self.h = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])

        vocab_sizes = torch.tensor(config.vocab_size_list) + 1
        self.register_buffer("vocab_sizes", vocab_sizes)
        
        level_mask_tokens = vocab_sizes.cumsum(dim=0) - 1
        self.register_buffer("level_mask_tokens", level_mask_tokens)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        past_key_values = past_key_values or [None] * self.config.num_hidden_layers
        
        hidden_states = self.wte(input_ids)
        hidden_states = norm(hidden_states)

        # The mask is now a simple causal mask. The GAT logic is handled by pruning the cache.
        q_len = input_ids.shape[1]
        past_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
        kv_len = q_len + past_len
        
        def causal_mask(b, h, q_idx, kv_idx):
            full_seq_idx = past_len + q_idx
            return full_seq_idx >= kv_idx

        block_mask = create_block_mask(causal_mask, None, None, q_len, kv_len, device=input_ids.device, _compile=self.config._compile)

        next_past_key_values = () if use_cache else None
        for i, (block, past_kv) in enumerate(zip(self.h, past_key_values)):
            hidden_states, present = block(
                hidden_states, block_mask, cache=past_kv
            )
            if use_cache:
                next_past_key_values = next_past_key_values + (present,)

        hidden_states = norm(hidden_states)
        return hidden_states, next_past_key_values


class GATForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = GATConfig

    def __init__(self, config: GATConfig):
        super().__init__(config)
        self.model = GATModel(config)
        self.lm_head = CastedLinear(config.hidden_size, config.total_vocab_size)
        self.lm_head.weight.data.zero_()

        vocab_sizes = torch.tensor(config.vocab_size_list) + 1
        self.register_buffer("vocab_sizes", vocab_sizes)
        
        level_mask_tokens = vocab_sizes.cumsum(dim=0) - 1
        self.register_buffer("level_mask_tokens", level_mask_tokens)

        level_vocab_starts = torch.cat([torch.tensor([0.]), vocab_sizes.cumsum(dim=0)])[:-1]
        self.register_buffer("level_vocab_starts", level_vocab_starts)

        level_vocab_ends = vocab_sizes.cumsum(dim=0)
        self.register_buffer("level_vocab_ends", level_vocab_ends)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # We don't need `levels_cache` here as it's a generation-only concept
    ) -> CausalLMOutputWithPast:
        
        hidden_states, presents = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        loss = None
        if labels is not None:
            logits_for_loss = logits.view(-1, logits.size(-1))
            labels_for_loss = labels.view(-1)
            loss = F.cross_entropy(logits_for_loss, labels_for_loss, reduction="none")
            loss = loss.view(input_ids.shape[0], input_ids.shape[1])
            loss[torch.isin(labels, self.level_mask_tokens)] = 0

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        levels_cache = kwargs.get("levels_cache")

        # Update the levels cache with the new token's level
        new_levels = infer_level(input_ids[:, -1:], self.model.vocab_sizes, self.model.level_mask_tokens[0])
        if levels_cache is not None:
            levels_cache = torch.cat([levels_cache, new_levels], dim=1)
        else:
            # First step: calculate levels for the entire prompt
            levels_cache = infer_level(input_ids, self.model.vocab_sizes, self.model.level_mask_tokens[0])

        pruned_pkv = None
        pruned_levels = levels_cache

        if past_key_values is not None:
            seq_len = past_key_values[0][0].shape[2]
            
            device = levels_cache.device
            is_recent_arange = torch.arange(seq_len, device=device)
            is_recent = is_recent_arange >= (seq_len - self.config.memory_span)
            is_high_level = (levels_cache[:, :seq_len] > 0).squeeze(0)

            keep_mask = is_recent | is_high_level
            keep_indices = torch.where(keep_mask)[0]

            pruned_pkv = []
            for k, v in past_key_values:
                pruned_k = k[:, :, keep_indices, :]
                pruned_v = v[:, :, keep_indices, :]
                pruned_pkv.append((pruned_k, pruned_v))
            
            pruned_pkv = tuple(pruned_pkv)
            pruned_levels = torch.cat([levels_cache[:, keep_indices], new_levels], dim=1)

        return {
            "input_ids": input_ids[:, -1:], # We only need to process the last token
            "past_key_values": pruned_pkv,
            "use_cache": kwargs.get("use_cache"),
            "levels_cache": pruned_levels,
        }

    def denoise(self, idx: torch.Tensor, denoise_mask: torch.Tensor, denoise_levels: torch.Tensor, temperature: float = 0.0): 
        hidden_states, _ = self.model(idx, use_cache=False)
        rep_mask = torch.roll(denoise_mask, -1, dims=1)
        logits = self.lm_head(hidden_states[rep_mask])
        next_token = self._decode(logits, levels=denoise_levels, temperature=temperature)
        idx[denoise_mask] = next_token
        return idx
    
    def _decode(self, logits: torch.Tensor, levels: Optional[torch.Tensor] = None, temperature: float = 0.0):
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        logits[:, self.level_mask_tokens] = float('-inf')

        if levels is not None:
            assert levels.shape == logits.shape[:-1]
            start_logits = self.level_vocab_starts[levels]
            end_logits = self.level_vocab_ends[levels]

            vocab_indices = torch.arange(logits.size(-1), device=self.device)
            mask = (vocab_indices >= start_logits.unsqueeze(-1)) & (vocab_indices < end_logits.unsqueeze(-1))
            logits = torch.where(mask, logits, torch.tensor(-float('inf')))

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            next_token = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)
        return next_token
