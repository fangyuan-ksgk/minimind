import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from transformers import PreTrainedModel, GenerationMixin, Qwen2ForCausalLM, Qwen2Config, PretrainedConfig, AutoConfig
from transformers.cache_utils import DynamicCache

from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

# --------------------------------------------------------------------------------------------------#
#                                             SORL Utilities
# --------------------------------------------------------------------------------------------------#

def infer_level(indices: torch.Tensor, vocab_sizes: torch.Tensor, pad_token: int):
    """
    Infers the abstraction level of each token based on its ID and the vocabulary sizes.
    """
    device = indices.device
    vocab_sizes = vocab_sizes.to(device)
    pad_token = pad_token.to(device) if torch.is_tensor(pad_token) else torch.tensor(pad_token, device=device)
    indices_expanded = indices.unsqueeze(-1)
    levels = (indices_expanded < vocab_sizes.cumsum(dim=0)).int().argmax(dim=-1)
    padding_mask = (indices == pad_token)
    final_levels = torch.where(padding_mask, -1, levels.long())
    return final_levels

def memory_pruning(current_pkv, levels_cache: torch.Tensor, generated_ids: torch.Tensor, memory_span: int):
    """
    Prunes the KV cache, levels cache, and generated IDs based on a memory span and token abstraction level.
    This function handles both tuple-based KV caches (e.g., MiniMind) and DynamicCache objects (e.g., Qwen2).
    """
    seq_len = levels_cache.shape[1]
    device = levels_cache.device

    is_recent = torch.arange(seq_len, device=device) >= (seq_len - memory_span)
    is_high_level = (levels_cache > 0).squeeze(0)
    keep_mask = is_recent | is_high_level
    keep_indices = torch.where(keep_mask)[0]

    pruned_levels_cache = levels_cache[:, keep_indices]
    pruned_generated_ids = generated_ids[:, keep_indices]

    if isinstance(current_pkv, DynamicCache):
        assert current_pkv[0][0].shape[2] == seq_len, "CRITICAL: State mismatch before pruning!"
        
        pruned_layers = []
        for k_layer, v_layer in current_pkv:
            pruned_k = k_layer[:, :, keep_indices, :]
            pruned_v = v_layer[:, :, keep_indices, :]
            pruned_layers.append((pruned_k, pruned_v))
        
        pruned_pkv = DynamicCache.from_legacy_cache(past_key_values=tuple(pruned_layers))

    elif isinstance(current_pkv, tuple) or isinstance(current_pkv, list):
        assert current_pkv[0][0].shape[1] == seq_len, "CRITICAL: State mismatch before pruning!"
        
        pruned_pkv_list = []
        for k, v in current_pkv:
            pruned_k = k[:, keep_indices, :, :]
            pruned_v = v[:, keep_indices, :, :]
            pruned_pkv_list.append((pruned_k, pruned_v))
        pruned_pkv = tuple(pruned_pkv_list)
    else:
        raise TypeError(f"Unsupported KV cache type for pruning: {type(current_pkv)}")

    return pruned_pkv, pruned_levels_cache, pruned_generated_ids


# --------------------------------------------------------------------------------------------------#
#                                       SORL Model Wrapper
#---------------------------------------------------------------------------------------------------#

SUPPORTED_MODELS = {
    "minimind": (MiniMindForCausalLM, MiniMindConfig),
    "qwen2": (Qwen2ForCausalLM, Qwen2Config),
}

# By inheriting from PreTrainedModel, the wrapper gains methods like .to(), .eval(), .train(),
# and becomes a proper PyTorch nn.Module that can be managed easily.
class SorlModelWrapper(PreTrainedModel, GenerationMixin):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        
        model_type = getattr(config, "model_type", None)
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_type = model_type
        ModelClass, _ = SUPPORTED_MODELS[model_type]
        
        self.model = ModelClass(config)
        self.memory_span = None
        self.full_vocab_size_list = None
        self.pad_token_id = None
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, abstract_vocab_size_list: List[int], memory_span: int, pad_token_id: int, **kwargs) -> "SorlModelWrapper":
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(config)
        wrapper.model = wrapper.model.__class__.from_pretrained(model_name_or_path, **kwargs)

        wrapper.memory_span = memory_span
        wrapper.pad_token_id = pad_token_id
        base_vocab_size = config.vocab_size
        wrapper.full_vocab_size_list = [base_vocab_size] + abstract_vocab_size_list
        wrapper._setup_vocabulary()
        
        new_total_vocab_size = wrapper.total_vocab_size.item()
        if wrapper.model.config.vocab_size != new_total_vocab_size:
            wrapper.model.resize_token_embeddings(new_total_vocab_size)
            wrapper.model.config.vocab_size = new_total_vocab_size
            wrapper.config.vocab_size = new_total_vocab_size

        return wrapper
    
    @classmethod
    def from_scratch(cls, config: PretrainedConfig, full_vocab_size_list: List[int], memory_span: int, pad_token_id: int) -> "SorlModelWrapper":
        """A custom initializer for creating a SORL model from scratch."""
        wrapper = cls(config)
        wrapper.memory_span = memory_span
        wrapper.pad_token_id = pad_token_id
        wrapper.full_vocab_size_list = full_vocab_size_list
        wrapper._setup_vocabulary()
        
        new_total_vocab_size = wrapper.total_vocab_size.item()
        wrapper.model.resize_token_embeddings(new_total_vocab_size)
        wrapper.config.vocab_size = new_total_vocab_size
        return wrapper

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))

        levels = infer_level(input_ids, self.vocab_sizes, -1)
        q_pos = torch.arange(seq_len, device=device).view(-1, 1)
        k_pos = torch.arange(seq_len, device=device).view(1, -1)
        is_recent = q_pos >= (k_pos - self.memory_span)
        is_high_level = (levels > 0).unsqueeze(1)
        sorl_causal_mask = causal_mask.unsqueeze(0) & (is_recent.unsqueeze(0) | is_high_level)

        if attention_mask is not None:
            sorl_causal_mask = sorl_causal_mask & attention_mask.unsqueeze(1).unsqueeze(2)
        final_attention_mask = sorl_causal_mask.unsqueeze(1)

        return self.model.forward(input_ids=input_ids, attention_mask=final_attention_mask, **kwargs)


    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": kwargs.get("attention_mask"),
        }

    def _setup_vocabulary(self):
        device = self.device
        
        base_vocab_size = self.full_vocab_size_list[0]
        abstract_vocab_sizes = self.full_vocab_size_list[1:]

        vocab_sizes_list = [base_vocab_size] + [size + 1 for size in abstract_vocab_sizes]
        self.vocab_sizes = torch.tensor(vocab_sizes_list, device=device)

        self.total_vocab_size = self.vocab_sizes.sum()
        self.level_vocab_ends = self.vocab_sizes.cumsum(dim=0)
        self.level_vocab_starts = torch.cat([torch.tensor([0], device=device), self.level_vocab_ends[:-1]])

        l0_mask_token = torch.tensor([self.pad_token_id], device=device, dtype=torch.long)
        
        if len(self.vocab_sizes) > 1:
            abstract_mask_tokens = (self.level_vocab_ends - 1)[1:]
            self.level_mask_tokens = torch.cat([l0_mask_token, abstract_mask_tokens])
        else:
            self.level_mask_tokens = l0_mask_token

        self.l0_mask = torch.zeros(self.total_vocab_size, dtype=torch.bool, device=device)
        self.l0_mask[:self.vocab_sizes[0]] = True
        self.abs_mask = ~self.l0_mask
        self.abs_mask[self.level_mask_tokens[1:]] = False

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int = 50,
        force_abstraction_every_n: Optional[int] = None,
    ):

        self.model.eval()

        generated_ids = input_ids.clone()
        past_key_values = None
        levels_cache = infer_level(generated_ids, self.vocab_sizes, -1) # Use -1 for pad_token, assuming it won't be in the prompt

        for step in range(max_new_tokens):
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=generated_ids, past_key_values=past_key_values
            )
            
            outputs = self.model.forward(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]
            current_pkv = outputs.past_key_values

            past_key_values, levels_cache, generated_ids = memory_pruning(current_pkv, levels_cache, generated_ids, 5)

            if force_abstraction_every_n is not None and (step + 1) % force_abstraction_every_n == 0:
                next_token_logits.masked_fill_(self.l0_mask, -float("inf"))
            else: 
                next_token_logits.masked_fill_(self.abs_mask, -float("inf"))

            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                next_token_id = top_k_indices[0, torch.multinomial(top_k_probs, num_samples=1)[0]]
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1)

            next_token_id = next_token_id.unsqueeze(0)
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            new_level = infer_level(next_token_id, self.vocab_sizes, -1)
            levels_cache = torch.cat([levels_cache, new_level], dim=1)
            
        return generated_ids


    def _parallel_decode(self, logits: torch.Tensor, levels: Optional[torch.Tensor] = None, temperature: float = 0.0):
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        
        total_vocab_size = logits.shape[-1]
        valid_mask_tokens = self.level_mask_tokens[self.level_mask_tokens < total_vocab_size]
        logits[:, valid_mask_tokens] = float('-inf')

        if levels is not None:
            assert levels.size(0) == logits.shape[0], f"[Denoise level & mask mismatch] \n - Need to denoise on {logits.shape[0]} tokens, but got {levels.shape[-1]} levels"
            start_logits = self.level_vocab_starts.to(levels.device)[levels]
            end_logits = self.level_vocab_ends.to(levels.device)[levels]
            vocab_indices = torch.arange(logits.size(-1), device=logits.device)
            mask = (vocab_indices >= start_logits.unsqueeze(-1)) & (vocab_indices < end_logits.unsqueeze(-1))
            logits = torch.where(mask, logits, torch.tensor(-float('inf'), device=logits.device))

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            next_token = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)
        return next_token


    def denoise(self, idx: torch.Tensor, denoise_mask: torch.Tensor, denoise_levels: torch.Tensor, temperature: float = 0.0): 
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "minimind":
                outputs = self.forward(input_ids=idx, use_cache=False, attention_mask=None)
                hidden_states = outputs.last_hidden_state
            else: 
                outputs = self.forward(input_ids=idx, use_cache=False, attention_mask=None, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
            rep_mask = torch.roll(denoise_mask, -1, dims=1)
            new_tokens = self._parallel_decode(self.model.lm_head(hidden_states[rep_mask]), levels=denoise_levels, temperature=temperature)
            denoised_idx = idx.clone()
            denoised_idx[denoise_mask] = new_tokens

        return denoised_idx
