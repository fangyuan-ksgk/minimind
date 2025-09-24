import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from transformers import PreTrainedModel, LogitsProcessor

from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                     Compatibility Subclass
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

class MiniMindForGeneration(MiniMindForCausalLM):
    """
    A subclass of MiniMindForCausalLM that adds the required `prepare_inputs_for_generation`
    method, making it fully compatible with the Hugging Face generate() loop.
    """
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # This is a standard implementation for decoder-only models
        if past_key_values:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": kwargs.get("attention_mask"),
        }

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             SORL Utilities
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

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

class ForceAbstractionLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor to force the model to select a specific abstraction token.
    """
    def __init__(self, abstraction_token_id: int):
        self.abstraction_token_id = abstraction_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, self.abstraction_token_id] = False
        scores.masked_fill_(mask, -float("inf"))
        return scores

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                       SORL Model Wrapper
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

class SorlModelWrapper:
    def __init__(self, model: MiniMindForCausalLM, sorl_config: dict):
        # Use our new compatible subclass, copying the weights from the original model
        self.model = MiniMindForGeneration(model.config)
        self.model.load_state_dict(model.state_dict())
        self.model.to(model.device) # Ensure it's on the same device
        self.config = sorl_config
        
        # SORL specific vocabulary setup
        vocab_sizes = torch.tensor(self.config["vocab_size_list"]) + 1
        self.vocab_sizes = vocab_sizes
        self.level_mask_tokens = self.vocab_sizes.cumsum(dim=0) - 1

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
        device = self.model.device
        
        generated_ids = input_ids.clone()
        levels_cache = infer_level(input_ids, self.vocab_sizes, self.level_mask_tokens[0])
        past_key_values = None
        
        for step in range(max_new_tokens):
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids=generated_ids, past_key_values=past_key_values
            )
            
            outputs = self.model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]
            
            # This is the new, unpruned cache from the model's output
            current_pkv = outputs.past_key_values

            # --- Pruning logic for the *next* iteration's state ---
            if current_pkv is not None:
                seq_len = current_pkv[0][0].shape[2]
                
                # Ensure levels_cache corresponds to the KV cache we are about to prune
                assert seq_len == levels_cache.shape[1], f"State mismatch: KV cache length {seq_len} != levels_cache length {levels_cache.shape[1]}"

                is_recent = torch.arange(seq_len, device=device) >= (seq_len - self.config["memory_span"])
                is_high_level = (levels_cache > 0).squeeze(0)

                keep_mask = is_recent | is_high_level
                keep_indices = torch.where(keep_mask)[0]
                
                pruned_pkv = []
                for k, v in current_pkv:
                    pruned_k = k[:, :, keep_indices, :]
                    pruned_v = v[:, :, keep_indices, :]
                    pruned_pkv.append((pruned_k, pruned_v))
                
                # Update state for the next iteration
                past_key_values = tuple(pruned_pkv)
                levels_cache = levels_cache[:, keep_indices] # CRITICAL: Prune levels_cache
            else:
                 past_key_values = current_pkv

            logits_processor = None
            if force_abstraction_every_n is not None and (step + 1) % force_abstraction_every_n == 0:
                abstraction_token = int(self.vocab_sizes[:-1].sum())
                logits_processor = ForceAbstractionLogitsProcessor(abstraction_token)
                next_token_logits = logits_processor(generated_ids, next_token_logits)

            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                next_token_id = top_k_indices[0, torch.multinomial(top_k_probs, num_samples=1)[0]]
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1)

            next_token_id = next_token_id.unsqueeze(0)
            
            # Update generated sequence and levels_cache with the new token
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            new_level = infer_level(next_token_id, self.vocab_sizes, self.level_mask_tokens[0])
            levels_cache = torch.cat([levels_cache, new_level], dim=1)
            
        return generated_ids
