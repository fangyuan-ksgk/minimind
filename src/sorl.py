from src.gat import GAT
from src.utils import infer_level, infer_timestamp, infer_rythmic_insertion_mask, insert_tokens, infer_spike_insertion_mask, infer_valid_masks, group_argmax, group_mean, compute_switch_ratio, combine_rollout
from copy import deepcopy
import torch
import torch.nn.functional as F
from typing import Optional, Union 
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from model.model_sorl import SorlModelWrapper


def compute_per_token_loss(model: SorlModelWrapper, idx: torch.Tensor) -> torch.Tensor:
    """
    Computes per-token loss (negative log-likelihood) using a standard forward pass.
    This replaces the custom `model(idx, target)` call from the original GAT model.
    """
    targets = idx[:, 1:].contiguous()
    inputs = idx[:, :-1].contiguous()
    
    outputs = model.forward(input_ids=inputs, output_hidden_states=False)
    logits = outputs.logits
    
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    
    pad_token_id = model.level_mask_tokens[0].item()
    losses = F.cross_entropy(logits_flat, targets_flat, reduction='none', ignore_index=pad_token_id)
    
    ppt = losses.view(targets.shape)
    return ppt


@dataclass 
class SORLConfig: 
    n: int # number of candidates to rollout 
    temperature: float 
    K: int # Rythmic stride for level-1 abstraction
    
    causal_rollout: bool = False # whether to use causal rollout
    budget: Optional[int] = None # max number of abstract tokens allowed

    # parameter used if not causal rollout
    l: Optional[int] = None # level of abstraction to search & learn
    steps: Optional[int] = None # steps for chunk-wise denoise
    max_t_search: Optional[int] = None # max number of tokens to search for abstraction
    start_ts: Optional[torch.Tensor] = None # start timestamp for adding placeholders | one for each sample
    end_ts: Optional[torch.Tensor] = None # end timestamp for adding placeholders | one for each sample
    abstract_budget: int = 5 # max number of spiky abstraction allowed
    use_rhythmic_placeholders: bool = True # whether to use rhythmic placeholders
    use_spike_placeholders: bool = True # whether to use spike placeholders
    temperature_flip: bool = False # Whether to alternate temperature between 0.0 and the specified value

    # incremental abstraction search 
    curriculum_ratio: float = 0.6 # ratio of curriculum iterations (after this total_step * ratio, abstraction is full-length)

    # memory fading 
    max_seq_len: int = 1024 # max sequence length for training data, useful for memory fading
    use_fade_memory: bool = False # whether to use memory fading
    min_keep: int = 1024 # default to same value as max_length
    use_compression_mask: bool = True # whether to use compression mask

    # dataset specific
    train_dataset_path: Optional[str] = None
    val_dataset_path: Optional[str] = None
    train_batch_size: int = 128
    val_batch_size: int = 128
    train_iterations: int = 1000
    val_iterations: int = 10
    max_length: int = 1024

    # optimization specific 
    learning_rate: float = 1e-3
    log_interval: int = 100

    # GAPT
    default_phase: Optional[int] = None # default phase for GAPT | when used, GAPT is disabled
    delta: float = 0.01
    tau: float = 0.1
    p_m: int = 10
    p_c: int = 10



# Placeholder addition function (for parallel search & training)
# -----------------------------------------------------------------------------------------------------
def add_rhythmic_placeholders(model: SorlModelWrapper, idx: torch.Tensor, l: int, K: int, t_search: Optional[Union[int, torch.Tensor]] = None, start_ts: Optional[torch.Tensor] = None, end_ts: Optional[torch.Tensor] = None): 
    
    tokens = deepcopy(idx)
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, K, l)
    insert_masks = infer_rythmic_insertion_mask(levels, timestamps, K, l)

    valid_masks = infer_valid_masks(timestamps, start_ts, t_search, end_ts)
    insert_masks *= valid_masks

    tokens = insert_tokens(tokens, insert_masks.int(), model.level_mask_tokens[l], model.level_mask_tokens[0])

    return tokens
    
def add_spike_placeholders(model: SorlModelWrapper, data: torch.Tensor, l: int, K: int, abstract_budget: int, t_search: Optional[Union[int, torch.Tensor]] = None, start_ts: Optional[torch.Tensor] = None, end_ts: Optional[torch.Tensor] = None): 

    tokens = deepcopy(data)
    idx = tokens[:, :-1].contiguous()
    target = tokens[:, 1:].contiguous()
    with torch.no_grad():
        ppt = compute_per_token_loss(model, tokens)

    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, K, l)
    insert_masks = infer_spike_insertion_mask(levels, ppt, l, abstract_budget)

    valid_masks = infer_valid_masks(timestamps, start_ts, t_search, end_ts)
    insert_masks *= valid_masks

    tokens = insert_tokens(tokens, insert_masks, model.level_mask_tokens[l], model.level_mask_tokens[0])

    return tokens

def pad_abstract_tokens(tokens: torch.Tensor, 
                        model: SorlModelWrapper,
                        l: int,
                        config: SORLConfig,
                        t_search: Optional[Union[int, torch.Tensor]] = None,
                        start_ts: Optional[torch.Tensor] = None, 
                        end_ts: Optional[torch.Tensor] = None):

    if config.use_spike_placeholders:
        assert config.abstract_budget is not None, "abstract_budgets must be provided for spike placeholders"

    if config.use_spike_placeholders:
        batch_data = add_spike_placeholders(model, tokens, l, config.K, config.abstract_budget, t_search, start_ts, end_ts)

    if config.use_rhythmic_placeholders:
        batch_data = add_rhythmic_placeholders(model, tokens, l, config.K, t_search, start_ts, end_ts)

    return batch_data

def repad_abstract_tokens(tokens: torch.Tensor, model: SorlModelWrapper, l: int, K: int, start_ts: torch.Tensor): 
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, K, l)
    repad_mask = (timestamps >= start_ts.unsqueeze(1)) & (levels == l)
    tokens[repad_mask] = model.level_mask_tokens[l]
    return tokens

def prep_denoise(tokens: torch.Tensor, model: SorlModelWrapper):
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    denoise_mask = torch.isin(tokens, model.level_mask_tokens[1:])
    denoise_levels = levels[denoise_mask]
    return denoise_mask, denoise_levels

def chunk_denoise(data: torch.Tensor, model: SorlModelWrapper, l: int, K: int, steps: int, 
                   max_t_search: Optional[int] = None, temperature: float = 0.0):

    tokens = deepcopy(data)
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, K, l)
    max_ts = timestamps.max(dim=1).values
    if max_t_search is not None:
        max_ts = torch.minimum(max_ts, torch.tensor(max_t_search))

    search_ts = torch.ceil(max_ts / steps)

    for step in range(steps): 

        start_ts = (search_ts * step).int()

        # What we actually need here is "repad" -- not new tricks required, just replace abstract tokens with "mask" from certain timestamp onwards
        tokens = repad_abstract_tokens(tokens, model, l, K, start_ts)

        denoise_mask, denoise_levels = prep_denoise(tokens, model)

        if denoise_levels.numel() > 0:
            with torch.no_grad():  
                tokens = model.denoise(tokens, denoise_mask, denoise_levels, temperature=temperature)

        if (start_ts >= max_ts).all(): 
            break 

    return tokens

def heuristic_rollout(data: torch.Tensor, model: SorlModelWrapper, l: int, n: int, config: SORLConfig):
    """Heuristic-based decision on when to generate abstraction"""

    data_idx = torch.arange(data.shape[0], device=data.device)

    repeat_data = data.repeat_interleave(n, dim=0)
    repeat_data_idx = data_idx.repeat_interleave(n, dim=0)

    repeat_data = pad_abstract_tokens(repeat_data, model, l, config, t_search=config.max_t_search, start_ts=config.start_ts, end_ts=config.end_ts)

    repeat_data = chunk_denoise(repeat_data, model, l, config.K, steps=config.steps, max_t_search=config.max_t_search, temperature=config.temperature)

    return repeat_data, repeat_data_idx

# Most beautiful way of doing rollout -- no heuristic, no manual placeholders, just model-based decision to maximize 'reward'
# - Issue: don't know when to stop (could generate abstraction forever)
def causal_generate(data: torch.Tensor, model: SorlModelWrapper, temperature: float, budget: int): 
    raise NotImplementedError("Causal generation is not ALLOWED!")
    """Model-based decision on when to generate abstraction"""

    pad_token_id = model.level_mask_tokens[0]   
    kv_cache, levels = None, None
    progress_idx = torch.zeros(data.size(0), dtype=torch.long)
    current_idx = data[torch.arange(data.size(0)), progress_idx].contiguous()

    new_data = current_idx.unsqueeze(1)
    abstract_tokens_used = torch.zeros(data.size(0), dtype=torch.long, device=data.device)

    while torch.any(progress_idx < data.size(1) - 1): 

        not_finished_mask = progress_idx < data.size(1) - 1
        next_idx, kv_cache, levels = model.generate(current_idx.unsqueeze(1), temperature=temperature, kv_cache=kv_cache, levels=levels)
        next_level = infer_level(next_idx, model.vocab_sizes, model.level_mask_tokens[0])
        traj_mask = (next_level == 0)
        effective_traj_mask = traj_mask & not_finished_mask

        current_idx = next_idx
       
        if torch.any(effective_traj_mask):
            safe_indices = torch.clamp(progress_idx + 1, max=data.size(1) - 1)
            gt_tokens = data[torch.arange(data.size(0), device=data.device), safe_indices]
            current_idx[effective_traj_mask] = gt_tokens[effective_traj_mask]
        
        progress_idx += effective_traj_mask.long()
        next_column_to_append = torch.full_like(current_idx, fill_value=pad_token_id)
        next_column_to_append[not_finished_mask] = current_idx[not_finished_mask]
        new_data = torch.cat([new_data, next_column_to_append.unsqueeze(1)], dim=1)

    return new_data


# def causal_generate(data: torch.Tensor, model: GAT, temperature: float, budget: int): 
#     """Model-based decision on when to generate abstraction"""

#     pad_token_id = model.level_mask_tokens[0]   
#     kv_cache, levels = None, None
#     progress_idx = torch.zeros(data.size(0), dtype=torch.long)
#     current_idx = data[torch.arange(data.size(0)), progress_idx].contiguous()

#     new_data = current_idx.unsqueeze(1)
    
#     abstract_tokens_used = torch.zeros(data.size(0), dtype=torch.long, device=data.device)

#     while torch.any(progress_idx < data.size(1) - 1): 

#         not_finished_mask = progress_idx < data.size(1) - 1
#         next_idx, kv_cache, levels = model.generate(current_idx.unsqueeze(1), temperature=temperature, kv_cache=kv_cache, levels=levels)
#         next_level = infer_level(next_idx, model.vocab_sizes, model.level_mask_tokens[0])
        
#         abstract_mask = (next_level > 0) & not_finished_mask
        
#         budget_exceeded_mask = (abstract_tokens_used >= budget) & abstract_mask
        
#         traj_mask = (next_level == 0) | budget_exceeded_mask
#         effective_traj_mask = traj_mask & not_finished_mask

#         actual_abstract_mask = (next_level > 0) & not_finished_mask & ~budget_exceeded_mask
#         abstract_tokens_used += actual_abstract_mask.long()

#         current_idx = next_idx
        
#         if torch.any(budget_exceeded_mask):
#             safe_indices = torch.clamp(progress_idx + 1, max=data.size(1) - 1)
#             gt_tokens = data[torch.arange(data.size(0)), safe_indices]
#             current_idx[budget_exceeded_mask] = gt_tokens[budget_exceeded_mask]
#             effective_traj_mask = effective_traj_mask | budget_exceeded_mask
       
#         if torch.any(effective_traj_mask):
#             safe_indices = torch.clamp(progress_idx + 1, max=data.size(1) - 1)
#             gt_tokens = data[torch.arange(data.size(0)), safe_indices]
#             current_idx[effective_traj_mask] = gt_tokens[effective_traj_mask]
        
#         progress_idx += effective_traj_mask.long()
#         next_column_to_append = torch.full_like(current_idx, fill_value=pad_token_id)
#         next_column_to_append[not_finished_mask] = current_idx[not_finished_mask]
#         new_data = torch.cat([new_data, next_column_to_append.unsqueeze(1)], dim=1)

#     return new_data


def causal_rollout(data: torch.Tensor, model: SorlModelWrapper, temperature: float, n: int, budget: int):
    """The simplificty of this procedure speaks for itself"""
    repeat_data = data.repeat_interleave(n, dim=0)
    repeat_data_idx = torch.arange(data.shape[0], device=data.device).repeat_interleave(n, dim=0)
    repeat_data = causal_generate(repeat_data, model, temperature, budget)
    return repeat_data, repeat_data_idx


def sorl_search(data: torch.Tensor, model: SorlModelWrapper, config: SORLConfig): 

    # greedy-involved rollout
    assert config.n > 1, "n must be greater than 1"
    with torch.no_grad(): 
        greedy_config = deepcopy(config)
        greedy_config.temperature = 0.0
        search_config = deepcopy(config)
        search_config.temperature = config.temperature

        greedy_data, greedy_data_idx = heuristic_rollout(data, model, l=config.l, n=1, config=greedy_config)
        search_data, search_data_idx = heuristic_rollout(data, model, l=config.l, n=config.n-1, config=search_config)

    combined_data, combined_data_idx = combine_rollout(greedy_data, greedy_data_idx, search_data, search_data_idx, model.level_mask_tokens[0])

    with torch.no_grad():
        ppt = compute_per_token_loss(model, combined_data)

    # select best for each sample idx
    idx_max = group_argmax(ppt.mean(axis=1), combined_data_idx)
    best_data = combined_data[idx_max]

    switch_ratio = compute_switch_ratio(idx_max, data.size(0))

    return best_data, switch_ratio

# loss computation (level-0 is trajectory loss, level >= 1 is abstract loss)
# ------------------------------------------------------------------------------------------------
def compute_loss(data: torch.Tensor, model: SorlModelWrapper, ppt: torch.Tensor): 
    levels = infer_level(data, model.vocab_sizes, model.level_mask_tokens[0])
    level_loss = {l: torch.tensor(0.) for l in range(len(model.full_vocab_size_list))}
    level_loss.update(group_mean(ppt, levels[:, 1:]))
    return level_loss[0], sum(level_loss[l] for l in level_loss if l > 0)

# Sub-optimal way of evaluating search improvement || We'd like to have "evaluate" function that properly does token-by-token generation
# ---------------------------------------------------------------------------------------------------------------------------------------
def compute_vocab_utilization_rate(data: torch.Tensor, model: SorlModelWrapper):
    si, ei = model.vocab_sizes.cumsum(dim=0)
    return data[(data >= si) & (data < ei)].unique().size(0) / (ei - si).item()


def evaluate(data: torch.Tensor, model: SorlModelWrapper, n: int, config: SORLConfig): 
    """Causal rollout do not assume full trajectory to add abstraction, it instead perform causal generation, suited for evaluation"""

    with torch.no_grad():        
        assert n > 1, "n must be greater than 1"
        
        greedy_config = deepcopy(config)
        greedy_config.temperature = 0.0
        search_config = deepcopy(config)
        search_config.temperature = 100.0

        greedy_data, _ = heuristic_rollout(data, model, l=config.l, n=1, config=greedy_config)
        search_data, _ = heuristic_rollout(data, model, l=config.l, n=n-1, config=search_config)

        greedy_ppt = compute_per_token_loss(model, greedy_data)
        search_ppt = compute_per_token_loss(model, search_data)

        greedy_ppl = compute_loss(greedy_data, model, greedy_ppt)[0]
        search_ppl = compute_loss(search_data, model, search_ppt)[0]

    improve_ppl_percentage = (search_ppl - greedy_ppl) / search_ppl # percentage of improvement in ppl

    vocab_utilization_rate = compute_vocab_utilization_rate(greedy_data, model)

    return greedy_ppl, improve_ppl_percentage.mean() * 100, greedy_data, vocab_utilization_rate



class SearchScheduler: 
    def __init__(self, sorl_config: SORLConfig): 
        self.sorl_config = sorl_config
        self.K = sorl_config.K
        self.max_ts = min(sorl_config.max_length // self.K, sorl_config.max_t_search)
        self.curriculum_iterations = int(sorl_config.train_iterations * sorl_config.curriculum_ratio)
        if self.curriculum_iterations > 0:
            self.t_delta = self.max_ts / self.curriculum_iterations
        else:
            self.t_delta = 0
        self.drop_ratio_delta = 1.0 / self.curriculum_iterations # static curriculum till full compression
        self.t_search = 0 
        self.drop_ratio = 0.0

        # memory fading experiment || gradually decrease 't_keep' at the later half of training
        self.use_fade_memory = sorl_config.use_fade_memory
        self.max_memory_span = sorl_config.max_seq_len
        self.min_memory_span = sorl_config.min_keep
        self.use_compression_mask = sorl_config.use_compression_mask
        self.memory_span = self.max_memory_span

    def step(self): 
        self.t_search = min(self.t_search + self.t_delta, self.max_ts)
        if self.use_fade_memory:
            new_memory_span = max(self.max_memory_span - (self.t_search * self.K), self.min_memory_span)
            # Apply smoothing to avoid abrupt changes
            self.memory_span = max(int(0.7 * self.memory_span + 0.3 * new_memory_span), self.min_memory_span)
        if self.use_compression_mask:
            self.drop_ratio = min(self.drop_ratio + self.drop_ratio_delta, 1.0)
        return int(self.t_search), self.drop_ratio


# Phase change regulator
class GatedPhaseTransition:
    """
    Gated Phase Transition (GAPT) : https://arxiv.org/pdf/2505.08727
    """
    def __init__(self, delta: float, tau: float, p_m: int, p_c: int):

        # Hyperparameters
        self.delta = delta
        self.tau = tau
        self.p_m = p_m
        self.p_c = p_c

        # State
        self.phi = 1  # 1 for memorization, 2 for compression
        self.s_m = 0
        self.s_c = 0
        self.E_min = float('inf')
        self.abs_loss_min = float('inf')

    def step(self, ssl_loss: float, abs_loss: float):

        L_ce = ssl_loss

        delta_E = self.E_min - L_ce
        self.E_min = min(self.E_min, L_ce)

        if self.phi == 1:  # Memorization phase
            if delta_E > self.delta:
                self.s_m = 0
            else:
                self.s_m += 1
            
            if self.s_m >= self.p_m:
                self.phi = 2
                self.s_c = 0
                self.E_min = float('inf')  # Reset for the new phase
                self.abs_loss_min = float('inf')

        elif self.phi == 2:  # Compression phase
            # Condition 1: SSL loss spikes
            if self.E_min != float('inf') and L_ce > self.E_min * (1 + self.tau):
                self.phi = 1
                self.s_m = 0
            else:
                # Condition 2: Abstraction loss plateaus
                delta_M = self.abs_loss_min - abs_loss
                self.abs_loss_min = min(self.abs_loss_min, abs_loss)

                if delta_M > self.delta:
                    self.s_c = 0
                else:
                    self.s_c += 1
                
                if self.s_c >= self.p_c:
                    self.phi = 1
                    self.s_m = 0
        
        return self.phi 