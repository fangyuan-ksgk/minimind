# Information Gain Reward for Abstraction 
# Source: https://github.com/NVlabs/RLP?tab=readme-ov-file#paper

# InfoGain(a) = Policy(x_{i} | x_{<i}, a) - EMA(Policy(x_{i} | x_{<i}))

from copy import deepcopy
import torch
from src.model import SorlModelWrapper

class EMAModel:
    """
    Maintains an Exponential Moving Average of a model's parameters.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, main_model: torch.nn.Module):
        training_mode = main_model.training
        main_model.eval()

        for ema_param, main_param in zip(self.ema_model.parameters(), main_model.parameters()):
            ema_param.data.mul_(self.decay).add_(main_param.data, alpha=1 - self.decay)

        if training_mode:
            main_model.train()

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


# Note 1. Sorl_search 