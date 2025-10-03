import random, torch
from typing import List, Tuple
from dataset.base import SavableDataset
import tqdm
from transformers import PreTrainedTokenizer

# ----------------------------------------------------------------------------------
# 3. New Dataset for "Hidden Information" Task
# ----------------------------------------------------------------------------------
class HiddenInfoDataset(SavableDataset):
    """
    Creates a dataset for the task: <number>A -> <number>.
    The format is `<num1>=<num2>`, which is transformed into `<num1_toks>=<abs_tok><num2_toks>`.
    The loss is only calculated on the prediction of `<num2_toks>`.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_samples: int,
        abstract_token_id: int,
        max_length: int = 64,
        min_digits: int = 2,
        max_digits: int = 8,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        self.min_digits = min_digits
        self.max_digits = max_digits

        self.equals_token_id = self.tokenizer.encode("=", add_special_tokens=False)
        assert len(self.equals_token_id) == 1, "The '=' character should be a single token."
        self.equals_token_id = self.equals_token_id[0]

        self.data = self._generate_data()

    def _generate_data(self) -> List[str]:
        """Generates the raw <num1>=<num2> strings."""
        print(f"Generating {self.num_samples} samples...")
        samples = []
        for _ in tqdm.tqdm(range(self.num_samples), desc="Generating samples"):
            _len = random.randint(self.min_digits, self.max_digits)
            min_val = 10**(_len - 1) if _len > 1 else 0
            max_val = (10**_len) - 1
            num = random.randint(min_val, max_val)
            samples.append(f"{num}={num}")
        return samples

    @property
    def has_custom_loss_mask(self) -> bool:
        return True

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text = self.data[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        try:
            equals_idx = tokens.index(self.equals_token_id)
        except ValueError:
            return self.__getitem__((idx + 1) % len(self))

        seq_len = len(tokens)
        loss_mask = torch.zeros(seq_len, dtype=torch.long)
        loss_mask[equals_idx + 1:] = 1

        tokens = tokens[:self.max_length]
        loss_mask = loss_mask[:self.max_length]

        return torch.tensor(tokens, dtype=torch.long), loss_mask