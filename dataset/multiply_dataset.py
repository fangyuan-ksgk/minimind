import random
import torch
from typing import List, Tuple
from dataset.base import SavableDataset
import tqdm
from transformers import PreTrainedTokenizer

# =======================================================
# Following multiplication experiment in iCoT: 
#   - https://github.com/da03/implicit_chain_of_thought
#   - CoT, reverse digits multiplication dataset
# =======================================================
# Question 1. why is the split pattern necessary, why the extra '||' is necessary?
# - Answer 1. #### differentiates answer from CoT. '||' is rather redundant, just a placeholder for preprocessing etc.
# -           <end-of-text> is quite redundant, too, but fine

# Reflection 1. Do note that we'd like to store things as binary files

def generate_multiplication_line(num1: int, num2: int, use_cot: bool, reverse_digits: bool) -> str:
    """Generates a multiplication problem string with optional schoolbook-style Chain-of-Thought."""
    
    def fmt(n: int, pad_len: int = 0) -> str:
        """Format a number with optional padding and digit reversal."""
        s = str(n).zfill(pad_len)
        return ''.join(s[::-1] if reverse_digits else s)
    
    question = f"{fmt(num1)} * {fmt(num2)}"
    final_answer = fmt(num1 * num2)
    
    if not use_cot:
        return f"{question} = <answer> {final_answer}"
    
    partial_products = [int(d) * num2 for d in str(num1)[::-1]]
    cot_parts = []
    current_sum = 0
    
    pad_len = len(str(num2))
    cot_parts.append(fmt(partial_products[0], pad_len))
    current_sum = partial_products[0]
    
    for i, p_prod in enumerate(partial_products[1:], start=1):
        step_product = p_prod * (10**i)
        current_sum += step_product
        pad_len += 1
        
        summand = fmt(step_product, pad_len)
        cot_parts.append(f"{summand} ({fmt(current_sum, pad_len)})" if i < len(partial_products) - 1 else summand)
    
    return f"{question} = {' + '.join(cot_parts)} = <answer> {final_answer}"


def extract_answer(text):
    """Extracts the answer from the format '...<answer> answer'."""
    split_pattern = '<answer>'
    if split_pattern not in text:
        return text.strip().replace(',', '')
    else:
        _, ans = text.strip().split('<answer>', 1)
        ans = '<answer>' + ans
        ans = ans.strip().replace(',', '')
        return ans

def extract_cot(text):
    """Extracts the chain of thought from the format 'cot<answer>...'."""
    split_pattern = '<answer>'
    if split_pattern not in text:
        return None
    else:
        cot, _ = text.strip().split('<answer>', 1)
        cot = cot.strip()
        return cot

class MultiplyDataset(SavableDataset):

    def __init__(self, tokenizer, num_samples: int, min_digits: int, max_digits: int, max_length=1024, max_size=-1, use_cot: bool = True, reverse_digits: bool = False, answer_mask: bool = False):
        # answer_mask: mask out query & cot loss || no answer_mask: only mask out query loss
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.max_length = max_length
        self.max_size = max_size
        self.use_cot = use_cot
        self.reverse_digits = reverse_digits

        self.data = self._generate_data()

        # Tokenizer specifics (when ' =' is a single token, we can't use '=' to separate the token ids)
        self.equals_token_id = self.tokenizer.encode(" =", add_special_tokens=False)
        assert len(self.equals_token_id) == 1, "The ' =' word should be a single token."
        self.equals_token_id = self.equals_token_id[0]

        self.answer_token = "<answer>" # followed by the answer
        self.answer_token_id = self.tokenizer.encode(self.answer_token, add_special_tokens=False)
        assert len(self.answer_token_id) == 1, "The '<answer>' token should be a single token."
        self.answer_token_id = self.answer_token_id[0]

        self.answer_mask = answer_mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.tokenizer.encode(self.data[i], add_special_tokens=False)
        input_ids = torch.tensor(text, dtype=torch.long)[:self.max_length]
        loss_mask = torch.ones(len(input_ids), dtype=torch.bool)

        if self.answer_mask: 
            sep_idx = torch.where(input_ids == self.answer_token_id)[0][0] + 1
            loss_mask[:sep_idx] = False
        else:
            sep_idx = torch.where(input_ids == self.equals_token_id)[0][0] + 1
            loss_mask[:sep_idx] = False
            
        return (input_ids,loss_mask)

    def _generate_data(self) -> List[str]:
        """Generates the raw <num1>*<num2>=<product> strings."""
        print(f"Generating {self.num_samples} multiplication samples for {self.min_digits}-digit numbers...")
        samples = []
        min_val = 10**(self.min_digits - 1) if self.min_digits > 1 else 0
        max_val = (10**self.max_digits) - 1
        
        for _ in tqdm.tqdm(range(self.num_samples), desc="Generating samples"):
            num1 = random.randint(min_val, max_val)
            num2 = random.randint(min_val, max_val)
            text = generate_multiplication_line(num1, num2, use_cot=self.use_cot, reverse_digits=self.reverse_digits)
            samples.append(text)
        return samples

    def __len__(self) -> int:
        return self.num_samples