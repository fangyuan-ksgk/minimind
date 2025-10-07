# Train BPE tokenizer on multiplication dataset
# ------------------------------------------------------------

import os
import json
import random
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Digits, Sequence, Whitespace
from transformers import PreTrainedTokenizerFast
from dataset.multiply_dataset import generate_multiplication_line


def create_training_corpus(num_samples=20000, min_digits=1, max_digits=4):
    """Generates a text corpus of multiplication problems for training a tokenizer."""
    print(f"Generating a training corpus of {num_samples*2} multiplication examples...")
    corpus = []
    for _ in range(num_samples):
        d1 = random.randint(min_digits, max_digits)
        d2 = random.randint(min_digits, max_digits)
        num1 = random.randint(10**(d1-1) if d1 > 1 else 0, 10**d1 - 1)
        num2 = random.randint(10**(d2-1) if d2 > 1 else 0, 10**d2 - 1)
        corpus.append(generate_multiplication_line(num1, num2, use_cot=True, reverse_digits=False))
        corpus.append(generate_multiplication_line(num1, num2, use_cot=False, reverse_digits=False))
    return corpus


def train_bpe_multiply(vocab_size: int):
    """Trains a BPE tokenizer on multiplication data and saves it."""
    print(f"--- Training BPE Tokenizer with Vocab Size: {vocab_size} ---")
    output_path = f"tokenizer/bpe_tokenizer_v{vocab_size}"
    
    # 1. Create the training corpus
    training_corpus = create_training_corpus()

    # 2. Initialize and configure the BPE tokenizer
    special_tokens = ["[UNK]", "[PAD]", "<answer>", "<eos>"]
    unk_token = "[UNK]"
    pad_token = "[PAD]"

    bpe_tokenizer = Tokenizer(BPE(unk_token=unk_token))
    bpe_tokenizer.pre_tokenizer = Whitespace()

    # 3. Train the new tokenizer
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    bpe_tokenizer.train_from_iterator(training_corpus, trainer=trainer)

    # 4. Save the tokenizer and its config file
    _save_tokenizer_files(bpe_tokenizer, output_path, unk_token, pad_token)
    
    # 5. Show an example
    _test_tokenizer(output_path, "99 * 30 = <answer> 2970 <eos>")
    

def build_and_save_baseline_tokenizer():
    """Builds and saves a simple, character-level tokenizer for digits and symbols."""
    print("--- Building Baseline Digit-Level Tokenizer ---")
    output_path = "tokenizer/digit_tokenizer"

    unk_token = "[UNK]"
    pad_token = "[PAD]"
    answer_token = "<answer>"
    eos_token = "<eos>"
    vocab = [unk_token, pad_token, answer_token] + list('0123456789*=+() ')
    
    simple_tokenizer = Tokenizer(WordLevel({token: i for i, token in enumerate(vocab)}, unk_token=unk_token))
    simple_tokenizer.add_special_tokens([unk_token, pad_token, answer_token, eos_token])
    
    simple_tokenizer.pre_tokenizer = Sequence([
        Whitespace(), 
        Digits(individual_digits=True)
    ])

    _save_tokenizer_files(simple_tokenizer, output_path, unk_token, pad_token)

    _test_tokenizer(output_path, "1912 * 2025 = <answer> 3868800 <eos>")


def _save_tokenizer_files(tokenizer: Tokenizer, path: str, unk_token: str, pad_token: str):
    """Helper function to save tokenizer.json and tokenizer_config.json."""
    os.makedirs(path, exist_ok=True)
    tokenizer_file = os.path.join(path, "tokenizer.json")
    tokenizer.save(tokenizer_file)

    config = {
      "tokenizer_class": "PreTrainedTokenizerFast",
      "unk_token": unk_token,
      "pad_token": pad_token,
      "model_max_length": 512,
      "additional_special_tokens": ["<answer>", "<eos>"]
    }
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Tokenizer saved to '{path}'")


def _test_tokenizer(path: str, example_str: str):
    """Loads a saved tokenizer and prints an example of its tokenization."""
    tokenizer_file = os.path.join(path, "tokenizer.json")
    loaded_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    loaded_tokenizer.pad_token = "[PAD]"
    tokens = loaded_tokenizer.tokenize(example_str)
    
    print("\n--- Example Tokenization ---")
    print(f"Expression: '{example_str}'")
    print(f"  -> Tokens: {tokens}")
    print(f"Vocabulary size: {loaded_tokenizer.vocab_size}\n")


if __name__ == "__main__":
    # Example of how to run the functions
    
    # 1. Build the simple baseline tokenizer
    build_and_save_baseline_tokenizer()

    for vocab_size in [50, 100, 150, 200, 250]:
        train_bpe_multiply(vocab_size=vocab_size)