from dataset.hidden_info_dataset import HiddenInfoDataset
from transformers import AutoTokenizer

def main():
    # --- Configuration ---
    TOKENIZER_PATH = "model/"
    OUTPUT_FILE = "dataset/hidden_info.bin"
    NUM_SAMPLES = 20
    MAX_LENGTH = 64
    MIN_DIGITS = 1
    MAX_DIGITS = 1

    # --- Initialization ---
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # The first token ID after the base vocabulary is our abstraction token
    abstract_token_id = tokenizer.vocab_size

    # --- Create and Save Dataset ---
    dataset = HiddenInfoDataset(
        tokenizer=tokenizer,
        num_samples=NUM_SAMPLES,
        abstract_token_id=abstract_token_id,
        max_length=MAX_LENGTH,
        min_digits=MIN_DIGITS,
        max_digits=MAX_DIGITS,
    )
    
    dataset.save_to_bin(OUTPUT_FILE)
    print(f"\nSuccessfully created 'hidden info' dataset at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()