from dataset.multiply_dataset import MultiplyDataset
from transformers import AutoTokenizer

def main():
    # --- Configuration ---
    TOKENIZER_PATH = "model/"
    TRAIN_OUTPUT_FILE = "dataset/multiply_train.bin"
    VAL_OUTPUT_FILE = "dataset/multiply_val.bin"
    TRAIN_SAMPLES = 80800
    VAL_SAMPLES = 1200
    MAX_LENGTH = 128
    NUM_DIGITS = 4  # For 4x4 digit multiplication

    # --- Initialization ---
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # --- Create and Save Training Dataset ---
    train_dataset = MultiplyDataset(
        tokenizer=tokenizer,
        num_samples=TRAIN_SAMPLES,
        max_length=MAX_LENGTH,
        num_digits=NUM_DIGITS,
    )
    
    train_dataset.save_to_bin(TRAIN_OUTPUT_FILE)
    print(f"\nSuccessfully created multiplication training dataset at: {TRAIN_OUTPUT_FILE}")

    # --- Create and Save Validation Dataset ---
    print("\n--- Generating validation set ---")
    val_dataset = MultiplyDataset(
        tokenizer=tokenizer,
        num_samples=VAL_SAMPLES,
        max_length=MAX_LENGTH,
        num_digits=NUM_DIGITS,
    )
    
    val_dataset.save_to_bin(VAL_OUTPUT_FILE)
    print(f"\nSuccessfully created multiplication validation dataset at: {VAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
