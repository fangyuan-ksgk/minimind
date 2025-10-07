from dataset.multiply_dataset import MultiplyDataset
from transformers import AutoTokenizer

def main(args):
    # --- Configuration ---
    TOKENIZER_PATH = args.tokenizer_path
    TRAIN_SAMPLES = 80800
    VAL_SAMPLES = 1200
    MAX_LENGTH = 128
    NUM_DIGITS = args.num_digits
    TRAIN_OUTPUT_FILE = f"dataset/multiply/multiply_{NUM_DIGITS}x{NUM_DIGITS}_train.bin"
    VAL_OUTPUT_FILE = f"dataset/multiply/multiply_{NUM_DIGITS}x{NUM_DIGITS}_val.bin"

    # --- Initialization ---
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # --- Create and Save Training Dataset ---
    train_dataset = MultiplyDataset(
        tokenizer=tokenizer,
        num_samples=TRAIN_SAMPLES,
        max_length=MAX_LENGTH,
        min_digits=NUM_DIGITS,
        max_digits=NUM_DIGITS,
        use_cot=False,
        reverse_digits=False
    )
    
    train_dataset.save_to_bin(TRAIN_OUTPUT_FILE)
    print(f"\nSuccessfully created multiplication training dataset at: {TRAIN_OUTPUT_FILE}")

    # --- Create and Save Validation Dataset ---
    print("\n--- Generating validation set ---")
    val_dataset = MultiplyDataset(
        tokenizer=tokenizer,
        num_samples=VAL_SAMPLES,
        max_length=MAX_LENGTH,
        min_digits=NUM_DIGITS,
        max_digits=NUM_DIGITS,
        use_cot=False,
        reverse_digits=False
    )
    
    val_dataset.save_to_bin(VAL_OUTPUT_FILE)
    print(f"\nSuccessfully created multiplication validation dataset at: {VAL_OUTPUT_FILE}")

    # --- CoT included ----
    TRAIN_OUTPUT_FILE = f"dataset/multiply/multiply_{NUM_DIGITS}x{NUM_DIGITS}_train_cot.bin"
    VAL_OUTPUT_FILE = f"dataset/multiply/multiply_{NUM_DIGITS}x{NUM_DIGITS}_val_cot.bin"

    train_dataset = MultiplyDataset(
        tokenizer=tokenizer,
        num_samples=TRAIN_SAMPLES,
        max_length=MAX_LENGTH,
        min_digits=NUM_DIGITS,
        max_digits=NUM_DIGITS,
        use_cot=True,
        reverse_digits=False
    )
    
    train_dataset.save_to_bin(TRAIN_OUTPUT_FILE)
    print(f"\nSuccessfully created multiplication training dataset at: {TRAIN_OUTPUT_FILE}")

    print("\n--- Generating validation set ---")
    val_dataset = MultiplyDataset(
        tokenizer=tokenizer,
        num_samples=VAL_SAMPLES,
        max_length=MAX_LENGTH,
        min_digits=NUM_DIGITS,
        max_digits=NUM_DIGITS,
        use_cot=True,
        reverse_digits=False
    )
    
    val_dataset.save_to_bin(VAL_OUTPUT_FILE)
    print(f"\nSuccessfully created multiplication validation dataset at: {VAL_OUTPUT_FILE}")

    # --- reverse digits included ----
    TRAIN_OUTPUT_FILE = f"dataset/multiply/multiply_{NUM_DIGITS}x{NUM_DIGITS}_train_reverse.bin"
    VAL_OUTPUT_FILE = f"dataset/multiply/multiply_{NUM_DIGITS}x{NUM_DIGITS}_val_reverse.bin"

    train_dataset = MultiplyDataset(
        tokenizer=tokenizer,
        num_samples=TRAIN_SAMPLES,
        max_length=MAX_LENGTH,
        min_digits=NUM_DIGITS,
        max_digits=NUM_DIGITS,
        use_cot=False,
        reverse_digits=True
    )
    
    train_dataset.save_to_bin(TRAIN_OUTPUT_FILE)
    print(f"\nSuccessfully created multiplication training dataset at: {TRAIN_OUTPUT_FILE}")

    print("\n--- Generating validation set ---")
    val_dataset = MultiplyDataset(
        tokenizer=tokenizer,
        num_samples=VAL_SAMPLES,
        max_length=MAX_LENGTH,
        min_digits=NUM_DIGITS,
        max_digits=NUM_DIGITS,
        use_cot=False, 
        reverse_digits=True
    )
    
    val_dataset.save_to_bin(VAL_OUTPUT_FILE)
    print(f"\nSuccessfully created multiplication validation dataset at: {VAL_OUTPUT_FILE}")


    # --- cot included & reverse digits included ----
    TRAIN_OUTPUT_FILE = f"dataset/multiply/multiply_{NUM_DIGITS}x{NUM_DIGITS}_train_reverse_cot.bin"
    VAL_OUTPUT_FILE = f"dataset/multiply/multiply_{NUM_DIGITS}x{NUM_DIGITS}_val_reverse_cot.bin"

    train_dataset = MultiplyDataset(
        tokenizer=tokenizer,
        num_samples=TRAIN_SAMPLES,
        max_length=MAX_LENGTH,
        min_digits=NUM_DIGITS,
        max_digits=NUM_DIGITS,
        use_cot=True,
        reverse_digits=True
    )
    
    train_dataset.save_to_bin(TRAIN_OUTPUT_FILE)
    print(f"\nSuccessfully created multiplication training dataset at: {TRAIN_OUTPUT_FILE}")

    print("\n--- Generating validation set ---")
    val_dataset = MultiplyDataset(
        tokenizer=tokenizer,
        num_samples=VAL_SAMPLES,
        max_length=MAX_LENGTH,
        min_digits=NUM_DIGITS,
        max_digits=NUM_DIGITS,
        use_cot=True,
        reverse_digits=True
    )
    
    val_dataset.save_to_bin(VAL_OUTPUT_FILE)
    print(f"\nSuccessfully created multiplication validation dataset at: {VAL_OUTPUT_FILE}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniMind Multiply Dataset")
    parser.add_argument("--tokenizer_path", type=str, default="model/")
    parser.add_argument("--num_digits", type=int, default=2)
    args = parser.parse_args()
    main(args)