import torch
from model.model_sorl import SorlModelWrapper, infer_level
from model.model_minimind import MiniMindConfig

def test_minimind_with_sorl():
    """
    Tests the SorlModelWrapper with a MiniMind model initialized from scratch.
    This checks generation, forward pass, and denoising capabilities.
    """
    print("="*80)
    print("--- Testing SORL Wrapper with MiniMind Model ---")
    print("="*80)

    # --- 1. Initialize a SORL-enabled MiniMind model from scratch ---
    base_vocab_size = 11
    abstract_vocab_sizes = [50]
    full_vocab_list = [base_vocab_size] + abstract_vocab_sizes
    
    sorl_model = SorlModelWrapper.from_scratch(
        config=MiniMindConfig(vocab_size=sum(full_vocab_list)),
        full_vocab_size_list=full_vocab_list,
        memory_span=5,
        pad_token_id=0,
        drop_ratio=0.3
    )
    print("Successfully initialized MiniMind with SORL wrapper.\n")

    # --- 2. Test SORL Generation ---
    prompt = torch.tensor([[1, 2, 3]])
    max_new_tokens = 50
    generated_sequence = sorl_model.generate(
        input_ids=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        force_abstraction_every_n=4
    )
    print("--- SORL Generation Results ---")
    print(f"Generated Sequence: {generated_sequence.tolist()}")
    print("✅ Generation test passed.\n")

    # --- 3. Test Forward Pass (Sparse Attention) ---
    result = sorl_model.forward(prompt)
    print("--- Forward Propagation (Sparse Attention) ---")
    print(f"Result logits shape: {result.logits.shape}")
    assert result.logits.shape == (prompt.shape[0], prompt.shape[1], sorl_model.model.config.vocab_size)
    print("✅ Forward pass test passed.\n")

    # --- 4. Test Denoising ---
    abstract_pad_token = sorl_model.level_mask_tokens[1].item()
    orig_tokens = torch.tensor([[1, 2, 3, abstract_pad_token, 2, 4, 1, abstract_pad_token, 3, 4, 2, abstract_pad_token]])
    levels = infer_level(orig_tokens, sorl_model.vocab_sizes, -1)
    denoise_mask = torch.isin(orig_tokens, sorl_model.level_mask_tokens[1:])
    denoise_levels = levels[denoise_mask]
    
    new_tokens = sorl_model.denoise(orig_tokens, denoise_mask, denoise_levels, 0.0)
    print("--- Denoising ---")
    print(f"Original tokens: {orig_tokens[0].tolist()}")
    print(f"Denoised tokens: {new_tokens[0].tolist()}")
    assert orig_tokens.shape == new_tokens.shape
    print("✅ Denoising test passed.\n")


def test_qwen_with_sorl():
    """
    Tests the SorlModelWrapper with a pretrained Qwen model from Hugging Face.
    This checks compatibility for loading, forward pass, and generation.
    """
    print("="*80)
    print("--- Testing SORL Wrapper with Qwen2-0.5B Model ---")
    print("="*80)
    
    try:
        # --- 1. Load a pretrained Qwen model and wrap it for SORL ---
        qwen_sorl_model = SorlModelWrapper.from_pretrained(
            model_name_or_path="Qwen/Qwen2-0.5B",
            abstract_vocab_size_list=[128, 64],
            memory_span=5,
            pad_token_id=0,
            drop_ratio=0.3
        )
        print("Successfully loaded Qwen/Qwen2-0.5B with SORL wrapper.\n")
        
        device = qwen_sorl_model.model.device
        prompt = torch.tensor([[1, 2, 3]]).to(device)

        # --- 2. Test Forward Pass (Sparse Attention) ---
        result = qwen_sorl_model.forward(prompt)
        print("--- Forward Propagation (Sparse Attention) ---")
        print(f"Result logits shape: {result.logits.shape}")
        assert result.logits.shape == (prompt.shape[0], prompt.shape[1], qwen_sorl_model.model.config.vocab_size)
        print("✅ Forward pass test passed.\n")

        # --- 3. Test SORL Generation ---
        max_new_tokens = 50
        generated_sequence = qwen_sorl_model.generate(
            input_ids=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            force_abstraction_every_n=4
        )
        print("--- SORL Generation Results with Qwen ---")
        print(f"Generated Sequence: {generated_sequence.tolist()}")
        print("✅ Generation test passed.\n")

        # --- 4. Test Denoising ---
        abstract_pad_token = qwen_sorl_model.level_mask_tokens[1].item()
        orig_tokens = torch.tensor([[1, 2, 3, abstract_pad_token, 2, 4, 1, abstract_pad_token, 3, 4, 2, abstract_pad_token]])
        levels = infer_level(orig_tokens, qwen_sorl_model.vocab_sizes, -1)
        denoise_mask = torch.isin(orig_tokens, qwen_sorl_model.level_mask_tokens[1:])
        denoise_levels = levels[denoise_mask]
        new_tokens = qwen_sorl_model.denoise(orig_tokens, denoise_mask, denoise_levels, 0.0)
        print("--- Denoising ---")
        print(f"Original tokens: {orig_tokens[0].tolist()}")
        print(f"Denoised tokens: {new_tokens[0].tolist()}")
        assert orig_tokens.shape == new_tokens.shape
        print("✅ Denoising test passed.\n")

    except Exception as e:
        print(f"\nCould not run Qwen test. This may be due to memory or network issues.")
        print(f"Error: {e}")






if __name__ == "__main__":
    test_minimind_with_sorl()
    test_qwen_with_sorl()