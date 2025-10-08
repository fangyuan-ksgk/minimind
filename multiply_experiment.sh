# Generate Dataset with different tokenizer

digit_tokenizer_path="tokenizer/digit_tokenizer"
num_digits=3

# (1). Train all tokenizers (run once)
# python -m tokenizer.train_bpe

# --- Experiment Loop ---
tokenizer_paths=("tokenizer/digit_tokenizer" "tokenizer/bpe_tokenizer_v50" "tokenizer/bpe_tokenizer_v100") # Add more as needed

for tokenizer_path in "${tokenizer_paths[@]}"; do
    tokenizer_name=$(basename "$tokenizer_path")
    echo "--- Starting experiments for tokenizer: $tokenizer_name ---"

    # (A). Generate all dataset variants for the current tokenizer
    echo "Generating datasets for $tokenizer_name..."
    python -m dataset.prep_multiply_dataset --tokenizer_path "$tokenizer_path" --num_digits "$num_digits"
    python -m dataset.prep_multiply_dataset --tokenizer_path "$tokenizer_path" --num_digits "$num_digits" --use_cot
    python -m dataset.prep_multiply_dataset --tokenizer_path "$tokenizer_path" --num_digits "$num_digits" --reverse
    python -m dataset.prep_multiply_dataset --tokenizer_path "$tokenizer_path" --num_digits "$num_digits" --reverse --use_cot
    echo "Dataset generation complete for $tokenizer_name."

    # (B). Train models on the generated datasets
    data_formats=("" "_cot" "_reverse" "_reverse_cot")
    model_configs=("2 4" "4 8") # Pairs of "num_hidden_layers num_attention_heads"

    for data_format in "${data_formats[@]}"; do
        train_path="dataset/multiply/multiply_${num_digits}x${num_digits}_train${data_format}.bin"
        val_path="dataset/multiply/multiply_${num_digits}x${num_digits}_val${data_format}.bin"
        
        data_format_name=${data_format:-_naive} 
        data_format_name=${data_format_name:1}

        for config in "${model_configs[@]}"; do
            read -r num_hidden_layers num_attention_heads <<< "$config"
            
            wandb_run_name="${num_digits}x${num_digits}-${tokenizer_name}-${data_format_name}-L${num_hidden_layers}-H${num_attention_heads}"

            echo "--- Starting training run: $wandb_run_name ---"

            python -m trainer.train_sorl_multiply \
                --train_data_path "$train_path" \
                --val_data_path "$val_path" \
                --tokenizer_path "$tokenizer_path" \
                --batch_size 256 \
                --epoch 4 \
                --log_interval 200 \
                --num_hidden_layers "$num_hidden_layers" \
                --num_attention_heads "$num_attention_heads" \
                --wandb_run_name "$wandb_run_name"

            echo "--- Finished run: $wandb_run_name ---"
            echo
        done
    done
    echo "--- All experiments finished for tokenizer: $tokenizer_name ---"
done

echo "--- All experiments complete. ---"