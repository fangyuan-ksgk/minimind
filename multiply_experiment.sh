# Generate Dataset with different tokenizer

digit_tokenizer_path="tokenizer/digit_tokenizer"
num_digits=3

# (1). Train tokenizer
# python -m tokenizer.train_bpe

# (1). Generate dataset
# python -m dataset.prep_multiply_dataset --tokenizer_path $digit_tokenizer_path --num_digits $num_digits 

# (2). Train model
# (a). no reverse, no CoT
train_path="dataset/multiply/multiply_3x3_train.bin"
val_path="dataset/multiply/multiply_3x3_val.bin"

python -m trainer.train_sorl_multiply --use_wandb --train_data_path $train_path --val_data_path $val_path

# (b). reverse, no CoT
train_path="dataset/multiply/multiply_3x3_train_reverse.bin"
val_path="dataset/multiply/multiply_3x3_val_reverse.bin"

python -m trainer.train_sorl_multiply --use_wandb --train_data_path $train_path --val_data_path $val_path