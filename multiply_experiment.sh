# Generate Dataset with different tokenizer

digit_tokenizer_path="tokenizer/digit_tokenizer"
num_digits=3

# (1). Generate dataset
python dataset/prep_multiply_dataset.py --tokenizer_path $digit_tokenizer_path --num_digits $num_digits 

# (2). Train model (TBD)