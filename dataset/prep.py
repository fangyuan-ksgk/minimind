from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset, SFTDataset

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('model/')

# save pre-train dataset into .bin file
dataset = PretrainDataset(data_path='dataset/pretrain_hq.jsonl', tokenizer=tokenizer, max_length=256)
dataset.save_to_bin('dataset/pretrain_hq.bin')

# save sft dataset into .bin file
dataset = SFTDataset(data_path="dataset/sft_1024.jsonl", tokenizer=tokenizer, max_length=1024)
dataset.save_to_bin('dataset/sft_1024.bin')