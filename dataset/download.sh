# download data 
hf download --repo-type dataset jingyaogong/minimind_dataset pretrain_hq.jsonl --local-dir dataset/
hf download --repo-type dataset jingyaogong/minimind_dataset sft_1024.jsonl --local-dir dataset/

# prep .bin files
python -m dataset.prep