from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import json

import numpy as np
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------------
# 1. New Abstract Base Class with Saving Functionality
# ----------------------------------------------------------------------------------
class SavableDataset(Dataset):
    """An abstract dataset that can be saved to a .bin file."""
    @property
    def has_custom_loss_mask(self) -> bool:
        """Subclasses should override this if they have a complex loss mask to save."""
        return False
        
    def save_to_bin(self, filepath: str):
        """Iterates through the dataset and saves sequences (and optionally masks) to a .bin file."""
        print(f"Saving dataset to {filepath}...")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_mask = self.has_custom_loss_mask

        with open(filepath, 'wb') as f:
            header = np.array([20241220, 2, len(self), int(save_mask)], dtype=np.int32)
            f.write(header.tobytes())
            
            for i in tqdm.tqdm(range(len(self)), desc="Saving to binary file"):
                X, loss_mask = self[i]
                seq = X.numpy().astype(np.int32)
                
                f.write(np.int32(len(seq)).tobytes())
                f.write(seq.tobytes())

                if save_mask:
                    mask = loss_mask.numpy().astype(np.int32)
                    f.write(mask.tobytes())
        
        config_path = filepath.replace('.bin', '_config.json')
        with open(config_path, 'w') as f:
            config = {'max_length': self.max_length, 'pad_token_id': self.tokenizer.pad_token_id, 'tokenizer_path': self.tokenizer.name_or_path}
            json.dump(config, f, indent=2)
        print("Save complete.")


# ----------------------------------------------------------------------------------
# 2. Upgraded MemLoader to Handle Custom Loss Masks
# ----------------------------------------------------------------------------------
class MemLoader:
    def __init__(self, filepath, device="cpu", rank=0, world_size=1):
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self._load_config(filepath)
        with open(filepath, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=4)
            magic_number, _, self.total_sequences, self.has_loss_mask = header
            assert magic_number == 20241220, "Invalid binary file."
            data_offset = f.tell()

        self.data = np.memmap(filepath, dtype=np.int32, mode='r', offset=data_offset)
        
        self._build_index()
        self.local_indices = np.array_split(np.arange(self.total_sequences), self.world_size)[self.rank]

    def __len__(self):
        return len(self.local_indices)

    def _load_config(self, filepath):
        config_path = filepath.replace('.bin', '_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.max_length = config['max_length']
        self.pad_token = config['pad_token_id']
        self.tokenizer_path = config['tokenizer_path']

    def _build_index(self):
        self.index = np.zeros((self.total_sequences, 2), dtype=np.int64)
        current_offset = 0
        for i in range(self.total_sequences):
            seq_len = self.data[current_offset]
            self.index[i, 0] = current_offset + 1  
            self.index[i, 1] = seq_len
            stride = 1 + seq_len
            if self.has_loss_mask:
                stride += seq_len  
            current_offset += stride

    def get_batch(self, batch_size):

        random_start = np.random.randint(0, len(self.local_indices) - batch_size + 1)
        batch_global_indices = self.local_indices[random_start : random_start + batch_size]
        
        batch_indices = self.index[batch_global_indices]

        sequences = [self.data[start:start+length] for start, length in batch_indices]
        max_len = max(len(s) for s in sequences)

        data_batch = torch.full((batch_size, max_len), self.pad_token, dtype=torch.long, device=self.device)
        padded_masks = torch.zeros_like(data_batch)

        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            data_batch[i, :seq_len] = torch.from_numpy(seq.astype(np.int64)).to(self.device)
            
            if self.has_loss_mask:
                mask_start = batch_indices[i, 0] + seq_len
                mask = self.data[mask_start : mask_start + seq_len]
                padded_masks[i, :seq_len] = torch.from_numpy(mask.astype(np.int64)).to(self.device)
            else:
                padded_masks[i, :seq_len] = 1 # Default: mask is valid where data is not padding

        return data_batch, padded_masks


# util functions 
# ----------------------------------------------------------------------------------

def save_tokenized_data(filepath: str, sequences: list, config: dict):
    """
    Saves tokenized sequences to a .bin file and metadata to a _config.json file,
    matching the efficient format from your project's `_save` method.
    """
    print(f"Saving tokenized data to {filepath}...")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        header = np.array([20241220, 1, len(sequences)], dtype=np.int32)
        f.write(header.tobytes())
        
        for seq in tqdm(sequences, desc="Writing to binary file"):
            f.write(np.int32(len(seq)).tobytes())
            f.write(np.array(seq, dtype=np.int32).tobytes())
    
    config_path = filepath.replace('.bin', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"Successfully saved {len(sequences)} sequences to {filepath}")
    print(f"Configuration saved to {config_path}")


# save .bin functional 
# ----------------------------------------------------------------------------------