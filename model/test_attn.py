import torch
from src.stat import AttentionMaskVisualizer, save_gif
from model.model_sorl import infer_level
import numpy as np 
from model.model_sorl import compute_attn_mask


# --- Interactive Experiment ---

# Configuration
VOCAB_SIZES = torch.tensor([100, 50]) # Base vocab: 0-99, Abstract vocab: 100-149
MEMORY_SPAN = 256
DEVICE = "cpu"

# --- Toy Example 3: Multiple Abstraction Tokens ---
# Sequence: T A T A T T T T
input_ids_3 = torch.tensor([[10, 20, 30, 45, 125, 10, 20, 30, 45, 125, 10, 20, 30, 45, 125, 10, 20, 30, 45, 125, 10, 20, 30, 45, 125, 10, 20, 30, 45, 125, 10, 20, 30, 45, 125, 10, 20, 30, 45, 125, 10, 20, 30, 45, 125]], device=DEVICE)

frames = [] 
for drop_ratio in np.linspace(0.0, 1.0, 11):
    mask3 = compute_attn_mask(input_ids_3, VOCAB_SIZES, MEMORY_SPAN, drop_ratio=drop_ratio)
    levels3 = infer_level(input_ids_3, VOCAB_SIZES, -1)
    visualizer3 = AttentionMaskVisualizer(levels3, mask3)
    s = "".join([["T", "A"][i] for i in levels3[0].int().tolist()])
    img = visualizer3.plot(f"Drop Ratio: {drop_ratio:.1f} || Attention Mask")

    frames.append(img)

for _ in range(5):
    frames.append(img)

# create GIF
save_gif(frames, "attention_mask_gif.gif", fps=1)