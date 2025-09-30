import torch
import matplotlib.pyplot as plt
import seaborn as sns
from model.model_sorl import infer_level
from io import BytesIO
from PIL import Image
import numpy as np 
from pathlib import Path
from model.model_sorl import compute_attn_mask


# --- Visualization Class ---
class AttentionMaskVisualizer:
    def __init__(self, levels, mask):
        self.levels = levels.squeeze(0).cpu().numpy()
        self.mask = mask.squeeze(0).cpu().numpy()
        self.seq_len = self.mask.shape[0]
        self.labels = [("A" if level > 0 else "T") for level in self.levels]

    def plot(self, title="Attention Mask") -> Image.Image:
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(self.mask.astype(int), cmap="Oranges", linewidths=0.5, cbar=False, annot=False, ax=ax)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.set_xticks(torch.arange(self.seq_len) + 0.5)
        ax.set_yticks(torch.arange(self.seq_len) + 0.5)
        ax.set_xticklabels(self.labels, rotation=90)
        ax.set_yticklabels(self.labels, rotation=0)
        ax.set_xlabel("Key (Memory)")
        ax.set_ylabel("Query (Current Token)")
        ax.set_title(title)

        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)

        buf.seek(0)
        return Image.open(buf)


def save_gif(frames, path, fps=6):
    """frames: list[Image.Image] -> gif at `path`"""
    if not frames:
        raise ValueError("No frames provided.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = int(1000 / fps)
    base = frames[0].convert("P", palette=Image.ADAPTIVE)
    tail = [im.convert("P", palette=Image.ADAPTIVE) for im in frames[1:]]

    base.save(
        path,
        format="GIF",
        save_all=True,
        append_images=tail,
        loop=0,
        duration=duration_ms,
        disposal=2,
    )


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