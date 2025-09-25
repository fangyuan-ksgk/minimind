import os
import sys
import argparse
import warnings
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from contextlib import nullcontext
from transformers import AutoTokenizer

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig
from model.model_sorl import SorlModelWrapper
from dataset.base import MemLoader
from src.sorl import SORLConfig, sorl_search, compute_loss, evaluate, SearchScheduler, compute_per_token_loss

warnings.filterwarnings('ignore')

# --- Globals for DDP ---
DDP = False
RANK = 0
WORLD_SIZE = 1

def logger(content):
    if not DDP or RANK == 0:
        print(content)

def setup_ddp():
    global DDP, RANK, WORLD_SIZE
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        DDP = True
        RANK = int(os.environ["RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(RANK)
        print(f"DDP enabled. Rank: {RANK}, World Size: {WORLD_SIZE}")

def cleanup_ddp():
    if DDP:
        dist.destroy_process_group()

def train(args):
    """Main training and validation loop for SORL."""
    if DDP:
        torch.manual_seed(args.seed + RANK)

    # 1. Initialize Tokenizer and Data Loaders
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    pad_token_id = tokenizer.pad_token_id
    
    train_loader = MemLoader(args.train_data_path, device=f"cuda:{RANK}" if DDP else args.device, rank=RANK, world_size=WORLD_SIZE)
    val_loader = MemLoader(args.val_data_path, device=f"cuda:{RANK}" if DDP else args.device, rank=RANK, world_size=WORLD_SIZE)
    logger("Data loaders initialized.")

    # 2. Initialize Model
    base_vocab_size = tokenizer.vocab_size
    abstract_vocab_sizes = [int(v) for v in args.abstract_vocab_sizes.split(',')]
    full_vocab_list = [base_vocab_size] + abstract_vocab_sizes
    
    minimind_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=sum(full_vocab_list) # Initial vocab size
    )
    
    model = SorlModelWrapper.from_scratch(
        config=minimind_config,
        full_vocab_size_list=full_vocab_list,
        memory_span=args.memory_span,
        pad_token_id=pad_token_id
    ).to(f"cuda:{RANK}" if DDP else args.device)
    
    if DDP:
        model = DistributedDataParallel(model, device_ids=[RANK])
    
    logger(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # 3. Setup SORL Config, Optimizer, and Schedulers
    # This now correctly maps all relevant arguments from the command line
    # to the SORL configuration object.
    sorl_config = SORLConfig(
        n=args.n_rollout,
        temperature=args.temperature,
        K=args.K,
        l=1,  # Target abstraction level is currently fixed at 1
        steps=args.denoise_steps,
        max_t_search=args.max_t_search,
        
        # Placeholders
        use_rhythmic_placeholders=args.use_rhythmic_placeholders,
        use_spike_placeholders=args.use_spike_placeholders,
        abstract_budget=args.abstract_budget,

        # Stability strategies
        temperature_flip=args.temperature_flip,
        
        # Curriculum and Memory
        curriculum_ratio=args.curriculum_ratio,
        use_fade_memory=args.use_fade_memory,
        min_keep=args.memory_span,
        max_seq_len=train_loader.max_length,
        
        # Dataset and Run Config
        train_dataset_path=args.train_data_path,
        val_dataset_path=args.val_data_path,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        train_iterations=args.train_iterations,
        val_iterations=args.val_iterations,
        max_length=train_loader.max_length,

        # Optimization
        learning_rate=args.learning_rate,
        log_interval=args.log_interval
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    search_scheduler = SearchScheduler(sorl_config)

    # 4. Initialize Logging (wandb)
    wandb = None
    if args.use_wandb and RANK == 0:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
    
    # --- Training Loop ---
    model.train()
    for i in range(args.train_iterations):
        global_step = i

        if args.temperature_flip:
            sorl_config.temperature = args.temperature if i % 2 != 0 else 0.0

        t_search = search_scheduler.step()
        sorl_config.max_t_search = t_search
        
        # Get data and perform SORL search
        data, _ = train_loader.get_batch(args.batch_size)
        with torch.no_grad():
            search_data, switch_ratio = sorl_search(data, model.module if DDP else model, sorl_config)
            
        # Compute loss
        ppt = compute_per_token_loss(model.module if DDP else model, search_data)
        ssl_loss, abs_loss = compute_loss(search_data, model.module if DDP else model, ppt)
        total_loss = ssl_loss + abs_loss

        # Optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Logging and Validation
        if global_step % args.log_interval == 0 and RANK == 0:
            logger(f"Iter {global_step}/{args.train_iterations} | Loss: {total_loss.item():.4f} | SSL: {ssl_loss.item():.4f} | Abs: {abs_loss.item():.4f}")
            if wandb:
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/ssl_loss": ssl_loss.item(),
                    "train/abs_loss": abs_loss.item(),
                    "train/switch_ratio": switch_ratio,
                    "progress/t_search": t_search,
                }, step=global_step)
    
    if RANK == 0 and wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SORL Training Script")
    # Paths
    parser.add_argument("--train_data_path", type=str, default="dataset/pretrain_hq.bin")
    parser.add_argument("--val_data_path", type=str, default="dataset/pretrain_hq.bin")
    parser.add_argument("--tokenizer_path", type=str, default="model/")
    # Model Config
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--num_attention_heads', default=8, type=int)
    parser.add_argument("--abstract_vocab_sizes", type=str, default="128", help="Comma-separated list of abstract vocab sizes.")
    # Training Config
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_iterations", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_iterations", type=int, default=10, help="Number of validation batches to run.")
    
    # SORL Config
    parser.add_argument("--n_rollout", type=int, default=4, help="Number of candidates to roll out.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for stochastic rollout.")
    parser.add_argument("--temperature_flip", action="store_true", help="Alternate temperature between 0.0 and the specified value.")
    parser.add_argument("--K", type=int, default=8, help="Rhythmic stride for abstraction.")
    parser.add_argument("--denoise_steps", type=int, default=4, help="Steps for chunk-wise denoising.")
    parser.add_argument("--curriculum_ratio", type=float, default=0.6, help="Ratio of curriculum iterations for t_search.")
    parser.add_argument("--max_t_search", type=int, default=1024, help="Max number of abstract timestamps to search within.")
    
    # Placeholder types
    parser.add_argument("--use_rhythmic_placeholders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_spike_placeholders", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--abstract_budget", type=int, default=5, help="Max number of spiky abstractions allowed.")

    # Memory fading
    parser.add_argument("--memory_span", type=int, default=1024, help="Min # of vivid tokens to keep in memory (used for min_keep).")
    parser.add_argument("--use_fade_memory", action="store_true", help="Enable memory fading during training.")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SORL")
    parser.add_argument("--wandb_run_name", type=str, default="sorl-training-run")
    
    args = parser.parse_args()
    
    setup_ddp()
    train(args)
    cleanup_ddp()