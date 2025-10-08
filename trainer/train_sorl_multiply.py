# Multiplication experiment with SoRL
import os
import sys
import argparse
import warnings
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer

# Import the unified evaluation function
from eval_multiply import evaluate_on_loader

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig
from model.model_sorl import SorlModelWrapper
from dataset.base import MemLoader
from src.sorl import SORLConfig, sorl_search, compute_loss, SearchScheduler, compute_per_token_loss, GatedPhaseTransition, evaluate

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        logger(f"DDP enabled. Rank: {RANK}, World Size: {WORLD_SIZE}")

def cleanup_ddp():
    if DDP:
        dist.destroy_process_group()

def train(args):
    """Main training and validation loop for SORL on multiplication task."""
    if DDP:
        torch.manual_seed(args.seed + RANK)

    # 1. Initialize Tokenizer and Data Loaders
    device = f"cuda:{RANK}" if DDP else args.device
    train_loader = MemLoader(args.train_data_path, device=device, rank=RANK, world_size=WORLD_SIZE)
    val_loader = MemLoader(args.val_data_path, device=device, rank=RANK, world_size=WORLD_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(train_loader.tokenizer_path)
    pad_token_id = tokenizer.pad_token_id
    logger("Data loaders initialized.")

    # 2. Initialize Model
    base_vocab_size = len(tokenizer)
    abstract_vocab_sizes = [int(v) for v in args.abstract_vocab_sizes.split(',')]
    full_vocab_list = [base_vocab_size] + abstract_vocab_sizes
    
    minimind_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=sum(full_vocab_list)
    )
    
    model = SorlModelWrapper.from_scratch(
        config=minimind_config,
        full_vocab_size_list=full_vocab_list,
        memory_span=args.memory_span,
        pad_token_id=pad_token_id
    ).to(device)
    
    if DDP:
        model = DistributedDataParallel(model, device_ids=[RANK])
    
    logger(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # 3. Setup SORL Config, Optimizer, and Schedulers
    sorl_config = SORLConfig(
        n=args.n_rollout, temperature=args.temperature, K=args.K, l=1,
        steps=args.denoise_steps, max_t_search=args.max_t_search,
        use_rhythmic_placeholders=args.use_rhythmic_placeholders,
        use_spike_placeholders=args.use_spike_placeholders,
        abstract_budget=args.abstract_budget,
        temperature_flip=args.temperature_flip,
        curriculum_ratio=args.curriculum_ratio,
        use_fade_memory=args.use_fade_memory,
        use_compression_mask=args.use_compression_mask,
        min_keep=args.memory_span, max_seq_len=train_loader.max_length,
        train_iterations=args.train_iterations,
        train_batch_size=args.batch_size, val_batch_size=args.batch_size,
        max_length=train_loader.max_length,
        default_phase=args.default_phase, delta=args.delta, tau=args.tau,
        p_m=args.p_m, p_c=args.p_c
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    search_scheduler = SearchScheduler(sorl_config)
    gapt = GatedPhaseTransition(sorl_config.delta, sorl_config.tau, sorl_config.p_m, sorl_config.p_c)

    # 4. Initialize Logging (wandb)
    wandb = None
    if args.use_wandb and RANK == 0:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
    
    # --- Training Loop ---
    logger("\n--- Starting training loop ---")
    model.train()
    for i in range(args.train_iterations):
        global_step = i

        t_search, drop_ratio = search_scheduler.step()
        sorl_config.max_t_search = t_search
        model_instance = model.module if DDP else model
        model_instance.drop_ratio = drop_ratio

        data, loss_mask = train_loader.get_batch(args.batch_size)
        with torch.no_grad():
            search_data, switch_ratio = sorl_search(data, loss_mask, model_instance, sorl_config)
            
        ppt = compute_per_token_loss(model_instance, search_data)
        ssl_loss, abs_loss = compute_loss(search_data, model_instance, ppt, loss_mask)
        
        current_phase = gapt.step(ssl_loss, abs_loss)
        if sorl_config.default_phase is not None:
            current_phase = sorl_config.default_phase
             
        total_loss = ssl_loss + abs_loss if current_phase == 2 else ssl_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if RANK == 0:
            print(f"\rIter {global_step+1}/{args.train_iterations} | Loss: {total_loss.item():.4f}", end="")

        if (global_step + 1) % args.log_interval == 0 and RANK == 0:
            print() # Newline after the progress indicator
            
            with torch.no_grad():
                greedy_advantage, _, greedy_info_gain, _, _ = evaluate(data, loss_mask, sorl_config, model_instance, search_n=1)

                model_instance.eval()
                eval_results = evaluate_on_loader(
                    model=model_instance, 
                    tokenizer=tokenizer, 
                    val_loader=val_loader, 
                    num_samples_to_eval=args.num_eval_samples, 
                    batch_size=args.batch_size, 
                    K=args.K
                )
                accuracy = eval_results.get('accuracy', 0)
                top_sim = eval_results.get('top_sim_score', 0)
                model_instance.train() # Switch back to train mode

            logger(
                f"Iter {global_step+1} | SSL: {ssl_loss.item():.3f}, Abs: {abs_loss.item():.3f} | "
                f"Phase: {current_phase} | t_search: {t_search}"
            )
            logger(
                f"  ├─ Info-Gain: {greedy_info_gain:.1f}% | Advantage: {greedy_advantage:.1f}%"
            )
            logger(f"  └─ Eval Accuracy: {accuracy:.2f}% | Topological Similarity: {top_sim:.2f}")


            if wandb:
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/ssl_loss": ssl_loss.item(),
                    "train/abs_loss": abs_loss.item(),
                    "train/switch_ratio": switch_ratio,
                    "progress/t_search": t_search,
                    "progress/phase": current_phase,
                    "eval/accuracy": accuracy,
                    "eval/top_sim_score": top_sim,
                    "eval/info_gain": greedy_info_gain,
                    "eval/advantage": greedy_advantage
                }, step=global_step)
    
    if RANK == 0 and wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SORL Training Script for Multiplication")
    # Paths
    parser.add_argument("--train_data_path", type=str, default="dataset/multiply/multiply_2x2_train.bin")
    parser.add_argument("--val_data_path", type=str, default="dataset/multiply/multiply_2x2_val.bin")
    
    # Model Config
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=4, type=int)
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument("--abstract_vocab_sizes", type=str, default="8", help="Comma-separated abstract vocab sizes.")
    
    # Training Config
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    # SORL Config
    parser.add_argument("--n_rollout", type=int, default=5, help="Number of candidates to roll out.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for stochastic rollout.")
    parser.add_argument("--temperature_flip", action="store_true", help="Alternate temperature.")
    parser.add_argument("--K", type=int, default=4, help="Rhythmic stride for abstraction.")
    parser.add_argument("--denoise_steps", type=int, default=1, help="Steps for chunk-wise denoising.")
    parser.add_argument("--curriculum_ratio", type=float, default=0.6, help="Ratio of curriculum iterations for t_search.")
    parser.add_argument("--max_t_search", type=int, default=0, help="Max abstract timestamps to search.")
    
    # Placeholder types
    parser.add_argument("--use_rhythmic_placeholders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_spike_placeholders", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--abstract_budget", type=int, default=5, help="Max spiky abstractions allowed.")

    # Memory fading
    parser.add_argument("--memory_span", type=int, default=128, help="Min # of vivid tokens to keep.")
    parser.add_argument("--use_fade_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_compression_mask", action=argparse.BooleanOptionalAction, default=False)
    
    # GAPT
    parser.add_argument("--default_phase", type=int, default=None, help="Default phase for GAPT (1 or 2).")
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--p_m", type=int, default=10)
    parser.add_argument("--p_c", type=int, default=10)

    # Logging & Evaluation
    parser.add_argument("--log_interval", type=int, default=100, help="Interval for detailed logging and evaluation.")
    parser.add_argument("--num_eval_samples", type=int, default=200, help="Number of samples for periodic evaluation.")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SORL-Multiply")
    parser.add_argument("--wandb_run_name", type=str, default="sorl-multiply-run")
    
    args = parser.parse_args()
    
    setup_ddp()
    train(args)
    cleanup_ddp()