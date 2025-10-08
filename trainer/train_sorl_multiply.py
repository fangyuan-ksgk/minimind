from types import SimpleNamespace
from transformers import AutoTokenizer
import torch
from model.model_minimind import MiniMindConfig
from model.model_sorl import SorlModelWrapper
from dataset.base import MemLoader
from src.sorl import SORLConfig, evaluate, compute_per_token_loss, compute_loss, sorl_search, SearchScheduler, GatedPhaseTransition
from eval_multiply import evaluate_on_loader
import os 
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def initialize_components(args):
    print("--- Initializing components ---")
    # --- Tokenizer and Data ---
    train_loader = MemLoader(args.train_data_path, device=args.device)
    val_loader = MemLoader(args.val_data_path, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(train_loader.tokenizer_path) # data is tokenized
    pad_token_id = tokenizer.pad_token_id

    # --- Model ---
    base_vocab_size = len(tokenizer)
    abstract_vocab_sizes = [int(v) for v in args.abstract_vocab_sizes.split(',')]
    full_vocab_list = [base_vocab_size] + abstract_vocab_sizes

    # 2 layer 4 head
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
    ).to(args.device)

    if torch.cuda.is_available():
        model = torch.compile(model)

    print(f"Model initialized on {args.device} with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # --- SORL Config and Schedulers ---
    sorl_config = SORLConfig(
        n=args.n_rollout, 
        temperature=args.temperature, 
        K=args.K,
        l=1, 
        steps=args.denoise_steps, 
        max_t_search=args.max_t_search,
        use_rhythmic_placeholders=args.use_rhythmic_placeholders,
        use_spike_placeholders=args.use_spike_placeholders,
        use_special_placeholders=args.use_special_placeholders,
        special_token_id=args.special_token_id,
        abstract_budget=args.abstract_budget,
        temperature_flip=args.temperature_flip,
        curriculum_ratio=args.curriculum_ratio,
        use_fade_memory=args.use_fade_memory,
        use_compression_mask=args.use_compression_mask,
        compression_curriculum_ratio=args.compression_curriculum_ratio,
        min_keep=args.memory_span, 
        max_seq_len=train_loader.max_length,
        train_iterations=args.train_iterations, 
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_interval=args.log_interval,
        max_length=train_loader.max_length,
        default_phase=args.default_phase, 
        delta=args.delta, tau=args.tau,
        p_m=args.p_m, p_c=args.p_c
    )
    return train_loader, val_loader, tokenizer, model, sorl_config



def train_sorl_multiply(sorl_config, model, train_loader, val_loader, tokenizer, optimizer, search_scheduler, gapt, wandb):

    optimizer = torch.optim.Adam(model.parameters(), lr=sorl_config.learning_rate)
    search_scheduler = SearchScheduler(sorl_config)
    gapt = GatedPhaseTransition(sorl_config.delta, sorl_config.tau, sorl_config.p_m, sorl_config.p_c)

    model.train()

    for i in range(sorl_config.train_iterations): # Run for 10 steps

        t_search, drop_ratio = search_scheduler.step()
        sorl_config.max_t_search = 0
        model.drop_ratio = 0.0

        data, loss_mask = train_loader.get_batch(sorl_config.train_batch_size)
        with torch.no_grad():
            search_data, switch_ratio = sorl_search(data, loss_mask, model, sorl_config)
            
        ppt = compute_per_token_loss(model, search_data)
        ssl_loss, abs_loss = compute_loss(search_data, model, ppt, loss_mask)
        
        total_loss = ssl_loss + abs_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        wandb.log({
            "train/loss": total_loss.item(),
            "train/ssl_loss": ssl_loss.item(),
            "train/abs_loss": abs_loss.item(),
            "train/switch_ratio": switch_ratio,
        })

        print(f"\rIter {i+1}/{sorl_config.train_iterations} | Loss: {total_loss.item():.4f}", end="")

        if i > 0 and i % sorl_config.log_interval == 0:
            print("Evaluating...")
            greedy_advantage, best_advantage, greedy_info_gain, _, a_loss = evaluate(data, loss_mask, sorl_config, model, search_n=1)
            eval_result = evaluate_on_loader(model, tokenizer, val_loader, batch_size=10, K=None)
            wandb.log({
                "train/greedy_advantage": greedy_advantage,
                "train/best_advantage": best_advantage,
                "train/greedy_info_gain": greedy_info_gain,
                "train/a_loss": a_loss,
                "eval/accuracy": eval_result["accuracy"],
                "eval/top_sim": eval_result["top_sim_score"],
            })

            print(
                f"Iter {i+1} | SSL: {ssl_loss.item():.3f}, Abs: {abs_loss.item():.3f} | "
                f"t_search: {t_search}"
            )
            print(
                f"  ├─ Info-Gain: {greedy_info_gain:.1f}% | Advantage: {greedy_advantage:.1f}%"
            )
            print(f"  └─ Eval Accuracy: {eval_result['accuracy']:.2f}% | Topological Similarity: {eval_result['top_sim_score']:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniMind SORL Training Script")
    
    # --- Paths ---
    parser.add_argument("--train_data_path", type=str, default="dataset/multiply/multiply_2x2_train.bin")
    parser.add_argument("--val_data_path", type=str, default="dataset/multiply/multiply_2x2_val.bin")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/digit_tokenizer")
    
    # --- Model Config ---
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--abstract_vocab_sizes", type=str, default="8")
    
    # --- Training Config ---
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    # --- SORL Config ---
    parser.add_argument("--n_rollout", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--denoise_steps", type=int, default=1)
    parser.add_argument("--max_t_search", type=int, default=0)
    parser.add_argument("--use_rhythmic_placeholders", action="store_true", default=True)
    parser.add_argument("--use_spike_placeholders", action="store_true", default=False)
    parser.add_argument("--use_special_placeholders", action="store_true", default=False)
    parser.add_argument("--special_token_id", type=int, default=31)
    parser.add_argument("--abstract_budget", type=int, default=5)
    parser.add_argument("--temperature_flip", action="store_true", default=False)
    parser.add_argument("--curriculum_ratio", type=float, default=0.6)
    parser.add_argument("--use_fade_memory", action="store_true", default=False)
    parser.add_argument("--use_compression_mask", action="store_true", default=False)
    parser.add_argument("--compression_curriculum_ratio", type=float, default=0.25)
    parser.add_argument("--memory_span", type=int, default=128)
    parser.add_argument("--default_phase", type=int, default=None)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--p_m", type=int, default=10)
    parser.add_argument("--p_c", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--train_iterations", type=int, default=2000)
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SORL-Multiply")
    parser.add_argument("--wandb_run_name", type=str, default="sorl-training-run")

    args = parser.parse_args()

    train_loader, val_loader, tokenizer, model, sorl_config = initialize_components(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=sorl_config.learning_rate)
    search_scheduler = SearchScheduler(sorl_config)
    gapt = GatedPhaseTransition(sorl_config.delta, sorl_config.tau, sorl_config.p_m, sorl_config.p_c)
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    train_sorl_multiply(sorl_config, model, train_loader, val_loader, tokenizer, optimizer, search_scheduler, gapt, wandb)