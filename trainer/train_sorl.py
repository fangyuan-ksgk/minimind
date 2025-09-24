import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ToBeFixed ...

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from model.model_gat import GATConfig, GATForCausalLM
from dataset.base import BaseDataset
from src.sorl import SearchScheduler, sorl_search, compute_loss, validate

warnings.filterwarnings('ignore')

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

def train_epoch(epoch, wandb, scheduler):
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        data = batch.to(args.device)
        
        # SORL specific logic
        t_search = scheduler.step()
        model.module.config.memory_span = scheduler.memory_span  # Update memory span

        with torch.no_grad():
            search_data, switch_ratio = sorl_search(data, model, t_search)
        
        outputs = model(input_ids=search_data[:, :-1].contiguous(), labels=search_data[:, 1:].contiguous())
        
        ssl_loss, abs_loss = compute_loss(search_data, model, outputs.loss)
        loss = abs_loss + ssl_loss
        loss = loss / args.accumulation_steps
        
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 and step > 0:
            spend_time = time.time() - start_time
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iter_per_epoch}) loss:{loss.item() * args.accumulation_steps:.4f} '
                f'abs_loss: {abs_loss.item():.4f}, ssl_loss: {ssl_loss.item():.4f}, t_search: {t_search}, '
                f'memory_span: {model.module.config.memory_span}'
            )
            
            if wandb is not None and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "train/loss": loss.item() * args.accumulation_steps,
                    "train/ssl_loss": ssl_loss.item(),
                    "train/abs_loss": abs_loss.item(),
                    "train/abstraction_switch_ratio": switch_ratio,
                    "progress/iteration": epoch * iter_per_epoch + step,
                    "progress/t_search": t_search,
                })
        
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            ckp_path = f'{args.save_dir}/sorl_gat.pth'
            torch.save(model.module.state_dict(), ckp_path)
            model.train()

def init_model(lm_config):
    model = GATForCausalLM(lm_config).to(args.device)
    Logger(f'Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SORL Training with GAT")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_--dargument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SORL")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument("--data_path", type=str, default="../dataset/sorl_data.bin")
    args = parser.parse_args()

    lm_config = GATConfig() # Using default GAT config
    os.makedirs(args.out_dir, exist_ok=True)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name="SORL-GAT-Training")
    else:
        wandb = None

    model = init_model(lm_config)
    
    # Data Loading
    train_ds = BaseDataset.from_file(args.data_path)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    
    # SORL Search Scheduler
    scheduler = SearchScheduler(K=lm_config.K, n_epoch=args.epochs, n_step_per_epoch=len(train_loader))

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb, scheduler)
