# SoRL & NIL training pipeline
# -------------------------------------------------------------
# (1). data loading (get_batch)
# (2). curriculum on t_search, t_keep
# (3). evaluate search improvement / ppl evaluation gadget
# -------------------------------------------------------------


import wandb, torch
from dataset.base import BaseDataset
from model.model_sorl import SorlModelWrapper
from src.sorl import SORLConfig, sorl_search, compute_loss, evaluate, SearchScheduler, compute_per_token_loss
from dataset.base import get_batch


# Validation loop
# ------------------------------------------------------------------------------------------------
def validate(val_dataset: BaseDataset, model: SorlModelWrapper,  config: SORLConfig): 

    total_improve_ppl = 0.0
    total_traj_ppl = 0.0

    num_iterations = max(1, config.val_iterations)
    for _ in range(num_iterations): 
        val_data = get_batch(val_dataset, config.val_batch_size, config.max_length, model.level_mask_tokens[0], device=model.model.device)
        with torch.no_grad(): 
            improve_ppl, traj_ppl, greedy_str, vocab_utilization_rate = evaluate(val_data, model, 5, config)
            total_improve_ppl += improve_ppl
            total_traj_ppl += traj_ppl
        del val_data

    avg_improve_ppl = total_improve_ppl / num_iterations
    avg_traj_ppl = total_traj_ppl / num_iterations
    return avg_improve_ppl, avg_traj_ppl, greedy_str, vocab_utilization_rate


# Self organizing reinforcement learning (SoRL)
# ------------------------------------------------------------------------------------------------
def self_organizing_reinforcement_learning(train_dataset: BaseDataset, val_dataset: BaseDataset, model: SorlModelWrapper, config: SORLConfig, start_step: int = 0): 
    
    optimizer = torch.optim.Adam(model.model.parameters(), lr=config.learning_rate)
    scheduler = SearchScheduler(config)

    for i in range(config.train_iterations):
        # config.temperature = 0.0 if i % 2 == 0 else 1.0
        global_step = start_step + i
        model.model.train() 

        t_search = scheduler.step()
        config.max_t_search = t_search
        model.memory_span = scheduler.memory_span # memory fading

        data = get_batch(train_dataset, config.train_batch_size, config.max_length, model.level_mask_tokens[0], device=model.model.device)

        with torch.no_grad(): 

            search_data, switch_ratio = sorl_search(data, model, config)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        ppt = compute_per_token_loss(model, search_data)

        ssl_loss, abs_loss = compute_loss(search_data, model, ppt)
        loss = abs_loss + ssl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if global_step % config.log_interval == 0 and global_step > 0:
            
            # Validation needs to be more rigorous : more samples
            model.model.eval()

            with torch.no_grad(): 
                _, improve_ppl_train, _, _ = evaluate(data, model, 5, config)
                validate_improve_ppl, validate_traj_ppl, validate_greedy_str, validate_vocab_utilization_rate = validate(val_dataset, model, config)

            
            wandb.log({
                f"train/loss": loss.item(), 
                f"train/ssl_loss": ssl_loss.item(), 
                f"train/abs_loss": abs_loss.item(),

                f"train/improve_ppl_percentage": improve_ppl_train.item(), 
                f"train/abstraction_switch_ratio": switch_ratio, # how often greedy sampled abstraction is rejected for other abstraction
                f"val(in-domain)/improve_ppl_percentage": validate_improve_ppl.item(), 
                f"val(in-domain)/traj_ppl": validate_traj_ppl.item(), 
                f"val(in-domain)/vocab_utilization_rate": validate_vocab_utilization_rate, 

                f"progress/iteration": global_step, 
                f"progress/t_search": t_search, 
            }, step=global_step)

            model.model.train()

        print(f"Iteration {i+1}/{config.train_iterations} "
                            f"- loss: {loss.item():.4f}, abs_loss: {abs_loss.item():.4f}, ssl_loss: {ssl_loss.item():.4f}, t_search: {t_search}, memory_span: {model.memory_span}")


        del loss, abs_loss, ssl_loss, ppt

    return model, start_step + config.train_iterations