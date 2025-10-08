import torch
import tqdm
from scipy.stats import pearsonr
import numpy as np
import torch.nn.functional as F

try:
    import Levenshtein
except ImportError:
    print("Warning: Levenshtein library not found. Please run: pip install python-Levenshtein")
    Levenshtein = None

# --- Utility Functions ---

def _extract_answer_from_ids(token_ids, tokenizer) -> str:
    """Decodes token IDs and extracts the answer before the first EOS token."""
    full_str = tokenizer.decode(token_ids, skip_special_tokens=False)
    before_eos = full_str.split('<eos>')[0].strip()
    if '<answer>' in before_eos:
        parts = before_eos.split('<answer>')
        if len(parts) > 1:
            answer_str = parts[1].strip()
        else:
            answer_str = ""
    else:
        answer_str = ""
    return answer_str

def _get_query_and_gt_ids(input_ids, equal_token_id, answer_token_id):
    """
    Splits the input_ids into query and ground truth based on the first occurrence of 
    equal_token_id or answer_token_id.
    """
    try:
        # Find the first index of either token
        equal_indices = (input_ids == equal_token_id).nonzero(as_tuple=True)[1]
        answer_indices = (input_ids == answer_token_id).nonzero(as_tuple=True)[1]
        
        # Use the first token that appears
        split_idx = -1
        if len(equal_indices) > 0 and len(answer_indices) > 0:
            split_idx = min(equal_indices[0], answer_indices[0])
        elif len(equal_indices) > 0:
            split_idx = equal_indices[0]
        elif len(answer_indices) > 0:
            split_idx = answer_indices[0]
        else:
            raise ValueError("Neither '=' nor '<answer>' token found in input_ids")

        query_ids = input_ids[:, :split_idx + 1]
        ground_truth_ids = input_ids[:, split_idx + 1:]
        
        return query_ids, ground_truth_ids

    except IndexError:
        # This handles cases where tokens are not found
        raise ValueError("Error splitting input_ids: make sure input contains '=' or '<answer>' token.")


def _str_to_int(s: str):
    """Safely converts a string to an integer, handling spaces and errors."""
    try:
        return int("".join(s.split()))
    except (ValueError, TypeError):
        return None



def compute_topological_similarity(model, tokenizer, string_list):
    """
    Computes the correlation between string edit-distance and embedding cosine similarity.
    A strong negative correlation means textually similar numbers have similar embeddings.
    """

    model.eval()
    num_strings = len(string_list)
    embedding_dim = model.config.hidden_size

    embeddings = torch.zeros((num_strings, embedding_dim), device=model.device)

    with torch.no_grad():
        for i, s in enumerate(string_list):
            input_ids = tokenizer.encode(s, add_special_tokens=False, return_tensors='pt').to(model.device)
            if input_ids.nelement() == 0:
                continue

            outputs = model(input_ids)
            last_hidden_state = outputs.last_hidden_state
            
            embeddings[i] = last_hidden_state[:, -1, :] # last token rep 
            # embeddings[i] = last_hidden_state.mean(dim=1) # avg token rep

    cos_sim_matrix = F.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
    )

    edit_dist_matrix = np.zeros((num_strings, num_strings))
    for i in range(num_strings):
        for j in range(num_strings):
            edit_dist_matrix[i, j] = Levenshtein.distance(string_list[i], string_list[j])

    triu_indices = np.triu_indices(num_strings, k=1)
    cos_sim_vector = cos_sim_matrix.cpu().numpy()[triu_indices]
    edit_dist_vector = edit_dist_matrix[triu_indices]

    correlation, p_value = pearsonr(cos_sim_vector, edit_dist_vector)
    return -correlation


# --- Main Evaluation Functions ---

def evaluate_on_loader(model, tokenizer, val_loader, num_samples_to_eval=200, batch_size=100, K=None):
    """
    Evaluates the model on a data loader, calculating accuracy based on exact match.
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    log_file = open("eval_log.txt", "w")
    
    answer_token_id = tokenizer.encode('<answer>', add_special_tokens=False)[0]
    equal_token_id = tokenizer.encode('=', add_special_tokens=False)[0]

    num_batches = (num_samples_to_eval + batch_size - 1) // batch_size
    top_sim_score = []
    pbar = tqdm.tqdm(range(num_batches), desc="Evaluating Batches")
    
    for _ in pbar:
        batch_input_ids, _ = val_loader.get_batch(batch_size)

        prompts = [tokenizer.decode(id) for id in batch_input_ids]
        sim_score = compute_topological_similarity(model, tokenizer, prompts)
        top_sim_score.append(sim_score)
        
        for i in range(batch_input_ids.size(0)):
            if total_samples >= num_samples_to_eval:
                break

            input_ids = batch_input_ids[i:i+1]

            query_ids, ground_truth_ids = _get_query_and_gt_ids(input_ids, equal_token_id, answer_token_id)

            with torch.no_grad():
                output = model.generate(
                    query_ids,
                    max_new_tokens=10,
                    temperature=0.0,
                    force_abstraction_every_n=K

                )
            
            generated_response = tokenizer.decode(output[0], skip_special_tokens=False)

            generated_ids = output[:, query_ids.shape[1]:]
            
            # Use utility functions for parsing and comparison
            generated_answer = _extract_answer_from_ids(generated_ids[0], tokenizer)
            ground_truth_answer = _extract_answer_from_ids(ground_truth_ids[0], tokenizer)
            
            log_file.write(f"Generated Response: {generated_response}\n")
            log_file.write(f"Generated Answer: {generated_answer}\n")
            log_file.write(f"Ground Truth Answer: {ground_truth_answer}\n")
            log_file.write("-" * 20 + "\n")
            
            generated_num = _str_to_int(generated_answer)
            ground_truth_num = _str_to_int(ground_truth_answer)
            
            if generated_num is not None and generated_num == ground_truth_num:
                correct_predictions += 1
            
            total_samples += 1
            pbar.set_postfix({"Accuracy": f"{(correct_predictions / total_samples) * 100:.2f}%"})

    log_file.close()
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    top_sim_score = sum(top_sim_score) / len(top_sim_score)
    print("\n--- Evaluation Summary ---")
    print(f"Samples Evaluated: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Topological Similarity: {top_sim_score:.2f}")
    return {"accuracy": accuracy, "correct": correct_predictions, "total": total_samples, "top_sim_score": top_sim_score}


def compute_int_emb_similarity(model, tokenizer, string_list): 
    # distance(int1, int2) correlation with cos_sim(emb(int1), emb(int2))
    raise NotImplementedError("Not implemented")

    

def evaluate_multiplication(model, tokenizer, prompt: str, K=None):
    """Generates an answer for a single prompt and prints a detailed comparison."""

    model.eval()
    
    answer_token_id = tokenizer.encode('<answer>', add_special_tokens=False)[0]
    equal_token_id = tokenizer.encode('=', add_special_tokens=False)[0]

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    query_ids, ground_truth_ids = _get_query_and_gt_ids(input_ids, equal_token_id, answer_token_id)
    
    with torch.no_grad():
        output = model.generate(
                query_ids,
                max_new_tokens=10,
                temperature=0.0,
                force_abstraction_every_n=K
        )   

    # Use utility function for parsing
    generated_ids = output[:, query_ids.shape[1]:]
    generated_response = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    generated_answer = _extract_answer_from_ids(generated_ids[0], tokenizer)
    ground_truth_answer = _extract_answer_from_ids(ground_truth_ids[0], tokenizer)
    
    print(f"Query: {tokenizer.decode(query_ids[0], skip_special_tokens=True)}")
    print(f"Generated Response: {generated_response.strip()}")
    print(f"Expected Answer:  {ground_truth_answer}")
    print(f"Generated Answer: {generated_answer}")
    return generated_answer, ground_truth_answer