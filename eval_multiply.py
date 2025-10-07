import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
import torch
import Levenshtein

# Evaluation Gadget for Digit Multiplication 
# ------------------------------------------------------------------------------

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


def evaluate_multiplication(model, tokenizer, prompt: str):
    model.eval()
    answer_token_id = tokenizer.encode('<answer>', add_special_tokens=False)[0]
    
    # Prepare model inputs
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    # Find the position of <answer> to split query and ground truth
    answer_indices = torch.where(input_ids == answer_token_id)
    if len(answer_indices[0]) == 0:
        print("Error: <answer> token not found in prompt.")
        return
    
    answer_idx = answer_indices[1][0]
    query_ids = input_ids[:, :answer_idx + 1]
    ground_truth_ids = input_ids[:, answer_idx + 1:]
    
    # Generate the answer from the model
    with torch.no_grad():
        output = model.generate(
            query_ids,
            max_new_tokens=10,
            temperature=0.0
        )
    
    # Decode and compare
    generated_ids = output[:, query_ids.shape[1]:]
    generated_answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    ground_truth_answer = tokenizer.decode(ground_truth_ids[0], skip_special_tokens=True).strip()
    
    print(f"Query: {tokenizer.decode(query_ids[0])}")
    print(f"Generated Response: {tokenizer.decode(generated_ids[0])}")
    print(f"Expected Answer:  {ground_truth_answer}")
    print(f"Generated Answer: {generated_answer}")