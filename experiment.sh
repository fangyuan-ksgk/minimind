#!/bin/bash
# ==============================================================================
# SORL Comprehensive "All-Nighter" Experiment Suite v2
# ==============================================================================
# This script systematically evaluates a wide range of SORL hyperparameters,
# with a deep dive into the interaction between memory fading and temperature.
# Best run on a multi-GPU machine using: torchrun --nproc_per_node=NUM_GPUS experiment.sh

# --- Common Parameters ---
TRAIN_ITERATIONS=15000
BATCH_SIZE=8
MAX_T_SEARCH=128
WANDB_PROJECT="SORL-Overnight-Deep-Dive"
# --- FIX: Define separate memory spans for fading vs. non-fading experiments ---
# For non-fading runs, this sets the base memory span.
MEMORY_SPAN_BASE=1024
# For fading runs, this sets the minimum number of tokens to keep.
# This MUST be less than max_seq_len (typically 1024) to have an effect.
FADE_MIN_KEEP=512
FADE_MIN_KEEP_LESS=256


# --- Group 1: Baseline & Core Temperature Probes (NO Memory Fading) ---
# Goal: Find the most effective temperature and stability strategy without memory fading.

# Exp 1: Baseline (No SORL Abstraction)
echo "--- Starting Exp 1.0: Baseline ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $MEMORY_SPAN_BASE \
    --max_t_search 0 --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "1.0-baseline"

# Exp 2: High Temperature (1.0) Series
echo "--- Starting Exp 1.1: Temp 1.0 ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $MEMORY_SPAN_BASE \
    --max_t_search $MAX_T_SEARCH --temperature 1.0 --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "1.1-temp-1.0"

echo "--- Starting Exp 1.2: Temp 1.0 with Flip ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $MEMORY_SPAN_BASE \
    --max_t_search $MAX_T_SEARCH --temperature 1.0 --temperature_flip --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "1.2-temp-1.0-flip"

# Exp 3: Medium Temperature (0.5) Series
echo "--- Starting Exp 1.3: Temp 0.5 ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $MEMORY_SPAN_BASE \
    --max_t_search $MAX_T_SEARCH --temperature 0.5 --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "1.3-temp-0.5"

echo "--- Starting Exp 1.4: Temp 0.5 with Flip ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $MEMORY_SPAN_BASE \
    --max_t_search $MAX_T_SEARCH --temperature 0.5 --temperature_flip --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "1.4-temp-0.5-flip"

# --- Group 2: Deep Dive into Memory Fading ---
# Goal: Systematically evaluate the impact of memory fading across all temperature strategies.

echo "--- Starting Exp 2.1: FadeMem + Temp 1.0 ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP \
    --max_t_search $MAX_T_SEARCH --temperature 1.0 --use_fade_memory --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "2.1-fademem-temp-1.0"

echo "--- Starting Exp 2.2: FadeMem + Temp 1.0 (Less Memory) ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP_LESS \
    --max_t_search $MAX_T_SEARCH --temperature 1.0 --use_fade_memory --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "2.2-fademem-temp-1.0-less"

echo "--- Starting Exp 2.3: FadeMem + Temp 1.0 with Flip ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP \
    --max_t_search $MAX_T_SEARCH --temperature 1.0 --temperature_flip --use_fade_memory --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "2.3-fademem-temp-1.0-flip"

echo "--- Starting Exp 2.4: FadeMem + Temp 1.0 with Flip (Less Memory) ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP_LESS \
    --max_t_search $MAX_T_SEARCH --temperature 1.0 --temperature_flip --use_fade_memory --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "2.4-fademem-temp-1.0-flip-less"

echo "--- Starting Exp 2.5: FadeMem + Temp 0.5 ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP \
    --max_t_search $MAX_T_SEARCH --temperature 0.5 --use_fade_memory --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "2.5-fademem-temp-0.5"

echo "--- Starting Exp 2.6: FadeMem + Temp 0.5 (Less Memory) ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP_LESS \
    --max_t_search $MAX_T_SEARCH --temperature 0.5 --use_fade_memory --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "2.6-fademem-temp-0.5-less"

echo "--- Starting Exp 2.7: FadeMem + Temp 0.5 with Flip ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP \
    --max_t_search $MAX_T_SEARCH --temperature 0.5 --temperature_flip --use_fade_memory --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "2.7-fademem-temp-0.5-flip"

echo "--- Starting Exp 2.8: FadeMem + Temp 0.5 with Flip (Less Memory) ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP_LESS \
    --max_t_search $MAX_T_SEARCH --temperature 0.5 --temperature_flip --use_fade_memory --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "2.8-fademem-temp-0.5-flip-less"

echo "--- Starting Exp 2.9: FadeMem + Temp 0.2 (Greedy) ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP \
    --max_t_search $MAX_T_SEARCH --temperature 0.2 --use_fade_memory --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "2.9-fademem-temp-0.2"

echo "--- Starting Exp 2.10: FadeMem + Temp 0.2 (Greedy, Less Memory) ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP_LESS \
    --max_t_search $MAX_T_SEARCH --temperature 0.2 --use_fade_memory --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "2.10-fademem-temp-0.2-less"

# --- Group 3: Placeholder and Budget Analysis (using best config from Group 2) ---
# Goal: Test the effect of spike-based placeholders. Assumes memory fading + medium temp is a good combo.

echo "--- Starting Exp 3.1: Rhythmic + Spike (Budget 10) ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP \
    --max_t_search $MAX_T_SEARCH --temperature 0.5 --use_fade_memory \
    --use_spike_placeholders --abstract_budget 10 \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "3.1-spike-budget-10"

echo "--- All experiments complete. ---"
