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
MAX_T_SEARCH=128 # max_len // K = 128 // 8 = 128 this is correct
WANDB_PROJECT="SORL-Overnight-Deep-Dive"
# --- FIX: Define separate memory spans for fading vs. non-fading experiments ---
# For non-fading runs, this sets the base memory span.
MEMORY_SPAN_BASE=1024
# For fading runs, this sets the minimum number of tokens to keep.
# This MUST be less than max_seq_len (typically 1024) to have an effect.
FADE_MIN_KEEP=512
FADE_MIN_KEEP_LESS=256


# ---- Group 4.Gated Phase Transition (in order to balancee the two targets) ----------
# (4.1) check what happens when default_phase is set to 1 (so abstraction is basically always a noise)
# (4.2) use GAPT to balance the two targets | rest same as (1.1) - temp 1.0, no memory degradation 
# (4.3) use GAPT to balance the two targets | rest same as (2.1) - temp 1.0, memory degradation

echo "--- Starting Exp 4.1: GAPT Default Phase 1 ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $MEMORY_SPAN_BASE \
    --max_t_search $MAX_T_SEARCH --temperature 1.0 --no-use_spike_placeholders \
    --default_phase 1 \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "4.1-gapt-default1-temp1.0"

echo "--- Starting Exp 4.2: GAPT + Temp 1.0 ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $MEMORY_SPAN_BASE \
    --max_t_search $MAX_T_SEARCH --temperature 1.0 --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "4.2-gapt-temp1.0"

echo "--- Starting Exp 4.3: GAPT + FadeMem + Temp 1.0 ---"
python trainer/train_sorl.py \
    --train_iterations $TRAIN_ITERATIONS --batch_size $BATCH_SIZE --memory_span $FADE_MIN_KEEP \
    --max_t_search $MAX_T_SEARCH --temperature 1.0 --use_fade_memory --no-use_spike_placeholders \
    --use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name "4.3-gapt-fademem-temp1.0"


echo "--- All experiments complete. ---"
