#!/bin/bash
set -x

source /home/user5/kg_reasoning_vg/TunePPO/.venv/bin/activate

ROOT="/home/user5/kg_reasoning_vg/TunePPO"

export PYTHONPATH="$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_HOME=/usr/local/cuda-12.3
export CUDA_VISIBLE_DEVICES=1

SCRATCH=/dev/shm/user5
EXP_NAME=VERL-QWEN3-0.6B-EDGE
RAY_TMP=$SCRATCH/ray_tmp_$CUDA_VISIBLE_DEVICES
REWARD_LOG_DIR=$SCRATCH/reward_logs/$EXP_NAME
mkdir -p $RAY_TMP $REWARD_LOG_DIR $SCRATCH/hf_cache

export RAY_TMPDIR=$RAY_TMP
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export WANDB_DIR=$SCRATCH/wandb
mkdir -p $WANDB_DIR
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=1
export RAY_BACKEND_LOG_LEVEL=warning
export MASTER_PORT=$((29500 + CUDA_VISIBLE_DEVICES))
export RAY_ADDRESS=local


python3 -m verl.trainer.main_ppo \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.1 \
    data.train_files=$ROOT/data/ruletaker_kg/train.json \
    data.val_files=$ROOT/data/ruletaker_kg/val.json \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.n=8 \
    data.seed=42 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=40960 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=40960 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward.num_workers=16 \
    reward.custom_reward_function.path=$ROOT/verl_exp/edge_reasoning_reward.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.graph_validity_scale=1.0 \
    +reward.custom_reward_function.reward_kwargs.graph_validity_reward=100.0 \
    +reward.custom_reward_function.reward_kwargs.correct_answer_reward=100.0 \
    +reward.custom_reward_function.reward_kwargs.graph_parse_bonus=5.0 \
    +reward.custom_reward_function.reward_kwargs.log_file=$REWARD_LOG_DIR/reward_log.jsonl \
    +algorithm.reward_log_keys=[success_rate,graph_coverage,graph_parse_ok,reward_think_tags,reward_answer_tags,reward_graph_parse,reward_correct_answer,reward_graph_validity,reward_format] \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=MULTIHOP \
    trainer.experiment_name=$EXP_NAME \
    trainer.val_before_train=True \
    data.val_max_samples=1000 \
    trainer.test_freq=10 \
    trainer.log_val_generations=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    "$@"
