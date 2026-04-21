#!/bin/bash
set -x

source /home/vasgreg/TunePPO/venv/bin/activate

export PYTHONPATH="/home/vasgreg/TunePPO/verl${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_HOME=/usr/local/cuda-12.3
export VLLM_USE_V1=1

LLM_GRAPH_EXTRACTOR_ENABLED=${RULETAKER_LLM_GRAPH_EXTRACTOR_ENABLED:-false}
LLM_GRAPH_EXTRACTOR_MODEL=${RULETAKER_LLM_GRAPH_EXTRACTOR_MODEL:-null}
LLM_GRAPH_EXTRACTOR_BASE_URL=${RULETAKER_LLM_GRAPH_EXTRACTOR_BASE_URL:-null}
LLM_GRAPH_EXTRACTOR_TIMEOUT=${RULETAKER_LLM_GRAPH_EXTRACTOR_TIMEOUT:-30.0}
LLM_GRAPH_EXTRACTOR_TEMPERATURE=${RULETAKER_LLM_GRAPH_EXTRACTOR_TEMPERATURE:-0.0}
LLM_GRAPH_EXTRACTOR_MAX_RETRIES=${RULETAKER_LLM_GRAPH_EXTRACTOR_MAX_RETRIES:-2}

python3 -m verl.trainer.main_ppo \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.1 \
    data.train_files=/home/vasgreg/TunePPO/data/ruletaker/train.json \
    data.val_files=/home/vasgreg/TunePPO/data/ruletaker/val.json \
    data.train_batch_size=4 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=meta-llama/Meta-Llama-3.1-8B-Instruct \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.num_workers=2 \
    reward.custom_reward_function.path=/home/vasgreg/TunePPO/ruletaker_reward.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.log_file=/home/vasgreg/TunePPO/reward_log.jsonl \
    +reward.custom_reward_function.reward_kwargs.graph_coverage_scale=1.0 \
    +reward.custom_reward_function.reward_kwargs.llm_graph_extractor_enabled=${LLM_GRAPH_EXTRACTOR_ENABLED} \
    +reward.custom_reward_function.reward_kwargs.llm_graph_extractor_model=${LLM_GRAPH_EXTRACTOR_MODEL} \
    +reward.custom_reward_function.reward_kwargs.llm_graph_extractor_base_url=${LLM_GRAPH_EXTRACTOR_BASE_URL} \
    +reward.custom_reward_function.reward_kwargs.llm_graph_extractor_timeout=${LLM_GRAPH_EXTRACTOR_TIMEOUT} \
    +reward.custom_reward_function.reward_kwargs.llm_graph_extractor_temperature=${LLM_GRAPH_EXTRACTOR_TEMPERATURE} \
    +reward.custom_reward_function.reward_kwargs.llm_graph_extractor_max_retries=${LLM_GRAPH_EXTRACTOR_MAX_RETRIES} \
    +algorithm.reward_log_keys=[success_rate,coverage,reward_think_tags,reward_answer_tags,reward_correct_answer,reward_reasoning,reward_format] \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=MULTIHOP \
    trainer.experiment_name=VERL-QWEN3-Ruletaker-GRPO-LLM_MATCH \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    "$@"
