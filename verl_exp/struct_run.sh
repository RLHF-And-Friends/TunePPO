#!/bin/bash
set -x

source /home/vasgreg/TunePPO/venv/bin/activate

export PYTHONPATH="/home/vasgreg/TunePPO/verl${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_HOME=/usr/local/cuda-12.3
export VLLM_USE_V1=1

python3 -m verl.trainer.main_ppo \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.1 \
    data.train_files=/home/vasgreg/TunePPO/data/ruletaker/train.json \
    data.val_files=/home/vasgreg/TunePPO/data/ruletaker/val.json \
    data.train_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=allenai/Olmo-3-7B-Think \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.num_workers=2 \
    reward.custom_reward_function.path=/home/vasgreg/TunePPO/verl_exp/struct_reasoning_reward.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.log_file=/home/vasgreg/TunePPO/verl_exp/reward_log.jsonl \
    +reward.custom_reward_function.reward_kwargs.graph_coverage_scale=1.0 \
    +algorithm.reward_log_keys=[success_rate,coverage,graph_parse_ok,reward_think_tags,reward_answer_tags,reward_graph_tags,reward_graph_parse,reward_correct_answer,reward_reasoning,reward_format] \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=MULTIHOP \
    trainer.experiment_name=VERL-OLMO-Ruletaker-GRPO-STRUCT \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    +reward.custom_reward_function.reward_kwargs.graph_tag_reward=5.0 \
    +reward.custom_reward_function.reward_kwargs.graph_parse_bonus=5.0 \
    "$@"
