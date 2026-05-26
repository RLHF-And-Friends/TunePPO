export HF_TOKEN=

CKPT=/home/user5/kg_reasoning_vg/TunePPO/checkpoints/MULTIHOP/VERL-QWEN3-0.6B-STRUCT-H100-cov100-ans100-bigval/global_step_140/actor
OUT=~/kg_reasoning_vg/TunePPO/checkpoints/cov100-ans100-step140

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CKPT" \
    --target_dir "$OUT" \
    --hf_upload_path VavGreg/qwen3-06b-ruletaker-grpo-step140 \
    --private