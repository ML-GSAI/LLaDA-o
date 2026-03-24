# Copyright 2025 AntGroup Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_DEBUG=WARN  # 只显示警告和错误

# replace the variables with your own
# Download the released model from https://huggingface.co/GSAI-ML/LLaDA-o
# and point both MODEL_PATH and RESUME_FROM to that local directory for the
# first finetuning run.
MODEL_PATH=${MODEL_PATH:-"/path/to/local/GSAI-ML-LLaDA-o"}
RESUME_FROM=${RESUME_FROM:-"${MODEL_PATH}"}
RESULTS_DIR=${RESULTS_DIR:-"/path/to/your/finetune_run"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${RESULTS_DIR}/checkpoints"}
WANDB_LOG_DIR=${WANDB_LOG_DIR:-"${RESULTS_DIR}"}
DATASET_CONFIG_FILE=${DATASET_CONFIG_FILE:-"${REPO_ROOT}/data/configs/example.yaml"}
LLADAO_DATA_ROOT=${LLADAO_DATA_ROOT:-"/path/to/local/huggingface_datasets"}
LLADAO_T2I_2M_DIR=${LLADAO_T2I_2M_DIR:-"${LLADAO_DATA_ROOT}/text-to-image-2M"}
LLADAO_VLM_BEE_DIR=${LLADAO_VLM_BEE_DIR:-"${LLADAO_DATA_ROOT}/Honey-Data-15M"}

export LLADAO_DATA_ROOT
export LLADAO_T2I_2M_DIR
export LLADAO_VLM_BEE_DIR

# 使用位置参数，设置默认值
NNODE=${1:-1}
NPROC_PER_NODE=${2:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}
echo "Using nnode=$NNODE, nproc_per_node=$NPROC_PER_NODE"

cd "${REPO_ROOT}"

python -m atorch.distributed.run --fault_tolerant --network-check --max_restarts=5 --nnode=$NNODE --nproc_per_node=$NPROC_PER_NODE --rdzv_conf join_timeout=5400 --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
  train/pretrain_unified_navit.py \
  --dataset_config_file "${DATASET_CONFIG_FILE}" \
  --layer_module LLaDAMoTDecoderLayer \
  --results_dir "${RESULTS_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --wandb_log_dir "${WANDB_LOG_DIR}" \
  --wandb_offline True \
  --visual_gen True \
  --visual_und True \
  --total_steps 5100 \
  --warmup_steps 300 \
  --lr_scheduler constant \
  --lr 2.5e-5 \
  --freeze_llm False \
  --freeze_vit False \
  --num_workers 1 \
  --use_flex True \
  --auto_resume True \
  --resume_from "${RESUME_FROM}" \
  --model_path "${MODEL_PATH}" \
  --finetune_from_hf True \
  --sharding_strategy 'FULL_SHARD' \
  --llm_qk_norm True \
  --resume_model_only True \
  --finetune_from_ema True \
  --timestep_shift 4.0 \
  --ce_loss_reweighting True \
  --ce_weight 0.25 \
  --mse_weight 1.0 \
  --ema 0.995 \
  --max_latent_size 64 \
  --ada_len True
