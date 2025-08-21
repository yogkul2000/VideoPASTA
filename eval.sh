#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_ENABLE_HF_TRANSFER="1"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG

MODEL_PATH=""
OUTPUT_PATH="./logs"
LOG_SUFFIX="vllm"
INFERENCE_TP_SIZE=4
BATCH_SIZE=1

mkdir -p "$OUTPUT_PATH"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Model Path: $MODEL_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Log Suffix: $LOG_SUFFIX"
echo "Batch Size: $BATCH_SIZE"
echo "Tensor Parallel Size: $INFERENCE_TP_SIZE"
echo "==============================================="
echo ""

echo "Starting evaluation..."
python3 -m lmms_eval \
    --model vllm \
    --model_args model="$MODEL_PATH",tensor_parallel_size="$INFERENCE_TP_SIZE",gpu_memory_utilization=0.9 \
    --tasks longvideobench_val_v \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH"