export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

export PYTHONPATH=/opt/its/z30055003/zero_interrupt/vllm-ascend:$PYTHONPATH
export PYTHONPATH=/opt/its/z30055003/zero_interrupt/vllm:$PYTHONPATH
source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

vllm serve /opt/its/model/DeepSeek-V4-Flash-w8a8-mtp-self \
    --max-model-len 1048576 \
    --max-num-batched-tokens 10240 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --api-server-count 1 \
    --max-num-seqs 64 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --no-enable-eplb \
    --heterogeneous-dp-config '[
      {"dp_rank": 0, "tp_size": 3, "tp_sharding_ratios": [2, 1, 1]},
      {"dp_rank": 1, "tp_size": 4},
      {"dp_rank": 2, "tp_size": 4},
      {"dp_rank": 3, "tp_size": 4}
    ]' \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --safetensors-load-strategy 'prefetch' \
    --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
    --quantization ascend \
    --seed 1024 \
    --port 8900 \
    --block-size 128 \
    --enforce-eager \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex":true,
        "enable_static_kernel":false
        },
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert":true}'
