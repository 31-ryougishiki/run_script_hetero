nic_name="eth2"
local_ip=7.246.78.76
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export VLLM_RPC_TIMEOUT=3600000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=2560
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_OP_EXPANSION_MODE="AIV"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

# ===== 脚本2额外设置 =====
export PYTHONPATH=/opt/its/z30055003/zero_interrupt/vllm-ascend:$PYTHONPATH
export PYTHONPATH=/opt/its/z30055003/zero_interrupt/vllm:$PYTHONPATH
source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 可见卡必须在 set_env.bash 之后设置：某些 CANN set_env 脚本会重写或
# 重置 ASCEND_RT_VISIBLE_DEVICES，导致 launcher 传入的 4 卡变成 3 卡。
export ASCEND_RT_VISIBLE_DEVICES=$1

# ===== vllm serve 参数（按脚本1顺序重排） =====
# 参数说明：$1=可见卡列表, $2=vllm端口, $3=DP总数, $4=当前DP rank,
#          $5=DP master地址, $6=DP rpc端口, $7=当前DP rank的TP size。
# 异构场景下 launcher 会为 dp0 传 3、dp1..3 传 4；
# --heterogeneous-dp-config 仍必须保留，vLLM 会再次按 dp_rank 校验/覆写。
vllm serve /opt/its/model/DeepSeek-V4-Flash-w8a8-mtp-self \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv4 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --no-disable-hybrid-kv-cache-manager \
    --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
    --no-enable-prefix-caching \
    --safetensors-load-strategy 'prefetch' \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
    --block-size 128 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --enforce-eager \
    --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true,  "enable_dsa_cp": true}' \
    --no-enable-eplb \
    --heterogeneous-dp-config '[
      {"dp_rank": 0, "tp_size": 3, "tp_sharding_ratios": [2, 1, 1]},
      {"dp_rank": 1, "tp_size": 4},
      {"dp_rank": 2, "tp_size": 4},
      {"dp_rank": 3, "tp_size": 4}
    ]' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 4,
                        "tp_size": 4
                },
                "decode": {
                        "dp_size": 16,
                        "tp_size": 1
                }
        }
    }'