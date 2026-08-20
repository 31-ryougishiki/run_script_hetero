# ===== 参数说明 =====
# $1=可见卡列表, $2=vllm端口, $3=DP总数, $4=当前DP rank,
# $5=DP master地址, $6=DP rpc端口, $7=当前DP rank的TP size。
#
# 以下环境变量由 launch_online_dp.py 注入：
#   HETERO_DP_CONFIG_JSON       异构 DP/TP 配置（可选）
#   KV_TRANSFER_CONFIG_JSON     KV connector 完整配置（含 prefill/decode 拓扑）
#   ADDITIONAL_CONFIG_JSON      additional_config（含 enable_dsa_cp 开关）
#   VLLM_ASCEND_ENABLE_FLASHCOMM1  按 --enable-sp 注入

nic_name="${DECODE_NIC_NAME:-eth2}" # change to your own nic name
local_ip="${DECODE_LOCAL_IP:-7.246.78.75}" # change to your own ip
model_path=/opt/its/model/DeepSeek-V4-Flash-w8a8-mtp-self

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_IF_IP=$local_ip
export VLLM_HOST_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
# Mooncake/ADXL 超时（ms）：首个跨节点 HcclCommPrepare 在大量
# prefill/decode worker 并发启动时可能超过默认值，适当放大建链窗口。
export ASCEND_CONNECT_TIMEOUT=180000
export ASCEND_TRANSFER_TIMEOUT=300000
# Mooncake Python 侧 batch 整体超时（s，默认仅 30s）。必须大于
# ASCEND_CONNECT_TIMEOUT，否则第一次建链未完成 batch_transfer_sync_read
# 就提前返回 -1，表现为 "Sync batch data transfer timeout"。
export MC_TRANSFER_TIMEOUT=600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024

export PYTHONPATH=/opt/its/z30055003/zero_interrupt/vllm-ascend:$PYTHONPATH
export PYTHONPATH=/opt/its/z30055003/zero_interrupt/vllm:$PYTHONPATH
source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

if [ "${HETERO_ENABLE_SP:-0}" = "1" ]; then
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
else
    unset VLLM_ASCEND_ENABLE_FLASHCOMM1
fi
export ASCEND_RT_VISIBLE_DEVICES=$1

: "${KV_TRANSFER_CONFIG_JSON:?launch_online_dp.py must export KV_TRANSFER_CONFIG_JSON}"
: "${ADDITIONAL_CONFIG_JSON:?launch_online_dp.py must export ADDITIONAL_CONFIG_JSON}"
HETERO_DP_ARGS=()
if [ -n "${HETERO_DP_CONFIG_JSON:-}" ]; then
    HETERO_DP_ARGS=(--heterogeneous-dp-config "$HETERO_DP_CONFIG_JSON")
fi

vllm serve "$model_path" \
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
    --max-num-batched-tokens 120 \
    --max-num-seqs 60 \
    --async-scheduling \
    --block-size 128 \
    --no-disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --safetensors-load-strategy 'prefetch' \
    --trust-remote-code \
    --tokenizer-mode deepseek_v4 \
    --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    "${HETERO_DP_ARGS[@]}" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG_JSON" \
    --additional-config "$ADDITIONAL_CONFIG_JSON"
