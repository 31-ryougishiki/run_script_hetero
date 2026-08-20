#!/bin/bash
# Decode 节点启动器。
#
# 用法:
#   start_server.sh <decode_dp> <decode_tp> [hetero_tp_sizes] \
#                   <prefill_dp> <prefill_tp>
#
# 场景示例:
#   decode dp16tp1 + prefill dp15tp1:  ./start_server.sh 16 1 "" 15 1
#   decode dp8tp2  + prefill dp4tp4:   ./start_server.sh 8 2 "" 4 4
#   原异构 prefill dp4tp(3,4,4,4):      ./start_server.sh 16 1 "" 4 4
#
# 注意: DeepSeek-V4 单 DP TP 必须 <= o_groups(8)，tp>8 不支持。

DECODE_DP_SIZE=${1:-16}
DECODE_TP_SIZE=${2:-1}
HETERO_TP_SIZES=${3:-}
PREFILL_DP_SIZE=${4:-4}
PREFILL_TP_SIZE=${5:-4}

HETERO_ARGS=()
DECODE_TP_DESCRIPTOR_ARGS=(--decode-tp-size "$DECODE_TP_SIZE")
if [ -n "$HETERO_TP_SIZES" ]; then
    HETERO_ARGS=(--hetero-tp-sizes "$HETERO_TP_SIZES")
    # 异构 decode 的逻辑池描述符取 per-DP tp 的最大值，由 launcher 计算。
    DECODE_TP_DESCRIPTOR_ARGS=()
fi

python launch_online_dp.py \
  --dp-size "$DECODE_DP_SIZE" \
  --tp-size "$DECODE_TP_SIZE" \
  "${HETERO_ARGS[@]}" \
  --dp-size-local "$DECODE_DP_SIZE" \
  --dp-rank-start 0 \
  --dp-address 7.246.78.75 \
  --dp-rpc-port 12321 \
  --vllm-start-port 7100 \
  --kv-role kv_consumer \
  --kv-port 36200 \
  --engine-id 1 \
  --prefill-dp-size "$PREFILL_DP_SIZE" \
  --prefill-tp-size "$PREFILL_TP_SIZE" \
  --decode-dp-size "$DECODE_DP_SIZE" \
  "${DECODE_TP_DESCRIPTOR_ARGS[@]}"
