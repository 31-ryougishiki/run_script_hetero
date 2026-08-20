#!/bin/bash
# Prefill 节点启动器。
#
# 用法:
#   start_server.sh <prefill_dp> <prefill_tp> [hetero_tp_sizes] \
#                   [decode_dp] [decode_tp]
#
# - prefill_dp / prefill_tp: 本节点 DP / 统一 TP。
# - hetero_tp_sizes: 可选，逗号分隔的 per-DP TP，例如 3,4,4,4。
# - decode_dp / decode_tp: 远端 decode 池拓扑（默认 16 / 1）。
#
# 场景示例:
#   dp15tp1 + decode dp16tp1:  ./start_server.sh 15 1 "" 16 1
#   dp4tp4  + decode dp8tp2:   ./start_server.sh 4 4 "" 8 2
#   原异构 dp4tp(3,4,4,4):      ./start_server.sh 4 4 3,4,4,4 16 1
#
# 注意: DeepSeek-V4 单 DP TP 必须 <= o_groups(8)，tp>8 不支持。

PREFILL_DP_SIZE=${1:-4}
PREFILL_TP_SIZE=${2:-4}
HETERO_TP_SIZES=${3:-}
DECODE_DP_SIZE=${4:-16}
DECODE_TP_SIZE=${5:-1}

HETERO_ARGS=()
PREFILL_TP_DESCRIPTOR_ARGS=(--prefill-tp-size "$PREFILL_TP_SIZE")
if [ -n "$HETERO_TP_SIZES" ]; then
    HETERO_ARGS=(--hetero-tp-sizes "$HETERO_TP_SIZES")
    # 异构 prefill 的逻辑池描述符取 per-DP tp 的最大值，由 launcher 计算。
    PREFILL_TP_DESCRIPTOR_ARGS=()
fi

python launch_online_dp.py \
  --dp-size "$PREFILL_DP_SIZE" \
  --tp-size "$PREFILL_TP_SIZE" \
  "${HETERO_ARGS[@]}" \
  --dp-size-local "$PREFILL_DP_SIZE" \
  --dp-rank-start 0 \
  --dp-address 7.246.78.76 \
  --dp-rpc-port 12321 \
  --vllm-start-port 7100 \
  --kv-role kv_producer \
  --kv-port 36000 \
  --engine-id 0 \
  --prefill-dp-size "$PREFILL_DP_SIZE" \
  "${PREFILL_TP_DESCRIPTOR_ARGS[@]}" \
  --decode-dp-size "$DECODE_DP_SIZE" \
  --decode-tp-size "$DECODE_TP_SIZE"
