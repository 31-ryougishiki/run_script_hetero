#!/bin/bash
# Decode 节点启动器。
#
# 用法:
#   start_server.sh <decode_dp> <decode_tp> [hetero_tp_sizes] \
#                   <prefill_dp> <prefill_tp> [enable_dsa_cp] [enable_sp]
#
# - decode_dp / decode_tp: 本节点 DP / 统一 TP。
# - hetero_tp_sizes: 可选，逗号分隔的 per-DP TP。
# - prefill_dp / prefill_tp: 远端 prefill 池拓扑（默认 4 / 4）。
# - enable_dsa_cp: 1/0，默认 0。
# - enable_sp: 1/0，默认 0。
#
# 场景示例:
#   decode dp16tp1 + prefill dp15tp1: ./start_server.sh 16 1 "" 15 1 0 0
#   decode dp8tp2  + prefill dp4tp4:  ./start_server.sh 8 2 "" 4 4 0 0
#   原异构 prefill dp4tp(3,4,4,4):    ./start_server.sh 16 1 "" 4 4 0 0
#
# 注意: DeepSeek-V4 单 DP TP 必须 <= o_groups(8)，tp>8 不支持。

DECODE_DP_SIZE=${1:-16}
DECODE_TP_SIZE=${2:-1}
HETERO_TP_SIZES=${3:-}
PREFILL_DP_SIZE=${4:-4}
PREFILL_TP_SIZE=${5:-4}
ENABLE_DSA_CP=${6:-0}
ENABLE_SP=${7:-0}

HETERO_ARGS=()
DECODE_TP_DESCRIPTOR_ARGS=(--decode-tp-size "$DECODE_TP_SIZE")
if [ -n "$HETERO_TP_SIZES" ]; then
    HETERO_ARGS=(--hetero-tp-sizes "$HETERO_TP_SIZES")
    # 异构 decode 的逻辑池描述符取 per-DP tp 的最大值，由 launcher 计算。
    DECODE_TP_DESCRIPTOR_ARGS=()
fi

DSA_CP_ARGS=(--no-enable-dsa-cp)
if [ "$ENABLE_DSA_CP" = "1" ]; then
    DSA_CP_ARGS=(--enable-dsa-cp)
fi
SP_ARGS=(--no-enable-sp)
if [ "$ENABLE_SP" = "1" ]; then
    SP_ARGS=(--enable-sp)
fi

python launch_online_dp.py \
  --dp-size "$DECODE_DP_SIZE" \
  --tp-size "$DECODE_TP_SIZE" \
  "${HETERO_ARGS[@]}" \
  --dp-size-local "$DECODE_DP_SIZE" \
  --dp-rank-start 0 \
  --device-start "${DECODE_DEVICE_START:-0}" \
  --dp-address "${DECODE_DP_ADDRESS:-7.246.78.75}" \
  --dp-rpc-port "${DECODE_DP_RPC_PORT:-12321}" \
  --vllm-start-port "${DECODE_VLLM_START_PORT:-7100}" \
  --kv-role kv_consumer \
  --kv-port "${DECODE_KV_PORT:-36200}" \
  --engine-id 1 \
  --prefill-dp-size "$PREFILL_DP_SIZE" \
  --prefill-tp-size "$PREFILL_TP_SIZE" \
  --decode-dp-size "$DECODE_DP_SIZE" \
  "${DECODE_TP_DESCRIPTOR_ARGS[@]}" \
  "${DSA_CP_ARGS[@]}" \
  "${SP_ARGS[@]}"
