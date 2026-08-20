#!/bin/bash
# PD 代理启动器。prefill/decode DP 数量变化时只需要设置下列环境变量:
#   PREFILL_DP_SIZE  (default 4)
#   DECODE_DP_SIZE   (default 16)
#   PREFILL_HOST     (default 7.246.78.76)
#   DECODE_HOST      (default 7.246.78.75)
#   PREFILL_PORT_BASE(default 7100)
#   DECODE_PORT_BASE (default 7100)
#   PROXY_PORT       (default 9000)
#   PROXY_HOST       (default 7.246.78.76)
#
# 场景:
#   场景2: PREFILL_DP_SIZE=15 DECODE_DP_SIZE=16 ./proxy.sh
#   场景3: PREFILL_DP_SIZE=4  DECODE_DP_SIZE=8  ./proxy.sh

PREFILL_DP_SIZE=${PREFILL_DP_SIZE:-4}
DECODE_DP_SIZE=${DECODE_DP_SIZE:-16}
PREFILL_HOST=${PREFILL_HOST:-7.246.78.76}
DECODE_HOST=${DECODE_HOST:-7.246.78.75}
PREFILL_PORT_BASE=${PREFILL_PORT_BASE:-7100}
DECODE_PORT_BASE=${DECODE_PORT_BASE:-7100}
PROXY_PORT=${PROXY_PORT:-9000}
PROXY_HOST=${PROXY_HOST:-7.246.78.76}

PREFILL_HOSTS_ARGS=()
PREFILL_PORTS_ARGS=()
for ((i = 0; i < PREFILL_DP_SIZE; i++)); do
    PREFILL_HOSTS_ARGS+=("$PREFILL_HOST")
    PREFILL_PORTS_ARGS+=("$((PREFILL_PORT_BASE + i))")
done

DECODE_HOSTS_ARGS=()
DECODE_PORTS_ARGS=()
for ((i = 0; i < DECODE_DP_SIZE; i++)); do
    DECODE_HOSTS_ARGS+=("$DECODE_HOST")
    DECODE_PORTS_ARGS+=("$((DECODE_PORT_BASE + i))")
done

python load_balance_proxy_server_example.py \
  --port "$PROXY_PORT" \
  --host "$PROXY_HOST" \
  --prefiller-hosts "${PREFILL_HOSTS_ARGS[@]}" \
  --prefiller-ports "${PREFILL_PORTS_ARGS[@]}" \
  --decoder-hosts "${DECODE_HOSTS_ARGS[@]}" \
  --decoder-ports "${DECODE_PORTS_ARGS[@]}"
