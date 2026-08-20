#!/bin/bash
# 按场景启动 prefill / decode / proxy。
#
# 用法:
#   ./run_scenario.sh <2|3|hetero> <prefill|decode|proxy>
#
# 场景:
#   2: prefill dp15tp1  + decode dp16tp1
#   3: prefill dp4tp4   + decode dp8tp2
#   hetero: prefill dp4tp(3,4,4,4) + decode dp16tp1 (当前基线)
#
# 场景1（prefill dp1tp15）依赖 DSA-CP tp>8 支持，已按需求回退，不再提供。
SCENARIO=${1:?usage: run_scenario.sh <2|3|hetero> <prefill|decode|proxy>}
ROLE=${2:?usage: run_scenario.sh <2|3|hetero> <prefill|decode|proxy>}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$SCENARIO" in
    1)
        echo "scenario 1 (prefill dp1tp15) requires tp>8, which is not supported." >&2
        exit 1
        ;;
    2)
        PREFILL_DP=15; PREFILL_TP=1;  PREFILL_HETERO=""
        DECODE_DP=16;  DECODE_TP=1;  DECODE_HETERO=""
        ;;
    3)
        PREFILL_DP=4;  PREFILL_TP=4;  PREFILL_HETERO=""
        DECODE_DP=8;   DECODE_TP=2;  DECODE_HETERO=""
        ;;
    hetero)
        PREFILL_DP=4;  PREFILL_TP=4;  PREFILL_HETERO="3,4,4,4"
        DECODE_DP=16;  DECODE_TP=1;  DECODE_HETERO=""
        ;;
    *)
        echo "Unknown scenario: $SCENARIO (expected 2, 3 or hetero)" >&2
        exit 1
        ;;
esac

case "$ROLE" in
    prefill)
        cd "$SCRIPT_DIR/prefill" || exit 1
        exec bash ./start_server.sh \
            "$PREFILL_DP" "$PREFILL_TP" "$PREFILL_HETERO" \
            "$DECODE_DP" "$DECODE_TP"
        ;;
    decode)
        cd "$SCRIPT_DIR/decode" || exit 1
        exec bash ./start_server.sh \
            "$DECODE_DP" "$DECODE_TP" "$DECODE_HETERO" \
            "$PREFILL_DP" "$PREFILL_TP"
        ;;
    proxy)
        export PREFILL_DP_SIZE="$PREFILL_DP"
        export DECODE_DP_SIZE="$DECODE_DP"
        cd "$SCRIPT_DIR" || exit 1
        exec bash ./proxy.sh
        ;;
    *)
        echo "Unknown role: $ROLE (expected prefill, decode or proxy)" >&2
        exit 1
        ;;
esac
