#!/bin/bash
# 在容器内执行的服务启停控制脚本。
#
# 用法（容器内）:
#   remote_ctl.sh start-prefill <prefill_dp> <prefill_tp> [hetero_tp_sizes] \
#                  <decode_dp> <decode_tp> <enable_dsa_cp> <enable_sp>
#   remote_ctl.sh start-decode  <decode_dp> <decode_tp> [hetero_tp_sizes] \
#                  <prefill_dp> <prefill_tp> <enable_dsa_cp> <enable_sp>
#   remote_ctl.sh start-proxy   <prefill_dp> <decode_dp>
#   remote_ctl.sh stop-prefill
#   remote_ctl.sh stop-decode
#   remote_ctl.sh stop-proxy
#
# 由 automated_hetero_test.sh 通过 `docker exec -d` 远程调用。

HETERO_DIR=${HETERO_DIR:-/opt/its/z30055003/hetero}
ACTION=${1:-}
shift || true

case "$ACTION" in
    start-prefill)
        pref_dp=${1:?prefill dp}
        pref_tp=${2:?prefill tp}
        pref_hetero=${3:-}
        [ "$pref_hetero" = "@EMPTY@" ] && pref_hetero=""
        shift 3
        cd "$HETERO_DIR/prefill" || exit 1
        exec bash ./start_server.sh \
            "$pref_dp" "$pref_tp" "$pref_hetero" "$@"
        ;;
    start-decode)
        dec_dp=${1:?decode dp}
        dec_tp=${2:?decode tp}
        dec_hetero=${3:-}
        [ "$dec_hetero" = "@EMPTY@" ] && dec_hetero=""
        shift 3
        cd "$HETERO_DIR/decode" || exit 1
        exec bash ./start_server.sh \
            "$dec_dp" "$dec_tp" "$dec_hetero" "$@"
        ;;
    start-proxy)
        PREFILL_DP_SIZE=${1:?prefill dp}
        DECODE_DP_SIZE=${2:?decode dp}
        export PREFILL_DP_SIZE DECODE_DP_SIZE
        cd "$HETERO_DIR" || exit 1
        exec bash ./proxy.sh
        ;;
    stop-prefill)
        pkill -f 'run_dp_template.sh' || true
        pkill -f 'launch_online_dp.py' || true
        pkill -f 'vllm serve /opt/its/model' || true
        ;;
    stop-decode)
        pkill -f 'run_dp_template.sh' || true
        pkill -f 'launch_online_dp.py' || true
        pkill -f 'vllm serve /opt/its/model' || true
        ;;
    stop-proxy)
        pkill -f 'load_balance_proxy_server_example.py' || true
        ;;
    *)
        echo "unknown action: $ACTION" >&2
        exit 1
        ;;
esac
