#!/bin/bash
# 单独测试 P 或 D 实例（不通过 PD 分离代理，实例自身直接处理请求）。
#
# 保留 kv_producer / kv_consumer 角色启动，但不要求对端在线；请求直接发到
# 本实例的每个 vLLM engine 端口。
#
# 用法:
#   bash single_instance_test.sh <prefill|decode> list
#   bash single_instance_test.sh <prefill|decode> all
#   bash single_instance_test.sh <prefill|decode> <case>
#   bash single_instance_test.sh <prefill|decode> custom
#
# custom 通过环境变量覆盖:
#   DP_SIZE TP_SIZE HETERO_TP_SIZES DSA_CP SP REMOTE_DP
#   REMOTE_DECODE_TP / REMOTE_PREFILL_TP LOCAL_IP NPU_TOTAL PORT_BASE
#
# tp16/tp15 这类参数可传入，但当前实现限制 TP <= o_groups(8)，
# 脚本会在启动前明确报错，不会拉起进程。

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE=${1:?usage: single_instance_test.sh <prefill|decode> <case|list|custom>}
MODE=${2:-list}
MAX_TP=${MAX_TP:-8}
RESULT_ROOT=${RESULT_ROOT:-./single_instance_results}
LOCAL_IP=${LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
LOCAL_IP=${LOCAL_IP:-127.0.0.1}
NPU_TOTAL=${NPU_TOTAL:-16}
MIN_TEST_CARDS=${MIN_TEST_CARDS:-7}
PORT_BASE=${PORT_BASE:-7100}
REQUESTS=${REQUESTS:-16}
CONCURRENCY=${CONCURRENCY:-4}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-900}
KEEP_RUNNING=${KEEP_RUNNING:-0}
REMOTE_DP=${REMOTE_DP:-16}
REMOTE_DECODE_TP=${REMOTE_DECODE_TP:-1}
REMOTE_PREFILL_TP=${REMOTE_PREFILL_TP:-}
INSTANCE_LOG=/tmp/single_instance_test.log

log() { echo "[$(date '+%F %T')] $*"; }

cleanup() {
    if [ "$KEEP_RUNNING" = "1" ]; then
        return
    fi
    log "清理单实例服务..."
    pkill -f 'launch_online_dp.py' >/dev/null 2>&1 || true
    pkill -f 'run_dp_template.sh' >/dev/null 2>&1 || true
    pkill -f 'vllm serve /opt/its/model' >/dev/null 2>&1 || true
    sleep 2
}
trap cleanup EXIT

wait_for_ports() {
    local host="$1" base="$2" count="$3" timeout="$4" label="$5"
    python3 - "$host" "$base" "$count" "$timeout" "$label" <<'PY'
import http.client
import sys
import time

host = sys.argv[1]
base = int(sys.argv[2])
count = int(sys.argv[3])
timeout = int(sys.argv[4])
label = sys.argv[5]
remaining = set(range(base, base + count))
deadline = time.time() + timeout
print(f"[{label}] waiting for ports {base}..{base + count - 1} on {host}",
      flush=True)
while time.time() < deadline:
    for port in sorted(remaining):
        try:
            conn = http.client.HTTPConnection(host, port, timeout=3)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                remaining.discard(port)
                print(f"[{label}] {host}:{port} ready", flush=True)
        except Exception:
            pass
    if not remaining:
        print(f"[{label}] all ports ready", flush=True)
        sys.exit(0)
    time.sleep(10)
print(f"[{label}] timeout, remaining: {sorted(remaining)}", flush=True)
sys.exit(1)
PY
}

sum_hetero_cards() {
    local sizes="$1" total=0 size
    IFS=',' read -ra parts <<<"$sizes"
    for size in "${parts[@]}"; do
        total=$((total + size))
    done
    echo "$total"
}

max_tp() {
    local sizes="$1" tp="$2" size max="$tp"
    if [ -n "$sizes" ]; then
        IFS=',' read -ra parts <<<"$sizes"
        for size in "${parts[@]}"; do
            [ "$size" -gt "$max" ] && max="$size"
        done
    fi
    echo "$max"
}

validate_tp() {
    local hetero="$1" tp="$2" size
    if [ -n "$hetero" ]; then
        IFS=',' read -ra sizes <<<"$hetero"
        for size in "${sizes[@]}"; do
            [ "$size" -le "$MAX_TP" ] || return 1
        done
        return 0
    fi
    [ "$tp" -le "$MAX_TP" ] || return 1
}

list_cases() {
    if [ "$ROLE" = "prefill" ]; then
        echo "dp3_tp4 dp2_tp43 dp1_tp7 dp4_tp3444 dp2_tp43_no_sp_no_dsa dp4_tp3444_no_sp_no_dsa dp2_tp43_no_dsa_sp dp4_tp3444_no_dsa_sp"
    elif [ "$ROLE" = "decode" ]; then
        echo "dp4_tp2122 dp4_tp2212"
    else
        echo "unknown role: $ROLE" >&2
        exit 2
    fi
}

set_case_vars() {
    case "$ROLE:$MODE" in
        prefill:dp3_tp4)
            DP_SIZE=3; TP_SIZE=4; HETERO="4,4,4"; DSA_CP=1; SP=1 ;;
        prefill:dp2_tp43)
            DP_SIZE=2; TP_SIZE=4; HETERO="4,3"; DSA_CP=1; SP=1 ;;
        prefill:dp1_tp7)
            DP_SIZE=1; TP_SIZE=7; HETERO="7"; DSA_CP=1; SP=1 ;;
        prefill:dp4_tp3444)
            DP_SIZE=4; TP_SIZE=4; HETERO="3,4,4,4"; DSA_CP=1; SP=1 ;;
        prefill:dp2_tp43_no_sp_no_dsa)
            DP_SIZE=2; TP_SIZE=4; HETERO="4,3"; DSA_CP=0; SP=0 ;;
        prefill:dp4_tp3444_no_sp_no_dsa)
            DP_SIZE=4; TP_SIZE=4; HETERO="3,4,4,4"; DSA_CP=0; SP=0 ;;
        prefill:dp2_tp43_no_dsa_sp)
            DP_SIZE=2; TP_SIZE=4; HETERO="4,3"; DSA_CP=0; SP=1 ;;
        prefill:dp4_tp3444_no_dsa_sp)
            DP_SIZE=4; TP_SIZE=4; HETERO="3,4,4,4"; DSA_CP=0; SP=1 ;;
        decode:dp4_tp2122)
            DP_SIZE=4; TP_SIZE=2; HETERO="2,1,2,2"; DSA_CP=0; SP=0 ;;
        decode:dp4_tp2212)
            DP_SIZE=4; TP_SIZE=2; HETERO="2,2,1,2"; DSA_CP=0; SP=0 ;;
        prefill:custom)
            DP_SIZE=${DP_SIZE:-2}; TP_SIZE=${TP_SIZE:-4}
            HETERO=${HETERO_TP_SIZES:-4,3}
            DSA_CP=${DSA_CP:-1}; SP=${SP:-1}
            ;;
        decode:custom)
            DP_SIZE=${DP_SIZE:-4}; TP_SIZE=${TP_SIZE:-2}
            HETERO=${HETERO_TP_SIZES:-2,1,2,2}
            DSA_CP=${DSA_CP:-0}; SP=${SP:-0}
            ;;
        *)
            echo "unknown case for role $ROLE: $MODE" >&2
            list_cases >&2
            exit 2
            ;;
    esac
}

if [ "$MODE" = "list" ]; then
    list_cases
    exit 0
fi

case "$ROLE" in
    prefill|decode) ;;
    *) echo "role must be prefill or decode" >&2; exit 2 ;;
esac

if [ "$MODE" = "all" ]; then
    failed_cases=""
    log "按顺序自动执行 $ROLE 全部内置用例: $(list_cases)"
    for single_case in $(list_cases); do
        log "========== 开始执行 $ROLE/$single_case =========="
        if KEEP_RUNNING=0 bash "$SCRIPT_DIR/single_instance_test.sh" \
            "$ROLE" "$single_case"; then
            log "========== $ROLE/$single_case 通过 =========="
        else
            log "========== $ROLE/$single_case 失败 =========="
            failed_cases="$failed_cases $single_case"
        fi
    done
    if [ -n "$failed_cases" ]; then
        log "以下用例失败:$failed_cases"
        exit 1
    fi
    log "$ROLE 全部内置用例通过。"
    exit 0
fi

set_case_vars

if ! validate_tp "$HETERO" "$TP_SIZE"; then
    echo "不支持的 TP 配置: tp=$TP_SIZE hetero='$HETERO'。"
    echo "当前实现限制 TP <= o_groups($MAX_TP)，tp16/tp15 等场景会在此报错。"
    exit 1
fi

if [ -n "$HETERO" ]; then
    CARDS=$(sum_hetero_cards "$HETERO")
else
    CARDS=$((DP_SIZE * TP_SIZE))
fi
if [ "$CARDS" -gt "$NPU_TOTAL" ]; then
    echo "需要 ${CARDS} 张 NPU，超过 NPU_TOTAL=${NPU_TOTAL}。" >&2
    exit 1
fi
if [ "$CARDS" -lt "$MIN_TEST_CARDS" ]; then
    echo "测试用例需要至少 ${MIN_TEST_CARDS} 张 NPU，当前只有 ${CARDS} 张。" >&2
    exit 1
fi

OUTDIR="$RESULT_ROOT/${ROLE}_${MODE}"
mkdir -p "$OUTDIR"
LOCAL_MAX_TP=$(max_tp "$HETERO" "$TP_SIZE")
log "单实例测试: role=$ROLE case=$MODE (dp=$DP_SIZE tp=$TP_SIZE "
log "hetero='$HETERO' dsa_cp=$DSA_CP sp=$SP cards=$CARDS)"

cleanup

(
    if [ "$ROLE" = "prefill" ]; then
        export PREFILL_LOCAL_IP="$LOCAL_IP"
        export PREFILL_DP_ADDRESS="$LOCAL_IP"
        export PREFILL_DP_RPC_PORT="${DP_RPC_PORT:-12321}"
        export PREFILL_VLLM_START_PORT="$PORT_BASE"
        export PREFILL_KV_PORT="${KV_PORT:-36000}"
        export PREFILL_DEVICE_START=0
        cd "$SCRIPT_DIR/prefill" || exit 1
        bash ./start_server.sh "$DP_SIZE" "$TP_SIZE" "$HETERO" \
            "$REMOTE_DP" "$REMOTE_DECODE_TP" "$DSA_CP" "$SP"
    else
        # decode 单实例没有对端，但 connector 仍要求 prefill_tp >= decode_tp，
        # 因此远端 prefill 池描述符默认取本实例最大 TP。
        single_remote_prefill_tp="${REMOTE_PREFILL_TP:-$LOCAL_MAX_TP}"
        export DECODE_LOCAL_IP="$LOCAL_IP"
        export DECODE_DP_ADDRESS="$LOCAL_IP"
        export DECODE_DP_RPC_PORT="${DP_RPC_PORT:-12322}"
        export DECODE_VLLM_START_PORT="$PORT_BASE"
        export DECODE_KV_PORT="${KV_PORT:-36200}"
        export DECODE_DEVICE_START=0
        cd "$SCRIPT_DIR/decode" || exit 1
        bash ./start_server.sh "$DP_SIZE" "$TP_SIZE" "$HETERO" \
            "$REMOTE_DP" "$single_remote_prefill_tp" "$DSA_CP" "$SP"
    fi
) >"$INSTANCE_LOG" 2>&1 &

log "实例后台日志: $INSTANCE_LOG"

if ! wait_for_ports "$LOCAL_IP" "$PORT_BASE" "$DP_SIZE" \
                   "$STARTUP_TIMEOUT" "$ROLE"; then
    log "实例就绪失败，查看 $INSTANCE_LOG"
    exit 1
fi

BACKENDS=""
for ((i = 0; i < DP_SIZE; i++)); do
    if [ -n "$BACKENDS" ]; then
        BACKENDS="$BACKENDS,"
    fi
    BACKENDS="${BACKENDS}http://${LOCAL_IP}:$((PORT_BASE + i))"
done

log "向后端发送请求: $BACKENDS"
python3 "$SCRIPT_DIR/request_hetero_test.py" \
    --target-urls "$BACKENDS" \
    --num-requests "$REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --outdir "$OUTDIR"

cp -f "$INSTANCE_LOG" "$OUTDIR/instance.log" 2>/dev/null || true
log "结果目录: $OUTDIR"
log "单实例测试完成。"
