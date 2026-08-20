#!/bin/bash
# 一键自动化异构 PD 测试。
#
# 功能:
#   1. 在 76 节点拉起 prefill，75 节点拉起 decode（通过 ssh + docker exec）。
#   2. 轮询两侧所有 vllm engine 端口（默认 7100 起，DP 数决定端口数），
#      端口可访问且 /health 返回 200 后认为服务就绪。
#   3. 在 76 节点拉起 proxy.sh，并用 request_hetero_test.py 并行发送中文请求。
#   4. 每个场景的结果保存到 <RESULT_ROOT>/<case>/，并抓取两侧容器日志。
#
# 用法:
#   ./automated_hetero_test.sh all                 # 运行全部场景
#   ./automated_hetero_test.sh hetero_baseline     # 只运行指定场景
#
# 常用环境变量:
#   NODE_PREFILL / NODE_DECODE / CONTAINER / HETERO_DIR / RESULT_ROOT
#   STARTUP_TIMEOUT（单侧服务就绪超时秒，默认 1800）
#   KEEP_RUNNING=1（全部结束后不清理服务）

set -u

NODE_PREFILL=${NODE_PREFILL:-7.246.78.76}
NODE_DECODE=${NODE_DECODE:-7.246.78.75}
CONTAINER=${CONTAINER:-vllm_v23_30055003}
HETERO_DIR=${HETERO_DIR:-/opt/its/z30055003/hetero}
RESULT_ROOT=${RESULT_ROOT:-./hetero_test_results}
VLLM_PORT_BASE=${VLLM_PORT_BASE:-7100}
PROXY_PORT=${PROXY_PORT:-9000}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}
REQUESTS_PER_CASE=${REQUESTS_PER_CASE:-32}
REQUEST_CONCURRENCY=${REQUEST_CONCURRENCY:-8}
KEEP_RUNNING=${KEEP_RUNNING:-0}
MIN_TEST_CARDS=${MIN_TEST_CARDS:-7}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=${1:-all}

log() { echo "[$(date '+%F %T')] $*"; }

stop_prefill() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NODE_PREFILL" \
        "docker exec $CONTAINER bash -lc 'cd $HETERO_DIR && bash ./remote_ctl.sh stop-prefill'" \
        >/dev/null 2>&1 || true
}
stop_decode() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NODE_DECODE" \
        "docker exec $CONTAINER bash -lc 'cd $HETERO_DIR && bash ./remote_ctl.sh stop-decode'" \
        >/dev/null 2>&1 || true
}
stop_proxy() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NODE_PREFILL" \
        "docker exec $CONTAINER bash -lc 'cd $HETERO_DIR && bash ./remote_ctl.sh stop-proxy'" \
        >/dev/null 2>&1 || true
}
stop_all() {
    log "停止两侧残留服务..."
    stop_proxy
    stop_prefill
    stop_decode
    sleep 3
}

start_prefill() {
    local pref_dp="$1" pref_tp="$2" pref_hetero="${3:-@EMPTY@}" \
          dec_dp="$4" dec_tp="$5" pref_dsa="$6" pref_sp="$7"
    local args="$pref_dp $pref_tp $pref_hetero $dec_dp $dec_tp $pref_dsa $pref_sp"
    local log_file="/tmp/hetero_test_prefill.log"
    local inner="cd $HETERO_DIR && bash ./remote_ctl.sh start-prefill $args > $log_file 2>&1"
    log "在 ${NODE_PREFILL} 拉起 prefill: $args"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NODE_PREFILL" \
        "docker exec -d $CONTAINER bash -lc '$inner'" || {
        log "启动 prefill 失败"; return 1
    }
}
start_decode() {
    local dec_dp="$1" dec_tp="$2" dec_hetero="${3:-@EMPTY@}" \
          pref_dp="$4" pref_tp="$5" dec_dsa="$6" dec_sp="$7"
    local args="$dec_dp $dec_tp $dec_hetero $pref_dp $pref_tp $dec_dsa $dec_sp"
    local log_file="/tmp/hetero_test_decode.log"
    local inner="cd $HETERO_DIR && bash ./remote_ctl.sh start-decode $args > $log_file 2>&1"
    log "在 ${NODE_DECODE} 拉起 decode: $args"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NODE_DECODE" \
        "docker exec -d $CONTAINER bash -lc '$inner'" || {
        log "启动 decode 失败"; return 1
    }
}
start_proxy() {
    local prefill_dp="$1" decode_dp="$2"
    local log_file="/tmp/hetero_test_proxy.log"
    local inner="cd $HETERO_DIR && PREFILL_DP_SIZE=$prefill_dp DECODE_DP_SIZE=$decode_dp bash ./proxy.sh > $log_file 2>&1"
    log "在 ${NODE_PREFILL} 拉起 proxy (P_DP=$prefill_dp, D_DP=$decode_dp)"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NODE_PREFILL" \
        "docker exec -d $CONTAINER bash -lc '$inner'" || {
        log "启动 proxy 失败"; return 1
    }
}

sum_hetero_cards() {
    local sizes="$1" total=0 size
    IFS=',' read -ra parts <<<"$sizes"
    for size in "${parts[@]}"; do
        total=$((total + size))
    done
    echo "$total"
}

wait_for_services() {
    local host="$1" port_base="$2" port_count="$3" timeout="$4" label="$5"
    python3 - "$host" "$port_base" "$port_count" "$timeout" "$label" <<'PY'
import http.client
import sys
import time

host = sys.argv[1]
port_base = int(sys.argv[2])
port_count = int(sys.argv[3])
timeout = int(sys.argv[4])
label = sys.argv[5]
deadline = time.time() + timeout
remaining = set(range(port_base, port_base + port_count))
print(f"[{label}] waiting for {port_count} ports on {host} "
      f"({port_base}..{port_base + port_count - 1})", flush=True)
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
print(f"[{label}] timeout after {timeout}s, still waiting: "
      f"{sorted(remaining)}", flush=True)
sys.exit(1)
PY
}

wait_for_port() {
    local host="$1" port="$2" timeout="$3" label="$4"
    python3 - "$host" "$port" "$timeout" "$label" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = int(sys.argv[3])
label = sys.argv[4]
deadline = time.time() + timeout
print(f"[{label}] waiting for {host}:{port}", flush=True)
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=3):
            print(f"[{label}] {host}:{port} ready", flush=True)
            sys.exit(0)
    except OSError:
        time.sleep(5)
print(f"[{label}] timeout waiting for {host}:{port}", flush=True)
sys.exit(1)
PY
}

collect_log() {
    local node="$1" log_file="$2" output="$3"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$node" \
        "docker exec $CONTAINER bash -lc 'cat $log_file 2>/dev/null || true'" \
        >"$output" 2>/dev/null || true
    [ -s "$output" ] || echo "(no log captured)" >"$output"
}

collect_case_logs() {
    local outdir="$1"
    mkdir -p "$outdir"
    collect_log "$NODE_PREFILL" /tmp/hetero_test_prefill.log "$outdir/prefill.log"
    collect_log "$NODE_DECODE" /tmp/hetero_test_decode.log "$outdir/decode.log"
    collect_log "$NODE_PREFILL" /tmp/hetero_test_proxy.log "$outdir/proxy.log"
}

run_case() {
    local name="$1" pref_dp="$2" pref_tp="$3" pref_hetero="$4" \
          dec_dp="$5" dec_tp="$6" pref_dsa="$7" pref_sp="$8"
    local outdir="$RESULT_ROOT/$name"
    local pref_cards dec_cards total_cards
    if [ -n "$pref_hetero" ]; then
        pref_cards=$(sum_hetero_cards "$pref_hetero")
    else
        pref_cards=$((pref_dp * pref_tp))
    fi
    dec_cards=$((dec_dp * dec_tp))
    total_cards=$((pref_cards + dec_cards))
    if [ "$total_cards" -lt "$MIN_TEST_CARDS" ]; then
        log "场景 $name 只有 ${total_cards} 张卡，少于 MIN_TEST_CARDS=${MIN_TEST_CARDS}，跳过。"
        return 1
    fi
    mkdir -p "$outdir"
    log "========== 开始场景: $name (共 ${total_cards} 卡) =========="
    log "prefill dp=$pref_dp tp=$pref_tp hetero='$pref_hetero' dsa_cp=$pref_dsa sp=$pref_sp"
    log "decode  dp=$dec_dp tp=$dec_tp dsa_cp=0 sp=0"

    stop_all

    if ! start_prefill "$pref_dp" "$pref_tp" "$pref_hetero" \
                      "$dec_dp" "$dec_tp" "$pref_dsa" "$pref_sp"; then
        collect_case_logs "$outdir"
        return 1
    fi
    if ! start_decode "$dec_dp" "$dec_tp" "" "$pref_dp" "$pref_tp" 0 0; then
        collect_case_logs "$outdir"
        return 1
    fi

    if ! wait_for_services "$NODE_PREFILL" "$VLLM_PORT_BASE" "$pref_dp" \
                          "$STARTUP_TIMEOUT" "prefill"; then
        collect_case_logs "$outdir"
        return 1
    fi
    if ! wait_for_services "$NODE_DECODE" "$VLLM_PORT_BASE" "$dec_dp" \
                          "$STARTUP_TIMEOUT" "decode"; then
        collect_case_logs "$outdir"
        return 1
    fi

    if ! start_proxy "$pref_dp" "$dec_dp"; then
        collect_case_logs "$outdir"
        return 1
    fi
    if ! wait_for_port "$NODE_PREFILL" "$PROXY_PORT" 120 "proxy"; then
        collect_case_logs "$outdir"
        return 1
    fi

    log "发送 ${REQUESTS_PER_CASE} 个并发中文请求..."
    if ! python3 "$SCRIPT_DIR/request_hetero_test.py" \
        --proxy-url "http://${NODE_PREFILL}:${PROXY_PORT}" \
        --num-requests "$REQUESTS_PER_CASE" \
        --concurrency "$REQUEST_CONCURRENCY" \
        --outdir "$outdir"; then
        log "场景 $name 存在失败请求，继续收集日志。"
    fi

    collect_case_logs "$outdir"
    log "场景 $name 结果目录: $outdir"
    log "========== 场景 $name 完成 =========="
}

# 场景矩阵（| 分隔，pref_hetero 为空表示统一 TP）:
#   name|pref_dp|pref_tp|pref_hetero|dec_dp|dec_tp|pref_dsa_cp|pref_sp
CASES=(
    "hetero_baseline|4|4|3,4,4,4|16|1|1|1"
    "hetero_no_sp_no_dsa|4|4|3,4,4,4|16|1|0|0"
    "tp43_no_sp_no_dsa|2|4|4,3|8|1|0|0"
    "hetero_no_dsa_sp|4|4|3,4,4,4|16|1|0|1"
    "tp43_no_dsa_sp|2|4|4,3|8|1|0|1"
    "pure_dp_no_sp_no_dsa|15|1||16|1|0|0"
    "tp4_dsa_cp_sp|4|4||8|2|1|1"
    "tp4_no_dsa_cp_sp|4|4||8|2|0|1"
    "tp4_no_dsa_cp_no_sp|4|4||8|2|0|0"
    "odd_tp_dsa_cp_sp|1|3|3|16|1|1|1"
)

run_all() {
    local failed=0
    for entry in "${CASES[@]}"; do
        IFS='|' read -r name pref_dp pref_tp pref_hetero \
            dec_dp dec_tp pref_dsa pref_sp <<<"$entry"
        if ! run_case "$name" "$pref_dp" "$pref_tp" "$pref_hetero" \
                      "$dec_dp" "$dec_tp" "$pref_dsa" "$pref_sp"; then
            log "场景 $name 测试失败。"
            failed=1
        fi
    done
    return "$failed"
}

trap 'if [ "$KEEP_RUNNING" != "1" ]; then stop_all; fi' EXIT

case "$MODE" in
    all)
        run_all
        ;;
    list)
        for entry in "${CASES[@]}"; do
            echo "${entry%%|*}"
        done
        ;;
    *)
        found=0
        for entry in "${CASES[@]}"; do
            IFS='|' read -r name pref_dp pref_tp pref_hetero \
                dec_dp dec_tp pref_dsa pref_sp <<<"$entry"
            if [ "$name" = "$MODE" ]; then
                found=1
                run_case "$name" "$pref_dp" "$pref_tp" "$pref_hetero" \
                         "$dec_dp" "$dec_tp" "$pref_dsa" "$pref_sp"
            fi
        done
        if [ "$found" != "1" ]; then
            echo "unknown case: $MODE (use 'list' to show cases)" >&2
            exit 2
        fi
        ;;
esac
