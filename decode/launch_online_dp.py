import argparse
import json
import multiprocessing
import os
import subprocess
import sys

# DeepSeek-V4 attention projection groups. Used to derive group-aligned
# ``tp_sharding_ratios`` for per-DP TP sizes that do not divide the model's
# 64 heads / 8 o_groups evenly (e.g. tp=3 -> [3, 3, 2]).
_DSV4_O_GROUPS = 8


def auto_tp_sharding_ratios(tp_size: int) -> list[int] | None:
    """Return group-aligned sharding ratios for one DP rank, or None.

    Only TP sizes that can shard DeepSeek-V4's 8 o_groups are supported.
    ``tp_size > o_groups`` is rejected because DSA-CP's o_proj sharding
    cannot represent more TP ranks than projection groups.
    """
    if tp_size > _DSV4_O_GROUPS:
        raise ValueError(
            f"DeepSeek-V4 does not support tp_size={tp_size} > "
            f"o_groups={_DSV4_O_GROUPS}."
        )
    if tp_size <= 1:
        return None
    if tp_size == 3:
        # Keep the validated heterogeneous-TP baseline layout
        # (50% / 25% / 25% for 64 heads and 8 o_groups).
        return [2, 1, 1]
    if _DSV4_O_GROUPS % tp_size == 0:
        return None
    base, remainder = divmod(_DSV4_O_GROUPS, tp_size)
    return [base + 1] * remainder + [base] * (tp_size - remainder)


def build_heterogeneous_dp_config(per_dp_tp_sizes: list[int]) -> list[dict]:
    config = []
    for dp_rank, tp_size in enumerate(per_dp_tp_sizes):
        entry: dict = {"dp_rank": dp_rank, "tp_size": tp_size}
        ratios = auto_tp_sharding_ratios(tp_size)
        if ratios is not None:
            entry["tp_sharding_ratios"] = ratios
        config.append(entry)
    return config


def build_kv_transfer_config(
    role: str,
    kv_port: int,
    engine_id: str,
    prefill_dp_size: int,
    prefill_tp_size: int,
    decode_dp_size: int,
    decode_tp_size: int,
) -> dict:
    return {
        "kv_connector": "MooncakeHybridConnector",
        "kv_role": role,
        "kv_port": kv_port,
        "engine_id": engine_id,
        "kv_connector_extra_config": {
            "prefill": {
                "dp_size": prefill_dp_size,
                "tp_size": prefill_tp_size,
            },
            "decode": {
                "dp_size": decode_dp_size,
                "tp_size": decode_tp_size,
            },
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True,
                        help="Data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size for the uniform DP/TP case.")
    parser.add_argument(
        "--hetero-tp-sizes", type=str, default=None,
        help=("Comma-separated per-DP tensor parallel sizes. When set, "
              "--heterogeneous-dp-config is generated automatically."),
    )
    parser.add_argument("--dp-size-local", type=int, default=-1,
                        help="Local data parallel size.")
    parser.add_argument("--dp-rank-start", type=int, default=0,
                        help="Starting rank for data parallel.")
    parser.add_argument("--device-start", type=int, default=0,
                        help="Physical device id offset for this service.")
    parser.add_argument("--dp-address", type=str, required=True,
                        help="IP address for data parallel master node.")
    parser.add_argument("--dp-rpc-port", type=str, default=12345,
                        help="Port for data parallel master node.")
    parser.add_argument("--vllm-start-port", type=int, default=9000,
                        help="Starting port for the engine.")
    parser.add_argument("--kv-role", type=str, default="kv_consumer",
                        choices=("kv_producer", "kv_consumer"),
                        help="KV connector role of this node.")
    parser.add_argument("--kv-port", type=int, default=36200,
                        help="Base handshake port of this node.")
    parser.add_argument("--engine-id", type=str, default="1",
                        help="KV connector engine id.")
    parser.add_argument("--prefill-dp-size", type=int, default=4,
                        help="Remote prefill pool DP size.")
    parser.add_argument("--prefill-tp-size", type=int, default=4,
                        help=("Remote prefill pool logical TP size. For "
                              "heterogeneous prefill this is the pool "
                              "descriptor (e.g. max per-DP tp)."))
    parser.add_argument("--decode-dp-size", type=int, default=None,
                        help="Remote decode pool DP size (defaults to --dp-size).")
    parser.add_argument("--decode-tp-size", type=int, default=None,
                        help=("Remote decode pool logical TP size. For "
                              "heterogeneous decode this is the pool "
                              "descriptor (e.g. max per-DP tp)."))
    parser.add_argument("--enable-dsa-cp", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Enable DSA context parallelism (default: off).")
    parser.add_argument("--enable-sp", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Enable FlashComm1 sequence parallelism (default: off).")
    return parser.parse_args()


args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
hetero_tp_sizes = args.hetero_tp_sizes
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
device_start = args.device_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port

if args.enable_dsa_cp and not args.enable_sp:
    print("--enable-dsa-cp requires --enable-sp for DeepSeek-V4.")
    sys.exit(1)
if args.enable_sp:
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "1"
    os.environ["HETERO_ENABLE_SP"] = "1"
else:
    os.environ.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
    os.environ["HETERO_ENABLE_SP"] = "0"

additional_config = {
    "ascend_compilation_config": {
        "enable_npugraph_ex": True,
        "enable_static_kernel": False,
    },
    "enable_cpu_binding": True,
    "multistream_overlap_shared_expert": True,
    "recompute_scheduler_enable": True,
}
if args.enable_dsa_cp:
    additional_config["enable_dsa_cp"] = True
os.environ["ADDITIONAL_CONFIG_JSON"] = json.dumps(additional_config)

if hetero_tp_sizes is not None:
    per_dp_tp_sizes = [int(x) for x in hetero_tp_sizes.split(",")]
    if len(per_dp_tp_sizes) != dp_size:
        print(
            f"--hetero-tp-sizes length ({len(per_dp_tp_sizes)}) must equal "
            f"--dp-size ({dp_size})."
        )
        sys.exit(1)
    if any(size > _DSV4_O_GROUPS for size in per_dp_tp_sizes):
        print(
            "DeepSeek-V4 does not support TP sizes greater than "
            f"o_groups={_DSV4_O_GROUPS}; got {per_dp_tp_sizes}."
        )
        sys.exit(1)
    hetero_config = build_heterogeneous_dp_config(per_dp_tp_sizes)
    os.environ["HETERO_DP_CONFIG_JSON"] = json.dumps(hetero_config)
    logical_pool_tp_size = max(per_dp_tp_sizes)
else:
    per_dp_tp_sizes = [tp_size] * dp_size
    os.environ.pop("HETERO_DP_CONFIG_JSON", None)
    logical_pool_tp_size = tp_size

unsupported_tp_sizes = [
    size for size in per_dp_tp_sizes if size > _DSV4_O_GROUPS
]
if unsupported_tp_sizes:
    print(
        "DeepSeek-V4 does not support TP sizes greater than "
        f"o_groups={_DSV4_O_GROUPS}; got {sorted(set(unsupported_tp_sizes))}."
    )
    sys.exit(1)

decode_dp_size = args.decode_dp_size
if decode_dp_size is None:
    decode_dp_size = dp_size
decode_tp_size = args.decode_tp_size
if decode_tp_size is None:
    decode_tp_size = logical_pool_tp_size

os.environ["KV_TRANSFER_CONFIG_JSON"] = json.dumps(
    build_kv_transfer_config(
        role=args.kv_role,
        kv_port=args.kv_port,
        engine_id=args.engine_id,
        prefill_dp_size=args.prefill_dp_size,
        prefill_tp_size=args.prefill_tp_size,
        decode_dp_size=decode_dp_size,
        decode_tp_size=decode_tp_size,
    )
)


def run_command(visible_devices, dp_rank, vllm_engine_port, rank_tp_size):
    command = [
        "bash",
        "./run_dp_template.sh",
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(rank_tp_size),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    template_path = "./run_dp_template.sh"
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        sys.exit(1)

    processes = []
    device_offset = device_start + sum(per_dp_tp_sizes[:dp_rank_start])
    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        rank_tp_size = per_dp_tp_sizes[dp_rank]
        vllm_engine_port = vllm_start_port + i
        visible_devices = ",".join(
            str(x) for x in range(device_offset, device_offset + rank_tp_size)
        )
        device_offset += rank_tp_size
        process = multiprocessing.Process(
            target=run_command,
            name=f"dp{dp_rank}",
            args=(visible_devices, dp_rank, vllm_engine_port, rank_tp_size),
        )
        processes.append(process)
        process.start()

    failed = False
    for process in processes:
        process.join()
        if process.exitcode not in (0, None):
            print(
                f"DP engine process {process.name} failed with "
                f"exit code {process.exitcode}."
            )
            failed = True
    if failed:
        sys.exit(1)
