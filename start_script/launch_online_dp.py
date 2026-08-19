import argparse
import multiprocessing
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dp-size",
        type=int,
        required=True,
        help="Data parallel size."
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for the uniform DP/TP case."
    )
    parser.add_argument(
        "--hetero-tp-sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated per-DP tensor parallel sizes for heterogeneous "
            "DP/TP, e.g. 3,4,4,4. Must match --heterogeneous-dp-config in "
            "run_dp_template.sh."
        ),
    )
    parser.add_argument(
        "--dp-size-local",
        type=int,
        default=-1,
        help="Local data parallel size."
    )
    parser.add_argument(
        "--dp-rank-start",
        type=int,
        default=0,
        help="Starting rank for data parallel."
    )
    parser.add_argument(
        "--dp-address",
        type=str,
        required=True,
        help="IP address for data parallel master node."
    )
    parser.add_argument(
        "--dp-rpc-port",
        type=str,
        default=12345,
        help="Port for data parallel master node."
    )
    parser.add_argument(
        "--vllm-start-port",
        type=int,
        default=9000,
        help="Starting port for the engine."
    )
    return parser.parse_args()

args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
hetero_tp_sizes = args.hetero_tp_sizes
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port

if hetero_tp_sizes is not None:
    per_dp_tp_sizes = [int(x) for x in hetero_tp_sizes.split(",")]
    if len(per_dp_tp_sizes) != dp_size:
        print(
            f"--hetero-tp-sizes length ({len(per_dp_tp_sizes)}) must equal "
            f"--dp-size ({dp_size})."
        )
        sys.exit(1)
else:
    per_dp_tp_sizes = [tp_size] * dp_size

total_cards = sum(per_dp_tp_sizes)
local_cards = sum(
    per_dp_tp_sizes[dp_rank_start : dp_rank_start + dp_size_local]
)
print(
    f"Heterogeneous DP/TP launcher: per_dp_tp_sizes={per_dp_tp_sizes}, "
    f"total_cards={total_cards}, local_cards={local_cards}, "
    f"local_dp_ranks={list(range(dp_rank_start, dp_rank_start + dp_size_local))}"
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
    device_offset = sum(per_dp_tp_sizes[:dp_rank_start])
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
            args=(visible_devices, dp_rank, vllm_engine_port, rank_tp_size),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
