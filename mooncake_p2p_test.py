#!/usr/bin/env python3
"""Minimal two-node Mooncake/Ascend P2P communication test.

This mirrors exactly what the PD connector does on the decode side:
``TransferEngine`` with ``P2PHANDSHAKE`` + ``ascend`` protocol, one device
buffer registered on each node, and ``batch_transfer_sync_read`` pulling
from the remote node.  After a successful read the client writes its own
pattern back so the server can also verify the reverse direction.

Run it inside the same environment as ``run_dp_template.sh`` (same
``ASCEND_RT_VISIBLE_DEVICES``, ``HCCL_*`` and ``ASCEND_*`` exports).

Usage
-----
Server (prefill node, 7.246.78.76):
    python mooncake_p2p_test.py --role server --ip 7.246.78.76
    # prints: SERVER_INFO RPC_PORT=20123 REMOTE_ADDR=... SIZE=4194304
    # then waits until the client writes the client pattern.

Client (decode node, 7.246.78.75); copy RPC_PORT and REMOTE_ADDR from the
server output:
    python mooncake_p2p_test.py --role client --ip 7.246.78.75 \
        --peer 7.246.78.76:20123 --remote-addr <REMOTE_ADDR> \
        --size 4194304

Both sides print ``P2P_TEST_OK`` and exit 0 on success.
"""

import argparse
import os
import sys
import time

SERVER_MAGIC = 0xA5
CLIENT_MAGIC = 0x5A
DEFAULT_SIZE = 4 * 1024 * 1024  # 4 MiB, small but exercises a real RDMA transfer


def _set_test_env_defaults() -> None:
    # Keep the same timeout semantics as the PD scripts.  MC_TRANSFER_TIMEOUT
    # must be set before the TransferEngine object is constructed.
    os.environ.setdefault("ASCEND_CONNECT_TIMEOUT", "30000")
    os.environ.setdefault("ASCEND_TRANSFER_TIMEOUT", "30000")
    os.environ.setdefault("MC_TRANSFER_TIMEOUT", "180")


_set_test_env_defaults()

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401
from mooncake.engine import TransferEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("server", "client"), required=True)
    parser.add_argument("--ip", required=True, help="Local IP used by TransferEngine")
    parser.add_argument(
        "--peer",
        default="",
        help="Client only: remote engine address, e.g. 7.246.78.76:20123",
    )
    parser.add_argument(
        "--remote-addr",
        type=lambda value: int(value, 0),
        default=0,
        help="Client only: remote registered buffer address printed by the server",
    )
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--device", type=int, default=None, help="Visible device index")
    return parser.parse_args()


def select_device(device_index: int | None) -> int:
    visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "0").split(",")[0]
    selected = int(visible) if device_index is None else device_index
    torch.npu.set_device(selected)
    return selected


def make_engine(local_ip: str) -> TransferEngine:
    engine = TransferEngine()
    ret = engine.initialize(local_ip, "P2PHANDSHAKE", "ascend", "")
    if ret != 0:
        raise RuntimeError(f"TransferEngine.initialize failed, ret={ret}")
    return engine


def register_buffer(
    engine: TransferEngine, size: int, fill: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    # HCCS requires device-memory page-table alignment (2 MiB).  torch's
    # allocator does not guarantee it, so over-allocate and take an aligned
    # view; keep the raw tensor alive for the lifetime of the test.
    alignment = 2 * 1024 * 1024
    raw = torch.empty(size + alignment, dtype=torch.uint8, device="npu")
    offset = (-raw.data_ptr()) % alignment
    tensor = raw.narrow(0, offset, size)
    tensor.fill_(fill)
    torch.npu.synchronize()
    addr = tensor.data_ptr()
    assert addr % alignment == 0
    ret = engine.register_memory(addr, size)
    if ret != 0:
        raise RuntimeError(f"register_memory failed for {addr:#x}, ret={ret}")
    print(f"Registered buffer addr={addr:#x} size={size}")
    return tensor, raw, addr


def run_server(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    engine = make_engine(args.ip)
    rpc_port = engine.get_rpc_port()
    tensor, _raw, addr = register_buffer(engine, args.size, SERVER_MAGIC)
    print(
        f"SERVER_INFO DEVICE={device} RPC_PORT={rpc_port} "
        f"REMOTE_ADDR={addr:#x} SIZE={args.size}",
        flush=True,
    )
    print("Waiting for the client to write CLIENT_MAGIC into the remote buffer...", flush=True)

    deadline = time.time() + 600
    while time.time() < deadline:
        if int(tensor[0].item()) == CLIENT_MAGIC:
            break
        time.sleep(1)
    else:
        print("SERVER_TIMEOUT: remote buffer was never updated", file=sys.stderr)
        sys.exit(1)

    torch.npu.synchronize()
    if not bool(torch.all(tensor.eq(CLIENT_MAGIC)).item()):
        bad = int((tensor != CLIENT_MAGIC).sum().item())
        print(f"SERVER_CHECK_FAILED: {bad} bytes do not match CLIENT_MAGIC", file=sys.stderr)
        sys.exit(1)

    engine.unregister_memory(addr)
    print("P2P_TEST_OK server: read+write verified")


def run_client(args: argparse.Namespace) -> None:
    if not args.peer or args.remote_addr == 0:
        print(
            "--peer and --remote-addr are required in client mode "
            "(copy them from the server output)",
            file=sys.stderr,
        )
        sys.exit(2)

    device = select_device(args.device)
    engine = make_engine(args.ip)
    tensor, _raw, local_addr = register_buffer(engine, args.size, CLIENT_MAGIC)

    print(
        f"Client device={device} local_addr={local_addr:#x} "
        f"remote={args.peer} remote_addr={args.remote_addr:#x}",
        flush=True,
    )

    # Same call direction as decode pulling prefill KV: remote -> local.
    ret = engine.batch_transfer_sync_read(
        args.peer, [local_addr], [args.remote_addr], [args.size]
    )
    if ret < 0:
        print(f"batch_transfer_sync_read failed, ret={ret}", file=sys.stderr)
        sys.exit(1)

    torch.npu.synchronize()
    if not bool(torch.all(tensor.eq(SERVER_MAGIC)).item()):
        bad = int((tensor != SERVER_MAGIC).sum().item())
        print(f"CLIENT_READ_CHECK_FAILED: {bad} bytes do not match SERVER_MAGIC", file=sys.stderr)
        sys.exit(1)
    print("CLIENT_READ_OK: received SERVER_MAGIC from remote buffer")

    # Exercise the reverse direction as well: local -> remote.
    tensor.fill_(CLIENT_MAGIC)
    torch.npu.synchronize()
    ret = engine.batch_transfer_sync_write(
        args.peer, [local_addr], [args.remote_addr], [args.size]
    )
    if ret < 0:
        print(f"batch_transfer_sync_write failed, ret={ret}", file=sys.stderr)
        sys.exit(1)

    engine.unregister_memory(local_addr)
    print("P2P_TEST_OK client: read+write verified")


def main() -> None:
    args = parse_args()
    print(
        "TEST_ENV HCCL_IF_IP=%s HCCL_SOCKET_IFNAME=%s "
        "ASCEND_RT_VISIBLE_DEVICES=%s ASCEND_CONNECT_TIMEOUT=%s "
        "ASCEND_TRANSFER_TIMEOUT=%s MC_TRANSFER_TIMEOUT=%s",
        os.getenv("HCCL_IF_IP", ""),
        os.getenv("HCCL_SOCKET_IFNAME", ""),
        os.getenv("ASCEND_RT_VISIBLE_DEVICES", ""),
        os.getenv("ASCEND_CONNECT_TIMEOUT", ""),
        os.getenv("ASCEND_TRANSFER_TIMEOUT", ""),
        os.getenv("MC_TRANSFER_TIMEOUT", ""),
    )
    if args.role == "server":
        run_server(args)
    else:
        run_client(args)


if __name__ == "__main__":
    main()
