# 异构 PD 启动脚本

`run_scenario.sh` 覆盖三种部署：

| 场景 | prefill | decode | 说明 |
|---|---|---|---|
| 2 | dp15 tp1 | dp16 tp1 | 15P + 16D，prefill 纯 DP |
| 3 | dp4 tp4 | dp8 tp2 | 16P + 16D，prefill/decode 均为 TP×DP |
| hetero | dp4 tp(3,4,4,4) | dp16 tp1 | 原异构基线（15P + 16D） |

> 场景1（prefill dp1tp15）依赖 DSA-CP `tp>8` 支持，已按需求回退，
> 不再提供。DeepSeek-V4 单 DP TP 必须 `<= o_groups(8)`。

prefill 与 decode 在两个节点上分别执行：

```bash
# prefill 节点
./run_scenario.sh 2 prefill

# decode 节点
./run_scenario.sh 2 decode

# 代理节点
./run_scenario.sh 2 proxy
```

等价的手动命令：

```bash
# 场景2
prefill/start_server.sh 15 1 "" 16 1
decode/start_server.sh 16 1 "" 15 1

# 场景3
prefill/start_server.sh 4 4 "" 8 2
decode/start_server.sh 8 2 "" 4 4

# 原异构基线
prefill/start_server.sh 4 4 3,4,4,4 16 1
decode/start_server.sh 16 1 "" 4 4
```

## 参数说明

`prefill/start_server.sh <dp> <tp> [hetero_tp_sizes] <decode_dp> <decode_tp>`

- 统一 TP 时 `hetero_tp_sizes` 传空字符串。
- 异构 TP 时传逗号分隔的 per-DP TP，例如 `3,4,4,4`。launcher 会自动生成
  `--heterogeneous-dp-config`；对不能整除 8 个 o_groups 的 TP（如 3）自动
  生成 group 对齐的 `tp_sharding_ratios`（tp=3 保持基线 `[2,1,1]`，
  其他值使用平衡拆分）。
- 任一 per-DP TP 超过 8 会直接报错退出。

`decode/start_server.sh <dp> <tp> [hetero_tp_sizes] <prefill_dp> <prefill_tp>`

- decode 侧同样支持 per-DP 异构 TP（同样要求 TP<=8）。
- prefill 为异构 TP 时，`prefill_tp` 只作为远端逻辑池描述符，传 per-DP
  TP 的最大值即可。

`proxy.sh` 通过环境变量生成代理参数：

```bash
PREFILL_DP_SIZE=4 DECODE_DP_SIZE=8 ./proxy.sh
```

## 实现要点

- `launch_online_dp.py` 不再依赖 `run_dp_template.sh` 中硬编码的
  `--heterogeneous-dp-config`，而是通过环境变量
  `HETERO_DP_CONFIG_JSON` / `KV_TRANSFER_CONFIG_JSON` 注入完整拓扑。
- prefill/decode 的 `dp_size`、`tp_size` 在 `kv_connector_extra_config`
  中任意搭配（TP 受 `<= o_groups(8)` 约束）；DeepSeek-V4 的 KV cache
  在每个 prefill TP rank 上全量复制，`MooncakeHybridConnector` 按请求
  哈希选择一个 prefill TP rank 拉取，不再要求 `prefill_tp >= decode_tp`。
- `run_dp_template.sh` 中仍需按节点修改 `nic_name`、`local_ip`、
  `model_path` 等环境。
