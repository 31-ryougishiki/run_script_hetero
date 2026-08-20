# 异构 PD 启动脚本

`run_scenario.sh` 覆盖三种部署：

| 场景 | prefill | decode | 说明 |
|---|---|---|---|
| 2 | dp15 tp1 | dp16 tp1 | 15P + 16D，prefill 纯 DP（SP/DSA-CP 关闭） |
| 3 | dp4 tp4 | dp8 tp2 | 16P + 16D，prefill DSA-CP+SP 开启 |
| hetero | dp4 tp(3,4,4,4) | dp16 tp1 | 原异构基线（15P + 16D，DSA-CP+SP 开启） |

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
# 场景2（纯 DP，SP/DSA-CP 必须关闭）
prefill/start_server.sh 15 1 "" 16 1 0 0
decode/start_server.sh 16 1 "" 15 1 0 0

# 场景3（prefill DSA-CP+SP 开启）
prefill/start_server.sh 4 4 "" 8 2 1 1
decode/start_server.sh 8 2 "" 4 4 0 0

# 原异构基线（prefill DSA-CP+SP 开启）
prefill/start_server.sh 4 4 3,4,4,4 16 1 1 1
decode/start_server.sh 16 1 "" 4 4 0 0
```

## 参数说明

`prefill/start_server.sh <dp> <tp> [hetero_tp_sizes] <decode_dp> <decode_tp> [enable_dsa_cp] [enable_sp]`

- 统一 TP 时 `hetero_tp_sizes` 传空字符串。
- 异构 TP 时传逗号分隔的 per-DP TP，例如 `3,4,4,4`。launcher 会自动生成
  `--heterogeneous-dp-config`；对不能整除 8 个 o_groups 的 TP（如 3）自动
  生成 group 对齐的 `tp_sharding_ratios`（tp=3 保持基线 `[2,1,1]`，
  其他值使用平衡拆分）。
- `enable_dsa_cp` / `enable_sp`：1 开启，0 关闭，默认 1/1。
  DSA-CP 依赖 SP，传 `1 0` 会被 launcher 拒绝。纯 DP 场景必须传 `0 0`。
- 任一 per-DP TP 超过 8 会直接报错退出。

`decode/start_server.sh <dp> <tp> [hetero_tp_sizes] <prefill_dp> <prefill_tp> [enable_dsa_cp] [enable_sp]`

- decode 侧同样支持 per-DP 异构 TP（同样要求 TP<=8）。
- prefill 为异构 TP 时，`prefill_tp` 只作为远端逻辑池描述符，传 per-DP
  TP 的最大值即可。
- decode 的 DSA-CP/SP 默认关闭。

`proxy.sh` 通过环境变量生成代理参数：

```bash
PREFILL_DP_SIZE=4 DECODE_DP_SIZE=8 ./proxy.sh
```

## 自动化测试

在任一能 ssh 到 75/76 的节点上执行（`remote_ctl.sh` 与脚本目录需同时
同步到 76/75 两台节点的 `/opt/its/z30055003/hetero`）：

```bash
# 查看测试场景
bash automated_hetero_test.sh list

# 运行全部场景
bash automated_hetero_test.sh all

# 只运行一个场景
bash automated_hetero_test.sh hetero_baseline
```

自动化流程：

1. 通过 `ssh` + `docker exec -d vllm_v23_30055003` 在 76 拉起 prefill、
   75 拉起 decode；
2. 轮询两侧 `7100..7100+dp-1` 端口及 `/health`，全部 200 后认为就绪；
3. 在 76 拉起 `proxy.sh`，等待 `9000` 端口；
4. 用 `request_hetero_test.py` 并行发送中文请求（默认 32 条、并发 8）；
5. 请求响应、summary、prefill/decode/proxy 日志保存到
   `./hetero_test_results/<case>/`；
6. 每个场景结束后清理服务，继续下一场景。

默认覆盖矩阵：

| case | prefill | decode | prefill DSA-CP | prefill SP |
|---|---|---|---|---|
| hetero_baseline | dp4 tp(3,4,4,4) | dp16 tp1 | 1 | 1 |
| hetero_no_sp_no_dsa | dp4 tp(3,4,4,4) | dp16 tp1 | 0 | 0 |
| tp43_no_sp_no_dsa | dp2 tp(4,3) | dp8 tp1 | 0 | 0 |
| hetero_no_dsa_sp | dp4 tp(3,4,4,4) | dp16 tp1 | 0 | 1 |
| tp43_no_dsa_sp | dp2 tp(4,3) | dp8 tp1 | 0 | 1 |
| pure_dp_no_sp_no_dsa | dp15 tp1 | dp16 tp1 | 0 | 0 |
| tp4_dsa_cp_sp | dp4 tp4 | dp8 tp2 | 1 | 1 |
| tp4_no_dsa_cp_sp | dp4 tp4 | dp8 tp2 | 0 | 1 |
| tp4_no_dsa_cp_no_sp | dp4 tp4 | dp8 tp2 | 0 | 0 |
| odd_tp_dsa_cp_sp | dp1 tp3 | dp16 tp1 | 1 | 1 |

可通过环境变量覆盖节点/容器/超时：

```bash
NODE_PREFILL=7.246.78.76 \
NODE_DECODE=7.246.78.75 \
CONTAINER=vllm_v23_30055003 \
STARTUP_TIMEOUT=1800 \
KEEP_RUNNING=1 \
bash automated_hetero_test.sh all
```

## 单独测试 P 或 D 实例

不通过 PD 分离代理，直接启动单个 prefill 或 decode 实例（仍保留
`kv_producer` / `kv_consumer` 角色），请求直接发到实例的 engine 端口：

```bash
# 查看角色可用场景
bash single_instance_test.sh prefill list
bash single_instance_test.sh decode list

# 单测 prefill：按顺序自动跑完全部内置用例
bash single_instance_test.sh prefill all

# 单测 prefill（仅异构用例，均至少 7 卡）
bash single_instance_test.sh prefill dp3_tp4                    # 12 卡，DSA-CP+SP
bash single_instance_test.sh prefill dp2_tp43                   # 7 卡，DSA-CP+SP
bash single_instance_test.sh prefill dp1_tp7                    # 7 卡，ratios [2,1,1,1,1,1,1]
bash single_instance_test.sh prefill dp4_tp3444                 # 15 卡，DSA-CP+SP
bash single_instance_test.sh prefill dp2_tp43_no_sp_no_dsa      # 7 卡，异构 no-SP/no-DSA
bash single_instance_test.sh prefill dp4_tp3444_no_sp_no_dsa    # 15 卡，异构 no-SP/no-DSA
bash single_instance_test.sh prefill dp2_tp43_no_dsa_sp         # 7 卡，异构 no-DSA-CP/SP
bash single_instance_test.sh prefill dp4_tp3444_no_dsa_sp       # 15 卡，异构 no-DSA-CP/SP

# 单测 decode：按顺序自动跑完全部内置用例
bash single_instance_test.sh decode all

# 单测 decode（仅异构用例，均 7 卡，no-SP/no-DSA）
bash single_instance_test.sh decode dp4_tp2122    # 2+1+2+2=7 卡
bash single_instance_test.sh decode dp4_tp2212    # 2+2+1+2=7 卡
```

custom 场景：

```bash
LOCAL_IP=7.246.78.76 \
DP_SIZE=2 TP_SIZE=4 HETERO_TP_SIZES=4,3 DSA_CP=1 SP=1 \
REMOTE_DP=16 REMOTE_DECODE_TP=1 \
bash single_instance_test.sh prefill custom
```

脚本接受任意 tp 参数；但当前实现限制 TP `<= o_groups(8)`，
因此 `tp15/tp16` 会在启动前明确报错，不会拉起进程。后续恢复
`tp>8` 支持后，无需修改测试脚本即可用 custom 模式直接测试
`tp15 -> tp16` 等 P/D 组合。

所有测试脚本都有 `MIN_TEST_CARDS=7` 卡数下限校验（可通过环境变量
覆盖）。当前单实例内置用例为 7~15 卡，双节点 PD 内置用例为 19~32 卡；
单节点 PD 测试已移除。

## 实现要点

- `launch_online_dp.py` 通过环境变量注入
  `HETERO_DP_CONFIG_JSON` / `KV_TRANSFER_CONFIG_JSON` /
  `ADDITIONAL_CONFIG_JSON`，并注入 `VLLM_ASCEND_ENABLE_FLASHCOMM1` 和
  `HETERO_ENABLE_SP` 标记；模板在 `source set_env.bash` 后按标记恢复，
  避免 CANN set_env 脚本覆盖 SP 开关。
- prefill/decode 的 `dp_size`、`tp_size` 在 `kv_connector_extra_config`
  中参数化（TP 受 `<= o_groups(8)` 约束）。TP 关系仍保持原仓约束：
  要求 `prefill_tp >= decode_tp`。
- `run_dp_template.sh` 中仍需按节点修改 `nic_name`、`local_ip`、
  `model_path` 等环境。
