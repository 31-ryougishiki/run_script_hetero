#!/bin/bash
# PD proxy for the heterogeneous DeepSeek-V4 deployment:
#   - prefill node 7.246.78.76 runs dp0..dp3 on vllm ports 7100..7103
#   - decode  node 7.246.78.75 runs dp0..dp15 on vllm ports 7100..7115
# Start from hetero_cp/run_script_hetero so the proxy script can import the
# bundled load_balance_proxy_server_example.py.

python load_balance_proxy_server_example.py \
  --host 7.246.78.75 \
  --port 1999 \
  --prefiller-hosts \
    7.246.78.76 7.246.78.76 7.246.78.76 7.246.78.76 \
  --prefiller-ports \
    7100 7101 7102 7103 \
  --decoder-hosts \
    7.246.78.75 7.246.78.75 7.246.78.75 7.246.78.75 \
    7.246.78.75 7.246.78.75 7.246.78.75 7.246.78.75 \
    7.246.78.75 7.246.78.75 7.246.78.75 7.246.78.75 \
    7.246.78.75 7.246.78.75 7.246.78.75 7.246.78.75 \
  --decoder-ports \
    7100 7101 7102 7103 7104 7105 7106 7107 \
    7108 7109 7110 7111 7112 7113 7114 7115
