#!/bin/bash
# Heterogeneous DP/TP prefill node: dp0 tp=3, dp1..3 tp=4, total 15 NPUs.
# The per-DP tp sizes must match --heterogeneous-dp-config in run_dp_template.sh.
python launch_online_dp.py \
  --dp-size 4 \
  --hetero-tp-sizes 3,4,4,4 \
  --dp-size-local 4 \
  --dp-rank-start 0 \
  --dp-address 7.246.78.76 \
  --dp-rpc-port 12321 \
  --vllm-start-port 7100
