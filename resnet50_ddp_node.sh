#!/bin/bash
set -euo pipefail

# Every node independently selects the NIC with the cluster-private IPv4 address.
NET_IFACE=$(ip -o -4 addr show | awk '$4 ~ /^10[.]210[.]45[.]/ {print $2;}'| tail -n 1)
if [[ -z "$NET_IFACE" ]]; then
  echo "ERROR on $(hostname): no 10.210.45.x interface" >&2
  exit 2
fi
export NCCL_SOCKET_IFNAME="=$NET_IFACE"
export GLOO_SOCKET_IFNAME="$NET_IFACE"
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "worker=$(hostname) iface=$NET_IFACE node_count=$SLURM_JOB_NUM_NODES" >&2
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT" >&2

exec python -m torch.distributed.run \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --nproc-per-node=1 \
  --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --max-restarts=0 \
  resnet50_ddp_benchmark.py \
  --batch-size="$BENCH_BATCH_SIZE" \
  --warmup="$BENCH_WARMUP" \
  --steps="$BENCH_STEPS" \
  --output="results/resnet50-$SLURM_JOB_ID.json"
