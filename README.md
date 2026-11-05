# ResNet-50 distributed GPU benchmark for Slurm

This is **synthetic ResNet-50 training** using PyTorch DDP, NCCL, BF16 mixed precision, and one GPU per compute node. It measures **weak scaling** as the number of nodes increases, keeping **batch per GPU fixed**. It does not measure accuracy or NFS throughput.

## Prerequisites

- One GPU on every allocated node, visible in Slurm as `--gres=gpu:1`.
- One compatible shared Python environment with CUDA-enabled `torch` and a matching `torchvision`, accessible on all nodes. Edit the environment activation section in `resnet50_ddp.sbatch`.
- A private `192.168.250.x` IPv4 address on every compute node, mutually reachable on the cluster network. The scripts discover the address/interface and use them for rendezvous and NCCL.
- GPU jobs can open intra-cluster ports; check UFW/firewall policies. The rendezvous port is 20000 + (SLURM_JOB_ID modulo 20000); NCCL may need additional dynamically selected ports.
- Submit from a **shared directory** accessible to all compute nodes (`/home/...` or `/projects/...`).
- Verify the physical 10GbE links negotiate **10000Mb/s** (an earlier reported link was at **1000Mb/s**).

## First, smoke test the environment

```bash
python -c 'import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.version.cuda,torch.cuda.is_available())'
sinfo -N
```

Run the check in an actual GPU allocation if the login/head node has no GPU. Verify per-node network addresses: `ip -br -4 addr`; check `sudo ethtool <PRIVATE_NIC>`.

## Submit the scaling runs

```bash
sbatch --nodes=1 resnet50_ddp.sbatch
sbatch --nodes=2 resnet50_ddp.sbatch
sbatch --nodes=4 resnet50_ddp.sbatch
sbatch --nodes=8 resnet50_ddp.sbatch
```

Start with 1 and 2 nodes; use 4 and 8 only after successful results. These scripts default to `--batch-size 32`, `--warmup 20`, `--steps 200` per GPU, with no dataset download.

To adjust workload, use e.g.:

```bash
sbatch --nodes=2 --export=ALL,BENCH_BATCH_SIZE=64,BENCH_STEPS=300 resnet50_ddp.sbatch
```

Monitor with `squeue -u "$USER"`, `tail -f resnet50-ddp-<JOB_ID>.log`. A completed run produces `results/resnet50-<JOB_ID>.json` and a `BENCH_RESULT=...` line in the Slurm log.

## Interpret results

- Throughput: global training images per second, measured using elapsed time of the **slowest rank**.
- Speedup(N) = throughput(N) / throughput(1).
- Scaling efficiency(N) = speedup(N) / N * 100%.
- Keep batch/GPU, image size, BF16, warmup, and measured steps identical across runs.
- Scaling shortfalls may come from 10GbE bandwidth, network interface choices, NCCL gradient all-reduce, power/thermal limits, or node differences.

For a proper network performance baseline, also run `iperf3` between compute nodes and (optionally) NVIDIA `nccl-tests`. For real model-training validation, use an actual dataset plus `DistributedSampler` and track validation accuracy.
