#!/usr/bin/env python3
"""Synthetic ResNet-50 DDP training throughput benchmark: one process per GPU."""
import argparse
import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet50


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32, help="Images per GPU per step")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup training steps")
    parser.add_argument("--steps", type=int, default=200, help="Measured training steps")
    parser.add_argument("--output", type=Path, default=None, help="JSON result path (rank 0 only)")
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 1 or args.steps < 1:
        parser.error("batch-size, warmup, and steps must be positive")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable. Check GPU allocation/PyTorch CUDA build.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", local_rank)
        torch.manual_seed(12345)
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

        model = resnet50(weights=None).to(device).train()
        ddp = DDP(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)

        # Generate data on-device and reuse it: no dataset download, NFS, or data-loader noise.
        torch.manual_seed(12345 + rank)
        images = torch.randn(args.batch_size, 3, 224, 224, device=device)
        targets = torch.randint(0, 1000, (args.batch_size,), device=device)

        print(
            f"rank={rank}/{world_size} host={socket.gethostname()} "
            f"gpu={torch.cuda.get_device_name(local_rank)} "
            f"visible={os.getenv('CUDA_VISIBLE_DEVICES', 'unset')} "
            f"nccl_iface={os.getenv('NCCL_SOCKET_IFNAME', 'auto')}",
            flush=True,
        )

        def training_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = ddp(images)
                loss = F.cross_entropy(logits, targets)
            loss.backward()  # DDP uses NCCL to synchronize gradients among all ranks.
            optimizer.step()
            return loss

        for _ in range(args.warmup):
            loss = training_step()
        
        print(f"rank={rank} warmup last_loss={loss.detach().float().item():.5f}", flush=True)

        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            loss = training_step()
            if _ % 10 == 0 and rank == 0:
                print(f"step={_} last_loss={loss.detach().float().item():.5f}", flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Global throughput must use the time of the slowest participating rank.
        elapsed_max = torch.tensor([elapsed], dtype=torch.float64, device=device)
        dist.all_reduce(elapsed_max, op=dist.ReduceOp.MAX)
        peak_allocated = torch.tensor(
            [torch.cuda.max_memory_allocated(device) / (1024**3)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(peak_allocated, op=dist.ReduceOp.MAX)

        if rank == 0:
            total_images = args.batch_size * args.steps * world_size
            seconds = elapsed_max.item()
            result = {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
                "world_size_gpus": world_size,
                "nodes": int(os.environ.get("SLURM_JOB_NUM_NODES", world_size)),
                "batch_size_per_gpu": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "warmup_steps": args.warmup,
                "measured_steps": args.steps,
                "training_seconds": round(seconds, 4),
                "global_images_per_second": round(total_images / seconds, 2),
                "worst_rank_peak_allocated_gpu_gib": round(peak_allocated.item(), 2),
                "last_loss_rank0": round(loss.detach().float().item(), 5),
                "gpu_rank0": torch.cuda.get_device_name(local_rank),
                "pytorch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "precision": "bfloat16 autocast",
                "data": "synthetic resident on GPU",
            }
            print("BENCH_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
