#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from megatron.core.extensions.transformer_engine import MXSimLinear
    from megatron.core.pipeline_parallel.async_comm import get_async_comm_queue
except ImportError:
    MXSimLinear = None
    get_async_comm_queue = None

try:
    from megatron.core.model_parallel_config import ModelParallelConfig
except Exception:
    ModelParallelConfig = None

try:
    import torch.cuda.nvtx as nvtx
except Exception:
    nvtx = None

try:
    import torch.cuda.profiler as profiler
except Exception:
    profiler = None


class _UnwrapTensor(nn.Module):
    def forward(self, x):
        while isinstance(x, tuple):
            x = x[0]
        return x


@dataclass
class BenchmarkConfig:
    seq_length: int = 64
    batch_size: int = 8
    hidden_size: int = 1024
    num_steps: int = 5
    device: str = "cuda"


def build_baseline_model(cfg: BenchmarkConfig) -> nn.Module:
    return nn.Sequential(
        nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True),
        nn.ReLU(),
        nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True),
    ).to(cfg.device)


def build_mxfp8_model(cfg: BenchmarkConfig) -> nn.Module:
    if MXSimLinear is None:
        raise RuntimeError("MXSimLinear is not available in the current environment.")
    if ModelParallelConfig is None:
        raise RuntimeError("ModelParallelConfig is not available in the current environment.")
    mp_cfg = ModelParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    # The TE wrappers expect a few optional config fields that may not exist
    # on older/newer config variants; provide safe defaults for this benchmark.
    _defaults = {
        "symmetric_ar_type": None,
        "disable_parameter_transpose_cache": False,
        "init_model_with_meta_device": False,
    }
    for _k, _v in _defaults.items():
        if not hasattr(mp_cfg, _k):
            setattr(mp_cfg, _k, _v)
    return nn.Sequential(
        MXSimLinear(
            input_size=cfg.hidden_size,
            output_size=cfg.hidden_size,
            config=mp_cfg,
            init_method=torch.nn.init.normal_,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        ),
        _UnwrapTensor(),
        nn.ReLU(),
        MXSimLinear(
            input_size=cfg.hidden_size,
            output_size=cfg.hidden_size,
            config=mp_cfg,
            init_method=torch.nn.init.normal_,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        ),
        _UnwrapTensor(),
    ).to(cfg.device)


def random_input(cfg: BenchmarkConfig) -> torch.Tensor:
    return torch.randn(cfg.batch_size, cfg.seq_length, cfg.hidden_size, device=cfg.device)


def gather_quant_metrics() -> tuple[float, float, float]:
    try:
        from megatron.core.quantization.metrics import get_global_collector
        collector = get_global_collector()
        collector.sync_and_update()
        metrics_records = []
        dlrc_trigger_count = 0
        for layer_metrics in collector._metrics.values():
            sign_history = getattr(layer_metrics, "sign_rate_history", [])
            cos_history = getattr(layer_metrics, "cos_sim_history", [])
            for sign_rate, cos_sim in zip(sign_history, cos_history):
                metrics_records.append((sign_rate, cos_sim))
                if sign_rate < 0.95 or cos_sim < 0.98:
                    dlrc_trigger_count += 1

        if not metrics_records:
            return 1.0, 1.0, 0.0

        avg_sign = sum(r[0] for r in metrics_records) / len(metrics_records)
        avg_cos = sum(r[1] for r in metrics_records) / len(metrics_records)
        freq = float(dlrc_trigger_count) / len(metrics_records) * 100.0
        return avg_sign, avg_cos, freq
    except Exception:
        return 1.0, 1.0, 0.0


def drain_world_comm_queue() -> float:
    if get_async_comm_queue() is None:
        return 0.0
    queue = get_async_comm_queue()
    total_wait_ms = 0.0
    while not queue.empty():
        handle = queue.pop()
        if handle is None:
            break
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            handle.wait()
        except Exception:
            pass
        end.record()
        torch.cuda.synchronize()
        total_wait_ms += start.elapsed_time(end)
    return total_wait_ms


def benchmark_step(
    cfg: BenchmarkConfig,
    model: nn.Module,
    data: torch.Tensor,
    mode: str,
    step_index: Optional[int] = None,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()

    profiling = step_index == 9
    if profiling and nvtx is not None:
        nvtx.range_push("PROFILE_STEP_10")
    if profiling and profiler is not None:
        profiler.start()

    torch.cuda.reset_peak_memory_stats(cfg.device)
    step_start = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)

    hidden_wait_ms = 0.0
    step_start.record()

    optimizer.zero_grad(set_to_none=True)
    out = model(data)
    if isinstance(out, tuple):
        out = out[0]
    loss = loss_fn(out, torch.zeros_like(out))
    loss.backward()

    if mode == "mxfp8_zb":
        hidden_wait_ms += drain_world_comm_queue()

    optimizer.step()

    if mode == "mxfp8_zb":
        hidden_wait_ms += drain_world_comm_queue()

    step_end.record()
    torch.cuda.synchronize()

    if profiling and profiler is not None:
        profiler.stop()
    if profiling and nvtx is not None:
        nvtx.range_pop()

    total_time_ms = step_start.elapsed_time(step_end)
    peak_memory = int(torch.cuda.max_memory_allocated(cfg.device))
    hidden_ratio = (hidden_wait_ms / total_time_ms * 100.0) if total_time_ms > 0 else 0.0
    return total_time_ms, peak_memory, hidden_wait_ms, hidden_ratio


def run_benchmark(args):
    cfg = BenchmarkConfig(
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_steps=args.steps,
        device=args.device,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    if args.mode == "bf16_baseline":
        model = build_baseline_model(cfg)
    elif args.mode in {"mxfp8_sim", "mxfp8_zb"}:
        model = build_mxfp8_model(cfg)
    else:
        raise ValueError(f"Unknown benchmark mode: {args.mode}")

    data = random_input(cfg)
    if args.mode in {"mxfp8_sim", "mxfp8_zb"}:
        data = data.to(torch.bfloat16)

    print(
        "CONFIG:"
        f" batch_size={cfg.batch_size}"
        f" seq_length={cfg.seq_length}"
        f" hidden_size={cfg.hidden_size}"
        f" mode={args.mode}"
        f" steps={cfg.num_steps}"
        f" warmup={args.warmup}"
        f" device={cfg.device}"
    )

    for _ in range(args.warmup):
        _ = benchmark_step(cfg, model, data, args.mode)

    step_times = []
    peak_memories = []
    hidden_waits = []
    hidden_percents = []
    for step_i in range(cfg.num_steps):
        total_time_ms, peak_memory, hidden_wait_ms, hidden_ratio = benchmark_step(
            cfg, model, data, args.mode, step_index=step_i
        )
        step_times.append(total_time_ms)
        peak_memories.append(peak_memory)
        hidden_waits.append(hidden_wait_ms)
        hidden_percents.append(hidden_ratio)
        print(
            f"STEP {step_i + 1}/{cfg.num_steps}: time_ms={total_time_ms:.3f}"
            f" hidden_wait_ms={hidden_wait_ms:.3f} hidden_ratio={hidden_ratio:.2f}"
        )

    avg_time = sum(step_times) / len(step_times)
    avg_mem = sum(peak_memories) // len(peak_memories)
    avg_hidden = sum(hidden_waits) / len(hidden_waits)
    avg_hidden_pct = sum(hidden_percents) / len(hidden_percents)

    print("=== Benchmark Summary ===")
    print(f"Mode: {args.mode}")
    print(f"  avg step time (ms): {avg_time:.3f}")
    print(f"  peak memory (bytes): {avg_mem}")
    print(f"  avg hidden wait (ms): {avg_hidden:.3f}")
    print(f"  hidden latency %: {avg_hidden_pct:.2f}%")

    if args.mode in {"mxfp8_sim", "mxfp8_zb"}:
        avg_sign, avg_cos, dlrc_freq = gather_quant_metrics()
        print(
            "DAQ_METRICS:"
            f" avg_sign_rate={avg_sign:.4f}"
            f" avg_cos_sim={avg_cos:.4f}"
            f" dlrc_trigger_percent={dlrc_freq:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MXFP8 and Zero-Bubble performance")
    parser.add_argument("--mode", choices=["bf16_baseline", "mxfp8_sim", "mxfp8_zb"], required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_benchmark(args)
