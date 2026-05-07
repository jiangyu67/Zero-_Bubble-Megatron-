# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense pipeline DualPipe helpers: secondary weight replica on the primary PP chunk."""

from __future__ import annotations

import torch

from megatron.training.global_vars import get_args
from megatron.training.utils import unwrap_model

# Stored on the forward (DDP) pipeline chunk; not in `model` list for the optimizer.
MEGATRON_DUAL_PIPE_BWD_ATTR = "_megatron_dual_pipe_bwd_module"


def get_dual_pipe_bwd_module(model_chunk0: torch.nn.Module):
    return getattr(model_chunk0, MEGATRON_DUAL_PIPE_BWD_ATTR, None)


def dual_pipe_zero_grad_extra(model_list: list) -> None:
    """Clear any stale grads on the backward weight replica (not wrapped by DDP)."""
    args = get_args()
    if getattr(args, "pipeline_schedule", None) != "dual_pipe":
        return
    if not model_list:
        return
    bwd = get_dual_pipe_bwd_module(model_list[0])
    if bwd is None:
        return
    for p in bwd.parameters():
        if p.grad is not None:
            p.grad = None


def merge_dual_pipe_bwd_grads_into_fwd(model_chunk0: torch.nn.Module) -> None:
    """Add gradients from the non-DDP replica into the primary (DDP) ``main_grad`` buffers.

    Odd-indexed microbatches run forward on the replica; their autograd leaves grads on replica
    parameters. Megatron DDP only hooks the primary chunk, so we accumulate here before
    ``finalize_model_grads``.
    """
    bwd = get_dual_pipe_bwd_module(model_chunk0)
    if bwd is None:
        return
    uw_f = unwrap_model(model_chunk0)
    uw_b = unwrap_model(bwd)
    for p_f, p_b in zip(uw_f.parameters(), uw_b.parameters()):
        if p_b.grad is None:
            continue
        g_b = p_b.grad
        if hasattr(p_f, "main_grad") and p_f.main_grad is not None:
            p_f.main_grad.add_(g_b.to(dtype=p_f.main_grad.dtype))
        elif p_f.grad is not None:
            p_f.grad.add_(g_b)
        else:
            raise RuntimeError(
                "dual_pipe merge: primary parameter has neither main_grad nor grad; "
                "unsupported DDP/optimizer layout for this parameter."
            )
        p_b.grad = None


def dual_pipe_sync_bwd_weights_from_fwd(model_list: list) -> None:
    """Keep the backward-oriented weight view identical to the primary weights after an update."""
    args = get_args()
    if getattr(args, "pipeline_schedule", None) != "dual_pipe":
        return
    if not model_list:
        return
    m0 = model_list[0]
    bwd = get_dual_pipe_bwd_module(m0)
    if bwd is None:
        return
    with torch.no_grad():
        for p_f, p_b in zip(
            unwrap_model(m0).parameters(),
            unwrap_model(bwd).parameters(),
        ):
            p_b.data.copy_(p_f.data)
