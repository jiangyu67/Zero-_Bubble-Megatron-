# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Avoid importing `schedules` at module import time; it pulls in large portions of
# Megatron-Core and can create circular-import chains for lightweight utilities.
def get_forward_backward_func(*args, **kwargs):
    from .schedules import get_forward_backward_func as _get_forward_backward_func

    return _get_forward_backward_func(*args, **kwargs)


__all__ = ["get_forward_backward_func"]
