# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import importlib
from typing import Any

from megatron.core import parallel_state
from megatron.core.package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
    "utils",
    "DistributedDataParallel",
    "InferenceParams",
    "ModelParallelConfig",
    "Timers",
    "__contact_emails__",
    "__contact_names__",
    "__description__",
    "__download_url__",
    "__homepage__",
    "__keywords__",
    "__license__",
    "__package_name__",
    "__repository_url__",
    "__shortversion__",
    "__version__",
]

def __getattr__(name: str) -> Any:
    """Lazy-import heavyweight submodules to avoid circular imports at import time."""
    if name in {"tensor_parallel", "utils"}:
        return importlib.import_module(f"megatron.core.{name}")
    if name == "DistributedDataParallel":
        from megatron.core.distributed import DistributedDataParallel as _DistributedDataParallel

        return _DistributedDataParallel
    if name == "InferenceParams":
        from megatron.core.inference_params import InferenceParams as _InferenceParams

        return _InferenceParams
    if name == "ModelParallelConfig":
        from megatron.core.model_parallel_config import ModelParallelConfig as _ModelParallelConfig

        return _ModelParallelConfig
    if name == "Timers":
        from megatron.core.timers import Timers as _Timers

        return _Timers
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals().keys())
        | {
            "tensor_parallel",
            "utils",
            "DistributedDataParallel",
            "InferenceParams",
            "ModelParallelConfig",
            "Timers",
        }
    )


try:
    import torch
    from packaging.version import Version as _PkgVersion

    _torch_ver = str(getattr(torch, "__version__", "0")).split("+")[0]
    if _PkgVersion(_torch_ver) >= _PkgVersion("2.6.0a0"):
        from .safe_globals import register_safe_globals

        register_safe_globals()
except Exception:
    # If torch (or packaging) isn't importable here, skip safe globals registration.
    pass
