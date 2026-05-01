# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import collections
from typing import Optional

import torch


class AsyncCommQueue:
    """Queue for asynchronous communication handles and bubble-fill tasks."""

    def __init__(self):
        self._queue = collections.deque()

    def push(self, handle: torch._C._distributed_c10d.Work) -> None:
        """Push an async work handle into the queue."""
        self._queue.append(handle)

    def pop(self) -> Optional[torch._C._distributed_c10d.Work]:
        """Pop the oldest pending async work handle."""
        if self.empty():
            return None
        return self._queue.popleft()

    def empty(self) -> bool:
        return len(self._queue) == 0

    def clear(self) -> None:
        self._queue.clear()


_global_async_comm_queue = AsyncCommQueue()


def get_async_comm_queue() -> AsyncCommQueue:
    """Get the shared async communication queue."""
    return _global_async_comm_queue
