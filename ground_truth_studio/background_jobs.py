"""Bounded background preparation; no FFmpeg subprocess runs on the UI thread."""
from __future__ import annotations

import atexit
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class BackgroundJobs:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gt-source")
        self.futures: dict[tuple[Any, ...], Future] = {}
        self.lock = threading.Lock()
        atexit.register(self.executor.shutdown, wait=False, cancel_futures=True)

    def request(self, key: tuple[Any, ...], operation: Callable[[], Any]) -> Future:
        """Reuse one preparation job per source version; do not duplicate workers."""
        with self.lock:
            if key not in self.futures:
                self.futures[key] = self.executor.submit(operation)
            return self.futures[key]
