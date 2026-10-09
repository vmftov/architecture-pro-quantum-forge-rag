import time


class SumTimeCounter:
    def __init__(self):
        self._start: float | None = None
        self._elapsed: float = 0.0

    def reset(self):
        self._start = None
        self._elapsed = 0.0

    def start(self):
        if self._start is None:
            self._start = time.perf_counter()

    def stop(self):
        if self._start is not None:
            self._elapsed += time.perf_counter() - self._start
            self._start = None

    def elapsed(self) -> float:
        if self._start is not None:
            return self._elapsed + time.perf_counter() - self._start
        return self._elapsed

    def elapsed_fmt(self) -> str:
        elapsed = self.elapsed()
        if elapsed < 1e-3:
            return f"{elapsed * 1e6:.2f} µs"
        if elapsed < 1:
            return f"{elapsed * 1e3:.2f} ms"
        return f"{elapsed:.3f} s"
