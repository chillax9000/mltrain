import time


class Clock:
    def __init__(self):
        self._init_ref = None
        self._last_ref = None

    def start(self):
        self._init_ref = time.perf_counter()
        self._last_ref = self._init_ref

    def silent_call(self):
        self._last_ref = time.perf_counter()

    def elapsed_since_last_call(self) -> float:
        now_ref = time.perf_counter()
        elapsed = now_ref - self._last_ref
        self._last_ref = now_ref
        return elapsed

    def print_elapsed_since_last_call(self, comment=None) -> float:
        elapsed = self.elapsed_since_last_call()
        comment = comment if comment else "elapsed since last call"
        print(comment + ":", display_time(elapsed))
        return elapsed

    def elapsed_since_start(self) -> float:
        elapsed = time.perf_counter() - self._init_ref
        return elapsed

    def print_elapsed_since_start(self, comment=None) -> float:
        elapsed = self.elapsed_since_start()
        comment = comment if comment else "elapsed since start"
        print(comment + ":", display_time(elapsed))
        return elapsed


def display_time(t: float) -> str:
    return f"{t: .2}s"
