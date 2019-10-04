import time

default_comment_last_call = "elapsed since last call"
default_comment_start = "elapsed since start"


class Clock:
    """
    Simple class for monitoring execution time. Relies on python standard library time (using .perf_counter).
    times: elapsed_since_start and elapsed_since_last_call, call meaning one of the following functions call (or start
    of the clock)
    functions:
        get, print: get/print elapsed time
        call: silent method to reset elapsed_since_last_call
    the silent version of a function does not trigger a call (i.e. does not reset elapsed_since_last_call)
    """

    def __init__(self, default_rounding_precision=2):
        self._init_ref = None
        self._last_ref = None
        self.default_rounding_precision = default_rounding_precision

    @classmethod
    def started(cls):
        clock = cls()
        clock.start()
        return clock

    def start(self):
        self._init_ref = time.perf_counter()
        self._last_ref = self._init_ref

    def restart(self):
        self.start()

    def call(self):
        self._last_ref = time.perf_counter()

    def get_elapsed_since_last_call(self) -> float:
        prev_last_ref = self._last_ref
        self.call()
        return self._last_ref - prev_last_ref

    def _print_since(self, elapsed, comment, rounding_precision):
        rounding_precision = self.default_rounding_precision if rounding_precision is None else rounding_precision
        print_comment(elapsed, comment, rounding_precision)
        return elapsed

    def print_elapsed_since_last_call(self, comment=default_comment_last_call, rounding_precision=None) -> float:
        return self._print_since(self.get_elapsed_since_last_call(), comment, rounding_precision)

    def get_elapsed_since_start(self) -> float:
        self.call()
        elapsed = self._last_ref - self._init_ref
        return elapsed

    def print_elapsed_since_start(self, comment=default_comment_start, rounding_precision=None) -> float:
        return self._print_since(self.get_elapsed_since_start(), comment, rounding_precision)

    def get_elapsed_since_last_call_silent(self) -> float:
        return time.perf_counter() - self._last_ref

    def print_elapsed_since_last_call_silent(self, comment=default_comment_last_call, rounding_precision=None) -> float:
        return self._print_since(self.get_elapsed_since_last_call_silent(), comment, rounding_precision)

    def get_elapsed_since_start_silent(self) -> float:
        elapsed = time.perf_counter() - self._init_ref
        return elapsed

    def print_elapsed_since_start_silent(self, comment=default_comment_start, rounding_precision=None) -> float:
        return self._print_since(self.get_elapsed_since_start_silent(), comment, rounding_precision)


def format_time(t: float, rounding_precision) -> str:
    return f"{t:.{rounding_precision}f}s"


def print_comment(elapsed, comment, rounding_precision):
    print(f"{comment}:", format_time(elapsed, rounding_precision))
