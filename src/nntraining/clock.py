import time


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
    def __init__(self):
        self._init_ref = None
        self._last_ref = None

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

    def print_elapsed_since_last_call(self, comment=None) -> float:
        elapsed = self.get_elapsed_since_last_call()
        print_comment(elapsed, comment, "elapsed since last call")
        return elapsed

    def get_elapsed_since_start(self) -> float:
        self.call()
        elapsed = self._last_ref - self._init_ref
        return elapsed

    def print_elapsed_since_start(self, comment=None) -> float:
        elapsed = self.get_elapsed_since_start()
        print_comment(elapsed, comment, "elapsed since start")
        return elapsed

    def get_elapsed_since_last_call_silent(self) -> float:
        return time.perf_counter() - self._last_ref

    def print_elapsed_since_last_call_silent(self, comment=None) -> float:
        elapsed = self.get_elapsed_since_last_call_silent()
        print_comment(elapsed, comment, "elapsed since start")
        return elapsed

    def get_elapsed_since_start_silent(self) -> float:
        elapsed = time.perf_counter() - self._init_ref
        return elapsed

    def print_elapsed_since_start_silent(self, comment=None) -> float:
        elapsed = self.get_elapsed_since_start_silent()
        print_comment(elapsed, comment, "elapsed since start")
        return elapsed


def display_time(t: float) -> str:
    return f"{t: .2}s"


def print_comment(elapsed, comment, default):
    comment = comment if comment else default
    print(f"{comment}:", display_time(elapsed))
