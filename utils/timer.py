import statistics
import time
from collections import defaultdict

from utils.base import print_text


class TimerCtx:

    def __init__(self, timer, key, verbose=False):
        self.timer = timer
        self.key = key
        self.verbose = verbose

    def __enter__(self):
        self.timer.start_time_dict[self.key] = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        d = time.perf_counter() - self.timer.start_time_dict[self.key]
        self.timer.duration_dict[self.key].append(d)
        self.timer.count_dict[self.key] += 1
        if self.verbose:
            print_text(f"{self.key} cost {d:.3f}s", color="red")


class Timer:
    def __init__(self, name: str = "", verbose: bool = False, skip: int = 0):
        self.start_time_dict = {}
        self.duration_dict = defaultdict(list)
        self.count_dict = defaultdict(int)
        self.skip = skip
        self.verbose = verbose
        self.name = name  # name of this timer

    def summary(self):
        avg_dict = {}
        std_dict = {}
        total_dict = {}

        # compute mean/std/total for each task
        for key in self.duration_dict:
            data = self.duration_dict[key][self.skip :]
            avg = statistics.mean(data)
            std = statistics.stdev(data) if len(data) > 1 else 0.0
            total_time = sum(data)

            avg_dict[key] = avg
            std_dict[key] = std
            total_dict[key] = total_time

        # header
        lines = []
        timer_name = f" ({self.name})" if self.name else ""
        skip_info = f" (skip {self.skip})" if self.skip else ""
        lines.append(f"Timer Summary{timer_name}{skip_info}:")
        lines.append("-" * 74)
        lines.append(
            f"{'Task Name':<30}"
            f"{'Count':>8}"
            f"{'Mean (s)':>12}"
            f"{'Std (s)':>12}"
            f"{'Total (s)':>12}"
        )
        lines.append("-" * 74)

        # per-task rows
        overall_time = 0.0
        for key in self.duration_dict:
            count = self.count_dict[key]
            mean_v = avg_dict[key]
            std_v = std_dict[key]
            total_v = total_dict[key]
            overall_time += total_v

            lines.append(
                f"{key:<30}"
                f"{count:>8}"
                f"{mean_v:>12.2f}"
                f"{std_v:>12.2f}"
                f"{total_v:>12.2f}"
            )

        lines.append("-" * 74)

        # final row: overall total time
        lines.append(
            f"{'Overall Time':<30}{'':>8}{'':>12}{'':>12}{overall_time:>12.2f}"
        )
        lines.append("-" * 74)

        return "\n".join(lines)

    def timing(self, key):
        return TimerCtx(self, key, self.verbose)

    def start(self, key):
        self.start_time_dict[key] = time.perf_counter()
        return self.start_time_dict[key]

    def stop(self, key):
        d = time.perf_counter() - self.start_time_dict[key]
        self.duration_dict[key].append(d)
        self.count_dict[key] += 1
        if self.verbose:
            print_text(f"{key} cost {d:.3f}s", color="red")
        return

    def last_durations(self) -> str:
        """
        Return a string of the last recorded duration for each task,
        formatted as: "task1=1.23s, task2=0.87s".
        """
        items = []
        for key, durations in self.duration_dict.items():
            if len(durations) > 0:
                last = durations[-1]
                items.append(f"{key}={last:.3f}s")
            else:
                items.append(f"{key}=N/A")

        return ", ".join(items)

    # def add_duration_list(self, key, values):
    #     if key in self.duration_dict:
    #         print(f"{key} in timer.duration_dict, it will overwrite it!")
    #     self.duration_dict[key] = values
    #     self.count_dict[key] = len(values)

    # def add(self, key, value):
    #     self.duration_dict[key].append(value)
    #     self.count_dict[key] += 1

    # def get_times(self, key):
    #     return self.duration_dict[key]

    # def get_last_time(self, key):
    #     return self.duration_dict[key][-1]


if __name__ == "__main__":
    timer = Timer(verbose=True, skip=0)

    with timer.timing("task1"):
        time.sleep(1)

    with timer.timing("task1"):
        time.sleep(2)

    with timer.timing("task1"):
        time.sleep(1)

    timer.start("task2")
    time.sleep(2)
    timer.stop("task2")

    timer.start("task2")
    time.sleep(1)
    timer.stop("task2")

    with timer.timing("task3"):
        time.sleep(1)
        with timer.timing("task3.1"):
            time.sleep(1)

    print(timer.summary())
