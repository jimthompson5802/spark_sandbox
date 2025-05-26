import os
import psutil
import time
from typing import Tuple


class SystemMonitor:
    """
    A class to monitor system resource usage including memory (RSS) and CPU usage.
    Useful for profiling and benchmarking code execution.
    """

    def __init__(self):
        """Initialize the SystemMonitor with the current process."""
        self.process = psutil.Process(os.getpid())
        self.clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    def get_current_rss(self) -> float:
        """
        Get the current Resident Set Size (RSS) memory usage of the process.

        Returns:
            float: The current RSS memory usage in megabytes (MB)
        """
        # Get memory info and convert from bytes to MB
        mem_info = self.process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)  # Convert bytes to MB
        return rss_mb

    def snapshot_cpu(self) -> Tuple:
        """
        Take a snapshot of the current CPU usage.
        Reads user and system time from /proc/self/stat and captures wall time.

        Returns:
            Tuple: A tuple containing (utime, stime, wall_time) values
        """
        with open("/proc/self/stat", "r") as f:
            fields = f.read().split()
            utime = int(fields[13])  # 14th field: user mode jiffies
            stime = int(fields[14])  # 15th field: kernel mode jiffies
        wall_time = time.monotonic()
        return utime, stime, wall_time

    def compute_cpu_usage(self, before_snapshot: Tuple, after_snapshot: Tuple) -> float:
        """
        Compute the CPU usage percentage between two snapshots.

        Args:
            before_snapshot (Tuple): A tuple containing (utime, stime, wall_time) at start
            after_snapshot (Tuple): A tuple containing (utime, stime, wall_time) at end

        Returns:
            float: The CPU usage percentage
        """
        before_utime, before_stime, before_wall_time = before_snapshot
        after_utime, after_stime, after_wall_time = after_snapshot
        wall_time = after_wall_time - before_wall_time

        cpu_time = ((after_utime - before_utime) + (after_stime - before_stime)) / self.clk_tck

        cpu_usage = (cpu_time / wall_time) * 100 if wall_time > 0 else 0.0

        return cpu_usage


# Example usage:
if __name__ == "__main__":
    # Create monitor instance
    monitor = SystemMonitor()

    # Get current memory usage
    print(f"Current RSS memory usage: {monitor.get_current_rss():.2f} MB")

    # Take CPU snapshot
    before_snapshot = monitor.snapshot_cpu()
    print(f"CPU snapshot: {before_snapshot}")

    # Simulate some work
    for _ in range(10**8):
        pass

    # Take another snapshot
    after_snapshot = monitor.snapshot_cpu()
    print(f"CPU snapshot after work: {after_snapshot}")

    print(f"\nCPU utilization:{monitor.compute_cpu_usage(before_snapshot, after_snapshot):.2f}%")
