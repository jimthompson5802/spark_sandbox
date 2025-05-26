# Code Monitor

A Python utility for monitoring system resources during code execution.

## Overview

Code Monitor provides tools to track and analyze resource utilization such as memory (RSS) and CPU usage during program execution. It's particularly useful for profiling and benchmarking code performance.

## Features

- Real-time memory (RSS) monitoring
- CPU usage tracking and analysis
- Simple API for instrumenting existing code

## Installation

```bash
pip install code_monitor
```

Or, install directly from the repository:

```bash
git clone <repository-url>
cd code_monitor
pip install -e .
```

## Usage

```python
from code_monitor import SystemMonitor

# Create a monitor instance
monitor = SystemMonitor()

# Get current memory usage in MB
memory_usage = monitor.get_current_rss()
print(f"Current memory usage: {memory_usage} MB")

# Capture CPU usage before execution
before = monitor.snapshot_cpu()

# Run your code here
# ...

# Capture CPU usage after execution
after = monitor.snapshot_cpu()

# Calculate CPU usage percentage
cpu_percent = monitor.compute_cpu_usage(before, after)
print(f"CPU usage: {cpu_percent}%")
```

## Requirements

- Python 3.8+
- psutil

## License

This project is licensed under the MIT License - see the LICENSE file for details.
