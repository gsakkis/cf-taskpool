# cf-taskpool

An asynchronous task pool with a `concurrent.futures`-like API for executing coroutines using a pool of asyncio tasks.

## Features

- **Simple, familiar API** - If you've used `ThreadPoolExecutor` or `ProcessPoolExecutor`, you'll feel right at home
- **Plays well with asyncio** - Seamlessly integrates with `asyncio.wait()`, `asyncio.as_completed()`, and other asyncio utilities
- **100% test coverage** - Most tests ported from Python's `concurrent.futures` test suite
- **No 3rd party dependencies** - Pure Python, only requires Python 3.11+
- **MIT licensed** - Free to use, modify, and distribute

## Installation

```bash
pip install cf-taskpool
```

## Quick Start

```python
import asyncio
from cf_taskpool import TaskPoolExecutor


async def fetch_data(url: str) -> str:
    """Simulate an async operation."""
    await asyncio.sleep(0.1)
    return f"Data from {url}"


async def main():
    # Create an executor with a pool of 3 workers
    async with TaskPoolExecutor(max_workers=3) as executor:
        # Submit a single task
        future = executor.submit(fetch_data, "https://example.com")
        result = await future
        print(result)  # Data from https://example.com


asyncio.run(main())
```

## API Overview

### `TaskPoolExecutor(max_workers=None, task_name_prefix="")`

Creates a new task pool executor.

- `max_workers`: Maximum number of workers (defaults to `os.cpu_count()`)
- `task_name_prefix`: Optional prefix for worker task names

### `submit(fn, /, *args, **kwargs) -> asyncio.Future`

Submits a callable to be executed. Returns an `asyncio.Future`.

```python
async def multiply(x: int, y: int) -> int:
    await asyncio.sleep(0.1)
    return x * y


async with TaskPoolExecutor() as executor:
    future = executor.submit(multiply, 6, 7)
    result = await future
    print(result)  # 42
```

You can also submit an awaitable directly:

```python
async with TaskPoolExecutor() as executor:
    coro = multiply(6, 7)
    future = executor.submit(coro)
    result = await future
    print(result)  # 42
```

### `async map(fn, *iterables, buffersize=None) -> AsyncGenerator`

Returns an async iterator that yields results as they complete.

```python
async def process_item(item: int) -> int:
    await asyncio.sleep(0.1)
    return item * 2


async with TaskPoolExecutor(max_workers=3) as executor:
    results = []
    async for result in await executor.map(process_item, range(5)):
        results.append(result)
    print(results)  # [0, 2, 4, 6, 8]
```

The `buffersize` parameter controls how many items are pre-fetched:

```python
# Only buffer 2 items at a time (useful for large or infinite iterables)
async for result in await executor.map(process_item, range(100), buffersize=2):
    print(result)
```

### `async shutdown(wait=True, cancel_futures=False)`

Signals the executor to stop accepting new tasks and cleans up resources.

- `wait`: If `True`, waits for pending tasks to complete
- `cancel_futures`: If `True`, cancels all pending futures

```python
executor = TaskPoolExecutor(max_workers=3)
# ... submit tasks ...
await executor.shutdown(wait=True)
```

Or use as a context manager (recommended):

```python
async with TaskPoolExecutor(max_workers=3) as executor:
    # ... submit tasks ...
    # shutdown() is called automatically on exit
```

## Integration with asyncio

### Using `asyncio.wait()`

```python


async def task(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return name


async with TaskPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(task, "fast", 0.1),
        executor.submit(task, "medium", 0.2),
        executor.submit(task, "slow", 0.3),
    ]

    # Wait for the first task to complete
    done, pending = await asyncio.wait(
        futures, return_when=asyncio.FIRST_COMPLETED
    )
    for future in done:
        print(f"Completed: {await future}")

    # Wait for all tasks to complete
    done, pending = await asyncio.wait(
        pending, return_when=asyncio.ALL_COMPLETED
    )
    for future in done:
        print(f"Completed: {await future}")
```

### Using `asyncio.as_completed()`

```python
async with TaskPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(task, "task1", 0.3),
        executor.submit(task, "task2", 0.1),
        executor.submit(task, "task3", 0.2),
    ]

    # Process results as they complete
    for coro in asyncio.as_completed(futures):
        result = await coro
        print(f"Got result: {result}")
```

### Using `asyncio.gather()`

```python
async with TaskPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(task, "task1", 0.3),
        executor.submit(task, "task2", 0.1),
        executor.submit(task, "task3", 0.2),
    ]

    # Wait for all and collect results
    results = await asyncio.gather(*futures)
    print(results)  # ['task1', 'task2', 'task3']
```

## Advanced Examples

### Handling Exceptions

```python
async def failing_task():
    await asyncio.sleep(0.1)
    raise ValueError("Something went wrong")


async with TaskPoolExecutor() as executor:
    future = executor.submit(failing_task)

    try:
        await future
    except ValueError as e:
        print(f"Caught exception: {e}")

    # Or check the exception later
    print(future.exception())  # ValueError("Something went wrong")
```

### Cancellation

```python
async def long_running_task():
    await asyncio.sleep(10)
    return "Done"


async with TaskPoolExecutor() as executor:
    future = executor.submit(long_running_task)

    # Cancel the task
    future.cancel()

    try:
        await future
    except asyncio.CancelledError:
        print("Task was cancelled")
```

### Processing Multiple Iterables with `map()`

```python
async def multiply(x: int, y: int) -> int:
    await asyncio.sleep(0.1)
    return x * y


async with TaskPoolExecutor(max_workers=3) as executor:
    # Multiply corresponding elements from two iterables
    async for result in await executor.map(multiply, range(5), range(5, 10)):
        print(result)  # 0, 6, 14, 24, 36
```

### Buffered Processing with `map()`

```python
import itertools as it


async def process(n: int) -> int:
    await asyncio.sleep(0.1)
    return n * 2


async with TaskPoolExecutor(max_workers=3) as executor:
    # Process an infinite iterator with a buffer of 2
    async for result in await executor.map(process, it.count(), buffersize=2):
        print(result)
        if result >= 20:
            break
```

## Comparison with concurrent.futures

| Feature | ThreadPoolExecutor | ProcessPoolExecutor | TaskPoolExecutor         |
|---------|-------------------|---------------------|--------------------------|
| Executes | Functions in threads | Functions in processes | Coroutines/Awaitables in tasks   |
| Returns | `concurrent.futures.Future` | `concurrent.futures.Future` | `asyncio.Future`         |
| Use with | `concurrent.futures.wait()` | `concurrent.futures.wait()` | `asyncio.wait()`         |
| Use with | `concurrent.futures.as_completed()` | `concurrent.futures.as_completed()` | `asyncio.as_completed()` |
| Best for | I/O-bound blocking code | CPU-bound code | Async/await code         |
