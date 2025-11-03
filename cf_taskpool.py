import asyncio
import contextlib
import itertools
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from functools import partial
from typing import Any, Self
from weakref import ref

type _WorkItem[T] = tuple[asyncio.Future[T], Callable[[], Awaitable[T]]]
type _WorkQueue = asyncio.Queue[_WorkItem[Any] | None]


class TaskPoolExecutor:
    # Used to assign unique task names when task_name_prefix is not supplied
    _counter = itertools.count().__next__

    def __init__(self, max_workers: int | None = None, task_name_prefix: str = ""):
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self._max_workers = max_workers
        self._work_queue: _WorkQueue = asyncio.Queue()
        self._idle_semaphore = asyncio.Semaphore(0)
        self._tasks: set[asyncio.Task[None]] = set()
        self._shutdown = False
        self._shutdown_lock = asyncio.Lock()
        self._name_prefix = task_name_prefix or f"TaskPoolExecutor-{self._counter()}"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.shutdown()

    async def submit[T](
        self, fn: Callable[..., Awaitable[T]], /, *args: object, **kwargs: object
    ) -> asyncio.Future[T]:
        async with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            future: asyncio.Future[T] = asyncio.Future()
            await self._work_queue.put((future, partial(fn, *args, **kwargs)))
            await self._adjust_task_count()
            return future

    async def map[T](
        self,
        fn: Callable[..., Awaitable[T]],
        *iterables: object,
    ) -> AsyncGenerator[T]:
        fs = await asyncio.gather(
            *(self.submit(fn, *args) for args in zip(*iterables, strict=False))
        )

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        async def result_iterator() -> AsyncGenerator[T]:
            try:
                # Reverse to keep finishing order
                fs.reverse()
                while fs:
                    # Careful not to keep a reference to the popped future
                    yield await fs.pop()
            finally:
                for future in fs:
                    future.cancel()

        return result_iterator()

    async def shutdown(
        self, *, wait: bool = True, cancel_futures: bool = False
    ) -> None:
        async with self._shutdown_lock:
            self._shutdown = True
            if cancel_futures:
                # Drain all work items from the queue and cancel their associated futures
                while True:
                    try:
                        work_item = self._work_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if work_item is not None:
                        work_item[0].cancel()

            # Send a wake-up to prevent tasks from permanently blocking
            for _ in self._tasks:
                await self._work_queue.put(None)
        if wait and self._tasks:
            await asyncio.wait(self._tasks)

    async def _adjust_task_count(self) -> None:
        # If idle workers are available, don't spin new ones
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(0):
                if await self._idle_semaphore.acquire():
                    return

        num_tasks = len(self._tasks)
        if num_tasks < self._max_workers:
            work_queue = self._work_queue
            # When the executor gets lost, the weakref callback will wake up the workers
            self_ref = ref(self, lambda _: work_queue.put_nowait(None))
            name = f"{self._name_prefix}_{num_tasks}"
            task = asyncio.create_task(_worker(self_ref, work_queue), name=name)
            self._tasks.add(task)


async def _worker(executor_ref: ref[TaskPoolExecutor], work_queue: _WorkQueue) -> None:
    while True:
        try:
            work_item = work_queue.get_nowait()
        except asyncio.QueueEmpty:
            # Attempt to increment idle count if queue is empty
            executor = executor_ref()
            if executor is not None:
                executor._idle_semaphore.release()  # noqa: SLF001
            del executor
            work_item = await work_queue.get()

        # The executor that owns the worker has been shutdown or collected
        if work_item is None:
            break

        await _run(*work_item)
        # Delete references to object. See GH-60488
        del work_item


async def _run[T](future: asyncio.Future[T], fn: Callable[[], Awaitable[T]]) -> None:
    if future.cancelled():
        return
    try:
        result = await fn()
    except BaseException as exc:  # noqa: BLE001
        if not future.cancelled():
            future.set_exception(exc)
            # Break a reference cycle with the exception 'exc'
            del future
    else:
        if not future.cancelled():
            future.set_result(result)
