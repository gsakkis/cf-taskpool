import asyncio
import contextlib
import itertools
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Self
from weakref import ref


class TaskPoolExecutor:
    # Used to assign unique task names when task_name_prefix is not supplied
    _counter = itertools.count().__next__

    def __init__(self, max_workers: int | None = None, task_name_prefix: str = ""):
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self._max_workers = max_workers
        self._work_queue: asyncio.Queue[_WorkItem[Any] | None] = asyncio.Queue()
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

            work_item = _WorkItem(fn, args, kwargs)
            await self._work_queue.put(work_item)
            await self._adjust_task_count()
            return work_item.future

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
                    yield await _result_or_cancel(fs.pop())
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
                    if work_item is not None and work_item.future is not None:
                        work_item.future.cancel()

            # Send a wake-up to prevent tasks from permanently blocking
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


class _WorkItem[T]:
    def __init__(
        self,
        fn: Callable[..., Awaitable[T]],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ):
        self.future: asyncio.Future[T] = asyncio.Future()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    async def run(self) -> None:
        if not self.future.cancelled():
            try:
                self.future.set_result(await self.fn(*self.args, **self.kwargs))
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, asyncio.CancelledError):
                    self.future.cancel()
                else:
                    self.future.set_exception(exc)
                # Break a reference cycle with the exception 'exc'
                del self


async def _worker(
    executor_reference: ref[TaskPoolExecutor],
    work_queue: asyncio.Queue[_WorkItem[Any] | None],
) -> None:
    while True:
        try:
            work_item = work_queue.get_nowait()
        except asyncio.QueueEmpty:
            # Attempt to increment idle count if queue is empty
            executor = executor_reference()
            if executor is not None:
                executor._idle_semaphore.release()  # noqa: SLF001
            del executor
            work_item = await work_queue.get()

        if work_item is not None:
            await work_item.run()
            # Delete references to object. See GH-60488
            del work_item
            continue

        executor = executor_reference()
        # Exit if the executor that owns the worker has been collected or shutdown
        if executor is None or executor._shutdown:  # noqa: SLF001
            # Notice other workers
            await work_queue.put(None)
            return
        del executor


async def _result_or_cancel[T](fut: asyncio.Future[T]) -> T:
    try:
        try:
            return await fut
        finally:
            fut.cancel()
    finally:
        # Break a reference cycle with the exception in self._exception
        del fut
