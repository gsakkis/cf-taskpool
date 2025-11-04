import asyncio
import contextlib
import inspect
import itertools
import os
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Iterable
from typing import Any, ParamSpec, Self, TypeVar, overload

_WorkQueue = asyncio.Queue[tuple[asyncio.Future[Any], Awaitable[Any]] | None]

T = TypeVar("T")
P = ParamSpec("P")


class TaskPoolExecutor:
    # Used to assign unique task names when task_name_prefix is not supplied
    _counter = itertools.count().__next__

    def __init__(self, max_workers: int | None = None, task_name_prefix: str = ""):
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self._max_workers = max_workers
        self._work_queue = _WorkQueue()
        self._idle_semaphore = asyncio.Semaphore(0)
        self._tasks: set[asyncio.Task[None]] = set()
        self._shutdown = False
        self._shutdown_lock = asyncio.Lock()
        self._name_prefix = task_name_prefix or f"TaskPoolExecutor-{self._counter()}"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.shutdown()

    @overload
    async def submit(
        self, fn: Callable[P, Awaitable[T]], /, *args: P.args, **kwargs: P.kwargs
    ) -> asyncio.Future[T]: ...

    @overload
    async def submit(self, aw: Awaitable[T], /) -> asyncio.Future[T]: ...

    async def submit(
        self,
        aw_or_fn: Callable[P, Awaitable[T]] | Awaitable[T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> asyncio.Future[T]:
        future: asyncio.Future[T] = asyncio.Future()
        if isinstance(aw_or_fn, Awaitable):
            if args or kwargs:
                raise TypeError("Cannot pass args/kwargs when submitting an awaitable")
            awaitable = aw_or_fn
        else:
            awaitable = _shielded_run_coro(aw_or_fn, *args, **kwargs)

        if inspect.iscoroutine(awaitable):
            # When the future gets garbage collected, ensure the coroutine is closed
            weakref.finalize(future, _close_unawaited_coro, awaitable)

        async with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            await self._work_queue.put((future, awaitable))
            await self._adjust_task_count()
            return future

    async def map(
        self,
        fn: Callable[..., Awaitable[T]],
        *iterables: Iterable[Any],
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
            # When the executor gets garbage collected, put None into the work queue to
            # wake up the worker tasks
            weakref.finalize(self, self._work_queue.put_nowait, None)
            self._tasks.add(
                asyncio.create_task(
                    coro=_worker(self._work_queue, self._idle_semaphore),
                    name=f"{self._name_prefix}_{num_tasks}",
                )
            )


async def _worker(work_queue: _WorkQueue, idle_semaphore: asyncio.Semaphore) -> None:
    while True:
        try:
            work_item = work_queue.get_nowait()
        except asyncio.QueueEmpty:
            if _current_task_cancelling():
                break
            # Attempt to increment idle count if queue is empty
            idle_semaphore.release()
            work_item = await work_queue.get()

        # The executor that owns the worker has been shutdown or collected
        if work_item is None:
            break

        await _run(*work_item)
        # Delete references to object. See GH-60488
        del work_item


async def _run(future: asyncio.Future[T], awaitable: Awaitable[T]) -> None:
    if future.cancelled():
        return
    try:
        result = await awaitable
    except asyncio.CancelledError:
        future.cancel()
    except BaseException as exc:  # noqa: BLE001
        if not future.cancelled():
            future.set_exception(exc)
            # Break a reference cycle with the exception 'exc'
            del future
    else:
        if not future.cancelled():
            future.set_result(result)


async def _shielded_run_coro(
    fn: Callable[P, Awaitable[T]], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    try:
        return await fn(*args, **kwargs)
    except asyncio.CancelledError:
        # Retry only if the _worker is cancelling (as opposed to fn raising)
        if _current_task_cancelling():
            return await fn(*args, **kwargs)
        raise


def _close_unawaited_coro(coro: Coroutine[Any, Any, Any]) -> None:
    if inspect.getcoroutinestate(coro) == inspect.CORO_CREATED:
        coro.close()


def _current_task_cancelling() -> bool:
    current_task = asyncio.current_task()
    return current_task is not None and current_task.cancelling() > 0
