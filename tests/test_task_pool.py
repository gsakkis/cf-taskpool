import asyncio
import contextlib
import os
import weakref
from functools import partial

import pytest

from cf_taskpool import TaskPoolExecutor

from . import adivmod, amul


class MyObject:
    def __init__(self, i):
        self.i = i

    async def my_method(self):
        await asyncio.sleep(0.01)
        return self.i

    @classmethod
    async def create(cls, i):
        await asyncio.sleep(0.01)
        return cls(i)


class FalseyBoolError(Exception):
    def __bool__(self):
        return False  # pragma: no cover


class FalseyLenError(Exception):
    def __len__(self):
        return 0  # pragma: no cover


class TestTaskPoolExecutor:
    async def test_submit(self, executor):
        future = await executor.submit(amul, 2, 8)
        assert await future == 16
        assert future.result() == 16

    async def test_submit_keyword(self, executor):
        async def acapture(*args, **kwargs):
            await asyncio.sleep(0.01)
            return args, kwargs

        future = await executor.submit(amul, 2, y=8)
        assert await future == 16
        assert future.result() == 16

        future = await executor.submit(acapture, 1, self=2, fn=3)
        assert await future == ((1,), {"self": 2, "fn": 3})
        assert future.result() == ((1,), {"self": 2, "fn": 3})

        with pytest.raises(TypeError):
            await executor.submit(fn=acapture, arg=1)
        with pytest.raises(TypeError):
            await executor.submit(arg=1)

    async def test_exception(self, executor):
        future = await executor.submit(adivmod, 2, 0)
        with pytest.raises(ZeroDivisionError) as exc_info:
            await future
        assert future.exception() is exc_info.value

    async def test_cancellation(self, executor):
        future = await executor.submit(adivmod, 2, 0, cancel_if_zero=True)
        with pytest.raises(asyncio.CancelledError):
            await future
        assert future.cancelled()

    async def test_map(self, executor):
        expected = await asyncio.gather(*map(amul, range(10), range(10)))
        agen = await executor.map(amul, range(10), range(10))
        assert [x async for x in agen] == expected

    @pytest.mark.parametrize(
        ("func", "exc_type"),
        [
            (adivmod, ZeroDivisionError),
            (partial(adivmod, cancel_if_zero=True), asyncio.CancelledError),
        ],
    )
    async def test_map_exception(self, executor, func, exc_type):
        agen = await executor.map(func, [1, 1, 1, 1], [2, 3, 0, 5])
        assert await agen.__anext__() == (0, 1)
        assert await agen.__anext__() == (0, 1)
        with pytest.raises(exc_type):
            await agen.__anext__()

    async def test_no_stale_references(self, executor):
        # Issue #16284: check that the executors don't unnecessarily hang onto
        # references.
        my_object = MyObject(1)
        my_object_collected = asyncio.Event()
        my_object_callback = weakref.ref(my_object, lambda _: my_object_collected.set())
        # Deliberately discarding the future.
        await executor.submit(my_object.my_method)
        del my_object
        try:
            await asyncio.wait_for(my_object_collected.wait(), timeout=1.0)
        except TimeoutError:  # pragma: no cover
            pytest.fail("Stale reference not collected within timeout.")
        assert my_object_callback() is None

    def test_max_workers_negative(self):
        for number in (0, -1):
            with pytest.raises(ValueError, match="max_workers must be greater than 0"):
                TaskPoolExecutor(max_workers=number)

    async def test_free_future_reference(self, executor):
        future = await executor.submit(MyObject.create, 1)
        await future

        wr = weakref.ref(future)
        del future
        # the future is not released exactly when the result is set but when the task's
        # wakeup callback runs via the event loop's call_soon()
        assert wr() is not None
        await asyncio.sleep(0)
        assert wr() is None

    async def test_free_result_reference(self, executor):
        # Issue #14406: Result iterator should not keep an internal
        # reference to result objects.
        async for obj in await executor.map(MyObject.create, range(10)):
            wr = weakref.ref(obj)
            del obj
            # the result may be still referenced by a future that is released when the
            # task's wakeup callback runs via the event loop's call_soon()
            if wr() is not None:
                await asyncio.sleep(0)
                assert wr() is None

    @pytest.mark.parametrize("exc_type", [FalseyBoolError, FalseyLenError])
    async def test_swallows_falsey_exceptions(self, executor, exc_type):
        # see gh-132063: Prevent exceptions that evaluate as falsey from being ignored.
        # Recall: `x` is falsey if `len(x)` returns 0 or `bool(x)` returns False.
        async def araise(exception):
            await asyncio.sleep(0.01)
            raise exception

        msg = "falsy"
        future = await executor.submit(araise, exc_type(msg))
        with pytest.raises(exc_type, match=msg):
            await future

    async def test_map_submits_without_iteration(self, executor):
        """Tests verifying issue 11777."""
        finished = []

        def record_finished(n):
            finished.append(n)

        await executor.map(record_finished, range(10))
        await executor.shutdown(wait=True)
        assert set(finished) == set(range(10))

    def test_default_workers(self):
        executor = TaskPoolExecutor()
        assert executor._max_workers == os.cpu_count()

    async def test_saturation(self):
        executor = TaskPoolExecutor(4)

        async def acquire_lock(lock):
            await lock.acquire()

        sem = asyncio.Semaphore(0)
        for _ in range(15 * executor._max_workers):
            await executor.submit(acquire_lock, sem)
        assert len(executor._tasks) == executor._max_workers
        for _ in range(15 * executor._max_workers):
            sem.release()
        await executor.shutdown(wait=True)

    async def test_idle_worker_reuse(self):
        executor = TaskPoolExecutor()
        assert await (await executor.submit(amul, 21, 2)) == 42
        assert await (await executor.submit(amul, 6, 7)) == 42
        assert await (await executor.submit(amul, 3, 14)) == 42
        assert len(executor._tasks) == 1
        await executor.shutdown(wait=True)

    async def test_executor_map_current_future_cancel(self):
        stop_event = asyncio.Event()
        log = []

        async def log_n_wait(ident):
            log.append(f"{ident=} started")
            try:
                await stop_event.wait()
            finally:
                log.append(f"{ident=} stopped")

        async with TaskPoolExecutor(max_workers=1) as pool:
            # submit work to saturate the pool
            fut = await pool.submit(log_n_wait, ident="first")
            try:
                agen = await pool.map(log_n_wait, ["second", "third"])
                with pytest.raises(TimeoutError):
                    async with contextlib.aclosing(agen), asyncio.timeout(0):
                        await anext(agen)
            finally:
                stop_event.set()
            await fut
        # ident='second' is cancelled as a result of raising a TimeoutError
        # ident='third' is cancelled because it remained in the collection of futures
        assert log == ["ident='first' started", "ident='first' stopped"]
