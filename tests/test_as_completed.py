import asyncio
import itertools
from contextlib import suppress

import pytest

Future = asyncio.Future[object]


def cancelled_future():
    f = Future()
    f.cancel()
    return f


def exception_future():
    f = Future()
    f.set_exception(OSError())
    return f


def successful_future():
    f = Future()
    f.set_result(42)
    return f


async def amul(x, y):
    await asyncio.sleep(0.01)
    return x * y


class TestTaskPoolAsCompleted:
    async def test_no_timeout(self, executor):
        future_c = cancelled_future()
        future_e = exception_future()
        future_s = successful_future()
        future1 = await executor.submit(amul, 2, 21)
        future2 = await executor.submit(amul, 7, 6)

        coros = list(
            asyncio.as_completed([future_c, future_e, future_s, future1, future2])
        )
        assert len(coros) == 5
        for coro in coros:
            with suppress(BaseException):
                await coro

        assert all(f.done() for f in (future_c, future_e, future_s, future1, future2))

    @pytest.mark.parametrize("timeout", [0, 0.1])
    async def test_future_times_out(self, executor, timeout):  # noqa: ASYNC109
        """Test asyncio.as_completed timing out before completing its final future."""
        already_completed = {
            cancelled_future(),
            exception_future(),
            successful_future(),
        }
        # Windows clock resolution is around 15.6 ms
        future = await executor.submit(asyncio.sleep, 1.0)
        results = []
        exception_types = set()
        for coro in asyncio.as_completed(already_completed | {future}, timeout=timeout):
            try:
                results.append(await coro)
            except (asyncio.CancelledError, OSError, TimeoutError) as ex:
                exception_types.add(type(ex))

        assert results == [42]
        assert exception_types == {asyncio.CancelledError, OSError, TimeoutError}
        # Check that ``future`` wasn't completed
        assert not future.done()

    async def test_duplicate_futures(self, executor):
        # Issue 20367. Duplicate futures should not raise exceptions or give duplicate
        # responses.
        # Issue #31641: accept arbitrary iterables.
        future1 = await executor.submit(asyncio.sleep, 0.1)
        results = [
            await coro for coro in asyncio.as_completed(itertools.repeat(future1, 3))
        ]
        assert len(results) == 1
        assert future1.done()
