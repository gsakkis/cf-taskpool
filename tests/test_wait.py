import asyncio
import sys
from concurrent import futures

from . import amul, cancelled_future, exception_future, successful_future


class TestTaskPoolWait:
    async def test_20369(self, executor):
        # See https://bugs.python.org/issue20369
        future = await executor.submit(amul, 1, 2)
        done, not_done = await asyncio.wait(
            [future, future], return_when=futures.ALL_COMPLETED
        )
        assert done == {future}
        assert not_done == set()

    async def test_first_completed(self, executor):
        event = asyncio.Event()
        future1 = await executor.submit(amul, 21, 2)
        future2 = await executor.submit(event.wait)
        try:
            done, not_done = await asyncio.wait(
                [future1, future2], return_when=futures.FIRST_COMPLETED
            )
            assert done == {future1}
            assert not_done == {future2}
        finally:
            event.set()
        await future2  # wait for job to finish

    async def test_first_completed_some_already_completed(self, executor):
        event = asyncio.Event()
        future_c = cancelled_future()
        future_s = successful_future()
        future1 = await executor.submit(event.wait)
        try:
            finished, pending = await asyncio.wait(
                [future_c, future_s, future1], return_when=futures.FIRST_COMPLETED
            )
            assert finished == {future_c, future_s}
            assert pending == {future1}
        finally:
            event.set()
        await future1  # wait for job to finish

    async def test_first_exception(self, executor):
        async def wait_and_raise(event):
            await event.wait()
            raise RuntimeError("this is an exception")

        event1 = asyncio.Event()
        event2 = asyncio.Event()
        try:
            future1 = await executor.submit(amul, 2, 21)
            future2 = await executor.submit(wait_and_raise, event1)
            future3 = await executor.submit(event2.wait)

            # Ensure that future1 is completed before future2 finishes
            async def wait_for_future1():
                await future1
                event1.set()

            task = asyncio.ensure_future(wait_for_future1())
            finished, pending = await asyncio.wait(
                [future1, future2, future3], return_when=futures.FIRST_EXCEPTION
            )
            assert task.done()
            assert finished == {future1, future2}
            assert pending == {future3}
        finally:
            event1.set()
            event2.set()
        await future3  # wait for job to finish

    async def test_first_exception_some_already_complete(self, executor):
        event = asyncio.Event()
        future_s = successful_future()
        future_c = cancelled_future()
        future1 = await executor.submit(divmod, 21, 0)
        future2 = await executor.submit(event.wait)
        try:
            finished, pending = await asyncio.wait(
                [future_s, future_c, future1, future2],
                return_when=futures.FIRST_EXCEPTION,
            )
            assert finished == {future_s, future_c, future1}
            assert pending == {future2}
        finally:
            event.set()
        await future2  # wait for job to finish

    async def test_first_exception_one_already_failed(self, executor):
        event = asyncio.Event()
        future_e = exception_future()
        future1 = await executor.submit(event.wait)
        try:
            finished, pending = await asyncio.wait(
                [future_e, future1], return_when=futures.FIRST_EXCEPTION
            )
            assert finished == {future_e}
            assert pending == {future1}
        finally:
            event.set()
        await future1  # wait for job to finish

    async def test_all_completed(self, executor):
        future_s = successful_future()
        future_c = cancelled_future()
        future_e = exception_future()
        future1 = await executor.submit(divmod, 2, 0)
        future2 = await executor.submit(amul, 2, 21)

        finished, pending = await asyncio.wait(
            [future_s, future_c, future_e, future1, future2],
            return_when=futures.ALL_COMPLETED,
        )
        assert finished == {future_s, future_c, future_e, future1, future2}
        assert pending == set()

    async def test_timeout(self, executor):
        short_timeout = 0.050
        event = asyncio.Event()
        future_c = cancelled_future()
        future_e = exception_future()
        future_s = successful_future()
        future = await executor.submit(event.wait)
        try:
            finished, pending = await asyncio.wait(
                [future_c, future_e, future_s, future],
                timeout=short_timeout,
                return_when=futures.ALL_COMPLETED,
            )
            assert finished == {future_c, future_e, future_s}
            assert pending == {future}
        finally:
            event.set()
        await future  # wait for job to finish

    async def test_pending_calls_race(self, executor):
        # Issue #14406: multi-threaded race condition when waiting on all futures.
        event = asyncio.Event()

        async def future_func():
            await event.wait()

        oldswitchinterval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            fs = {await executor.submit(future_func) for i in range(100)}
            event.set()
            await asyncio.wait(fs, return_when=futures.ALL_COMPLETED)
        finally:
            sys.setswitchinterval(oldswitchinterval)
