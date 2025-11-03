import asyncio
import re
import signal

import pytest
from test.support.script_helper import assert_python_ok

from cf_taskpool import TaskPoolExecutor


async def amul(x, y):
    await asyncio.sleep(0.01)
    return x * y


async def aabs(x):
    await asyncio.sleep(0.01)
    return abs(x)


def assert_run_taskpool_executor(expected_out, transient, shutdown=None):
    script = f"""
import asyncio
from cf_taskpool import TaskPoolExecutor

async def sleep_and_print(t, msg):
    await asyncio.sleep(t)
    print(msg)

async def main(executor):
    await executor.submit(sleep_and_print, 0.1, "apple")
    if {shutdown} is not None:
        await executor.shutdown(**{shutdown})

if __name__ == "__main__":
    if {transient}:
        asyncio.run(main(TaskPoolExecutor()))
    else:
        executor = TaskPoolExecutor()
        asyncio.run(main(executor))
    """
    _, out, err = assert_python_ok("-c", script, PYTHONPATH=".")
    assert err == b""
    assert out.strip() == expected_out


class TestTaskPoolShutdown:
    async def test_run_after_shutdown(self, executor):
        await executor.shutdown()
        with pytest.raises(RuntimeError):
            await executor.submit(amul, 2, 5)

    async def test_hang_issue12364(self, executor):
        fs = [await executor.submit(asyncio.sleep, 0.1) for _ in range(50)]
        await executor.shutdown()
        for f in fs:
            await f

    async def test_cancel_futures(self):
        executor = TaskPoolExecutor(max_workers=5)
        fs = [await executor.submit(asyncio.sleep, 0.1) for _ in range(50)]
        await executor.shutdown(cancel_futures=True)
        # 5 tasks were picked by the workers before the shutdown, 45 were cancelled
        cancelled = [fut for fut in fs if fut.cancelled()]
        assert len(cancelled) == 45

        # Ensure the other futures were able to finish.
        # Use "not fut.cancelled()" instead of "fut.done()" to include futures
        # that may have been left in a pending state.
        others = [fut for fut in fs if not fut.cancelled()]
        assert len(others) == 5
        for fut in others:
            assert fut.done()
            assert fut.exception() is None

    @pytest.mark.skipif(
        not hasattr(signal, "alarm"), reason="signal.alarm not available"
    )
    async def test_hang_gh94440(self):
        """shutdown(wait=True) doesn't hang when a future was submitted and
        quickly canceled right before shutdown.

        See https://github.com/python/cpython/issues/94440.
        """

        def timeout(_signum, _frame):
            raise RuntimeError("timed out waiting for shutdown")  # pragma: no cover

        executor = TaskPoolExecutor(max_workers=1)
        future = await executor.submit(amul, 2, 5)
        await future
        old_handler = signal.signal(signal.SIGALRM, timeout)
        try:
            signal.alarm(5)
            future = await executor.submit(amul, 2, 5)
            future.cancel()
            await executor.shutdown(wait=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    async def test_tasks_terminate(self, executor):
        async def acquire_lock(lock):
            await lock.acquire()

        sem = asyncio.Semaphore(0)
        for _ in range(3):
            await executor.submit(acquire_lock, sem)
        assert len(executor._tasks) == 3
        for _ in range(3):
            sem.release()
        await executor.shutdown()
        assert all(t.done() for t in executor._tasks)

    async def test_context_manager_shutdown(self):
        async with TaskPoolExecutor(max_workers=5) as executor:
            mapped_abs = [x async for x in await executor.map(aabs, range(-4, 4))]
            assert mapped_abs == list(map(abs, range(-4, 4)))
        assert all(t.done() for t in executor._tasks)

    @pytest.mark.parametrize("explicit_shutdown", [False, True])
    async def test_shutdown_no_wait(self, explicit_shutdown):
        executor = TaskPoolExecutor(max_workers=5)
        res = await executor.map(aabs, range(-5, 5))
        tasks = executor._tasks
        if explicit_shutdown:
            await executor.shutdown(wait=False)
        else:
            del executor

        assert all(not t.done() for t in tasks)
        await asyncio.wait(tasks)
        assert all(t.done() for t in tasks)

        # Make sure the results were all computed before the executor got shutdown
        mapped_abs = [x async for x in res]
        assert all(r == abs(v) for r, v in zip(mapped_abs, range(-5, 5), strict=True))

    @pytest.mark.parametrize(
        ("task_name_prefix", "task_name_pattern"),
        [
            (None, r"^TaskPoolExecutor-\d+_[0-4]$"),
            ("SpecialPool", r"^SpecialPool_[0-4]$"),
        ],
    )
    async def test_task_names(self, task_name_prefix, task_name_pattern):
        executor = TaskPoolExecutor(max_workers=5, task_name_prefix=task_name_prefix)
        await executor.map(aabs, range(-5, 5))
        tasks = executor._tasks
        del executor
        assert all(re.search(task_name_pattern, t.get_name()) for t in tasks)

    @pytest.mark.parametrize("transient", [True, False])
    @pytest.mark.parametrize("wait", [True, False])
    @pytest.mark.parametrize("cancel_futures", [True, False])
    def test_interpreter_shutdown(self, transient, wait, cancel_futures):
        shutdown = {"wait": wait, "cancel_futures": cancel_futures}
        expected = b"apple" if not cancel_futures else b""
        assert_run_taskpool_executor(expected, transient=transient, shutdown=shutdown)

    @pytest.mark.parametrize("transient", [True, False])
    def test_interpreter_shutdown_without_executor_shutdown(self, transient):
        assert_run_taskpool_executor(b"apple", transient=transient)
