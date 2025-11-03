import asyncio
from collections.abc import Awaitable, Callable

from cf_taskpool import TaskPoolExecutor

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


def submit[R, **P](
    executor: TaskPoolExecutor,
    as_awaitable: bool,  # noqa: FBT001
    func: Callable[P, Awaitable[R]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Awaitable[asyncio.Future[R]]:
    if as_awaitable:
        return executor.submit(func(*args, **kwargs))
    return executor.submit(func, *args, **kwargs)


async def aabs(x):
    await asyncio.sleep(0.01)
    return abs(x)


async def amul(x, y):
    await asyncio.sleep(0.01)
    return x * y


async def adivmod(x, y, *, cancel_if_zero=False):
    await asyncio.sleep(0.01)
    if y == 0 and cancel_if_zero:
        raise asyncio.CancelledError
    return divmod(x, y)
