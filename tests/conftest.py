import time

import pytest

from cf_taskpool import TaskPoolExecutor


@pytest.fixture
async def executor():
    t1 = time.monotonic()
    async with TaskPoolExecutor(max_workers=5) as executor:
        yield executor
    dt = time.monotonic() - t1
    assert dt < 300, "synchronization issue: test lasted too long"
