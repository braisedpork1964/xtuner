"""SessionServer pause gate.

Requests park at the gate while weights are swapped, instead of being aborted mid-turn.
"""

import asyncio

import pytest

from xtuner.v1.rl.rollout.session_server import SessionServer


@pytest.fixture
def gated():
    instance = SessionServer.__new__(SessionServer)
    instance._gate = asyncio.Event()
    instance._gate.set()
    instance._inflight = 0
    instance._drained = asyncio.Event()
    instance._drained.set()
    instance._model_step = None
    return instance


class TestSessionServerGate:
    @pytest.mark.asyncio
    async def test_open_gate_does_not_block(self, gated):
        await asyncio.wait_for(gated._gate.wait(), timeout=1)

    @pytest.mark.asyncio
    async def test_closed_gate_blocks_until_released(self, gated):
        gated.hold()
        waiter = asyncio.create_task(gated._gate.wait())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)
        gated.release(model_step=7)
        await asyncio.wait_for(waiter, timeout=1)
        assert gated._model_step == 7

    @pytest.mark.asyncio
    async def test_release_without_model_step_keeps_previous(self, gated):
        gated.release(model_step=3)
        gated.release()
        assert gated._model_step == 3

    @pytest.mark.asyncio
    async def test_drain_returns_zero_when_idle(self, gated):
        assert await gated.wait_drained(timeout_s=0.05) == 0

    @pytest.mark.asyncio
    async def test_drain_waits_for_inflight_request(self, gated):
        async def one_request():
            await gated._gate.wait()
            gated._inflight += 1
            gated._drained.clear()
            try:
                await asyncio.sleep(0.1)
            finally:
                gated._inflight -= 1
                if gated._inflight == 0:
                    gated._drained.set()

        task = asyncio.create_task(one_request())
        await asyncio.sleep(0.01)
        assert gated._inflight == 1
        assert await gated.wait_drained(timeout_s=2) == 0
        await task

    @pytest.mark.asyncio
    async def test_drain_reports_requests_that_outlast_the_timeout(self, gated):
        gated._inflight = 2
        gated._drained.clear()
        assert await gated.wait_drained(timeout_s=0.05) == 2

    def test_hold_and_release_are_idempotent(self, gated):
        gated.hold()
        gated.hold()
        assert not gated._gate.is_set()
        gated.release()
        gated.release()
        assert gated._gate.is_set()
