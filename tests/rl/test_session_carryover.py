"""Carrying unfinished sandbox rollouts across produce_batch calls.

With session carryover on, a rollout that outlives a weight sync stays pending instead of being cancelled, so its
sandbox survives and the trajectory continues under the new weights.
"""

import asyncio

import pytest

from xtuner.v1.rl.agent_loop_manager.produce_utils import produced_token_count
from xtuner.v1.rl.agent_loop_manager.producer import AsyncProduceStrategy, AsyncProduceStrategyConfig


def build_strategy(*, carryover=True, max_staleness=1, sync_weights_interval=1):
    return AsyncProduceStrategy(
        over_sample_threshold=0.0,
        enable_partial_rollout=False,
        enable_session_carryover=carryover,
        tail_batch_trigger_size=-1,
        max_staleness=max_staleness,
        max_token_staleness=None,
        sync_weights_interval=sync_weights_interval,
        is_valid_sample_fn=lambda samples: True,
        should_continue_fn=lambda completed, target, **kw: completed < target,
    )


class _Ctx:
    def __init__(self, train_step, model_step=0):
        self.train_step = train_step
        self.model_step = model_step
        self.task_name = "t"
        self.put_calls: list[object] = []

    async def put_generated_group(self, group):
        self.put_calls.append(group)
        return True


class TestCarryoverConfig:
    def test_carryover_requires_a_staleness_budget(self):
        with pytest.raises(ValueError, match="max_staleness >= 1"):
            AsyncProduceStrategyConfig(enable_session_carryover=True, max_staleness=0).build()

    def test_carryover_builds_with_a_staleness_budget(self):
        strategy = AsyncProduceStrategyConfig(enable_session_carryover=True, max_staleness=1).build()
        assert strategy.enable_session_carryover is True


class TestCarryOverPending:
    @pytest.mark.asyncio
    async def test_unfinished_task_is_kept_for_the_next_step(self):
        strategy = build_strategy()
        task = asyncio.create_task(asyncio.sleep(5))
        strategy._local_pending_tasks.add(task)
        strategy._task_start_step[task] = 1

        pause_time = await strategy.pause_produce(_Ctx(train_step=1), drain_pending=False)

        assert pause_time == 0.0
        assert strategy.pending_task_count() == 1
        assert strategy.consume_carried_counts() == (1, 0)
        task.cancel()

    @pytest.mark.asyncio
    async def test_finished_task_is_settled_immediately(self):
        strategy = build_strategy()

        async def done():
            return ["group"]

        task = asyncio.create_task(done())
        await task
        strategy._local_pending_tasks.add(task)
        strategy._task_start_step[task] = 1
        ctx = _Ctx(train_step=1)

        await strategy.pause_produce(ctx, drain_pending=False)

        assert ctx.put_calls == [["group"]]
        assert strategy.pending_task_count() == 0

    @pytest.mark.asyncio
    async def test_task_that_would_expire_next_step_is_cancelled(self):
        # stale_threshold = (max_staleness + 1) * interval = 2; started at step 0 and consumed at step 2 expires.
        strategy = build_strategy(max_staleness=1)
        task = asyncio.create_task(asyncio.sleep(5))
        strategy._local_pending_tasks.add(task)
        strategy._task_start_step[task] = 0

        await strategy.pause_produce(_Ctx(train_step=2), drain_pending=False)

        assert strategy.pending_task_count() == 0
        assert strategy.consume_carried_counts() == (0, 1)
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_drain_pending_clears_everything(self):
        strategy = build_strategy()

        async def done():
            return ["group"]

        task = asyncio.create_task(done())
        await task
        strategy._local_pending_tasks.add(task)
        strategy._task_start_step[task] = 1
        ctx = _Ctx(train_step=1)
        ctx.agent_loop = type("L", (), {"pause": staticmethod(lambda: asyncio.sleep(0))})()

        await strategy.pause_produce(ctx, drain_pending=True)

        assert strategy.pending_task_count() == 0
        assert strategy._task_start_step == {}

    @pytest.mark.asyncio
    async def test_carryover_off_always_drains(self):
        strategy = build_strategy(carryover=False)

        async def done():
            return ["group"]

        task = asyncio.create_task(done())
        await task
        strategy._local_pending_tasks.add(task)
        ctx = _Ctx(train_step=1)
        ctx.agent_loop = type("L", (), {"pause": staticmethod(lambda: asyncio.sleep(0))})()

        await strategy.pause_produce(ctx, drain_pending=False)

        assert strategy.pending_task_count() == 0


class TestProducedTokenCount:
    def test_unversioned_response_counts_in_full(self):
        state = type("S", (), {"response_ids": [1, 2, 3], "response_model_steps": None})()
        assert produced_token_count(state, model_step=4) == 3

    def test_only_tokens_from_this_window_count(self):
        # A carried-over trajectory: two turns from step 4, one from step 5.
        state = type("S", (), {"response_ids": [1, 2, 3], "response_model_steps": [4, 4, 5]})()
        assert produced_token_count(state, model_step=5) == 1

    def test_unversioned_tail_counts_too(self):
        # Token-level partial rollout: old tokens versioned, newly generated tail not yet stamped.
        state = type("S", (), {"response_ids": [1, 2, 3], "response_model_steps": [4]})()
        assert produced_token_count(state, model_step=5) == 2
