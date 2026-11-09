"""Per-token weight-version accounting for agentic traces.

A sandbox trajectory can span several weight syncs, so each turn records the version that generated it and the
agent loop turns that into ``response_model_steps``, which drives the existing staleness lifecycle.
"""

import pytest

from xtuner.v1.data_proto.rl_data import update_sample_version
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import _response_model_steps
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore, TokenizedSegment


@pytest.fixture
def store():
    # Bypass the ray actor wrapper; the storage logic under test is plain Python.
    return RolloutTraceStore.__ray_metadata__.modified_class()


class TestExportModelSteps:
    def test_model_steps_align_with_input_ids(self, store):
        store.insert("s", key="prompt", value=TokenizedSegment(text="prompt", token_ids=[1, 2]))
        store.insert(
            "s",
            key="promptA",
            value=TokenizedSegment(text="A", token_ids=[3, 4], labels=[3, 4], model_step=5),
        )
        store.insert(
            "s",
            key="promptAB",
            value=TokenizedSegment(text="B", token_ids=[5], labels=[5], model_step=6),
        )

        trace = store.export_training_trace("s", "promptAB")

        assert trace["input_ids"] == [1, 2, 3, 4, 5]
        assert len(trace["model_steps"]) == len(trace["input_ids"])
        # Input tokens carry no version; the two generated turns carry theirs.
        assert trace["model_steps"] == [None, None, 5, 5, 6]

    def test_model_steps_default_to_none(self, store):
        store.insert("s", key="prompt", value=TokenizedSegment(text="prompt", token_ids=[1]))
        store.insert("s", key="promptA", value=TokenizedSegment(text="A", token_ids=[2], labels=[2]))

        trace = store.export_training_trace("s", "promptA")

        assert trace["model_steps"] == [None, None]


class TestResponseModelSteps:
    def test_versions_follow_the_response_tokens(self):
        # labels[0] is dropped alongside input_ids[0], mirroring how response_ids is built.
        data = {"labels": [-100, -100, 3, 4, 5], "model_steps": [None, None, 7, 7, 8]}
        assert _response_model_steps(data) == [7, 7, 8]

    @pytest.mark.parametrize(
        "data",
        [
            {"labels": [-100, 2], "model_steps": []},
            {"labels": [-100, 2], "model_steps": None},
            {"labels": [-100, 2]},
            # One untracked turn makes the whole sequence unusable.
            {"labels": [-100, -100, 3, 4], "model_steps": [None, None, None, 8]},
            # No trainable token at all.
            {"labels": [-100, -100], "model_steps": [None, None]},
        ],
    )
    def test_incomplete_versions_fall_back_to_none(self, data):
        assert _response_model_steps(data) is None


class TestUpdateSampleVersion:
    def test_existing_versions_are_kept(self):
        state = type("S", (), {"response_ids": [1, 2, 3], "response_model_steps": [4, 4, 5]})()
        update_sample_version(state, model_step=9)
        assert state.response_model_steps == [4, 4, 5]

    def test_missing_versions_are_stamped_with_completion_step(self):
        state = type("S", (), {"response_ids": [1, 2, 3], "response_model_steps": None})()
        update_sample_version(state, model_step=9)
        assert state.response_model_steps == [9, 9, 9]
