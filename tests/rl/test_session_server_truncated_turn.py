"""SessionServer truncated-turn detection.

Covers the rule that decides whether an assistant turn reached a natural stop: engine aborts arrive as HTTP 200
with partial content, so the only reliable signal is a trailing eos token in ``output_ids``.
"""

import pytest

from xtuner.v1.rl.rollout.session_server import (
    FMT_ANTHROPIC,
    FMT_OPENAI,
    SessionServer,
    _response_output_ids,
    _response_routed_experts,
)


EOS = 151645


@pytest.fixture
def server():
    # Bypass __init__ so the test needs neither a tokenizer nor the trace-store actor.
    instance = SessionServer.__new__(SessionServer)
    instance.eos_token_id = EOS
    instance.reject_truncated_response = True
    return instance


def openai_response(output_ids, routed_experts=None, finish_reason="abort"):
    return {"choices": [{"message": {"role": "assistant", "content": "x"}, "output_ids": output_ids,
                         "finish_reason": finish_reason, "routed_experts": routed_experts}]}


def anthropic_response(output_ids, routed_experts=None, finish_reason="stop_sequence"):
    return {"role": "assistant", "content": [], "output_ids": output_ids, "stop_reason": finish_reason,
            "routed_experts": routed_experts}


class TestResponseFieldAccess:
    @pytest.mark.parametrize(
        "resp, fmt, expected",
        [
            (openai_response([1, 2, EOS]), FMT_OPENAI, [1, 2, EOS]),
            (anthropic_response([1, 2, EOS]), FMT_ANTHROPIC, [1, 2, EOS]),
            ({"choices": []}, FMT_OPENAI, None),
            ({}, FMT_OPENAI, None),
            ({}, FMT_ANTHROPIC, None),
        ],
    )
    def test_output_ids_read_from_either_format(self, resp, fmt, expected):
        assert _response_output_ids(resp, fmt) == expected

    @pytest.mark.parametrize(
        "resp, fmt",
        [
            (openai_response([EOS], routed_experts="key-abc"), FMT_OPENAI),
            (anthropic_response([EOS], routed_experts="key-abc"), FMT_ANTHROPIC),
        ],
    )
    def test_routed_experts_read_from_either_format(self, resp, fmt):
        assert _response_routed_experts(resp, fmt) == "key-abc"


class TestIsTruncatedTurn:
    @pytest.mark.parametrize("fmt, build", [(FMT_OPENAI, openai_response), (FMT_ANTHROPIC, anthropic_response)])
    def test_turn_ending_on_eos_is_complete(self, server, fmt, build):
        assert server.is_truncated_turn(build([10, 11, EOS]), fmt) is False

    @pytest.mark.parametrize("fmt, build", [(FMT_OPENAI, openai_response), (FMT_ANTHROPIC, anthropic_response)])
    def test_turn_without_trailing_eos_is_truncated(self, server, fmt, build):
        # What an engine abort looks like: HTTP 200, real tokens, no eos.
        assert server.is_truncated_turn(build([10, 11, 12]), fmt) is True

    @pytest.mark.parametrize("output_ids", [[], None])
    def test_turn_without_output_ids_is_truncated(self, server, output_ids):
        # A request aborted before inference started reports an empty generation.
        assert server.is_truncated_turn(openai_response(output_ids), FMT_OPENAI) is True

    def test_eos_in_the_middle_does_not_count(self, server):
        assert server.is_truncated_turn(openai_response([EOS, 10, 11]), FMT_OPENAI) is True

    @pytest.mark.parametrize(
        "fmt, build, reason",
        [
            (FMT_OPENAI, openai_response, "length"),
            (FMT_ANTHROPIC, anthropic_response, "max_tokens"),
            # lmdeploy rewrites stop/length to parse_error when its completeness check fails; the turn still
            # ended on its own terms. Observed on the startup health check, which caps max_tokens at 16.
            (FMT_OPENAI, openai_response, "parse_error"),
            (FMT_ANTHROPIC, anthropic_response, "parse_error"),
        ],
    )
    def test_self_terminated_turn_without_eos_is_not_truncated(self, server, fmt, build, reason):
        # The caller asked for a short answer: no eos, but the turn ended exactly as requested.
        assert server.is_truncated_turn(build([10, 11, 12], finish_reason=reason), fmt) is False

    def test_length_cap_still_requires_output(self, server):
        assert server.is_truncated_turn(openai_response([], finish_reason="length"), FMT_OPENAI) is True
