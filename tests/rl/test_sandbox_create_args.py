"""Sandbox create-arg expansion.

Per-task pool overrides carry secrets as ``${VAR}`` references so annotations stay shareable; they are resolved
from the environment right before the create call.
"""

import pytest

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import _expand_env_refs
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import SandboxSpec


class TestExpandEnvRefs:
    def test_whole_string_reference_is_resolved(self, monkeypatch):
        monkeypatch.setenv("SANDBOX_TEST_AKSK", "ak:sk")
        assert _expand_env_refs("${SANDBOX_TEST_AKSK}") == "ak:sk"

    @pytest.mark.parametrize(
        "value",
        ["plain", "prefix-${SANDBOX_TEST_AKSK}", "${SANDBOX_TEST_AKSK}-suffix", "$SANDBOX_TEST_AKSK", "${}"],
    )
    def test_only_a_whole_string_reference_counts(self, value, monkeypatch):
        monkeypatch.setenv("SANDBOX_TEST_AKSK", "ak:sk")
        assert _expand_env_refs(value) == value

    def test_nested_structures_are_walked(self, monkeypatch):
        monkeypatch.setenv("SANDBOX_TEST_AKSK", "ak:sk")
        payload = {
            "project": "ailab-llmit",
            "aksk": "${SANDBOX_TEST_AKSK}",
            "volumes": [{"hostPath": "/llm-ai-infra", "token": "${SANDBOX_TEST_AKSK}"}],
        }
        assert _expand_env_refs(payload) == {
            "project": "ailab-llmit",
            "aksk": "ak:sk",
            "volumes": [{"hostPath": "/llm-ai-infra", "token": "ak:sk"}],
        }

    def test_missing_env_var_fails_loudly(self, monkeypatch):
        monkeypatch.delenv("SANDBOX_TEST_MISSING", raising=False)
        with pytest.raises(KeyError, match="SANDBOX_TEST_MISSING"):
            _expand_env_refs({"aksk": "${SANDBOX_TEST_MISSING}"})

    @pytest.mark.parametrize("value", [None, 7, True, 1.5])
    def test_non_string_leaves_pass_through(self, value):
        assert _expand_env_refs(value) is value


class TestSandboxSpecDboxFields:
    def test_dbox_override_is_accepted(self):
        # Shape written by scripts/generate_dbox_annotations.py into pipeline_overrides.pool.specs.main.
        spec = SandboxSpec(
            image="registry.example/data_querying:v0.0.1",
            env="registry.example/data_querying:v0.0.1",
            cluster_name="dbox",
            extra_params={
                "project": "ailab-llmit",
                "aksk": "${SANDBOX_PROVIDER_AKSK}",
                "volumes": [{"hostPath": "/llm-ai-infra", "mountPath": "/sandbox-data/llm-ai-infra"}],
            },
        )
        assert spec.cluster_name == "dbox"
        assert spec.extra_params["aksk"] == "${SANDBOX_PROVIDER_AKSK}"

    def test_defaults_keep_non_dbox_runs_unchanged(self):
        spec = SandboxSpec(image="t-data-processing-v1")
        assert spec.env is None
        assert spec.extra_params == {}
        assert spec.user is None

    def test_user_can_be_an_env_reference(self, monkeypatch):
        # The gateway rejects a create without an owner (422 "user is required").
        monkeypatch.setenv("SANDBOX_TEST_USER", "wangziyi")
        spec = SandboxSpec(image="img", user="${SANDBOX_TEST_USER}")
        assert _expand_env_refs(spec.user) == "wangziyi"
