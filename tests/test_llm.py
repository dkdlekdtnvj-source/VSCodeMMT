"""`optimize.llm` 모듈에 대한 테스트."""
from types import SimpleNamespace

import optuna
from optuna.trial import TrialState

from optimize.llm import LLMSuggestions, generate_llm_candidates, _validate_candidate
from optimize.search_spaces import SpaceSpec


def test_validate_candidate_accepts_string_choices() -> None:
    space: SpaceSpec = {
        "optimizer": {"type": "str", "values": ["Adam", "SGD", "RMSprop"]}
    }
    candidate = {"optimizer": " adam "}

    validated = _validate_candidate(candidate, space)

    assert validated == {"optimizer": "Adam"}


def test_generate_llm_candidates_filters_unsupported_kwargs(monkeypatch) -> None:
    space: SpaceSpec = {"x": {"type": "int", "min": 0, "max": 5}}
    trial = optuna.trial.create_trial(
        params={"x": 1},
        distributions={"x": optuna.distributions.IntDistribution(0, 5)},
        value=1.0,
        state=TrialState.COMPLETE,
    )

    class DummyAnalysisResponse:
        def __init__(self):
            self.text = "{\"space\": {\"x\": {\"min\": 0, \"max\": 5}}}"

    class DummyCandidateResponse:
        def __init__(self):
            self.text = "{\"candidates\": [{\"x\": 1}], \"insights\": [\"ok\"]}"

    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise TypeError("generate_content() got an unexpected keyword argument 'system_instruction'")
            if len(self.calls) == 2:
                return DummyAnalysisResponse()
            return DummyCandidateResponse()

    dummy_models = DummyModels()
    dummy_client = SimpleNamespace(models=dummy_models)
    monkeypatch.setattr(
        "optimize.llm.genai",
        SimpleNamespace(Client=lambda api_key: dummy_client, types=None),
    )
    monkeypatch.setattr(
        "optimize.llm.optuna.importance.get_param_importances",
        lambda study: {"x": 0.9},
    )

    config = {
        "enabled": True,
        "api_key": "dummy",
        "count": 1,
        "top_n": 1,
        "system_instruction": "Focus on profit factor",
        "candidate_count": 2,
    }

    suggestions: LLMSuggestions = generate_llm_candidates(space, [trial], config)

    assert suggestions.candidates == [{"x": 1}]
    assert suggestions.insights == ["ok"]
    assert len(dummy_models.calls) == 3
    assert "system_instruction" in dummy_models.calls[0]
    assert "system_instruction" not in dummy_models.calls[1]


def test_generate_llm_candidates_switches_model_on_permission_error(monkeypatch) -> None:
    space: SpaceSpec = {"x": {"type": "int", "min": 0, "max": 5}}
    trial = optuna.trial.create_trial(
        params={"x": 1},
        distributions={"x": optuna.distributions.IntDistribution(0, 5)},
        value=1.0,
        state=TrialState.COMPLETE,
    )

    class DummyAnalysisResponse:
        def __init__(self):
            self.text = "{\"space\": {\"x\": {\"min\": 0, \"max\": 5}}}"

    class DummyCandidateResponse:
        def __init__(self):
            self.text = "{\"candidates\": [{\"x\": 1}], \"insights\": []}"

    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            model_name = kwargs.get("model")
            if model_name in {"gemini-2.5-pro", "gemini-2.5-flash"}:
                raise RuntimeError("PERMISSION_DENIED: model not available")
            if len(self.calls) <= 3:
                return DummyAnalysisResponse()
            return DummyCandidateResponse()

    dummy_models = DummyModels()
    dummy_client = SimpleNamespace(models=dummy_models)
    monkeypatch.setattr(
        "optimize.llm.genai",
        SimpleNamespace(Client=lambda api_key: dummy_client, types=None),
    )
    monkeypatch.setattr(
        "optimize.llm.optuna.importance.get_param_importances",
        lambda study: {"x": 1.0},
    )

    config = {
        "enabled": True,
        "api_key": "dummy",
        "model": "gemini-2.5-pro",
        "fallback_models": ["gemini-2.5-flash"],
        "count": 1,
        "top_n": 1,
    }

    suggestions: LLMSuggestions = generate_llm_candidates(space, [trial], config)

    assert suggestions.candidates == [{"x": 1}]
    assert len(dummy_models.calls) == 4
    assert dummy_models.calls[0]["model"] == "gemini-2.5-pro"
    assert dummy_models.calls[1]["model"] == "gemini-2.5-flash"
    assert dummy_models.calls[2]["model"] == "gemini-2.0-flash"
    assert dummy_models.calls[3]["model"] == "gemini-2.0-flash"


def test_generate_llm_candidates_retries_on_rate_limit(monkeypatch) -> None:
    space: SpaceSpec = {"x": {"type": "int", "min": 0, "max": 5}}
    trial = optuna.trial.create_trial(
        params={"x": 2},
        distributions={"x": optuna.distributions.IntDistribution(0, 5)},
        value=1.0,
        state=TrialState.COMPLETE,
    )

    class DummyAnalysisResponse:
        def __init__(self):
            self.text = "{\"space\": {\"x\": {\"min\": 0, \"max\": 5}}}"

    class DummyCandidateResponse:
        def __init__(self):
            self.text = "{\"candidates\": [{\"x\": 2}], \"insights\": []}"

    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) < 3:
                raise RuntimeError("429 Too Many Requests: quota exceeded")
            if len(self.calls) == 3:
                return DummyAnalysisResponse()
            return DummyCandidateResponse()

    dummy_models = DummyModels()
    dummy_client = SimpleNamespace(models=dummy_models)
    monkeypatch.setattr(
        "optimize.llm.genai",
        SimpleNamespace(Client=lambda api_key: dummy_client, types=None),
    )
    monkeypatch.setattr(
        "optimize.llm.optuna.importance.get_param_importances",
        lambda study: {"x": 0.5},
    )
    monkeypatch.setattr("optimize.llm.time.sleep", lambda *_: None)

    config = {
        "enabled": True,
        "api_key": "dummy",
        "model": "gemini-2.5-pro",
        "count": 1,
        "top_n": 1,
    }

    suggestions: LLMSuggestions = generate_llm_candidates(space, [trial], config)

    assert suggestions.candidates == [{"x": 2}]
    assert len(dummy_models.calls) == 4
    assert all(call["model"] == "gemini-2.5-pro" for call in dummy_models.calls)


def test_generate_llm_candidates_parses_nested_string_payload(monkeypatch) -> None:
    space: SpaceSpec = {"x": {"type": "int", "min": 0, "max": 5}}
    trial = optuna.trial.create_trial(
        params={"x": 1},
        distributions={"x": optuna.distributions.IntDistribution(0, 5)},
        value=1.0,
        state=TrialState.COMPLETE,
    )

    class DummyAnalysisResponse:
        def __init__(self):
            self.text = "{\"space\": {\"x\": {\"min\": 0, \"max\": 5}}}"

    class DummyCandidateResponse:
        def __init__(self):
            self.text = (
                "{\"candidates\": [{\"content\": {\"parts\": [\"{'text': '{\\\"candidates\\\": [{\\\"x\\\": 4}], "
                "\\\"insights\\\": [\\\"deep insight\\\"]}', 'function_call': None}\"]}}]}"
            )

    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return DummyAnalysisResponse()
            return DummyCandidateResponse()

    dummy_models = DummyModels()
    dummy_client = SimpleNamespace(models=dummy_models)
    monkeypatch.setattr(
        "optimize.llm.genai",
        SimpleNamespace(Client=lambda api_key: dummy_client, types=None),
    )
    monkeypatch.setattr(
        "optimize.llm.optuna.importance.get_param_importances",
        lambda study: {"x": 0.5},
    )

    config = {
        "enabled": True,
        "api_key": "dummy",
        "count": 1,
        "top_n": 1,
    }

    suggestions: LLMSuggestions = generate_llm_candidates(space, [trial], config)

    assert suggestions.candidates == [{"x": 4}]
    assert suggestions.insights == ["deep insight"]
    assert len(dummy_models.calls) == 2
