from types import SimpleNamespace
from optuna.trial import TrialState

from optimize.llm import LLMSuggestions
from optimize.run import LLMCandidateRefresher


class DummyStudy:
    def __init__(self) -> None:
        self.enqueued = []
        self.trials = []

    def enqueue_trial(self, params, skip_if_exists=True):  # noqa: D401 - 단순 저장
        self.enqueued.append(params)


def _completed_trial(params, number):
    return SimpleNamespace(state=TrialState.COMPLETE, params=params, number=number)


def test_llm_refresher_enqueues_candidates(monkeypatch):
    space = {"a": {"type": "int", "min": 1, "max": 5, "step": 1}}
    config = {"enabled": True, "refresh_trials": 2, "count": 2}
    refresher = LLMCandidateRefresher(
        space,
        config,
        forced_params={"fixed": 99},
        use_basic_factors=False,
    )

    assert refresher.enabled

    suggestions = LLMSuggestions(
        candidates=[{"a": 3}, {"a": 4}],
        insights=["추가 튜닝 제안"],
    )

    calls = []

    def fake_generate(space_arg, trials_arg, cfg_arg):
        calls.append((space_arg, trials_arg, cfg_arg))
        return suggestions

    monkeypatch.setattr("optimize.run.generate_llm_candidates", fake_generate)

    study = DummyStudy()

    first = _completed_trial({"a": 1}, 0)
    study.trials.append(first)
    refresher(study, first)

    assert not study.enqueued
    assert not calls

    second = _completed_trial({"a": 2}, 1)
    study.trials.append(second)
    refresher(study, second)

    assert len(calls) == 1
    assert len(study.enqueued) == 2
    assert all(candidate["fixed"] == 99 for candidate in study.enqueued)
    assert refresher.total_enqueued == 2
    assert refresher.total_refreshes == 1
    assert "추가 튜닝 제안" in refresher.collected_insights
    assert calls[0][2]["top_n"] == 10
    assert calls[0][2]["bottom_n"] == 10
    assert refresher.config.get("top_n") is None


def test_llm_refresher_expands_rank_windows(monkeypatch):
    space = {"a": {"type": "int", "min": 1, "max": 5, "step": 1}}
    config = {"enabled": True, "refresh_trials": 1, "top_n": 5, "bottom_n": 7, "count": 1}
    refresher = LLMCandidateRefresher(space, config)

    suggestions = LLMSuggestions(candidates=[{"a": 3}], insights=[])

    captured_configs = []

    def fake_generate(space_arg, trials_arg, cfg_arg):
        captured_configs.append(cfg_arg)
        return suggestions

    monkeypatch.setattr("optimize.run.generate_llm_candidates", fake_generate)

    study = DummyStudy()

    for index in range(4):
        trial = _completed_trial({"a": index + 1}, index)
        study.trials.append(trial)
        refresher(study, trial)

    assert len(captured_configs) == 4
    assert captured_configs[0]["top_n"] == 5
    assert captured_configs[0]["bottom_n"] == 7
    assert captured_configs[1]["top_n"] == 15
    assert captured_configs[1]["bottom_n"] == 17
    assert captured_configs[2]["top_n"] == 25
    assert captured_configs[2]["bottom_n"] == 27
    assert captured_configs[3]["top_n"] == 35
    assert captured_configs[3]["bottom_n"] == 37
