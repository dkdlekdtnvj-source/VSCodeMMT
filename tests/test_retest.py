import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from optimize import constants as opt_constants
from optimize import retest
from optimize.main_loop import _enforce_exit_guards


@pytest.mark.parametrize("threshold_input", ["600", "600.0"])
def test_run_retest_builds_retest_namespace(tmp_path, monkeypatch, threshold_input):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    run_dir = report_root / "20250101-TEST"
    run_dir.mkdir()

    manifest = {
        "created_at": "2025-01-01T00:00:00Z",
        "run": {"tag": "20250101-TEST", "symbol": "BTCUSDT", "timeframe": "1m"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    seed_payload = {
        "params": {
            "search": {"n_trials": 100, "algo": "bayes"},
            "space": {
                "fixedStopPct": {"type": "float", "min": 0.0, "max": 5.0, "step": 0.5},
                "atrStopLen": {"type": "int", "min": 5, "max": 40, "step": 1},
                "atrStopMult": {"type": "float", "min": 0.5, "max": 5.0, "step": 0.1},
                "useChandelierExit": {"type": "bool"},
                "chandelierLen": {"type": "int", "min": 5, "max": 50, "step": 1},
                "chandelierMult": {"type": "float", "min": 1.0, "max": 4.0, "step": 0.1},
                "useSarExit": {"type": "bool"},
                "sarStart": {"type": "float", "min": 0.005, "max": 0.05, "step": 0.005},
                "sarIncrement": {"type": "float", "min": 0.005, "max": 0.05, "step": 0.005},
                "sarMaximum": {"type": "float", "min": 0.05, "max": 0.4, "step": 0.01},
                "leverage": {"type": "int", "min": 2, "max": 40, "step": 1},
                "usePyramiding": {"type": "bool"},
            },
            "overrides": {"entry_tf": "1m"},
            "llm": {"enabled": False},
        },
        "backtest": {"fees": {"maker": 0.0002}},
    }
    (run_dir / "seed.yaml").write_text(yaml.safe_dump(seed_payload, sort_keys=False), encoding="utf-8")

    df = pd.DataFrame(
        [
            {
                "TotalAssets": 650.0,
                "fixedStopPct": 1.5,
                "atrStopLen": 14,
                "atrStopMult": 2.0,
                "useChandelierExit": True,
                "chandelierLen": 22,
                "chandelierMult": 2.5,
                "useSarExit": False,
                "sarStart": 0.02,
                "sarIncrement": 0.02,
                "sarMaximum": 0.2,
                "leverage": 7,
                "usePyramiding": True,
                "Score": 12.3,
                "EntryTF": "1m",
            },
            {
                "TotalAssets": 580.0,
                "fixedStopPct": 2.0,
                "atrStopLen": 10,
                "atrStopMult": 1.5,
                "useChandelierExit": False,
                "chandelierLen": 0,
                "chandelierMult": 0.0,
                "useSarExit": False,
                "sarStart": 0.02,
                "sarIncrement": 0.02,
                "sarMaximum": 0.2,
                "leverage": 5,
                "usePyramiding": False,
                "Score": 9.1,
                "EntryTF": "3m",
            },
        ]
    )
    df.to_csv(run_dir / "results.csv", index=False)

    monkeypatch.setattr(opt_constants, "DEFAULT_REPORT_ROOT", report_root)
    monkeypatch.setattr(opt_constants, "DEFAULT_STORAGE_ROOT", tmp_path / "storage")

    inputs = iter(["1", threshold_input])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    captured: dict[str, object] = {}

    def _fake_execute(args, argv):
        params_cfg = yaml.safe_load(Path(args.params).read_text(encoding="utf-8"))
        backtest_cfg = yaml.safe_load(Path(args.backtest).read_text(encoding="utf-8"))
        bank_payload = json.loads(Path(args.resume_from).read_text(encoding="utf-8"))
        captured["params"] = params_cfg
        captured["backtest"] = backtest_cfg
        captured["bank"] = bank_payload
        captured["output"] = Path(args.output)
        captured["argv"] = list(argv)

    retest.run_retest(execute_fn=_fake_execute)

    params_cfg = captured["params"]
    assert params_cfg["llm"]["enabled"] is True
    assert set(params_cfg["space"].keys()) == set(retest.RETEST_SPACE_KEYS)
    for key, value in retest.BASE_PARAMETER_OVERRIDES.items():
        assert params_cfg["overrides"][key] == value
    assert params_cfg["overrides"]["entry_tf"] == "1m"

    bank_payload = captured["bank"]
    assert params_cfg["search"]["n_trials"] == len(bank_payload["entries"]) * 10
    assert len(bank_payload["entries"]) == 1
    entry_params = bank_payload["entries"][0]["params"]
    assert entry_params["leverage"] == 7
    assert bank_payload["space_hash"]

    output_dir = captured["output"]
    assert output_dir.parent == report_root
    assert output_dir.name.startswith("Retest")
    assert "20250101-TEST" in output_dir.name

    argv = captured["argv"]
    assert "--resume-from" in argv


def test_retest_scales_trials_with_multiple_candidates(tmp_path, monkeypatch):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    run_dir = report_root / "20250102-TEST"
    run_dir.mkdir()

    manifest = {
        "created_at": "2025-01-02T00:00:00Z",
        "run": {"tag": "20250102-TEST", "symbol": "ETHUSDT", "timeframe": "3m"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    seed_payload = {
        "params": {
            "search": {"n_trials": 50},
            "space": {
                key: {"type": "float", "min": 0.0, "max": 5.0, "step": 0.5}
                for key in ["fixedStopPct", "atrStopMult", "chandelierMult"]
            },
        },
    }
    (run_dir / "seed.yaml").write_text(yaml.safe_dump(seed_payload, sort_keys=False), encoding="utf-8")

    df = pd.DataFrame(
        [
            {"TotalAssets": 700.0, "fixedStopPct": 1.0},
            {"TotalAssets": 680.0, "fixedStopPct": 1.5},
            {"TotalAssets": 660.0, "fixedStopPct": 2.0},
        ]
    )
    df.to_csv(run_dir / "results.csv", index=False)

    monkeypatch.setattr(opt_constants, "DEFAULT_REPORT_ROOT", report_root)
    monkeypatch.setattr(opt_constants, "DEFAULT_STORAGE_ROOT", tmp_path / "storage")

    inputs = iter(["1", "670"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    captured: dict[str, object] = {}

    def _fake_execute(args, argv):
        captured["params"] = yaml.safe_load(Path(args.params).read_text(encoding="utf-8"))
        captured["bank"] = json.loads(Path(args.resume_from).read_text(encoding="utf-8"))

    retest.run_retest(execute_fn=_fake_execute)

    params_cfg = captured["params"]
    bank_payload = captured["bank"]
    assert len(bank_payload["entries"]) == 2
    assert params_cfg["search"]["n_trials"] == len(bank_payload["entries"]) * 10


@pytest.mark.parametrize(
    "params, expected_exit, expected_mom",
    [
        (
            {"exitOpposite": False, "useMomFade": False},
            True,
            False,
        ),
        (
            {"exitOpposite": False, "useMomFade": False, "fixedStopPct": 1.0},
            False,
            False,
        ),
        (
            {"exitOpposite": False, "useMomFade": False, "atrStopLen": 10},
            True,
            False,
        ),
        (
            {"exitOpposite": False, "useMomFade": False, "useChandelierExit": True},
            False,
            False,
        ),
        (
            {
                "exitOpposite": False,
                "useMomFade": False,
                "atrStopLen": 14,
                "atrStopMult": 1.8,
            },
            False,
            False,
        ),
        (
            {
                "exitOpposite": False,
                "useMomFade": False,
                "useSarExit": True,
                "atrStopLen": 14,
                "atrStopMult": None,
            },
            False,
            False,
        ),
    ],
)
def test_enforce_exit_guards_honours_alternative_exits(params, expected_exit, expected_mom):
    result = _enforce_exit_guards(params, context="unit-test")
    assert result["exitOpposite"] is expected_exit
    assert result["useMomFade"] is expected_mom
