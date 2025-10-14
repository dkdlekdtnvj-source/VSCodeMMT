import json
from pathlib import Path
from typing import Optional

import matplotlib
import pandas as pd

from optimize.report import generate_reports, write_trials_dataframe

matplotlib.use("Agg")


def _make_dataset(symbol: str, timeframe: str, htf: Optional[str], metrics: dict) -> dict:
    meta = {
        "symbol": symbol,
        "source_symbol": symbol,
        "timeframe": timeframe,
        "from": "2024-01-01",
        "to": "2025-09-25",
        "htf_timeframe": htf,
    }
    payload = {"name": f"{symbol}_{timeframe}_{htf}", "meta": meta, "metrics": metrics}
    return payload


def test_generate_reports_emits_timeframe_summary(tmp_path: Path) -> None:
    dataset_metrics_a = {
        "Valid": True,
        "NetProfit": 0.25,
        "TotalAssets": 680.0,
        "Liquidations": 0.0,
        "MaxDD": -0.12,
        "WinRate": 0.58,
        "WeeklyNetProfit": 0.015,
        "Trades": 140,
    }
    dataset_metrics_b = {
        "Valid": True,
        "NetProfit": 0.42,
        "TotalAssets": 720.0,
        "Liquidations": 1.0,
        "MaxDD": -0.09,
        "WinRate": 0.61,
        "WeeklyNetProfit": 0.019,
        "Trades": 128,
    }
    dataset_metrics_c = {
        "Valid": True,
        "NetProfit": 0.3,
        "TotalAssets": 640.0,
        "Liquidations": 0.0,
        "MaxDD": -0.15,
        "WinRate": 0.52,
        "WeeklyNetProfit": 0.012,
        "Trades": 150,
    }
    dataset_metrics_d = {
        "Valid": True,
        "NetProfit": 0.18,
        "TotalAssets": 580.0,
        "Liquidations": 2.0,
        "MaxDD": -0.2,
        "WinRate": 0.48,
        "WeeklyNetProfit": 0.01,
        "Trades": 90,
    }

    results = [
        {
            "trial": 0,
            "score": 1.0,
            "params": {
                "oscLen": 20,
                "statThreshold": 38.0,
                "useChandelierExit": True,
                "chandelierLen": 15,
                "chandelierMult": 2.2,
                "useSarExit": True,
                "sarStart": 0.02,
                "sarIncrement": 0.02,
                "sarMaximum": 0.2,
            },
            "metrics": {"NetProfit": 0.25, "TotalAssets": 680.0, "Liquidations": 0.0},
            "datasets": [
                _make_dataset("BINANCE:ENAUSDT", "1m", "15m", dataset_metrics_a),
                _make_dataset("BINANCE:ENAUSDT", "3m", "1h", dataset_metrics_b),
            ],
        },
        {
            "trial": 1,
            "score": 1.2,
            "params": {
                "oscLen": 22,
                "statThreshold": 42.0,
                "useChandelierExit": False,
                "useSarExit": False,
            },
            "metrics": {"NetProfit": 0.3, "TotalAssets": 640.0, "Liquidations": 2.0},
            "datasets": [
                _make_dataset("BINANCE:ENAUSDT", "1m", "15m", dataset_metrics_c),
                _make_dataset("BINANCE:ENAUSDT", "5m", None, dataset_metrics_d),
            ],
        },
    ]

    best = {
        "params": {"oscLen": 20, "statThreshold": 38.0},
        "metrics": {"NetProfit": 0.25, "TotalAssets": 680.0, "Liquidations": 0.0},
        "score": 1.0,
    }
    wf_summary = {}

    generate_reports(
        results,
        best,
        wf_summary,
        ["NetProfit", {"name": "TotalAssets", "weight": 3.0}],
        tmp_path,
    )

    results_path = tmp_path / "results.csv"
    summary_path = tmp_path / "results_timeframe_summary.csv"
    ranking_path = tmp_path / "results_timeframe_rankings.csv"

    assert results_path.exists()
    assert summary_path.exists()
    assert ranking_path.exists()

    results_df = pd.read_csv(results_path, keep_default_na=False)
    summary_df = pd.read_csv(summary_path, keep_default_na=False)
    ranking_df = pd.read_csv(ranking_path, keep_default_na=False)

    assert results_df.columns[0] == "TotalAssets"
    assert results_df.columns[1] == "Liquidations"
    osc_idx = results_df.columns.get_loc("oscLen")
    stat_idx = results_df.columns.get_loc("statThreshold")
    assert osc_idx < stat_idx
    assert "timeframe" in summary_df.columns
    assert "htf_timeframe" not in summary_df.columns
    assert (summary_df["timeframe"] == "1m").any()
    assert "TotalAssets_mean" in ranking_df.columns
    assert "htf_timeframe" not in ranking_df.columns
    for column in [
        "useChandelierExit",
        "chandelierLen",
        "chandelierMult",
        "useSarExit",
        "sarStart",
        "sarIncrement",
        "sarMaximum",
    ]:
        assert column in results_df.columns


def test_generate_reports_writes_monte_carlo_summary(tmp_path: Path) -> None:
    trades = [
        {"profit": 50.0, "return_pct": 0.02},
        {"profit": -25.0, "return_pct": -0.01},
        {"profit": 75.0, "return_pct": 0.03},
        {"profit": -10.0, "return_pct": -0.005},
    ]

    best = {
        "params": {"oscLen": 20},
        "metrics": {"NetProfit": 0.25, "TotalAssets": 700.0, "TradesList": trades},
        "datasets": [
            {
                "name": "BINANCE:ENAUSDT_1m",
                "meta": {"timeframe": "1m"},
                "metrics": {"Valid": True, "Trades": 120, "TradesList": trades, "TotalAssets": 700.0},
            }
        ],
        "score": 1.0,
    }

    results = [
        {
            "trial": 0,
            "score": 1.0,
            "params": {"oscLen": 20},
            "metrics": {"NetProfit": 0.25, "TotalAssets": 680.0},
            "datasets": [],
        }
    ]

    generate_reports(
        results,
        best,
        {},
        ["NetProfit", {"name": "TotalAssets", "weight": 3.0}],
        tmp_path,
    )

    monte_path = tmp_path / "monte_carlo.json"
    assert monte_path.exists()

    payload = json.loads(monte_path.read_text(encoding="utf-8"))
    assert payload["iterations"] == 500
    assert payload["sample_size"] == len(trades)
    assert "net_profit" in payload
    assert "max_drawdown" in payload

    best_payload = json.loads((tmp_path / "best.json").read_text(encoding="utf-8"))
    assert "monte_carlo" in best_payload


def test_generate_reports_handles_non_numeric_objectives(tmp_path: Path) -> None:
    results = [
        {
            "trial": 0,
            "score": 1.0,
            "params": {},
            "metrics": {"NetProfit": "체크 필요", "TotalAssets": "overfactor"},
            "datasets": [],
        },
        {
            "trial": 1,
            "score": 1.2,
            "params": {},
            "metrics": {"NetProfit": 0.3, "TotalAssets": 780.0},
            "datasets": [],
        },
    ]

    best = {
        "params": {},
        "metrics": {"NetProfit": 0.3, "TotalAssets": 780.0},
        "score": 1.2,
    }

    generate_reports(
        results,
        best,
        {},
        ["NetProfit", {"name": "TotalAssets", "weight": 3.0}],
        tmp_path,
    )

    results_df = pd.read_csv(tmp_path / "results.csv")
    assert "CompositeScore" in results_df.columns
    assert "NetProfit_z" in results_df.columns
    assert (results_df["NetProfit_z"] == 0).all()
    assert "TotalAssets_z" in results_df.columns
    assert (results_df["TotalAssets_z"] == 0).all()


def test_write_trials_dataframe_removes_empty_parameter_column(tmp_path: Path) -> None:
    class DummyStudy:
        def trials_dataframe(self, attrs):
            return pd.DataFrame(
                {
                    "number": [0],
                    "value": [1.0],
                    "state": ["COMPLETE"],
                    "datetime_start": [pd.Timestamp("2024-01-01")],
                    "datetime_complete": [pd.Timestamp("2024-01-02")],
                    "params": [
                        {
                            "": 5.0,
                            "p_exit_chandelier": 18,
                            "p_exit_sar": 0.04,
                        }
                    ],
                    "user_attrs": [
                        {
                            "metrics": {"TotalAssets": 520.0, "Liquidations": 0.0, "Valid": True},
                            "score": 1.23,
                            "valid": True,
                            "trades": 140,
                            "dataset_key": {"timeframe": "1m"},
                        }
                    ],
                }
            )

    study = DummyStudy()
    write_trials_dataframe(study, tmp_path)
    csv_path = tmp_path / "trials.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert "" not in df.columns
