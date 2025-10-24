import sys
import subprocess

import pandas as pd

from tools.compare_pine_csv import compare_metric_frames, compare_trade_reasons


def test_compare_metric_frames_matches():
    pine = pd.DataFrame(
        [{"TotalAssets": 500.0, "NetProfit": 0.05, "Trades": 10, "Wins": 6, "Losses": 4, "Liquidations": 0}]
    )
    python = pd.DataFrame(
        [{"TotalAssets": 500.0, "NetProfitAbs": 0.05, "Trades": 10, "Wins": 6, "Losses": 4, "Liquidations": 0}]
    )
    diffs = list(compare_metric_frames(pine, python))
    assert all(diff.matches for diff in diffs)


def test_compare_metric_frames_detects_difference():
    pine = pd.DataFrame([{"TotalAssets": 500.0}])
    python = pd.DataFrame([{"TotalAssets": 505.0}])
    diffs = list(compare_metric_frames(pine, python, tolerance=0.1))
    mismatch = next(diff for diff in diffs if diff.name == "TotalAssets")
    assert not mismatch.matches
    assert mismatch.delta == 5.0


def test_compare_trade_reasons_counts():
    pine = pd.DataFrame({"Reason": ["Stop Long", "Stop Long", "Guard Exit"]})
    python = pd.DataFrame({"Reason": ["Stop Long", "Guard Exit", "Guard Exit"]})
    table = compare_trade_reasons(pine, python)
    assert table is not None
    assert table.loc["Stop Long", "delta"] == -1
    assert table.loc["Guard Exit", "delta"] == 1


def test_compare_pine_cli_invocation(tmp_path):
    pine_metrics = pd.DataFrame(
        [{"TotalAssets": 500.0, "NetProfit": 0.05, "Trades": 10, "Wins": 6, "Losses": 4, "Liquidations": 0}]
    )
    python_metrics = pd.DataFrame(
        [{"TotalAssets": 500.0, "NetProfitAbs": 0.05, "Trades": 10, "Wins": 6, "Losses": 4, "Liquidations": 0}]
    )
    pine_trades = pd.DataFrame({"Reason": ["Stop Long", "Guard Exit"]})
    python_trades = pd.DataFrame({"Reason": ["Stop Long", "Guard Exit"]})

    pine_metrics_path = tmp_path / "pine_metrics.csv"
    python_metrics_path = tmp_path / "python_metrics.csv"
    pine_trades_path = tmp_path / "pine_trades.csv"
    python_trades_path = tmp_path / "python_trades.csv"

    pine_metrics.to_csv(pine_metrics_path, index=False)
    python_metrics.to_csv(python_metrics_path, index=False)
    pine_trades.to_csv(pine_trades_path, index=False)
    python_trades.to_csv(python_trades_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.compare_pine_csv",
            "--pine-metrics",
            str(pine_metrics_path),
            "--python-metrics",
            str(python_metrics_path),
            "--pine-trades",
            str(pine_trades_path),
            "--python-trades",
            str(python_trades_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Metrics match" in result.stdout
    assert "Trade exit reasons match" in result.stdout
