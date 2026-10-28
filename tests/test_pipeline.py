"""基本測試：python -m pytest -q（或 python tests/test_pipeline.py）

重點是「不偷看未來」：截斷未來資料後，過去每一天的特徵必須完全相同。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import backtest, config, demo, features, portfolio  # noqa: E402

CFG = config.load_config()


def _raw():
    if not (config.RAW / "price.csv").exists():
        demo.generate()
    return features.load_raw()


def test_no_lookahead_in_features():
    raw = _raw()
    _, X_full, _ = features.build(CFG, raw)
    cut = X_full.index[-300]
    trunc = {k: v[v.index <= cut] for k, v in raw.items()}
    _, X_cut, _ = features.build(CFG, trunc)
    common = X_cut.index
    a = X_full.loc[common]
    b = X_cut.loc[common]
    # 還原價會因之後的除息而整體縮放，比值類特徵不受影響；允許極小浮點誤差
    diff = (a - b).abs().max()
    assert (diff.fillna(0) < 1e-9).all(), f"特徵在截斷後改變（前視偏差）：\n{diff[diff > 1e-9]}"


def test_adjusted_price_removes_dividend_gap():
    s = pd.Series([100.0, 100.0, 98.0, 98.0])
    d = pd.Series([0.0, 0.0, 2.0, 0.0])
    adj = features.adjusted_close(s, d)
    assert abs(adj.iloc[2] / adj.iloc[1] - 1) < 1e-12   # 除息日還原報酬為 0
    assert adj.iloc[-1] == 98.0                          # 最新價不變


def test_targets_use_future_only():
    raw = _raw()
    base, _, Y = features.build(CFG, raw)
    a = base["adj"]
    i = 100
    assert abs(Y["fwd60"].iloc[i] - (a.iloc[i + 60] / a.iloc[i] - 1)) < 1e-12
    assert Y["fwd60"].iloc[-60:].isna().all()


def test_dca_buys_every_month():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    price = pd.Series(np.linspace(30, 36, len(idx)), index=idx)
    r = backtest.simulate(price, CFG, None)
    assert r["n_buys"] == 12
    from src.report import eval_budget
    assert r["invested"] == 12 * eval_budget(CFG)


def test_portfolio_and_dividends(tmp_path, monkeypatch=None):
    raw = _raw()
    base, _, _ = features.build(CFG, raw)
    backup = config.TRADES_CSV.with_suffix(".bak")
    had = config.TRADES_CSV.exists()
    if had:
        shutil.copy(config.TRADES_CSV, backup)
        config.TRADES_CSV.unlink()
    try:
        d0 = base.index[-400]
        portfolio.add_trade(d0.strftime("%Y-%m-%d"), 2000, 30.0, fee=20)
        portfolio.add_trade(base.index[-100].strftime("%Y-%m-%d"), 1000, 33.0, fee=20)
        s = portfolio.summary(base, CFG)
        assert s["shares"] == 3000
        assert abs(s["avg_cost"] - (2000 * 30 + 20 + 1000 * 33 + 20) / 3000) < 1e-9
        assert s["cum_dividends"] > 0          # 期間有除息，應自動算出配息
    finally:
        config.TRADES_CSV.unlink(missing_ok=True)
        if had:
            shutil.move(backup, config.TRADES_CSV)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(None) if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount] else fn()
            print("通過", name)
