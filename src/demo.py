"""產生模擬資料，用於沒有網路時測試整套流程（python run_daily.py --demo）。

模擬資料不是真實行情，只用來驗證程式邏輯。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import RAW, ensure_dirs, load_config


def generate(seed: int = 7, end: str | None = None) -> dict:
    ensure_dirs()
    cfg = load_config()
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(cfg["history_start"], end)
    n = len(days)

    # 市場因子：帶波動叢聚的報酬
    vol = np.empty(n)
    vol[0] = 0.010
    shocks = rng.standard_normal(n)
    for i in range(1, n):
        vol[i] = np.sqrt(0.000002 + 0.08 * (vol[i - 1] * shocks[i - 1]) ** 2 + 0.90 * vol[i - 1] ** 2)
    mkt = 0.0003 + vol * shocks
    twii = 12000 * np.exp(np.cumsum(mkt))

    # 含息總報酬（對數）= 長期趨勢 + 跟隨大盤的隨機漫步 + 均值回歸偏離（已知可預測成分，用來驗證模型會學）
    trend = np.log(15.0) + 0.00036 * np.arange(n)            # 約 9%/年 含息報酬
    walk = 0.6 * np.cumsum(mkt - mkt.mean())
    dev = np.zeros(n)
    for i in range(1, n):
        dev[i] = 0.985 * dev[i - 1] + 0.006 * rng.standard_normal()
    total_idx = np.exp(trend + walk + dev)

    # 配息：config 的 ex_months 每月 18 日附近除息，約 1.6%/季
    df = pd.DataFrame(index=days)
    ex_dates = []
    for y in range(days[0].year, days[-1].year + 1):
        for m in cfg.get("ex_months", (1, 4, 7, 10)):
            d = pd.Timestamp(year=y, month=m, day=18)
            pos = days.searchsorted(d)
            if pos < n:
                ex_dates.append(days[pos])

    close = np.empty(n)
    factor = 1.0
    divs = []
    ex_set = set(ex_dates)
    for i, d in enumerate(days):
        if d in ex_set and i > 0:
            dv = round(close[i - 1] * rng.uniform(0.014, 0.022) * 4 / len(cfg.get("ex_months", (1, 4, 7, 10))), 3)
            factor *= (close[i - 1] - dv) / close[i - 1]
            divs.append((d.strftime("%Y-%m-%d"), dv))
        close[i] = round(total_idx[i] * factor, 2)

    noise = np.abs(rng.standard_normal(n)) * 0.004
    price = pd.DataFrame({
        "date": days.strftime("%Y-%m-%d"),
        "open": np.round(close * (1 + rng.uniform(-0.003, 0.003, n)), 2),
        "high": np.round(close * (1 + noise), 2),
        "low": np.round(close * (1 - noise), 2),
        "close": close,
        "volume": (rng.lognormal(16, 0.5, n)).astype(int),
    })
    inst = pd.DataFrame({
        "date": days.strftime("%Y-%m-%d"),
        "foreign_net": (rng.standard_normal(n) * 2e6).astype(int),
        "trust_net": (rng.standard_normal(n) * 5e5).astype(int),
    })
    price.to_csv(RAW / "price.csv", index=False)
    pd.DataFrame(divs, columns=["date", "cash_dividend"]).to_csv(RAW / "dividends.csv", index=False)
    pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "close": np.round(twii, 2)}).to_csv(RAW / "taiex.csv", index=False)
    inst.to_csv(RAW / "inst.csv", index=False)
    meta = {"fetched_at": end, "sources": {k: "DEMO" for k in ("price", "dividends", "taiex", "inst")},
            "warnings": ["這是模擬資料，不是真實行情"], "last_price_date": price["date"].iloc[-1]}
    with open(RAW / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta
