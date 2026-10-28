"""回測：照訊號加碼 vs 每月第一個交易日定期定額，以及模型樣本外評估。

兩種策略每月第一個交易日都存入同樣的預算，差別只在「什麼時候買」。
交易與估值都用還原權息價（等同配息再投入），可以公平比較總報酬。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .score import AGGRESSIVE, NORMAL, PAUSE


def _fee(cost: float, cfg: dict) -> float:
    return max(cfg["fee_min"], cost * cfg["fee_rate"] * cfg["fee_discount"])


def _xirr(flows: list[tuple[pd.Timestamp, float]]) -> float:
    t0 = flows[0][0]
    ts = np.array([(d - t0).days / 365.0 for d, _ in flows])
    cf = np.array([v for _, v in flows])

    def npv(r):
        return float(np.sum(cf / (1 + r) ** ts))

    lo, hi = -0.9, 1.0
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def simulate(price: pd.Series, cfg: dict, signals: pd.Series | None = None) -> dict:
    """signals=None 為定期定額；否則依訊號加碼。"""
    from .report import eval_budget
    B = eval_budget(cfg)
    max_idle = cfg["backtest"]["max_idle_months"]
    idx = price.index
    ym = idx.year * 12 + idx.month
    first = np.r_[True, ym[1:] != ym[:-1]]
    last = np.r_[ym[1:] != ym[:-1], True]

    cash = shares = spent = invested = 0.0
    spent_month = 0.0
    idle = 0
    flows, buys, equity = [], [], []

    def buy(d, p, amount):
        nonlocal cash, shares, spent, spent_month
        amount = min(amount, cash)
        n = math.floor(amount / (p * (1 + cfg["fee_rate"] * cfg["fee_discount"])))
        while n > 0 and n * p + _fee(n * p, cfg) > cash + 1e-9:
            n -= 1
        if n <= 0:
            return
        cost = n * p + _fee(n * p, cfg)
        cash -= cost
        shares += n
        spent += cost
        spent_month += cost
        buys.append((d, p, n, cost))

    for i, d in enumerate(idx):
        p = float(price.iloc[i])
        if first[i]:
            cash += B
            invested += B
            flows.append((d, -B))
            spent_month = 0.0
        if signals is None:
            if first[i]:
                buy(d, p, cash)
        else:
            s = signals.get(d, NORMAL)
            if first[i] and idle >= max_idle:
                buy(d, p, B)
            if s == AGGRESSIVE:
                buy(d, p, cash)
            elif s == NORMAL and spent_month == 0:
                buy(d, p, B / 2)
            if last[i] and s != PAUSE and spent_month < B:
                buy(d, p, B - spent_month)
        if last[i]:
            idle = 0 if spent_month > 0 else idle + 1
        equity.append(shares * p + cash)

    eq = pd.Series(equity, index=idx)
    final = float(eq.iloc[-1])
    ratio = eq / pd.Series(np.cumsum([B if f else 0 for f in first]), index=idx)
    flows.append((idx[-1], final))
    return {
        "invested": invested,
        "final_value": final,
        "total_return": final / invested - 1,
        "xirr": _xirr(flows),
        "avg_cost": spent / shares if shares else float("nan"),
        "n_buys": len(buys),
        "cash_left": cash,
        "max_drawdown": float((ratio / ratio.cummax() - 1).min()),
        "buys": buys,
        "equity": eq,
    }


def rule_signals(pos120: pd.Series) -> pd.Series:
    """簡單規則基準線：價格在 120 日區間下 1/3 積極、上 1/3 暫緩，其餘一般。"""
    return pd.Series(np.where(pos120 < 1 / 3, AGGRESSIVE, np.where(pos120 > 2 / 3, PAUSE, NORMAL)), index=pos120.index)


def oracle_signals(fwd60: pd.Series) -> pd.Series:
    """完美預知（偷看未來 60 日報酬）：只用來估計擇時效益的理論上限，不可能實際做到。"""
    return pd.Series(np.where(fwd60 > 0.08, AGGRESSIVE, np.where(fwd60 < 0, PAUSE, NORMAL)), index=fwd60.index)


def compare(adj: pd.Series, scored: pd.DataFrame, cfg: dict, Y: pd.DataFrame | None = None) -> dict:
    start = max(pd.Timestamp(cfg["backtest"]["start"]), scored.index.min())
    price = adj[adj.index >= start]
    sig = simulate(price, cfg, scored["signal"])
    dca = simulate(price, cfg, None)
    out = {
        "start": price.index[0], "end": price.index[-1],
        "signal": sig, "dca": dca,
        "cost_diff": sig["avg_cost"] / dca["avg_cost"] - 1,
        "signal_counts": scored.loc[price.index, "signal"].value_counts().to_dict(),
    }
    if "pos120" in scored:
        out["rule"] = simulate(price, cfg, rule_signals(scored["pos120"]))
    if Y is not None:
        # 最後 60 天沒有答案，完美預知在那段退回一般加碼
        out["oracle"] = simulate(price, cfg, oracle_signals(Y["fwd60"].reindex(price.index)).where(
            Y["fwd60"].reindex(price.index).notna(), NORMAL))
    return out


def evaluate(scored: pd.DataFrame, Y: pd.DataFrame) -> dict:
    """樣本外指標（PRD 第 8 節，預測期間改為存股用的 20 / 60 日）。"""
    df = scored.join(Y[["fwd20", "fwd60"]], how="inner")
    res = {}
    for h in (20, 60):
        d = df[df[f"fwd{h}"].notna()]
        up = d[f"fwd{h}"] > 0
        res[f"hit{h}"] = float(((d[f"p_up{h}"] > 0.5) == up).mean())
        res[f"base{h}"] = float(max(up.mean(), 1 - up.mean()))   # 永遠猜多數方向
        res[f"cover{h}"] = float(((d[f"fwd{h}"] >= d[f"q10_{h}"]) & (d[f"fwd{h}"] <= d[f"q90_{h}"])).mean())
        res[f"n{h}"] = int(len(d))
    d = df[df["fwd60"].notna()]
    bins = pd.cut(d["p_up60"], [0, 0.4, 0.5, 0.55, 0.65, 0.75, 1.0])
    cal = d.groupby(bins, observed=True).agg(n=("p_up60", "size"), predicted=("p_up60", "mean"),
                                             actual=("fwd60", lambda s: (s > 0).mean()))
    res["calibration"] = cal
    mid = cal[cal.index.astype(str) == "(0.55, 0.65]"]
    res["calib_err_60"] = float(abs(mid["predicted"] - mid["actual"]).iloc[0]) if len(mid) and mid["n"].iloc[0] >= 30 else float("nan")
    return res
