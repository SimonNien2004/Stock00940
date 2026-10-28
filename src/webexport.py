"""輸出網頁版 APP 用的 reports/web.json。

每日同步時，Claude 會把這個檔案的四個區塊寫進網頁 APP 的資料庫：
  report     → app/report
  series     → app/series
  backtest   → app/backtest
  dividends  → app/dividends
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import REPORTS
from .score import AGGRESSIVE, NORMAL, PAUSE

SIG_CODE = {AGGRESSIVE: "A", NORMAL: "N", PAUSE: "P"}


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), nd)


def export(*, base, scored, today, js_report, bt, ev, live, cfg, meta, days: int = 500) -> dict:
    tail = base.iloc[-days:]
    sc = scored.reindex(tail.index)
    # 今日不在樣本外序列中：以今日評分補上
    sc.loc[tail.index[-1], ["score", "signal"]] = [today["score"], today["signal"]]
    series = {
        "dates": [d.strftime("%Y-%m-%d") for d in tail.index],
        "close": [_f(x, 2) for x in tail["close"]],
        "score": [_f(x, 1) for x in sc["score"]],
        "signal": [SIG_CODE.get(s, "") if isinstance(s, str) else "" for s in sc["signal"]],
    }

    def month_end(eq: pd.Series, invested: pd.Series) -> list:
        r = (eq / invested).resample("ME").last()
        return [_f(x, 4) for x in r]

    s_eq, d_eq = bt["signal"]["equity"], bt["dca"]["equity"]
    idx = s_eq.index
    first = np.r_[True, (idx.year * 12 + idx.month)[1:] != (idx.year * 12 + idx.month)[:-1]]
    invested = pd.Series(np.cumsum(np.where(first, cfg["monthly_budget"], 0)), index=idx)
    months = [d.strftime("%Y-%m") for d in s_eq.resample("ME").last().index]

    rows = []
    c = bt["dca"]
    for key, name in (("dca", "定期定額（基準）"), ("rule", "簡單規則（120 日位置）"),
                      ("signal", "本系統訊號"), ("oracle", "完美預知（理論上限）")):
        r = bt.get(key)
        if r is None:
            continue
        rows.append({"key": key, "name": name, "xirr": _f(r["xirr"]), "total_return": _f(r["total_return"]),
                     "vs_dca": _f(r["xirr"] - c["xirr"]), "cost_vs_dca": _f(r["avg_cost"] / c["avg_cost"] - 1),
                     "max_drawdown": _f(r["max_drawdown"]), "n_buys": int(r["n_buys"])})

    backtest = {
        "start": bt["start"].strftime("%Y-%m-%d"), "end": bt["end"].strftime("%Y-%m-%d"),
        "monthly_budget": __import__("src.report", fromlist=["eval_budget"]).eval_budget(cfg), "rows": rows,
        "signal_counts": {k: int(v) for k, v in bt["signal_counts"].items()},
        "equity": {"months": months, "signal": month_end(s_eq, invested), "dca": month_end(d_eq, invested)},
        "evaluation": {k: _f(v) if isinstance(v, (int, float, np.floating)) else v
                       for k, v in ev.items() if k != "calibration"},
        "live": {k: _f(v) for k, v in (live or {}).items()},
    }

    divs = base[base["div"] > 0]
    dividends = {"items": [{"date": d.strftime("%Y-%m-%d"), "cash": _f(v, 4)} for d, v in divs["div"].items()]}

    report = {
        "date": js_report["date"], "close": js_report["close"], "score": js_report["score"],
        "signal": js_report["signal"], "confidence": js_report["confidence"],
        "advice": js_report["advice"], "reasons": js_report["reasons"],
        "prediction": {k: _f(v) for k, v in js_report["prediction"].items()},
        "pos120": _f(today["pos120"]),
        "next_ex_dividend_est": js_report["next_ex_dividend_est"],
        "next_rebalance_est": js_report["next_rebalance_est"],
        "ttm_dividend": _f(base.loc[base.index > base.index[-1] - pd.Timedelta(days=365), "div"].sum()),
        "config": {k: cfg.get(k) for k in ("monthly_budget", "reserve", "fee_rate", "fee_discount", "fee_min", "dca", "extra_pool")}
                  | {"thresholds": cfg["thresholds"]},
        "engine": js_report["engine"], "data_source": meta.get("sources", {}).get("price"),
        "demo": "DEMO" in meta.get("sources", {}).values(),
        "warnings": meta.get("warnings", []), "generated_at": js_report["generated_at"],
    }
    out = {"report": report, "series": series, "backtest": backtest, "dividends": dividends}
    (REPORTS / "web.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out
