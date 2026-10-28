"""每日報告（reports/latest.md、latest.json）、評估報告與預測紀錄對帳。"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from .config import PREDICTIONS_CSV, REPORTS
from .features import FEATURE_TEXT, next_rebalance
from .portfolio import what_if
from .score import AGGRESSIVE, NORMAL, PAUSE

TPE = timezone(timedelta(hours=8))
PRED_COLS = ["date", "close", "score", "signal", "confidence", "pos120", "p_up20", "p_up60", "p_dip",
             "q10_20", "q50_20", "q90_20", "q10_60", "q50_60", "q90_60", "engine", "act20", "act60"]


# ---------------------------------------------------------------- 預測紀錄與對帳
def log_prediction(today: pd.Series, date: pd.Timestamp, close: float, engine: str) -> None:
    row = {c: today.get(c) for c in PRED_COLS if c in today.index}
    row.update({"date": date.strftime("%Y-%m-%d"), "close": close, "engine": engine,
                "act20": np.nan, "act60": np.nan})
    if PREDICTIONS_CSV.exists():
        log = pd.read_csv(PREDICTIONS_CSV)
        log = log[log["date"] != row["date"]]          # 同一天重跑就覆蓋
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])
    log[PRED_COLS].sort_values("date").to_csv(PREDICTIONS_CSV, index=False, float_format="%.6g")


def reconcile(adj: pd.Series) -> pd.DataFrame:
    """把到期的預測填上實際結果。"""
    if not PREDICTIONS_CSV.exists():
        return pd.DataFrame(columns=PRED_COLS)
    log = pd.read_csv(PREDICTIONS_CSV)
    pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(adj.index)}
    for h in (20, 60):
        col = f"act{h}"
        for k, d in log["date"].items():
            i = pos.get(d)
            if i is not None and i + h < len(adj):
                log.loc[k, col] = adj.iloc[i + h] / adj.iloc[i] - 1
    log.to_csv(PREDICTIONS_CSV, index=False, float_format="%.6g")
    return log


def live_accuracy(log: pd.DataFrame) -> dict:
    out = {}
    for h in (20, 60):
        d = log[log[f"act{h}"].notna()]
        if len(d):
            out[f"n{h}"] = len(d)
            out[f"hit{h}"] = float(((d[f"p_up{h}"] > 0.5) == (d[f"act{h}"] > 0)).mean())
            out[f"cover{h}"] = float(((d[f"act{h}"] >= d[f"q10_{h}"]) & (d[f"act{h}"] <= d[f"q90_{h}"])).mean())
    return out


# ---------------------------------------------------------------- 文字
def next_dca(d: pd.Timestamp, cfg: dict) -> pd.Timestamp | None:
    """下一次定期定額扣款日（日曆日，遇休市券商會順延）。"""
    days = sorted((cfg.get("dca") or {}).get("days") or [])
    if not days:
        return None
    for k in range(3):
        m = (d + pd.offsets.MonthBegin(k)) if k else d.replace(day=1)
        last = (m + pd.offsets.MonthEnd(0)).day
        for day in days:
            t = m.replace(day=min(day, last))
            if t > d:
                return t
    return None


def next_ex_estimate(base: pd.DataFrame, d: pd.Timestamp, cfg: dict) -> pd.Timestamp | None:
    """下次除息日（估）：優先用 config 的 ex_months / ex_day；沒有設定時用過去除息間隔的中位數。"""
    months, day = cfg.get("ex_months"), int(cfg.get("ex_day", 18))
    if months:
        for y in (d.year, d.year + 1):
            for m in sorted(months):
                last = (pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)).day
                t = pd.Timestamp(year=y, month=m, day=min(day, last))
                if t > d:
                    return t
    ex_dates = base.index[base["div"] > 0]
    if not len(ex_dates):
        return None
    gaps = np.diff(ex_dates.values).astype("timedelta64[D]").astype(int)[-6:]
    step = pd.Timedelta(days=int(np.median(gaps)) if len(gaps) else 91)
    nxt = ex_dates[-1] + step
    while nxt <= d:
        nxt += step
    return nxt


def eval_budget(cfg: dict) -> float:
    """回測用的每月額外預算：未設定時，以每月定期定額總額作為假設金額。"""
    if cfg.get("monthly_budget"):
        return float(cfg["monthly_budget"])
    dca = cfg.get("dca") or {}
    return float(dca.get("amount", 0) * len(dca.get("days", []))) or 10000.0


def advice(sig: str, cfg: dict, close: float, pos: dict) -> list[str]:
    B, R = cfg.get("monthly_budget", 0), cfg.get("reserve", 0)
    dca = cfg.get("dca") or {}
    lines, amt = [], 0
    nd = next_dca(pd.Timestamp(pos.get("date", pd.Timestamp.today())).normalize(), cfg)
    if dca.get("amount") and nd is not None:
        days = "、".join(str(x) for x in dca["days"])
        lines.append(f"定期定額照常：每月 {days} 日各 {dca['amount']:,.0f} 元，下次約 {nd:%m/%d} 扣款（遇休市順延）。")
    aim = cfg["thresholds"]["aggressive"]
    pool = pos.get("pool")
    if pool:
        shared = f"，與 {'、'.join(pool['shared_with'])} 共用" if pool.get("shared_with") else ""
        left = f"資金池剩 {pool['remaining']:,.0f} / {pool['total']:,.0f} 元{shared}"
        if pool["remaining"] <= 0:
            lines.append(f"額外加碼：資金池已用完（{left}）。")
        elif sig == AGGRESSIVE:
            amt = pool["cap_aggressive"]
            if amt >= 100:
                lines.append(f"額外加碼：本月最多可動用 {amt // 100 * 100:,.0f} 元（月初剩餘資金的 {pool['aggressive_pct']:.0%}"
                             f"，已扣本月已用 {pool['used_month']:,.0f} 元；{left}）。")
            else:
                lines.append(f"額外加碼：本月額度已用完，下個月再動用（{left}）。")
        elif sig == NORMAL:
            amt = pool["cap_normal"]
            if amt >= 100:
                lines.append(f"額外加碼：本月最多可動用 {amt // 100 * 100:,.0f} 元（月初剩餘資金的 {pool['normal_pct']:.0%}；{left}）；"
                             f"想投入更多，等評分 ≥ {aim} 再用。")
            else:
                lines.append(f"額外加碼：一般加碼的本月額度已用完；評分 ≥ {aim} 時可再動用到 {pool['aggressive_pct']:.0%}（{left}）。")
        else:
            lines.append(f"額外加碼：先保留資金池，定期定額照常即可（{left}）。")
    elif sig == AGGRESSIVE:
        if B:
            amt = B
            lines.append(f"額外加碼：可投入本月全部額外預算 {B:,.0f} 元" + (f"；預備金也可視情況動用（上限 {R:,.0f} 元）。" if R else "。"))
        else:
            lines.append("額外加碼：適合動用定期定額以外的資金（額外預算尚未設定，告訴 Claude 金額後會換算股數）。")
    elif sig == NORMAL:
        if B:
            amt = B / 2
            lines.append(f"額外加碼：本月額外預算先投入一半（約 {B / 2:,.0f} 元），其餘等評分 ≥ {aim} 再用。")
        else:
            lines.append(f"額外加碼：可小額加碼；想一次投入較多，等評分 ≥ {aim} 再用。")
    else:
        lines.append("額外加碼：先保留，定期定額照常即可。")
    amt = int(amt // 100 * 100)   # 取整到百元
    w_amt = amt or dca.get("amount", 0)
    if w_amt:
        w = what_if(pos, w_amt, close, cfg)
        if w["shares"]:
            what = "下次定期定額" if not amt else "額外加碼"
            line = f"以今日收盤 {close:.2f} 估算，{what} {w_amt:,.0f} 元約可買 {w['shares']:,} 股"
            if pos.get("has_position"):
                line += f"，你的均價會從 {w['old_avg']:.2f} 變為 {w['new_avg']:.2f}"
            lines.append(line + "。")
    return lines


def reasons(contrib: pd.Series, xrow: pd.Series, k: int = 3) -> list[str]:
    out = []
    for f in contrib.abs().sort_values(ascending=False).index[:k]:
        v = xrow[f]
        if pd.isna(v):
            continue
        tmpl, _ = FEATURE_TEXT[f]
        tag = "有利加碼" if contrib[f] > 0 else "不利加碼"
        out.append(f"{tmpl.format(v)}（{tag}）")
    return out


def _pct(x):
    return "—" if x is None or pd.isna(x) else f"{x:.1%}"


# ---------------------------------------------------------------- 主報告
def write_daily(*, base, X, today, contrib, pos, bt, ev, live, cfg, engine, meta) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    d = base.index[-1]
    close = float(base["close"].iloc[-1])
    sig, sc, conf = today["signal"], float(today["score"]), today["confidence"]
    next_ex = next_ex_estimate(base, d, cfg)
    next_rb = next_rebalance(d, cfg)
    now = datetime.now(TPE).strftime("%Y-%m-%d %H:%M")

    L = [f"# {cfg['symbol']} {cfg.get('name', '')} 每日加碼報告｜{d:%Y-%m-%d}（收盤 {close:.2f}）", ""]
    if "DEMO" in meta.get("sources", {}).values():
        L += ["> ⚠️ 本報告使用模擬資料，僅供測試程式，不代表真實行情。", ""]
    stale = int(meta.get("stale_days", 0) or 0)
    if stale:
        L += [f"> ⚠️ 資料只到 {d:%Y-%m-%d}，比最近交易日少 {stale} 天（資料源尚未更新）；以下評分用的是舊價格。", ""]
    L += [f"資料截止：**{d:%Y-%m-%d}**" + ("" if stale else "（已是最新交易日）"), ""]
    L += ["## 結論", "",
          f"**加碼評分：{sc:.0f} / 100 → {sig}**（信心 {conf}）", ""]
    L += [f"- {x}" for x in advice(sig, cfg, close, pos)]
    L += ["", "## 預測（含息報酬）", "",
          "| 期間 | 上漲機率 | 預期區間（10%–90%） | 中位數 | 換算價格區間 |",
          "| --- | --- | --- | --- | --- |"]
    for h in (20, 60):
        q10, q50, q90 = today[f"q10_{h}"], today[f"q50_{h}"], today[f"q90_{h}"]
        L.append(f"| {h} 個交易日 | {today[f'p_up{h}']:.0%} | {q10:+.1%} ～ {q90:+.1%} | {q50:+.1%} | "
                 f"{close * (1 + q10):.2f} ～ {close * (1 + q90):.2f} |")
    L += ["", f"60 日內再跌 3% 以上的機率：{today['p_dip']:.0%}。價格區間為含息換算，期間若除息，市價會再扣除配息金額。", ""]

    L += ["## 主要依據", ""]
    L += [f"{i}. {r}" for i, r in enumerate(reasons(contrib, X.iloc[-1]), 1)]
    L += ["", "依對 60 日上漲機率的影響大小排序；「有利」表示目前數值讓上漲機率高於一般情況。", ""]

    L += ["## 你的部位", ""]
    if pos.get("has_position"):
        L += ["| 項目 | 數值 |", "| --- | --- |",
              f"| 持股 | {pos['shares']:,.0f} 股（{pos['lots']:.3f} 張） |",
              f"| 平均成本（含手續費） | {pos['avg_cost']:.2f} |",
              f"| 市值 | {pos['market_value']:,.0f} 元 |",
              f"| 未實現損益 | {pos['unrealized']:+,.0f} 元（{pos['unrealized_pct']:+.1%}） |",
              f"| 累計領息（扣補充保費） | {pos['cum_dividends']:,.0f} 元 |",
              f"| 含息總損益 | {pos['total_pl']:+,.0f} 元 |",
              f"| 成本殖利率（近 12 個月） | {pos['cost_yield']:.2%} |"]
        if pos.get("pool"):
            p = pos["pool"]
            detail = (f"（本檔 {p['used_self']:,.0f}、{'、'.join(p['shared_with'])} {p['used_others']:,.0f}）"
                      if p.get("shared_with") else "")
            L += [f"| 額外資金池 | 已用 {p['used']:,.0f}／{p['total']:,.0f} 元{detail}，剩 {p['remaining']:,.0f} 元 |"]
        L += [""]
    else:
        L += ["尚未記錄持股。在 Claude 對話中說「我有 20 張，均價 33.5」或回報交易即可建立。", ""]

    L += ["## 近期事件", ""]
    if next_ex is not None:
        line = f"- 下次除息（估）：{next_ex:%Y-%m-%d} 前後，距今 {(next_ex - d).days} 天；上次每單位配息 {pos['last_div']:.3f} 元"
        if pos.get("has_position"):
            line += f"，以目前持股估約 {pos['next_div_est']:,.0f} 元"
            if pos.get("next_div_nhi"):
                line += f"（超過 2 萬元，將扣二代健保補充保費約 {pos['next_div_nhi']:,.0f} 元）"
        L.append(line + "。")
    L.append(f"- 下次成分股調整（估）：{next_rb:%Y-%m-%d} 前後，距今 {(next_rb - d).days} 天。")
    L += [f"- 近 12 個月殖利率（以今日收盤計）：{pos['market_yield']:.2%}。", ""]

    s, c = bt["signal"], bt["dca"]
    L += ["## 系統表現", "",
          f"回測 {bt['start']:%Y-%m-%d} ～ {bt['end']:%Y-%m-%d}，每月 {eval_budget(cfg):,.0f} 元{'' if cfg.get('monthly_budget') else ('（以定期定額月總額作為回測假設）' if (cfg.get('dca') or {}).get('amount') else '（未設定預算，以 10,000 元作為回測假設）')}，僅用樣本外預測：", "",
          "| 策略 | 年化報酬（XIRR） | 總報酬 | 平均買入價（還原價） | 最大回撤 |", "| --- | --- | --- | --- | --- |",
          f"| 照訊號加碼 | {_pct(s['xirr'])} | {_pct(s['total_return'])} | {s['avg_cost']:.2f} | {_pct(s['max_drawdown'])} |",
          f"| 定期定額 | {_pct(c['xirr'])} | {_pct(c['total_return'])} | {c['avg_cost']:.2f} | {_pct(c['max_drawdown'])} |",
          "", f"照訊號加碼的平均買入價比定期定額{'低' if bt['cost_diff'] < 0 else '高'} {abs(bt['cost_diff']):.2%}。"
          f"樣本外 60 日方向命中率 {ev['hit60']:.1%}（基準 {ev['base60']:.1%}），區間覆蓋率 {ev['cover60']:.1%}。", ""]
    if live:
        L += [f"實盤對帳：已到期 60 日預測 {live.get('n60', 0)} 筆，命中率 {_pct(live.get('hit60'))}；"
              f"20 日預測 {live.get('n20', 0)} 筆，命中率 {_pct(live.get('hit20'))}。", ""]
    else:
        L += ["實盤對帳：尚無到期的預測（20 日後開始有結果）。", ""]

    L += ["---", f"資料日期 {d:%Y-%m-%d}｜資料來源 {meta.get('sources', {}).get('price', '?')}｜模型 {engine}｜產生時間 {now}（台北）", ""]
    for w in meta.get("warnings", []):
        L.append(f"- 注意：{w}")
    L += ["", "本報告僅供個人研究參考，不構成投資建議；過去績效不代表未來表現。"]
    (REPORTS / "latest.md").write_text("\n".join(L), encoding="utf-8")

    js = {
        "date": d.strftime("%Y-%m-%d"), "close": close, "score": sc, "signal": sig, "confidence": conf,
        "advice": advice(sig, cfg, close, pos),
        "prediction": {k: float(today[k]) for k in today.index if k.startswith(("p_", "q"))},
        "reasons": reasons(contrib, X.iloc[-1]),
        "position": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in pos.items() if k not in ("dividends", "last_trade", "date")},
        "next_ex_dividend_est": next_ex.strftime("%Y-%m-%d") if next_ex is not None else None,
        "next_rebalance_est": next_rb.strftime("%Y-%m-%d"),
        "backtest": {"signal_xirr": s["xirr"], "dca_xirr": c["xirr"], "cost_diff": bt["cost_diff"]},
        "evaluation": {k: v for k, v in ev.items() if k != "calibration"},
        "live": live, "engine": engine, "data_source": meta.get("sources"), "generated_at": now,
        "data_stale_days": int(meta.get("stale_days", 0) or 0), "data_warnings": meta.get("warnings", []),
    }
    (REPORTS / "latest.json").write_text(json.dumps(js, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return js


def write_evaluation(bt: dict, ev: dict, cfg: dict, engine: str) -> None:
    s, c = bt["signal"], bt["dca"]
    L = ["# 模型與策略評估（樣本外）", "",
         f"walk-forward 擴張視窗、每 {cfg['model']['retrain_every']} 個交易日重訓、訓練與測試間隔 60 日。模型：{engine}。", "",
         "## PRD 第 8 節驗收對照", "",
         "| 指標 | 結果 | 門檻 | 通過 |", "| --- | --- | --- | --- |"]

    def ok(b):
        return "是" if b else "否"

    L.append(f"| 60 日方向命中率 | {ev['hit60']:.1%}（{ev['n60']} 筆） | ≥ 55% 且高於基準 {ev['base60']:.1%} | "
             f"{ok(ev['hit60'] >= 0.55 and ev['hit60'] > ev['base60'])} |")
    L.append(f"| 20 日方向命中率（參考） | {ev['hit20']:.1%} | — | — |")
    ce = ev["calib_err_60"]
    L.append(f"| 機率校準（預測 55–65% 區） | 誤差 {_pct(ce)} | ≤ 5 個百分點 | {ok(not pd.isna(ce) and ce <= 0.05)} |")
    L.append(f"| 60 日區間覆蓋率 | {ev['cover60']:.1%} | 75%–85% | {ok(0.75 <= ev['cover60'] <= 0.85)} |")
    L.append(f"| 策略年化報酬 | {_pct(s['xirr'])} vs 定期定額 {_pct(c['xirr'])} | 高於定期定額 | {ok(s['xirr'] > c['xirr'])} |")
    L.append(f"| 最大回撤 | {_pct(s['max_drawdown'])} vs {_pct(c['max_drawdown'])} | 低於定期定額 | "
             f"{ok(s['max_drawdown'] >= c['max_drawdown'])} |")
    L += ["", "注意：60 日預測的樣本彼此重疊，實際獨立樣本約為筆數 / 60，命中率的不確定性比筆數看起來大。", "",
          "## 擇時效益對照（同樣每月預算）", "",
          "| 策略 | 年化報酬（XIRR） | 相對定期定額 | 平均買入價相對定期定額 |", "| --- | --- | --- | --- |"]
    rows = [("定期定額（基準）", c), ("簡單規則（120 日位置）", bt.get("rule")),
            ("本系統訊號", s), ("完美預知（理論上限）", bt.get("oracle"))]
    for name, r in rows:
        if r is None:
            continue
        L.append(f"| {name} | {_pct(r['xirr'])} | {r['xirr'] - c['xirr']:+.2%} | {r['avg_cost'] / c['avg_cost'] - 1:+.2%} |")
    L += ["", "「完美預知」偷看了未來報酬，實際不可能做到；它代表在這套每月預算規則下，擇時最多能多賺多少。"
          "本系統若贏不了「簡單規則」，依 PRD 的 Go / No-Go，應改用規則訊號。", "",
          "## 機率校準表（60 日）", "", "| 預測機率區間 | 筆數 | 平均預測 | 實際上漲比例 |", "| --- | --- | --- | --- |"]
    for k, r in ev["calibration"].iterrows():
        L.append(f"| {k} | {int(r['n'])} | {r['predicted']:.1%} | {r['actual']:.1%} |")
    L += ["", "## 回測明細", "",
          f"期間 {bt['start']:%Y-%m-%d} ～ {bt['end']:%Y-%m-%d}；訊號分布：" +
          "、".join(f"{k} {v} 天" for k, v in bt["signal_counts"].items()), "",
          "| 項目 | 照訊號加碼 | 定期定額 |", "| --- | --- | --- |",
          f"| 總投入 | {s['invested']:,.0f} | {c['invested']:,.0f} |",
          f"| 期末價值 | {s['final_value']:,.0f} | {c['final_value']:,.0f} |",
          f"| 買進次數 | {s['n_buys']} | {c['n_buys']} |",
          f"| 期末未投入現金 | {s['cash_left']:,.0f} | {c['cash_left']:,.0f} |",
          f"| 平均買入價（還原價） | {s['avg_cost']:.3f} | {c['avg_cost']:.3f} |", ""]
    (REPORTS / "evaluation.md").write_text("\n".join(L), encoding="utf-8")
