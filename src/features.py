"""還原權息價、特徵與預測目標。

原則：第 t 天的特徵只能用第 t 天收盤（含）以前的資料，避免前視偏差。
目標（fwd20、fwd60、dip）則使用 t 之後的資料，只在訓練時使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW

FEATURES = [
    "pos120", "pos240", "bias20", "bias60", "bias240", "ret20", "ret60", "vol20",
    "rsi14", "boll", "vol_ratio", "yield_ttm", "days_since_ex", "days_to_ex",
    "days_to_rebal", "twii_bias60", "twii_ret20", "inst_net20",
]

FEATURE_TEXT = {
    "pos120": ("收盤價位於近 120 日區間的 {:.0%} 位置", "pct"),
    "pos240": ("收盤價位於近 240 日區間的 {:.0%} 位置", "pct"),
    "bias20": ("與 20 日均線乖離 {:+.1%}", "pct"),
    "bias60": ("與 60 日均線乖離 {:+.1%}", "pct"),
    "bias240": ("與 240 日均線乖離 {:+.1%}", "pct"),
    "ret20": ("近 20 日還原報酬 {:+.1%}", "pct"),
    "ret60": ("近 60 日還原報酬 {:+.1%}", "pct"),
    "vol20": ("20 日年化波動 {:.1%}", "pct"),
    "rsi14": ("RSI(14) = {:.0f}", "num"),
    "boll": ("布林通道位置 {:+.2f}（-1 下軌、+1 上軌）", "num"),
    "vol_ratio": ("5 日均量為 60 日均量的 {:.2f} 倍", "num"),
    "yield_ttm": ("近 12 個月殖利率 {:.2%}", "pct"),
    "days_since_ex": ("距上次除息 {:.0f} 天", "num"),
    "days_to_ex": ("距下次除息（估）{:.0f} 天", "num"),
    "days_to_rebal": ("距下次成分股調整（估）{:.0f} 天", "num"),
    "twii_bias60": ("加權指數與 60 日均線乖離 {:+.1%}", "pct"),
    "twii_ret20": ("加權指數近 20 日 {:+.1%}", "pct"),
    "inst_net20": ("近 20 日外資＋投信淨買超占成交量 {:+.1%}", "pct"),
}


def load_raw() -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("price", "dividends", "taiex", "inst"):
        df = pd.read_csv(RAW / f"{name}.csv")
        df["date"] = pd.to_datetime(df["date"])
        out[name] = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    return out


def adjusted_close(close: pd.Series, div: pd.Series) -> pd.Series:
    """向後還原：除息日之前的價格乘上 (前收 - 股利) / 前收。"""
    prev = close.shift(1)
    f = ((prev - div) / prev).where(div > 0, 1.0).fillna(1.0)
    factor = f[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return close * factor


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _range_pos(s: pd.Series, w: int, minp: int | None = None) -> pd.Series:
    lo, hi = s.rolling(w, min_periods=minp).min(), s.rolling(w, min_periods=minp).max()
    return ((s - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5)


def _next_rebalance(d: pd.Timestamp, months=(5, 11), day: int = 31) -> pd.Timestamp:
    """下次成分股調整日（近似）：config 的 rebalance_months / rebalance_day。"""
    for y in (d.year, d.year + 1):
        for m in sorted(months):
            last = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
            t = pd.Timestamp(year=y, month=m, day=min(day, last.day))
            if t >= d:
                return t
    return d


def next_rebalance(d: pd.Timestamp, cfg: dict) -> pd.Timestamp:
    return _next_rebalance(d, cfg.get("rebalance_months", (5, 11)), cfg.get("rebalance_day", 31))


def build(cfg: dict, raw: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """回傳 (base, X, Y)。base 含 close/adj/div；X 為特徵；Y 為目標。"""
    raw = raw if raw is not None else load_raw()
    px = raw["price"]
    base = pd.DataFrame(index=px.index)
    base["close"] = px["close"].astype(float)
    base["volume"] = px["volume"].astype(float)

    # 除息日對齊到交易日（若剛好不是交易日，移到下一個交易日）
    base["div"] = 0.0
    for d, row in raw["dividends"].iterrows():
        pos = base.index.searchsorted(d)
        if pos < len(base) and pos > 0:
            base.iloc[pos, base.columns.get_loc("div")] += float(row["cash_dividend"])
    base["adj"] = adjusted_close(base["close"], base["div"])

    a = base["adj"]
    # 暖機期：240 日指標至少要有幾天資料才開始算（歷史很短的 ETF 可調小，預設 240）
    warm = int(cfg.get("warmup_days", 240))
    X = pd.DataFrame(index=base.index)
    X["pos120"] = _range_pos(a, 120)
    X["pos240"] = _range_pos(a, 240, min(warm, 240))
    for w in (20, 60):
        X[f"bias{w}"] = a / a.rolling(w).mean() - 1
    X["bias240"] = a / a.rolling(240, min_periods=min(warm, 240)).mean() - 1
    X["ret20"] = a.pct_change(20)
    X["ret60"] = a.pct_change(60)
    X["vol20"] = a.pct_change().rolling(20).std() * np.sqrt(252)
    X["rsi14"] = _rsi(a)
    m, s = a.rolling(20).mean(), a.rolling(20).std()
    X["boll"] = (a - m) / (2 * s)
    X["vol_ratio"] = base["volume"].rolling(5).mean() / base["volume"].rolling(60).mean()
    X["yield_ttm"] = base["div"].rolling("365D").sum() / base["close"]

    # 事件曆：除息與成分股調整
    ex_dates = base.index[base["div"] > 0]
    since, to_ex, to_rb = [], [], []
    for d in base.index:
        past = ex_dates[ex_dates <= d]
        if len(past):
            last = past[-1]
            since.append((d - last).days)
            # 以「到當天為止」的歷史除息間隔中位數推估（季配約 91 天、半年配約 182 天），不偷看未來
            gaps = np.diff(past.values).astype("timedelta64[D]").astype(int)[-6:]
            step = pd.Timedelta(days=int(np.median(gaps)) if len(gaps) else 91)
            nxt = last + step
            while nxt < d:
                nxt += step
            to_ex.append((nxt - d).days)
        else:
            since.append(np.nan)
            to_ex.append(np.nan)
        to_rb.append((next_rebalance(d, cfg) - d).days)
    X["days_since_ex"], X["days_to_ex"], X["days_to_rebal"] = since, to_ex, to_rb

    tw = raw["taiex"]["close"].reindex(base.index).ffill() if len(raw["taiex"]) else pd.Series(np.nan, index=base.index)
    X["twii_bias60"] = tw / tw.rolling(60).mean() - 1
    X["twii_ret20"] = tw.pct_change(20)

    inst = raw["inst"]
    if len(inst):
        net = (inst["foreign_net"] + inst["trust_net"]).reindex(base.index).fillna(0)
        X["inst_net20"] = net.rolling(20).sum() / base["volume"].rolling(20).sum()
    else:
        X["inst_net20"] = np.nan

    # 目標
    Y = pd.DataFrame(index=base.index)
    for h in (20, 60):
        Y[f"fwd{h}"] = a.shift(-h) / a - 1
        Y[f"up{h}"] = (Y[f"fwd{h}"] > 0).astype(float).where(Y[f"fwd{h}"].notna())
    fut_min = a[::-1].rolling(60, min_periods=60).min()[::-1].shift(-1)
    Y["dip"] = (fut_min <= a * (1 - cfg["dip_pct"])).astype(float).where(fut_min.notna())

    # 去掉暖機期（warmup_days，預設 240 日）
    valid = X["bias240"].notna()
    return base[valid], X.loc[valid, FEATURES], Y[valid]
