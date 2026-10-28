"""資料抓取：FinMind 為主、yfinance 為備援。

輸出統一格式到 data/raw/：
  price.csv      date, open, high, low, close, volume     （原始價，未還原）
  dividends.csv  date, cash_dividend                        （除息交易日、每單位現金股利）
  taiex.csv      date, close                                （加權指數）
  inst.csv       date, foreign_net, trust_net               （外資、投信對本標的淨買超股數）
  meta.json      各資料的來源與最後日期
"""
from __future__ import annotations

import json
import os
import time
from datetime import date

import pandas as pd
import requests

from .config import RAW, ensure_dirs

FINMIND_API = "https://api.finmindtrade.com/api/v4/data"


# ---------------------------------------------------------------- FinMind
def _finmind(dataset: str, data_id: str, start: str, token: str | None) -> pd.DataFrame:
    params = {"dataset": dataset, "data_id": data_id, "start_date": start}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.get(FINMIND_API, params=params, headers=headers, timeout=60)
            r.raise_for_status()
            js = r.json()
            if js.get("status", 200) != 200:
                raise RuntimeError(f"FinMind {dataset}: {js.get('msg')}")
            return pd.DataFrame(js.get("data", []))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"FinMind {dataset} 抓取失敗：{last_err}")


def _fm_price(symbol: str, start: str, token: str | None) -> pd.DataFrame:
    df = _finmind("TaiwanStockPrice", symbol, start, token)
    if df.empty:
        raise RuntimeError(f"FinMind 沒有回傳 {symbol} 價格")
    df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
    df = df[["date", "open", "high", "low", "close", "volume"]]
    return df[df["close"] > 0]


def _fm_dividends(symbol: str, start: str, token: str | None) -> pd.DataFrame:
    df = _finmind("TaiwanStockDividendResult", symbol, start, token)
    if df.empty:
        return pd.DataFrame(columns=["date", "cash_dividend"])
    cash = pd.to_numeric(df.get("stock_and_cache_dividend"), errors="coerce")
    fallback = pd.to_numeric(df["before_price"], errors="coerce") - pd.to_numeric(df["after_price"], errors="coerce")
    df["cash_dividend"] = cash.where(cash > 0, fallback).round(4)
    return df[["date", "cash_dividend"]][df["cash_dividend"] > 0]


def _fm_taiex(start: str, token: str | None) -> pd.DataFrame:
    df = _finmind("TaiwanStockPrice", "TAIEX", start, token)
    return df[["date", "close"]]


def _fm_inst(symbol: str, start: str, token: str | None) -> pd.DataFrame:
    df = _finmind("TaiwanStockInstitutionalInvestorsBuySell", symbol, start, token)
    if df.empty:
        return pd.DataFrame(columns=["date", "foreign_net", "trust_net"])
    df["net"] = pd.to_numeric(df["buy"], errors="coerce") - pd.to_numeric(df["sell"], errors="coerce")
    pv = df.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0)
    out = pd.DataFrame(index=pv.index)
    foreign_cols = [c for c in pv.columns if c.startswith("Foreign")]
    out["foreign_net"] = pv[foreign_cols].sum(axis=1) if foreign_cols else 0
    out["trust_net"] = pv["Investment_Trust"] if "Investment_Trust" in pv else 0
    return out.reset_index()


# ---------------------------------------------------------------- yfinance 備援
def _yf_price_and_div(symbol: str, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    h = yf.Ticker(f"{symbol}.TW").history(start=start, auto_adjust=False, actions=True)
    if h.empty:
        raise RuntimeError("yfinance 沒有回傳資料")
    h.index = h.index.tz_localize(None).strftime("%Y-%m-%d")
    price = pd.DataFrame({
        "date": h.index, "open": h["Open"].values, "high": h["High"].values,
        "low": h["Low"].values, "close": h["Close"].values, "volume": h["Volume"].values,
    })
    d = h[h["Dividends"] > 0]
    div = pd.DataFrame({"date": d.index, "cash_dividend": d["Dividends"].values})
    return price, div


def _yf_taiex(start: str) -> pd.DataFrame:
    import yfinance as yf

    h = yf.Ticker("^TWII").history(start=start, auto_adjust=False)
    h.index = h.index.tz_localize(None).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": h.index, "close": h["Close"].values})


# ---------------------------------------------------------------- 主流程
def update_all(cfg: dict) -> dict:
    """抓取全部資料（每次全量重抓，資料量小，避免增量出錯）。"""
    ensure_dirs()
    symbol, start = cfg["symbol"], cfg["history_start"]
    token = os.environ.get("FINMIND_TOKEN") or None
    meta: dict = {"fetched_at": date.today().isoformat(), "sources": {}, "warnings": []}

    try:
        price = _fm_price(symbol, start, token)
        div = _fm_dividends(symbol, start, token)
        meta["sources"]["price"] = meta["sources"]["dividends"] = "FinMind"
    except Exception as e:  # noqa: BLE001
        meta["warnings"].append(f"FinMind 價格失敗，改用 yfinance：{e}")
        price, div = _yf_price_and_div(symbol, start)
        meta["sources"]["price"] = meta["sources"]["dividends"] = "yfinance"

    try:
        taiex = _fm_taiex(start, token)
        meta["sources"]["taiex"] = "FinMind"
    except Exception as e:  # noqa: BLE001
        meta["warnings"].append(f"FinMind 加權指數失敗，改用 yfinance：{e}")
        try:
            taiex = _yf_taiex(start)
            meta["sources"]["taiex"] = "yfinance"
        except Exception as e2:  # noqa: BLE001
            meta["warnings"].append(f"加權指數無法取得：{e2}")
            taiex = pd.DataFrame(columns=["date", "close"])

    try:
        inst = _fm_inst(symbol, start, token)
        meta["sources"]["inst"] = "FinMind"
    except Exception as e:  # noqa: BLE001
        meta["warnings"].append(f"法人買賣超無法取得（模型會略過此特徵）：{e}")
        inst = pd.DataFrame(columns=["date", "foreign_net", "trust_net"])

    # FinMind 偶爾晚一兩天更新：用 yfinance 補上最近缺的交易日
    if meta["sources"]["price"] == "FinMind":
        try:
            yf_price, _ = _yf_price_and_div(symbol, (pd.Timestamp(price["date"].max()) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
            newer = yf_price[yf_price["date"] > str(price["date"].max())]
            # yfinance 在資料還沒出來時會塞一筆「成交量 0、開高低收全同」的佔位資料，丟掉
            fake = newer[pd.to_numeric(newer["volume"], errors="coerce").fillna(0) <= 0]
            if len(fake):
                meta["warnings"].append(f"yfinance 的 {', '.join(fake['date'])} 成交量為 0（佔位資料），已略過")
                newer = newer.drop(fake.index)
            if len(newer):
                price = pd.concat([price, newer], ignore_index=True)
                meta["sources"]["price"] = "FinMind＋yfinance"
                meta["warnings"].append(f"FinMind 尚未更新 {', '.join(newer['date'])}，這幾天的價格由 yfinance 補上")
        except Exception as e:  # noqa: BLE001
            meta["warnings"].append(f"yfinance 補抓最新價格失敗：{e}")

    last = pd.Timestamp(price["date"].max())
    expected = pd.Timestamp(date.today()) - pd.offsets.BDay(0 if pd.Timestamp.now(tz="Asia/Taipei").hour >= 15 else 1)
    holidays = {str(h) for h in cfg.get("market_holidays", [])}
    missing = [d for d in pd.bdate_range(last + pd.Timedelta(days=1), expected) if f"{d:%Y-%m-%d}" not in holidays]
    behind = len(missing)
    meta["expected_date"] = f"{expected:%Y-%m-%d}"
    meta["stale_days"] = behind
    if behind >= 1:
        meta["warnings"].append(f"價格資料停在 {last:%Y-%m-%d}，比最近的工作日少 {behind} 天（可能是休市，或資料源還沒更新）")

    price.sort_values("date").to_csv(RAW / "price.csv", index=False)
    div.sort_values("date").to_csv(RAW / "dividends.csv", index=False)
    taiex.sort_values("date").to_csv(RAW / "taiex.csv", index=False)
    inst.sort_values("date").to_csv(RAW / "inst.csv", index=False)

    meta["last_price_date"] = str(price["date"].max())
    with open(RAW / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta
