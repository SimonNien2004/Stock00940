"""持股與成本紀錄。

data/trades.csv 欄位：date, action, shares, price, fee, note
  action：buy（買進）、sell（賣出，存股少用）
  shares：股數（1 張 = 1000 股，零股照實際股數）
配息不用手動記：依除息日前一天的持股 × 每單位配息自動計算。

指令列用法：
  python -m src.portfolio add --date 2026-10-23 --shares 5000 --price 36.2
  python -m src.portfolio summary
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import POOL_LEDGER_CSV, TRADES_CSV, ensure_dirs, load_config

COLUMNS = ["date", "action", "shares", "price", "fee", "note"]
NHI_RATE, NHI_THRESHOLD = 0.0211, 20000   # 二代健保補充保費：單筆股利 ≥ 2 萬元扣 2.11%


def load_trades() -> pd.DataFrame:
    ensure_dirs()
    if not TRADES_CSV.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(TRADES_CSV, index=False)
    df = pd.read_csv(TRADES_CSV, dtype={"note": str})
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df["note"] = df["note"].fillna("")
    for c in ("shares", "price", "fee"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def default_fee(shares: float, price: float, cfg: dict) -> float:
    return round(max(cfg["fee_min"], shares * price * cfg["fee_rate"] * cfg["fee_discount"]))


def add_trade(date: str, shares: int, price: float, action: str = "buy", fee: float | None = None,
              note: str = "", cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    if action not in ("buy", "sell"):
        raise ValueError("action 只能是 buy 或 sell")
    if shares <= 0 or price <= 0:
        raise ValueError("股數與價格需大於 0")
    fee = default_fee(shares, price, cfg) if fee is None else fee
    row = {"date": pd.Timestamp(date).strftime("%Y-%m-%d"), "action": action, "shares": int(shares),
           "price": float(price), "fee": float(fee), "note": note}
    load_trades()
    pd.DataFrame([row]).to_csv(TRADES_CSV, mode="a", header=False, index=False)
    return row


EXTRA_TAG = "額外加碼"


LEDGER_COLUMNS = ["date", "symbol", "amount", "note"]


def load_ledger(exclude_symbol: str | None = None) -> pd.DataFrame:
    """共用資金池帳本：其他標的的額外加碼（日期、標的、金額含手續費）。"""
    if not POOL_LEDGER_CSV.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS).assign(date=pd.to_datetime([]))
    df = pd.read_csv(POOL_LEDGER_CSV, dtype={"symbol": str, "note": str})
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    if exclude_symbol:
        df = df[df["symbol"].astype(str) != str(exclude_symbol)]
    return df


def add_ledger(date: str, symbol: str, amount: float, note: str = "") -> dict:
    ensure_dirs()
    if not POOL_LEDGER_CSV.exists():
        pd.DataFrame(columns=LEDGER_COLUMNS).to_csv(POOL_LEDGER_CSV, index=False)
    row = {"date": pd.Timestamp(date).strftime("%Y-%m-%d"), "symbol": str(symbol), "amount": round(float(amount), 2), "note": note}
    pd.DataFrame([row]).to_csv(POOL_LEDGER_CSV, mode="a", header=False, index=False)
    return row


def pool_status(cfg: dict, trades: pd.DataFrame | None = None, today: pd.Timestamp | None = None) -> dict | None:
    """額外加碼資金池：總額、已用、剩餘、本月已用、本月各訊號可動用上限。"""
    pool = cfg.get("extra_pool") or {}
    total = float(pool.get("total") or 0)
    if total <= 0:
        return None
    trades = load_trades() if trades is None else trades
    today = (today or pd.Timestamp.today()).normalize()
    start = pd.Timestamp(pool.get("start", "1900-01-01"))
    if len(trades):
        t = trades[(trades["action"] == "buy") & (trades["date"] >= start)
                   & trades["note"].astype(str).str.contains(EXTRA_TAG)]
    else:
        t = trades
    own = pd.DataFrame({"date": t["date"], "amount": t["shares"] * t["price"] + t["fee"]}) if len(t) \
        else pd.DataFrame(columns=["date", "amount"])
    # 共用資金池：加上其他標的（記在 pool_ledger.csv）的額外加碼支出
    others = load_ledger(cfg["symbol"])
    others = others[others["date"] >= start]
    allx = pd.concat([own, others[["date", "amount"]]], ignore_index=True)
    used = float(allx["amount"].sum()) if len(allx) else 0.0
    month0 = today.replace(day=1)
    used_month = float(allx.loc[pd.to_datetime(allx["date"]) >= month0, "amount"].sum()) if len(allx) else 0.0
    used_self = float(own["amount"].sum()) if len(own) else 0.0
    remaining = max(0.0, total - used)
    at_month_start = remaining + used_month
    npct, apct = float(pool.get("normal_pct", 0.10)), float(pool.get("aggressive_pct", 0.30))
    return {"total": total, "used": used, "used_self": used_self, "used_others": used - used_self,
            "shared_with": pool.get("shared_with") or [],
            "remaining": remaining, "used_month": used_month,
            "cap_normal": min(max(0.0, at_month_start * npct - used_month), remaining),
            "cap_aggressive": min(max(0.0, at_month_start * apct - used_month), remaining),
            "normal_pct": npct, "aggressive_pct": apct}


def with_pool(res: dict, cfg: dict) -> dict:
    p = pool_status(cfg)
    if p:
        res["pool"] = p
    return res


def holdings_on(trades: pd.DataFrame, before: pd.Timestamp) -> float:
    t = trades[trades["date"] < before]
    return float(t.loc[t["action"] == "buy", "shares"].sum() - t.loc[t["action"] == "sell", "shares"].sum())


def summary(base: pd.DataFrame, cfg: dict) -> dict:
    """base 需含 close（原始價）與 div（除息日每單位股利）。"""
    trades = load_trades()
    last_date = base.index[-1]
    close = float(base["close"].iloc[-1])
    ttm_div = float(base.loc[base.index > last_date - pd.Timedelta(days=365), "div"].sum())
    last_div_rows = base[base["div"] > 0]
    last_div = float(last_div_rows["div"].iloc[-1]) if len(last_div_rows) else 0.0

    res = {"has_position": False, "close": close, "date": last_date, "ttm_div": ttm_div,
           "last_div": last_div, "market_yield": ttm_div / close if close else 0.0}
    if trades.empty:
        return with_pool(res, cfg)

    shares = cost = 0.0
    for _, t in trades.iterrows():
        if t["action"] == "buy":
            shares += t["shares"]
            cost += t["shares"] * t["price"] + t["fee"]
        else:   # 賣出：依平均成本等比例扣除成本
            if shares > 0:
                cost -= cost * min(t["shares"], shares) / shares
            shares -= t["shares"]

    # 自動計算已領配息
    first = trades["date"].min()
    divs = []
    for d, row in base[(base["div"] > 0) & (base.index > first)].iterrows():
        held = holdings_on(trades, d)
        if held > 0:
            amt = held * row["div"]
            divs.append({"ex_date": d, "shares": held, "per_unit": row["div"], "amount": amt,
                         "nhi": amt * NHI_RATE if amt >= NHI_THRESHOLD else 0.0})
    cum_div = sum(x["amount"] - x["nhi"] for x in divs)
    avg = cost / shares if shares > 0 else 0.0
    mv = shares * close
    next_div = shares * last_div
    res.update({
        "has_position": shares > 0, "shares": shares, "lots": shares / 1000, "cost": cost,
        "avg_cost": avg, "market_value": mv, "unrealized": mv - cost,
        "unrealized_pct": (mv - cost) / cost if cost else 0.0, "cum_dividends": cum_div,
        "total_pl": mv - cost + cum_div, "cost_yield": ttm_div / avg if avg else 0.0,
        "dividends": divs, "next_div_est": next_div,
        "next_div_nhi": next_div * NHI_RATE if next_div >= NHI_THRESHOLD else 0.0,
        "n_trades": len(trades), "last_trade": trades.iloc[-1].to_dict(),
    })
    return with_pool(res, cfg)


def what_if(pos: dict, amount: float, price: float, cfg: dict) -> dict:
    """若以 price 投入 amount 元，股數與新均價。"""
    n = int(amount // (price * (1 + cfg["fee_rate"] * cfg["fee_discount"])))
    if n <= 0:
        return {"shares": 0}
    fee = default_fee(n, price, cfg)
    shares = pos.get("shares", 0) + n
    cost = pos.get("cost", 0) + n * price + fee
    return {"shares": n, "new_avg": cost / shares, "old_avg": pos.get("avg_cost", 0), "total_shares": shares}


def _cli():
    ap = argparse.ArgumentParser(description=f"{load_config()['symbol']} 持股紀錄")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="新增一筆交易")
    a.add_argument("--date", required=True)
    a.add_argument("--shares", type=int, required=True, help="股數（1 張 = 1000）")
    a.add_argument("--price", type=float, required=True)
    a.add_argument("--action", default="buy", choices=["buy", "sell"])
    a.add_argument("--fee", type=float, default=None)
    a.add_argument("--note", default="")
    a.add_argument("--extra", action="store_true", help="這筆是動用額外資金池的加碼（定期定額不要加）")
    lg = sub.add_parser("ledger", help="記錄其他標的動用共用資金池的金額")
    lg.add_argument("--date", required=True)
    lg.add_argument("--symbol", required=True, help="其他標的代號，例如 00878")
    lg.add_argument("--amount", type=float, required=True, help="含手續費的總金額")
    lg.add_argument("--note", default="")
    sub.add_parser("summary", help="顯示持股摘要")
    args = ap.parse_args()
    if args.cmd == "add":
        note = f"{EXTRA_TAG} {args.note}".strip() if args.extra and EXTRA_TAG not in args.note else args.note
        print("已記錄：", add_trade(args.date, args.shares, args.price, args.action, args.fee, note))
        cfg = load_config()
        p = pool_status(cfg)
        if p:
            print(f"額外資金池：已用 {p['used']:,.0f}／{p['total']:,.0f} 元，剩 {p['remaining']:,.0f} 元")
        if EXTRA_TAG in note and p and p["shared_with"]:
            amt = args.shares * args.price + (args.fee if args.fee is not None else default_fee(args.shares, args.price, cfg))
            for other in p["shared_with"]:
                print(f"共用資金池：請到 Stock{other} 執行 python -m src.portfolio ledger --date {args.date} "
                      f"--symbol {cfg['symbol']} --amount {amt:.0f}")
    elif args.cmd == "ledger":
        print("已記錄到共用資金池帳本：", add_ledger(args.date, args.symbol, args.amount, args.note))
        p = pool_status(load_config())
        if p:
            print(f"額外資金池：已用 {p['used']:,.0f}／{p['total']:,.0f} 元，剩 {p['remaining']:,.0f} 元")
    elif args.cmd == "summary":
        from .features import build
        cfg = load_config()
        base, _, _ = build(cfg)
        s = summary(base, cfg)
        if not s["has_position"]:
            print("目前沒有持股紀錄")
            return
        print(f"持股 {s['shares']:,.0f} 股（{s['lots']:.3f} 張），均價 {s['avg_cost']:.2f}，"
              f"市值 {s['market_value']:,.0f}，未實現 {s['unrealized']:+,.0f}，累計領息 {s['cum_dividends']:,.0f}")


if __name__ == "__main__":
    _cli()
