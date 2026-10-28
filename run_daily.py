"""每日主流程：抓資料 → 特徵 → 模型 → 評分 → 回測 → 報告 → 對帳。

用法：
  python run_daily.py            # 正式：抓真實資料（GitHub Actions 每日執行）
  python run_daily.py --demo     # 測試：用模擬資料跑完整流程
  python run_daily.py --skip-fetch   # 用 data/raw 既有資料重算
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from src import backtest, demo, features, fetch, model, portfolio, report, score, webexport
from src.config import RAW, load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.time()

    if args.demo:
        meta = demo.generate()
    elif args.skip_fetch:
        meta = json.loads((RAW / "meta.json").read_text(encoding="utf-8"))
    else:
        meta = fetch.update_all(cfg)
    print(f"[1/5] 資料就緒：最後日期 {meta['last_price_date']}，來源 {meta['sources']}")

    base, X, Y = features.build(cfg)
    eng = model.engine()
    print(f"[2/5] 特徵 {X.shape[1]} 個、樣本 {len(X)} 天，模型引擎 {eng}")

    oos, raw_oos = model.walk_forward(X, Y, cfg)
    scored = score.apply(oos, X["pos120"], cfg["thresholds"])
    pred, _, contrib = model.fit_today(X, Y, raw_oos, cfg)
    today = score.apply(pred.to_frame().T.set_index(X.index[[-1]]), X["pos120"], cfg["thresholds"]).iloc[0]
    print(f"[3/5] 今日評分 {today['score']:.0f} → {today['signal']}（樣本外預測 {len(oos)} 天）")

    bt = backtest.compare(base["adj"], scored, cfg, Y)
    ev = backtest.evaluate(scored, Y)
    print(f"[4/5] 回測 XIRR 訊號 {bt['signal']['xirr']:.2%} / 定期定額 {bt['dca']['xirr']:.2%}；"
          f"60 日命中率 {ev['hit60']:.1%}（基準 {ev['base60']:.1%}）")

    pos = portfolio.summary(base, cfg)
    report.log_prediction(today, base.index[-1], float(base["close"].iloc[-1]), eng)
    log = report.reconcile(base["adj"])
    live = report.live_accuracy(log)
    js = report.write_daily(base=base, X=X, today=today, contrib=contrib, pos=pos, bt=bt, ev=ev,
                            live=live, cfg=cfg, engine=eng, meta=meta)
    report.write_evaluation(bt, ev, cfg, eng)
    webexport.export(base=base, scored=scored, today=today, js_report=js, bt=bt, ev=ev, live=live,
                     cfg=cfg, meta=meta)
    print(f"[5/5] 報告已寫入 reports/latest.md、latest.json、evaluation.md、web.json（耗時 {time.time() - t0:.0f} 秒）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
