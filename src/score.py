"""加碼評分（0–100）與訊號（PRD 第 7 節）。

評分組成：
  35%  價格便宜度      1 - 120 日區間位置
  35%  60 日上漲機率    (p_up60 - 0.35) / 0.40，截在 0–1
  15%  不會再跌 3%      1 - p_dip
  15%  60 日中位數報酬  (q50_60 + 5%) / 15%，截在 0–1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGGRESSIVE, NORMAL, PAUSE = "積極加碼", "一般加碼", "暫緩加碼"


def score(pos120, p_up60, p_dip, q50_60):
    value = 1 - np.asarray(pos120, dtype=float)
    pu = np.clip((np.asarray(p_up60, dtype=float) - 0.35) / 0.40, 0, 1)
    nd = 1 - np.asarray(p_dip, dtype=float)
    med = np.clip((np.asarray(q50_60, dtype=float) + 0.05) / 0.15, 0, 1)
    return np.round(100 * (0.35 * value + 0.35 * pu + 0.15 * nd + 0.15 * med), 1)


def confidence(p_up60, q50_60):
    """高：機率明確且與中位數方向一致；低：兩個模型方向相反。"""
    p = np.asarray(p_up60, dtype=float)
    q = np.asarray(q50_60, dtype=float)
    agree = (p > 0.5) == (q > 0)
    out = np.where(np.abs(p - 0.5) >= 0.15, "高", "中")
    return np.where(~agree & (np.abs(p - 0.5) > 0.05), "低", out)


def signal(sc, pos120, p_up60, p_dip, q50_60, conf, th: dict):
    sc, pos120 = np.asarray(sc, float), np.asarray(pos120, float)
    p_up60, p_dip, conf = np.asarray(p_up60, float), np.asarray(p_dip, float), np.asarray(conf)
    q50_60 = np.asarray(q50_60, float)
    aggressive = (sc >= th["aggressive"]) & (pos120 <= th["aggressive_max_pos120"]) & (p_up60 >= th["aggressive_min_p_up"])
    # 高股息 ETF 長期向上，延後買進只在預期報酬為負時才划算：評分低「且」中位數預期為負，或很可能再跌 3%
    pause = ((sc < th["pause"]) & (q50_60 < 0)) | (p_dip >= th["dip_pause_prob"])
    out = np.where(aggressive, AGGRESSIVE, np.where(pause, PAUSE, NORMAL))
    return np.where(conf == "低", NORMAL, out)   # 信心低 → 退回定期定額


def apply(pred: pd.DataFrame, pos120: pd.Series, th: dict) -> pd.DataFrame:
    """對一整段預測（樣本外或今日）計算評分、信心、訊號。"""
    out = pred.copy()
    pos = pos120.reindex(out.index)
    out["pos120"] = pos
    out["score"] = score(pos, out["p_up60"], out["p_dip"], out["q50_60"])
    out["confidence"] = confidence(out["p_up60"], out["q50_60"])
    out["signal"] = signal(out["score"], pos, out["p_up60"], out["p_dip"], out["q50_60"], out["confidence"], th)
    return out
