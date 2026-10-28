"""預測模型：LightGBM（GitHub Actions 上）或 scikit-learn HistGradientBoosting（備援）。

輸出九個量：
  p_up20, p_up60      未來 20 / 60 個交易日還原報酬 > 0 的機率
  p_dip               未來 60 日內出現比今日低 3% 以上價格的機率
  q10/q50/q90_20/60   未來 20 / 60 日還原報酬的 10% / 50% / 90% 分位

驗證方式：walk-forward（擴張視窗），訓練資料與測試起點之間留 60 日空窗，
避免目標值重疊造成的資訊外洩。
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

CLF_TARGETS = {"p_up20": "up20", "p_up60": "up60", "p_dip": "dip"}
REG_TARGETS = {
    "q10_20": ("fwd20", 0.1), "q50_20": ("fwd20", 0.5), "q90_20": ("fwd20", 0.9),
    "q10_60": ("fwd60", 0.1), "q50_60": ("fwd60", 0.5), "q90_60": ("fwd60", 0.9),
}
OUTPUTS = list(CLF_TARGETS) + list(REG_TARGETS)
PURGE = 60


def engine() -> str:
    if os.environ.get("ETF_ENGINE") == "sklearn":
        return "sklearn"
    try:
        import lightgbm  # noqa: F401
        return "lightgbm"
    except ImportError:
        return "sklearn"


HORIZON = {"p_up20": 20, "p_up60": 60, "p_dip": 60}

# 刻意保守的參數：金融資料樣本少、標籤高度重疊，複雜模型很容易過度擬合
N_TREES, LR, LEAVES, MIN_LEAF = 150, 0.05, 7, 30


def _make_clf():
    if engine() == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=N_TREES, learning_rate=LR, num_leaves=LEAVES,
                              min_child_samples=MIN_LEAF, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=0)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=N_TREES, learning_rate=LR, max_leaf_nodes=LEAVES,
                                          min_samples_leaf=MIN_LEAF, l2_regularization=5.0, random_state=0)


def _make_reg(alpha: float):
    if engine() == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=N_TREES, learning_rate=LR,
                             num_leaves=LEAVES, min_child_samples=MIN_LEAF, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, random_state=0)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=N_TREES, learning_rate=LR,
                                         max_leaf_nodes=LEAVES, min_samples_leaf=MIN_LEAF, random_state=0)


def _fit_all(Xtr: pd.DataFrame, Ytr: pd.DataFrame, stride: int = 1) -> dict:
    # 每 stride 天取一筆：降低相鄰樣本高度相關造成的過度擬合，也加快訓練
    if stride > 1:
        keep = np.arange(len(Xtr) - 1, -1, -stride)[::-1]
        Xtr, Ytr = Xtr.iloc[keep], Ytr.iloc[keep]
    models = {}
    for out, col in CLF_TARGETS.items():
        m = Ytr[col].notna()
        y = Ytr.loc[m, col].astype(int)
        if y.nunique() < 2:  # 全部同一類時，退化為常數機率
            models[out] = float(y.iloc[0]) if len(y) else 0.5
            continue
        models[out] = _make_clf().fit(Xtr[m], y)
    for out, (col, alpha) in REG_TARGETS.items():
        m = Ytr[col].notna()
        models[out] = _make_reg(alpha).fit(Xtr[m], Ytr.loc[m, col])
    return models


def _predict_all(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for k in CLF_TARGETS:
        m = models[k]
        out[k] = np.full(len(X), m) if isinstance(m, float) else m.predict_proba(X)[:, 1]
    for k in REG_TARGETS:
        out[k] = models[k].predict(X)
    # 修正分位數交叉
    for h in (20, 60):
        cols = [f"q10_{h}", f"q50_{h}", f"q90_{h}"]
        out[cols] = np.sort(out[cols].values, axis=1)
    return out


def walk_forward(X: pd.DataFrame, Y: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """樣本外預測（每個日期只用當時已知的資料訓練），並逐段校準。回傳 (校準後, 原始)。"""
    min_train = cfg["model"]["min_train_days"]
    step = cfg["model"]["retrain_every"]
    stride = cfg["model"].get("train_stride", 5)
    n = len(X)
    parts = []
    for t0 in range(min_train + PURGE, n, step):
        models = _fit_all(X.iloc[: t0 - PURGE], Y.iloc[: t0 - PURGE], stride)
        raw = _predict_all(models, X.iloc[t0:t0 + step])
        raw["pos"] = np.arange(t0, min(t0 + step, n))
        parts.append(raw)
    if not parts:
        raise RuntimeError("歷史資料不足，無法做 walk-forward 驗證")
    raw = pd.concat(parts)
    cal = []
    for t0, block in raw.groupby((raw["pos"] - raw["pos"].iloc[0]) // step, sort=True):
        start = int(block["pos"].iloc[0])
        cal.append(calibrate(block, raw[raw["pos"] < start], Y, start))
    return pd.concat(cal).drop(columns="pos"), raw


def fit_today(X: pd.DataFrame, Y: pd.DataFrame, raw_oos: pd.DataFrame, cfg: dict) -> tuple[pd.Series, dict, pd.Series]:
    """用全部已知標籤訓練，預測最後一天並校準；回傳 (預測, 模型, 特徵貢獻)。"""
    models = _fit_all(X, Y, cfg["model"].get("train_stride", 5))
    last = X.iloc[[-1]]
    raw = _predict_all(models, last)
    pred = calibrate(raw, raw_oos, Y, len(X)).iloc[0]
    contrib = contributions(models["p_up60"], X, last)
    return pred, models, contrib


# ---------------------------------------------------------------- 逐段校準
# 只用「在該時點之前、標籤已經揭曉」的樣本外預測來校準，不會偷看未來。
MIN_CAL_ROWS = 250
TARGET_TAIL = 0.10   # 區間兩端各 10%，即 80% 區間


def calibrate(block: pd.DataFrame, hist: pd.DataFrame, Y: pd.DataFrame, start: int) -> pd.DataFrame:
    """block：待校準的預測；hist：更早的樣本外預測（含 pos 欄）；start：block 的起始位置。"""
    from sklearn.linear_model import LogisticRegression

    out = block.copy()
    for k, h in HORIZON.items():
        known = hist[hist["pos"] <= start - h - 1]
        y = Y[CLF_TARGETS[k]].reindex(known.index)
        m = y.notna()
        p = out[k].clip(0.01, 0.99)
        if m.sum() >= MIN_CAL_ROWS and y[m].nunique() == 2:
            z = np.log(known.loc[m, k].clip(0.01, 0.99) / (1 - known.loc[m, k].clip(0.01, 0.99)))
            lr = LogisticRegression(C=1.0).fit(z.to_frame(), y[m].astype(int))
            out[k] = lr.predict_proba(np.log(p / (1 - p)).to_frame(name=z.name))[:, 1]
        else:   # 樣本不足：往 50% 收縮一半
            out[k] = 0.5 + 0.5 * (p - 0.5)
    for h in (20, 60):
        q10, q50, q90 = f"q10_{h}", f"q50_{h}", f"q90_{h}"
        known = hist[hist["pos"] <= start - h - 1]
        y = Y[f"fwd{h}"].reindex(known.index)
        m = y.notna()
        if m.sum() >= MIN_CAL_ROWS:
            k_ = known[m]
            wl = (k_[q50] - k_[q10]).clip(lower=1e-4)
            wu = (k_[q90] - k_[q50]).clip(lower=1e-4)
            ml = float(np.quantile((k_[q50] - y[m]) / wl, 1 - TARGET_TAIL))
            mu = float(np.quantile((y[m] - k_[q50]) / wu, 1 - TARGET_TAIL))
        else:
            ml = mu = 1.5
        ml, mu = max(ml, 0.5), max(mu, 0.5)
        lo = out[q50] - ml * (out[q50] - out[q10]).clip(lower=1e-4)
        hi = out[q50] + mu * (out[q90] - out[q50]).clip(lower=1e-4)
        out[q10], out[q90] = lo, hi
    return out


def contributions(clf, X: pd.DataFrame, last: pd.DataFrame) -> pd.Series:
    """各特徵對「60 日上漲機率」的影響：把該特徵換成歷史中位數後，機率改變多少。

    正值 = 目前這個特徵值讓上漲機率變高（有利加碼）。適用於任何模型引擎。
    """
    if isinstance(clf, float):
        return pd.Series(0.0, index=X.columns)
    med = X.median()
    rows = pd.concat([last] * len(X.columns), ignore_index=True).astype(float)
    for i, c in enumerate(X.columns):
        rows.loc[i, c] = med[c]
    p_now = clf.predict_proba(last)[0, 1]
    p_alt = clf.predict_proba(rows)[:, 1]
    return pd.Series(p_now - p_alt, index=X.columns)
