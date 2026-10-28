# 00940 存股加碼助理

每個交易日盤後自動判斷「今天是不是好的 00940（元大台灣價值高息）加碼點」，給出 0–100 的加碼評分與建議（積極加碼／一般加碼／暫緩加碼），記錄你的持股與成本，並持續拿「照訊號加碼」和「定期定額」比較。你直接在 Claude 對話中提問與回報交易。

> 僅供本人研究參考，不構成投資建議。

## 00940 的特性

| 項目 | 00940 | 說明 |
| --- | --- | --- |
| 名稱 | 元大台灣價值高息 | 高股息 ETF，50 檔成分股 |
| 追蹤指數 | 臺灣指數公司特選臺灣價值高息指數 | |
| 上市日 | 2024-04-01 | 歷史只有約 2 年半，是所有版本中最短的 |
| 配息 | 月配，除息多在每月 6～11 日 | `ex_months` 為 1～12 月、`ex_day` 為 9 |
| 成分股調整 | 5、11 月 | `rebalance_months`、`rebalance_day` |
| 暖機與訓練 | 240 日指標暖機縮短為 120 日（`warmup_days`）；最少訓練 250 日 | 其他版本暖機為 240 日 |

**歷史太短的影響**：樣本外驗證不到 1 年，涵蓋的行情很少，機率也無法校準。評估報告的數字幾乎沒有統計意義，請以定期定額為主，訊號只當參考。累積更多資料後（例如 2027 年底），可以把 `warmup_days` 調回 240、`min_train_days` 調高。

## 運作方式

1. **GitHub Actions（雲端排程）**：週一至週五 16:00 抓資料 → 算特徵 → 模型預測 → 評分 → 回測 → 寫報告，結果 commit 回你的私人 repo。
2. **Claude 對話**：在對話中加入這個 repo，Claude 讀 `reports/latest.json` 回答你，並把你回報的交易寫入 `data/trades.csv`。工作規則寫在 `CLAUDE.md`。

## 一次性設定（約 15 分鐘）

1. **私人 repo**：程式已推到 `SimonNien2004/Stock00940`。
2. **FinMind token**：到 [FinMind](https://finmindtrade.com/) 免費註冊，在使用者頁面複製 API token。回到 repo → Settings → Secrets and variables → Actions → New repository secret，名稱 `FINMIND_TOKEN`，貼上 token。沒有 token 也能跑，只是額度較低；FinMind 失敗時會自動改用 yfinance。
3. **開啟寫入權限**：repo → Settings → Actions → General → Workflow permissions → 選 **Read and write permissions** → Save。
4. **改預算**：編輯 `config.yaml` 的 `monthly_budget`（每月加碼預算）與 `reserve`（預備金）。
5. **第一次執行**：repo → Actions → 「00940 每日加碼報告」→ Run workflow。約 3–5 分鐘後，`reports/latest.md` 就是第一份報告，`reports/evaluation.md` 是用真實資料做的評估。
6. **建立持股**：在 Claude 對話中說「我目前有 20 張，均價 33.5」或逐筆回報交易。

## 在 Claude 對話中使用

建議建立一個 Claude 專案，在專案指示貼上：

```
這是我的 00940 存股加碼助理。每次對話先加入 GitHub repo SimonNien2004/Stock00940，
讀 CLAUDE.md 後依其規則回答。回報交易時先複誦確認再寫入。
```

可以這樣問：
- 「今天適合加碼嗎？」
- 「我有 20 萬，現在買還是等？」
- 「今天買了 3 張，36.2」
- 「我現在持股狀況？」
- 「除息前買還是除息後買比較好？」
- 「回測評分 80 以上才積極加碼，結果如何？」
- 「上個月的預測準嗎？」

## 檔案結構

| 路徑 | 用途 |
| --- | --- |
| `config.yaml` | 預算、手續費、訊號門檻、模型與回測參數 |
| `run_daily.py` | 每日主流程 |
| `src/fetch.py` | 資料抓取（FinMind 主、yfinance 備援） |
| `src/features.py` | 還原權息價、18 個特徵、預測目標 |
| `src/model.py` | LightGBM 分類與分位數模型、walk-forward 驗證、逐段校準 |
| `src/score.py` | 加碼評分與訊號規則 |
| `src/backtest.py` | 訊號加碼 vs 定期定額 vs 簡單規則 vs 完美預知 |
| `src/portfolio.py` | 持股、均價、自動計算配息與二代健保 |
| `src/report.py` | 每日報告、評估報告、預測對帳 |
| `src/demo.py` | 模擬資料（離線測試用） |
| `tests/` | 前視偏差與核心邏輯測試 |
| `data/` | 原始資料、交易紀錄、預測紀錄（由排程更新） |
| `reports/` | `latest.md`、`latest.json`、`evaluation.md` |

## 本機執行（選用）

```bash
pip install -r requirements.txt
python run_daily.py --demo        # 用模擬資料測試整套流程
python run_daily.py               # 抓真實資料（需可連到 FinMind）
python tests/test_pipeline.py     # 執行測試
```

## 方法與限制

- **模型**：預測未來 20、60 個交易日含息報酬的機率與 10%／50%／90% 分位，以及 60 日內再跌 3% 的機率。訓練採 walk-forward（訓練與測試間隔 60 日），每 5 天取一筆樣本降低重疊，並只用當時已揭曉的樣本外結果校準機率與區間。
- **評分**：價格便宜度 35%、60 日上漲機率 35%、不再跌 3% 15%、60 日中位數報酬 15%。
- **暫緩加碼很少出現是刻意的**：ETF 長期向上，延後買進只有在預期報酬為負時才划算。
- **擇時效益有天花板**：以每月預算為單位，即使完美預知未來，年化報酬也只比定期定額多一點點。這套系統最大的價值是避免在相對高點重押、在相對低點動用預備金，以及把持股、配息、成本殖利率整理清楚。`reports/evaluation.md` 會用真實資料告訴你實際效益；若贏不了「簡單規則」，就該改用規則訊號。
- 遇到國定假日，排程仍會執行，但資料日期不變，報告內容與前一日相同。
