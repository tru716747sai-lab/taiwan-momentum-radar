# Taiwan Momentum Radar

台股動能雷達：自動取得上市/上櫃股票清單與行情，做資料品質檢查、流動性過濾、動能/趨勢/風險評分，輸出 Top 10；另含歷史回測與 walk-forward 驗證。

> 研究工具，不構成投資建議。排名使用已完成交易日資料，避免未來資訊洩漏。

## 功能
- 自動建立 TWSE / TPEx 股票 universe
- Yahoo Finance 日線行情（可替換 provider）
- 資料品質 Gate：缺值、資料長度、成交金額
- Momentum：20/60/120/252 日報酬
- Trend：價格相對 20/60/120 日均線
- Risk：20 日波動度、60 日最大回撤
- Liquidity：20 日平均成交金額
- 橫斷面 percentile ranking，產出 0–100 總分
- Top 10 CSV + Markdown 報告
- Walk-forward 月度換倉回測
- GitHub Actions：台灣交易日晚間自動執行並 commit 報告

## 預設評分
`score = 45% momentum + 25% trend + 15% liquidity + 15% risk`

Momentum 內部權重：20D 10%、60D 25%、120D 30%、252D 35%。  
Risk 使用「低波動 + 低回撤」相對排名，不把高風險誤當強勢。

所有權重都在 `config/settings.yaml`，可直接調整。

## 快速開始
```bash
python -m pip install -r requirements.txt
python -m radar.cli run
python -m radar.cli backtest
```

輸出：
- `reports/latest.csv`
- `reports/latest.md`
- `reports/backtest_summary.json`
- `reports/backtest_equity.csv`

## GitHub Actions
`.github/workflows/daily-radar.yml` 於週一至週五台灣時間 19:10 執行。GitHub cron 使用 UTC，因此設定為 `10 11 * * 1-5`。也支援手動 Run workflow。

## 設計原則
1. Data Gate 先於排名：資料不足不硬算。
2. 排名與門檻分離：先過流動性/價格/歷史長度，再做橫斷面排名。
3. 無 look-ahead：訊號只使用當日以前資料，回測以次一交易期報酬衡量。
4. 可稽核：每次輸出保留各子分數與原始因子。
5. 可替換：行情與 universe provider 與 scoring 分離。

## 專案結構
```text
src/radar/
  universe.py   股票清單
  market.py     行情下載
  features.py   因子
  scoring.py    排名
  pipeline.py   每日流程
  backtest.py   walk-forward
  cli.py        CLI
config/settings.yaml
.github/workflows/daily-radar.yml
tests/
```

## 注意
Yahoo Finance 偶爾會對大量 ticker 節流；程式已採批次下載與失敗容忍。正式長期使用若需更高穩定性，可把 `market.py` 換成券商或付費資料源，其他模組不用改。
