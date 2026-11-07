# データの方針

研究用コードと市場データの利用権は別です。このリポジトリには、原データと日次系列を含めません。

- **出典:** Yahoo Finance（yfinance経由）の `^SP500TR`、`^GSPC`、`^VIX`、`^IRX`
- **本レポートの取得日:** 2026-11-07
- **含めないもの:** `data/market_snapshot.csv`、`results/s1_wealth_paths.csv`、`results/s4_bands_h21.csv`、`results/s4_bands_h63.csv`
- **含めるもの:** コード、集計値のCSV、レポート、図

Yahoo Financeデータの再配布許諾は確認できていません。データを取得する場合は、利用者自身の用途で許諾を確認してください。

調整済みデータは後日改訂される場合があります。再取得したデータでは、`REPORT.md` の数値と完全には一致しないことがあります。
